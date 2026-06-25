import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .transformer import DecoderLayer
from .layers import RMSNorm
from .attention import precompute_freqs_cis


class SuperCodeurModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim
        )
        self.dropout = (
            nn.Dropout(config.dropout)
            if config.dropout > 0
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_dim, config.norm_eps)

        self.lm_head = nn.Linear(
            config.hidden_dim, config.vocab_size, bias=False
        )
        if config.tie_embeddings:
            # Weight tying: share the embedding matrix with the output
            # projection (standard for decoder-only LMs, saves vocab*dim params).
            self.lm_head.weight = self.token_embedding.weight

        cos, sin = precompute_freqs_cis(
            config.head_dim, config.max_seq_len * 2, config.rope_theta
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual projections: keeps the variance
        # of the residual stream bounded as depth grows.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        ignore_index: int = -100,
    ):
        """Forward pass.

        If ``targets`` is given, returns ``(None, loss)``: the loss is computed
        on the next-token objective in memory-bounded chunks and the full logit
        tensor is never materialized (the trainer only needs the loss).
        Otherwise returns ``(logits, present_kv)`` for generation/inference.
        """
        batch, seq_len = input_ids.shape

        # Position offset for RoPE when decoding with a KV-cache.
        past_len = 0
        if past_kv is not None and past_kv[0] is not None:
            past_len = past_kv[0][0].shape[2]

        h = self.dropout(self.token_embedding(input_ids))

        cos = self.cos[past_len : past_len + seq_len]
        sin = self.sin[past_len : past_len + seq_len]

        use_ckpt = (
            self.config.gradient_checkpointing
            and self.training
            and not use_cache
        )

        new_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            layer_past = past_kv[i] if past_kv is not None else None
            if use_ckpt:
                # Recompute this layer's activations in the backward pass
                # instead of holding them in memory through the whole forward.
                h = checkpoint(
                    self._layer_forward, layer, h, cos, sin,
                    use_reentrant=False,
                )
                present = None
            else:
                h, present = layer(
                    h, cos, sin, mask=None,
                    past_kv=layer_past, use_cache=use_cache,
                )
            if use_cache:
                new_kv.append(present)

        h = self.norm(h)

        if targets is not None:
            # Next-token objective: predict token t+1 from the state at t.
            # Full logits matrix at batch=4, seq=1024, vocab=5000 is ~20 MB
            # in fp16 — well within T4 memory. Direct computation is ~10×
            # faster than chunked + checkpointed loss.
            logits = self.lm_head(h)
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_targets = targets[:, 1:].reshape(-1)
            loss = F.cross_entropy(
                shift_logits.float(),
                shift_targets,
                ignore_index=ignore_index,
            )
            return None, loss

        logits = self.lm_head(h)
        return logits, (new_kv if use_cache else None)

    @staticmethod
    def _layer_forward(layer, h, cos, sin):
        # Checkpoint-friendly wrapper: tensor in, tensor out (training has no
        # KV-cache, so the present_kv output is always None and dropped here).
        out, _ = layer(h, cos, sin, mask=None, past_kv=None, use_cache=False)
        return out

    def _chunked_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Memory-bounded cross-entropy over the tied LM head.

        Projects and scores ``chunk_size`` rows at a time. Each chunk's logits
        are recomputed in the backward pass (checkpointing), so peak logit
        memory is ``chunk_size * vocab`` rather than ``B*T * vocab``.
        """
        weight = self.lm_head.weight

        def score(h_chunk: torch.Tensor, lbl_chunk: torch.Tensor) -> torch.Tensor:
            logits = F.linear(h_chunk, weight)
            return F.cross_entropy(
                logits.float(),
                lbl_chunk,
                ignore_index=ignore_index,
                reduction="sum",
            )

        total = hidden.new_zeros(())
        recompute = self.training and torch.is_grad_enabled()
        for start in range(0, hidden.size(0), chunk_size):
            h_chunk = hidden[start : start + chunk_size]
            lbl_chunk = labels[start : start + chunk_size]
            if recompute:
                total = total + checkpoint(
                    score, h_chunk, lbl_chunk, use_reentrant=False
                )
            else:
                total = total + score(h_chunk, lbl_chunk)

        valid = (labels != ignore_index).sum().clamp(min=1)
        return total / valid

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.15,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()

        past_kv = None
        cur = input_ids
        generated_ids = input_ids[0].tolist()

        for _ in range(max_new_tokens):
            # Trim context to the model's window (only matters without cache).
            if past_kv is None and cur.shape[1] > self.config.max_seq_len:
                cur = cur[:, -self.config.max_seq_len :]

            logits, past_kv = self.forward(cur, use_cache=True, past_kv=past_kv)
            logits = logits[:, -1, :].float()

            # Repetition penalty over the full running sequence.
            if repetition_penalty != 1.0:
                for tok in set(generated_ids):
                    if logits[0, tok] > 0:
                        logits[0, tok] /= repetition_penalty
                    else:
                        logits[0, tok] *= repetition_penalty

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(
                        logits, descending=True
                    )
                    cum = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    remove = cum > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    idx_remove = remove.scatter(1, sorted_idx, remove)
                    logits = logits.masked_fill(idx_remove, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            generated_ids.append(token_id)
            # With the KV-cache we only feed the single new token next round.
            cur = next_token

            if eos_token_id is not None and token_id == eos_token_id:
                break

        return torch.tensor([generated_ids], device=input_ids.device)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
