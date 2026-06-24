import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def precompute_freqs_cis(
    dim: int, max_seq_len: int, theta: float = 10000.0
):
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embeddings.

    Args:
        x: [batch, num_heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim // 2] already sliced to the positions
            covered by x (the caller offsets for KV-cache).
    """
    half_dim = x.shape[-1] // 2
    x_reshaped = x.view(*x.shape[:-1], half_dim, 2)
    x_even = x_reshaped[..., 0]
    x_odd = x_reshaped[..., 1]

    cos = cos.view(1, 1, x.shape[-2], half_dim)
    sin = sin.view(1, 1, x.shape[-2], half_dim)

    x_even_new = x_even * cos - x_odd * sin
    x_odd_new = x_odd * cos + x_even * sin

    x_rotated = torch.stack([x_even_new, x_odd_new], dim=-1).view(*x.shape)
    return x_rotated


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x.unsqueeze(2)
        .expand(batch, num_heads, n_rep, seq_len, head_dim)
        .reshape(batch, num_heads * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads

        self.wq = nn.Linear(
            config.hidden_dim,
            config.num_heads * config.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            config.hidden_dim,
            config.num_kv_heads * config.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            config.hidden_dim,
            config.num_kv_heads * config.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            config.num_heads * config.head_dim,
            config.hidden_dim,
            bias=False,
        )
        self.dropout_p = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        batch, seq_len, _ = x.shape

        xq = self.wq(x).view(batch, seq_len, self.num_heads, self.head_dim)
        xk = self.wk(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        xq = apply_rotary_emb(xq, cos, sin)
        xk = apply_rotary_emb(xk, cos, sin)

        # KV-cache: prepend previously computed keys/values (incremental decode)
        if past_kv is not None:
            past_k, past_v = past_kv
            xk = torch.cat([past_k, xk], dim=2)
            xv = torch.cat([past_v, xv], dim=2)
        present_kv = (xk, xv) if use_cache else None

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Fused, memory-efficient attention (FlashAttention / mem-efficient
        # kernels when available). is_causal handles the triangular mask
        # without materializing a [seq, seq] tensor.
        if mask is not None:
            attn_mask = mask
            is_causal = False
        elif past_kv is None and seq_len > 1:
            attn_mask = None
            is_causal = True
        else:
            # Single-token decode (q_len == 1) attends to full cached context.
            attn_mask = None
            is_causal = False

        dropout_p = self.dropout_p if self.training else 0.0
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        output = (
            output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        )
        output = self.wo(output)
        return output, present_kv
