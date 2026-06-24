from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ModelConfig:
    vocab_size: int = 50000
    hidden_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    ffn_hidden_dim: int = 5632
    max_seq_len: int = 4096
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Recompute layer activations during backward instead of storing them.
    # Trades ~30% extra compute for a large drop in activation memory —
    # essential for fitting 300M+ on a 16 GB T4.
    gradient_checkpointing: bool = False

    @classmethod
    def from_preset(cls, name: str, vocab_size: int = 32000) -> "ModelConfig":
        presets: Dict[str, Dict] = {
            "nano": dict(
                hidden_dim=128, num_layers=6, num_heads=8,
                num_kv_heads=4, head_dim=16, ffn_hidden_dim=352,
                max_seq_len=256,
            ),
            "10M": dict(
                hidden_dim=256, num_layers=12, num_heads=8,
                num_kv_heads=4, head_dim=32, ffn_hidden_dim=704,
                max_seq_len=512,
            ),
            "50M": dict(
                hidden_dim=512, num_layers=16, num_heads=8,
                num_kv_heads=4, head_dim=64, ffn_hidden_dim=1408,
                max_seq_len=512,
            ),
            "100M": dict(
                hidden_dim=768, num_layers=12, num_heads=12,
                num_kv_heads=4, head_dim=64, ffn_hidden_dim=2048,
                max_seq_len=1024,
            ),
            "300M": dict(
                hidden_dim=1024, num_layers=20, num_heads=16,
                num_kv_heads=4, head_dim=64, ffn_hidden_dim=2816,
                max_seq_len=1024,
            ),
            "350M": dict(
                hidden_dim=1024, num_layers=24, num_heads=16,
                num_kv_heads=4, head_dim=64, ffn_hidden_dim=3072,
                max_seq_len=1024,
            ),
        }
        if name not in presets:
            raise ValueError(f"Preset must be one of {list(presets.keys())}")
        return cls(vocab_size=vocab_size, tie_embeddings=True, **presets[name])

    @property
    def total_params_estimated(self) -> int:
        emb = self.vocab_size * self.hidden_dim
        attn_q = self.hidden_dim * self.num_heads * self.head_dim
        attn_k = self.hidden_dim * self.num_kv_heads * self.head_dim
        attn_v = self.hidden_dim * self.num_kv_heads * self.head_dim
        attn_o = self.num_heads * self.head_dim * self.hidden_dim
        attn = attn_q + attn_k + attn_v + attn_o
        ffn = 3 * self.hidden_dim * self.ffn_hidden_dim
        per_layer = attn + ffn + 2 * self.hidden_dim
        total = emb + per_layer * self.num_layers + self.hidden_dim
        if not self.tie_embeddings:
            total += self.vocab_size * self.hidden_dim
        return total

    def __str__(self) -> str:
        return (
            f"ModelConfig(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_heads={self.num_heads},\n"
            f"  num_kv_heads={self.num_kv_heads},\n"
            f"  head_dim={self.head_dim},\n"
            f"  ffn_hidden_dim={self.ffn_hidden_dim},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f"  norm_eps={self.norm_eps},\n"
            f"  rope_theta={self.rope_theta},\n"
            f"  dropout={self.dropout},\n"
            f"  tie_embeddings={self.tie_embeddings},\n"
            f"  estimated_params={self.total_params_estimated:,}\n"
            f")"
        )
