import torch.nn as nn

from .attention import GroupedQueryAttention
from .layers import RMSNorm, SwiGLUFFN


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLUFFN(config)
        self.attention_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

    def forward(self, x, cos, sin, mask=None, past_kv=None, use_cache=False):
        h = self.attention_norm(x)
        h, present_kv = self.attention(
            h, cos, sin, mask, past_kv=past_kv, use_cache=use_cache
        )
        h = self.dropout(h)
        x = x + h

        h = self.ffn_norm(x)
        h = self.feed_forward(h)
        h = self.dropout(h)
        x = x + h

        return x, present_kv
