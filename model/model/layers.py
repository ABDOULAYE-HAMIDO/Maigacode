import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in float32 for numerical stability under mixed precision,
        # then cast back to the input dtype.
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return (x_normed * self.weight).to(dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_dim, config.ffn_hidden_dim, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_dim, config.ffn_hidden_dim, bias=False
        )
        self.down_proj = nn.Linear(
            config.ffn_hidden_dim, config.hidden_dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down( SiLU(gate(x)) * up(x) ), SiLU(z) = z * sigmoid(z).
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
