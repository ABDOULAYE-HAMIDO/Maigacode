import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model.model.config import ModelConfig
from model.model.model import SuperCodeurModel

config = ModelConfig(
    vocab_size=50000,
    hidden_dim=64,
    num_layers=3,
    num_heads=4,
    num_kv_heads=2,
    head_dim=16,
    ffn_hidden_dim=176,
    max_seq_len=256,
    tie_embeddings=True,
)

print(f"Nano Model: {config.total_params_estimated:,} params")

model = SuperCodeurModel(config)
print(f"  Réel: {model.count_params():,} params")

x = torch.randint(0, 1000, (2, 32))
with torch.no_grad():
    out = model(x)
print(f"  Forward: {out.shape}")

out = model.generate(x[:1, :8], max_new_tokens=10, temperature=0.8)
print(f"  Generate: {out.shape}")

print(f"\nModèle nano prêt — {model.count_params()/1e6:.2f}M paramètres, entraînable sur PC")
