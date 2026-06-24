import sys, time, json, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from model.model.config import ModelConfig
from model.model.model import SuperCodeurModel
from model.tokenizer import BPE
from model.training.trainer import Trainer
from model.data.prepare import prepare_data

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("  ENTRAÎNEMENT SUR DONNÉES RÉELLES")
print("=" * 60)

# Prepare data from real code
prepare_data(
    input_dir="./data/raw_big",
    output_path="./data/training",
    tokenizer_path="./checkpoints/tokenizer",
    vocab_size=32000,
    max_files=50000,
    seq_len=256,
)

# Dynamic vocab size
with open("./checkpoints/tokenizer/tokenizer.json") as f:
    vocab_size = json.load(f)["vocab_size"]
print(f"Tokenizer vocab: {vocab_size}")

CONFIG = ModelConfig(
    vocab_size=vocab_size,
    hidden_dim=128,
    num_layers=6,
    num_heads=8,
    num_kv_heads=4,
    head_dim=16,
    ffn_hidden_dim=352,
    max_seq_len=256,
    tie_embeddings=True,
)

model = SuperCodeurModel(CONFIG).to(device)
if device == "cuda" and hasattr(torch, "compile"):
    model = torch.compile(model)
print(f"Modèle: {model.count_params():,} params")

print("\nChargement du dataset...")
train_data = torch.load("./data/training/train.pt")
val_data = torch.load("./data/training/val.pt")
print(f"Train: {len(train_data)} sequences")
print(f"Val:   {len(val_data)} sequences")

batch_size = min(32, len(train_data))
num_workers = min(4, os.cpu_count() or 1)
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True,
    pin_memory=(device == "cuda"), num_workers=num_workers,
)
val_loader = DataLoader(
    val_data, batch_size=min(32, len(val_data)), shuffle=False,
    pin_memory=(device == "cuda"), num_workers=num_workers,
)
print(f"Batches: {len(train_loader)} train, {len(val_loader)} val")

grad_accum = max(1, 256 // batch_size)
trainer = Trainer(
    model=model, learning_rate=3e-4, weight_decay=0.1, grad_clip=1.0,
    grad_accum_steps=grad_accum, warmup_steps=100,
    precision="bf16" if device == "cuda" else "fp32",
)
nb = len(train_loader)

trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=5,
    log_interval=max(1, nb // 5),
    val_interval=max(1, nb),
    save_dir="./checkpoints",
)

# Test generation
best = Path("./checkpoints/best_model.pt")
if best.exists():
    tokenizer = BPE()
    tokenizer.load("./checkpoints/tokenizer")
    model = SuperCodeurModel(CONFIG).to(device)
    model.load_state_dict(torch.load(str(best), map_location=device)["model_state_dict"])

    for prompt in ["def fibonacci", "def add", "class Stack", "function ", "import "]:
        ids = tokenizer.encode(prompt)
        inp = torch.tensor([ids], device=device)
        out = model.generate(inp, max_new_tokens=60, temperature=0.6, top_k=40, top_p=0.9,
                            repetition_penalty=1.15, eos_token_id=2)
        print(f"\n>> {prompt}")
        print(tokenizer.decode(out[0].tolist())[:300])

print("\n✓ TERMINÉ")
