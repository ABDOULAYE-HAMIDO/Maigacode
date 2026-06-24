import sys, json, time, torch, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from torch.utils.data import DataLoader, TensorDataset
from model.model.config import ModelConfig
from model.model.model import SuperCodeurModel
from model.training.trainer import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

# Load only 30K sequences for fast training
train_data = torch.load("data/training/train.pt")[:30000]
val_data = torch.load("data/training/val.pt")[:1000]

with open("checkpoints/tokenizer/tokenizer.json") as f:
    vocab_size = json.load(f)["vocab_size"]

config = ModelConfig(
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

model = SuperCodeurModel(config).to(device)
if device == "cuda" and hasattr(torch, "compile"):
    model = torch.compile(model)
    print("  torch.compile activé")
print(f"Modele: {model.count_params():,} params")

bs = min(16, len(train_data))
num_workers = min(4, os.cpu_count() or 1)
train_loader = DataLoader(
    train_data, batch_size=bs, shuffle=True, drop_last=True,
    pin_memory=(device == "cuda"), num_workers=num_workers,
)
val_loader = DataLoader(
    val_data, batch_size=bs, shuffle=False,
    pin_memory=(device == "cuda"), num_workers=num_workers,
)
print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

grad_accum = max(1, 256 // bs)  # target ~256 tokens per step
trainer = Trainer(
    model=model, learning_rate=3e-4, weight_decay=0.1, grad_clip=1.0,
    grad_accum_steps=grad_accum, warmup_steps=100,
    precision="bf16" if device == "cuda" else "fp32",
)
nb = len(train_loader)

trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=3,
    log_interval=max(1, nb // 5),
    val_interval=nb,
    save_dir="./checkpoints",
)

# Generation test
best = Path("checkpoints/best_model.pt")
if best.exists():
    from model.tokenizer import BPE
    tokenizer = BPE()
    tokenizer.load("checkpoints/tokenizer")
    model = SuperCodeurModel(config).to(device)
    model.load_state_dict(torch.load(str(best), map_location=device)["model_state_dict"])
    for prompt in ["def fibonacci", "def add", "function hello"]:
        inp = torch.tensor([tokenizer.encode(prompt)], device=device)
        out = model.generate(inp, max_new_tokens=40, temperature=0.6, top_k=40,
                             repetition_penalty=1.15, eos_token_id=2)
        print(f"\n>> {prompt}\n{tokenizer.decode(out[0].tolist())[:250]}")

print(f"\nTermine en {time.time()-t0:.1f}s")
