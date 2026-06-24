#!/usr/bin/env python3
"""
Colab training script — run on Google Colab (T4/V100/A100).
Usage:
  !python run_colab.py --config 100M --epochs 5 --batch_size 8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from model.model.config import ModelConfig
from model.model.model import SuperCodeurModel
from model.tokenizer import BPE, CodeCorpus
from model.training.trainer import Trainer
from model.data.prepare import prepare_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="100M", choices=["nano", "10M", "50M", "100M", "300M", "350M"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--no_grad_ckpt", action="store_true", help="Disable gradient checkpointing (faster, more VRAM)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (A100/H100 only, not T4)")
    parser.add_argument("--max_files", type=int, default=50000)
    parser.add_argument("--data_dir", type=str, default="./data/raw_big")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    if device == "cuda":
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    tokenizer_path = output_dir / "tokenizer"
    data_path = output_dir / "data"

    data_exists = (data_path / "train.pt").exists()
    tokenizer_exists = (tokenizer_path / "tokenizer.json").exists()

    if not data_exists or not tokenizer_exists:
        print("\n=== Preparing data ===")
        prepare_data(
            input_dir=args.data_dir,
            output_path=str(data_path),
            tokenizer_path=str(tokenizer_path),
            vocab_size=32000,
            max_files=args.max_files,
            seq_len=args.seq_len,
        )

    with open(tokenizer_path / "tokenizer.json") as f:
        vocab_size = json.load(f)["vocab_size"]

    # ── Model ─────────────────────────────────────────────────────────────
    config = ModelConfig.from_preset(args.config, vocab_size=vocab_size)
    config.max_seq_len = args.seq_len
    # Gradient checkpointing on by default on GPU: keeps activation memory low
    # enough to fit 300M+ on a 16 GB T4. Disable with --no_grad_ckpt.
    config.gradient_checkpointing = (device == "cuda") and not args.no_grad_ckpt
    print(f"\n=== Model config: {args.config} ===")
    print(config)
    print(f"Expected params: {config.total_params_estimated:,}")

    model = SuperCodeurModel(config).to(device)
    # torch.compile is OFF by default: on a T4 it gives no speedup (no native
    # bf16) and eats ~2 GB. Enable only on A100/H100 with --compile.
    if args.compile and device == "cuda":
        model = torch.compile(model)
        print("torch.compile enabled")
    print(f"Actual params: {model.count_params():,}")

    # ── DataLoader ────────────────────────────────────────────────────────
    print("\n=== Loading dataset ===")
    train_data = torch.load(data_path / "train.pt")
    val_data = torch.load(data_path / "val.pt")
    print(f"Train: {len(train_data)} sequences")
    print(f"Val:   {len(val_data)} sequences")

    # Drop last batch if not full (avoid shape errors in compiled mode)
    train_data = train_data[:len(train_data) - len(train_data) % args.batch_size] if len(train_data) % args.batch_size != 0 else train_data

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
        pin_memory=(device == "cuda"), num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=(device == "cuda"), num_workers=num_workers,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Trainer ───────────────────────────────────────────────────────────
    grad_accum = max(1, 512 // args.batch_size)  # ~512 tokens per step
    max_steps = len(train_loader) * args.epochs // grad_accum
    warmup_steps = min(200, max_steps // 10)

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=0.1,
        grad_clip=1.0,
        grad_accum_steps=grad_accum,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        betas=(0.9, 0.95),
        precision="bf16" if (device == "cuda" and torch.cuda.is_bf16_supported()) else "fp16" if device == "cuda" else "fp32",
    )

    print(f"\n=== Training ({args.config}, {args.epochs} epochs) ===")
    print(f"  Grad accum: {grad_accum}, Warmup: {warmup_steps}, Max steps: {max_steps}")
    print(f"  Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print(f"  Effective batch: {args.batch_size * grad_accum} seq/step")

    log_interval = max(1, (len(train_loader) // grad_accum) // 10)
    val_interval = max(1, len(train_loader) // grad_accum)
    t0 = time.time()
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        log_interval=log_interval,
        val_interval=val_interval,
        save_dir=str(output_dir),
    )
    print(f"\nTraining time: {(time.time() - t0) / 60:.1f} minutes")

    # ── Test generation ───────────────────────────────────────────────────
    best = output_dir / "best_model.pt"
    if best.exists():
        print("\n=== Testing generation ===")
        tokenizer = BPE()
        tokenizer.load(str(tokenizer_path))
        model = SuperCodeurModel(config).to(device)
        model.load_state_dict(torch.load(str(best), map_location=device)["model_state_dict"])

        prompts = ["def fibonacci", "def add", "class Stack", "function hello", "import numpy"]
        for prompt in prompts:
            inp = torch.tensor([tokenizer.encode(prompt)], device=device)
            out = model.generate(inp, max_new_tokens=80, temperature=0.6, top_k=40, top_p=0.9,
                                repetition_penalty=1.15, eos_token_id=2)
            print(f"\n>> {prompt}\n{tokenizer.decode(out[0].tolist())}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
