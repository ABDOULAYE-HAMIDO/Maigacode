import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Linear warmup followed by cosine decay to ``min_lr``."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


class Trainer:
    def __init__(
        self,
        model,
        device="cuda",
        learning_rate: float = 3e-4,
        min_lr: float = 3e-5,
        weight_decay: float = 0.1,
        betas: tuple = (0.9, 0.95),
        grad_clip: float = 1.0,
        warmup_steps: int = 100,
        max_steps: int = 10000,
        grad_accum_steps: int = 1,
        pad_token_id: int = -100,
        precision: str = "bf16",  # "bf16" | "fp16" | "fp32"
    ):
        self.model = model
        self.device = device
        self.grad_clip = grad_clip
        self.max_lr = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.pad_token_id = pad_token_id

        # Decoupled weight decay: only decay 2D weights (matmuls/embeddings),
        # not norms/biases — the standard nanoGPT recipe.
        decay, no_decay = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
            fused=(device == "cuda"),
        )

        # Mixed precision setup.
        use_cuda = device == "cuda" and torch.cuda.is_available()
        if precision == "bf16" and use_cuda and torch.cuda.is_bf16_supported():
            self.autocast = torch.autocast("cuda", dtype=torch.bfloat16)
            self.scaler = None
        elif precision == "fp16" and use_cuda:
            self.autocast = torch.autocast("cuda", dtype=torch.float16)
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.autocast = nullcontext()
            self.scaler = None

        self.step = 0

    def _set_lr(self):
        lr = get_lr(
            self.step,
            self.warmup_steps,
            self.max_steps,
            self.max_lr,
            self.min_lr,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    @torch.no_grad()
    def validate(self, val_loader, max_batches=50):
        self.model.eval()
        total_loss, n = 0.0, 0
        for x in val_loader:
            x = x.to(self.device, non_blocking=True)
            with self.autocast:
                _, loss = self.model(
                    x, targets=x, ignore_index=self.pad_token_id
                )
            total_loss += loss.item()
            n += 1
            if n >= max_batches:
                break
        self.model.train()
        return total_loss / max(n, 1)

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 1,
        log_interval: int = 50,
        val_interval: int = 500,
        save_dir: str = "./checkpoints",
    ):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.train()
        best_val_loss = float("inf")
        t0 = time.time()
        running = 0.0

        for epoch in range(num_epochs):
            print(f"\n{'='*60}\n  EPOCH {epoch + 1}/{num_epochs}\n{'='*60}")
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, x in enumerate(train_loader):
                x = x.to(self.device, non_blocking=True)
                with self.autocast:
                    _, loss = self.model(
                        x, targets=x, ignore_index=self.pad_token_id
                    )
                    loss = loss / self.grad_accum_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                running += loss.item() * self.grad_accum_steps

                # Optimizer step every grad_accum_steps micro-batches.
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    lr = self._set_lr()
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1

                    if self.step % log_interval == 0:
                        avg = running / (log_interval * self.grad_accum_steps)
                        running = 0.0
                        dt = time.time() - t0
                        tok = (
                            self.step
                            * x.size(0)
                            * x.size(1)
                            * self.grad_accum_steps
                        )
                        print(
                            f"  Step {self.step:>6d} | Loss: {avg:.4f} | "
                            f"LR: {lr:.2e} | {tok/dt/1e3:.1f}k tok/s"
                        )

                    if self.step % val_interval == 0:
                        val_loss = self.validate(val_loader)
                        star = "★" if val_loss < best_val_loss else " "
                        print(
                            f"  ── Validation | Loss: {val_loss:.4f} | {star}"
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save(save_dir / "best_model.pt", val_loss)
                            print("  ── Meilleur modèle sauvegardé !")

                    if self.step >= self.max_steps:
                        break

            val_loss = self.validate(val_loader)
            print(f"\n  Fin epoch {epoch + 1} | Val Loss: {val_loss:.4f}")
            if self.step >= self.max_steps:
                break

        print(f"\n  Entraînement terminé ! Best val loss: {best_val_loss:.4f}")
        return best_val_loss

    def save(self, path, val_loss):
        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model.config,
                "loss": val_loss,
            },
            path,
        )
