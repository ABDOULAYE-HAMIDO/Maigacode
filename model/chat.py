import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .model.config import ModelConfig
from .model.model import SuperCodeurModel
from .tokenizer import BPE


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Conversation:
    def __init__(self, max_turns: int = 4, max_context_len: int = 512):
        self.turns: List[Tuple[str, str]] = []
        self.max_turns = max_turns
        self.max_context_len = max_context_len

    def add_turn(self, user: str, assistant: str):
        self.turns.append((user, assistant))
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def format_prompt(self, query: str) -> str:
        parts = []
        for user, assistant in self.turns:
            parts.append(f"User: {user}\nAssistant: {assistant}")
        parts.append(f"User: {query}\nAssistant:")
        return "\n\n".join(parts)

    def clear(self):
        self.turns.clear()


class SuperCodeurChat:
    def __init__(self, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model: Optional[SuperCodeurModel] = None
        self.tokenizer: Optional[BPE] = None
        self.config: Optional[ModelConfig] = None
        self.conversation = Conversation()
        self._max_tokens_generated = 200

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def find_checkpoint(self) -> Optional[Path]:
        paths = [
            PROJECT_ROOT / "checkpoints" / "best_model.pt",
            PROJECT_ROOT / "checkpoints" / "checkpoint_final.pt",
        ]
        for p in paths:
            if p.exists():
                return p
        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        if checkpoints_dir.exists():
            ckpts = sorted(checkpoints_dir.glob("*.pt"))
            if ckpts:
                return ckpts[-1]
        return None

    def load(self, config_name: str = "100M", checkpoint: Optional[str] = None):
        checkpoint_path = Path(checkpoint) if checkpoint else self.find_checkpoint()
        tokenizer_path = PROJECT_ROOT / "checkpoints" / "tokenizer"

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Train the tokenizer first with download_code.py + prepare_data."
            )

        self.tokenizer = BPE()
        self.tokenizer.load(str(tokenizer_path))
        vocab_size = self.tokenizer.vocab_size()

        self.config = ModelConfig.from_preset(config_name, vocab_size=vocab_size)
        self.model = SuperCodeurModel(self.config).to(self.device)

        if checkpoint_path and checkpoint_path.exists():
            state = torch.load(str(checkpoint_path), map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            print(f"Loaded checkpoint: {checkpoint_path.name} (val_loss: {state.get('loss', 'N/A'):.4f})")
        else:
            print(f"No checkpoint found. Using untrained {config_name} model.")

        self.model.eval()

    def respond(self, query: str, temperature: float = 0.6, top_k: int = 40,
                top_p: float = 0.9, repetition_penalty: float = 1.15,
                max_new_tokens: int = 200) -> str:
        if not self.is_loaded:
            return "⚠️ Modèle non chargé. Utilise `load()` d'abord."

        with torch.no_grad():
            prompt = self.conversation.format_prompt(query)
            input_ids = self.tokenizer.encode(prompt)
            if len(input_ids) > self.config.max_seq_len // 2:
                input_ids = input_ids[-(self.config.max_seq_len // 2):]

            inp = torch.tensor([input_ids], device=self.device)
            out = self.model.generate(
                inp,
                max_new_tokens=min(max_new_tokens, self._max_tokens_generated),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=2,
            )
            full = self.tokenizer.decode(out[0].tolist())
            response = full[len(self.tokenizer.decode(input_ids)):].strip()

        self.conversation.add_turn(query, response)
        return response

    def clear(self):
        self.conversation.clear()

    def set_max_tokens(self, n: int):
        self._max_tokens_generated = min(n, 1024)
