import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model.tokenizer import BPE, CodeCorpus


def prepare_data(
    input_dir: str,
    output_path: str,
    tokenizer_path: str,
    vocab_size: int = 50000,
    max_files: int = 5000,
    seq_len: int = 256,
    train_split: float = 0.95,
):
    print("=" * 60)
    print("  DATA PREPARATION")
    print("=" * 60)

    corpus = CodeCorpus()
    texts = corpus.from_directory(input_dir)

    if max_files and len(texts) > max_files:
        texts = corpus.sample(texts, max_files)

    stats = corpus.stats(texts)
    print(f"  Fichiers: {stats['num_files']}")
    print(f"  Taille:   {stats['total_bytes'] / 1024**2:.1f} MB")

    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.exists():
        print(f"  Chargement du tokenizer existant...")
        tokenizer = BPE()
        tokenizer.load(str(tokenizer_path))
    else:
        print(f"  Entraînement du tokenizer BPE...")
        tokenizer = BPE()
        tokenizer.train(texts, vocab_size=vocab_size, verbose=True)
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        print(f"  Tokenizer sauvegardé dans {tokenizer_path}")

    print(f"  Tokenisation des données...")
    all_tokens = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        if (i + 1) % 500 == 0:
            print(f"    Tokenisé {i+1}/{len(texts)} fichiers ({len(all_tokens):,} tokens)")

    print(f"  Total tokens: {len(all_tokens):,}")

    all_tokens = all_tokens[: (len(all_tokens) // seq_len) * seq_len]
    data = torch.tensor(all_tokens, dtype=torch.long)
    data = data.view(-1, seq_len)

    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(train_data, output_path / "train.pt")
    torch.save(val_data, output_path / "val.pt")

    print(f"\n  Dataset:")
    print(f"    Train: {train_data.shape} ({train_data.numel():,} tokens)")
    print(f"    Val:   {val_data.shape} ({val_data.numel():,} tokens)")
    print(f"    Sauvegardé dans {output_path}")

    return train_data, val_data, tokenizer


if __name__ == "__main__":
    prepare_data(
        input_dir="./data/raw",
        output_path="./data/training",
        tokenizer_path="./checkpoints/tokenizer",
        vocab_size=50000,
        max_files=5000,
        seq_len=256,
    )
