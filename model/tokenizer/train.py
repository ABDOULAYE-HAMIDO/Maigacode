import argparse
import sys
import time
from pathlib import Path

from .bpe import BPE
from .corpus import CodeCorpus


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer from scratch")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing code files for training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tokenizer_output",
        help="Directory to save the trained tokenizer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50000,
        help="Target vocabulary size (default: 50000)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to use for training",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="File extensions to include (e.g. .py .js .rs)",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="+",
        default=["<|BOS|>", "<|EOS|>", "<|PAD|>", "<|UNK|>", "<|SEP|>", "<|CLS|>"],
        help="Special tokens for the tokenizer",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' not found")
        sys.exit(1)

    print("=" * 60)
    print("  BPE TOKENIZER TRAINING")
    print("=" * 60)
    print(f"  Input:      {input_dir}")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Extensions: {args.extensions or 'all code files'}")
    print(f"  Max files:  {args.max_files or 'unlimited'}")
    print("=" * 60)

    corpus = CodeCorpus()
    texts = corpus.from_directory(
        input_dir,
        recursive=True,
        extensions=set(args.extensions) if args.extensions else None,
    )

    if args.max_files:
        texts = corpus.sample(texts, args.max_files)

    stats = corpus.stats(texts)
    print(f"\n  Loaded {stats['num_files']} files")
    print(f"  Total chars: {stats['total_chars']:,}")
    print(f"  Total bytes: {stats['total_bytes']:,}")

    if stats['num_files'] == 0:
        print("Error: No code files found in the input directory")
        sys.exit(1)

    tokenizer = BPE()
    special = args.special_tokens

    print(f"\n  Special tokens: {special}")
    print(f"  Starting training...\n")

    start_time = time.time()

    tokenizer.train(
        texts=texts,
        vocab_size=args.vocab_size,
        special_tokens=special,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")

    output_dir = Path(args.output_dir)
    tokenizer.save(output_dir)
    print(f"  Tokenizer saved to: {output_dir}/")
    print(f"  Vocab size: {tokenizer.vocab_size()}")

    sample_text = "def hello_world():\n    print('hello, world!')"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n  Encoding test:")
    print(f"    Input:  {sample_text}")
    print(f"    Tokens: {encoded}")
    print(f"    Decoded: {decoded}")
    print(f"    Compression: {len(sample_text.encode('utf-8'))} bytes -> {len(encoded)} tokens")
    print("=" * 60)


if __name__ == "__main__":
    main()
