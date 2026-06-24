import argparse
import json
import os
import sys
from pathlib import Path


def download_from_huggingface(output_dir: str, dataset: str, split: str, max_samples: int):
    print(f"Loading dataset '{dataset}' (split: {split}, max: {max_samples})...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    ds = load_dataset(dataset, split=split, streaming=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, example in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        content = None
        if 'content' in example:
            content = example['content']
        elif 'code' in example:
            content = example['code']
        elif 'text' in example:
            content = example['text']
        elif 'source' in example:
            content = example['source']
        else:
            continue

        if not content or not content.strip():
            continue

        lang = example.get('language', example.get('lang', 'unknown'))
        ext = LANG_EXT.get(lang.lower(), '.txt')

        file_path = output_path / f"sample_{i:07d}{ext}"
        try:
            file_path.write_text(content, encoding='utf-8', errors='replace')
            count += 1
        except Exception:
            pass

        if count > 0 and count % 1000 == 0:
            print(f"  Downloaded {count} files...")

    print(f"Done. Downloaded {count} code samples to '{output_dir}'")


def download_from_github(output_dir: str, language: str, max_repos: int):
    print(f"Downloading {language} code from GitHub (max repos: {max_repos})...")
    print("Please install and use 'gh' CLI for GitHub authentication.")
    print("Run: gh repo clone <repo> <dir>")


LANG_EXT = {
    'python': '.py', 'javascript': '.js', 'typescript': '.ts',
    'jsx': '.jsx', 'tsx': '.tsx', 'go': '.go', 'rust': '.rs',
    'c': '.c', 'cpp': '.cpp', 'c++': '.cpp', 'java': '.java',
    'ruby': '.rb', 'php': '.php', 'swift': '.swift', 'kotlin': '.kt',
    'scala': '.scala', 'r': '.r', 'dart': '.dart', 'lua': '.lua',
    'perl': '.pl', 'haskell': '.hs', 'elixir': '.exs',
    'clojure': '.clj', 'solidity': '.sol', 'zig': '.zig',
    'nim': '.nim', 'crystal': '.cr', 'sass': '.scss',
    'scss': '.scss', 'html': '.html', 'css': '.css',
    'shell': '.sh', 'bash': '.sh', 'powershell': '.ps1',
    'sql': '.sql', 'markdown': '.md', 'json': '.json',
    'yaml': '.yaml', 'xml': '.xml',
}


def download_from_the_stack(output_dir: str, max_samples: int):
    return download_from_huggingface(
        output_dir=output_dir,
        dataset="bigcode/the-stack-dedup",
        split="train",
        max_samples=max_samples,
    )


def download_from_codeparrot(output_dir: str, max_samples: int):
    return download_from_huggingface(
        output_dir=output_dir,
        dataset="codeparrot/github-code",
        split="train",
        max_samples=max_samples,
    )


AVAILABLE_DATASETS = {
    'the-stack': download_from_the_stack,
    'codeparrot': download_from_codeparrot,
}


def main():
    parser = argparse.ArgumentParser(description="Download code datasets for tokenizer training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="the-stack",
        choices=list(AVAILABLE_DATASETS.keys()),
        help="Dataset to download (default: the-stack)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Output directory for downloaded code",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Maximum number of code samples to download (default: 50000)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  CODE DATASET DOWNLOADER")
    print("=" * 60)
    print(f"  Dataset:   {args.dataset}")
    print(f"  Max files: {args.max_samples}")
    print(f"  Output:    {args.output_dir}")
    print("=" * 60)

    AVAILABLE_DATASETS[args.dataset](
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
