import sys
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from model.tokenizer.corpus import EXTENSIONS

output_dir = Path("./data/raw_big")
output_dir.mkdir(parents=True, exist_ok=True)

# Top languages for code training
files_to_download = [
    ("data/python/data-00000-of-00001.parquet", ".py"),
    ("data/javascript/data-00000-of-00001.parquet", ".js"),
    ("data/typescript/data-00000-of-00001.parquet", ".ts"),
    ("data/go/data-00000-of-00001.parquet", ".go"),
    ("data/rust/data-00000-of-00001.parquet", ".rs"),
    ("data/java/data-00000-of-00001.parquet", ".java"),
    ("data/c/data-00000-of-00001.parquet", ".c"),
    ("data/cpp/data-00000-of-00001.parquet", ".cpp"),
]

try:
    import pyarrow.parquet as pq
except ImportError:
    print("Installing pyarrow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
    import pyarrow.parquet as pq

total_saved = 0
max_per_lang = 20000  # Save at most 20K files per language

for parquet_path, ext in files_to_download:
    lang = parquet_path.split("/")[1]
    print(f"\nDownloading {lang}...")

    try:
        local_path = hf_hub_download(
            repo_id="bigcode/the-stack-dedup",
            filename=parquet_path,
            repo_type="dataset",
        )
        print(f"  Downloaded to: {local_path}")

        table = pq.read_table(local_path, columns=["content"])
        print(f"  Rows: {len(table)}")

        lang_output = output_dir / lang
        lang_output.mkdir(exist_ok=True)

        saved = 0
        for i, row in enumerate(table):
            if saved >= max_per_lang:
                break
            content = row["content"].as_py()
            if content and len(content) > 50 and len(content) < 10000:
                file_path = lang_output / f"{lang}_{saved:06d}{ext}"
                file_path.write_text(content, encoding="utf-8", errors="replace")
                saved += 1

        total_saved += saved
        print(f"  Saved: {saved} files")

    except Exception as e:
        print(f"  Error: {e}")

print(f"\n{'='*60}")
print(f"Total: {total_saved} files in {output_dir}")
print(f"Size: {sum(f.stat().st_size for f in output_dir.rglob('*')) / 1024**2:.1f} MB")
