import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))

output_dir = Path("./data/raw_big")
output_dir.mkdir(parents=True, exist_ok=True)

langs = [
    ("Python", ".py", 10000),
    ("JavaScript", ".js", 5000),
    ("Go", ".go", 3000),
    ("Rust", ".rs", 2000),
]

total = 0
for lang, ext, limit in langs:
    print(f"\n{lang}...", end=" ", flush=True)
    try:
        local = hf_hub_download(
            repo_id="hasankursun/github-code-2025-language-split",
            filename=f"{lang}/train-00000.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(local, columns=["content"])

        lang_dir = output_dir / lang.lower()
        lang_dir.mkdir(exist_ok=True)
        saved = 0

        for content in table.column("content"):
            if saved >= min(limit, len(table)):
                break
            c = content.as_py()
            if c and len(c) > 100 and len(c) < 10000:
                fp = lang_dir / f"{lang.lower()}_{saved:06d}{ext}"
                fp.write_text(c, encoding="utf-8", errors="replace")
                saved += 1

        total += saved
        print(f"{saved} files")

    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nTotal: {total} files")
print(f"Size: {sum(f.stat().st_size for f in output_dir.rglob('*')) / 1024**2:.1f} MB")
