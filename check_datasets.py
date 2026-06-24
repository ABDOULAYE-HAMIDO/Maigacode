from huggingface_hub import HfApi

api = HfApi()

datasets_to_check = [
    "hasankursun/github-code-2025-language-split",
    "sahil2801/CodeAlpaca-20k",
    "codeparrot/codeparrot-clean",
]

for ds in datasets_to_check:
    try:
        files = api.list_repo_files(ds, repo_type="dataset")
        parquet = [f for f in files if f.endswith(".parquet")]
        jsonl = [f for f in files if f.endswith(".jsonl")]
        print(f"\n{ds}:")
        print(f"  Total files: {len(files)}")
        print(f"  Parquet: {len(parquet)}")
        print(f"  JSONL: {len(jsonl)}")
        if parquet:
            print(f"  First: {parquet[0]}")
        if jsonl:
            print(f"  First: {jsonl[0]}")
    except Exception as e:
        print(f"\n{ds}: ERROR - {e}")
