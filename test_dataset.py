from datasets import load_dataset

print("Testing codeparrot/github-code...")
ds = load_dataset("codeparrot/github-code", split="train", streaming=True, trust_remote_code=True)
count = 0
for i, ex in enumerate(ds):
    if i >= 10:
        break
    code = ex.get("code", "")
    if code:
        lang = ex.get("language", "?")
        print(f"  Sample {i}: {len(code)} chars, lang={lang}")
        count += 1
print(f"OK: {count} samples loaded")
