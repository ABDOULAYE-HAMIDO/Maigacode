import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from model.tokenizer import BPE, CodeCorpus

output_path = Path("./data/training")
output_path.mkdir(parents=True, exist_ok=True)
tokenizer_path = Path("./checkpoints/tokenizer")
tokenizer_path.mkdir(parents=True, exist_ok=True)

# Load files from list
print("Loading files...")
with open("data/filelist.txt") as f:
    files = [line.strip() for line in f if line.strip()][:10000]

print(f"Reading {len(files)} files...")
texts = []
for fp in files:
    try:
        text = Path(fp).read_text(encoding="utf-8", errors="replace")
        if text.strip():
            texts.append(text)
    except:
        pass

print(f"Loaded {len(texts)} files, {sum(len(t) for t in texts):,} chars")

# Train tokenizer on subset
print("\nTraining BPE tokenizer...")
t0 = time.time()
tokenizer = BPE()
tokenizer.train(texts[:2000], vocab_size=5000, verbose=True)
tokenizer.save(str(tokenizer_path))
print(f"Tokenizer: {tokenizer.vocab_size()} tokens in {time.time()-t0:.1f}s")

# Tokenize all files
print(f"\nTokenizing {len(texts)} files...")
all_tokens = []
for i, text in enumerate(texts):
    all_tokens.extend(tokenizer.encode(text))
    if (i+1) % 2000 == 0:
        print(f"  {i+1}/{len(texts)} files, {len(all_tokens):,} tokens")

print(f"Total tokens: {len(all_tokens):,}")

# Create sequences
seq_len = 256
all_tokens = all_tokens[: (len(all_tokens) // seq_len) * seq_len]
data = torch.tensor(all_tokens, dtype=torch.long).view(-1, seq_len)

split = int(len(data) * 0.95)
torch.save(data[:split], output_path / "train.pt")
torch.save(data[split:], output_path / "val.pt")

print(f"\nTrain: {data[:split].shape}")
print(f"Val:   {data[split:].shape}")
print(f"Done!")
