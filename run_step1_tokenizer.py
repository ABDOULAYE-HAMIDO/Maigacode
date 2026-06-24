import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from model.tokenizer import BPE

files = list(Path("data/raw_big").rglob("*.*"))[:2000]
texts = []
for f in files:
    try:
        texts.append(f.read_text(encoding="utf-8", errors="replace"))
    except:
        pass

print(f"Loaded {len(texts)} files")

tokenizer = BPE()
tokenizer.train(texts, vocab_size=5000, verbose=True)

Path("checkpoints/tokenizer").mkdir(parents=True, exist_ok=True)
tokenizer.save("checkpoints/tokenizer")
print(f"Done! Vocab: {tokenizer.vocab_size()} tokens")
