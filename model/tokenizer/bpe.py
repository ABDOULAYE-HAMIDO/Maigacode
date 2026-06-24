import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import regex as re


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPE:
    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}
        self.pattern: str = PATTERN
        self.compiled_pattern = re.compile(self.pattern)

    def _pre_tokenize(self, text: str) -> List[str]:
        words = []
        for match in self.compiled_pattern.finditer(text):
            words.append(match.group())
        if not words:
            words = [text]
        return words

    def _byte_encode(self, text: str) -> bytes:
        return text.encode('utf-8')

    def train(
        self,
        texts: List[str],
        vocab_size: int = 50000,
        special_tokens: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        if special_tokens is None:
            special_tokens = ["<|BOS|>", "<|EOS|>", "<|PAD|>", "<|UNK|>"]

        next_id = 256
        for token in special_tokens:
            self.special_tokens[token] = next_id
            self.inverse_special_tokens[next_id] = token
            self.vocab[next_id] = token.encode('utf-8')
            next_id += 1

        if verbose:
            print(f"[Tokenizer] Base vocab: {next_id} tokens ({256} bytes + {len(special_tokens)} special)")

        all_word_freqs = Counter()
        total_texts = len(texts)

        for idx, text in enumerate(texts):
            words = self._pre_tokenize(text)
            for word in words:
                byte_ids = tuple(self._byte_encode(word))
                all_word_freqs[byte_ids] += 1
            if verbose and (idx + 1) % 1000 == 0:
                print(f"[Tokenizer] Processed {idx + 1}/{total_texts} texts")

        if verbose:
            print(f"[Tokenizer] Corpus: {len(all_word_freqs)} unique words")
            print(f"[Tokenizer] Starting BPE merges (target: {vocab_size} tokens)...")

        num_merges = vocab_size - next_id

        # Build initial pair frequency counter from all words.
        stats = Counter()
        word_ids_cache = {}
        for word_bytes, freq in all_word_freqs.items():
            ids = list(word_bytes)
            word_ids_cache[word_bytes] = ids
            for pair in zip(ids, ids[1:]):
                stats[pair] += freq

        for i in range(num_merges):
            if not stats:
                if verbose:
                    print(f"[Tokenizer] No more pairs to merge at step {i}")
                break

            most_frequent_pair = stats.most_common(1)[0][0]

            new_id = next_id
            next_id += 1
            self.merges[most_frequent_pair] = new_id

            token0 = self.vocab[most_frequent_pair[0]]
            token1 = self.vocab[most_frequent_pair[1]]
            self.vocab[new_id] = token0 + token1

            # Incrementally update: only modify words that contain the merged pair.
            new_word_freqs = Counter()
            stats = +stats

            for word_bytes, freq in list(all_word_freqs.items()):
                ids = word_ids_cache.get(word_bytes)
                if ids is None:
                    ids = list(word_bytes)
                    word_ids_cache[word_bytes] = ids

                # Check if this word contains the pair (fast scan).
                contains = False
                for j in range(len(ids) - 1):
                    if ids[j] == most_frequent_pair[0] and ids[j + 1] == most_frequent_pair[1]:
                        contains = True
                        break
                if not contains:
                    new_word_freqs[word_bytes] = freq
                    continue

                # Remove old pair counts for this word.
                for pair in zip(ids, ids[1:]):
                    stats[pair] -= freq
                    if stats[pair] <= 0:
                        del stats[pair]

                # Apply merge to this word.
                new_ids = tuple(merge(ids, most_frequent_pair, new_id))
                word_ids_cache[word_bytes] = list(new_ids)
                new_word_freqs[new_ids] += freq

                # Add new pair counts for the merged word.
                for pair in zip(new_ids, new_ids[1:]):
                    stats[pair] += freq

            all_word_freqs = new_word_freqs

            if verbose and (i + 1) % 1000 == 0:
                print(f"[Tokenizer] Merged {i + 1}/{num_merges} pairs (vocab: {next_id})")

        if verbose:
            print(f"[Tokenizer] Training complete. Vocab size: {len(self.vocab)}")

    def encode_word(self, word: str) -> List[int]:
        word_bytes = list(word.encode('utf-8'))
        while len(word_bytes) >= 2:
            best_pair = None
            best_priority = None
            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                if pair in self.merges:
                    priority = self.merges[pair]
                    if best_priority is None or priority < best_priority:
                        best_priority = priority
                        best_pair = i
            if best_pair is None:
                break
            new_id = self.merges[(word_bytes[best_pair], word_bytes[best_pair + 1])]
            word_bytes = (
                word_bytes[:best_pair]
                + [new_id]
                + word_bytes[best_pair + 2:]
            )
        return word_bytes

    def encode(self, text: str) -> List[int]:
        words = self._pre_tokenize(text)
        tokens = []
        for word in words:
            word_tokens = self.encode_word(word)
            tokens.extend(word_tokens)
        return tokens

    def decode(self, ids: List[int]) -> str:
        byte_parts = []
        for token_id in ids:
            if token_id in self.inverse_special_tokens:
                byte_parts.append(
                    self.inverse_special_tokens[token_id].encode('utf-8')
                )
            elif token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
            else:
                byte_parts.append(b"\xef\xbf\xbd")
        return b"".join(byte_parts).decode('utf-8', errors='replace')

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        merges_list = [
            [int(k[0]), int(k[1]), int(v)] for k, v in self.merges.items()
        ]

        data = {
            'pattern': self.pattern,
            'merges': merges_list,
            'special_tokens': dict(self.special_tokens),
            'vocab_size': len(self.vocab),
        }

        with open(path / 'tokenizer.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        with open(path / 'merges.txt', 'w', encoding='utf-8') as f:
            for (a, b), new_id in sorted(self.merges.items(), key=lambda x: x[1]):
                f.write(f"{a} {b}\n")

    def load(self, path: Union[str, Path]):
        path = Path(path)

        with open(path / 'tokenizer.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.pattern = data.get('pattern', PATTERN)
        self.compiled_pattern = re.compile(self.pattern)

        self.vocab = {i: bytes([i]) for i in range(256)}

        self.special_tokens = {}
        self.inverse_special_tokens = {}
        for name, token_id in data['special_tokens'].items():
            self.special_tokens[name] = token_id
            self.inverse_special_tokens[token_id] = name
            self.vocab[token_id] = name.encode('utf-8')

        self.merges = {}
        for a, b, new_id in data['merges']:
            self.merges[(a, b)] = new_id
            token0 = self.vocab[a]
            token1 = self.vocab[b]
            self.vocab[new_id] = token0 + token1

    def vocab_size(self) -> int:
        return len(self.vocab)
