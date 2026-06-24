import os
import random
from pathlib import Path
from typing import Iterator, List, Optional, Union


EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.c', '.h',
    '.cpp', '.hpp', '.java', '.rb', '.php', '.swift', '.kt', '.scala',
    '.r', '.m', '.mm', '.hs', '.erl', '.ex', '.exs', '.clj', '.cljs',
    '.vue', '.svelte', '.css', '.scss', '.less', '.html', '.xml',
    '.yaml', '.yml', '.toml', '.json', '.md', '.rst', '.txt',
    '.sh', '.bash', '.zsh', '.ps1', '.bat', '.fish',
    '.sql', '.graphql', '.proto', '.zig', '.nim', '.cr',
    '.lua', '.pl', '.pm', '.t', '.sml', '.fs', '.fsx',
    '.dart', '.coffee', '.litcoffee',
}


def is_code_file(path: Path) -> bool:
    return path.suffix.lower() in EXTENSIONS and path.stat().st_size > 0


class CodeCorpus:
    def __init__(
        self,
        max_file_size: int = 1024 * 1024,
        encoding: str = 'utf-8',
    ):
        self.max_file_size = max_file_size
        self.encoding = encoding

    def from_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
        extensions: Optional[set] = None,
    ) -> List[str]:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        texts = []
        pattern = '**/*' if recursive else '*'

        for file_path in dir_path.glob(pattern):
            if not file_path.is_file():
                continue
            if extensions and file_path.suffix.lower() not in extensions:
                continue
            if not is_code_file(file_path):
                continue
            if file_path.stat().st_size > self.max_file_size:
                continue

            try:
                text = file_path.read_text(encoding=self.encoding, errors='replace')
                if text.strip():
                    texts.append(text)
            except Exception:
                pass

        return texts

    def from_file(self, file_path: Union[str, Path]) -> str:
        file_path = Path(file_path)
        return file_path.read_text(encoding=self.encoding, errors='replace')

    def from_files(self, file_paths: List[Union[str, Path]]) -> List[str]:
        texts = []
        for fp in file_paths:
            try:
                texts.append(self.from_file(fp))
            except Exception:
                pass
        return texts

    def from_text_list(self, texts: List[str]) -> List[str]:
        return texts

    def sample(
        self,
        texts: List[str],
        sample_size: int,
        seed: Optional[int] = None,
    ) -> List[str]:
        if seed is not None:
            random.seed(seed)
        return random.sample(texts, min(sample_size, len(texts)))

    def stats(self, texts: List[str]) -> dict:
        total_chars = sum(len(t) for t in texts)
        total_bytes = sum(len(t.encode('utf-8')) for t in texts)
        return {
            'num_files': len(texts),
            'total_chars': total_chars,
            'total_bytes': total_bytes,
            'avg_chars_per_file': total_chars // max(len(texts), 1),
        }
