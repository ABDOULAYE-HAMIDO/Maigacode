import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
from model.tokenizer.corpus import EXTENSIONS

output_dir = Path("./data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading code dataset...")

try:
    ds = load_dataset(
        "codeparrot/github-code",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
except Exception as e:
    print(f"Error: {e}")
    print("\nCreating synthetic dataset instead...")
    from model.data.prepare import prepare_data
    
    # Create basic samples
    samples = [
        # Python
        "def add(a, b): return a + b\n",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()\n",
        "import json\nimport os\n\ndef load_config(path):\n    with open(path, 'r') as f:\n        return json.load(f)\n",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)\n",
        # JavaScript
        "function greet(name) {\n    return `Hello, ${name}!`;\n}\n",
        "const debounce = (fn, delay) => {\n    let timeout;\n    return (...args) => {\n        clearTimeout(timeout);\n        timeout = setTimeout(() => fn(...args), delay);\n    };\n};\n",
        "class EventEmitter {\n    constructor() {\n        this.events = {};\n    }\n    emit(event, ...args) {\n        const listeners = this.events[event] || [];\n        listeners.forEach(l => l(...args));\n    }\n}\n",
        # Rust
        "pub fn factorial(n: u64) -> u64 {\n    match n {\n        0 | 1 => 1,\n        _ => n * factorial(n - 1),\n    }\n}\n",
        "pub struct Point {\n    x: f64,\n    y: f64,\n}\nimpl Point {\n    pub fn distance(&self, other: &Point) -> f64 {\n        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()\n    }\n}\n",
        # Go
        "package main\nimport \"fmt\"\nfunc main() {\n    fmt.Println(\"Hello, World!\")\n}\n",
        "package main\nfunc add(a, b int) int { return a + b }\n",
        "type Server struct {\n    Port int\n    Host string\n}\nfunc NewServer(port int) *Server {\n    return &Server{Port: port, Host: \"localhost\"}\n}\n",
        # Java
        "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello!\");\n    }\n}\n",
        "public class Calculator {\n    public int add(int a, int b) { return a + b; }\n    public int sub(int a, int b) { return a - b; }\n}\n",
        # C
        "#include <stdio.h>\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}\n",
        "int factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}\n",
        # C++
        "#include <iostream>\nint main() {\n    std::cout << \"Hello\" << std::endl;\n    return 0;\n}\n",
        "class Stack {\nprivate:\n    int items[100];\n    int top;\npublic:\n    Stack() : top(-1) {}\n    void push(int x) { items[++top] = x; }\n    int pop() { return items[top--]; }\n};\n",
        # TypeScript
        "interface User {\n    name: string;\n    age: number;\n    email: string;\n}\nfunction greet(user: User): string {\n    return `Hello, ${user.name}!`;\n}\n",
        "type Predicate<T> = (value: T) => boolean;\nfunction filter<T>(arr: T[], pred: Predicate<T>): T[] {\n    return arr.filter(pred);\n}\n",
        # More Python
        "def read_csv(path):\n    import csv\n    with open(path, 'r') as f:\n        return list(csv.DictReader(f))\n",
        "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\n",
        "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance\n",
        "def to_binary(n):\n    if n == 0: return '0'\n    result = ''\n    while n > 0:\n        result = str(n % 2) + result\n        n //= 2\n    return result\n",
        "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n",
        "def chunks(lst, n):\n    for i in range(0, len(lst), n):\n        yield lst[i:i + n]\n",
        "class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = {}\n        self.order = []\n    def get(self, key):\n        if key in self.cache:\n            self.order.remove(key)\n            self.order.append(key)\n            return self.cache[key]\n        return -1\n",
        "def timer(fn):\n    import time\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = fn(*args, **kwargs)\n        print(f'{fn.__name__}: {time.time() - start:.4f}s')\n        return result\n    return wrapper\n",
    ]
    
    for i, sample in enumerate(samples):
        with open(output_dir / f"sample_{i:04d}.py", "w") as f:
            f.write(sample)
    
    print(f"Created {len(samples)} synthetic files")

count = len(list(output_dir.glob("*")))
total_bytes = sum(f.stat().st_size for f in output_dir.glob("*"))
print(f"\nTotal: {count} files, {total_bytes/1024:.1f} KB")
print("Ready for training!")
