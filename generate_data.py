import random
from pathlib import Path

random.seed(42)
output_dir = Path("./data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

def gen_python(i):
    funcs = [
        f"def func_{i}(x, y):\n    result = x + y\n    if result > 100:\n        return result * 2\n    return result\n",
        f"class Class_{i}:\n    def __init__(self, value={i}):\n        self.value = value\n    def get_value(self):\n        return self.value\n    def set_value(self, v):\n        self.value = v\n",
        f"def parse_data_{i}(data):\n    import json\n    try:\n        parsed = json.loads(data)\n        return parsed.get('result', None)\n    except json.JSONDecodeError:\n        return None\n",
        f"def filter_items_{i}(items, threshold={i}):\n    filtered = []\n    for item in items:\n        if item > threshold:\n            filtered.append(item)\n        elif item == 0:\n            filtered.append(threshold)\n    return sorted(filtered, reverse=True)\n",
        f"def process_file_{i}(path):\n    with open(path, 'r') as f:\n        content = f.read()\n    lines = content.split('\\n')\n    return [(j, line) for j, line in enumerate(lines) if line]\n",
    ]
    return funcs[i % len(funcs)]

def gen_js(i):
    funcs = [
        f"function process_{i}(data) {{\n    const result = data.map(x => x * {i});\n    return result.filter(x => x > 0);\n}}\n",
        f"class Handler{i} {{\n    constructor() {{\n        this.events = new Map();\n    }}\n    on(event, cb) {{\n        if (!this.events.has(event)) this.events.set(event, []);\n        this.events.get(event).push(cb);\n    }}\n}}\n",
        f"const utils_{i} = {{\n    sum: (a, b) => a + b + {i},\n    product: (a, b) => a * b,\n    negate: x => -x,\n}};\n",
        f"async function fetch_{i}(url) {{\n    const response = await fetch(url);\n    const data = await response.json();\n    return {{ status: response.status, data }};\n}}\n",
    ]
    return funcs[i % len(funcs)]

def gen_rs(i):
    funcs = [
        f"pub fn compute_{i}(x: i32, y: i32) -> i32 {{\n    let result = x * y + {i};\n    if result > 100 {{\n        return result / 2;\n    }}\n    result\n}}\n",
        f"pub struct Container{i} {{\n    data: Vec<String>,\n    count: usize,\n}}\nimpl Container{i} {{\n    pub fn new() -> Self {{\n        Self {{ data: Vec::new(), count: 0 }}\n    }}\n}}\n",
        f"pub fn find_{i}(arr: &[i32], target: i32) -> Option<usize> {{\n    for (idx, val) in arr.iter().enumerate() {{\n        if *val == target {{\n            return Some(idx);\n        }}\n    }}\n    None\n}}\n",
    ]
    return funcs[i % len(funcs)]

def gen_go(i):
    funcs = [
        f"func Process{i}(items []int) []int {{\n    result := make([]int, 0)\n    for _, item := range items {{\n        if item > {i} {{\n            result = append(result, item*2)\n        }}\n    }}\n    return result\n}}\n",
        f"type Server{i} struct {{\n    Port int\n    Name string\n}}\n\nfunc NewServer{i}(port int) *Server{i} {{\n    return &Server{i}{{Port: port, Name: \"svc\"}}\n}}\n",
        f"func calculate{i}(a, b int) (int, error) {{\n    if b == 0 {{\n        return 0, fmt.Errorf(\"division by zero\")\n    }}\n    return (a + b) / b, nil\n}}\n",
    ]
    return funcs[i % len(funcs)]

def gen_java(i):
    funcs = [
        f"public class Processor{i} {{\n    private int value;\n    public Processor{i}(int v) {{ this.value = v; }}\n    public int compute(int x) {{ return x * value + {i}; }}\n}}\n",
        f"public interface Handler{i} {{\n    void handle(String event);\n    default void log() {{ System.out.println(\"handled\"); }}\n}}\n",
    ]
    return funcs[i % len(funcs)]

def gen_c(i):
    funcs = [
        f"int compute_{i}(int x, int y) {{\n    int result = x + y + {i};\n    if (result > 100) {{\n        return result / 2;\n    }}\n    return result;\n}}\n",
        f"typedef struct {{ int x; int y; }} Point{i};\nPoint{i} make_point{i}(int x, int y) {{ Point{i} p = {{x, y}}; return p; }}\n",
    ]
    return funcs[i % len(funcs)]

generators = {
    ".py": gen_python,
    ".js": gen_js,
    ".rs": gen_rs,
    ".go": gen_go,
    ".java": gen_java,
    ".c": gen_c,
}

count = 0
for i in range(1500):
    ext = random.choices(
        list(generators.keys()),
        weights=[0.35, 0.25, 0.15, 0.1, 0.1, 0.05],
        k=1,
    )[0]
    gen = generators[ext]
    code = gen(i)
    file_path = output_dir / f"sample_{count:06d}{ext}"
    file_path.write_text(code)
    count += 1

total_bytes = sum(f.stat().st_size for f in output_dir.glob("*"))
print(f"Generated {count} code files ({total_bytes/1024:.1f} KB)")
