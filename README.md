# LLVM-CUDA Optimizer (Flask Edition)

An LLVM-accelerated, LLM-guided CUDA kernel optimizer with a robust Flask web interface. This tool empowers GPU developers to automatically transform and optimize CUDA code using techniques like shared memory, loop unrolling, and memory coalescing—backed by benchmarking and insightful visualizations.

---

## ✨ Features

- 🤖 LLM-assisted kernel analysis and optimization
- ⚙️ Multiple optimization strategies:
  - Shared memory utilization
  - Loop unrolling
  - Register usage tuning
  - Memory coalescing
- 🖥️ Web UI (Flask) + CLI mode
- 📊 Benchmarking, speedup charts & execution-time plots
- 📝 Auto-generated reports (JSON) and kernel diffs
- 🧠 Interactive & intelligent optimization suggestions (auto mode)

---

## 🗂️ Project Structure

```
.
├── src/
│   ├── kernels/                # CUDA kernels (original & transformed)
│   ├── ui/
│   │   └── cli.py      # CLI entrypoint
│   │   
│   ├── profiler/              # Benchmarking & chart generation
│   │   └── profiler.py
├── results/                   # JSON reports, PNG charts
│   ├── execution_time_*.png
│   ├── speedup_*.png
│   └── report_*.json
├── templates/                 # Flask templates (HTML)
│   ├── index.html
│   └── results.html
├── requirements.txt
└── README.md
└── app.py              # Flask-based web interface
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA Toolkit with nvcc
- NVIDIA GPU with compute capability

### Setup

```bash
git clone https://github.com/your-username/llvm-cuda-optimizer.git
cd llvm-cuda-optimizer

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Check your CUDA environment:

```bash
nvcc --version
nvidia-smi
```

---

## 🚀 Usage

### 🔧 CLI Mode

```bash
python src/ui/cli_patched.py --src src/kernels/your_kernel.cu --opt all --benchmark --report
```

Common options:
- `--src <file>`: Input CUDA kernel
- `--opt`: Optimization strategy (`shared_memory`, `unroll`, `register`, `memory_coalescing`, `auto`, `all`)
- `--benchmark`: Compare execution time with baseline
- `--report`: Output JSON + PNG charts
- `--interactive`: Get suggestions & choose interactively

Example:

```bash
python src/ui/cli_patched.py --src src/kernels/vector_add_runner.cu --opt auto --benchmark --report
```

### 🌐 Web App (Flask)

Launch the interface:

```bash
python src/ui/app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

You can:
- Upload a custom .cu file
- Choose optimizations interactively
- View speedup and execution time charts
- Download the performance report (JSON)

---

## 📈 Output Example

The tool produces:
- ✅ Transformed kernels: src/kernels/<name>_<opt>.cu
- ✅ Benchmarked reports: results/report_<kernel>_<timestamp>.json
- ✅ Execution time chart: results/execution_time_<kernel>.png
- ✅ Speedup chart: results/speedup_<kernel>.png

Example output (from report):

```json
"results": {
  "baseline": 1.866,
  "shared_memory": 1.089,
  "unroll": 0.711,
  "register": 0.785,
  "all_combined": 0.988
},
"speedups": {
  "shared_memory": 1.71,
  "unroll": 2.62,
  "register": 2.37,
  "all_combined": 1.88
}
```

---

## 🧪 Sample Kernel

Located at: `src/kernels/vector_add_runner.cu`

```cpp
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
```

---

## 🧠 Tips & Troubleshooting

- Kernels without loops or __global__ may be skipped during transformation
- Use `--interactive` to get suggestions when you're unsure
- Charts only generate if benchmark + report flags are used
- Make sure `nvcc` is in your system path
- Use `sys.executable` in subprocess for Python path consistency (web app)

---

## 🛣️ Roadmap

- [ ] Visual diff view (original vs optimized kernel)
- [ ] Multi-kernel queue in UI
- [ ] LLM explanation of applied optimizations
- [ ] Docker support & cloud deployment
- [ ] PyTorch kernel export hooks

---

## 🤝 Contributing

We welcome contributions!

```bash
git checkout -b feat/your-feature
# Make changes
git commit -m "Add your feature"
git push origin feat/your-feature
```

Then open a PR 🚀

---

## 📬 Contact

Maintainer: Rajbir Ray  
Email: rajbirray701@gmail.com  
GitHub: https://github.com/rayrajbir/LLVM-CUDA-GPU-Kernel-Optimizer.git
