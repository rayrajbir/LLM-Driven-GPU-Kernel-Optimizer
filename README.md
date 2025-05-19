# LLM-Accelerated CUDA Kernel Optimizer

An intelligent CUDA kernel optimization system powered by LLM-assisted transformations. This tool helps GPU developers improve code efficiency through automated optimization strategies, backed by performance benchmarking and visualization.

## ✨ Features

* ✅ Natural language prompts for CUDA optimization
* ✅ Multiple optimization strategies:
  * Shared memory utilization
  * Loop unrolling
  * Thread coarsening
  * Memory access patterns
* ✅ Dual interfaces: CLI & Gradio web UI
* ✅ Execution time benchmarking & performance visualization
* ✅ Auto-generated optimized CUDA kernels
* ✅ Comprehensive performance reports (JSON/CSV) and charts (PNG)

## 🗂️ Project Structure

```
.
├── src/
│   ├── kernels/
│   │   └── vector_add_runner.cu   # Base CUDA kernel
│   ├── ui/
│   │   ├── cli.py                 # CLI pipeline for optimization
│   │   └── web_app.py             # Gradio web UI interface
│   ├── profiler/
│   │   └── profiler.py            # Benchmark & chart generator
│   └── __init__.py
├── results/
│   ├── report_vector_add_runner_*.json   # Performance reports
│   ├── execution_time_vector_add_runner_*.png   # Charts
├── .gradio/
│   └── flagged/
│       └── dataset1.csv           # Gradio logging output
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA Toolkit
- NVIDIA GPU with CUDA support

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm-cuda-optimizer.git
   cd llm-cuda-optimizer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv op.env
   source op.env/bin/activate  # On Windows: op.env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify CUDA is installed:
   ```bash
   nvcc --version
   nvidia-smi
   ```

## 🚀 Usage

### 🔧 CLI Mode

Run optimizations directly from the command line:

```bash
python src/ui/cli.py --src src/kernels/vector_add_runner.cu --opt auto --benchmark
```

**Optional flags:**
- `--report`: Save performance report in JSON format
- `--exe`: Execute transformed kernel and compare with original
- `--opt <strategy>`: Apply a specific optimization (e.g., `shared_memory`, `loop_unroll`)

### 🌐 Gradio Web UI

Launch the web interface for interactive optimization:

```bash
python src/ui/web_app.py
```

Visit: http://127.0.0.1:7860

Try prompts like:
- "Optimize using shared memory"
- "Use shared memory and unroll loops"
- "Optimize memory access patterns and thread utilization"

## 📦 Output

The tool generates several outputs to help you analyze optimizations:

- Transformed CUDA kernel files in `src/kernels/`
- Performance report JSONs in `results/`
- Execution time comparison charts as PNG
- CSV logs from Gradio UI at `.gradio/flagged/`

## 🧪 Sample Kernel

Example kernel in `src/kernels/vector_add_runner.cu`:

```cpp
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
```

## 🔍 Advanced Use Cases

- **Custom Kernel Optimization**: Submit your own CUDA kernels for optimization
- **Strategy Chaining**: Apply multiple optimization techniques in sequence
- **Performance Analysis**: Compare execution times across different strategies
- **LLM-guided Optimization**: Get natural language explanations for optimization choices

## 🧠 Troubleshooting

- Make sure matplotlib is installed for chart generation
- Use `sys.executable` in web_app.py for correct Python environment
- Ensure PYTHONPATH is set to project root when calling subprocess
- Check CUDA compatibility with: `python -c "import torch; print(torch.cuda.is_available())"`

## 📌 Roadmap

- Upload custom kernels via UI
- Optimization diff view (original vs optimized)
- Auto-tuning + multi-GPU support
- LLM chat integration for optimization explanation
- Support for more complex GPU algorithms (reduction, scan, etc.)
- Integration with popular DL frameworks (PyTorch, TensorFlow)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

Your Name - rajbirray701@gmail.com

Project Link: [https://github.com/rayrajbir/llm-cuda-optimizer](https://github.com/rayrajbir/LLM-Driven-GPU-Kernel-Optimizer.git)
