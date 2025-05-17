# 🧠 LLVM-CUDA-NLP: LLM-Driven GPU Kernel Optimizer

A research-grade prototype that uses NLP (via LLMs) to generate and optimize CUDA GPU kernels, compiled with LLVM for performance analysis. This project bridges high-level natural language requests with low-level GPU code using a combination of transformers, CUDA, and LLVM IR passes.

---

## 🚀 Project Goals

* 🗣️ Accept natural language input like:

  > "Optimize this kernel for matrix multiplication with shared memory and warp-level primitives"

* 🤖 Use an LLM (like FLAN-T5 or OpenAI) to:

  * Interpret the optimization request
  * Suggest kernel transformations or parameters
  * Optionally generate new CUDA code

* ⚙️ Compile the CUDA kernel using `nvcc` and analyze with LLVM passes.

* 📊 Provide hooks for performance tuning and benchmarking.

---

## 🧪 Requirements

### Python

* `transformers`
* `torch`
* `openai` *(optional)*

Install with:

```bash
pip install -r requirements.txt
```

---

### System

* **CUDA Toolkit** (with `nvcc`)
* **LLVM** (v10+)
* **CMake** (v3.10+)

---

## 📁 Project Structure

```
LLVM-CUDA-NLP/
├── models/                  # LLM interfaces and prompt templates
├── kernels/                 # CUDA kernel sources
├── llvm_passes/            # LLVM IR manipulation and analysis passes
├── benchmarks/             # Performance tests and metrics
├── utils/                  # Utilities for I/O, compilation, logging
├── main.py                 # Main CLI entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Usage Example

```bash
python main.py \
  --prompt "Optimize for 2D convolution with shared memory and minimal bank conflicts" \
  --input-kernel kernels/conv.cu \
  --output-kernel kernels/conv_optimized.cu
```

You can also use the OpenAI API for enhanced interpretation:

```bash
python main.py --use-openai --api-key <your-key> [...]
```

---

## 🔍 Features

* LLM-powered prompt interpretation
* Auto-suggestion of CUDA best practices (e.g., loop unrolling, memory coalescing)
* LLVM IR analysis hooks (e.g., register pressure, instruction counts)
* Integration with performance profilers like Nsight
* Extensible for reinforcement learning-based tuning (future)

---

## 📈 Benchmarking

Benchmark utilities provided in `benchmarks/`:

```bash
python benchmarks/benchmark_runner.py --kernel kernels/conv_optimized.cu
```

Results include:

* Execution time
* Occupancy
* Warp efficiency
* Shared memory utilization

---

## 🤝 Contributing

We welcome PRs and ideas! To contribute:

1. Fork the repository
2. Create a new branch
3. Submit a pull request with a clear description

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙌 Acknowledgments

* OpenAI for GPT APIs
* Hugging Face Transformers
* NVIDIA CUDA Toolkit
* LLVM community
