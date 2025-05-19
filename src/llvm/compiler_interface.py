import subprocess
import os

def compile_cuda_to_ptx(source_path: str, output_path: str):
    """Compile a CUDA source file to PTX using NVCC."""
    print(f"🔧 Compiling {source_path} -> {output_path}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    result = subprocess.run(
        ["nvcc", "-ptx", source_path, "-o", output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("❌ nvcc compilation error:\n", result.stderr)
        raise RuntimeError(f"Compilation failed: {result.stderr}")

    print(f"✅ Compilation successful: {output_path}")
    return output_path
