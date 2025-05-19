import gradio as gr
import subprocess
import os
import json
import glob
import sys
from pathlib import Path

def optimize_kernel(prompt):
    try:
        # 1. Define exact paths based on your project structure
        project_root = r"C:\Users\User\Desktop\llvm-cuda-optimizer"
        kernel_path = os.path.join(project_root, "src", "kernels", "vector_add_runner.cu")
        cli_path = os.path.join(project_root, "src", "ui", "cli.py")
        
        # 2. Create results directory if it doesn't exist
        results_dir = os.path.join(project_root, "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            print(f"Created results directory at: {results_dir}")
        
        # 3. Double-check file existence
        if not os.path.exists(kernel_path):
            return f"❌ Kernel file not found at: {kernel_path}", None
            
        if not os.path.exists(cli_path):
            return f"❌ CLI script not found at: {cli_path}", None

        # 4. Prepare command with exact paths
        command = [
            sys.executable,  # Use the current Python interpreter
            cli_path,
            "--src", kernel_path,
            "--opt", "auto",
            "--benchmark",
            "--report"
        ]

        # 5. Set environment variables to ensure proper module imports
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root  # Set project root in PYTHONPATH
        
        # 6. Run the CLI command with proper environment
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            error_message = f"❌ CLI error (return code {result.returncode}):\n\n"
            error_message += f"STDERR:\n{result.stderr}\n\n"
            error_message += f"STDOUT:\n{result.stdout}"
            return error_message, None

        # 7. Find latest report file
        report_files = sorted(glob.glob(os.path.join(results_dir, "report_vector_add_runner_*.json")), reverse=True)
        
        if not report_files:
            return "⚠️ No report file found in the results directory. Check CLI output for errors.", None

        # 8. Parse and display report
        with open(report_files[0], 'r') as f:
            report = json.load(f)

        summary = f"### 📊 Kernel: `{report['kernel']}`\n"
        summary += f"**Timestamp**: {report['timestamp']}\n\n"
        summary += "#### Execution Times (ms):\n"
        for opt, time in report["results"].items():
            summary += f"- **{opt}**: {time:.4f} ms\n"

        if report.get("speedups"):
            summary += "\n#### Speedups over baseline:\n"
            for opt, sp in report["speedups"].items():
                summary += f"- **{opt}**: {sp:.2f}×\n"

        # 9. Get latest chart image
        chart_files = sorted(glob.glob(os.path.join(results_dir, "execution_time_vector_add_runner*.png")), reverse=True)
        chart_path = chart_files[0] if chart_files else None

        return summary, chart_path

    except Exception as e:
        import traceback
        return f"❌ Exception: {str(e)}\n\n{traceback.format_exc()}", None

iface = gr.Interface(
    fn=optimize_kernel,
    inputs=gr.Textbox(label="Optimization Prompt", placeholder="E.g., Use shared memory and unroll loops"),
    outputs=[
        gr.Markdown(label="Optimization Summary"),
        gr.Image(label="Execution Time Chart")
    ],
    title="🚀 LLM-Accelerated CUDA Kernel Optimizer",
    description="Analyze and optimize your CUDA kernels using LLM-assisted strategies.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(debug=True)