import sys
import os
import argparse
import subprocess
import sys
print("🐍 sys.path:", sys.path)
import json
import time
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.kernels.runner_transformer import KernelRunnerTransformer
from src.profiler.profiler import profile_kernel
from src.llvm.compiler_interface import compile_cuda_to_ptx

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
print(f"Added {project_root} to Python path")
# Constants
RESULTS_DIR = "results"
KERNELS_DIR = "src/kernels"
BUILD_DIR = "build"

class OptimizationBenchmark:
    """Class to handle benchmarking different optimizations"""
    
    def __init__(self, kernel_path: str):
        self.kernel_path = kernel_path
        self.base_name = os.path.splitext(os.path.basename(kernel_path))[0]
        self.results = {}
        self.baseline_time = None
        
        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(BUILD_DIR, exist_ok=True)
        
    def run_baseline(self) -> float:
        """Compile and run the baseline kernel to get reference performance"""
        print("🔍 Running baseline kernel...")
        binary_path = os.path.join(BUILD_DIR, f"{self.base_name}_baseline.exe")
        
        if compile_kernel(self.kernel_path, binary_path):
            execution_time = profile_kernel(binary_path)
            self.baseline_time = execution_time
            self.results["baseline"] = execution_time
            print(f"⏱️  Baseline execution time: {execution_time:.4f} ms")
            return execution_time
        return None
        
    def run_optimization(self, opt_name: str, transformer_func, **kwargs) -> float:
        """Apply optimization, compile and run"""
        print(f"🔧 Applying {opt_name} optimization...")
        transformer = KernelRunnerTransformer(self.kernel_path)
        
        # Apply the optimization function
        transformer_func(transformer, **kwargs)
        
        # Add a comment about the optimization
        transformer.add_comment(f"Applied {opt_name} optimization with parameters: {kwargs}")
        
        # Output paths
        opt_kernel_path = os.path.join(KERNELS_DIR, f"{self.base_name}_{opt_name}.cu")
        opt_binary_path = os.path.join(BUILD_DIR, f"{self.base_name}_{opt_name}.exe")
        
        # Save the transformed kernel
        transformer.save(opt_kernel_path)
        print(f"📝 Saved optimized kernel to {opt_kernel_path}")
        
        # Compile and profile
        if compile_kernel(opt_kernel_path, opt_binary_path):
            execution_time = profile_kernel(opt_binary_path)
            self.results[opt_name] = execution_time
            
            # Calculate speedup
            if self.baseline_time:
                speedup = self.baseline_time / execution_time if execution_time > 0 else 0
                print(f"⏱️  Execution time: {execution_time:.4f} ms (Speedup: {speedup:.2f}x)")
            else:
                print(f"⏱️  Execution time: {execution_time:.4f} ms")
                
            return execution_time
        return None
    
    def generate_report(self, output_file: str = None) -> Dict:
        """Generate a performance report of all optimizations"""
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(RESULTS_DIR, f"report_{self.base_name}_{timestamp}.json")
        
        report = {
            "kernel": self.base_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": self.results,
            "speedups": {}
        }
        
        # Calculate speedups
        if self.baseline_time:
            for opt, time in self.results.items():
                if opt != "baseline" and time > 0:
                    report["speedups"][opt] = self.baseline_time / time
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Saved performance report to {output_file}")
        return report
    
    def generate_charts(self, report: Dict = None):
        """Generate performance visualization charts"""
        if not report:
            report = self.generate_report()
        
        # Create execution time comparison chart
        plt.figure(figsize=(10, 6))
        optimizations = list(report["results"].keys())
        times = [report["results"][opt] for opt in optimizations]
        
        plt.bar(optimizations, times, color='skyblue')
        plt.title(f'Kernel Execution Time Comparison - {self.base_name}')
        plt.xlabel('Optimization')
        plt.ylabel('Execution Time (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = os.path.join(RESULTS_DIR, f"execution_time_{self.base_name}.png")
        plt.savefig(chart_path)
        print(f"📈 Saved execution time chart to {chart_path}")
        
        # Create speedup chart if there's a baseline
        if "speedups" in report and report["speedups"]:
            plt.figure(figsize=(10, 6))
            opts = list(report["speedups"].keys())
            speedups = [report["speedups"][opt] for opt in opts]
            
            plt.bar(opts, speedups, color='lightgreen')
            plt.title(f'Performance Speedup - {self.base_name}')
            plt.xlabel('Optimization')
            plt.ylabel('Speedup (×)')
            plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)  # Baseline reference
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart_path = os.path.join(RESULTS_DIR, f"speedup_{self.base_name}.png")
            plt.savefig(chart_path)
            print(f"📈 Saved speedup chart to {chart_path}")

def compile_kernel(kernel_path, binary_path):
    """Compile a CUDA kernel to executable"""
    print(f"🔨 Compiling {kernel_path} to {binary_path}...")
    result = subprocess.run(
        ["nvcc", kernel_path, "-o", binary_path],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ Compilation failed:\n", result.stderr.strip() or result.stdout.strip())
        return False
    print("✅ Compilation successful.")
    return True

import re

def detect_best_optimizations(kernel_path: str) -> Dict:
    """Analyze kernel and suggest optimal optimization strategies"""
    print(f"🔍 Analyzing kernel: {kernel_path}")
    
    with open(kernel_path, 'r') as f:
        code = f.read()
    
    suggestions = {}

    # ✅ Improved shared memory detection logic
    shared_memory_candidates = re.findall(r'(\w+)\[.*threadIdx\.x.*\]', code)
    if shared_memory_candidates:
        array_counts = {var: code.count(f"{var}[") for var in set(shared_memory_candidates)}
        likely_shared = [var for var, count in array_counts.items() if count > 1]

        if likely_shared:
            suggestions["shared_memory"] = {
                "confidence": "high",
                "reason": f"Detected repeated access to {', '.join(likely_shared)} across threads",
                "params": {"block_size": 256}
            }

    # 🔄 Loop unrolling detection
    loop_pattern = r'for\s*\(\s*int\s+(\w+)\s*=\s*(\w+|\d+)\s*;\s*\1\s*<\s*([^;]+)\s*;\s*\1(\+\+|\+=\s*1)\s*\)'
    loops = re.findall(loop_pattern, code)
    if loops:
        suggestions["loop_unrolling"] = {
            "confidence": "medium",
            "reason": f"Found {len(loops)} loops that could be unrolled",
            "params": {"unroll_factor": 4}
        }

    # 📦 Memory coalescing detection
    if "threadIdx.x" in code and "[" in code and "]" in code:
        suggestions["memory_coalescing"] = {
            "confidence": "medium",
            "reason": "Detected array accesses that might benefit from coalescing",
            "params": {}
        }

    # 🧠 Register optimization detection
    if code.count("{") > 3:  # heuristic: multiple scopes
        suggestions["register_optimization"] = {
            "confidence": "low",
            "reason": "Complex kernel might benefit from register usage optimization",
            "params": {"max_registers": 32}
        }
        
    return suggestions


def apply_shared_memory_opt(transformer, block_size=256, remove_existing_index=True):
    """Apply shared memory optimization"""
    transformer.apply_shared_memory(remove_existing_index=remove_existing_index)
    return True

def apply_loop_unrolling_opt(transformer, unroll_factor=4):
    """Apply loop unrolling optimization"""
    transformer.apply_loop_unrolling(unroll_factor=unroll_factor)
    return True

def apply_memory_coalescing_opt(transformer):
    """Apply memory coalescing optimization"""
    transformer.apply_memory_coalescing()
    return True

def apply_register_optimization_opt(transformer, max_registers=32):
    """Apply register optimization"""
    transformer.apply_register_optimization(max_registers=max_registers)
    return True

def main():
    parser = argparse.ArgumentParser(
        description="CUDA Kernel Optimizer - A tool for applying and benchmarking CUDA kernel optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply shared memory optimization and compile to executable
  python cli.py --src src/kernels/vector_add_runner.cu --opt shared_memory
  
  # Apply loop unrolling with factor 8
  python cli.py --src src/kernels/vector_add_runner.cu --opt unroll --unroll-factor 8
  
  # Run all optimizations and generate report
  python cli.py --src src/kernels/vector_add_runner.cu --opt all --benchmark
  
  # Run in interactive mode to get optimization suggestions
  python cli.py --src src/kernels/vector_add_runner.cu --interactive
"""
    )
    parser.add_argument("--src", default="src/kernels/vector_add_runner.cu", help="Source CUDA kernel file")
    parser.add_argument("--opt", choices=["shared_memory", "unroll", "memory_coalescing", "register", "all", "auto"], 
                        help="Optimization to apply")
    parser.add_argument("--mode", choices=["ptx", "exe"], default="exe", 
                        help="Output compilation target: PTX or executable")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmark comparing baseline with optimized version")
    parser.add_argument("--unroll-factor", type=int, default=4,
                        help="Loop unrolling factor (default: 4)")
    parser.add_argument("--block-size", type=int, default=256,
                        help="CUDA block size for shared memory optimization (default: 256)")
    parser.add_argument("--max-registers", type=int, default=32,
                        help="Maximum registers per thread for register optimization (default: 32)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode with optimization suggestions")
    parser.add_argument("--report", action="store_true",
                        help="Generate detailed report with visualizations")
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.src):
        print(f"❌ Error: Source file '{args.src}' not found.")
        return

    # Create output directories
    os.makedirs(BUILD_DIR, exist_ok=True)
    os.makedirs(KERNELS_DIR, exist_ok=True)
    
    # Interactive mode
    if args.interactive:
        suggestions = detect_best_optimizations(args.src)
        
        print(f"\n🔍 Kernel Analysis for {args.src}")
        print("-" * 60)
        
        if not suggestions:
            print("No specific optimization opportunities detected.")
        else:
            print("Suggested optimizations:")
            for opt, details in suggestions.items():
                print(f"  • {opt} (confidence: {details['confidence']})")
                print(f"    Reason: {details['reason']}")
                print(f"    Suggested parameters: {details['params']}")
                print()
                
            # Ask user which optimizations to apply
            print("\nWhich optimizations would you like to apply?")
            print("1. Apply all suggested optimizations")
            print("2. Select specific optimizations")
            print("3. Quit")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == "1":
                # Set args for all suggested optimizations
                args.opt = "auto"
                args.benchmark = True
                args.report = True
            elif choice == "2":
                # Let user select specific optimizations
                selected_opts = []
                for opt in suggestions:
                    apply_this = input(f"Apply {opt}? (y/n): ").lower()
                    if apply_this == 'y':
                        selected_opts.append(opt)
                
                if not selected_opts:
                    print("No optimizations selected. Exiting.")
                    return
                
                # Just use the first selected optimization for now
                args.opt = selected_opts[0]
                
                # Ask if user wants benchmark
                run_benchmark = input("Run benchmark comparison? (y/n): ").lower()
                args.benchmark = (run_benchmark == 'y')
                
                # Ask if user wants report
                gen_report = input("Generate detailed report with visualizations? (y/n): ").lower()
                args.report = (gen_report == 'y')
            else:
                print("Exiting.")
                return
    
    # Initialize benchmark if requested
    if args.benchmark:
        benchmark = OptimizationBenchmark(args.src)
        baseline_time = benchmark.run_baseline()
        
        if baseline_time is None:
            print("❌ Baseline compilation or execution failed. Exiting.")
            return
    
    # Apply optimizations based on choice
    base_name = os.path.splitext(os.path.basename(args.src))[0]
    
    if args.opt == "shared_memory":
        transformer = KernelRunnerTransformer(args.src)
        transformer.apply_shared_memory(remove_existing_index=True)
        transformer.add_comment(f"Applied shared memory optimization with block size {args.block_size}.")
        out_kernel = f"{KERNELS_DIR}/{base_name}_shared_memory.cu"
        out_binary = f"{BUILD_DIR}/{base_name}_shared_memory.exe"
        out_ptx = f"{BUILD_DIR}/{base_name}_shared_memory.ptx"
        transformer.save(out_kernel)
        
        if args.benchmark:
            benchmark.run_optimization("shared_memory", apply_shared_memory_opt, 
                                      block_size=args.block_size, remove_existing_index=True)
        
    elif args.opt == "unroll":
        transformer = KernelRunnerTransformer(args.src)
        transformer.apply_loop_unrolling(unroll_factor=args.unroll_factor)
        transformer.add_comment(f"Applied loop unrolling with factor {args.unroll_factor}.")
        out_kernel = f"{KERNELS_DIR}/{base_name}_unroll.cu"
        out_binary = f"{BUILD_DIR}/{base_name}_unroll.exe"
        out_ptx = f"{BUILD_DIR}/{base_name}_unroll.ptx"
        transformer.save(out_kernel)
        
        if args.benchmark:
            benchmark.run_optimization("unroll", apply_loop_unrolling_opt, 
                                      unroll_factor=args.unroll_factor)
    
    elif args.opt == "memory_coalescing":
        transformer = KernelRunnerTransformer(args.src)
        transformer.apply_memory_coalescing()
        transformer.add_comment("Applied memory coalescing optimization.")
        out_kernel = f"{KERNELS_DIR}/{base_name}_coalesced.cu"
        out_binary = f"{BUILD_DIR}/{base_name}_coalesced.exe"
        out_ptx = f"{BUILD_DIR}/{base_name}_coalesced.ptx"
        transformer.save(out_kernel)
        
        if args.benchmark:
            benchmark.run_optimization("memory_coalescing", apply_memory_coalescing_opt)
    
    elif args.opt == "register":
        transformer = KernelRunnerTransformer(args.src)
        transformer.apply_register_optimization(max_registers=args.max_registers)
        transformer.add_comment(f"Applied register optimization with max {args.max_registers} registers.")
        out_kernel = f"{KERNELS_DIR}/{base_name}_register.cu"
        out_binary = f"{BUILD_DIR}/{base_name}_register.exe"
        out_ptx = f"{BUILD_DIR}/{base_name}_register.ptx"
        transformer.save(out_kernel)
        
        if args.benchmark:
            benchmark.run_optimization("register", apply_register_optimization_opt, 
                                      max_registers=args.max_registers)
    
    elif args.opt == "all":
        # Apply all optimizations one by one and benchmark each
        if args.benchmark:
            benchmark.run_optimization("shared_memory", apply_shared_memory_opt, 
                                      block_size=args.block_size, remove_existing_index=True)
            benchmark.run_optimization("unroll", apply_loop_unrolling_opt, 
                                      unroll_factor=args.unroll_factor)
            benchmark.run_optimization("memory_coalescing", apply_memory_coalescing_opt)
            benchmark.run_optimization("register", apply_register_optimization_opt, 
                                     max_registers=args.max_registers)
            
            # Apply all optimizations together
            transformer = KernelRunnerTransformer(args.src)
            transformer.apply_shared_memory(remove_existing_index=True)
            transformer.apply_loop_unrolling(unroll_factor=args.unroll_factor)
            transformer.apply_memory_coalescing()
            transformer.apply_register_optimization(max_registers=args.max_registers)
            transformer.add_comment("Applied all optimizations: shared memory, loop unrolling, memory coalescing, and register optimization.")
            out_kernel = f"{KERNELS_DIR}/{base_name}_all_opt.cu"
            transformer.save(out_kernel)
            
            # Compile and benchmark the combined optimization
            benchmark.run_optimization("all_combined", lambda t, **kwargs: t.save(out_kernel))
    
    elif args.opt == "auto":
        # Apply optimizations based on analysis
        suggestions = detect_best_optimizations(args.src)
        
        for opt, details in suggestions.items():
            if opt == "shared_memory":
                benchmark.run_optimization("shared_memory", apply_shared_memory_opt, 
                                          **details["params"])
            elif opt == "loop_unrolling":
                benchmark.run_optimization("unroll", apply_loop_unrolling_opt, 
                                          **details["params"])
            elif opt == "memory_coalescing":
                benchmark.run_optimization("memory_coalescing", apply_memory_coalescing_opt, 
                                          **details["params"])
            elif opt == "register_optimization":
                benchmark.run_optimization("register", apply_register_optimization_opt, 
                                          **details["params"])
    
    # Compile to PTX if requested
    if args.mode == "ptx" and "out_kernel" in locals() and "out_ptx" in locals():
        compile_cuda_to_ptx(out_kernel, out_ptx)
    
    # Generate report if requested
    if args.benchmark and args.report:
        benchmark.generate_report()
        benchmark.generate_charts()
        
    print("\n✨ Optimization process completed.")

if __name__ == "__main__":
    main()