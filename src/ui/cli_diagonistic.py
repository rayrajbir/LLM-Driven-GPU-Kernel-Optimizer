import os
import sys
import subprocess

def run_cli_diagnostic():
    """Run a diagnostic check on the CLI script"""
    
    # Define paths
    project_root = r"C:\Users\User\Desktop\llvm-cuda-optimizer"
    cli_path = os.path.join(project_root, "src", "ui", "cli.py")
    kernel_path = os.path.join(project_root, "src", "kernels", "vector_add_runner.cu")
    
    # Verify files exist
    print(f"Checking files...")
    print(f"CLI path exists: {os.path.exists(cli_path)}")
    print(f"Kernel path exists: {os.path.exists(kernel_path)}")
    
    # Check Python path
    print("\nCurrent Python path:")
    for p in sys.path:
        print(f"  - {p}")
    
    # Try to import from CLI directly to see errors
    print("\nAttempting to import CLI module...")
    try:
        sys.path.append(os.path.dirname(os.path.dirname(cli_path)))  # Add src directory to path
        print(f"Added {os.path.dirname(os.path.dirname(cli_path))} to Python path")
        
        # Try running the CLI with minimal arguments
        cmd = [sys.executable, cli_path, "--src", kernel_path, "--interactive"]
        print(f"\nRunning CLI with command: {' '.join(cmd)}")
        
        # Set environment variable for Python path
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        
        # Run with full output capture
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
    
    # Check module dependencies
    print("\nChecking for required modules...")
    modules_to_check = [
        "matplotlib", 
        "numpy",
        "json",
        "pathlib",
        "re"
    ]
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {module} - Available")
        except ImportError:
            print(f"❌ {module} - Missing")
    
    # Check for project-specific modules
    print("\nChecking project module imports...")
    try:
        sys.path.append(project_root)
        print(f"Added {project_root} to Python path")
        
        # Print all .py files in the src directory to understand the module structure
        print("\nPython files in project:")
        src_dir = os.path.join(project_root, "src")
        if os.path.exists(src_dir):
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.endswith('.py'):
                        rel_path = os.path.relpath(os.path.join(root, file), project_root)
                        print(f"  - {rel_path}")
    
    except Exception as e:
        print(f"Error examining project structure: {str(e)}")

if __name__ == "__main__":
    run_cli_diagnostic()