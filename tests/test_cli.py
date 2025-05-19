import subprocess

def test_cli_help():
    result = subprocess.run(["python", "src/ui/cli.py", "--help"], capture_output=True, text=True)
    assert "CUDA Kernel Optimizer" in result.stdout
