import subprocess
import re

def profile_kernel(binary_path):
    import subprocess
    import re

    try:
        result = subprocess.run([binary_path], capture_output=True, text=True, timeout=10)
        stdout = result.stdout
        match = re.search(r"Execution time:\s*([\d.]+)\s*ms", stdout)
        if match:
            return float(match.group(1))
        else:
            print("⚠️ Could not parse kernel execution time. Trying nvprof...")

        # Fallback: use nvprof
        nvprof_result = subprocess.run(
            ["nvprof", binary_path],
            capture_output=True, text=True
        )
        nvprof_out = nvprof_result.stderr  # nvprof writes to stderr
        nvprof_match = re.search(r"(?i)kernel.*?([\d.]+)\s+ms", nvprof_out)
        if nvprof_match:
            return float(nvprof_match.group(1))
        else:
            print("⚠️ nvprof also failed to extract timing.")
            return None
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        return None
