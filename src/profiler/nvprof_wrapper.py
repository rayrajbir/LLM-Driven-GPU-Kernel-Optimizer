import subprocess

def profile_with_nvprof(executable_path):
    cmd = ["nvprof", executable_path]
    subprocess.run(cmd)
