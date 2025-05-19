import os
from src.kernels.runner_transformer import KernelRunnerTransformer

def test_add_comment(tmp_path):
    src_file = tmp_path / "vector_add_runner.cu"
    src_file.write_text("__global__ void foo() {}")

    transformer = KernelRunnerTransformer(str(src_file))
    transformer.add_comment("Test optimization comment")

    output = transformer.code
    assert "// Test optimization comment" in output
