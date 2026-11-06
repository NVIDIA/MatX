from pathlib import Path
import re


include = Path(__file__).resolve().parents[2] / "include"
seen = set()
blocked = {
    "matx.h",
    "matx/core/print.h",
    "matx/file_io/file_io.h",
    "matx/operators/operators.h",
    "matx/transforms/transforms.h",
    "matx/operators/matmul.h",
    "cublas_v2.h",
    "cusparse.h",
}


def visit(path):
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)

    source = path.read_text()
    for name in re.findall(r'^\s*#\s*include\s*[<"]([^>"]+)', source, re.M):
        if name in blocked:
            raise AssertionError(f"{path.relative_to(include)} includes {name}")

        child = include / name
        if not child.is_file():
            child = path.parent / name
        if child.is_file():
            visit(child)


visit(include / "matx/core/matx_tensor.h")
