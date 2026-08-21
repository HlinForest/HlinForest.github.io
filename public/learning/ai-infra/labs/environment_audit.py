"""Collect a reproducible, secret-safe environment snapshot."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys


def run(command: list[str]) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        return "not found"
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"failed: {type(exc).__name__}: {exc}"
    output = (result.stdout or result.stderr).strip()
    return f"exit={result.returncode}\n{output}"


def torch_report() -> str:
    if importlib.util.find_spec("torch") is None:
        return "PyTorch: not installed"
    import torch

    lines = [
        f"PyTorch: {torch.__version__}",
        f"built CUDA: {torch.version.cuda}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"device count: {torch.cuda.device_count()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"device 0: {torch.cuda.get_device_name(0)}")
    return "\n".join(lines)


def main() -> None:
    print("== Runtime ==")
    print("platform:", platform.platform())
    print("python:", sys.version.replace("\n", " "))
    print("executable:", sys.executable)
    print("cwd:", os.getcwd())
    print("virtual env:", os.environ.get("VIRTUAL_ENV", "not set"))
    print("conda env:", os.environ.get("CONDA_DEFAULT_ENV", "not set"))

    print("\n== Toolchain ==")
    for command in (
        ["git", "--version"],
        ["cmake", "--version"],
        ["g++", "--version"],
        ["nvcc", "--version"],
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    ):
        print(f"\n$ {' '.join(command)}")
        print(run(command))

    print("\n== Framework ==")
    print(torch_report())

    print("\n== Selected paths ==")
    for name in ("PATH", "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"):
        value = os.environ.get(name)
        if value is None:
            print(f"{name}: not set")
        elif name == "PATH":
            entries = value.split(os.pathsep)
            print(f"{name}: {len(entries)} entries")
            for entry in entries[:10]:
                print("  ", entry)
        else:
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()

# EXERCISES
# 1. Save two reports from different Python environments and diff them.
# 2. Add a check for the C++ compiler target architecture.
# 3. Explain why this script intentionally does not dump every environment variable.
