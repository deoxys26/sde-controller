"""Pinned runtime policy and environment recorder for the Colab adapter."""

from __future__ import annotations

import importlib.metadata
import argparse
import json
import platform
import sys
from typing import Any


EXPECTED_PYTHON = (3, 12)
PINNED_VERSIONS = {
    "numpy": "1.26.4",
    "torch": "2.12.0",
    "psutil": "7.2.2",
    "pytest": "9.1.1",
}


class EnvironmentMismatch(RuntimeError):
    """The installed runtime differs from the reviewed migration pins."""


def collect_environment() -> dict[str, Any]:
    import numpy
    import psutil
    import torch

    try:
        pytest_version = importlib.metadata.version("pytest")
    except importlib.metadata.PackageNotFoundError:
        pytest_version = None
    return {
        "python": sys.version,
        "python_major_minor": list(sys.version_info[:2]),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torch_base_version": torch.__version__.split("+", 1)[0],
        "psutil": psutil.__version__,
        "pytest": pytest_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_build": torch.version.cuda,
        "device_policy": "CPU only for this migration protocol",
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
    }


def verify_environment(*, require_pytest: bool = True) -> dict[str, Any]:
    info = collect_environment()
    mismatches: list[str] = []
    if tuple(info["python_major_minor"]) != EXPECTED_PYTHON:
        mismatches.append(f"Python expected {EXPECTED_PYTHON}, observed {info['python_major_minor']}")
    observed = {
        "numpy": info["numpy"], "torch": info["torch_base_version"], "psutil": info["psutil"],
    }
    if require_pytest:
        observed["pytest"] = info["pytest"]
    for package, actual in observed.items():
        expected = PINNED_VERSIONS[package]
        if actual != expected:
            mismatches.append(f"{package} expected {expected}, observed {actual}")
    if mismatches:
        raise EnvironmentMismatch("; ".join(mismatches))
    return info


def print_environment(*, verify: bool = True, require_pytest: bool = False) -> dict[str, Any]:
    info = verify_environment(require_pytest=require_pytest) if verify else collect_environment()
    print(json.dumps(info, indent=2, sort_keys=True))
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true",
                        help="report installed versions without enforcing migration pins")
    parser.add_argument("--require-pytest", action="store_true")
    args = parser.parse_args(argv)
    print_environment(verify=not args.report_only, require_pytest=args.require_pytest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
