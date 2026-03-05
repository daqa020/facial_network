"""
gpu_check.py – Hardware detection module.

Detects the operating system, Python version, NVIDIA GPU presence,
and CUDA availability.  All checks are best-effort; the application
never fails if CUDA / GPU drivers are missing.
"""

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    """Information about a detected NVIDIA GPU."""
    name: str = "Unknown"
    driver_version: str = "Unknown"
    memory_total_mb: Optional[int] = None


@dataclass
class SystemInfo:
    """Aggregated system / hardware report."""
    os_name: str = ""
    os_version: str = ""
    architecture: str = ""
    python_version: str = ""
    gpu: Optional[GPUInfo] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None


# ─── Detection Functions ─────────────────────────────────────────────────────

def detect_system_info() -> dict:
    """Return basic OS and Python information as a dict."""
    return {
        "os_name": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }


def detect_gpu() -> Optional[GPUInfo]:
    """
    Attempt to detect an NVIDIA GPU by calling ``nvidia-smi``.

    Returns a :class:`GPUInfo` on success, or ``None`` if no NVIDIA driver
    is installed or the command is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]  # first GPU only
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None

        return GPUInfo(
            name=parts[0],
            driver_version=parts[1],
            memory_total_mb=int(parts[2]) if parts[2].isdigit() else None,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


def detect_cuda() -> tuple[bool, Optional[str]]:
    """
    Check for CUDA availability via environment variables and ``nvcc``.

    Returns
    -------
    (cuda_found, version_string_or_None)
    """
    # 1. Environment variable check
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    cuda_found = cuda_path is not None

    # 2. Try nvcc --version
    version: Optional[str] = None
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            cuda_found = True
            for token in result.stdout.split():
                if token.startswith("V"):
                    version = token.lstrip("V")
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return cuda_found, version


# ─── Combined Report ─────────────────────────────────────────────────────────

def gather_full_report() -> SystemInfo:
    """Run all detection routines and return a :class:`SystemInfo`."""
    sys_info = detect_system_info()
    gpu = detect_gpu()
    cuda_found, cuda_ver = detect_cuda()

    return SystemInfo(
        os_name=sys_info["os_name"],
        os_version=sys_info["os_version"],
        architecture=sys_info["architecture"],
        python_version=sys_info["python_version"],
        gpu=gpu,
        cuda_available=cuda_found,
        cuda_version=cuda_ver,
    )


def print_hardware_report(info: Optional[SystemInfo] = None) -> SystemInfo:
    """
    Pretty-print the hardware / environment report to stdout.

    If *info* is ``None`` a fresh report is gathered first.
    Returns the :class:`SystemInfo` so callers can inspect it.
    """
    if info is None:
        info = gather_full_report()

    sep = "\u2500" * 52
    print(f"\n{sep}")
    print("  SYSTEM & HARDWARE REPORT")
    print(sep)
    print(f"  OS            : {info.os_name} {info.os_version}")
    print(f"  Architecture  : {info.architecture}")
    print(f"  Python        : {info.python_version}")
    print(sep)

    if info.gpu is not None:
        vram = (
            f"{info.gpu.memory_total_mb} MB"
            if info.gpu.memory_total_mb
            else "N/A"
        )
        print(f"  NVIDIA GPU    : {info.gpu.name}")
        print(f"  Driver        : {info.gpu.driver_version}")
        print(f"  VRAM          : {vram}")
    else:
        print("  NVIDIA GPU    : Not detected")

    if info.cuda_available:
        ver = info.cuda_version or "detected (version unknown)"
        print(f"  CUDA          : {ver}")
    else:
        print("  CUDA          : Not available")

    print(sep)
    if info.gpu is None:
        print("  \u24d8  No GPU detected \u2013 running on CPU (fully supported).")
    else:
        print("  \u2713  GPU detected \u2013 MediaPipe uses CPU; GPU info logged.")
    print(f"{sep}\n")

    return info


if __name__ == "__main__":
    print_hardware_report()
