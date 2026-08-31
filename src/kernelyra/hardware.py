from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess  # nosec B404
from typing import Any

# Hardware detection invokes only an absolute nvidia-smi path returned by the OS.


def _ram_bytes() -> int:
    if os.name == "nt":
        try:
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            return 0
    try:
        sysconf = getattr(os, "sysconf")
        return int(sysconf("SC_PAGE_SIZE") * sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 0


def detect_hardware() -> dict[str, Any]:
    ram_bytes = _ram_bytes()
    nvidia: list[dict[str, Any]] = []
    accelerator_hint = os.environ.get("KERNELYRA_ACCELERATOR", "").strip().lower()
    generic_accelerators: list[dict[str, Any]] = []
    if accelerator_hint in {"cuda", "rocm", "metal", "directml", "opencl"}:
        # Non-NVIDIA drivers are framework-specific.  The hint keeps startup
        # light and lets the isolated torch/tensorflow worker do final probing.
        generic_accelerators.append({"kind": accelerator_hint, "source": "KERNELYRA_ACCELERATOR"})
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            raise FileNotFoundError("nvidia-smi was not found")
        # Absolute executable path and fixed arguments; shell execution is disabled.
        result = subprocess.run(  # nosec B603
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=True,
        )
        for line in result.stdout.splitlines()[:16]:
            name, memory, driver = [part.strip() for part in line.split(",", 2)]
            nvidia.append(
                {"name": name[:160], "vram_gb": round(float(memory) / 1024, 1), "driver": driver[:80]}
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    system = platform.system().lower()
    return {
        "cpu_threads": os.cpu_count() or 1,
        "ram_gb": round(ram_bytes / 1024**3, 1) if ram_bytes else 8.0,
        "nvidia_gpus": nvidia,
        "accelerators": generic_accelerators,
        "gpu_available": bool(nvidia or generic_accelerators),
        "tensorflow_devices": [],
        "engine_loading": False,
        "engine_error": None,
        "acceleration": "Heavy ML backends load only inside a spawned run worker",
        "resource_enforcement": {
            "scheduler": "available",
            "windows_job_objects": "available" if system == "windows" else "not_applicable",
            "linux_cgroup_v2": "best_effort" if system == "linux" else "not_applicable",
            "posix_rlimit": "available" if os.name != "nt" else "not_applicable",
            "gpu": "backend_specific",
        },
        "detection": "lightweight",
    }


AUTOMATIC_EXECUTION_POLICY: dict[str, Any] = {
    "data_workers": 2,
    "prefetch": 2,
    "stream_limit": 256 * 1024 * 1024,
    # CPU is already a caller-controlled ceiling.  A run explicitly granted
    # 100% CPU must be allowed to use all detected worker threads instead of
    # being silently capped to an arbitrary fraction.
    "native_thread_fraction": 1.0,
    "bulk_step_cap": 32,
    "arena_bytes": 96 * 1024 * 1024,
    "chunk_target_records": 4096,
    "hidden_layers": (64, 32),
    "strategy": "automatic bounded execution",
    "cpu_backends": ("native", "numpy", "torch", "tensorflow"),
    # Native and NumPy are intentionally CPU-only.  Do not silently accept a
    # hybrid request and run it on the CPU: callers requesting a GPU deserve a
    # clear incompatibility error rather than a misleading successful run.
    "hybrid_backends": ("torch", "tensorflow"),
}


def resolve_execution_target(value: str | None, hardware: dict[str, Any]) -> str:
    """Resolve CPU-only or CPU+GPU execution without guessing a device class."""
    target = str(value or "auto").strip().lower()
    if target == "auto":
        return "hybrid" if bool(hardware.get("gpu_available")) else "cpu"
    if target not in {"cpu", "hybrid"}:
        raise KeyError("execution must be cpu, hybrid, or auto")
    if target == "hybrid" and not bool(hardware.get("gpu_available")):
        raise KeyError(
            "Hybrid execution requires a detected accelerator. Buy or enable a compatible GPU, "
            "then retry; otherwise use execution='cpu'."
        )
    return target


def execution_policy(
    hardware: dict[str, Any],
    *,
    execution_target: str | None = "auto",
) -> dict[str, Any]:
    """Resolve one predictable policy; resource limits remain caller-owned."""
    target = resolve_execution_target(execution_target, hardware)
    policy = dict(AUTOMATIC_EXECUTION_POLICY)
    policy["mode"] = "automatic"
    policy["execution"] = target
    policy["backend_order"] = policy["hybrid_backends"] if target == "hybrid" else policy["cpu_backends"]
    return policy
