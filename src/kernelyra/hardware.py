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


def recommend_profile(hardware: dict[str, Any]) -> str:
    threads = int(hardware.get("cpu_threads") or 1)
    ram = float(hardware.get("ram_gb") or 0)
    gpu_vram = max(
        (float(item.get("vram_gb", 0)) for item in hardware.get("nvidia_gpus", [])), default=0
    )
    if threads <= 6 or ram < 12:
        return "low-memory"
    if threads >= 32 and ram >= 64 and (gpu_vram >= 16 or (threads >= 64 and ram >= 128)):
        return "workstation"
    if threads >= 16 and ram >= 32 and gpu_vram >= 8:
        return "performance"
    return "balanced"


PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "low-memory": {
        "label": "Low memory",
        "execution_mode": "weak",
        "cpu": 30,
        "ram": 35,
        "gpu": 30,
        "max_steps": 1000,
        "batch_size": 32,
        "model": "16 → 8",
    },
    "eco": {
        "label": "Low memory (legacy eco alias)",
        "execution_mode": "weak",
        "cpu": 30,
        "ram": 35,
        "gpu": 30,
        "max_steps": 1000,
        "batch_size": 32,
        "model": "16 → 8",
    },
    "balanced": {
        "label": "Balanced",
        "execution_mode": "balanced",
        "cpu": 55,
        "ram": 55,
        "gpu": 55,
        "max_steps": 2400,
        "batch_size": 64,
        "model": "32 → 16",
    },
    "performance": {
        "label": "Performance",
        "execution_mode": "performance",
        "cpu": 80,
        "ram": 75,
        "gpu": 75,
        "max_steps": 5000,
        "batch_size": 128,
        "model": "64 → 32 → 16",
    },
    "workstation": {
        "label": "Workstation",
        "execution_mode": "workstation",
        # Workstations are throughput-oriented: CPU/GPU are uncapped while RAM
        # retains a small emergency reserve for the OS, filesystem cache and
        # checkpoint writer.
        "cpu": 100,
        "ram": 95,
        "gpu": 100,
        "reserve_ram_gb": 3,
        "max_steps": 9000,
        "batch_size": 256,
        "model": "256 → 128 → 64",
    },
    "custom": {
        "label": "Custom",
        "execution_mode": "balanced",
        "cpu": 55,
        "ram": 55,
        "gpu": 0,
        "max_steps": 2400,
        "batch_size": 64,
        "model": "backend-defined",
    },
}

# Four execution programs behind one library API.  Profiles such as ``eco``
# remain accepted as compatibility aliases, but resolve to one of these four
# runtime strategies rather than creating a fifth behavior.
EXECUTION_MODES: dict[str, dict[str, Any]] = {
    "weak": {
        "label": "Weak PC",
        "data_workers": 0,
        "prefetch": 0,
        "stream_limit": 128 * 1024 * 1024,
        "native_thread_fraction": .25,
        "bulk_step_cap": 8,
        "arena_bytes": 32 * 1024 * 1024,
        "strategy": "stream-first, low-copy, low-latency",
        "cpu_backends": ("native", "numpy", "torch", "tensorflow"),
        "gpu_backends": ("native", "numpy", "torch", "tensorflow"),
    },
    "balanced": {
        "label": "Balanced PC",
        "data_workers": 2,
        "prefetch": 2,
        "stream_limit": 256 * 1024 * 1024,
        "native_thread_fraction": .5,
        "bulk_step_cap": 32,
        "arena_bytes": 96 * 1024 * 1024,
        "strategy": "balanced parallelism with bounded reuse",
        "cpu_backends": ("native", "numpy", "torch", "tensorflow"),
        "gpu_backends": ("native", "torch", "tensorflow", "numpy"),
    },
    "performance": {
        "label": "Powerful PC",
        "data_workers": 6,
        "prefetch": 4,
        "stream_limit": 384 * 1024 * 1024,
        "native_thread_fraction": .75,
        "bulk_step_cap": 100,
        "arena_bytes": 256 * 1024 * 1024,
        "strategy": "throughput-first OpenMP and bulk dispatch",
        "cpu_backends": ("native", "numpy", "torch", "tensorflow"),
        "gpu_backends": ("torch", "tensorflow", "native", "numpy"),
    },
    "workstation": {
        "label": "Workstation",
        "data_workers": 12,
        "prefetch": 8,
        "stream_limit": 2**63 - 1,
        "native_thread_fraction": 1.0,
        "bulk_step_cap": 100,
        "arena_bytes": 768 * 1024 * 1024,
        "strategy": "maximum local throughput with large reusable arena",
        "cpu_backends": ("native", "numpy", "torch", "tensorflow"),
        "gpu_backends": ("torch", "tensorflow", "native", "numpy"),
    },
}


def execution_mode(profile: str) -> str:
    """Map public profiles and legacy aliases to one of four engine programs."""
    preset = PROFILE_PRESETS.get(profile)
    if preset is None:
        raise KeyError(f"Unknown hardware profile: {profile}")
    return str(preset["execution_mode"])


def execution_policy(profile: str, hardware: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the selected program with CPU/GPU backend priority."""
    mode = execution_mode(profile)
    policy = dict(EXECUTION_MODES[mode])
    # NVIDIA is detected without importing heavy ML runtimes. Other accelerators
    # are still usable when a user explicitly selects torch/tensorflow; their
    # framework performs final device discovery inside the isolated worker.
    has_detected_gpu = bool(hardware.get("gpu_available") or hardware.get("nvidia_gpus"))
    policy["mode"] = mode
    policy["backend_order"] = policy["gpu_backends"] if has_detected_gpu else policy["cpu_backends"]
    return policy
