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
        "gpu_available": bool(nvidia),
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
        "cpu": 30,
        "ram": 35,
        "gpu": 30,
        "max_steps": 1000,
        "batch_size": 32,
        "model": "16 → 8",
    },
    "eco": {
        "label": "Low memory (legacy eco alias)",
        "cpu": 30,
        "ram": 35,
        "gpu": 30,
        "max_steps": 1000,
        "batch_size": 32,
        "model": "16 → 8",
    },
    "balanced": {
        "label": "Balanced",
        "cpu": 55,
        "ram": 55,
        "gpu": 55,
        "max_steps": 2400,
        "batch_size": 64,
        "model": "32 → 16",
    },
    "performance": {
        "label": "Performance",
        "cpu": 80,
        "ram": 75,
        "gpu": 75,
        "max_steps": 5000,
        "batch_size": 128,
        "model": "64 → 32 → 16",
    },
    "workstation": {
        "label": "Workstation",
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
        "cpu": 55,
        "ram": 55,
        "gpu": 0,
        "max_steps": 2400,
        "batch_size": 64,
        "model": "backend-defined",
    },
}
