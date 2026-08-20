from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Any


def requested_limits(resource_limits: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_bytes": int(resource_limits.get("memory_bytes") or 0),
        "cpu_percent": int(resource_limits.get("cpu_percent") or 0),
        "gpu_memory_mb": int(resource_limits.get("gpu_memory_mb") or 0),
    }


def apply_child_limits(resource_limits: dict[str, Any]) -> dict[str, Any]:
    """Apply limits available inside a spawned POSIX worker without privileges."""
    requested = requested_limits(resource_limits)
    result: dict[str, Any] = {
        "requested": requested,
        "scheduler_enforced": True,
        "os_enforced": {},
        "backend_enforced": {
            "gpu_memory": "pending_backend_confirmation" if requested["gpu_memory_mb"] else "not_requested"
        },
        "unsupported": [],
        "degraded": [],
        "platform": platform.system().lower(),
    }
    if os.name == "nt":
        result["os_enforced"]["assignment"] = "daemon_job_object"
        return result
    try:
        getattr(os, "setsid")()
        result["os_enforced"]["process_group"] = True
    except OSError as error:
        result["degraded"].append(f"process_group: {type(error).__name__}")

    try:
        import resource

        memory_bytes = requested["memory_bytes"]
        if memory_bytes > 0:
            setrlimit = getattr(resource, "setrlimit")
            rlimit_as = getattr(resource, "RLIMIT_AS")
            setrlimit(rlimit_as, (memory_bytes, memory_bytes))
            result["os_enforced"]["memory"] = "rlimit_as"
        else:
            result["unsupported"].append("memory_not_requested")
    except (ImportError, OSError, ValueError) as error:
        result["degraded"].append(f"memory_rlimit: {type(error).__name__}")

    if sys.platform.startswith("linux"):
        cgroup_root = Path("/sys/fs/cgroup")
        controllers = cgroup_root / "cgroup.controllers"
        if controllers.exists() and os.access(cgroup_root, os.W_OK):
            group = cgroup_root / f"kernelyra-{os.getpid()}"
            try:
                group.mkdir(exist_ok=False)
                if requested["memory_bytes"]:
                    (group / "memory.max").write_text(str(requested["memory_bytes"]), encoding="ascii")
                if requested["cpu_percent"]:
                    quota = max(1000, min(100000, requested["cpu_percent"] * 1000))
                    (group / "cpu.max").write_text(f"{quota} 100000", encoding="ascii")
                (group / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
                result["os_enforced"]["cgroup_v2"] = str(group)
            except OSError as error:
                result["degraded"].append(f"cgroup_v2: {type(error).__name__}")
        else:
            result["degraded"].append("cgroup_v2 unavailable or not writable; rlimit/process-group fallback")
    elif sys.platform == "darwin":
        result["degraded"].append("macOS has no cgroup CPU-rate equivalent; process-group/rlimit used")
    if requested["cpu_percent"] and "cgroup_v2" not in result["os_enforced"]:
        result["unsupported"].append("cpu_rate")
    return result


class WindowsJob:
    """Best-effort Windows Job Object with tree kill, memory and CPU-rate limits."""

    def __init__(self, pid: int, resource_limits: dict[str, Any]):
        self.handle: int | None = None
        self.status: dict[str, Any] = {
            "requested": requested_limits(resource_limits),
            "scheduler_enforced": True,
            "os_enforced": {},
            "backend_enforced": {
                "gpu_memory": "pending_backend_confirmation" if resource_limits.get("gpu_memory_mb") else "not_requested"
            },
            "unsupported": [],
            "degraded": [],
            "platform": "windows",
        }
        if os.name != "nt":
            self.status["unsupported"].append("windows_job_object")
            return
        try:
            self._assign(pid, resource_limits)
        except (OSError, ValueError) as error:
            self.status["degraded"].append(f"job_object: {type(error).__name__}: {str(error)[:120]}")

    def _assign(self, pid: int, resource_limits: dict[str, Any]) -> None:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = int(job)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        limits = EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        memory_bytes = int(resource_limits.get("memory_bytes") or 0)
        if memory_bytes:
            limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY
            limits.ProcessMemoryLimit = memory_bytes
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())

        cpu_percent = int(resource_limits.get("cpu_percent") or 0)
        if cpu_percent:
            class CPU_RATE(ctypes.Structure):
                _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]

            cpu = CPU_RATE(0x1 | 0x4, max(100, min(10000, cpu_percent * 100)))
            if kernel32.SetInformationJobObject(job, 15, ctypes.byref(cpu), ctypes.sizeof(cpu)):
                self.status["os_enforced"]["cpu_rate"] = "job_object_hard_cap"
            else:
                self.status["degraded"].append("Windows Job Object CPU rate unavailable")
        process = kernel32.OpenProcess(0x0100 | 0x0400, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel32.AssignProcessToJobObject(job, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel32.CloseHandle(process)
        self.status["os_enforced"]["process_tree"] = "job_object"
        if memory_bytes:
            self.status["os_enforced"]["memory"] = "job_object_process_memory"

    def terminate(self, exit_code: int = 1) -> None:
        if self.handle and os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.TerminateJobObject(ctypes.c_void_p(self.handle), exit_code)

    def close(self) -> None:
        if self.handle and os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle(ctypes.c_void_p(self.handle))
            self.handle = None
