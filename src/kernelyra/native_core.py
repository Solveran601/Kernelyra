from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .models import TaskType

ABI_VERSION = 3
COMPONENT_ZIG_MEMORY = 1
COMPONENT_FORTRAN_NUMERIC = 2
COMPONENT_ALL = 3
_TASK_IDS = {
    TaskType.BINARY_CLASSIFICATION.value: 0,
    TaskType.MULTICLASS_CLASSIFICATION.value: 1,
    TaskType.REGRESSION.value: 2,
}


class NativeCoreError(RuntimeError):
    pass


class _ModelConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("task", ctypes.c_uint32),
        ("features", ctypes.c_uint32),
        ("classes", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("learning_rate", ctypes.c_float),
        ("weight_decay", ctypes.c_float),
        ("target_mean", ctypes.c_float),
        ("target_std", ctypes.c_float),
    ]


def _library_names() -> tuple[str, ...]:
    system = platform.system().lower()
    if system == "windows":
        return ("kernelyra_core.dll",)
    if system == "darwin":
        return ("libkernelyra_core.dylib", "kernelyra_core.dylib")
    return ("libkernelyra_core.so", "kernelyra_core.so")


def native_core_candidates() -> tuple[Path, ...]:
    configured = os.environ.get("KERNELYRA_NATIVE_CORE")
    package_dir = Path(__file__).resolve().parent / "native_bin"
    candidates = [Path(configured).expanduser().resolve()] if configured else []
    candidates.extend(package_dir / name for name in _library_names())
    return tuple(candidates)


def resolve_native_core() -> Path | None:
    return next((path for path in native_core_candidates() if path.is_file()), None)


def native_core_status() -> dict[str, Any]:
    path = resolve_native_core()
    if path is None:
        return {
            "available": False,
            "path": None,
            "version": None,
            "features": None,
            "diagnostic": "Native core binary is not bundled; run `kernelyra native build` or install a binary wheel",
        }
    try:
        core = NativeCore(path)
        return {
            "available": True,
            "path": str(path),
            "version": core.version,
            "features": core.features,
            "components": core.components,
            "component_mask": core.component_mask,
            "enabled_component_mask": core.enabled_component_mask,
            "diagnostic": None,
        }
    except (NativeCoreError, OSError) as error:
        return {
            "available": False,
            "path": str(path),
            "version": None,
            "features": None,
            "diagnostic": f"Native core failed to load: {type(error).__name__}: {str(error)[:180]}",
        }


def build_native_core(output_dir: str | Path | None = None) -> Path:
    """Build the Windows C ABI with Fortran training and Zig memory kernels."""
    root = Path(__file__).resolve().parents[2]
    source = root / "native" / "bridge" / "cpp" / "core_abi.cpp"
    headers = root / "native" / "include"
    if not source.is_file():
        raise NativeCoreError("Native C++ ABI bridge source is not present in this installation")
    destination = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "native_bin"
    destination.mkdir(parents=True, exist_ok=True)
    system = platform.system().lower()
    if system == "windows":
        compiler = shutil.which("g++")
        if not compiler:
            raise NativeCoreError("g++ was not found; install the MinGW toolchain or use a binary wheel")
        output = destination / "kernelyra_core.dll"
        gfortran = shutil.which("gfortran")
        zig = shutil.which("zig")
        if not gfortran or not zig:
            missing = ", ".join(name for name, value in (("gfortran", gfortran), ("zig", zig)) if not value)
            raise NativeCoreError(
                f"Windows source build requires {missing}; install the toolchain or use the bundled Windows wheel"
            )
        component_sources = {
            "zig": root / "native" / "core" / "zig" / "memory_kernels.zig",
            "fortran": root / "native" / "core" / "fortran" / "training_kernels.f90",
        }
        with tempfile.TemporaryDirectory(prefix="kernelyra-native-build-") as temporary:
            build = Path(temporary)
            objects: list[Path] = []
            defines: list[str] = []

            def compile_component(arguments: list[str], expected: Path, name: str) -> None:
                result = subprocess.run(  # nosec B603
                    arguments,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=300,
                    check=False,
                )
                if result.returncode or not expected.is_file():
                    detail = (result.stderr or result.stdout)[-2000:]
                    raise NativeCoreError(f"{name} component build failed: {detail}")
                objects.append(expected)

            if component_sources["zig"].is_file():
                zig_object = build / "memory_zig.obj"
                compile_component(
                    [
                        zig,
                        "build-obj",
                        str(component_sources["zig"]),
                        "-O",
                        "ReleaseFast",
                        "-target",
                        "x86_64-windows-gnu",
                        "-femit-bin=" + str(zig_object),
                    ],
                    zig_object,
                    "Zig memory",
                )
                defines.append("-DKR_HAS_ZIG_MEMORY=1")
            if component_sources["fortran"].is_file():
                fortran_object = build / "numeric_fortran.o"
                compile_component(
                    [
                        gfortran,
                        "-O3",
                        "-ffast-math",
                        "-funroll-loops",
                        "-fno-protect-parens",
                        "-fopenmp",
                        "-c",
                        str(component_sources["fortran"]),
                        "-o",
                        str(fortran_object),
                    ],
                    fortran_object,
                    "Fortran numeric",
                )
                defines.append("-DKR_HAS_FORTRAN_NUMERIC=1")
            command = [
                compiler,
                "-std=c++17",
                "-O3",
                "-mtune=generic",
                "-fno-math-errno",
                "-fno-trapping-math",
                "-ffp-contract=fast",
                "-fopenmp",
                "-I",
                str(headers),
                *defines,
                "-shared",
                "-static",
                "-static-libgcc",
                "-static-libstdc++",
                str(source),
                *(str(item) for item in objects),
                "-o",
                str(output),
            ]
            if "-DKR_HAS_FORTRAN_NUMERIC=1" in defines:
                command.extend(["-static-libgfortran", "-lquadmath"])
            completed = subprocess.run(  # nosec B603
                command,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                check=False,
            )
            if completed.returncode or not output.is_file():
                detail = (completed.stderr or completed.stdout)[-2000:]
                raise NativeCoreError(f"Native core build failed: {detail}")
            NativeCore(output)
            return output
    else:
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise NativeCoreError("No C++17 compiler was found; use a binary wheel")
        suffix = ".dylib" if system == "darwin" else ".so"
        output = destination / f"libkernelyra_core{suffix}"
        command = [
            compiler,
            "-std=c++17",
            "-O3",
            "-mtune=generic",
            "-fno-math-errno",
            "-fno-trapping-math",
            "-ffp-contract=fast",
            "-shared",
            "-fPIC",
            "-I",
            str(headers),
            str(source),
            "-o",
            str(output),
        ]
    completed = subprocess.run(  # nosec B603
        command,
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if completed.returncode or not output.is_file():
        detail = (completed.stderr or completed.stdout)[-2000:]
        raise NativeCoreError(f"Native core build failed: {detail}")
    NativeCore(output)
    return output


class NativeCore:
    def __init__(self, path: str | Path | None = None):
        library_path = Path(path) if path else resolve_native_core()
        if library_path is None:
            raise NativeCoreError("Kernelyra native core is unavailable")
        try:
            self.library = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise NativeCoreError(str(error)) from None
        self.path = library_path
        self._bind()

    def _bind(self) -> None:
        library = self.library
        library.kr_core_version.restype = ctypes.c_char_p
        library.kr_core_features.restype = ctypes.c_char_p
        library.kr_core_components.restype = ctypes.c_char_p
        library.kr_core_component_mask.restype = ctypes.c_uint32
        library.kr_core_enabled_component_mask.restype = ctypes.c_uint32
        library.kr_core_set_component_mask.argtypes = [ctypes.c_uint32]
        library.kr_core_set_component_mask.restype = ctypes.c_uint32
        library.kr_last_error.restype = ctypes.c_char_p
        library.kr_memory_alloc_aligned.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        library.kr_memory_alloc_aligned.restype = ctypes.c_void_p
        library.kr_memory_free_aligned.argtypes = [ctypes.c_void_p]
        library.kr_memory_normalize_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        library.kr_memory_copy_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        library.kr_memory_zero_f32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        library.kr_numeric_gradient_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        library.kr_kernel_dot_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        library.kr_kernel_dot_f32.restype = ctypes.c_float
        library.kr_model_create.argtypes = [ctypes.POINTER(_ModelConfig)]
        library.kr_model_create.restype = ctypes.c_void_p
        library.kr_model_destroy.argtypes = [ctypes.c_void_p]
        library.kr_model_train_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        library.kr_model_train_step.restype = ctypes.c_int
        library.kr_model_train_random_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        library.kr_model_train_random_step.restype = ctypes.c_int
        library.kr_model_predict.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        library.kr_model_predict.restype = ctypes.c_int
        library.kr_model_weight_count.argtypes = [ctypes.c_void_p]
        library.kr_model_weight_count.restype = ctypes.c_size_t
        library.kr_model_bias_count.argtypes = [ctypes.c_void_p]
        library.kr_model_bias_count.restype = ctypes.c_size_t
        library.kr_model_export.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        library.kr_model_export.restype = ctypes.c_int
        library.kr_model_import.argtypes = library.kr_model_export.argtypes
        library.kr_model_import.restype = ctypes.c_int
        library.kr_csv_load_numeric.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char]
        library.kr_csv_load_numeric.restype = ctypes.c_void_p
        library.kr_csv_destroy.argtypes = [ctypes.c_void_p]
        library.kr_csv_rows.argtypes = [ctypes.c_void_p]
        library.kr_csv_rows.restype = ctypes.c_size_t
        library.kr_csv_features.argtypes = [ctypes.c_void_p]
        library.kr_csv_features.restype = ctypes.c_size_t
        library.kr_csv_target_name.argtypes = [ctypes.c_void_p]
        library.kr_csv_target_name.restype = ctypes.c_char_p
        library.kr_csv_feature_name.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.kr_csv_feature_name.restype = ctypes.c_char_p
        library.kr_csv_feature_mean.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.kr_csv_feature_mean.restype = ctypes.c_float
        library.kr_csv_feature_std.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.kr_csv_feature_std.restype = ctypes.c_float
        library.kr_csv_copy.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        library.kr_csv_copy.restype = ctypes.c_int
        library.kr_csv_stream_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        library.kr_csv_stream_open.restype = ctypes.c_void_p
        library.kr_csv_stream_destroy.argtypes = [ctypes.c_void_p]
        library.kr_csv_stream_next_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        library.kr_csv_stream_next_batch.restype = ctypes.c_int
        library.kr_csv_stream_rows_consumed.argtypes = [ctypes.c_void_p]
        library.kr_csv_stream_rows_consumed.restype = ctypes.c_uint64
        library.kr_csv_stream_restore.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        library.kr_csv_stream_restore.restype = ctypes.c_int
        library.kr_csv_scan_numeric.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char]
        library.kr_csv_scan_numeric.restype = ctypes.c_void_p
        library.kr_csv_scan_destroy.argtypes = [ctypes.c_void_p]
        library.kr_csv_scan_rows.argtypes = [ctypes.c_void_p]
        library.kr_csv_scan_rows.restype = ctypes.c_uint64
        library.kr_csv_scan_split_rows.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        library.kr_csv_scan_split_rows.restype = ctypes.c_uint64
        library.kr_csv_scan_features.argtypes = [ctypes.c_void_p]
        library.kr_csv_scan_features.restype = ctypes.c_size_t
        library.kr_csv_scan_columns.argtypes = [ctypes.c_void_p]
        library.kr_csv_scan_columns.restype = ctypes.c_size_t
        library.kr_csv_scan_column_name.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.kr_csv_scan_column_name.restype = ctypes.c_char_p
        library.kr_csv_scan_target_column.argtypes = [ctypes.c_void_p]
        library.kr_csv_scan_target_column.restype = ctypes.c_uint32
        library.kr_csv_scan_target_name.argtypes = [ctypes.c_void_p]
        library.kr_csv_scan_target_name.restype = ctypes.c_char_p
        library.kr_csv_scan_feature_name.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.kr_csv_scan_feature_name.restype = ctypes.c_char_p
        library.kr_csv_scan_feature_mean.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.kr_csv_scan_feature_mean.restype = ctypes.c_float
        library.kr_csv_scan_feature_std.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.kr_csv_scan_feature_std.restype = ctypes.c_float
        library.kr_csv_scan_target_values.argtypes = [ctypes.c_void_p]
        library.kr_csv_scan_target_values.restype = ctypes.c_size_t
        library.kr_csv_scan_target_value.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.kr_csv_scan_target_value.restype = ctypes.c_char_p

    @property
    def version(self) -> str:
        return str(self.library.kr_core_version().decode("ascii", "replace"))

    @property
    def features(self) -> str:
        return str(self.library.kr_core_features().decode("ascii", "replace"))

    @property
    def components(self) -> str:
        return str(self.library.kr_core_components().decode("ascii", "replace"))

    @property
    def component_mask(self) -> int:
        return int(self.library.kr_core_component_mask())

    @property
    def enabled_component_mask(self) -> int:
        return int(self.library.kr_core_enabled_component_mask())

    def set_component_mask(self, mask: int) -> int:
        return int(self.library.kr_core_set_component_mask(mask))

    @staticmethod
    def _native_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
        if not isinstance(values, np.ndarray) or values.dtype != np.float32 or not values.flags.c_contiguous:
            raise NativeCoreError(f"{name} must be a C-contiguous float32 NumPy array")
        return values

    def copy_f32(self, destination: np.ndarray, source: np.ndarray) -> None:
        """Copy equal-sized float32 arrays through the selected memory kernel."""
        destination = self._native_float_array(destination, name="destination")
        source = self._native_float_array(source, name="source")
        if destination.size != source.size:
            raise NativeCoreError("Native memory copy requires equal-sized arrays")
        self.library.kr_memory_copy_f32(
            destination.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            destination.size,
        )

    def zero_f32(self, values: np.ndarray) -> None:
        """Zero a float32 array through the selected memory kernel."""
        values = self._native_float_array(values, name="values")
        self.library.kr_memory_zero_f32(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), values.size
        )

    def error(self) -> NativeCoreError:
        value = self.library.kr_last_error()
        return NativeCoreError(value.decode("utf-8", "replace") if value else "Unknown native core error")


class NativeModel:
    def __init__(
        self,
        *,
        task: str,
        features: int,
        classes: int = 1,
        seed: int = 42,
        learning_rate: float = .03,
        weight_decay: float = 0.0,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        threads: int | None = None,
        core: NativeCore | None = None,
    ):
        if task not in _TASK_IDS:
            raise NativeCoreError(f"Unsupported native task: {task}")
        self.core = core or NativeCore()
        self.task = task
        self.features = int(features)
        self.classes = int(classes if task == TaskType.MULTICLASS_CLASSIFICATION.value else 1)
        config = _ModelConfig(
            ABI_VERSION,
            _TASK_IDS[task],
            self.features,
            self.classes,
            max(1, int(threads if threads is not None else (os.cpu_count() or 1))),
            int(seed),
            float(learning_rate),
            float(weight_decay),
            float(target_mean),
            float(target_std),
        )
        self.handle = self.core.library.kr_model_create(ctypes.byref(config))
        if not self.handle:
            raise self.core.error()

    @staticmethod
    def _float_pointer(array: np.ndarray) -> Any:
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        rows = np.ascontiguousarray(x, dtype=np.float32)
        targets = np.ascontiguousarray(y, dtype=np.float32).reshape(-1)
        if rows.ndim != 2 or rows.shape != (len(targets), self.features) or len(targets) == 0:
            raise NativeCoreError("Native train batch shape mismatch")
        loss = ctypes.c_float()
        ok = self.core.library.kr_model_train_step(
            self.handle,
            self._float_pointer(rows),
            self._float_pointer(targets),
            len(targets),
            ctypes.byref(loss),
        )
        if not ok:
            raise self.core.error()
        return float(loss.value)

    def train_random_step(self, x: np.ndarray, y: np.ndarray, batch_size: int) -> float:
        rows = np.ascontiguousarray(x, dtype=np.float32)
        targets = np.ascontiguousarray(y, dtype=np.float32).reshape(-1)
        if rows.ndim != 2 or rows.shape != (len(targets), self.features) or len(targets) == 0:
            raise NativeCoreError("Native training matrix shape mismatch")
        if batch_size < 1:
            raise NativeCoreError("Native batch size must be positive")
        loss = ctypes.c_float()
        ok = self.core.library.kr_model_train_random_step(
            self.handle,
            self._float_pointer(rows),
            self._float_pointer(targets),
            len(targets),
            int(batch_size),
            ctypes.byref(loss),
        )
        if not ok:
            raise self.core.error()
        return float(loss.value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        rows = np.ascontiguousarray(x, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.features:
            raise NativeCoreError("Native predict matrix shape mismatch")
        shape = (len(rows), self.classes) if self.task == TaskType.MULTICLASS_CLASSIFICATION.value else (len(rows),)
        output = np.empty(shape, dtype=np.float32)
        ok = self.core.library.kr_model_predict(
            self.handle,
            self._float_pointer(rows),
            len(rows),
            self._float_pointer(output),
            output.size,
        )
        if not ok:
            raise self.core.error()
        return output

    def export_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        weight_count = int(self.core.library.kr_model_weight_count(self.handle))
        bias_count = int(self.core.library.kr_model_bias_count(self.handle))
        weights: np.ndarray = np.empty(weight_count, dtype=np.float32)
        bias = np.empty(bias_count, dtype=np.float32)
        ok = self.core.library.kr_model_export(
            self.handle,
            self._float_pointer(weights),
            weight_count,
            self._float_pointer(bias),
            bias_count,
        )
        if not ok:
            raise self.core.error()
        if self.task == TaskType.MULTICLASS_CLASSIFICATION.value:
            weights = weights.reshape(self.features, self.classes)
        return weights, bias

    def import_parameters(self, weights: np.ndarray, bias: np.ndarray | float) -> None:
        weight_values = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
        bias_values = np.ascontiguousarray(np.atleast_1d(bias), dtype=np.float32).reshape(-1)
        ok = self.core.library.kr_model_import(
            self.handle,
            self._float_pointer(weight_values),
            weight_values.size,
            self._float_pointer(bias_values),
            bias_values.size,
        )
        if not ok:
            raise self.core.error()

    def close(self) -> None:
        if self.handle:
            self.core.library.kr_model_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> NativeModel:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


class NativeNumericCsv:
    """Owned numeric CSV dataset decoded and standardized by the native core."""

    def __init__(
        self,
        path: str | Path,
        target: str | None = None,
        delimiter: str = ",",
        *,
        core: NativeCore | None = None,
    ):
        if len(delimiter.encode("utf-8")) != 1:
            raise NativeCoreError("Native CSV delimiter must be one byte")
        self.core = core or NativeCore()
        self.handle = self.core.library.kr_csv_load_numeric(
            str(Path(path).resolve()).encode("utf-8"),
            (target or "").encode("utf-8"),
            delimiter.encode("ascii"),
        )
        if not self.handle:
            raise self.core.error()
        self.rows = int(self.core.library.kr_csv_rows(self.handle))
        self.features = int(self.core.library.kr_csv_features(self.handle))
        self.target = self.core.library.kr_csv_target_name(self.handle).decode("utf-8", "replace")
        self.feature_names = tuple(
            self.core.library.kr_csv_feature_name(self.handle, index).decode("utf-8", "replace")
            for index in range(self.features)
        )
        self.means = tuple(
            float(self.core.library.kr_csv_feature_mean(self.handle, index))
            for index in range(self.features)
        )
        self.stds = tuple(
            float(self.core.library.kr_csv_feature_std(self.handle, index))
            for index in range(self.features)
        )

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.empty((self.rows, self.features), dtype=np.float32)
        y = np.empty(self.rows, dtype=np.float32)
        ok = self.core.library.kr_csv_copy(
            self.handle,
            NativeModel._float_pointer(x),
            x.size,
            NativeModel._float_pointer(y),
            y.size,
        )
        if not ok:
            raise self.core.error()
        return x, y

    def close(self) -> None:
        if self.handle:
            self.core.library.kr_csv_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> NativeNumericCsv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


class NativeNumericCsvStream:
    """Bounded-memory numeric CSV batches encoded entirely by the C++ core."""

    def __init__(
        self,
        spec: dict[str, Any],
        *,
        split: str = "train",
        core: NativeCore | None = None,
    ):
        if spec.get("format") not in {"csv", "tsv"} or spec.get("encoding") not in {
            "utf-8",
            "utf-8-sig",
        }:
            raise NativeCoreError("Native streaming requires UTF-8 CSV or TSV")
        if spec.get("categorical_columns"):
            raise NativeCoreError("Categorical streaming uses the general feature-hashing source")
        columns = [str(value) for value in spec["columns"]]
        feature_names = [str(value) for value in spec["numeric_columns"]]
        try:
            feature_columns = np.asarray([columns.index(name) for name in feature_names], dtype=np.uint32)
            target_column = columns.index(str(spec["target"]))
            means = np.asarray([spec["means"][name] for name in feature_names], dtype=np.float32)
            stds = np.asarray([spec["stds"][name] for name in feature_names], dtype=np.float32)
            task = str(spec["task_type"])
            classes = np.asarray([float(value) for value in spec.get("classes", ())], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise NativeCoreError("Native streaming metadata is not numeric or is incomplete") from error
        if task not in _TASK_IDS or not feature_names:
            raise NativeCoreError("Native streaming task or feature set is unsupported")
        self.core = core or NativeCore()
        self.features = len(feature_names)
        self.train_records = max(1, int(spec["split_records"]["train"]))
        split_ids = {"train": 0, "validation": 1, "test": 2}
        if split not in split_ids:
            raise NativeCoreError(f"Unknown native stream split: {split}")
        self.split = split
        self.selected_records = max(1, int(spec["split_records"][split]))
        class_pointer = (
            classes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if classes.size
            else ctypes.POINTER(ctypes.c_float)()
        )
        self.handle = self.core.library.kr_csv_stream_open(
            str(Path(str(spec["path"])).resolve()).encode("utf-8"),
            str(spec["delimiter"]).encode("ascii"),
            feature_columns.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            feature_columns.size,
            target_column,
            NativeModel._float_pointer(means),
            NativeModel._float_pointer(stds),
            _TASK_IDS[task],
            class_pointer,
            classes.size,
            split_ids[split],
            self.selected_records,
        )
        if not self.handle:
            raise self.core.error()

    @property
    def rows_consumed(self) -> int:
        return int(self.core.library.kr_csv_stream_rows_consumed(self.handle))

    @property
    def epoch(self) -> int:
        return self.rows_consumed // self.selected_records

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_size < 1:
            raise NativeCoreError("Native stream batch size must be positive")
        x = np.empty((batch_size, self.features), dtype=np.float32)
        y = np.empty(batch_size, dtype=np.float32)
        ok = self.core.library.kr_csv_stream_next_batch(
            self.handle,
            batch_size,
            NativeModel._float_pointer(x),
            x.size,
            NativeModel._float_pointer(y),
            y.size,
        )
        if not ok:
            raise self.core.error()
        return x, y

    def state(self) -> dict[str, int]:
        return {"stream_epoch": self.epoch, "stream_rows_consumed": self.rows_consumed}

    def restore_rows(self, rows_consumed: int) -> None:
        if rows_consumed < 0:
            raise NativeCoreError("Native stream checkpoint cursor cannot be negative")
        if not self.core.library.kr_csv_stream_restore(self.handle, rows_consumed):
            raise self.core.error()

    def close(self) -> None:
        if self.handle:
            self.core.library.kr_csv_stream_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> NativeNumericCsvStream:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


class NativeNumericCsvScan:
    """One-pass bounded-memory statistics for numeric UTF-8 CSV/TSV."""

    def __init__(
        self,
        path: str | Path,
        target: str | None = None,
        delimiter: str = ",",
        *,
        core: NativeCore | None = None,
    ):
        self.core = core or NativeCore()
        self.handle = self.core.library.kr_csv_scan_numeric(
            str(Path(path).resolve()).encode("utf-8"),
            (target or "").encode("utf-8"),
            delimiter.encode("ascii"),
        )
        if not self.handle:
            raise self.core.error()
        library = self.core.library
        self.rows = int(library.kr_csv_scan_rows(self.handle))
        self.split_records = {
            "train": int(library.kr_csv_scan_split_rows(self.handle, 0)),
            "validation": int(library.kr_csv_scan_split_rows(self.handle, 1)),
            "test": int(library.kr_csv_scan_split_rows(self.handle, 2)),
        }
        features = int(library.kr_csv_scan_features(self.handle))
        columns = int(library.kr_csv_scan_columns(self.handle))
        self.columns = tuple(
            library.kr_csv_scan_column_name(self.handle, index).decode("utf-8", "replace")
            for index in range(columns)
        )
        self.target_column = int(library.kr_csv_scan_target_column(self.handle))
        self.target = library.kr_csv_scan_target_name(self.handle).decode("utf-8", "replace")
        self.feature_names = tuple(
            library.kr_csv_scan_feature_name(self.handle, index).decode("utf-8", "replace")
            for index in range(features)
        )
        self.means = {
            name: float(library.kr_csv_scan_feature_mean(self.handle, index))
            for index, name in enumerate(self.feature_names)
        }
        self.stds = {
            name: float(library.kr_csv_scan_feature_std(self.handle, index))
            for index, name in enumerate(self.feature_names)
        }
        declared_values = int(library.kr_csv_scan_target_values(self.handle))
        values = [
            library.kr_csv_scan_target_value(self.handle, index).decode("utf-8", "replace")
            for index in range(declared_values)
        ]
        self.target_values = tuple(value for value in values if value)
        self.target_values_overflow = len(self.target_values) != declared_values

    def close(self) -> None:
        if self.handle:
            self.core.library.kr_csv_scan_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> NativeNumericCsvScan:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass
