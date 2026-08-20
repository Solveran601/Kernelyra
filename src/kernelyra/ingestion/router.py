from __future__ import annotations

import json
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

from ..architectures import CHECKPOINT_FORMATS
from ..errors import DatasetError
from ..formats import FORMAT_COUNT, format_for_path
from ..streaming import MAX_FOLDER_FILES, STREAM_EXTENSIONS
from .registry import IngestorRegistry

# Native probes are resolved to explicit executables and invoked without a shell.
MODEL_FORMATS = CHECKPOINT_FORMATS
NATIVE_PROTOCOL = "kernelyra-native-probe/1"
MAX_NATIVE_OUTPUT = 64 * 1024


def _model_format(path: Path) -> dict[str, Any] | None:
    lower = path.name.lower()
    for model in MODEL_FORMATS:
        if any(lower.endswith(str(extension)) for extension in model["extensions"]):
            return model
    return None


class FormatRouter:
    """Route files through a built-in catalogue; no dynamic extensions."""

    route_count = FORMAT_COUNT

    def __init__(self, native_probe: Path | None = None, registry: IngestorRegistry | None = None):
        self.native_probe = native_probe
        self.registry = registry or IngestorRegistry()

    def inspect(self, raw_path: str | Path) -> dict[str, Any]:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise DatasetError("Path was not found; check the disk, folder and file name")
        if path.is_file():
            model = _model_format(path)
            if model is not None:
                can_train = bool(model.get("training_output") or model.get("fine_tune"))
                return {
                    "path": str(path),
                    "exists": True,
                    "kind": "model",
                    "format": model["id"],
                    "format_label": model["id"],
                    "fine_tune": bool(model.get("fine_tune")),
                    "trainable": can_train,
                    "bytes": path.stat().st_size,
                    "message": model["note"],
                }
            descriptor = format_for_path(path)
            native = self._probe(path)
            ingestor = self.registry.for_path(path)
            format_id = descriptor.id if descriptor else native.get("format", path.suffix.lstrip(".") or "unknown")
            result = {
                "path": str(path),
                "exists": True,
                "kind": "dataset",
                "format": format_id,
                "format_label": format_id,
                "category": descriptor.category if descriptor else "unknown",
                "modality": descriptor.modality if descriptor else native.get("modality", "binary"),
                "handler": descriptor.handler if descriptor else "unknown",
                "recognized": descriptor is not None,
                "extractable": bool(descriptor and descriptor.training in {"extract", "train"}),
                "bytes": path.stat().st_size,
                "engine": native.get("engine", "python-router"),
                "trainable": ingestor is not None,
                "adapter": getattr(ingestor, "name", None),
                "message": (
                    "Ready for direct training"
                    if ingestor
                    else "Recognized built-in route; direct trainer for this modality is not implemented"
                ),
            }
            if ingestor:
                result.update(ingestor.inspect(path))
                result["trainable"] = bool(result.get("suggested_target", True))
            return result
        if not path.is_dir():
            raise DatasetError("Path is neither a regular file nor a directory")

        counts: dict[str, int] = {}
        modalities: dict[str, int] = {}
        trainable_files: list[Path] = []
        total_bytes = 0
        file_count = 0
        truncated = False
        for item in sorted(path.rglob("*"), key=lambda value: value.as_posix().casefold()):
            if item.is_symlink() or not item.is_file():
                continue
            file_count += 1
            if file_count > MAX_FOLDER_FILES:
                truncated = True
                break
            total_bytes += item.stat().st_size
            descriptor = format_for_path(item)
            extension = descriptor.id if descriptor else item.suffix.lower().lstrip(".") or "unknown"
            counts[extension] = counts.get(extension, 0) + 1
            modality = descriptor.modality if descriptor else "unknown"
            modalities[modality] = modalities.get(modality, 0) + 1
            if item.suffix.lower() in STREAM_EXTENSIONS and self.registry.for_path(item) is not None:
                trainable_files.append(item)
        if truncated:
            raise DatasetError(f"Dataset folder exceeds the {MAX_FOLDER_FILES} file safety limit")

        directory_result: dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "kind": "dataset",
            "format": "directory",
            "format_label": "Dataset folder",
            "modality": "mixed" if len(modalities) > 1 else next(iter(modalities), "unknown"),
            "modalities": modalities,
            "file_count": file_count,
            "trainable_file_count": len(trainable_files),
            "formats": counts,
            "bytes": total_bytes,
            "trainable": bool(trainable_files),
            "message": (
                f"Ready for streaming training from {len(trainable_files)} compatible files"
                if trainable_files
                else "Folder recognized, but it contains no directly trainable tabular files"
            ),
        }
        if trainable_files:
            first = trainable_files[0]
            ingestor = self.registry.for_path(first)
            assert ingestor is not None
            inspected = ingestor.inspect(first)
            directory_result.update(
                {
                    "sampled_rows": int(inspected.get("sampled_rows") or 1),
                    "columns": list(inspected.get("columns") or []),
                    "suggested_target": inspected.get("suggested_target"),
                    "task_types": list(inspected.get("task_types") or []),
                    "preview": list(inspected.get("preview") or []),
                    "schema_probe": str(first),
                }
            )
        return directory_result

    def _probe(self, path: Path) -> dict[str, Any]:
        if self.native_probe and self.native_probe.exists():
            try:
                result = subprocess.run(  # nosec B603
                    [str(self.native_probe), "--json", str(path)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=3,
                    check=True,
                )
                if len(result.stdout) > MAX_NATIVE_OUTPUT or len(result.stderr) > MAX_NATIVE_OUTPUT:
                    raise DatasetError("Native probe exceeded its output limit")
                decoded = json.loads(result.stdout)
                if isinstance(decoded, list):
                    decoded = decoded[0] if decoded and isinstance(decoded[0], dict) else {}
                if not isinstance(decoded, dict) or decoded.get("protocol") != NATIVE_PROTOCOL:
                    raise DatasetError("Native probe protocol is incompatible")
                return decoded
            except (DatasetError, OSError, subprocess.SubprocessError, json.JSONDecodeError):
                pass
        return {
            "format": path.suffix.lower().lstrip(".") or "unknown",
            "modality": "table" if path.suffix.lower() in STREAM_EXTENSIONS else "binary",
            "engine": "python-router",
        }
