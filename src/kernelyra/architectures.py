"""Explicit architecture and checkpoint-format compatibility contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ArchitectureDescriptor:
    id: str
    modalities: tuple[str, ...]
    tasks: tuple[str, ...]
    backends: tuple[str, ...]
    implemented: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ARCHITECTURES: tuple[ArchitectureDescriptor, ...] = (
    ArchitectureDescriptor(
        "linear",
        ("table",),
        ("binary_classification", "multiclass_classification", "regression"),
        ("native", "numpy"),
        True,
        "Low-overhead linear model for weak devices and very wide tables.",
    ),
    ArchitectureDescriptor(
        "mlp",
        ("table",),
        ("binary_classification", "multiclass_classification", "regression"),
        ("torch", "tensorflow"),
        True,
        "Dense neural network for tabular features.",
    ),
    ArchitectureDescriptor("transformer", ("text", "multimodal"), ("language_modeling",), ("torch",), False, "Base edition does not yet contain a tokenizer and causal-LM trainer."),
    ArchitectureDescriptor("cnn", ("image", "audio"), ("classification",), ("torch", "tensorflow"), False, "Recognized contract; trainer is not implemented in the base edition."),
    ArchitectureDescriptor("vision-transformer", ("image",), ("classification",), ("torch",), False, "Recognized contract; trainer is not implemented in the base edition."),
    ArchitectureDescriptor("rnn", ("text", "audio", "timeseries"), ("sequence",), ("torch", "tensorflow"), False, "Recognized contract; trainer is not implemented in the base edition."),
    ArchitectureDescriptor("pointnet", ("3d",), ("classification", "segmentation"), ("torch",), False, "Recognized contract; trainer is not implemented in the base edition."),
    ArchitectureDescriptor("graph-neural-network", ("graph", "3d"), ("classification", "regression"), ("torch",), False, "Recognized contract; trainer is not implemented in the base edition."),
)

ARCHITECTURE_BY_ID = {item.id: item for item in ARCHITECTURES}

CHECKPOINT_FORMATS: tuple[dict[str, Any], ...] = (
    {
        "id": "kernelyra-npz",
        "extensions": [".npz"],
        "architectures": ["linear", "mlp"],
        "training_output": True,
        "fine_tune": True,
        "note": "Atomic Kernelyra checkpoint with tensors and optimizer state.",
    },
    {
        "id": "pytorch-state",
        "extensions": [".pt", ".pth"],
        "architectures": ["linear", "mlp", "transformer", "cnn", "vision-transformer", "rnn", "pointnet", "graph-neural-network"],
        "training_output": False,
        "fine_tune": True,
        "note": "Safe weights-only import for the PyTorch backend; not a base checkpoint output.",
    },
    {
        "id": "keras",
        "extensions": [".keras", ".h5", ".hdf5"],
        "architectures": ["linear", "mlp", "cnn", "rnn"],
        "training_output": False,
        "fine_tune": True,
        "note": "Safe Keras model import; not a base checkpoint output.",
    },
    {
        "id": "gguf",
        "extensions": [".gguf", ".ggml"],
        "architectures": ["transformer"],
        "training_output": False,
        "fine_tune": False,
        "note": "Inference/distribution container, not a generic training architecture or optimizer checkpoint.",
    },
    {
        "id": "safetensors",
        "extensions": [".safetensors"],
        "architectures": ["transformer", "cnn", "vision-transformer", "rnn", "pointnet", "graph-neural-network"],
        "training_output": False,
        "fine_tune": False,
        "note": "Recognized tensor container; architecture metadata and trainer integration are still required.",
    },
    {
        "id": "onnx",
        "extensions": [".onnx"],
        "architectures": ["linear", "mlp", "transformer", "cnn", "vision-transformer", "rnn", "pointnet", "graph-neural-network"],
        "training_output": False,
        "fine_tune": False,
        "note": "Inference exchange format in the base edition.",
    },
)

CHECKPOINT_BY_ID = {item["id"]: item for item in CHECKPOINT_FORMATS}


def resolve_training_contract(architecture: str, model_format: str, backend: str, task: str) -> tuple[str, str]:
    resolved_architecture = architecture.strip().lower()
    if resolved_architecture == "auto":
        resolved_architecture = "linear" if backend in {"native", "numpy"} else "mlp"
    descriptor = ARCHITECTURE_BY_ID.get(resolved_architecture)
    if descriptor is None:
        raise ConfigurationError(f"Unknown architecture '{architecture}'")
    if not descriptor.implemented:
        raise ConfigurationError(f"Architecture '{resolved_architecture}' is recognized but unavailable: {descriptor.note}")
    if backend not in descriptor.backends:
        raise ConfigurationError(f"Architecture '{resolved_architecture}' is not implemented by backend '{backend}'")
    if task not in descriptor.tasks:
        raise ConfigurationError(f"Architecture '{resolved_architecture}' does not support task '{task}'")

    resolved_format = model_format.strip().lower()
    if resolved_format == "auto":
        resolved_format = "kernelyra-npz"
    checkpoint = CHECKPOINT_BY_ID.get(resolved_format)
    if checkpoint is None:
        raise ConfigurationError(f"Unknown model format '{model_format}'")
    if resolved_architecture not in checkpoint["architectures"]:
        raise ConfigurationError(
            f"Model format '{resolved_format}' is incompatible with architecture '{resolved_architecture}'"
        )
    if not checkpoint["training_output"]:
        raise ConfigurationError(
            f"Model format '{resolved_format}' is recognized but cannot be produced by base training: {checkpoint['note']}"
        )
    return resolved_architecture, resolved_format


def describe_architectures() -> list[dict[str, Any]]:
    return [item.to_dict() for item in ARCHITECTURES]
