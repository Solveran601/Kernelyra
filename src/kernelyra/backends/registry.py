from __future__ import annotations

import importlib.util
from typing import Any

from ..errors import ConfigurationError
from .native_backend import NativeBackend
from .numpy_backend import NumpyBackend
from .tensorflow_backend import TensorFlowBackend
from .torch_backend import TorchBackend


class BackendRegistry:
    """Closed registry of reviewed backends shipped with Kernelyra."""

    def __init__(self) -> None:
        self._backends: dict[str, Any] = {
            "native": NativeBackend,
            "numpy": NumpyBackend,
            "tensorflow": TensorFlowBackend,
            "torch": TorchBackend,
        }

    def names(self) -> list[str]:
        return sorted(self._backends)

    def create(self, name: str) -> Any:
        factory = self._backends.get(name)
        if not factory:
            raise ConfigurationError(f"Backend '{name}' не установлен. Доступны: {', '.join(self.names())}")
        return factory()

    def describe(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for name in self.names():
            factory = self._backends[name]
            dependency = {"tensorflow": "tensorflow", "torch": "torch"}.get(name)
            diagnostic = None
            if name == "native":
                from ..native_core import native_core_status

                native = native_core_status()
                available = bool(native["available"])
                diagnostic = native["diagnostic"]
            else:
                available = dependency is None or importlib.util.find_spec(dependency) is not None
            result.append(
                {
                    "name": name,
                    "version": str(getattr(factory, "version", "0")),
                    "task_types": list(getattr(factory, "task_types", ("binary_classification",))),
                    "metrics": list(getattr(factory, "metrics", ("loss", "accuracy"))),
                    "export_formats": list(getattr(factory, "export_formats", ("run-manifest-json",))),
                    "available": available,
                    "diagnostic": diagnostic
                    if name == "native"
                    else None
                    if available
                    else f'Install optional dependency: pip install "kernelyra-ai[{name}]"',
                }
            )
        return result
