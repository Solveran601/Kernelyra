from __future__ import annotations

import importlib.util
from typing import Any

from .architectures import CHECKPOINT_FORMATS, describe_architectures
from .backends.registry import BackendRegistry
from .formats import FORMAT_COUNT, describe_formats, format_counts
from .ingestion.registry import IngestorRegistry
from .models import TaskType


class CapabilityRegistry:
    """Single capability source used by SDK, CLI, API and MCP."""

    def __init__(self, backends: BackendRegistry, ingestors: IngestorRegistry):
        self.backends = backends
        self.ingestors = ingestors

    def snapshot(self) -> dict[str, Any]:
        backend_items = self.backends.describe()
        ingestor_items = self.ingestors.describe()
        optional_dependencies = [
            {
                "name": "torch",
                "available": importlib.util.find_spec("torch") is not None,
                "extra": "torch",
                "purpose": "PyTorch training backend",
            },
            {
                "name": "tensorflow",
                "available": importlib.util.find_spec("tensorflow") is not None,
                "extra": "tensorflow",
                "purpose": "TensorFlow training backend",
            },
            {
                "name": "pandas",
                "available": importlib.util.find_spec("pandas") is not None,
                "extra": "data",
                "purpose": "Chunked tabular streaming",
            },
            {
                "name": "pyarrow",
                "available": importlib.util.find_spec("pyarrow") is not None,
                "extra": "data",
                "purpose": "Parquet dataset ingestion",
            },
            {
                "name": "mcp",
                "available": importlib.util.find_spec("mcp") is not None,
                "extra": "mcp",
                "purpose": "stdio MCP server",
            },
        ]
        return {
            "task_types": [item.value for item in TaskType],
            "backends": backend_items,
            "ingestors": ingestor_items,
            "input_formats": [item["id"] for item in describe_formats()],
            "format_catalogue": describe_formats(),
            "metrics": sorted({metric for item in backend_items for metric in item["metrics"]}),
            "export_formats": sorted(
                {format_name for item in backend_items for format_name in item["export_formats"]}
            ),
            "optional_dependencies": optional_dependencies,
            "architectures": describe_architectures(),
            "model_formats": list(CHECKPOINT_FORMATS),
            "recognized_format_routes": FORMAT_COUNT,
            "format_counts": format_counts(),
            "extensions_enabled": False,
            "contract_version": "kernelyra-capabilities/2",
        }
