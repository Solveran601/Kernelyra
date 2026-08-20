from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from .auto import AutoTrainer
from .errors import KernelyraError

PROTOCOL_VERSION = "kernelyra-jsonl/1"
MAX_REQUEST_BYTES = 1024 * 1024


def _write(output: TextIO, payload: dict[str, Any]) -> None:
    # ASCII is a strict UTF-8 subset. Escaping non-ASCII characters keeps the
    # JSONL wire format independent from the Windows console code page while
    # json decoders reconstruct the original Unicode strings.
    output.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    output.flush()


def run_stdio(
    workspace: str | Path,
    config: str | Path | None = None,
    *,
    input_stream: BinaryIO | None = None,
    output_stream: TextIO | None = None,
) -> int:
    """Serve the stable, language-neutral Kernelyra JSONL protocol over stdio."""
    source = input_stream or sys.stdin.buffer
    output = output_stream or sys.stdout
    with AutoTrainer(workspace, config=config) as trainer:
        _write(output, {"type": "ready", "protocol": PROTOCOL_VERSION})
        while True:
            raw = source.readline(MAX_REQUEST_BYTES + 1)
            if not raw:
                return 0
            if len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                _write(output, {"id": None, "ok": False, "error": "Request exceeds 1 MiB", "error_type": "ProtocolError"})
                continue
            request_id: Any = None
            try:
                request = json.loads(raw.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
                request_id = request.get("id")
                method = request.get("method")
                params = request.get("params") or {}
                if not isinstance(params, dict):
                    raise ValueError("params must be a JSON object")
                if method == "ping":
                    result: Any = {"protocol": PROTOCOL_VERSION, "status": "ready"}
                elif method == "capabilities":
                    result = trainer.workspace.capabilities
                elif method == "hardware":
                    result = trainer.workspace.hardware
                elif method == "plan":
                    dataset = params.pop("dataset")
                    result = trainer.plan(dataset, **params).to_dict()
                elif method == "train":
                    dataset = params.pop("dataset")
                    result = trainer.train(dataset, **params).to_dict()
                elif method == "finetune":
                    dataset = params.pop("dataset")
                    model = params.pop("model")
                    result = trainer.finetune(model, dataset, **params).to_dict()
                else:
                    raise ValueError(f"Unknown method '{method}'")
                _write(output, {"id": request_id, "ok": True, "result": result})
            except (KernelyraError, OSError, ValueError, KeyError, TypeError) as error:
                _write(
                    output,
                    {
                        "id": request_id,
                        "ok": False,
                        "error": str(error),
                        "error_type": type(error).__name__,
                    },
                )
