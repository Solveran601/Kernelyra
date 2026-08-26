"""Portable, reproducible experiment reports for completed or active runs."""

from __future__ import annotations

import html
import json
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import RunNotFoundError

if TYPE_CHECKING:
    from .workspace import Workspace


REPORT_CONTRACT = "kernelyra-experiment-report/1"


def build_experiment_report(workspace: Workspace, run_id: str) -> dict[str, Any]:
    run = workspace.storage.get_run(run_id)
    if run is None:
        raise RunNotFoundError("Run was not found")
    dataset = workspace.datasets.get(run.dataset)
    health = run.metrics.get("health", {}) if isinstance(run.metrics, dict) else {}
    return {
        "contract": REPORT_CONTRACT,
        "generated_at": time.time(),
        "run": run.to_dict(),
        "dataset": dataset.to_dict(),
        "data_contract": dataset.manifest.get("data_contract", run.data_contract),
        "data_health": dataset.manifest.get("data_health", {}),
        "execution": run.environment_manifest,
        "metrics": run.metrics,
        "health": health,
        "checkpoint": run.checkpoint,
        "reproduce": {
            "command": (
                f'kernelyra --workspace "{workspace.root}" train "{dataset.path}" '
                f'--target "{dataset.target}" --backend {run.backend} --execution {run.execution} '
                f"--pack {run.algorithm_pack} --cpu {run.cpu} --ram {run.ram} --gpu {run.gpu} "
                f"--threads {run.threads} --seed {run.seed} --max-steps {run.max_steps}"
            ),
            "dataset_sha256": dataset.sha256,
            "run_id": run.id,
        },
    }


def write_experiment_report(report: dict[str, Any], output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".html", ".htm"}:
        title = html.escape(f"Kernelyra experiment {report.get('run', {}).get('id', '')}")
        body = html.escape(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        content = f"<!doctype html><meta charset=\"utf-8\"><title>{title}</title><h1>{title}</h1><pre>{body}</pre>\n"
    else:
        content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    pending = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
    with pending.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, destination)
    return destination
