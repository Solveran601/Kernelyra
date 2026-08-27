"""Run a local, evidence-first Kernelyra v5 demonstration.

It creates a small tabular CSV and JSONL, shows their actual routing, trains
the CSV with the native backend, then checks UTF-8 text chunking and causal
loss masks.  The text result is preparation only; v5 has no LLM trainer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from kernelyra import Config, Engine, __version__, batch_masked_text_examples, prepare_masked_text_examples
from kernelyra.ingestion.router import FormatRouter
from kernelyra.native_core import NativeCore, NativeCoreError, NativeTensorArena, native_core_status


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def write_samples(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "live_binary.csv"
    jsonl_path = root / "live_binary.jsonl"
    rows: list[dict[str, float | int]] = []
    for index in range(192):
        x1 = math.sin(index / 7.0)
        x2 = math.cos(index / 11.0)
        x3 = (index % 17) / 17.0
        label = int(x1 * 1.3 + x2 * 0.9 + x3 > 0.35)
        rows.append({"x1": x1, "x2": x2, "x3": x3, "label": label})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("x1", "x2", "x3", "label"))
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return csv_path, jsonl_path


def route_summary(router: FormatRouter, path: Path) -> dict[str, Any]:
    inspected = router.inspect(path)
    keep = ("format", "adapter", "engine", "trainable", "suggested_target", "columns", "numeric_columns", "streaming")
    return {name: inspected[name] for name in keep if name in inspected}


def text_summary() -> dict[str, Any]:
    text = (
        "Первое предложение задаёт тему. Второе продолжает ту же мысль и должно получить контекст.\n\n"
        "Третий абзац завершает пример без разрыва UTF-8 символов."
    )
    try:
        chunks = NativeCore().plan_text_chunks(
            text, minimum_bytes=32, target_bytes=64, maximum_bytes=96, overlap_bytes=24
        )
    except NativeCoreError as error:
        return {
            "available": False,
            "status": "requires the local experimental native DLL",
            "reason": str(error),
            "v5_claim": "No built-in text/LLM trainer is present in v5.",
        }
    return {
        "available": True,
        "status": "preprocessing and loss-mask preparation only; no LLM trainer is present",
        "source_bytes": len(text.encode("utf-8")),
        "content_reconstructs_source": "".join(item["content"] for item in chunks) == text,
        "loss_mask": text_mask_summary(text),
        "chunks": [
            {
                "context_start_byte": item["context_start_byte"],
                "content_start_byte": item["content_start_byte"],
                "end_byte": item["end_byte"],
                "context_prefix_bytes": item["context_prefix_bytes"],
                "content_preview": item["content"][:80],
            }
            for item in chunks
        ],
    }


def text_mask_summary(text: str) -> dict[str, Any]:
    """Show the exact repeated-context masking contract without training a model."""
    examples = prepare_masked_text_examples(
        text, minimum_bytes=32, target_bytes=64, maximum_bytes=96, overlap_bytes=24
    )
    batch = batch_masked_text_examples(examples[: min(2, len(examples))])
    enabled = int(batch.loss_mask.sum())
    tokens = int(batch.attention_mask.sum())
    return {
        "examples": len(examples),
        "first_context_prefix_tokens": examples[0].context_prefix_tokens if examples else 0,
        "later_context_prefix_tokens": examples[1].context_prefix_tokens if len(examples) > 1 else 0,
        "supervised_target_tokens": enabled,
        "masked_context_or_padding_tokens": tokens - enabled,
        "batch_shape": list(batch.input_ids.shape),
        "contract": "loss_mask=0 for carried context/padding; loss_mask=1 only for new content",
    }


def memory_arena_summary() -> dict[str, Any]:
    """Exercise the bounded Zig arena and expose its real accounting trace."""
    try:
        core = NativeCore()
        with NativeTensorArena(core=core, byte_budget=4096) as arena:
            arena.acquire_float32((64,), tag="first")
            mark = arena.mark()
            arena.acquire_float32((128,), tag="second")
            before_rewind = arena.stats
            arena.rewind(mark)
            after_rewind = arena.stats
            arena.acquire_float32((32,), tag="after-rewind")
            arena.reset()
            after_reset = arena.stats
    except NativeCoreError as error:
        return {"available": False, "reason": str(error)}
    return {
        "available": True,
        "native_core_version": core.version,
        "contract": "single 4096-byte arena; rewind and reset release leases without reallocating the arena",
        "before_rewind": before_rewind,
        "after_rewind": after_rewind,
        "after_reset": after_reset,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".test_workspaces") / "live-v5-showcase")
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    root = args.work_dir.resolve()
    csv_path, jsonl_path = write_samples(root / "data")
    router = FormatRouter()
    settings = (
        Config()
        .target("label")
        .backend("native")
        .cpu_only()
        .pack("careful")
        .resources(cpu=50, ram=40, threads=2)
        .optimizer(learning_rate=0.08)
        .steps(args.steps)
        .seed(20260826)
    )
    with Engine(root / "workspace") as engine:
        plan = engine.plan(csv_path, settings=settings)
        result = engine.fit(csv_path, settings=settings)
    summary = {
        "kernelyra_version": __version__,
        "native_core": native_core_status(),
        "csv_route": route_summary(router, csv_path),
        "jsonl_route": route_summary(router, jsonl_path),
        "training": {
            "status": str(result.run.status),
            "checkpoint": result.checkpoint,
            "backend": result.plan.backend,
            "execution": result.plan.execution,
            "algorithm_pack": result.plan.algorithm_pack,
            "data_mode": result.plan.data_mode,
            "metrics": result.run.metrics,
        },
        "memory_arena": memory_arena_summary(),
        "text_chunking": text_summary(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
