"""Fast, self-contained verification for Kernelyra V3 data operations."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from kernelyra import Engine  # noqa: E402


def make_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_time", "income", "visits", "label"])
        writer.writeheader()
        for index in range(240):
            income = 24_000 + index * 155
            visits = index % 11
            label = int((income / 10_000) + visits * .25 >= 4.9)
            writer.writerow(
                {
                    "event_time": f"2026-01-{1 + index // 24:02d}",
                    "income": income,
                    "visits": visits,
                    "label": label,
                }
            )


def cli(workspace: Path, *arguments: str) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "kernelyra", "--workspace", str(workspace), "--json", *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="kernelyra-v3-") as temporary:
        root = Path(temporary)
        dataset = root / "events.csv"
        report_path = root / "report.json"
        make_csv(dataset)
        assert cli(root, "version")["version"] == "0.5.0a2"
        assert cli(root, "doctor")["ok"] is True
        cli_plan = cli(
            root,
            "plan",
            str(dataset),
            "--target",
            "label",
            "--backend",
            "numpy",
            "--profile",
            "eco",
            "--max-steps",
            "16",
        )
        assert cli_plan["split_policy"]["strategy"] == "temporal"
        cli_result = cli(
            root,
            "train",
            str(dataset),
            "--target",
            "label",
            "--backend",
            "numpy",
            "--profile",
            "eco",
            "--max-steps",
            "16",
            "--evaluation-interval",
            "4",
            "--target-metric",
            ".999",
        )
        assert cli_result["run"]["status"] == "completed"
        with Engine(root) as engine:
            doctor = engine.doctor(dataset, "label")
            assert doctor["contract"]["contract"] == "kernelyra-dataset-contract/1"
            assert doctor["contract"]["split_policy"]["strategy"] == "temporal"
            assert doctor["contract"]["chunk_policy"]["strategy"] == "adaptive_contiguous_ranges"

            plan = engine.plan(
                dataset,
                "label",
                backend="numpy",
                profile="eco",
                max_steps=16,
                evaluation_interval=4,
                target_metric=.999,
                seed=17,
            )
            assert plan.split_policy["strategy"] == "temporal"
            assert plan.data_contract["signature"]

            result = engine.fit(
                dataset,
                "label",
                backend="numpy",
                profile="eco",
                max_steps=16,
                evaluation_interval=4,
                target_metric=.999,
                seed=17,
            )
            report = engine.report(result.run.id, report_path)
            assert report["contract"] == "kernelyra-experiment-report/1"
            assert report_path.is_file()
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            assert payload["run"]["split_policy"]["strategy"] == "temporal"
            assert payload["data_contract"]["signature"]
    print("V3 data operations: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
