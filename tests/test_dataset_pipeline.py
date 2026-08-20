from __future__ import annotations

import csv
import json
import unittest

import numpy as np

from kernelyra.backends import BackendConfig
from kernelyra.backends.numpy_backend import NumpyBackend
from kernelyra.workspace import Workspace
from tests.helpers import isolated_workspace


class DatasetPipelineTests(unittest.TestCase):
    def test_csv_manifest_preprocessing_and_idempotent_import(self) -> None:
        with isolated_workspace() as temporary:
            source = temporary / "данные с пробелом.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["number", "category", "constant", "leak", "target"])
                for index in range(120):
                    target = index % 2
                    writer.writerow(
                        ["" if index % 17 == 0 else index / 10, ["red", "green", "blue"][index % 3], 1, target, target]
                    )
            workspace = Workspace.open(temporary / "project")
            first = workspace.datasets.import_file(source, "target")
            second = workspace.datasets.import_file(source, "target")
            self.assertEqual(first.id, second.id)
            self.assertEqual(first.sha256, first.manifest["sha256"])
            self.assertEqual(first.task_types, ["binary_classification"])
            self.assertNotIn(str(temporary), json.dumps(first.manifest, ensure_ascii=False))
            self.assertTrue(any("утечка target" in warning for warning in first.warnings))
            self.assertTrue(any("Константный" in warning for warning in first.warnings))
            x, y = workspace.datasets.load_arrays(first.id)
            self.assertTrue(np.isfinite(x).all())
            self.assertEqual(len(x), len(y))
            workspace.close()

    def test_jsonl_multiclass_and_npz_regression(self) -> None:
        with isolated_workspace() as temporary:
            jsonl = temporary / "multi.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for index in range(150):
                    handle.write(
                        json.dumps(
                            {"value": index / 10, "kind": f"k{index % 4}", "target": f"c{index % 3}"}
                        )
                        + "\n"
                    )
            npz = temporary / "regression.npz"
            rng = np.random.default_rng(5)
            x = rng.normal(size=(160, 3))
            y = x[:, 0] * 2.5 - x[:, 1] * .7 + .2
            np.savez(npz, x=x, y=y)

            workspace = Workspace.open(temporary / "project")
            multi = workspace.datasets.import_file(jsonl, "target")
            regression = workspace.datasets.import_file(npz)
            self.assertEqual(multi.task_types, ["multiclass_classification"])
            self.assertEqual(regression.task_types, ["regression"])
            self.assertIn("jsonl", workspace.capabilities["input_formats"])
            self.assertIn("npz", workspace.capabilities["input_formats"])

            backend = NumpyBackend()
            for dataset, task, metric in (
                (multi, "multiclass_classification", "macro_f1"),
                (regression, "regression", "r2"),
            ):
                arrays = workspace.datasets.load_arrays(dataset.id)
                session = backend.create_session(
                    BackendConfig(x=arrays[0], y=arrays[1], profile="low-memory", seed=11, task_type=task)
                )
                for _ in range(40):
                    backend.train_step(session, 16)
                result = backend.evaluate(session)
                self.assertIn(metric, result.metrics)
                self.assertTrue(np.isfinite(result.score))
            workspace.close()


if __name__ == "__main__":
    unittest.main()
