from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kernelyra import AutoTrainer, Config, Engine, finetune, fit, plan, train
from kernelyra.auto import _stream_limit
from kernelyra.backends.registry import BackendRegistry
from kernelyra.datasets import DatasetManager
from kernelyra.errors import ConfigurationError
from kernelyra.protocol import run_stdio
from kernelyra.streaming import StreamingTabularSource, build_stream_spec


def write_dataset(path: Path, rows: int = 400) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "group", "target"])
        writer.writerows((index / 10, f"g{index % 5}", index % 2) for index in range(rows))


class AutoTerminalTests(unittest.TestCase):
    def test_300_easy_api_variations_reach_the_real_planner(self) -> None:
        """Keep the 300-case contract in the suite without retaining a one-off harness."""
        profiles = {
            "low-memory": (30, 35),
            "balanced": (55, 55),
            "performance": (80, 75),
            "workstation": (100, 95),
            "custom": (47, 43),
        }
        batches: tuple[int | None, ...] = (None, 8, 16, 32, 64)
        precisions = ("auto", "float32", "float64")
        intervals: tuple[int | None, ...] = (None, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "matrix.csv"
            write_dataset(dataset, rows=160)
            defaults = Config().backend("numpy").target("target").stopping(maximum_steps=2, target_metric=.99)
            checked = 0
            with Engine(root / "workspace", settings=defaults) as engine:
                for profile, (cpu, ram) in profiles.items():
                    for batch in batches:
                        for precision in precisions:
                            for interval in intervals:
                                settings = (
                                    Config()
                                    .hardware(profile, cpu=cpu, ram=ram, gpu=0)
                                    .model(16, 8, precision=precision)
                                    .quality(evaluation_interval=interval, min_improvement=.0001)
                                    .guard(margin=.02, patience=2)
                                )
                                if batch is not None:
                                    settings.batch(batch, accept_risk=True)
                                for options in (settings, settings.to_dict()):
                                    resolved = engine.plan(dataset, settings=options)
                                    self.assertEqual(resolved.profile, profile)
                                    self.assertEqual(resolved.precision, precision)
                                    self.assertEqual(resolved.max_steps, 2)
                                    if batch is not None:
                                        self.assertEqual(resolved.batch_size, batch)
                                    if interval is not None:
                                        self.assertEqual(resolved.evaluation_interval, interval)
                                    json.dumps(resolved.to_dict(), ensure_ascii=False)
                                    checked += 1
            self.assertEqual(checked, 300)

    def test_easy_library_api_keeps_auto_simple_and_full_control_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "easy.csv"
            write_dataset(dataset)
            settings = (
                Config()
                .backend("numpy")
                .goal(.99)
                .steps(3)
                .batch(8)
                .resources(cpu=30, ram=35, gpu=0)
                .optimizer(learning_rate=.02, weight_decay=.001)
                .model(16, 8, precision="float32")
                .data(workers=0, prefetch=0)
                .quality(
                    evaluation_interval=1,
                    min_improvement=.001,
                    early_stopping_patience=5,
                    target_patience=2,
                )
                .guard(margin=.02, patience=2)
                .seed(9)
            )
            with Engine(root / "workspace") as engine:
                resolved = engine.plan(dataset, "target", settings=settings)
                result = engine.fit(dataset, "target", settings=settings)
            self.assertEqual(resolved.max_steps, 3)
            self.assertEqual(resolved.batch_size, 8)
            self.assertEqual(resolved.learning_rate, .02)
            self.assertEqual(resolved.hidden_layers, (16, 8))
            self.assertEqual(resolved.evaluation_interval, 1)
            self.assertEqual(resolved.min_improvement, .001)
            self.assertEqual(resolved.degradation_margin, .02)
            self.assertEqual(resolved.degradation_patience, 2)
            self.assertEqual(resolved.early_stopping_patience, 5)
            self.assertEqual(resolved.target_patience, 2)
            self.assertEqual(result.run.step, 3)
            self.assertTrue(Path(result.checkpoint or "").is_file())

            automatic = fit(
                dataset,
                "target",
                workspace=root / "automatic-workspace",
                backend="numpy",
                max_steps=1,
                target_metric=.99,
            )
            self.assertEqual(automatic.run.step, 1)

    def test_easy_configuration_composition_defaults_inspection_and_many(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            write_dataset(first, rows=80)
            write_dataset(second, rows=90)
            base = Config.from_mapping({"backend": "numpy", "max_steps": 2}).low_memory()
            changed = (
                base.copy()
                .merge({"target_metric": .98}, Config().quality(evaluation_interval=1))
                .hardware("custom", cpu=41, ram=42, gpu=0)
                .stopping(maximum_steps=3, early_stopping_patience=4, target_patience=2)
                .batch()
            )
            self.assertEqual(base.to_dict()["profile"], "low-memory")
            self.assertEqual(changed.to_dict()["profile"], "custom")
            self.assertNotIn("batch_size", changed.to_dict())
            with Engine(root / "workspace", settings=base) as engine:
                engine.configure(changed, precision="float32")
                inspected = engine.inspect(first)
                plans = engine.plan_many([first, second], "target")
                self.assertGreaterEqual(engine.hardware["cpu_threads"], 1)
                self.assertIn("backends", engine.capabilities)
            self.assertTrue(inspected["trainable"])
            self.assertEqual(len(plans), 2)
            self.assertTrue(all(item.backend == "numpy" for item in plans))
            self.assertTrue(all(item.max_steps == 3 for item in plans))
            self.assertTrue(all(item.profile == "custom" for item in plans))

            reset = changed.copy().automatic("profile", "cpu", "ram", "gpu")
            self.assertNotIn("profile", reset.to_dict())
            self.assertEqual(reset.automatic().to_dict(), {})

    def test_unknown_training_options_fail_instead_of_being_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.csv"
            write_dataset(dataset)
            with AutoTrainer(root / "workspace", environ={}) as trainer:
                with self.assertRaisesRegex(ConfigurationError, "Unknown training option"):
                    trainer.plan(dataset, target="target", backned="torch")

    def test_quality_guard_rejects_unsafe_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.csv"
            write_dataset(dataset)
            with AutoTrainer(root / "workspace", environ={}) as trainer:
                with self.assertRaisesRegex(ConfigurationError, "degradation_patience"):
                    trainer.plan(dataset, degradation_patience=0)
                with self.assertRaisesRegex(ConfigurationError, "evaluation_interval"):
                    trainer.plan(dataset, evaluation_interval=0)

    def test_public_api_is_exposed_and_auto_plan_respects_precedence(self) -> None:
        self.assertTrue(callable(plan))
        self.assertTrue(callable(train))
        self.assertTrue(callable(finetune))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.csv"
            config = root / "settings.toml"
            write_dataset(dataset)
            config.write_text(
                "[training]\nbackend = 'numpy'\nbatch_size = 8\nmax_steps = 250\n",
                encoding="utf-8",
            )
            with AutoTrainer(
                root / "workspace",
                config=config,
                environ={"KERNELYRA_BATCH_SIZE": "16"},
            ) as trainer:
                resolved = trainer.plan(dataset, batch_size=32, accept_batch_risk=True)
                self.assertEqual(resolved.backend, "numpy")
                self.assertEqual(resolved.batch_size, 32)
                self.assertEqual(resolved.max_steps, 250)
                self.assertEqual(resolved.sources["batch_size"], "explicit")
                self.assertEqual(resolved.sources["backend"], "config")
                self.assertEqual(resolved.target, "target")
                self.assertEqual(resolved.task, "binary_classification")

    def test_low_memory_stream_limit_is_below_the_hard_copy_limit(self) -> None:
        maximum = DatasetManager.MAX_IMPORT_BYTES
        self.assertEqual(_stream_limit("low-memory", maximum), 128 * 1024 * 1024)
        self.assertEqual(_stream_limit("workstation", maximum), maximum)

    def test_custom_profile_is_a_complete_manual_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.csv"
            write_dataset(dataset)
            with AutoTrainer(root / "workspace", environ={}) as trainer:
                resolved = trainer.plan(
                    dataset,
                    profile="custom",
                    cpu=47,
                    ram=43,
                    gpu=0,
                    backend="numpy",
                )
            self.assertEqual(resolved.profile, "custom")
            self.assertEqual((resolved.cpu, resolved.ram, resolved.gpu), (47, 43, 0))
            self.assertEqual(resolved.hidden_layers, (64, 32))

    def test_streaming_source_uses_bounded_batches_and_external_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "large.csv"
            write_dataset(dataset, rows=700)
            spec = build_stream_spec(dataset, "target")
            source = StreamingTabularSource(spec, seed=42)
            x, y = source.next_batch(17)
            self.assertEqual(x.shape, (17, spec["features"]))
            self.assertEqual(y.shape, (17,))
            self.assertLessEqual(len(source.validation_y), 4096)
            source.close()
            with AutoTrainer(root / "workspace", environ={}) as trainer:
                attached = trainer.workspace.datasets.attach_file(dataset, "target")
                self.assertEqual(attached.path, str(dataset.resolve()))
                self.assertEqual(attached.manifest["source_kind"], "external_stream")
                trainer.workspace.datasets.remove(attached.id)
                self.assertTrue(dataset.is_file(), "Removing metadata must never delete an external dataset")

    def test_stream_prefetch_workers_and_checkpoint_cursor_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "stream.tsv"
            write_dataset(dataset, rows=900)
            spec = build_stream_spec(dataset, "target")
            source = StreamingTabularSource(spec, seed=7, data_workers=2, prefetch=2)
            source.next_batch(23)
            cursor = source.state()["stream_rows_consumed"]
            expected_x, expected_y = source.next_batch(19)
            source.close()

            restored = StreamingTabularSource(spec, seed=7, data_workers=2, prefetch=1)
            restored.restore_rows(cursor)
            actual_x, actual_y = restored.next_batch(19)
            restored.close()
            np.testing.assert_allclose(actual_x, expected_x)
            np.testing.assert_array_equal(actual_y, expected_y)

    def test_jsonl_protocol_is_stable_and_bounded(self) -> None:
        requests = (
            json.dumps({"id": 1, "method": "ping", "params": {}}).encode("utf-8") + b"\n"
            + b"not-json\n"
        )
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            code = run_stdio(directory, input_stream=io.BytesIO(requests), output_stream=output)
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual(messages[0]["protocol"], "kernelyra-jsonl/1")
        self.assertTrue(messages[1]["ok"])
        self.assertFalse(messages[2]["ok"])
        self.assertEqual(messages[2]["error_type"], "JSONDecodeError")

    def test_jsonl_wire_output_is_codepage_independent(self) -> None:
        requests = json.dumps({"id": 1, "method": "unknown", "params": {}}).encode("utf-8") + b"\n"
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            run_stdio(directory, input_stream=io.BytesIO(requests), output_stream=output)
        wire = output.getvalue()
        wire.encode("ascii")
        self.assertEqual(json.loads(wire.splitlines()[-1])["error_type"], "ValueError")

    def test_numpy_executes_the_same_auto_plan_in_streaming_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "stream.csv"
            write_dataset(dataset, rows=700)
            with patch.object(DatasetManager, "MAX_IMPORT_BYTES", 1):
                with AutoTrainer(root / "workspace", environ={}) as trainer:
                    result = trainer.train(
                        dataset,
                        backend="numpy",
                        target="target",
                        max_steps=7,
                        target_metric=.99,
                    )
            self.assertEqual(result.plan.data_mode, "stream")
            self.assertEqual(result.plan.max_steps, 7)
            self.assertEqual(result.run.status, "completed")
            self.assertEqual(result.run.max_steps, 7)
            self.assertEqual(result.run.step, 7)
            self.assertEqual(result.run.effective_backend, "numpy")
            self.assertGreater(result.run.samples_seen, 0)
            self.assertIsNotNone(result.checkpoint)
            self.assertTrue(Path(result.checkpoint or "").is_file())

    def test_torch_backend_is_registered_without_importing_torch(self) -> None:
        described = {item["name"]: item for item in BackendRegistry().describe()}
        self.assertIn("torch", described)
        self.assertIn("available", described["torch"])


if __name__ == "__main__":
    unittest.main()
