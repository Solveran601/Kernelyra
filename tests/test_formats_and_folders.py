from __future__ import annotations

import csv
import unittest
from pathlib import Path

from kernelyra import AutoTrainer, RunConfig, Workspace, extract_folder, extract_text, text_format_count
from kernelyra.errors import ConfigurationError
from kernelyra.formats import FORMAT_COUNT, format_for_path
from kernelyra.streaming import StreamingTabularSource, build_stream_spec
from tests.helpers import isolated_workspace


class FormatCatalogueTests(unittest.TestCase):
    def test_catalogue_has_more_than_180_unique_builtin_routes(self) -> None:
        self.assertGreaterEqual(FORMAT_COUNT, 180)
        for path, modality in (("sample.md", "text"), ("sample.png", "image"), ("sample.mp4", "video"), ("sample.glb", "3d"), ("sample.gguf", "model")):
            descriptor = format_for_path(path)
            self.assertIsNotNone(descriptor)
            self.assertEqual(descriptor.modality, modality)

    def test_more_than_180_text_routes_have_real_streaming_extraction(self) -> None:
        self.assertGreaterEqual(text_format_count(), 180)
        with isolated_workspace() as temporary:
            with (temporary / "one.md").open("w", encoding="utf-8", newline="") as handle:
                handle.write("hello\r\nworld")
            (temporary / "two.rs").write_text("fn main() {}", encoding="utf-8")
            chunks = list(extract_folder(temporary))
            self.assertEqual([Path(item.path).name for item in chunks], ["one.md", "two.rs"])
            self.assertEqual("".join(chunk.text for chunk in extract_text(temporary / "one.md")), "hello\nworld")

    def test_transformer_gguf_is_rejected_without_fake_training(self) -> None:
        with isolated_workspace() as temporary:
            dataset = temporary / "data.csv"
            with dataset.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("x", "target"))
                writer.writerows((index, index % 2) for index in range(64))
            with AutoTrainer(temporary / "project") as trainer:
                with self.assertRaisesRegex(ConfigurationError, "transformer.*unavailable"):
                    trainer.plan(dataset, backend="torch", architecture="transformer", model_format="gguf")


class FolderStreamingTests(unittest.TestCase):
    def test_compatible_csv_folder_is_one_streaming_dataset(self) -> None:
        with isolated_workspace() as temporary:
            folder = temporary / "dataset"
            folder.mkdir()
            for part in range(2):
                with (folder / f"part-{part}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("x", "kind", "target"))
                    writer.writerows((index, f"k{index % 3}", index % 2) for index in range(part * 50, (part + 1) * 50))
            spec = build_stream_spec(folder, "target")
            self.assertEqual(spec["format"], "folder")
            self.assertEqual(spec["file_count"], 2)
            self.assertEqual(spec["records"], 100)
            source = StreamingTabularSource(spec, seed=42)
            batch_x, batch_y = source.next_batch(16)
            source.close()
            self.assertEqual(batch_x.shape[0], 16)
            self.assertEqual(batch_y.shape, (16,))

            workspace = Workspace.open(temporary / "project")
            inspection = workspace.datasets.inspect(folder)
            self.assertTrue(inspection["trainable"])
            attached = workspace.datasets.attach_path(folder, "target")
            workspace.close()
            self.assertEqual(attached.records, 100)
            self.assertEqual(attached.manifest["source_kind"], "external_stream_folder")

    def test_folder_schema_mismatch_fails_before_training(self) -> None:
        with isolated_workspace() as temporary:
            folder = temporary / "dataset"
            folder.mkdir()
            (folder / "a.csv").write_text("x,target\n1,0\n" * 40, encoding="utf-8")
            (folder / "b.csv").write_text("different,target\n1,1\n" * 40, encoding="utf-8")
            with self.assertRaisesRegex(Exception, "schema"):
                build_stream_spec(folder, "target")


class InferenceCheckTests(unittest.TestCase):
    def test_held_out_requests_leave_checkpoint_unchanged(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            run = workspace.create_run(
                RunConfig(
                    dataset="demo",
                    backend="numpy",
                    architecture="linear",
                    model_format="kernelyra-npz",
                    target_metric=.5,
                    max_steps=100,
                )
            ).start()
            import time

            deadline = time.monotonic() + 30
            while run.status not in {"completed", "error", "error_recoverable"} and time.monotonic() < deadline:
                time.sleep(.05)
                run = workspace.runs.get(run.id).info
            self.assertEqual(run.status, "completed")
            report = workspace.inference_check(run.id, 20)
            workspace.close()
            self.assertEqual(report["summary"]["requests"], 20)
            self.assertTrue(report["checkpoint_immutable"])
            self.assertFalse(report["chat_model"])


if __name__ == "__main__":
    unittest.main()
