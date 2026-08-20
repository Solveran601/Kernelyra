from __future__ import annotations

import ctypes
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kernelyra.backends.base import BackendConfig
from kernelyra.backends.native_backend import NativeBackend
from kernelyra.ingestion.csv_ingestor import CSVIngestor
from kernelyra.native_core import (
    COMPONENT_ALL,
    COMPONENT_FORTRAN_NUMERIC,
    COMPONENT_ZIG_MEMORY,
    NativeCore,
    NativeModel,
    NativeNumericCsv,
    NativeNumericCsvStream,
    native_core_status,
)
from kernelyra.streaming import StreamingTabularSource, build_stream_spec


@unittest.skipUnless(native_core_status()["available"], "native core binary is unavailable")
class NativeCoreTests(unittest.TestCase):
    @staticmethod
    def _pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def test_polyglot_kernels_match_numpy_and_cpp_fallbacks(self) -> None:
        core = NativeCore()
        self.assertTrue(core.component_mask & COMPONENT_FORTRAN_NUMERIC)
        self.assertTrue(core.component_mask & COMPONENT_ZIG_MEMORY)
        self.assertEqual(core.component_mask, COMPONENT_ALL)
        self.assertEqual(core.enabled_component_mask, COMPONENT_ALL)
        rng = np.random.default_rng(123)
        x = rng.normal(size=(257, 67)).astype(np.float32)
        errors = rng.normal(size=257).astype(np.float32)
        expected_gradient = x.T @ errors
        gradient = np.empty(67, dtype=np.float32)
        previous = core.set_component_mask(COMPONENT_FORTRAN_NUMERIC)
        try:
            core.library.kr_numeric_gradient_f32(
                self._pointer(x), self._pointer(errors), 257, 67, self._pointer(gradient)
            )
        finally:
            core.set_component_mask(previous)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-5, atol=2e-5)

        left = rng.normal(size=1031).astype(np.float32)
        right = rng.normal(size=1031).astype(np.float32)
        previous = core.set_component_mask(COMPONENT_FORTRAN_NUMERIC)
        try:
            fortran = float(
                core.library.kr_kernel_dot_f32(
                    self._pointer(left), self._pointer(right), len(left)
                )
            )
            core.set_component_mask(0)
            fallback = float(
                core.library.kr_kernel_dot_f32(
                    self._pointer(left), self._pointer(right), len(left)
                )
            )
        finally:
            core.set_component_mask(previous)
        expected_dot = float(left @ right)
        self.assertAlmostEqual(fortran, expected_dot, delta=abs(expected_dot) * 2e-5 + 2e-5)
        self.assertAlmostEqual(fallback, expected_dot, delta=abs(expected_dot) * 2e-5 + 2e-5)

        values = rng.normal(size=(73, 19)).astype(np.float32)
        means = values.mean(axis=0).astype(np.float32)
        stds = values.std(axis=0).astype(np.float32)
        expected = (values - means) / stds
        actual = values.copy()
        previous = core.set_component_mask(COMPONENT_ZIG_MEMORY)
        try:
            core.library.kr_memory_normalize_f32(
                self._pointer(actual), 73, 19, self._pointer(means), self._pointer(stds)
            )
        finally:
            core.set_component_mask(previous)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        if core.component_mask & COMPONENT_ZIG_MEMORY:
            self.assertIn("zig", core.components)
            pointer = core.library.kr_memory_alloc_aligned(4096, 64)
            self.assertTrue(pointer)
            self.assertEqual(int(pointer) % 64, 0)
            core.library.kr_memory_free_aligned(pointer)
        source = rng.normal(size=59).astype(np.float32)
        copied = np.empty_like(source)
        core.copy_f32(copied, source)
        np.testing.assert_array_equal(copied, source)
        core.zero_f32(copied)
        np.testing.assert_array_equal(copied, np.zeros_like(source))

    def test_binary_training_and_parameter_roundtrip(self) -> None:
        x = np.asarray(
            [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]], dtype=np.float32
        )
        y = np.asarray([0, 0, 1, 1], dtype=np.float32)
        with NativeModel(
            task="binary_classification", features=2, learning_rate=0.1, threads=2
        ) as model:
            for _ in range(150):
                loss = model.train_step(x, y)
            probabilities = model.predict(x)
            weights, bias = model.export_parameters()
            self.assertTrue(np.isfinite(loss))
            self.assertGreaterEqual(float(((probabilities >= 0.5) == y).mean()), 0.99)

        with NativeModel(task="binary_classification", features=2, threads=1) as restored:
            restored.import_parameters(weights, bias)
            np.testing.assert_allclose(restored.predict(x), probabilities, rtol=1e-6, atol=1e-6)

    def test_multiclass_probabilities_are_normalized(self) -> None:
        x = np.eye(3, dtype=np.float32)
        y = np.asarray([0, 1, 2], dtype=np.float32)
        with NativeModel(
            task="multiclass_classification",
            features=3,
            classes=3,
            learning_rate=0.1,
            threads=2,
        ) as model:
            for _ in range(100):
                model.train_step(x, y)
            probabilities = model.predict(x)
        self.assertEqual(probabilities.shape, (3, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)

    def test_numeric_csv_uses_native_standardization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "numeric.csv"
            rows = [f"{index},{index * 2},{index % 2}" for index in range(1, 41)]
            source.write_text("f1,f2,target\n" + "\n".join(rows) + "\n", encoding="utf-8")
            with NativeNumericCsv(source, "target") as native:
                x, y = native.arrays()
                self.assertEqual(native.feature_names, ("f1", "f2"))
                self.assertEqual(native.target, "target")
            np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=1e-6)
            np.testing.assert_allclose(x.std(axis=0), 1.0, atol=1e-6)
            np.testing.assert_array_equal(y, [index % 2 for index in range(1, 41)])

            metadata, imported_x, imported_y = CSVIngestor.import_file(source, "target")
            self.assertEqual(metadata["engine"], "kernelyra-native-csv/1")
            self.assertEqual(metadata["features"], 2)
            np.testing.assert_allclose(imported_x, x)
            np.testing.assert_array_equal(imported_y, y.astype(np.int64))

    def test_numeric_stream_matches_general_source_and_restores_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stream.csv"
            rows = [f"{index / 10},{index * 3},{index % 2}" for index in range(1, 201)]
            source.write_text("f1,f2,target\n" + "\n".join(rows) + "\n", encoding="utf-8")
            spec = build_stream_spec(source, "target")
            general = StreamingTabularSource(spec, seed=42)
            native = NativeNumericCsvStream(spec)
            try:
                expected_x, expected_y = general.next_batch(31)
                actual_x, actual_y = native.next_batch(31)
                np.testing.assert_allclose(actual_x, expected_x, rtol=2e-5, atol=2e-5)
                np.testing.assert_array_equal(actual_y, expected_y)
                native.restore_rows(0)
                restored_x, restored_y = native.next_batch(31)
                np.testing.assert_allclose(restored_x, expected_x, rtol=2e-5, atol=2e-5)
                np.testing.assert_array_equal(restored_y, expected_y)
                self.assertEqual(native.state()["stream_rows_consumed"], 31)
            finally:
                native.close()
                general.close()

    def test_native_backend_replaces_python_stream_and_checkpoints_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "backend-stream.csv"
            rows = [f"{index / 10},{index * 3},{index % 2}" for index in range(1, 401)]
            source.write_text("f1,f2,target\n" + "\n".join(rows) + "\n", encoding="utf-8")
            spec = build_stream_spec(source, "target")
            backend = NativeBackend()
            session = backend.create_session(
                BackendConfig(
                    x=None,
                    y=None,
                    profile="low-memory",
                    seed=11,
                    dataset_spec=spec,
                    resource_limits={"cpu_percent": 50},
                )
            )
            checkpoint = root / "stream.npz"
            try:
                self.assertEqual(session.metadata["stream_engine"], "kernelyra-native-csv-stream/1")
                self.assertEqual(backend.train_steps(session, 19, 3).samples, 57)
                backend.save_checkpoint(session, checkpoint, {"schema_version": 3})
                self.assertEqual(session.data_source.state()["stream_rows_consumed"], 57)
                backend.train_step(session, 7)
                backend.restore_checkpoint(session, checkpoint)
                self.assertEqual(session.data_source.state()["stream_rows_consumed"], 57)
            finally:
                backend.close_session(session)


if __name__ == "__main__":
    unittest.main()
