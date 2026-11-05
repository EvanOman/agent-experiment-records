"""Portable artifact, split, and input guards; accuracy is checked by replay CI."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "experiments/2026-11-04-mnist-direct-weights/reproduce.py"
SPEC = importlib.util.spec_from_file_location("mnist_reproduce", ENTRY)
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class MnistPackagingTests(unittest.TestCase):
    def test_checkpoint_and_inference_sources_match_the_frozen_artifacts(self):
        manifest = reproduce.verify_artifacts()
        self.assertEqual(manifest["candidate"], "c026")
        self.assertEqual(manifest["historical_weights_sha256"], reproduce.WEIGHTS_SHA256)
        for name, original in manifest["original_source_sha256"].items():
            if name != "materialize.py":
                self.assertEqual(reproduce.sha256(reproduce.MODEL_DIR / name), original, name)

    def test_primary_split_is_unique_stratified_and_reproducible(self):
        labels = np.repeat(np.arange(10, dtype=np.uint8), 6000)
        first = reproduce.primary_indices(labels)
        second = reproduce.primary_indices(labels)
        self.assertEqual(len(first), 10000)
        self.assertEqual(len(np.unique(first)), 10000)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.bincount(labels[first]), [1000] * 10)
        self.assertTrue(np.all((first >= 0) & (first < 60000)))

    def test_primary_split_rejects_noncanonical_labels(self):
        for labels in (np.zeros(10000, dtype=np.uint8),
                       np.full(60000, 10, dtype=np.uint8)):
            with self.assertRaises(ValueError):
                reproduce.primary_indices(labels)

    def test_invalid_replay_arguments_fail_before_dataset_access(self):
        nonexistent = Path("a-dataset-that-does-not-exist.npz")
        for limit, batch_size, expected in ((0, 100, None), (10001, 100, None),
                                            (1, 0, None), (1, 101, None), (1, 1, 2)):
            with self.assertRaises(ValueError):
                reproduce.replay(nonexistent, limit, batch_size, expected)

    def test_existing_bad_dataset_is_preserved_and_not_downloaded_over(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "mnist.npz"
            data.write_bytes(b"not canonical MNIST")
            with patch.object(reproduce.urllib.request, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    reproduce.obtain_data(data, download=True)
                network.assert_not_called()
            self.assertEqual(data.read_bytes(), b"not canonical MNIST")

    def test_model_manifest_cannot_escape_its_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (root / "outside.py").write_text("not an approved inference dependency")
            manifest = {"historical_weights_sha256": reproduce.WEIGHTS_SHA256,
                        "files": {"../outside.py": "unused"}}
            (model / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "ordinary contained file"):
                reproduce.verify_artifacts(model)

    def test_model_manifest_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "actual.py").write_text("approved filenames must still be ordinary files")
            (model / "alias.py").symlink_to("actual.py")
            manifest = {"historical_weights_sha256": reproduce.WEIGHTS_SHA256,
                        "files": {"alias.py": "unused"}}
            (model / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "ordinary contained file"):
                reproduce.verify_artifacts(model)


if __name__ == "__main__":
    unittest.main()
