"""Guards and deterministic split logic for the direct-authoring snapshot."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/2026-11-05-mnist-neural-direct-authoring"
SPEC = importlib.util.spec_from_file_location("direct_authoring_replay_tests", EXPERIMENT / "replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)
REPOSITORY_SPEC = importlib.util.spec_from_file_location("publication_repository_verifier_tests", ROOT / "scripts/verify_repository.py")
repository_verifier = importlib.util.module_from_spec(REPOSITORY_SPEC)
REPOSITORY_SPEC.loader.exec_module(repository_verifier)


class FakePublicArchive:
    def __init__(self):
        self.members = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def __getitem__(self, key):
        self.members.append(key)
        if key == "x_test":
            return np.zeros((10000, 28, 28), dtype=np.uint8)
        if key == "y_test":
            return np.zeros(10000, dtype=np.uint8)
        raise AssertionError("Canonical replay must not load a training member")


class DirectAuthoringPublicationTests(unittest.TestCase):
    def test_original_artifacts_and_score_attribution(self):
        result = replay.verification.validate_snapshot()
        self.assertEqual(result["development_results"], 108)
        self.assertEqual(result["best_development_validation"], {"candidate": "v095", "correct": 9528, "total": 10000})
        self.assertEqual(result["historical_canonical_confirmation"]["candidate"], "v087")
        self.assertEqual(result["historical_canonical_confirmation"]["correct"], 9461)
        self.assertFalse(result["historical_canonical_confirmation"]["threshold_met"])

    def test_allocation_keeps_stable_tie_order(self):
        np.testing.assert_array_equal(replay.allocation(np.array([1, 1, 1]), 2), [1, 1, 0])
        np.testing.assert_array_equal(replay.allocation(np.array([7, 2, 1]), 4), [3, 1, 0])

    def test_recovered_recipe_is_deterministic_and_returns_only_development(self):
        labels = np.repeat(np.arange(10, dtype=np.uint8), 6000)
        first = replay.partition_indices(labels)
        second = replay.partition_indices(labels)
        self.assertEqual(set(first), {"work", "validation"})
        for name, size in (("work", 30000), ("validation", 10000)):
            self.assertEqual(first[name].shape, (size,))
            self.assertEqual(first[name].dtype.str, "<i8")
            self.assertEqual(len(np.unique(first[name])), size)
            np.testing.assert_array_equal(first[name], second[name])
            np.testing.assert_array_equal(np.bincount(labels[first[name]], minlength=10), [size // 10] * 10)
        self.assertEqual(np.intersect1d(first["work"], first["validation"]).size, 0)

    def test_wrong_labels_cannot_claim_original_split_hash(self):
        with self.assertRaisesRegex(ValueError, "original declaration"):
            replay.development_indices(np.repeat(np.arange(10, dtype=np.uint8), 6000))
        for labels in (np.zeros(59999, dtype=np.uint8), np.full(60000, 10, dtype=np.uint8)):
            with self.assertRaises(ValueError):
                replay.partition_indices(labels)

    def test_invalid_replay_arguments_fail_before_dataset_access(self):
        with patch.object(replay, "obtain_data") as access:
            for candidate, split, limit, workers, expected in (
                ("v095", "canonical_test", 10000, 4, None),
                ("v087", "canonical_test", 0, 4, None),
                ("v087", "canonical_test", 10001, 4, None),
                ("v087", "canonical_test", 10000, 3, None),
                ("v087", "canonical_test", 10000, 4, 9500),
                ("v087", "canonical_test", 10, 4, 11),
            ):
                with self.assertRaises(ValueError):
                    replay.replay(candidate, split, Path("unused.npz"), limit=limit,
                                  workers=workers, expect_correct=expected)
            access.assert_not_called()

    def test_canonical_loads_only_test_members_and_enforces_exact_count(self):
        archive = FakePublicArchive()
        with patch.object(replay.verification, "verify_checkpoint", return_value={"fixed": True}), \
             patch.object(replay.verification, "sha256", return_value=replay.SOURCE_SHA256), \
             patch.object(replay, "obtain_data", return_value=Path("verified-public.npz")), \
             patch.object(replay.np, "load", return_value=archive), \
             patch.object(replay, "initialize_network"), \
             patch.object(replay, "batch_count", side_effect=lambda batch: (len(batch[1]), len(batch[1]))), \
             patch.object(replay, "source_revision", return_value={"status": "synthetic test"}):
            result = replay.replay("v087", "canonical_test", Path("unused.npz"), workers=1)
        self.assertEqual(archive.members, ["x_test", "y_test"])
        self.assertEqual(result["correct"], 10000)
        self.assertEqual(result["expected_correct"], 9461)
        self.assertEqual(result["status"], "count_mismatch")
        self.assertFalse(result["new_blind_confirmation"])

    def test_existing_bad_source_is_preserved_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mnist.npz"
            path.write_bytes(b"invalid source")
            with patch.object(replay.urllib.request, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    replay.obtain_data(path, download=True)
                network.assert_not_called()
            self.assertEqual(path.read_bytes(), b"invalid source")

    def test_failed_download_checksum_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mnist.npz"
            with patch.object(replay.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong download")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    replay.obtain_data(path, download=True)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_manifest_entries_cannot_escape_or_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ordinary.py").write_text("fixture")
            (root / "link.py").symlink_to("ordinary.py")
            for name in ("../outside", "link.py", "/absolute.py"):
                with self.assertRaisesRegex(ValueError, "ordinary contained file"):
                    replay.verification.contained_file(root, name)

    def test_publication_manifest_checks_paths_duplicates_and_complete_coverage(self):
        for fault in (None, "escape", "duplicate_public", "duplicate_whole", "omitted_file"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = root / "payload.txt"
                payload.write_text("fixture")
                digest = hashlib.sha256(payload.read_bytes()).hexdigest()
                records = [{"public_path": "../outside" if fault == "escape" else "payload.txt", "public_sha256": digest}]
                if fault == "duplicate_public":
                    records.append(records[0])
                publication = root / "PUBLICATION_MANIFEST.json"
                publication.write_text(json.dumps({"files": records}))
                mapping = {"PUBLICATION_MANIFEST.json": hashlib.sha256(publication.read_bytes()).hexdigest()}
                if fault != "omitted_file":
                    mapping["payload.txt"] = digest
                whole = root / "SHA256SUMS.json"
                encoded = json.dumps(mapping)
                if fault == "duplicate_whole":
                    encoded = encoded[:-1] + ', "payload.txt": "' + digest + '"}'
                whole.write_text(encoded)
                with patch.object(repository_verifier, "DIRECT_EXPERIMENT", root):
                    if fault is None:
                        result = repository_verifier.verify_direct_publication_manifests([payload, publication, whole])
                        self.assertEqual(result["whole_experiment_files"], 2)
                    else:
                        with self.assertRaises(ValueError):
                            repository_verifier.verify_direct_publication_manifests([payload, publication, whole])


if __name__ == "__main__":
    unittest.main()
