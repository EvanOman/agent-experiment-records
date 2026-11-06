"""Offline checks for the frozen direct-authoring publication snapshot.

This verifies saved artifacts and counts. Synthetic smoke inference is neither
MNIST accuracy measurement nor another blind confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parent
FROZEN = {
    "v001": {
        "graph.json": "4106d07bd8c7c1cf4b2c0ee3b356cf74a2e23fc7e028abac301edd53f8c49da5",
        "weights.npz": "0c276d650ea32e1a07615bf34ad7cd4a2c3ffb4d49cf1e88dd236f7b1ea1d605",
        "runner.py": "45fb2ed2704566cdb4055e0c07aa4ffebb892f08d2fcbee743394d073f4d35ea",
    },
    "v087": {
        "graph.json": "cd042f600410801e1464cc02c1acb6218f1cb60cd10bac06bbfc4d7c509f1649",
        "weights.npz": "8dbf117a9702b272113491b45ecf711771729e3275e0aefcfe36f4327f8e4ef4",
        "runner.py": "128299dfbdc1db4a535ebcde0d68231559e86b374b7354e4e47c81d0b4f1427a",
    },
    "v095": {
        "graph.json": "be60dbddd9af426d45de0f1263456dfdc5ad1304850a03d78da63bcc702a4576",
        "weights.npz": "3056fcdf1f821db073d9d029aec89c2e7319dc91db4771251a0768150e271a5f",
        "runner.py": "128299dfbdc1db4a535ebcde0d68231559e86b374b7354e4e47c81d0b4f1427a",
    },
}
EXPANDED_ENTRIES = {"v001": 4932, "v087": 1242619, "v095": 2471932}
NUMERICAL_SPECIFICATIONS = {"v001": 240, "v087": 1506, "v095": 1509}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_file(folder: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
        raise ValueError("Manifest entry must be an ordinary contained file")
    path = folder / name
    if not path.is_file() or path.is_symlink() or path.resolve().parent != folder.resolve():
        raise ValueError("Manifest entry must be an ordinary contained file")
    return path


def verify_checkpoint(candidate: str, folder: Path | None = None) -> dict:
    if candidate not in FROZEN:
        raise ValueError("Checkpoint is not part of the frozen publication")
    folder = ROOT / "checkpoints" / candidate if folder is None else Path(folder)
    manifest = json.loads(contained_file(folder, "manifest.json").read_text())
    if manifest.get("candidate") != candidate or not isinstance(manifest.get("files"), dict):
        raise ValueError("Checkpoint manifest identity or file mapping is invalid")
    for name, expected in manifest["files"].items():
        if sha256(contained_file(folder, name)) != expected:
            raise ValueError(f"Checkpoint file hash mismatch: {candidate}/{name}")
    for name, expected in FROZEN[candidate].items():
        if manifest["files"].get(name) != expected or sha256(contained_file(folder, name)) != expected:
            raise ValueError(f"Original frozen operational bytes changed: {candidate}/{name}")
    original = contained_file(folder, "SHA256SUMS.json")
    if sha256(original) != manifest["original_manifest_sha256"]:
        raise ValueError("Original artifact manifest does not match its recorded hash")
    for name, expected in json.loads(original.read_text()).items():
        if sha256(contained_file(folder, name)) != expected:
            raise ValueError(f"Original frozen artifact hash mismatch: {candidate}/{name}")
    with np.load(folder / "weights.npz", allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    if any(value.dtype != np.float32 or not np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError("Raw tensors must be finite non-pickled float32 arrays")
    count = sum(value.size for value in values.values())
    if count != EXPANDED_ENTRIES[candidate]:
        raise ValueError("Expanded raw tensor count changed")
    graph = json.loads((folder / "graph.json").read_text())
    known = {"pixels", *values}
    for node in graph["nodes"]:
        if node["name"] in known or not set(node["inputs"]) <= known:
            raise ValueError("Graph is not an ordered feed-forward graph")
        known.add(node["name"])
    if not set(graph["outputs"]) <= known or graph["prediction_logits"] not in known:
        raise ValueError("Graph output binding is invalid")
    return {
        "candidate": candidate, "files": dict(FROZEN[candidate]),
        "public_manifest_sha256": sha256(folder / "manifest.json"),
        "original_manifest_sha256": manifest["original_manifest_sha256"],
        "graph_nodes": len(graph["nodes"]), "raw_arrays": len(values),
        "expanded_tensor_entries": count,
        "independently_authored_numerical_specifications": NUMERICAL_SPECIFICATIONS[candidate],
        "counting_note": "The numerical-specification count excludes architectural decisions; expanded entries are not all independently typed.",
    }


def load_network(candidate: str, folder: Path | None = None):
    folder = ROOT / "checkpoints" / candidate if folder is None else Path(folder)
    specification = importlib.util.spec_from_file_location("frozen_" + candidate, folder / "runner.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.Network(folder)


def check_counts(result: dict, correct: int | None = None, total: int | None = None) -> None:
    n, c = result["total"], result["correct"]
    if type(c) is not int or type(n) is not int or not 0 <= c <= n or n <= 0:
        raise ValueError("Invalid aggregate count")
    if result["accuracy"] != c / n or (correct is not None and c != correct) or (total is not None and n != total):
        raise ValueError("Published aggregate score attribution does not match counts")
    if "confusion_true_rows_pred_columns" in result:
        confusion = np.asarray(result["confusion_true_rows_pred_columns"])
        if confusion.shape != (10, 10) or int(confusion.sum()) != n or int(np.trace(confusion)) != c:
            raise ValueError("Development aggregate confusion totals do not match")


def validate_snapshot() -> dict:
    snapshot = json.loads((ROOT / "snapshot.json").read_text())
    if (snapshot.get("schema") != "mnist-neural-direct-authoring-publication-v1" or
            snapshot.get("source_snapshot_manifest_sha256") != "5c10a58a630e234e062f15c3d1bfdeac8f3f1704695c33c6db01ab48d7878076" or
            snapshot.get("best_completed_development_candidate") != "v095" or
            snapshot.get("historical_confirmation_candidate") != "v087" or
            snapshot.get("holdout_threshold_achieved") is not False or
            snapshot.get("original_goal_status") != "active_unmet_at_cutoff" or
            snapshot.get("history", {}).get("completed_development_results") != 108):
        raise ValueError("Frozen publication snapshot identity or status drifted")
    artifacts = [verify_checkpoint(candidate) for candidate in FROZEN]
    results = {}
    for path in sorted((ROOT / "evidence/development").glob("v???_*.json")):
        result = json.loads(path.read_text())
        check_counts(result)
        key = (result["candidate"], result["split"])
        if key in results:
            raise ValueError("Duplicate development result")
        results[key] = result
    if len(results) != 108 or len({key[0] for key in results}) != 92:
        raise ValueError("Publication omits part of the 108-result, 92-candidate frozen history")
    for key, count in {("v001", "validation"): 5928, ("v001", "work"): 18087,
                       ("v087", "validation"): 9505, ("v095", "validation"): 9528}.items():
        check_counts(results[key], count, 30000 if key[1] == "work" else 10000)
    canonical = json.loads((ROOT / "evidence/confirmation/v087_result.json").read_text())
    check_counts(canonical, 9461, 10000)
    if canonical["candidate"] != "v087" or canonical["set_identity"] != "canonical_test" or canonical["status"] != "completed":
        raise ValueError("Canonical result belongs to a different checkpoint or set")
    for name, digest in FROZEN["v087"].items():
        if canonical["artifact_hashes"]["candidate"].get(name) != digest:
            raise ValueError("Canonical result is not bound to the published v087")
    for candidate in FROZEN:
        validation = results[(candidate, "validation")]
        record = snapshot["checkpoints"][candidate]
        check_counts(record["validation"], validation["correct"], validation["total"])
        if (record["independent_numerical_specifications"] != NUMERICAL_SPECIFICATIONS[candidate] or
                record["expanded_tensor_elements"] != EXPANDED_ENTRIES[candidate]):
            raise ValueError("Snapshot numerical-count attribution drifted")
        for name, digest in FROZEN[candidate].items():
            if validation["hashes"].get(name) != digest:
                raise ValueError("Validation result checkpoint binding changed")
    check_counts(snapshot["checkpoints"]["v087"]["canonical_test"], 9461, 10000)
    if (snapshot["checkpoints"]["v087"]["canonical_test"]["status"] != "completed_below_threshold" or
            snapshot["checkpoints"]["v095"]["canonical_test"]["status"] != "not_evaluated" or
            snapshot["checkpoints"]["v095"]["work"]["status"] != "pending_at_cutoff"):
        raise ValueError("Snapshot incorrectly transfers an unmeasured score or completion claim")
    history = json.loads((ROOT / "history/scores.json").read_text())
    if isinstance(history, dict):
        history = history["records"]
    if len(history) != len(results):
        raise ValueError("Score history omits or duplicates a completed development result")
    seen = set()
    for row in history:
        key = (row["candidate"], row["split"])
        if key in seen or key not in results:
            raise ValueError("Score history has an extra or duplicate result")
        seen.add(key)
        for field in ("correct", "total", "accuracy"):
            if row[field] != results[key][field]:
                raise ValueError("Score history disagrees with saved aggregate evidence")
    return {
        "schema": "mnist-neural-direct-authoring-verification-v1",
        "status": "passed", "kind": "offline_artifact_and_count_checks",
        "checkpoints": artifacts, "development_results": len(results),
        "best_development_validation": {"candidate": "v095", "correct": 9528, "total": 10000},
        "historical_canonical_confirmation": {"candidate": "v087", "correct": 9461, "total": 10000, "threshold_met": False},
        "dataset_loaded": False, "new_blind_confirmation": False,
    }


def smoke(candidate: str) -> dict:
    before = verify_checkpoint(candidate)
    images = np.stack([np.zeros((28, 28), dtype=np.uint8),
                       (np.arange(784).reshape(28, 28) % 256).astype(np.uint8)])
    network = load_network(candidate)
    batched = network.forward(images)
    separate = np.concatenate([network.forward(images[i:i + 1]) for i in range(len(images))])
    if (batched.shape != (2, 10) or not np.isfinite(batched).all() or
            not np.allclose(batched, separate, atol=1e-5, rtol=1e-6) or
            not np.array_equal(np.argmax(batched, axis=1), np.argmax(separate, axis=1))):
        raise ValueError("Synthetic inference shape, finiteness or batch parity failed")
    if verify_checkpoint(candidate) != before:
        raise ValueError("Checkpoint changed during smoke inference")
    return {
        "schema": "mnist-neural-direct-authoring-smoke-v1", "status": "passed",
        "candidate": candidate, "kind": "two_nonglyph_inputs_only", "images": 2,
        "logits_sha256": hashlib.sha256(batched.tobytes()).hexdigest(),
        "batch_single_max_absolute_difference": float(np.max(np.abs(batched - separate))),
        "batch_single_tolerance": {"atol": 1e-5, "rtol": 1e-6},
        "batch_single_argmax_exact": True,
        "artifact_hashes": before, "dataset_loaded": False, "accuracy_measured": False,
        "new_blind_confirmation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke"))
    parser.add_argument("--checkpoint", choices=tuple(FROZEN), default="v095")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_snapshot() if args.command == "verify" else smoke(args.checkpoint)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, ValueError, KeyError) as exc:
        sys.exit(str(exc))
