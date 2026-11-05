#!/usr/bin/env python3
"""Verify or replay c026 without training or contacting a model provider.

``verify`` reconstructs the saved tensors from directly authored parameters.
``smoke`` tests synthetic forward inference; it is not an accuracy benchmark.
``replay`` evaluates a reconstruction of the already-consumed historical split.
It is a public reproducibility check, not a new independent confirmation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import urllib.request

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
REPO_ROOT = HERE.parents[1]
SOURCE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SOURCE_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
WEIGHTS_SHA256 = "298c01852a29ce46f8b41f339a2adf47b4982e6d6b4ff5b23a7934bfcb946123"
SPLIT_SEED = 2026110402
PRIMARY_COUNT = 10000
HISTORICAL_CORRECT = 9603
MODULE_NAMES = (
    "model", "shared_features", "part_metric", "part_metric_v2",
    "part_metric_v2_fast", "topology_features", "materialize",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest() -> dict:
    return json.loads((MODEL_DIR / "manifest.json").read_text())


def verify_artifacts(model_dir: Path = MODEL_DIR) -> dict:
    """Validate the portable artifact manifest before any model code executes."""
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if manifest["historical_weights_sha256"] != WEIGHTS_SHA256:
        raise ValueError("Manifest does not identify the published frozen checkpoint")
    for name, expected in manifest["files"].items():
        path = model_dir / name
        if path.is_symlink() or not path.resolve().is_relative_to(model_dir.resolve()):
            raise ValueError(f"Artifact is not an ordinary contained file: {name}")
        if sha256(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    if sha256(model_dir / "weights.npz") != WEIGHTS_SHA256:
        raise ValueError("Checkpoint differs from the historical frozen numerical weights")
    return manifest


def load_model():
    """Load only the packaged modules, retaining their original import equations."""
    verify_artifacts()
    location = str(MODEL_DIR)
    if location not in sys.path:
        sys.path.insert(0, location)
    for name in MODULE_NAMES:
        existing = sys.modules.get(name)
        if existing is not None and Path(existing.__file__).resolve().parent != MODEL_DIR:
            raise RuntimeError(f"Unrelated module already loaded under {name!r}; use a fresh process")
    model = importlib.import_module("model")
    for name in MODULE_NAMES:
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != MODEL_DIR:
            raise RuntimeError(f"Inference dependency resolved outside the packaged model: {name}")
    weights = model.load_weights(MODEL_DIR / "weights.npz")
    for value in weights.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return model, weights


def verify_reconstruction() -> dict:
    """Expand authored parameters without opening an image dataset."""
    manifest = verify_artifacts()
    model, _ = load_model()
    writer = importlib.import_module("materialize")
    parameters = json.loads((MODEL_DIR / manifest["parameter_source"]).read_text())
    authored = [glyph for name in manifest["glyph_sources_order"]
                for glyph in json.loads((MODEL_DIR / name).read_text())]
    if len(authored) != manifest["glyph_count"] or len({g["name"] for g in authored}) != len(authored):
        raise ValueError("Authored glyph bank has missing or duplicate forms")
    glyphs = [{**glyph, "width": glyph["width"] * multiplier,
               "name": glyph["name"] + "__width" + str(multiplier)}
              for glyph in authored for multiplier in parameters["stroke_width_multipliers"]]
    if len(glyphs) != manifest["template_count"]:
        raise ValueError("Shared width expansion has the wrong template count")
    reconstructed = {}
    for index, deskew in enumerate(parameters["deskew_views"]):
        fields = writer.template_fields(glyphs, {**parameters, "deskew": deskew})
        prefix = "" if index == 0 else f"view{index}_"
        reconstructed.update({prefix + key: value for key, value in fields.items()})
    reconstructed["labels"] = np.array([g["label"] for g in glyphs])
    reconstructed["names"] = np.array([g["name"] for g in glyphs])
    arrays = {}
    with np.load(MODEL_DIR / "weights.npz", allow_pickle=False) as stored:
        if set(stored.files) != set(reconstructed) | {"parameters_json"}:
            raise ValueError("Saved checkpoint has unexpected fields")
        if json.loads(str(stored["parameters_json"])) != parameters:
            raise ValueError("Saved shared parameters differ from directly authored JSON")
        for key, value in reconstructed.items():
            saved = stored[key]
            if value.dtype != saved.dtype or not np.array_equal(value, saved):
                raise ValueError(f"Exact data-free tensor reconstruction failed: {key}")
            arrays[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                           "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    verify_artifacts()
    return {"schema": "public-mnist-verification-v1", "candidate": "c026",
            "weights_sha256": WEIGHTS_SHA256, "exact_reconstruction": True,
            "authored_glyphs": len(authored), "expanded_templates_per_view": len(glyphs),
            "reconstructed_fields": len(arrays), "arrays": arrays,
            "dataset_access": False, "conventional_optimizer_training": False,
            "data_fitted_parameter_generation": False,
            "limitation": "Exact expansion proves tensor reproducibility, not the complete historical authorship or absence of leakage."}


def synthetic_images() -> np.ndarray:
    images = np.zeros((4, 28, 28), dtype=np.uint8)
    images[1] = 255
    images[2, 4:24, 13:16] = 255
    y, x = np.indices((28, 28))
    radius = ((x - 13.5) / 7.5) ** 2 + ((y - 13.5) / 10) ** 2
    images[3, np.abs(radius - 1) < 0.22] = 255
    images.flags.writeable = False
    return images


def smoke_test() -> dict:
    model, weights = load_model()
    images = synthetic_images()
    params_before = json.dumps(weights["parameters"], sort_keys=True)
    scores = model.forward(images, weights)
    predictions = scores.argmax(axis=1)
    if scores.shape != (4, 10) or not np.isfinite(scores).all():
        raise ValueError("Synthetic forward inference produced invalid scores")
    if int(predictions[2]) != 1:
        raise ValueError("A synthetic vertical-stroke digit was not recognized as one")
    order = np.array([3, 0, 2, 1, 2, 0])
    permuted = model.predict(images[order], weights)
    singleton = np.array([model.predict(images[i:i + 1], weights)[0] for i in range(len(images))])
    if not np.array_equal(permuted, predictions[order]) or not np.array_equal(singleton, predictions):
        raise ValueError("Predictions depend on batch order, size, or duplication")
    if params_before != json.dumps(weights["parameters"], sort_keys=True):
        raise ValueError("Inference mutated shared parameters")
    verify_artifacts()
    return {"schema": "public-mnist-smoke-v1", "candidate": "c026",
            "weights_sha256": WEIGHTS_SHA256, "synthetic_examples": len(images),
            "predictions": predictions.tolist(), "finite_scores": True,
            "known_vertical_stroke_prediction": 1, "batch_order_invariance": True,
            "read_only_inputs_and_tensors": True, "dataset_access": False,
            "limitation": "Synthetic behavior checks only; this does not reproduce MNIST accuracy."}


def primary_indices(labels: np.ndarray) -> np.ndarray:
    """Recreate the predeclared historical stratified split, with no model access."""
    if labels.shape != (60000,) or not np.isin(labels, np.arange(10)).all():
        raise ValueError("Expected the canonical 60,000 MNIST training labels")
    rng = np.random.default_rng(SPLIT_SEED)
    counts = np.bincount(labels, minlength=10)
    ideal = counts / 6
    allocation = np.floor(ideal).astype(int)
    for digit in np.argsort(-(ideal - allocation), kind="stable")[:PRIMARY_COUNT - allocation.sum()]:
        allocation[digit] += 1
    holdout, development = [], []
    for digit in range(10):
        indices = rng.permutation(np.flatnonzero(labels == digit))
        holdout.extend(indices[:allocation[digit]])
        development.extend(indices[allocation[digit]:])
    holdout = rng.permutation(holdout)
    # The original setup also permuted development after reserving holdout.
    rng.permutation(development)
    return holdout


def obtain_data(path: Path, download: bool) -> None:
    if path.exists():
        if sha256(path) != SOURCE_SHA256:
            raise ValueError("Dataset checksum mismatch; existing files are never overwritten")
        return
    if not download:
        raise ValueError("Supply the canonical dataset using --data, or permit --download")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download")
    if temporary.exists():
        raise ValueError("A partial download already exists; inspect/remove it before retrying")
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, temporary.open("xb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha256(temporary) != SOURCE_SHA256:
            raise ValueError("Downloaded dataset differs from the canonical source checksum")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def replay(data: Path, limit: int, batch_size: int, expect_correct: int | None = None) -> dict:
    if not 1 <= limit <= PRIMARY_COUNT or not 1 <= batch_size <= 100:
        raise ValueError("Replay limit must be1..10000 and batch size1..100")
    if expect_correct is not None and not 0 <= expect_correct <= limit:
        raise ValueError("Expected correct count must lie within the requested sample count")
    if sha256(data) != SOURCE_SHA256:
        raise ValueError("Dataset checksum mismatch")
    model, weights = load_model()
    with np.load(data, allow_pickle=False) as archive:
        images, labels = archive["x_train"], archive["y_train"]
        if images.shape != (60000, 28, 28) or images.dtype != np.uint8:
            raise ValueError("Expected canonical MNIST training image shape and dtype")
        selected = primary_indices(labels)[:limit]
        pixels = images[selected]
        # Labels remain with the evaluator; they are never an inference argument.
        targets = labels[selected]
    pixels.flags.writeable = False
    before = time.monotonic()
    predictions = np.concatenate([model.predict(pixels[i:i + batch_size], weights)
                                  for i in range(0, limit, batch_size)])
    correct = int((predictions == targets).sum())
    verify_artifacts()
    report = {
        "schema": "public-mnist-replay-v1", "candidate": "c026",
        "experiment": "2026-11-04-mnist-direct-weights",
        "evaluation": "post-experiment replay of the historical primary split",
        "new_independent_confirmation": False, "canonical_benchmark_test_set": False,
        "split_seed": SPLIT_SEED, "n": limit, "correct": correct,
        "accuracy": correct / limit, "batch_size": batch_size,
        "expected_correct": expect_correct,
        "matches_expected": None if expect_correct is None else correct == expect_correct,
        "historical_full_result": {"correct": HISTORICAL_CORRECT, "n": PRIMARY_COUNT},
        "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
        "weights_sha256": WEIGHTS_SHA256,
        "model_manifest_sha256": sha256(MODEL_DIR / "manifest.json"),
        "replay_code_sha256": sha256(Path(__file__).resolve()),
        "selected_source_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "model_code_sha256": {name + ".py": sha256(MODEL_DIR / (name + ".py"))
                              for name in MODULE_NAMES if name != "materialize"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.system(),
                        "machine": platform.machine()},
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - before,
        "conventional_optimizer_training": False,
        "data_fitted_parameter_generation": False,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Offline checksums and exact data-free tensor reconstruction")
    commands.add_parser("smoke", help="Synthetic inference/order checks; no dataset accuracy claim")
    full = commands.add_parser("replay", help="Public replay of the consumed historical split")
    full.add_argument("--data", type=Path, default=REPO_ROOT / ".cache" / "mnist.npz")
    full.add_argument("--download", action="store_true")
    full.add_argument("--limit", type=int, default=PRIMARY_COUNT)
    full.add_argument("--batch-size", type=int, default=100)
    full.add_argument("--output", type=Path)
    full.add_argument("--expect-correct", type=int)
    args = parser.parse_args()
    if args.command == "verify":
        report = verify_reconstruction()
    elif args.command == "smoke":
        report = smoke_test()
    else:
        if args.output is not None and args.output.exists():
            parser.error("The output report already exists; choose a new output path")
        if not 1 <= args.limit <= PRIMARY_COUNT or not 1 <= args.batch_size <= 100:
            parser.error("--limit must be1..10000; --batch-size must be1..100")
        if args.expect_correct is not None and not 0 <= args.expect_correct <= args.limit:
            parser.error("--expect-correct must lie within the requested sample count")
        obtain_data(args.data, args.download)
        report = replay(args.data, args.limit, args.batch_size, args.expect_correct)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report.get("matches_expected") is False else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
