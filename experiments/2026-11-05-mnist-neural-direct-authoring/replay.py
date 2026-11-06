"""Public inference-only replay of saved direct-authoring checkpoints.

The v087 canonical replay is a rerun on an exposed public test set, not a new
blind confirmation. No prepared sealed file or confirmation ledger is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SOURCE_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
VALIDATION_INDEX_SHA256 = "c15721627096718f6f1a748dd8536a315be3c387000c9ab59a311b3cb7ea9449"
WORK_INDEX_SHA256 = "27d935c2c23391a653cd9328a5fb3a2f8b4de042c99aa09ceb085b89128dd2e1"
SPLIT_SEED = 2026110501
EXPECTED = {("v087", "canonical_test"): 9461,
            ("v001", "validation"): 5928,
            ("v087", "validation"): 9505,
            ("v095", "validation"): 9528}
_SPEC = importlib.util.spec_from_file_location("direct_authoring_verify", ROOT / "verify.py")
verification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verification)
_NETWORK = None


def obtain_data(path: Path, download: bool = False) -> Path:
    """Checksum first, then load; never replace an existing mismatched file."""
    path = Path(path)
    if path.exists():
        if not path.is_file() or path.is_symlink() or verification.sha256(path) != SOURCE_SHA256:
            raise ValueError("Public MNIST source checksum mismatch; existing file preserved")
        return path
    if not download:
        raise ValueError("Public MNIST is missing; supply --download or a checksum-matching --data file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="mnist-download-", suffix=".part", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if verification.sha256(temporary) != SOURCE_SHA256:
            raise ValueError("Downloaded public MNIST source checksum mismatch")
        # A hard link publishes a complete file exclusively, never clobbering a
        # file created by another process while this download was running.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def allocation(counts: np.ndarray, total: int) -> np.ndarray:
    """Exact original operation order, including stable remainder ordering."""
    ideal = counts * (total / counts.sum())
    out = np.floor(ideal).astype(int)
    for label in np.argsort(-(ideal - out), kind="stable")[:total - out.sum()]:
        out[label] += 1
    return out


def partition_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Recovered original recipe; return only work and validation indices.

    The first two permutations advance the original RNG state. Their contents
    are immediately discarded: reserve memberships are never returned, saved,
    displayed or used for inference. No image data is needed by this function.
    """
    y = np.asarray(labels)
    if y.shape != (60000,) or y.dtype.kind not in "iu" or np.any((y < 0) | (y > 9)):
        raise ValueError("Split reconstruction requires the canonical training-label schema")
    rng = np.random.default_rng(SPLIT_SEED)
    counts = np.bincount(y, minlength=10)
    reserve_counts = allocation(counts, 20000)
    r1_counts = allocation(reserve_counts, 10000)
    r1, r2, dev = [], [], []
    for label in range(10):
        ids = rng.permutation(np.flatnonzero(y == label))
        r1.extend(ids[:r1_counts[label]])
        r2.extend(ids[r1_counts[label]:reserve_counts[label]])
        dev.extend(ids[reserve_counts[label]:])
    rng.permutation(r1)
    rng.permutation(r2)
    del r1, r2
    dev = rng.permutation(dev)
    vcounts = allocation(np.bincount(y[dev], minlength=10), 10000)
    work, validation = [], []
    for label in range(10):
        ids = rng.permutation(dev[y[dev] == label])
        validation.extend(ids[:vcounts[label]])
        work.extend(ids[vcounts[label]:])
    work = rng.permutation(work).astype("<i8", copy=False)
    validation = rng.permutation(validation).astype("<i8", copy=False)
    return {"work": work, "validation": validation}


def development_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    indices = partition_indices(labels)
    for name, count, expected in (("work", 30000, WORK_INDEX_SHA256),
                                  ("validation", 10000, VALIDATION_INDEX_SHA256)):
        values = indices[name]
        if values.shape != (count,) or len(np.unique(values)) != count or hashlib.sha256(values.tobytes()).hexdigest() != expected:
            raise ValueError(f"Reconstructed {name} indices do not match the original declaration")
    if np.intersect1d(indices["work"], indices["validation"]).size:
        raise ValueError("Reconstructed development partitions overlap")
    return indices


def split_record(data: Path, download: bool = False) -> dict:
    source = obtain_data(data, download)
    with np.load(source, allow_pickle=False) as archive:
        indices = development_indices(archive["y_train"])
    return {
        "schema": "mnist-direct-authoring-development-split-v1", "status": "verified",
        "source_partition": "y_train", "source_sha256": SOURCE_SHA256,
        "declared_seed": SPLIT_SEED, "index_dtype": "<i8",
        "partitions": {name: {"n": len(values), "sha256": hashlib.sha256(values.tobytes()).hexdigest()}
                       for name, values in indices.items()},
        "index_hash_encoding": "C-order little-endian signed-64-bit raw array bytes",
        "recipe_status": "exact original preparation code recovered by the coordinator and independently checked against both original index digests",
        "dataset_images_loaded": False, "public_training_labels_loaded": True,
        "reserve_membership_returned_or_serialized": False,
        "reserve_files_accessed": False, "new_blind_confirmation": False,
    }


def initialize_network(candidate: str) -> None:
    global _NETWORK
    _NETWORK = verification.load_network(candidate)


def batch_count(batch: tuple[np.ndarray, np.ndarray]) -> tuple[int, int]:
    images, labels = batch
    logits = _NETWORK.forward(images)
    if not isinstance(logits, np.ndarray) or logits.shape != (len(images), 10) or not np.isfinite(logits).all():
        raise ValueError("Frozen runner produced invalid logits")
    return int(np.count_nonzero(np.argmax(logits, axis=1) == labels)), len(labels)


def source_revision() -> dict:
    """Report a commit only when these public replay inputs are tracked/clean."""
    try:
        repository = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip())
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        relative = ROOT.relative_to(repository).as_posix()
        status = subprocess.check_output(["git", "status", "--porcelain", "--", relative], cwd=repository, text=True)
        tracked = subprocess.check_output(["git", "ls-files", "--", relative + "/replay.py"], cwd=repository, text=True).strip()
        clean = bool(tracked) and not status
        return {"git_commit": commit if clean else None,
                "status": "clean committed snapshot" if clean else "uncommitted publication working tree",
                "replay_files_match_reported_commit": clean}
    except (subprocess.CalledProcessError, OSError, ValueError):
        return {"git_commit": None, "status": "no Git source binding available",
                "replay_files_match_reported_commit": False}


def replay(candidate: str, split: str, data: Path, *, download: bool = False,
           limit: int = 10000, workers: int = 4, expect_correct: int | None = None) -> dict:
    if (candidate, split) not in EXPECTED:
        raise ValueError("Only published checkpoint/split pairs may be replayed")
    if type(limit) is not int or not 1 <= limit <= 10000 or workers not in (1, 2, 4):
        raise ValueError("Replay limit or worker count is invalid")
    full = limit == 10000
    expected = EXPECTED[(candidate, split)] if full else expect_correct
    if expect_correct is not None and (not 0 <= expect_correct <= limit or
                                      (full and expect_correct != expected)):
        raise ValueError("Expected count differs from the recorded full replay count or sample size")
    before = verification.verify_checkpoint(candidate)
    entry_hash = verification.sha256(Path(__file__))
    verifier_hash = verification.sha256(ROOT / "verify.py")
    source = obtain_data(data, download)
    with np.load(source, allow_pickle=False) as archive:
        if split == "canonical_test":
            # NpzFile is lazy: canonical replay never loads training members.
            images, labels = archive["x_test"], archive["y_test"]
        else:
            indices = development_indices(archive["y_train"])["validation"]
            # Only the exact development selection is returned or evaluated.
            images, labels = archive["x_train"][indices], archive["y_train"][indices]
    if images.shape != (10000, 28, 28) or images.dtype != np.uint8 or labels.shape != (10000,) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels > 9)):
        raise ValueError("Public MNIST selected input schema is invalid")
    images, labels = images[:limit], labels[:limit]
    batches = [(images[i:i + 128], labels[i:i + 128]) for i in range(0, limit, 128)]
    started = time.monotonic()
    if workers == 1:
        initialize_network(candidate)
        counts = list(map(batch_count, batches))
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_network, initargs=(candidate,)) as pool:
            counts = list(pool.map(batch_count, batches))
    correct, total = (sum(pair[i] for pair in counts) for i in (0, 1))
    seconds = time.monotonic() - started
    del images, labels, batches
    if total != limit or verification.verify_checkpoint(candidate) != before:
        raise ValueError("Replay count or frozen checkpoint integrity changed")
    if verification.sha256(source) != SOURCE_SHA256 or verification.sha256(Path(__file__)) != entry_hash or verification.sha256(ROOT / "verify.py") != verifier_hash:
        raise ValueError("Public replay source or operational files changed during execution")
    matched = expected is None or correct == expected
    return {
        "schema": "mnist-neural-direct-authoring-public-replay-v1",
        "status": "passed" if matched else "count_mismatch",
        "candidate": candidate, "split": split,
        "kind": "full_public_replay" if full else "partial_public_smoke",
        "correct": correct, "total": total, "accuracy": correct / total,
        "expected_correct": expected, "full_recorded_count_checked": full,
        "count_threshold": 9500, "meets_95_percent_on_this_replayed_split": bool(full and correct >= 9500),
        "new_blind_confirmation": False, "confirmation_ledger_accessed": False,
        "public_source": {"url": SOURCE_URL, "sha256": SOURCE_SHA256,
                          "members_loaded": ["x_test", "y_test"] if split == "canonical_test" else ["x_train", "y_train"],
                          "selected_split": split},
        "validation_index_sha256": VALIDATION_INDEX_SHA256 if split == "validation" else None,
        "artifact_hashes": before, "replay_source_sha256": entry_hash,
        "verification_source_sha256": verifier_hash, "source_revision": source_revision(),
        "batch_size": 128, "workers": workers, "numerical_threads_per_worker": 1,
        "seconds": seconds, "seconds_semantics": "wall time of complete inference, excluding download and integrity checks",
        "limitations": "This is inference on public, already exposed data. Replaying a count does not establish blind generalization, authorship chronology, or absence of pretraining exposure.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    split_parser = subparsers.add_parser("split", help="Recreate and verify work/validation indices from public training labels only")
    split_parser.add_argument("--data", type=Path, default=Path(".cache/mnist-public-source.npz"))
    split_parser.add_argument("--download", action="store_true")
    split_parser.add_argument("--output", type=Path)
    replay_parser = subparsers.add_parser("replay", help="Inference-only public replay")
    replay_parser.add_argument("--checkpoint", choices=tuple(verification.FROZEN), default="v087")
    replay_parser.add_argument("--split", choices=("canonical_test", "validation"), default="canonical_test")
    replay_parser.add_argument("--data", type=Path, default=Path(".cache/mnist-public-source.npz"))
    replay_parser.add_argument("--download", action="store_true")
    replay_parser.add_argument("--limit", type=int, default=10000)
    replay_parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    replay_parser.add_argument("--expect-correct", type=int)
    replay_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError("Preserve the existing replay or split result; choose a new output")
    if args.command == "split":
        result = split_record(args.data, args.download)
    else:
        result = replay(args.checkpoint, args.split, args.data, download=args.download,
                        limit=args.limit, workers=args.workers, expect_correct=args.expect_correct)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0 if result["status"] in ("passed", "verified") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(str(exc))
