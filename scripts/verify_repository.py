"""Check the public record's reported counts, local links and publication hygiene."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/2026-11-04-mnist-direct-weights"
DIRECT_EXPERIMENT = ROOT / "experiments/2026-11-05-mnist-neural-direct-authoring"
TEXT_SUFFIXES = {".md", ".py", ".json", ".jsonl", ".txt", ".yml", ".yaml", ".toml", ".svg"}
PRIVATE_MARKERS = {
    "private home path": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "private socket": re.compile(r"/run/user/\d+/"),
    "private mesh host": re.compile(r"[a-z0-9.-]+\.ts\.net", re.I),
    "GitHub credential": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "provider credential": re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}"),
    "AWS access key": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def public_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    return sorted({ROOT / s.decode() for s in completed.stdout.split(b"\0") if s})


def verify_direct_publication_manifests(files: list[Path]) -> dict:
    def verified_entry(relative: str, digest: str) -> str:
        if (not isinstance(relative, str) or not relative or "\\" in relative or
                Path(relative).is_absolute() or ".." in Path(relative).parts or
                "." in Path(relative).parts or Path(relative).as_posix() != relative):
            raise ValueError("Unsafe direct-authoring public manifest path")
        path = DIRECT_EXPERIMENT / relative
        if (not path.is_file() or path.is_symlink() or
                path.resolve() != DIRECT_EXPERIMENT.resolve() / relative):
            raise ValueError("Direct-authoring manifest entry is not an ordinary contained file")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid public manifest digest")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Direct-authoring publication hash mismatch: {relative}")
        return relative

    publication = json.loads((DIRECT_EXPERIMENT / "PUBLICATION_MANIFEST.json").read_text())
    seen = set()
    for record in publication["files"]:
        relative = verified_entry(record["public_path"], record["public_sha256"])
        if relative in seen:
            raise ValueError("Duplicate entry in direct-authoring publication manifest")
        seen.add(relative)
    whole_path = DIRECT_EXPERIMENT / "SHA256SUMS.json"
    pairs = json.loads(whole_path.read_text(), object_pairs_hook=lambda value: value)
    if not isinstance(pairs, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs):
        raise ValueError("Whole-experiment checksum manifest must be a flat path/digest mapping")
    whole = set()
    for relative, digest in pairs:
        relative = verified_entry(relative, digest)
        if relative == "SHA256SUMS.json" or relative in whole:
            raise ValueError("Duplicate or circular whole-experiment checksum entry")
        whole.add(relative)
    published = {path.relative_to(DIRECT_EXPERIMENT).as_posix() for path in files
                 if path.is_relative_to(DIRECT_EXPERIMENT) and path != whole_path}
    if whole != published:
        raise ValueError("Whole-experiment checksums do not exactly cover all published experiment files")
    return {"publication_manifest_entries": len(seen), "whole_experiment_files": len(whole)}


def verify() -> dict:
    required = [ROOT / "README.md", ROOT / "LICENSE", EXPERIMENT / "README.md",
                EXPERIMENT / "prompt.md", EXPERIMENT / "transcript/coordinator.md",
                EXPERIMENT / "transcript/README.md", EXPERIMENT / "evidence/primary.report.json",
                EXPERIMENT / "model/weights.npz", EXPERIMENT / "reproduce.py"]
    for path in required:
        assert path.is_file(), f"Missing public artifact: {path.relative_to(ROOT)}"
    result = json.loads((EXPERIMENT / "evidence/primary.report.json").read_text())
    assert result["n"] == 10000 and result["correct"] == 9603
    assert result["accuracy"] == result["correct"] / result["n"] == 0.9603
    assert result["status"] == "passed"
    direct_spec = importlib.util.spec_from_file_location("direct_authoring_publication_checks", DIRECT_EXPERIMENT / "verify.py")
    direct_module = importlib.util.module_from_spec(direct_spec)
    direct_spec.loader.exec_module(direct_module)
    direct_result = direct_module.validate_snapshot()
    files = public_files()
    direct_manifests = verify_direct_publication_manifests(files)
    for path in files:
        assert path.is_file() and not path.is_symlink(), f"Unsafe public file: {path.relative_to(ROOT)}"
        assert path.stat().st_size < 90 * 1024 * 1024, f"Oversized Git artifact: {path.relative_to(ROOT)}"
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore"}:
            continue
        text = path.read_text()
        for name, pattern in PRIVATE_MARKERS.items():
            match = pattern.search(text)
            assert match is None, f"Publication check: {name} in {path.relative_to(ROOT)} (matched content withheld)"
    for path in (ROOT / "README.md", EXPERIMENT / "README.md", DIRECT_EXPERIMENT / "README.md"):
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if re.match(r"[a-z]+://", target) or target.startswith("#"):
                continue
            local = target.split("#", 1)[0]
            assert (path.parent / local).exists(), f"Broken local link in {path.relative_to(ROOT)}: {target}"
    return {"publication_files_checked": len(files), "reported_correct": result["correct"],
            "reported_n": result["n"], "reported_accuracy": result["accuracy"],
            "direct_authoring_snapshot": direct_result,
            "direct_authoring_publication_manifests": direct_manifests,
            "limitations": "Pattern scan and artifact checks are not a complete privacy or historical leakage proof."}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2))
    except (AssertionError, OSError, ValueError) as exc:
        sys.exit(str(exc))
