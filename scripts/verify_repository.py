"""Check the public record's reported counts, local links and publication hygiene."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/2026-11-04-mnist-direct-weights"
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
    files = public_files()
    for path in files:
        assert path.is_file() and not path.is_symlink(), f"Unsafe public file: {path.relative_to(ROOT)}"
        assert path.stat().st_size < 90 * 1024 * 1024, f"Oversized Git artifact: {path.relative_to(ROOT)}"
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore"}:
            continue
        text = path.read_text()
        for name, pattern in PRIVATE_MARKERS.items():
            match = pattern.search(text)
            assert match is None, f"Publication check: {name} in {path.relative_to(ROOT)} (matched content withheld)"
    for path in (ROOT / "README.md", EXPERIMENT / "README.md"):
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if re.match(r"[a-z]+://", target) or target.startswith("#"):
                continue
            local = target.split("#", 1)[0]
            assert (path.parent / local).exists(), f"Broken local link in {path.relative_to(ROOT)}: {target}"
    return {"publication_files_checked": len(files), "reported_correct": result["correct"],
            "reported_n": result["n"], "reported_accuracy": result["accuracy"],
            "limitations": "Pattern scan and artifact checks are not a complete privacy or historical leakage proof."}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2))
    except (AssertionError, OSError, ValueError) as exc:
        sys.exit(str(exc))
