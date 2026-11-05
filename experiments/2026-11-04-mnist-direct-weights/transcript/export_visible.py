#!/usr/bin/env python3
"""Render a reviewed, task-scoped visible-message export for publication.

Input is a JSON list of role/text/timestamp objects, such as a conversation-only
session-history export. This is an allowlist renderer, not a raw-log exporter or
a general guarantee that arbitrary new transcripts are safe to publish.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize(text: str) -> tuple[str, Counter]:
    changes: Counter = Counter()

    def replace(pattern: str, replacement: str, category: str, flags=0):
        nonlocal text
        text, count = re.subn(pattern, replacement, text, flags=flags)
        changes[category] += count

    replace(r"<(environment_context|INSTRUCTIONS|system|developer)>.*?</\1>",
            "[Injected context omitted.]", "injected_context_blocks", re.S)
    replace(r"The standard OMachine draft-only rule.*?externally\.",
            "[Local machine-operating policy omitted.]", "local_operating_policy", re.S)
    replace(r"/home/[^/\s]+/dev/mnist-astra-direct-weights/?", "<workspace>/",
            "workspace_paths")
    replace(r"/home/[^/\s]+/dev/mnist-direct-weights/\.venv312", "<python-environment>",
            "interpreter_paths")
    replace(r"/(?:home|Users)/[^/\s]+", "<home>", "other_home_paths")
    replace(r"/run/user/\d+/[^\s)\]`]+", "<local-runtime>", "local_runtime_paths")
    replace(r"/root/integrity_continued\b", "reviewer-1", "reviewer_aliases")
    replace(r"/root/[^\s`)'\"]+", "<agent>", "other_agent_aliases")
    replace(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            "<thread-id>", "internal_thread_ids", re.I)
    replace(r"\b[a-z0-9-]+\.(?:[a-z0-9-]+\.)*ts\.net\b", "<private-host>",
            "private_hostnames", re.I)
    replace(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})\b",
            "<redacted-secret>", "secret_shaped_values")
    replace(r"\bEvan(?:'s)?\b", "the user", "personal_name")
    replace(r"\bOMachine\b", "<local-host>", "machine_name")
    return text, +changes


def export(rows: list[dict]) -> tuple[list[dict], dict]:
    messages = []
    changes: Counter = Counter()
    dropped: Counter = Counter()
    for row in rows:
        role = row.get("role")
        if role not in ("user", "assistant"):
            dropped[str(role)] += 1
            continue
        if not isinstance(row.get("text"), str):
            dropped["non_text_message"] += 1
            continue
        text, redactions = sanitize(row["text"])
        changes.update(redactions)
        messages.append({"message": len(messages) + 1, "role": role,
                         "timestamp": row.get("timestamp"), "text": text})
    metadata = {
        "schema": "visible-conversation-publication-v1",
        "source_scope": "Coordinator visible user and assistant messages only; conversation-only session-history export",
        "source_messages_sha256": digest(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                                    separators=(",", ":")).encode()),
        "source_messages": len(rows), "published_messages": len(messages),
        "role_counts": dict(Counter(m["role"] for m in messages)),
        "dropped_input_roles": dict(dropped), "redactions": dict(changes),
        "omitted_from_source_export": ["internal reasoning", "system/developer messages",
                                       "injected context", "tool calls and tool outputs",
                                       "subagent conversations", "usage and account metadata"],
        "claims": "These are visible progress messages, not a complete execution trace or proof of the historical authoring process.",
    }
    return messages, metadata


def render_markdown(messages: list[dict]) -> str:
    lines = ["# Sanitized coordinator transcript", "",
             "This is the complete visible coordinator conversation: one user message and 54 assistant messages, in their original order with original UTC timestamps. Private paths/names and local operating details were replaced or omitted. No statements or scores were invented.", "",
             "Internal reasoning, system/developer context, tool calls/results, account metadata, and subagent conversations are excluded. The first message includes the expanded task contract that was sent as user-visible input. The original short request and the separately sanitized assignment are in [prompt.md](../prompt.md).", "",
             "This transcript records what the coordinator said. Inspect the frozen numerical artifacts, evidence, code, and public replay separately; the transcript alone does not prove that a process restriction was followed.", "",
             "Machine-readable version: [coordinator.jsonl](coordinator.jsonl). Redaction and omission record: [manifest.json](manifest.json).", ""]
    for message in messages:
        lines += [f"## {message['message']:02d} · {message['role']} · {message['timestamp']}", "",
                  message["text"].rstrip(), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Conversation-only JSON message list")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.source.read_text())
    if not isinstance(rows, list):
        raise ValueError("Expected a list of visible message objects")
    messages, metadata = export(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    jsonl = "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in messages)
    markdown = render_markdown(messages)
    (args.output / "coordinator.jsonl").write_text(jsonl)
    (args.output / "coordinator.md").write_text(markdown)
    metadata["public_sha256"] = {"coordinator.jsonl": digest(jsonl.encode()),
                                "coordinator.md": digest(markdown.encode())}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
