# Visible conversation export

[coordinator.md](coordinator.md) contains all **55 visible coordinator messages**: one user message and 54 assistant messages, with their original order and UTC timestamps. [coordinator.jsonl](coordinator.jsonl) contains the same sanitized text in a machine-readable form.

The first user message contains the expanded task contract. [The prompt document](../prompt.md) distinguishes the original short user request from that expanded assignment.

This is a conversation transcript, not the full execution record. Internal reasoning, system/developer context, injected environment blocks, tool calls/results, account metadata, and subagent conversations were excluded by the conversation-only source export. The later [trace review](../TRACE_REVIEW.md) discusses what the separately inspected recorded tool activity does and does not establish.

The publication transform replaced private workspace/interpreter paths and names, and omitted one paragraph of local machine-operating policy. It did not invent progress statements, scores, or messages. The exact categories and counts are in [manifest.json](manifest.json), together with the source visible-message hash and hashes of the two public renderings. The private source messages are not included, so the source hash is a provenance commitment rather than a self-contained reconstruction proof.

Historical coordinator statements remain attributed statements. For example, an agent's claim that the protocol was followed is evidence of what it reported; it is not external certification that no leak or unrecorded access could have occurred. Adaptive validation feedback is learning in the broad sense even though the recorded restrictions prohibit conventional gradient or optimizer training.

`export_visible.py` is the small allowlist renderer used for this publication. It expects a JSON list of visible `role`, `text`, and `timestamp` objects obtained through a conversation-only history export. It does not read account files or raw rollouts. New inputs still require task-scoped human/privacy review; its regular expressions are not a general secret-removal guarantee.

```bash
python experiments/2026-11-04-mnist-direct-weights/transcript/test_export_visible.py
```

No raw logs or private intermediates are required to read or verify the public export's checksums.
