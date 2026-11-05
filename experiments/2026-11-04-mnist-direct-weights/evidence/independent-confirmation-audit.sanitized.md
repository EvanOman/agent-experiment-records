# Independent post-confirmation audit: c026 passed

Reviewer: native OpenAI subagent `reviewer-1`.
Date: 2026-11-04.

The finalized primary confirmation report records **9,603 correct out of
10,000, or 96.03%**. The integer count exceeds the required minimum of 9,501.
Combined with the prior independent direct-authoring review, the measured
acceptance criteria are satisfied for this experiment's exact c026 bundle.

I independently audited the terminal report and exclusive reservation after
both were finalized at mode 0444. Their schemas, candidate/set identity,
chronology, strict acceptance threshold, integer counts, accuracy, environment,
and aggregate-only fields are consistent. The report binds the exact reservation,
and both bind the prior approval, frozen bundle, evaluator, and expected dataset
hash. The evaluator's recorded dataset hash agrees with both current and frozen
prepared manifest metadata. I did not reopen or independently deserialize the
confirmation dataset.

All 60 frozen files still match their recorded hashes; the exact file set,
read-only modes, and source/dependency closure are unchanged. The external
approval receipt, its final review note, and the current evaluator also retain
their approved hashes. Python, NumPy, and SciPy versions match the frozen
environment. No model or final-evaluator execution was performed during this
post-run audit.

| Artifact | SHA-256 |
|---|---|
| Frozen bundle manifest | `847e4cf9f44ad5b5d1fbf3e1f143bec53eef6eb78c2cf74954ffde00b1e66328` |
| Independent approval | `df0365537adbcdadcff57ed3e25ae37f40676cc8689481571f78c41deb514c52` |
| Final evaluator | `1705c1243630371a7a52a715d0bb3ffdcdda9c882a887fad7dfbf642974440f4` |
| Primary reservation | `dbd712ec9235bb4b79fdffd080cfeddbb7c38f26e3c0c0e606bc874ff1f82b3f` |
| Primary confirmation report | `66347e17e73f1e3ec3523ec53417cc887d21bff854513abc2a20abb033fc3426` |
| Independent post-confirmation receipt | `ba105e023aa23809edc32068f4ef863afbf62453adc159d34ec6d6ec59cc29dc` |

The receipt is
`protocol/reviews/c026_primary_post_confirmation.json`; its reproducible
metadata-only checker is `notes/verify_c026_confirmation.py`. The checker reads
frozen artifacts, source, JSON metadata, reservation and aggregate report only.
It imports neither NumPy nor the model/evaluator and never reads confirmation
image bytes. Its exact source hash is recorded in the receipt.

This primary set is permanently consumed. No repeat evaluation is authorized
or needed to substantiate this result. The reserve has no protocol reservation
or report and remains unused by this protocol. The audited evidence supports
goal completion; no unresolved integrity blocker remains.
