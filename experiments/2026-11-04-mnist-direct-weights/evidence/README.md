# Historical experiment evidence

The original one-use confirmation recorded **9,603/10,000 = 96.03%** on a fresh withheld split drawn from MNIST's training partition. The original acceptance threshold was 9,501 correct. This is separate from the repeatedly used validation partition and from any later public CI replay.

## Result and chronology

- [primary.report.json](primary.report.json): original aggregate result, unchanged bytes.
- [primary.reservation.json](primary.reservation.json): original one-use reservation, unchanged bytes and preceding the result.
- [approval.sanitized.json](approval.sanitized.json): the native OpenAI reviewer's approval of the exact frozen candidate before confirmation.
- [post-confirmation-review.sanitized.json](post-confirmation-review.sanitized.json): the recorded metadata-only audit after confirmation.
- [historical-freeze.json](historical-freeze.json): original manifest of 60 frozen files, unchanged bytes. It describes the historical bundle, not the smaller public package's current file set.

The reviewer's internal role label was replaced with the public alias `reviewer-1`. Hashes *inside* the historical result and reservation still refer to the **original** approval/review bytes. A sanitized receipt therefore has a different public hash. [publication-manifest.json](publication-manifest.json) records the original and public hash for each selected artifact and every transform. It does not rewrite the original attestation chain to pretend the sanitized bytes were present during the experiment.

Eight selected artifacts retain byte identity. Five selected review artifacts contain only the declared role-alias substitution. The original numerical model/weights are packaged separately under [model/](../model/); the public verification/replay checks apply to that package.

## Numerical construction and development

- [tensor-reproduction.json](tensor-reproduction.json): original data-free construction evidence and tensor hashes.
- [independent-tensor-reconstruction.sanitized.json](independent-tensor-reconstruction.sanitized.json): recorded independent reconstruction checks.
- [validation-order-invariance.json](validation-order-invariance.json): original full-validation order/read-only result.
- [development-history.json](development-history.json) and [CSV](development-history.csv): derived summaries of all 25 complete 5,000-image validation reports, plus the separately recorded incomplete initial candidate. These retain candidate IDs, counts, accuracy and original report hashes. Confusion matrices, per-image data, runtime details and other fields were omitted. No new evaluations were run to make this summary.
- [source-data-manifest.json](source-data-manifest.json) and [development-split.json](development-split.json): original source/split commitments, unchanged bytes. No image arrays are published here.
- [model-description.md](model-description.md): the original account of the fixed numerical graph, unchanged bytes.

The numerical authorship claim is about hand-authored stroke-coordinate/template parameters and shared coefficients, expanded through fixed data-free geometry/feature equations into saved tensors. It does not mean an agent independently typed every expanded tensor element. Repeated validation feedback informed revisions, so “no training” here means no recorded conventional gradient/optimizer/fitting process, not an absence of learning or model selection.

## Reviews and limits

[The final pre-confirmation audit](independent-final-audit.sanitized.md) and [post-confirmation audit](independent-confirmation-audit.sanitized.md) are historical native OpenAI reviewer documents. They are not external certifications. Stronger historical wording remains preserved as attributed evidence. Consult [TRACE_REVIEW.md](../TRACE_REVIEW.md) for the later read-only trace review and its qualification to the claimed timing of one bias-vector choice.

The original holdout boundary was procedural on a trusted shared host, not an OS-enforced isolation boundary. Recorded traces and hash checks cannot exclude all unobserved access, LLM pretraining exposure, or intentional actions that were never logged. A successful public replay can reproduce predictions and counts; it cannot independently prove the historical model-authoring process or restore the replayed data's unseen status.

Raw rollouts, hidden reasoning, account/machine configuration, datasets, sealed arrays, and per-image prediction/label records were not copied into this evidence directory.
