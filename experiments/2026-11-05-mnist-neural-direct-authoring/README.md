# Directly authored MNIST weights: a measured development snapshot

**The best completed development validation result is 95.28%, but the requested 95% holdout result has not been achieved.** This publication freezes the experiment at **2026-11-06 05:55:49 UTC**, before the publication team accessed the target repository. The original goal remains active; publication does not claim its completion.

| Checkpoint | What it represents | Development validation | Development work | Historical canonical-test confirmation |
|---|---|---:|---:|---:|
| [v001](checkpoints/v001/) | Substantive unaided first network, frozen before examples or feedback | **5,928/10,000 (59.28%)** | 18,087/30,000 (60.29%) | Not evaluated |
| [v087](checkpoints/v087/) | The checkpoint actually submitted for one-use confirmation | 9,505/10,000 (95.05%) | 28,392/30,000 (94.64%) | **9,461/10,000 (94.61%): failed the 9,500 threshold** |
| [v095](checkpoints/v095/) | Best completed development-validation checkpoint at cutoff | **9,528/10,000 (95.28%)** | Running at cutoff; no score claimed | **Not evaluated** |

These scores belong to different checkpoints and splits. **v095's 95.28% is not a holdout score; v087's 94.61% is not v095's score.** v087's canonical set is consumed. No unused reserve data or fresh confirmation was accessed for publication. Later work and newly authored but unscored v096 are outside the selected numerical snapshot.

Hosted CI reproduced **v087 at 9,461/10,000** from the checksum-verified public archive. The saved proof identifies the tested commit and preserves the exact aggregate bytes. [Hosted replay evidence](evidence/publication/hosted_ci_42179538672/README.md)

## Request and method

The [verbatim initiating prompt](PROMPT.md) asks Astra Ultra to directly author raw neural weights for MNIST, starting with a substantive complete network before seeing examples or feedback. [Supplemental experimental rules](EXPERIMENTAL_RULES.md) retain the source protocol with declared privacy redactions.

The author supplied a finite **hard-threshold/morphological feed-forward neural graph with arithmetic and pooling**, not a fixed affine-ReLU CNN. The generic runner executes saved affine maps, convolutions, hard gates, morphology, reductions and structural tensor operations. Geometric behavior is encoded in the graph and raw tensors; there is no separate procedural OCR classifier or template bank outside it. The [architecture explanation](ARCHITECTURE.md) describes the three released checkpoints and their limitations.

After the unaided first attempt, the author used development examples, errors, activations and manual prospective reasoning to revise values and structure. No optimizer, fitting routine or automated parameter search was used in the inspected workflow. **That is a narrower claim than “no learning”: feedback-guided authoring is adaptation and can overfit development data.** The original first score is permanently retained rather than replaced by the later result.

## What the numerical counts mean

| Checkpoint | Graph nodes | Raw arrays | Recorded independently authored numerical specifications | Expanded entries |
|---|---:|---:|---:|---:|
| v001 | 242 | 39 | 240 | 4,932 |
| v087 | 37,443 | 552 | **1,506** | 1,242,619 |
| v095 | 125,406 | 555 | **1,509** | 2,471,932 |

The 1,506 and 1,509 figures use the recorded sharing/accounting convention and **exclude architecture, connections, indexing and other design choices**. The increase of three comprises two rotation specifications and one cleanup-fraction specification. Copying, tying and analytical expansion produce many stored entries; **the expanded arrays were not all independently typed**. These counts are neither the number of distinct float values nor a measure of total design effort. The [independent inventory](evidence/checkpoint_inventory.md) verifies actual counts and exact file hashes.

## Files and evidence

- Each `checkpoints/vNNN/` contains the exact original raw `weights.npz`, explicit `graph.json`, transparent `runner.py`, original manifest and historical construction/provenance files. `manifest.json` is the public integrity wrapper. Numerical artifacts and runner bytes have not been rewritten.
- [All 108 completed development results](history/scores.json), also available as [CSV](history/scores.csv), cover **92 scored candidates**. The [96-row candidate inventory](history/candidates.json), [full archived log](history/CANDIDATE_LOG.md) and [failure summary](history/FAILURES.md) preserve declines, ties, failed construction/verification, interruptions and unscored entries.
- The [original v087 canonical aggregate](evidence/confirmation/v087_result.json) and [attempt ledger](evidence/confirmation/attempts.jsonl) preserve its failed historical confirmation. The [native final review](evidence/reviews/final_native_review_v087.md) was clean, but a clean review did not make that score pass.
- [v095's incremental review](evidence/reviews/open_nine_head_v095_peer_review.md), [broader provenance preparation](evidence/reviews/confirmation_review_preparation_05_report.md) and [observer-boundary review](evidence/reviews/confirmation_review_observation_boundary_05_report.md) record their scopes. These are point-in-time records; later aggregate results do not silently rewrite their earlier pending statements.
- The [sanitized visible transcript excerpt](transcript/VISIBLE_TRANSCRIPT.md) has an explicit [coverage and omission record](transcript/omissions.json). It is not a complete conversation or a private reasoning trace.
- [Snapshot status](snapshot.json), [source/public hash mapping](PUBLICATION_MANIFEST.json) and [payload redactions and omissions](PAYLOAD_REDACTIONS.json) distinguish original bytes from sanitized text derivatives. Source/report hashes inside historical reviews remain original bindings; redacted public text has its own public hash.

No dataset, per-example forward-output archive, private host instructions, account/credential material, internal reasoning or unused reserve content is bundled. The original construction scripts are historical evidence and may require omitted ancestor files to rebuild a checkpoint. **They are not inference dependencies.** Prediction needs only the chosen graph, tensors, runner, numerical packages and caller-supplied pixels.

## Local verification and prediction

Run these commands from the repository root with Python 3.12. The original numerical environment used NumPy 2.2.6 and SciPy 1.15.3; [requirements.txt](requirements.txt) pins them. No package-age restriction bypass is required.

```sh
E=experiments/2026-11-05-mnist-neural-direct-authoring
python -m pip install -r "$E/requirements.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$E/verify.py" verify
python "$E/verify.py" smoke --checkpoint v095 --output replay-output/direct-smoke.json
```

The smoke run uses generic nonglyph inputs. It verifies execution and integrity; it does **not** measure MNIST accuracy. The shared repository gates are `python -m unittest discover -s tests -v` and `python scripts/verify_repository.py`.

For your own input, save raw uint8 pixels shaped `(N,28,28)` in a non-pickled NumPy file. The graph performs its own scaling:

```sh
python "$E/checkpoints/v095/runner.py" "$E/checkpoints/v095" input_images.npy predictions.npy
```

The output is a NumPy array of predicted digit indices. Inference does not train, construct weights from examples or contact external services.

## Public reproduction is not blind confirmation

The replay utility downloads the public MNIST archive only when explicitly requested and checks its exact SHA-256 before use. Its [public data description](data.json) records the source digest and original work/validation index hashes. The original development split recipe was recovered and verified **after the publication cutoff**; that reproduction evidence is distinct from the frozen authoring results. Split verification loads public training labels to reproduce the declared stratified indices and emits only their hashes and counts, not images or unused-reserve membership.

```sh
python "$E/replay.py" split --download --output replay-output/direct-splits.json
```

Reproduce the **already-consumed v087 canonical-test result**, requiring exactly **9,461/10,000**:

```sh
python "$E/replay.py" replay --checkpoint v087 --split canonical_test --download --workers 4 --expect-correct 9461 --output replay-output/direct-v087-canonical.json
```

This is inference-only replay of public canonical test images. It cannot be called a fresh blind confirmation, and it does not authorize any unused reserve evaluation. The required count is enforced, not accepted as a substitute for inference.

An optional full replay of the best development-validation checkpoint is separately available:

```sh
python "$E/replay.py" replay --checkpoint v095 --split validation --download --workers 4 --expect-correct 9528 --output replay-output/direct-v095-validation.json
```

**The full v095 public replay is pending unless a separately dated execution record establishes its completion.** Smoke tests and verified split indices are not that full replay. Later local replay and hosted CI outcomes belong in separately dated reproduction/publication evidence; they do not alter the historical snapshot cutoff or establish absent authorship leakage. Earlier unsuccessful split-reconstruction checks, if any, remain in that reproduction record rather than being erased.

## Limits and publication authorization

The observed authoring record and native reviews support narrower claims about the inspected source, tensors and accesses. They cannot prove absolute first-observation chronology, exclude all unrecorded access, or rule out public MNIST appearing in model pretraining. The seals were procedural on a shared machine, not an OS-enforced security boundary. Hundreds of development mistakes remain, and the graph is large rather than latency-optimized.

The user subsequently authorized a concrete public repository snapshot, overriding only the assignment's earlier no-publication rule for this release. This did not authorize social posts, correspondence, fresh holdout consumption or a 95% holdout-success claim. Publication workers were isolated from ongoing model authors and did not inspect prior public experiment folders. The ongoing original goal remains separate from this publication.
