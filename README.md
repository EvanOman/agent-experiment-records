# Agent experiment records

Reproducible records of experiments with coding agents: the request, visible conversation, code, numerical artifacts, failed attempts, and measured result.

| Experiment | Agent | Result |
|---|---|---|
| [MNIST with directly authored template parameters](experiments/2026-11-04-mnist-direct-weights/) | GPT-6 Astra, Ultra | **96.03%** on a fresh 10,000-image holdout |
| [MNIST with a directly authored neural graph](experiments/2026-11-05-mnist-neural-direct-authoring/) | GPT-6 Astra, Ultra | **95.28% development validation** (v095); **94.61% canonical test** (v087), below the holdout goal |

[![Verification and MNIST replay](https://github.com/EvanOman/agent-experiment-records/actions/workflows/verify.yml/badge.svg)](https://github.com/EvanOman/agent-experiment-records/actions/workflows/verify.yml)

The first experiment asked Codex to exceed 95% on MNIST by editing model parameters directly. It designed a deformable stroke/template classifier, authored its digit shapes and shared coefficients, and revised them using development feedback. A data-free construction step expands those parameters into the saved tensors. No conventional optimizer-based training was used.

This does **not** demonstrate filling every entry of an ordinary CNN's weight matrices. The experiment README explains the architecture, what was authored, what was generated, and what the evidence can establish. The result uses a fresh split of MNIST's training partition; it is not a score on the canonical benchmark test partition.

## Direct neural graph snapshot

The [November 5 experiment](experiments/2026-11-05-mnist-neural-direct-authoring/) preserves a substantive unaided first network at **59.28% validation**, then direct authoring with development feedback. Its best completed validation checkpoint at the publication cutoff is **v095: 9,528/10,000 (95.28%)**. The separately confirmed **v087 scored 9,461/10,000 (94.61%) on the canonical test set**, so this snapshot does not establish the requested 95% holdout result. The ongoing experiment has not consumed another holdout for publication.

This is a hard-threshold/morphological neural graph using arithmetic and pooling, not a fixed affine-ReLU CNN. Its 1,509 independently specified numerical choices do not count architecture decisions or imply that every expanded array entry was typed independently. Development-guided authoring is adaptation; an absence of optimizer-based training is a narrower claim than an absence of learning. Raw checkpoints, score history including regressions, aggregate evidence, and a sanitized visible transcript are included with explicit scope and omissions.

The [experiment's reproduction instructions](experiments/2026-11-05-mnist-neural-direct-authoring/#local-verification-and-prediction) distinguish generic smoke checks, validation-split reconstruction and inference-only public replay of v087's already consumed canonical test. That replay must reproduce **9,461 correct** and is not fresh blind confirmation.

## Reproduce

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_repository.py
python -m unittest discover -s tests -v
python experiments/2026-11-04-mnist-direct-weights/reproduce.py verify
python experiments/2026-11-04-mnist-direct-weights/reproduce.py smoke
```

For the full 10,000-image replay, including a checksum-verified download of the public source dataset:

```bash
python experiments/2026-11-04-mnist-direct-weights/reproduce.py replay \
  --download --output replay-output/mnist.json --expect-correct 9603
```

The replay performs inference only and can take tens of minutes on a CPU. It reconstructs the recorded split and reuses the frozen model. It is a public reproduction of a consumed holdout, not another blind confirmation experiment.

## What CI checks

The [workflow](.github/workflows/verify.yml) checks artifact hashes, data-free tensor reconstruction, synthetic inference, prediction invariance, transcript-export tests, and publication hygiene. A separate job downloads the public dataset, reconstructs the 10,000-image holdout, and requires exactly **9,603 correct predictions**. Its aggregate JSON result is uploaded as a workflow artifact.

The [recorded hosted run](https://github.com/EvanOman/agent-experiment-records/actions/runs/41966207607) passed both jobs and reproduced **96.03% (9,603/10,000)**. The [saved CI result](ci-proofs/mnist-c026-replay.json) and [run/commit binding](ci-proofs/mnist-c026-run.json) preserve its aggregate evidence. Full inference took about 8 minutes 36 seconds on that runner.

CI establishes that the published checkpoint and code reproduce the result. It cannot prove every historical authoring action, exclude possible pretraining exposure to MNIST, or turn the same-machine holdout procedure into an isolation boundary.

## Add an experiment

Create a dated directory under `experiments/` with a README covering the exact request, model and reasoning setting, permitted feedback, measured result, architecture, limitations, and reproduction commands. Preserve unsuccessful attempts and distinguish historical evidence from regenerated checks. Publish only task-scoped visible conversation with a documented sanitization policy. Keep datasets and private runtime files out of Git.

Code and documentation are available under the [MIT license](LICENSE). MNIST is downloaded separately from its public distribution; no dataset images are included here.
