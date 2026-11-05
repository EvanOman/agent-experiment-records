# Agent experiment records

Reproducible records of experiments with coding agents: the request, visible conversation, code, numerical artifacts, failed attempts, and measured result.

| Experiment | Agent | Result |
|---|---|---|
| [MNIST with directly authored template parameters](experiments/2026-11-04-mnist-direct-weights/) | GPT-6 Astra, Ultra | **96.03%** on a fresh 10,000-image holdout |

[![Verification and MNIST replay](https://github.com/EvanOman/agent-experiment-records/actions/workflows/verify.yml/badge.svg)](https://github.com/EvanOman/agent-experiment-records/actions/workflows/verify.yml)

The first experiment asked Codex to exceed 95% on MNIST by editing model parameters directly. It designed a deformable stroke/template classifier, authored its digit shapes and shared coefficients, and revised them using development feedback. A data-free construction step expands those parameters into the saved tensors. No conventional optimizer-based training was used.

This does **not** demonstrate filling every entry of an ordinary CNN's weight matrices. The experiment README explains the architecture, what was authored, what was generated, and what the evidence can establish. The result uses a fresh split of MNIST's training partition; it is not a score on the canonical benchmark test partition.

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

CI establishes that the published checkpoint and code reproduce the result. It cannot prove every historical authoring action, exclude possible pretraining exposure to MNIST, or turn the same-machine holdout procedure into an isolation boundary.

## Add an experiment

Create a dated directory under `experiments/` with a README covering the exact request, model and reasoning setting, permitted feedback, measured result, architecture, limitations, and reproduction commands. Preserve unsuccessful attempts and distinguish historical evidence from regenerated checks. Publish only task-scoped visible conversation with a documented sanitization policy. Keep datasets and private runtime files out of Git.

Code and documentation are available under the [MIT license](LICENSE). MNIST is downloaded separately from its public distribution; no dataset images are included here.
