# Prompt and task contract

## Original user request

The user supplied this request. Its wording is preserved, including its original spelling:

> Now do the same thing with a Astra Ultra new thread. Tell it to not use Claude Bridge, just use its own OpenAI subagents, and then set a goal to keep working until ever reached above 95% on MNIST through direct model weights, no training. But iterating on the weights and observing activations, etc., is acceptable.

The earlier instruction to use a holdout test or cross-validation also remained in force.

## Expanded task contract

The launcher expanded the request into the following task-scoped assignment before the coordinator began. This is a sanitized copy of the assignment document, separately labeled so it is not confused with the short user request or with hidden system/developer context. Local paths and names were replaced; local machine-operating policy was omitted.

This contract defines the intended process. It is not, by itself, evidence that every restriction was followed. The visible conversation is in [the transcript](transcript/coordinator.md), original experiment evidence is under [evidence/](evidence/README.md), and the later read-only review is in [TRACE_REVIEW.md](TRACE_REVIEW.md).

```text
# Astra Ultra: directly authored MNIST weights above 95%

the user explicitly requests a NEW GPT-6 Astra thread at Ultra reasoning. Use your own native OpenAI subagents. **Do not use Claude Bridge, Claude, Fable, Overmind, any external-model API, or a provider/billing fallback.** This explicit provider choice overrides any standing preference for a Claude contribution. Keep subscription authentication. Use native OpenAI subagents for independent architecture ideas, bounded weight authoring, and integrity review, with clear ownership and the same constraints. You are not alone in the workspace; preserve existing files and other agents' work.

The user explicitly requests a persistent goal: **keep working until you achieve strictly greater than 95.0% MNIST accuracy through model weights directly authored and iterated by you and your OpenAI subagents, with no conventional training.** Inspect the goal already attached to THIS thread; do not create a duplicate. If the launcher could not attach it, create this goal yourself immediately. No token budget was requested. Do not stop because of a fixed number of iterations, a plateau, an unpromising first architecture, or the end of a turn. Change your approach and keep working. Mark the goal complete only after the measured acceptance criteria below are met. Preserve progress across compaction. A concrete external blocker or an explicit user pause/cancel is different from weak model performance.

User clarification: iterating on the weights, running forward inference, observing intermediate activations, and using evaluation feedback are acceptable. The NUMERICAL WEIGHTS must be authored directly by OpenAI agents. No training script, SGD, Adam, backpropagation, gradient calculation, least squares, regression fitting, data-computed class centroids/prototypes, pretrained digit-classifier weights, automatic parameter fitting, evolutionary/coordinate/random weight optimization, or downloaded solution weights. Ordinary deterministic tensor serialization and expansion for explicit weight sharing are allowed only when their numerical parameters/patterns were directly authored by the agents and are independent of data. Scripts may evaluate weights; they must not invent or fit weight updates.

No web search. The launcher requests `web_search="disabled"`. Do not invoke web/search tools, browse websites, look up published MNIST weights/solutions, or work around this using shell HTTP, browsers, MCP tools, another agent, or another model. All necessary raw MNIST data is already local. No additional package or compute spending beyond standing authorization. If packages are needed, obey the seven-day release-age rule.

Choose a reasonable, sufficiently capable numerical model architecture yourself. You are not bound to the previous experiment's architecture or its stopping rule, and you must not import its weights. You may improve the architecture, add genuinely shared hidden features, inspect activations, reason about confusions, and author new numerical parameters. Do not disguise an arbitrary digit-recognition program, a dataset lookup table, or a sample cache as a layer or weights. The model must use the same fixed numerical graph and parameters for every input and consume only image pixels. No sample IDs, positions, labels, filenames, hashes, or dataset membership may enter inference. Fixed image-only preprocessing is acceptable; learned/fitted preprocessing is subject to the same no-training restriction.

## Workspace and data

Work only on this experiment in `<workspace>/`. The launcher owns `runtime/native_rpc.py` and launch receipts. You own experiment implementation, numerical candidate artifacts, logs, reports, and local tests. No repository or machine configuration changes are needed. There is no existing project Git branch to switch. Use a dedicated worktree if you later work in an existing repository. [Local machine-operating policy omitted.]

Prepared files:

- `data/development_raw.npz`: 50,000 labeled MNIST images, arrays `images` (N×28×28 uint8) and `labels`. Split this deterministically into a working training/examples partition and a validation partition. Use validation and training data for all exploration and activation/error inspection. Record the split and candidate history.
- `data/manifest.json`: exact source hash and a newly randomized stratified split. The development and sealed partitions are disjoint. This experiment uses a fresh split; do not import prior experiment reports or test feedback.
- `sealed/mnist_holdout.npz`: 10,000 held-out MNIST images and labels. **Do not open/read/preprocess these arrays, inspect examples, compute features/activations, run predictions, or reveal labels during model development.** Do not give this path or its contents to weight-authoring subagents. Build a clearly separated final evaluator that can consume it only after the architecture/preprocessing/weights are frozen and an independent native OpenAI reviewer has checked the protocol.

An existing compatible Python environment with NumPy 2.2.6 and SciPy 1.15.3 is `<python-environment>/bin/python`. Reusing that interpreter is allowed; do not read that project's model, scores, test data, responses, or weights. You can instead create your own eligible environment. Package installs and data splitting are not model training.

## Evidence and completion criteria

Save every evaluated candidate's exact numerical weights, authoring provenance, immutable hash, architecture/preprocessing version, validation measurements, and the reason for the next authored revision. Keep evaluators separate from weight writers. Test inference for order/batch invariance and ensure only pixels and frozen parameters affect outputs. Use a native OpenAI integrity reviewer to check that no numerical parameters are derived by a fitting routine, no test labels leak, and no programmatic optimizer or web access is used.

Keep working through architectures and numerical revisions until validation accuracy exceeds 95.0% convincingly. Do not spend the sealed confirmation evaluation on an obviously unqualified candidate. Freeze the candidate, graph/preprocessing code, and all numerical parameters by hash before the final holdout evaluation. Use a one-use test reservation and immutable report; give the author no holdout feedback for further weight tuning. A 10,000-image confirmation set requires **at least 9,501 correct predictions** to exceed 95.0%; 9,500 is insufficient.

Complete the goal only when an integrity-checked, directly authored numerical model has **strictly greater than 95.0% accuracy on a genuinely withheld MNIST confirmation set**, with reproducible artifacts and no prohibited training or cheating. If a final confirmation fails, report that honestly, retire that exposed set from unseen-test claims, and continue legitimate development; do not repeatedly tune on or query the same exposed holdout to manufacture success. Preserve a defensible fresh confirmation or a predeclared nested validation procedure for any subsequent claim.

Return periodic concise progress updates with measured validation accuracy, observed activation/confusion evidence, the next concrete experiment, and any real blocker. Keep the goal active while meaningful authorized work remains. Do not claim success from a development score, artifact existence, or a narrow passing test.
```
