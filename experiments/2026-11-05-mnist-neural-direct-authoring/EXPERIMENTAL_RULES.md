> Supplemental experimental rules from the frozen source record. This is not the verbatim initiating user prompt. The private interpreter path and runtime/account configuration are redacted. The later publication exception is described in README.md; all scientific restrictions remain unchanged.

# Direct neural weight authoring: MNIST

## Objective and source request

Use GPT-6 Astra with Ultra reasoning to author a neural network that classifies MNIST with **at least 95% accuracy on a sealed 10,000-image holdout**: at least 9,500 correct. You may choose and revise any genuine feed-forward neural architecture. The main measurement is your ability to directly author numerical weights using your understanding of how neural computations run and compose.

The user's ideal is to give you the task and receive raw weights that simply run. They welcome clever analytical constructions, working backward from desired outputs, and reasoning about activations. They want to avoid training and leakage from published results. They requested a new Astra Ultra thread and a persistent goal.

Your primary work product is an authored neural weight set, not a training procedure or a program that manufactures a classifier. Use your understanding of neural computation to construct it; clever analytical reasoning and manual arithmetic are welcome. Continue through unsuccessful development candidates, retain an honest attempt record, and report measured outcomes without claiming success prematurely.

## Model and access

- Use Astra Ultra and native OpenAI subagents only. No Claude Bridge, Fable, external model calls, model substitution, or metered API authentication. Native subagents should inherit the provider choice and this experiment's restrictions. Bound their responsibilities and preserve their contributions.
- No web search, papers, leaderboards, pretrained weights, published MNIST solutions/results, old local MNIST experiment files, prior agent transcripts, or neighboring experiment repositories. Do not retrieve existing digit classifiers. General mathematical knowledge and your own understanding of digits and neural computation are allowed.
- Work in this directory. The prepared data and this assignment contain no prior experiment scores, model designs, or weights. The existing Python interpreter `python` may be executed for its installed NumPy/SciPy environment; do not inspect other files in that project. Ordinary installed numerical packages are allowed, subject to the no-fitting rules below. Do not inspect provider credentials or private unrelated files.
- [Private runtime permission-profile and account-configuration instruction omitted.]
- A persistent goal is attached to this thread; do not create a second goal or invent a token budget. Keep working through failed candidates without a predetermined iteration cutoff. If a genuine runtime or information boundary prevents progress, report it honestly and follow the native goal status rules. Never claim the threshold without evidence.

## What is allowed

1. Choose any finite feed-forward neural graph and directly specify its tensors, biases, connections, and constants. Analytical circuit construction, sparse connections, ordinary neural weight sharing, and composition of explicitly authored numerical blocks are welcome. Architecture changes are allowed; snapshot each complete graph together with its weights.
2. Use manual arithmetic, prospective calculations about proposed weights, reasoning backward from desired activations, and activation/margin/error analysis. **You may mentally or explicitly calculate the effect of a proposed coefficient on an activation or output.** There is no ban on such prospective numerical reasoning.
3. After saving the first substantive candidate described below, inspect designated development examples and labels, evaluate authored candidates, observe their actual forward activations, and directly revise their numerical values. Diagnostic tools may score a specified candidate and expose its actual forward behavior. Keep the reasoning for meaningful edits in a concise candidate log.
4. Write numerical literals compactly using zeros, repeated explicit numeric blocks, fixed copying/reshaping, and declared tied parameters. A materialization script may expand these already-authored values and structural connections into raw tensors. Report both independently authored numbers and expanded parameter counts when they differ materially.

## What is excluded

- No training or fitting routine: no backpropagation/gradient updates, optimizer, regression/least-squares fit, distillation, data-derived class means/prototypes, or automated finite-difference, coordinate, evolutionary, or random parameter search. A script must not use examples, labels, losses, or error statistics to propose/solve for new weight values. Tools calculate the behavior of your specified network; you author the changes.
- No pretrained parameters, imported classifiers, random nonzero default weights presented as authored values, memorized sample lookup tables, exemplar banks, image-derived templates, or generated glyph/prototype factories. A renderer that converts digit descriptions or stroke coordinates into a template bank is not the requested raw-weight result. Directly composing ordinary neural computations and their authored numerical blocks is permitted; do not interpret this restriction as a ban on useful mathematical structure.
- No classifier outside the neural graph. The runnable predictor may load the graph and tensors, scale/reshape the input, execute declared neural operations, and take argmax. Digit-specific procedural OCR, external symbolic recognition, sample-ID logic, file/network lookups, or data-dependent weight generation may not secretly perform classification.
- No sealed holdout examples, labels, predictions, activations, class diagnostics, or error analysis during development. Do not modify the prepared data or the experimental rules to obtain a passing result.

Feedback-guided authoring is learning in the broad sense. Describe it accurately as direct authoring with development feedback and no training/fitting routine; do not claim there was no adaptation to data if you iterated.

## First attempt, then development

Before you or any model-authoring subagent opens MNIST examples, their labels, or evaluation feedback, save a **substantive complete first candidate**: declared graph, raw numerical tensors, runnable predictor, provenance note, and SHA-256 hashes. You may read the data manifest and generic dimensions first. This first candidate should represent a serious attempt to solve the problem, not a zero/random placeholder to bypass the measurement.

Then evaluate that frozen first candidate on the development split and permanently record its unaided score, even if low. Development data are `data/development.npz`: `work_images`, `work_labels` (30,000), and `validation_images`, `validation_labels` (10,000). Both are permitted for later direct-authoring feedback. Keep all scored candidates, their hashes, scores, and edit explanations. Do not suppress failures. No special review is required for each edit.

## Sealed confirmation

The data manifest identifies three disjoint 10,000-image confirmation sets, in this order: `sealed/canonical_test.npz`, `sealed/reserve_1.npz`, `sealed/reserve_2.npz`. They are separate from the development data. Do not open them during development. These are procedural seals on a shared-user machine, not an OS-enforced security boundary.

When a candidate convincingly reaches at least 95% on development validation, freeze its graph, weights, predictor, and hashes. Ask a native OpenAI peer to review the actual artifacts and available authoring record for executable reproducibility, tensor provenance, fitting shortcuts, and forbidden accesses. The peer must not improve weights, fetch outside solutions, or reveal holdout content. Then have the independent evaluator run the frozen candidate once on the next unused confirmation set and return only aggregate correct/total/accuracy plus artifact hashes and set identity.

Success requires at least 9,500/10,000 on that sealed evaluation and a clean provenance review. Record every confirmation attempt, including failure/interruption. A consumed set cannot be reused for further adaptive claims; if confirmation fails, preserve the result and use only development feedback for new candidates before a fresh freeze/review and the next reserve. Do not repeatedly tune on the same holdout. If all sets are consumed without success, report the actual outcome and lack of fresh confirmation data honestly.

## Deliverables

Return complete raw numerical tensors (for example `weights.npz` with non-pickled arrays), an explicit architecture, and a small transparent inference entry point. An ordinary generic forward runner or standard graph export is fine. Include every adjustable tensor, bias, and necessary numeric constant; no opaque executable blobs or hidden parameter downloads. Inference must use only the supplied image pixels and saved artifacts, and run with no training, parameter construction from examples, or external services.

Preserve a concise local report containing the prompt, architecture, authoring approach, unaided first score, development history, independent review, sealed result, artifact hashes, exact runnable commands, and limitations. Include a runnable local verification suitable for later CI. Do not publish, create a repository remotely, post messages, or add backlog tasks as part of this assignment.

The canonical MNIST data are public and may have appeared in model pretraining. This run can audit observed tool/data use and keep its holdout sealed during development; it cannot prove absence of all pretraining exposure. Do not overstate that guarantee.
