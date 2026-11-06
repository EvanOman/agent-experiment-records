# What was authored

These artifacts are a **hard-threshold/morphological feed-forward neural graph with arithmetic and pooling**, not a fixed affine-ReLU convolutional network trained by an optimizer. The assignment allowed a finite feed-forward architecture and direct specification of values, connections and constants. The chosen representation makes the nonlinear geometric computations explicit inside the saved graph.

The generic runner loads `graph.json` and non-pickled float32 arrays from `weights.npz`, converts caller-supplied pixels to float32, executes the declared operations and returns the final ten logits. Prediction is argmax. Class-specific behavior is in the supplied graph and tensors; there is no separate procedural OCR classifier, template lookup, example bank, network call, hidden weight download or input-dependent parameter construction.

The supported operations include affine maps and convolutions; ReLU and inclusive step; addition, subtraction, multiplication and division; min/max pooling and cumulative maximum; reductions; and fixed slicing, padding, stacking and reshaping. Thresholded pixels, morphology and explicitly unrolled topology computations form the main representation. Hard gates and arithmetic mean the model is not restricted to a conventional differentiable CNN.

## Three released checkpoints

**v001** was the complete substantive first attempt, frozen before development examples, labels or feedback. Its 242-node graph uses authored morphology, directional gaps, regions, topology cues and a sparse class readout. It achieved 59.28% development validation, not a holdout score. It is retained to separate unaided authoring from later development-guided work.

**v087** has 37,443 nodes. It combines conservative shear handling, three tied threshold branches, skeleton/endpoint and cavity measurements, qualified contour/return cues, orientation filters and accumulated authored readout corrections. This is the checkpoint independently confirmed once on canonical MNIST test data: 94.61%, below the requested 95% threshold. Its own development validation was 95.05%.

**v095**, the best completed development-validation checkpoint at the snapshot cutoff, has 125,406 nodes. Its ancestry adds a fixed identity/+10°/−10° ensemble with shared classifiers, cleanup before those views, and an identity-anchored supported-two-hole consensus. Its new nine route copies one fixed connected upper-turn/lower-tail feature into each of the nine view/threshold branches. Threshold-median and orientation-maximum pooling feed explicit eligibility gates and signed readout residuals. The resulting validation score is 95.28%; v095 has no sealed-confirmation result in this snapshot.

The open-nine cue establishes an ordered connected walk. It can revisit pixels or retrace a spur, does not establish pen-stroke order or a simple path, and is not a proof of digit identity. The correction preserved old internal computations while adding readout support. Its completed validation tradeoff was 18 corrected nines versus 13 newly missed cases: 11 fours, one five and one eight. That feedback motivated later work, which is outside the selected numerical snapshot.

## Numerical accounting

| Checkpoint | Graph nodes | Raw arrays | Recorded independently authored numerical specifications | Expanded float32 entries |
|---|---:|---:|---:|---:|
| v001 | 242 | 39 | 240 | 4,932 |
| v087 | 37,443 | 552 | 1,506 | 1,242,619 |
| v095 | 125,406 | 555 | 1,509 | 2,471,932 |

The **1,506 versus 1,509 counts are an accounting convention, not a count of all design choices**. They exclude architecture, connections, indexing, routing, masks' structural placement and other design decisions. Tied constants are counted under their recorded sharing convention; explicitly assigned readout entries may count independently even when equal. These figures are not counts of distinct floating-point values or a measure of total authoring effort.

The increase of three is exactly two numerical specifications assigned to the rotation resampler plus one cleanup-fraction specification. Fixed copying, sharing and analytical expansion produce many stored entries from those specifications. Most of the later expanded entries belong to dense fixed resampling matrices. **The expanded arrays were not all independently typed.** Their numerical bytes are nevertheless saved completely and do not need to be constructed from examples at inference.

The independent [checkpoint inventory](evidence/checkpoint_inventory.md) checks raw-array shapes, finite float32 values, expanded totals, graph dependencies, specification sums and original hashes. The complete saved provenance dictionaries remain inside each checkpoint.

## Adaptation and reproducibility

After the first attempt, the author inspected permitted development behavior and explicitly revised numerical values and architecture. No optimizer, regression fit, gradient update or automated parameter search was used in the inspected workflow. **No optimizer is a narrower claim than no learning:** repeated feedback-guided authoring is adaptation and can overfit development data.

Some historical construction scripts execute fixed diagnostic checks after materializing specified values. They are retained as provenance, not invoked by prediction. Reconstructing a late checkpoint through its historical authoring script can require ancestor files not included in this release. Portable inference needs only the released graph, weights, runner, pinned numerical dependencies and input pixels.

Exact numerical files and runners retain their original bytes. Public manifests bind the released files; sanitization of review/history text is separately recorded. Generic smoke tests check execution and integrity, public replays check fixed predictions, and historical blind confirmation is a distinct event. None proves absence of unrecorded historical access or public-MNIST exposure in model pretraining.
