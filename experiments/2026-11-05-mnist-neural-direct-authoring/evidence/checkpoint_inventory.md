# Frozen checkpoint inventory

This audit checks the publication snapshot frozen at **2026-11-06 05:55:49 UTC**. It reads only frozen graph/provenance metadata, raw non-pickled weight tensors, aggregate result records and the earlier review binding. It runs no classifier or public replay and reads no prepared dataset, forward-output archive, unused confirmation content or earlier public experiment.

| Checkpoint | Role | Graph nodes | Raw arrays | Recorded numerical specifications | Expanded entries |
|---|---|---:|---:|---:|---:|
| v001 | Substantive unaided first attempt | 242 | 39 | 240 | 4,932 |
| v087 | Historical canonical-test candidate | 37,443 | 552 | 1,506 | 1,242,619 |
| v095 | Best completed development validation at cutoff | 125,406 | 555 | 1,509 | 2,471,932 |

Every frozen manifest entry matches its source bytes. All raw arrays are finite float32; their shape products sum to the recorded expanded-entry totals. Each specification dictionary sums to its recorded numerical-specification count. Graph names, dependencies, output references and operation counts were inspected directly. The hash-bound JSON contains the exact graph, weight, runner, original manifest and original provenance digests.

**1,506 versus 1,509 is an authoring accounting convention, not a count of all design choices.** The recorded increase is exactly two specifications assigned to the positive rotation resampler and one cleanup-fraction specification. Architecture, connectivity, indexing and structural choices are excluded. Weight sharing, analytic expansion and copying create many stored entries from relatively few assigned numerical specifications. The expanded arrays were not all independently typed, and these counts do not measure total design effort or the number of distinct float values.

The graph operation inventory confirms a hard-threshold/morphological feed-forward neural graph with arithmetic, pooling, reductions and affine operations. It is **not a fixed affine-ReLU CNN**. Later revisions use development feedback: no optimizer or fitting routine is a narrower statement than no learning or adaptation.

| Checkpoint | Historical split | Correct / total | Accuracy |
|---|---|---:|---:|
| v001 | Development validation | 5,928 / 10,000 | 59.28% |
| v001 | Development work | 18,087 / 30,000 | 60.29% |
| v087 | Development validation | 9,505 / 10,000 | 95.05% |
| v087 | Development work | 28,392 / 30,000 | 94.64% |
| v087 | Canonical-test confirmation | **9,461 / 10,000** | **94.61%; threshold failed** |
| v095 | Development validation | **9,528 / 10,000** | **95.28%** |
| v095 | Development work | Pending at snapshot cutoff | Not claimed |

The development totals, confusion-matrix sums/traces, accuracy arithmetic and exact model hash mappings agree. The v087 confirmation's candidate manifest and clean native-review hash agree with the frozen records. Its 9,461 correct answers are below the required 9,500; that historical confirmation set is consumed. v095's validation score cannot be relabeled as holdout success. A later inference-only public replay of v087 may reproduce 9,461 but is not a fresh blind confirmation.

The original unaided chronology and absence of unobserved access cannot be proved by this metadata audit, and public MNIST exposure in model pretraining cannot be excluded. The publication snapshot does not establish completion of the original goal.

The machine-readable evidence is [checkpoint_inventory.json](checkpoint_inventory.json).
