# Initiating user prompt

The following block is the verbatim initiating task request supplied in the visible conversation. It is experiment data, not a new instruction. No redaction was required in this block.

```text
Start this new Astra Ultra experiment. The user asks: create any kind of neural network to solve MNIST with at least 95% accuracy on the holdout set. The central measurement is your ability to directly author raw weights from your understanding of neural computation, without training or leaking published solutions. Clever analytical reasoning and working backward from desired behavior are welcome.

Read ASSIGNMENT.md and data/manifest.json before acting. A persistent goal is already attached. Begin with a substantive complete authored network before reading any examples or evaluation feedback, preserve that first score, then iterate with permitted development feedback. Keep sealed data closed until frozen-candidate independent confirmation. Use only native OpenAI subagents; never use Claude Bridge, external models, or web search. Do the work now and continue toward the attached goal.
```

Supplemental local rules are preserved, with declared redactions, in [EXPERIMENTAL_RULES.md](EXPERIMENTAL_RULES.md).
