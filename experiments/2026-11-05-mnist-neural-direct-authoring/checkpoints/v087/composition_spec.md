# Root-fixed v087 composition, declared before materialization

Root explicitly selects one new graph: COMPLETE frozen v086 logits plus EXACT
frozen v085 `cp85_correction` ONCE. Root supplied source85 validation9491 versus
parent79 validation9490, one corrected prediction and no lost correct predictions,
with changes confined to class3 logits on140 rows. The v086 validation run was
still pending when root selected this composition. These supplied observations
motivate the single choice; this worker reads no results, examples or labels.

Copy complete frozen v086, renaming only the final `logits` producer to
`co87_parent_logits`. Preserve all earlier order, especially complete79 calibration,
the filled-lower residual, then the source-seven correction with its association
`(terminal_projection + body9_projection) + body21_projection`. Copy only the
live v085 closure rooted at `cp85_correction`, stopping at exactly matching parent
node declarations and byte-identical tensors. Do not import v085 complete logits
or `cp85_parent_logits`; do not duplicate any residual. Append one final add:

```
logits = co87_parent_logits + cp85_correction
```

Retain the source closed-three computation exactly, in each of center/low/high:

```
permission = one - os73_<branch>_Qoutside
permitted_raw = <branch>v19_closed_three * permission
new = maximum(<branch>at_qualified_closed_three, permitted_raw)
delta = new - <branch>at_qualified_closed_three
weighted = delta * saved_branch_mix_weight
delta_mix = weighted_center + (weighted_low + weighted_high)
cp85_correction = reshape(delta_mix,[-1,1]) @ saved_row196 + zero
```

The actual old feature at row196 must remain the current qualified closed-three
feature. Saved row196 is `[0,0,0,4,0,0,0,0,0,0]`; center and side weights remain
the existing .5 and .25. Each maximum precedes mixing. Preserve the exact branch
mixture association, affine zero bias and final addition to COMPLETE v086.

Expected live closure20 nodes plus one final add gives37,443 total nodes.
All302 original feature columns remain. Copy all552 raw arrays, entire weights
archive and runner byte-for-byte from v086 (also identical to v079/v085).
Counts remain1506 independently authored specifications and1,242,619 expanded
entries. Zero new numbers, arrays, biases or tensor entries; no v081/v082,
new rotation or other changes.

Before freezing, check complete-parent byte-exact inverse, all common geometry
ancestors and tensor bytes, exact source closure, original parent and source
intermediates on generic fields, simultaneous nonzero parent class3, lower8/9,
seven and new class3 residuals, class3-only added delta and source association.
Retain generic closed-three floor/raw0/Q0/Q1/fractional/max-before-mixture cases.
Run ordinary generic verification and bounded native Astra Ultra actual-artifact
review. Preserve failures and source. Freeze unchanged once; root handles scoring.

This is direct authoring with development feedback, without training/fitting or
parameter search. The source gain may overlap prior gains; increased class3 can
create regressions. No accuracy or sealed-success claim follows from composition.
