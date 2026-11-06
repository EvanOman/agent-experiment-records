v095 is frozen once at `candidates/v095` and ready for root-owned population evaluation. It implements the single prospectively specified open-head readout composition on the complete frozen v094. This worker performed no development scoring or sealed evaluation; no accuracy claim is made.

The candidate retains all 555 raw arrays, 2,471,932 stored entries and 1,509 independently authored numerical specifications. `weights.npz` and `runner.py` are byte-for-byte unchanged from v094. The graph has 125,406 nodes: 113,943 unchanged parent nodes, nine exact 1,236-node geometry copies, and 339 new composition nodes. All 302 feature columns and 140 parent outputs remain; 112 additional diagnostics bring the output count to 252. Only the original final-logit node is renamed before the specified residual is appended.

This is a new, development-guided architectural decision using existing numerical values. The per-view threshold median, orientation maximum, strict clean-hole-absence gate, branch-local eligibility and readout combination are newly authored uses of those constants. Unchanged raw numerical values do not imply unchanged decision logic or an absence of adaptation. The fixed construction was saved and hashed before materialization, and was not amended after executable feedback.

The new readout preserves all old body-to-waist/bowl, route, cup, orientation and sc80 internal computations. It adds body/head evidence through saved rows 121/187, attenuates the cup through row 154, accounts for the inherited Tnew-product seven restorations, and updates only the positive nine-orientation permission. Its correction has support only in classes 4, 7, 9. The final association and single addition to complete parent logits are explicit float32 operations, not a rebuilt parent head.

Validation is complete:

- Static verification passed 208 named source assertions, exact serialized parent recovery, all nine geometry inverses, all eight feature-array ties, all 555 inherited raw-tensor records, and topological/finite-source checks.
- Independent generic verification passed 617,042 checks. All 113,943 inherited intermediates matched v094 byte-for-byte on three nonglyph fields. Nine copied features matched 162 retained output arrays against the frozen standalone feature on six fixtures, including positive Q. The 58 abstract cases/522 branch records cover strict absence and local eligibility boundaries, zero-Q identity, body/route preservation, effective-head positive parts, subtracting old products once, qualified Tnew, positive-orientation behavior and simultaneous family changes. Distinguishing float32 association counterexamples passed.
- Native actual-stage review was clean. Its separate executable check passed 251,477 assertions, independently reconstructed all 339 composition nodes, tested 33 abstract cases, matched all 25 standalone feature outputs across nine copies on three nonglyph masks, and compared every inherited intermediate on two additional generic inputs. It read and bound the separate generic verification.
- Ordinary verification passed artifact integrity, finite inference, batch/single consistency and argmax checks. All staged hashes remained unchanged, and the one frozen copy exactly matches the reviewed stage.

No constructor, model, ordinary-verification, generic-verification or peer execution failed. Minor source-inspection incidents are preserved in the audit and review records: premature or guessed source filenames, an empty inventory glob, and overlarge/truncated source displays. The standalone geometry's earlier reference-shape test failure remains in its preexisting review; it was not erased or counted as a v095 model failure. No examples, actual cue arrays, feedback/result files, sealed or confirmation data were read for this construction or review. No outside model, web search, training, fitting or parameter search was used.

The decisive SHA-256 bindings are:

| Artifact | SHA-256 |
|---|---|
| Frozen candidate manifest | `c26d3c32203d876613fa7cb6c8c7b4a4a3b5952b82e147a794c30216c4c42155` |
| Graph | `be60dbddd9af426d45de0f1263456dfdc5ad1304850a03d78da63bcc702a4576` |
| Raw weights | `3056fcdf1f821db073d9d029aec89c2e7319dc91db4771251a0768150e271a5f` |
| Runner | `128299dfbdc1db4a535ebcde0d68231559e86b374b7354e4e47c81d0b4f1427a` |
| Author | `5330da91667cd2f940743e23fc433d7dbb3e859126fd517a7b8659ffa8420f67` |
| Prospective specification | `a1dfe30afcdce6eb7529f8ebb2ead5aa3e8f931e56e8f73b092882258709dbc2` |
| Generic verification JSON | `05f41498c1ee8d403a0a5c047954fa4c094ad44aea540b7c710a0a30a84ec362` |
| Ordinary verification JSON | `6fe38e74b757b2832f56213cf194e237a64bc7ee42b9557870a4c799066f9f18` |
| Native peer review JSON | `4cd3432a6ffe49e3ebc9b38a698e337aee6e7041c80ed143d1f89ee85cfc9f2f` |
| Native peer review prose | `abd6eabb018d666dae17602f1df03a561442b9ca26495798ffd927d88f11f089` |

The full source/readout audit is `contributions/open_nine_readout_source_audit.md/.json`, with consolidated binding `open_nine_readout_source_SHA256SUMS.json` SHA-256 `596010ee8b2dfc3f7f4ed9cfdcc68e8accccc04785f2b29f24d8c480398a5228`. The v095 READY record and contribution hash manifest bind the complete local construction and verification record.

Ordinary local verification can be reproduced with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify.py candidates/v095
```

Inference uses the unchanged generic entry point:

```sh
python candidates/v095/runner.py candidates/v095 input.npy predictions.npy
```

The ordered-walk geometry can retrace spurs and discard alternative seeds; pooling can spread a false witness into other locally eligible branches. Generic consistency and a clean incremental review establish implementation readiness, not accuracy or population specificity. They do not certify every historical access or exclude pretraining exposure. Root owns the next full-population measurement and any later confirmation decision.
