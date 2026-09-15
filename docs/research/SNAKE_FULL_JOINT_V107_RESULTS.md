# V107 final comparison — completed, not adopted

The bounded V107 joint internal-efficacy/action-readout fit and matched fixed-internal control completed. Independent endpoint auditing passed, but the joint candidate failed two preregistered safety gates. No fresh V107 gameplay evaluation was authorized by those gates and none was run. The existing adapted V96 controller was selected for the final showcase.

The user changed the objective to finish the existing demo and stop. All autonomous training, evaluation and minimum replay capture stopped at **2026-09-15 07:43:42 UTC**, before the fixed 08:21 UTC cutoff. The three independent fits, 1,500 final games and broad causal acceptance program remain deferred. This report closes the bounded comparison; it does not qualify the original research objective.

## What learned

V107 used the unchanged full graph of 165,122 neurons and 25,563,197 measured directed pairs. Exact discrete truncated backpropagation jointly updated positive internal efficacy and the existing 2,179-parameter action readout. The image encoder and sensory interface stayed fixed. Training used previously recorded demonstrations, including training-only teacher and legality targets. There were no new teacher queries or game executions in this fit.

The final fit changed 14,718,078 eligible internal pairs and all 2,179 readout parameters. Logical work was 104,672 full-network slots and 2,051 updates to each optimizer. Actual execution, including exact recovery replay, was **104,896 slots and 2,058 updates to each optimizer**, with 102,581 active observations. Seven replayed chunks matched their original predictions, losses and both optimizer diagnostics exactly.

## Held-out recorded-prefix results

Positive cross-entropy reduction means improved prediction of the recorded teacher target; it is not a gameplay score. Intervals are paired bootstrap 95% intervals across recorded episodes, not uncertainty across independent training seeds.

| Comparison | Mean reduction | 95% interval |
| --- | ---: | --- |
| Joint versus source, all held-out prefixes | 0.038142 | [0.001266, 0.074508] |
| Joint versus source, new held-out prefixes | 0.070617 | [0.015637, 0.141498] |
| Joint versus matched fixed-internal control, all held-out prefixes | 0.040400 | [0.020309, 0.063403] |

Nine of eleven registered joint gates passed. The two failures were:

- Correct at least four new held-out fatal choices: **2 corrected**.
- Leave at most three avoidable unsafe choices in the new held-out set: **5 remained** (source 6; two corrected and one newly unsafe).

Old held-out unsafe choices fell from five to four with no new unsafe choices. Fitting-set unsafe choices fell from 32 to 18, with 17 corrected and three newly unsafe. These diagnostics do not justify overriding the failed game-admission gate. Conditional V107 game tools and reserved seeds were left unused.

## Actual cost and recovery

| Phase | External elapsed seconds | Scope |
| --- | ---: | --- |
| Joint fit, all three invocations | 2,569.091 | 42.82 minutes; pauses excluded |
| Matched fixed-internal control | 184.416 | Cached source outputs; zero recurrent observations or internal updates |
| Independent endpoint audit | 7.617 | CPU record, identity and numerical checks |
| Required test suite | 32.883 | 642 tests, 112 skips, successful exit |
| Final selected replay capture | 6.328 | 377 existing decision endpoints; zero new games or updates |

The joint worker reported 3.453 GB peak RSS; the external process measurements peaked at 3.483 GB. A first invocation stopped before learning because eight direct-logit components exceeded the original source tolerance. All actions agreed. That failure remains preserved; a separately qualified component check verified the full outputs and independent FP64 readout without changing the network or training schedule. A second invocation stopped at the disk guard. The final invocation restored the frozen checkpoint and replayed seven chunks exactly. No fourth invocation was made.

The independent audit verifies recorded endpoint coverage, source/cache alignment, original failures, both optimizer clocks, objectives and final gate counts. It does not claim to have independently rerun every training update. See `data/snake-whole-v107/review.json` and the preserved process reports.

## Frozen identities

- Main contract: `ef1d9dde5d29ef4c4eb106c790bc77942695bfe24fc2196447dc66887ce41030`.
- Source-check extension: `893887e89de90b35fee5745a5d09b27293217d4d956c13937a2045df0e2ef813`.
- Resource-recovery extension: `1ff226a6bc9cb4d858e5a3175ffbc86d423f1e7ea38469e795a41e01719220d9`.
- Matched-control contract: `d7cc4f994ca6ab29de76d4880f99974d8ed143ac784ab4b03dcfbfae8da3466e`.
- Independent audit: `ee8c1374432e3a95a877a97e530412f1254730768c0e81f0f0ee50303356c14a`.
- Unselected V107 final checkpoint: `73972223258e2753840f780e556778c7f4cc4f37fdd74f8c8a591870917a8300` at `data/snake-whole-v107/runs/joint-107001/final-checkpoint.npz`.

The selected adapted V96 controller, exact replay provenance and final release are documented in `SNAKE_SHOWCASE_FINAL_HANDOFF.md`. No further experiment is queued.
