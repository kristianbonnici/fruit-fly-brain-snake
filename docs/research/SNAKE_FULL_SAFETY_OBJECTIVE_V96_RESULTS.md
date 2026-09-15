# V96: targeted safety gain, failed offline advancement

V96 completed full internal learning and passed independent numerical/data/recording verification. It passes the new-correction prediction and legality gates, but fails the overall cross-entropy and old-stratum retention gates. These failures remain failures. The original full goal is incomplete, and no V96 live games or final evaluation have been run.

## Change and qualification

The complete 165,122-neuron / 25,563,197-pair graph, measured counts, V87 physiology, V84 image encoder and V69 sensory/action interfaces are unchanged. Only 14,737,539 eligible positive internal efficacies learn. Start at exact V90 final efficacy with fresh Adam; this is an objective variant of run 91001, not an independent training replicate.

The unchanged 7,875-row V95 mixture has 56 fitting prefixes (4,191 rows) and 40 held-out prefixes (3,684 rows). Training assigns 0.21875 mass to 21 recorded source-fatal fitting examples, 0.28125 to other corrective examples, and 0.5 to KL(source || current), temperature 1, on the original fitting examples. Source targets come from the independently verified V95 initial checkpoint, which uses V90 efficacy. All held-out training weights and distillation targets are zero. Loss labels and weights are separate from image-derived neural inputs; no inference filter is introduced.

CPU preparation took 0.361550 internal / 0.464810 external seconds. Native qualification used 28 fixed-head vector computations, with independent analytic and finite-difference checks using 6,156 CPU head-vector computations. Maximum native gradient error was 3.1428e-6; CPU finite-difference error 3.3036e-9. The source KL loss and native feature gradient were zero at the source outputs. Qualification took 0.687444 internal / 0.822524 external seconds and made no full-neural observations, encoder executions or parameter updates.

The required suite passed: 593 tests, 112 skips, 33.011 test-body / 34.786255 external seconds. Main initial replay reproduced 6,077 old predictions bit for bit and all 1,798 live-source actions; live-logit maximum difference was 2.7999281883e-5 under the registered tolerance. The independent audit also confirmed all 7,875 initial predictions equal V95 initial predictions exactly.

## Full learning and held-out results

Main: five epochs, 687 updates, 37,984 computed full slots, 36,705 active exposures, one invocation and no reserve work. Duration 887.528759 internal / 889.219161 external seconds (14.82 minutes); peak RSS 2,902,867,968 bytes. No encoder executions, teacher queries, policy games or final games. Source interfaces stayed fixed. Efficacy changed on 14,718,108 eligible pairs relative to V90, with range [0.4834156654, 4.0000000152]; the upper value is exp(FP32(log 4)).

| Held-out stratum | Prefixes | Initial CE | Final CE | Reduction |
|---|---:|---:|---:|---:|
| All | 40 | 0.660420414 | 0.640154658 | 0.020265756 |
| Original expert | 8 | 0.361385582 | 0.399289936 | -0.037904354 |
| Original learner | 8 | 0.737014394 | 0.745214204 | -0.008199809 |
| Earlier corrections | 8 | 0.687048752 | 0.678923735 | 0.008125017 |
| V73 corrections | 8 | 0.585906925 | 0.559052175 | 0.026854750 |
| V94 full-controller corrections | 8 | 0.930746418 | 0.818293240 | 0.112453178 |

All 40 prefixes: gain 0.020265756, paired 95% CI [−0.005646646, 0.048205136], failing the minimum 0.05 and positive lower bound. New 8 prefixes: gain 0.112453178, CI [0.055872265, 0.181326616], passing minimum 0.1 and positive lower bound. Both use 10,000 paired draws and seeds 95031/95032. Original expert CE worsens 0.037904354, exceeding the 0.02 retention limit.

On the 468 held-out V94 states, illegal actions fall from 8 to 2 and 6 of 8 original fatal choices are corrected. Both legality gates pass, with no newly illegal held-out action. The remaining source-fatal choices are prefix 7 / move 118 (right) and prefix 11 / move 34 (left). These are recorded-state decisions, not V96 game scores.

Overall held-out left/straight/right recalls are 0.407166 / 0.922263 / 0.566964; on the new corrections they are 0.188889 / 0.944223 / 0.472441. Every episode and confusion matrix is saved in the phase reports. The fixed-checkpoint fitting objective falls from 0.502852689 (CE 0.502852689, KL 0) to 0.366126679 (CE 0.348149741, KL 0.017976938).

## Independent audit and post-hoc diagnosis

The CPU audit passed, confirming every frozen gate result. It independently reconstructed weights and source targets, checked all unchanged inputs/schedules, every phase checkpoint/context/mask, current=final equality, every recorded full-node endpoint and every native loss component. It reconstructed 267 fixed-head vectors in NumPy FP64 with maximum error 1.0009964853e-6. Audit duration 5.312281 internal / 5.813587 external seconds; peak RSS 2,249,441,280 bytes; no new neural, encoder, update, teacher or game work.

A separately frozen post-hoc CPU diagnostic replayed all 6,077 old image/body histories from move zero, with original tail-moving collision rules. Two targeted tests passed, including comparisons with the original Snake engine and image-history tamper rejection. The diagnostic took 0.492234 internal / 0.579615 external seconds, peak RSS 391,151,616 bytes, with no new neural or teacher execution.

On 3,216 old held-out states, illegal choices fall from V90’s 15 to V96’s 4, with no newly illegal choice. Every one of V96’s 227 changed actions is immediately legal. In the expert stratum that failed CE retention, illegal choices fall from 9 to 2, while teacher agreement falls from 928/1,024 to 920/1,024; all 53 changed actions are legal. Thus reduced teacher agreement is not the same as increased immediate collision risk. This diagnostic does not alter the failed V96 gate, prove long-term safety, or measure live-game performance.

This new evidence supports a separately registered image-only gameplay diagnostic of the frozen V96 candidate. Its purpose is to measure whether the recorded-state safety gain transfers to actual play on unused development seeds. Keep the V96 offline failure explicit and retain the original three independent 500-game final acceptance requirements.

## Frozen provenance

Training contract: 720ecb4169bf150337ccfbab300c22156e89e9bde2244882b558808ce2c2c7f5.
Audit contract: 4700b49592ecd5b1d18f464bbd34520854039fb2163832dff418c882c32aa633.
Old-legality diagnostic: c022126d1da4f83b8b039c9c1eaac70118bf791b6ec7fe94d0d552553f4e9453.
Final checkpoint SHA256: de381ff7e64e615489c34fe6925173dbd5d116386d45d1787e92c0c328feffe0.
Final log-efficacy array hash: 02541c3b26cd9f7dc2493187d87a151fe40e97c012c84eb05ac73c1579095735.

Sustained mean training chunk: 0.903523332 seconds per 32 slots; prediction: 0.349811628 seconds. These measurements use cached frozen image features and full internal forward/adjoint computation. Training and review commands are in work/v96. Completed runs must not be relaunched. Protected services and existing viewers remain unchanged.
