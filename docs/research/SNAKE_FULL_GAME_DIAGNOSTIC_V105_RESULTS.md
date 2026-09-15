# V105 complete: decoder adaptation helps; additional V103 efficacy regresses

The separately registered three-condition development diagnostic and independent
audit completed. **The candidate incremental-learning progress gate failed.**
Exact V96 efficacy with its V104 adapted decoder is the strongest controller in
 this comparison. It remains learned with demonstrations and below the original
mean-score target. The V103 and V104 original offline failures remain unchanged.

| Completed original 16x16 games | Original V96 | Adapted V96 | Adapted V103 |
|---|---:|---:|---:|
| Games | 32 | 32 | 32 |
| Mean score | 4.40625 | 10.15625 | 7.75 |
| Median | 5.0 | 10.0 | 7.0 |
| Fraction scoring >=5 | 0.53125 | 0.8125 | 0.59375 |
| Decisions | 16235 | 6305 | 6552 |
| Full-state boundary parts | 273 | 117 | 117 |
| Dense full endpoints | 105 | 293 | 157 |
| Actual 10ms states | 140 | 320 | 320 |
| External seconds | 210.481369 | 83.780671 | 86.257511 |
| Peak RSS bytes | 2979561472 | 3526639616 | 3525722112 |
| Collision | 18 | 30 | 29 |
| No food for too long | 14 | 2 | 3 |

All paired bootstrap intervals use 10,000 pair resamples with seeds frozen
before play. These are game-seed diagnostic intervals for one continuing run
91001, not independent training-run uncertainty or final acceptance.

| Comparison | Mean gain | 95% interval | Mean >=1 and lower >0 |
|---|---:|---|---|
| Additional V103 internals, matched adapted heads | -2.40625 | [-4.75, -0.15625] | FAIL |
| V96 decoder adaptation | 5.75 | [3.46875, 8.09375] | PASS |
| Adapted V103 versus original V96 | 3.34375 | [0.96875, 5.90625] | PASS |

The registered primary criterion required both additional internal and total
change comparisons to pass. Only total change passed. Adapted V96 meets median10
and fraction>=5>=.8 descriptively, but mean10.15625<15. No arm is eligible for
final acceptance from this 32-game, single-continuation diagnostic.

## Every paired score

Seed =35184373138832 + zero-based index. All32seeds are now used development
seeds. No games were excluded, extended, substituted or artificially cut off.
None of these states may become fitting data or be called fresh again.

| Index | Original V96 | Adapted V96 | Adapted V103 |
|---|---:|---:|---:|
| 0 | 2 | 9 | 7 |
| 1 | 8 | 8 | 2 |
| 2 | 3 | 12 | 13 |
| 3 | 10 | 9 | 9 |
| 4 | 0 | 10 | 1 |
| 5 | 4 | 0 | 12 |
| 6 | 5 | 13 | 6 |
| 7 | 10 | 13 | 8 |
| 8 | 0 | 18 | 14 |
| 9 | 5 | 19 | 4 |
| 10 | 0 | 16 | 24 |
| 11 | 8 | 3 | 3 |
| 12 | 0 | 6 | 3 |
| 13 | 3 | 6 | 17 |
| 14 | 7 | 17 | 1 |
| 15 | 7 | 18 | 2 |
| 16 | 0 | 4 | 2 |
| 17 | 1 | 14 | 15 |
| 18 | 5 | 11 | 14 |
| 19 | 9 | 10 | 10 |
| 20 | 0 | 13 | 13 |
| 21 | 4 | 17 | 2 |
| 22 | 7 | 2 | 2 |
| 23 | 0 | 5 | 7 |
| 24 | 5 | 7 | 9 |
| 25 | 3 | 20 | 11 |
| 26 | 9 | 3 | 3 |
| 27 | 5 | 14 | 11 |
| 28 | 7 | 6 | 7 |
| 29 | 6 | 13 | 10 |
| 30 | 6 | 6 | 3 |
| 31 | 2 | 3 | 3 |

## Verified path, recording and cost

All arms used all165,122neurons/25,563,197directed pairs, original integer
counts/signs, V88 equations, fixed V84 image encoder and V69 sensory mapping.
Original V96 used its exact V69 head through V97; the two adapted arms used their
exact V104 heads. The nonlinear decoder has2,179parameters and reads only64
descending neural outputs. Input is rendered image history; persistent full
neural state advances ten10ms substeps per100ms decision. Actions are unfiltered
argmax. Encoder, interface and efficacy hashes were unchanged at the endpoint.
Trainer/teacher/loss imports were blocked. No optimizer update or teacher query
occurred during image qualification or gameplay. All original game rules apply.

Image qualification passed exact baseline-reference, adapted restore and
independent full-core checks:24full observations/128encoder executions in
4.796363external seconds, peak3,417,980,928bytes. Both adapted heads matched FP64
arithmetic; adapted V96 full states were bitwise identical to baseline on the
same images. Two CPU admission tests and the required625-test suite passed
(112optional skips),31.932483external seconds with unchanged bound sources.

Gameplay consumed29,092full observations/232,736encoder executions in
380.519552external seconds across three sequential completed invocations.
Including image qualification:29,116full observations/232,864encoder executions.
No recovery, repeated tail work, teacher queries, updates or final games.
All runs stayed within1,774internal/1,834external seconds and48,000aggregate
decisions per arm. Sparse recordings retain actual images, views, inferred
sensors, drive, logits, actions and64outputs each move; full-state boundaries
at most64moves apart; early dense endpoints and actual10ms samples as tabulated.
Unrecorded full states remain absent. No spike timestamps or interpolation.

Independent CPU audit replayed all29,092game transitions/images/RNG and checked
every frozen condition-specific head vector in FP64, rtol1e-4/atol3e-5, maximum
error1.26315556903e-06. It took12.642902body/
12.797471external seconds, peak798,081,024bytes, with zero neural work.

## Post-hoc failure diagnosis

A separately frozen CPU replay inspected all29,092already-used transitions,
using unchanged V97 cycle analysis and its original qualified fixtures.
It took24.867694body/24.989148external seconds, peak758,693,888bytes.
No neural/encoder/update/teacher/new-game work occurred. Legality is diagnostic
geometry only; it is absent from inference.

| Recorded failure evidence | Original V96 | Adapted V96 | Adapted V103 |
|---|---:|---:|---:|
| Collisions with a legal alternative | 16/18 | 26/30 | 28/29 |
| Trapped at collision (no legal action) | 2 | 4 | 1 |
| Fatal chosen obstacle inferred below .5 | 0 | 0 | 0 |
| Fatal left/straight/right choices | 3/8/7 | 13/17/0 | 22/6/1 |
| Timeouts with terminal board/action cycle | 14/14 | 2/2 | 3/3 |
| Food-offset MAE on visited states | .021596609 | .019330567 | .017468036 |

Adapted V96 fatal chosen-obstacle estimates were at least .52774012; margins
over its best legal action ranged .02581313 to3.75313783 where a legal action
existed. Thus many collisions persist despite perceiving the blocked direction.
The four trapped states also require earlier planning, not merely a different
final choice. These observations do not establish generally perfect perception.

Adapted V96 timeout periods were4and6 at scores0and2, suffixes1,006and998moves.
Adapted V103 timeout periods6,8,6 at scores1,3,1. Repeated recorded64outputs
differed at most2.982e-8 across the arms. This is evidence of board/action cycles
and near-identical outputs, not convergence of unrecorded full neural state.

## Selected next direction and preserved artifacts

Retain adapted V96 as the development starting point. Collect separately
registered fresh training-partition trajectories from that exact image-only
controller, then generate corrective teacher labels in a separate CPU process.
Inspect collision, trap and cycle coverage before allocating training. The
intended next learning comparison is jointly adapted internal efficacy and head
versus equally trained head-only control, rather than another efficacy-only
fit under a fixed head. New gradient/recovery qualification and a bounded plan
are required before that training. No new learning or games are allocated by
this report, and no V105 development state may enter its fitting inputs.

- Evaluation/audit binding:65fed7f9d2400b11a6b8e2d73613ee43429a38ba43f089424e93e621ac0ef272.
- Failure diagnostic binding:72c82554d47e3643789fdfd057c438e9fc3e6eb91ddd8248b510d336f4cc427b.
- Source policy:9f6419ed938acc187ca2138a9a39da936e58a80b542c1f9da96b515e6cdb48b4.
- V96 efficacy checkpoint:de381ff7e64e615489c34fe6925173dbd5d116386d45d1787e92c0c328feffe0.
- Adapted source head:b2d65045ecd21abc7a896ac6b12b13722ad857c2e17bd09376980050281f3b08.
- Stage evidence:data/snake-whole-v105. Actual NPZ recordings stay local.

The original goal remains active and incomplete. Three independent runs,500
fresh final games each, matched initial/disabled controls, temporal/intermediate/
surrounding-network causal evidence, exact selected-training recovery and an
honest completed viewer remain required. All1,500final seeds remain untouched.
No protected service, viewer, external system or old frozen report was changed.
