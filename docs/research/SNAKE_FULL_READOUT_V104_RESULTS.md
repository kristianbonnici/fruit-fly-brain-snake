# V104 completed: endpoint gates pass, one matched-comparison gate fails

Both small decoders passed all eight registered recorded-state endpoint gates.
The V103-efficacy decoder also improves over its original head and the equally
trained V96-efficacy decoder in held-prefix prediction loss. However, it makes
six held unsafe choices versus five for adapted V96. The additional safety
comparison therefore fails, and **V104's overall gameplay gate remains failed**.
Independent audit confirms this result. No V104 games were allocated.

This is **learned with demonstrations**. Each condition received the same
2,179-parameter V69 initial decoder,50epochs,9540fitting rows, batch128,
seed104001 permutations and fresh Adam. Only decoder parameters changed in
this stage. Complete165122-neuron/25563197-pair efficacy, image encoder and
sensory interface stayed fixed. Exact V103 CE.25/legal-set.25/KL.5 fitting
weights and V96 anchors were identical across conditions; no held labels fit.
Neither cached features nor legal flags are authorized runtime observations.

## Recorded endpoint comparison

| Metric | Adapted V96 | Adapted V103 |
|---|---:|---:|
| All48 CE reduction versus original V96 | .106361649238 | .121866221830 |
| 95% interval | [.070823235235,.146883349937] | [.079493462740,.170852597595] |
| New8 CE reduction versus original V96 | .273743826539 | .295298643033 |
| New8 interval | [.165398863361,.403445105261] | [.161413917072,.462748291851] |
| New source fatalities corrected | 5/6 | 5/6 |
| New held unsafe choices | 1; source6 | 2; source6 |
| Old held unsafe choices | 4; source6 | 4; source6 |
| Newly unsafe old held choices | 1 | 0 |
| Fitting unsafe choices | 11; source28 | 11; source28 |
| Newly unsafe fitting choices | 1 | 4 |
| Two held cycle agreement gains | +.502857,+.497297 | +.625714,+.497297 |
| Eight endpoint gates | All pass | All pass |

The V103 head's improvement over its own pre-adaptation head is .069605915347
CE,95% interval [.045978263123,.095350900898]. Its advantage over identically
adapted V96 is .015504572592,95% interval [.003407647347,.028474281297]. Fixed
bootstrap seeds104031/104032,10,000draws; endpoint seeds99031/99032 unchanged.
Five of the six additional aggregate gates pass. The unsafe-choice comparison
fails at6versus5. These48prefixes have been inspected during development;
they are not fresh games, independent training seeds, or final causal evidence.

The unsafe sets differ, not just their totals. Adapted V96 newly turns left at
old row7402, while adapted V103 retains source-fatal old row3608 and newly turns
left at new row11938. Both retain old rows3817/5344/5692 and new row14593. All
row identities and legal sets remain in the reports. No gate is retrospectively
relaxed, no alternative epoch selected, and the V104 seed block remains unused.

## Qualification, work and independent verification

The two synthetic CPU tests passed (mixed parameter finite differences,
zero-mass handling and exact complete Adam restoration). Original-source
qualification checked all29188cached head vectors, maximum absolute error
1.192092896e-6, then3discarded updates/384fitting exposures with exact loss,
parameter and moment restoration. Qualification .094098body /.204062external
seconds,106070016peak bytes; no new neural work.

Required suite passed623tests with112optional skips in30.305body seconds.
The frozen main completed7500decoder updates /954000fitting exposures in one
invocation, no recovery:8.859635body /8.956481external seconds, actual external
peak118865920bytes. Forecast26.553622seconds, limits120internal/150external.
No internal update, recurrent observation, encoder/teacher execution or game.
The earlier full-feature extraction cost remains separately recorded in
SNAKE_FULL_READOUT_V104_EXTRACTION_RESULTS.md (375.408950external seconds).

Independent audit reconstructed all29188adapted heads in FP64, maximum error
2.555445332e-6 within frozen rtol1e-4/atol3e-5. It rebuilt every episode/group
metric, all three objective terms, all gates and bootstrap intervals, identical
fitting permutations, complete checkpoints/moments/context, current/final
equality, all100epochs and actual process costs. Audit .237950body /
.312584external seconds,104628224peak bytes; zero updates/neural work.

Final weighted objectives (CE/KL/legal): adapted V96
.159353395287/.021599637379/.023610016645; adapted V103
.143522990252/.023844830023/.021673968050. These describe the fixed fitting
objective; they are not scores or proof of useful play.

- Qualification binding:68e81464a32112f104efb2d2faffceae5b588561fc8de538b7fc6ea909162412
- Training binding:b6a376698b311f91e88e2aa9140b7a1bd4f2984bc8c6184d3e79829245d45407
- Independent review:21b4f81ed396bcb521bbf8ff79af5ba5d1b877f8244ed7bcfce1d6149ca89ea6
- Source head:b2d65045ecd21abc7a896ac6b12b13722ad857c2e17bd09376980050281f3b08
- Learned head:94a6e7bd642f9bf11e0651eadda9042b97b73dcbfa3f362c469c0023cbf0e26c

All artifacts are in data/snake-whole-v104; immutable terminal heads are
decoder-source-final.npz and decoder-learned-final.npz. Do not relaunch fitting.
Reserved32development seeds35184373128832..863 are unused after3313priorJSON
files were scanned. The original1500final seeds remain unused.

The next selected question is actual play with the adapted heads, retaining
this explicitly failed comparative gate. A separately versioned post-hoc V105
diagnostic should compare original V96, adapted V96 and adapted V103 on fresh
paired games, after new exact image/head qualification and finite allocation.
This tests both adaptation and incremental internal efficacy. No V105 work is
allocated by this report; no existing V104 gate or final target is changed.
Three independent training runs,500finalgames each, all useful-play targets,
full causal evidence and an honest trained whole-network viewer remain incomplete.
