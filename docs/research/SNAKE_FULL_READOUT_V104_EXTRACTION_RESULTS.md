# V104 matched full-feature extraction verified; decoder fitting pending

Both exact V96 and V103 efficacy conditions replayed all14,594 original V103
input rows through the complete165,122-neuron/25,563,197-pair model. All original
encoder/sensory/decoder parameters and internal efficacies stayed fixed.
These actual64-output pre/post-state caches are for training only. No cache
lookup enters inference; no image encoder, teacher, game or update executed.

Required pre-extraction suite:621tests/112optional skips passed,30.116body /
31.206682external seconds. Native qualification computed96full slots: first32,
next32, restore and repeat next32. All64unique source predictions and complete
restored states/features matched bitwise. Qualification2.918378body /
3.852243external seconds, peak2,913,566,720bytes; save.043777/restore.040324seconds.

Main extraction completed29,760full slots /29,188active observations in one
invocation, no recovery or repeats. Actual374.515089body /375.408950external
seconds (6.26minutes), peak2,930,868,224bytes. Forecast363.815547seconds,
limits788internal/848external,250MB storage reserve above8GBfree floor.

The separate independent CPU audit passed. All14,594source predictions match
V103 initial (exact V96) and all14,594learned predictions match V103 final,
bitwise. Complete batch-four final neural states also match those endpoints.
Every extraction chunk prediction hash, including inactive slots, matches the
corresponding original V103 prediction-phase chunk. Current and immutable
cache endpoints agree in every array and metadata field; chronology, move-zero
reset, preceding-state copies, finite bounds, identities and cost checks pass.

Independent FP64 V69 decoder reconstruction checked all29,188vectors. Maximum
absolute error1.454284853e-6source /1.765919407e-6learned, within registered
rtol1e-4/atol3e-5. Output-state RMS .0005270218214source /.0005541567537learned;
this is activity description, not causal proof. Audit1.604151body /1.753224external
seconds, peak799,326,208bytes, zero neural work.

- Qualification binding: e5a3a55addcddb0c88b545f6f91c1b171056470886415f1e558afde31b7cb1cb
- Main extraction binding: 6b09aca3d74edfab47af2731067bfd5f9c2f5764b3ceef143fb5236a9faef71b
- Audit binding: 95e0eb626b06e268645ef38999a2f2fc3a7ebc615ac72f774d2357a3717f632c
- Source cache: d7109f2db121e635c192eae904d8a02fa36209af317088b80a842fbf88e3962a
- Learned cache: 89455086e7af0e2a88f9d45f080204394d69285c9e08b024b4a3f61ef283dbd6

Caches: data/snake-whole-v104/source-features.npz and learned-features.npz.
Extraction and audit are complete; do not relaunch them.

The small decoder's analytical gradient now supports the exact V103 CE/legal/KL
objective in full_decoder_v104.py. It retains V92's FP32 Adam/save/restore,
undoes V103's eight-position loss normalization for shuffled row minibatches,
and accepts loss metadata separately from neural features. Two small CPU
tests passed in.019seconds: mixed parameter finite differences, zero-mass/empty
legal labels, invalid weighted-empty rejection and exact complete Adam restore.
These fixture updates are not main fitting or source-parameter qualification.

The matched50-epoch experiment remains unallocated. Its runner, source-head
qualification, common seed104001 permutations, independent final reviewer,
new32development seed reservation and bounded fitting contract still need
implementation. SNAKE_FULL_READOUT_V104_PLAN.md freezes its data/loss/optimizer,
equal training opportunities, both endpoint reports, eight original gates and
additional own-head/matched-control eligibility checks. No head fitting,
conditional gameplay, final evaluation, causal proof or viewer claim yet.
