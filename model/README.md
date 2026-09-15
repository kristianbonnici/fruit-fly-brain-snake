# Research implementation

This directory contains the selected controller's source and its local Python import dependencies, copied byte-for-byte from the research checkout. The browser does not import or execute it.

## Start reading here

| File | Purpose |
| --- | --- |
| `snake_whole/full_policy_v105.py` | Selected adapted efficacy/readout assembly (`source` arm) |
| `snake_whole/full_visual_v89.py` | Image → sensor estimates → full network → action path |
| `snake_whole/full_core_v88.py` | Graded dynamics, CPU implementation, Metal kernels and adjoint |
| `snake_whole/perception_symmetry_v84.py` | Eight-view visual encoder |
| `snake_whole/structured_model_v69.py` | Learned sensory mapping and small action decoder |
| `snake_whole/full_fit_v96.py` | Internal efficacy training driver |
| `snake_whole/full_readout_fit_v104.py` | Decoder adaptation driver |
| `snake.py` | Original deterministic Snake rules |

Other files are retained because these implementations import them. Some reduced-network modules load historical interface weights; their presence does not mean the selected policy executes a reduced neural core. Source hashes are listed in `../docs/source-inventory.json`.

## Run synthetic numerical tests

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r model/requirements-test.txt
cd model
../.venv/bin/python -m unittest discover -s tests -v
```

These tests check a small synthetic graph against an independent dense calculation, finite-difference gradients, decoder loss/optimizer restoration, and recording invariants. They require no anatomical download and perform no full-network training. GPU checks are optional and skipped when MLX is absent.

## Full inference and training status

The original full image-policy path used Apple Silicon, MLX/Metal, NumPy and SciPy. Data preparation additionally uses pandas and PyArrow. The portable viewer needs none of these.

Full inference/training is **not a supported fresh-clone command in this release**. The frozen drivers require a long chain of prepared anatomy, exact source checkpoints, contracts, and historical corpora. Some historical artifacts were pruned before this release. Publishing source does not restore them. Running an old driver without those artifacts should fail its provenance checks; do not remove those checks to make it run.

This release supports replaying the captured result, inspecting the actual implementation, and checking its numerical building blocks. A portable frozen-checkpoint inference bundle or a new fully reproducible training recipe is separate future work. The supplied research entry points and reports explain the selected algorithm without claiming complete training reproducibility.
