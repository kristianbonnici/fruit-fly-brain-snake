"""Immutable first recurrent learning comparison; usable by isolated inference."""
import json
import os

from .common import ROOT, digest, stable
from .comparison_archive_v59 import read, install
from .comparison_contract_v59 import environment, resources

BASE = ROOT/'data/snake-whole-v69'


def precision():
    if os.environ.get('MLX_ENABLE_TF32') != '0':
        raise ValueError('Launch a fresh V69 process with MLX_ENABLE_TF32=0')
    return dict(MLX_ENABLE_TF32='0', dtype='float32', launch_time_configuration=True)


def contract():
    precision()
    path = BASE/'training-contract.json'
    plan = json.loads(path.read_text())
    if plan['status'] != 'allocated_structured_anatomical_comparison_v69' or plan['environment'] != environment():
        raise ValueError('Exact frozen recurrent comparison and runtime required')
    for p, sha in {**plan['sources'], **plan['evidence']}.items():
        if digest(ROOT/p) != sha:
            raise ValueError('Frozen recurrent evidence changed: '+p)
    return plan, digest(path)


def run_directory(plan, arm, seed):
    if arm not in ('joint', 'frozen') or seed != plan['training']['first_seed']:
        raise ValueError('Only the first registered matched pair is allocated')
    return BASE/f'runs/{arm}-{seed}'


def load_model(path, binding):
    import mlx.core as mx
    from .structured_model_v69 import Model as ReducedModel,load_anatomy
    from .structured_sensor_v69 import SPEC
    precision()
    arrays, meta = read(path)
    cfg = meta['configuration']
    if (meta['format'] != 'recurrent-training-boundary-v64' or cfg['allocation'] != binding
            or meta['binding'] != stable(cfg) or meta.get('comparison_version') != 69 or cfg.get('model_version') != 69 or cfg.get('sensor_spec') != SPEC
            or cfg['precision'] != precision() or cfg['runtime'] != environment()
            or cfg['device'] != str(mx.default_device()) or meta['arm'] not in ('joint','frozen')
            or cfg['train_internal'] != (meta['arm'] == 'joint')
            or meta['seed'] != cfg['seed'] or meta['completed_epoch'] < 0
            or (meta['completed_epoch'] == 0 and meta['cursor'] != 0)
            or (meta['completed_epoch'] > 0 and (meta['cursor'] != len(arrays['schedule.rows'])
                or meta['epoch'] != meta['completed_epoch']))):
        raise ValueError('Exact complete frozen V69 recurrent checkpoint required')
    for p, sha in cfg['sources'].items():
        if digest(ROOT/p) != sha:
            raise ValueError('Checkpoint model/training source changed: '+p)
    mx.random.seed(meta['seed'])
    model = ReducedModel(load_anatomy()[0], train_internal=cfg['train_internal'])
    if model._physics_identity != cfg['physics'] or model._anatomy_identity != cfg['anatomy']:
        raise ValueError('Exact retained anatomy and derived physics required')
    install(model, arrays)
    if model.capacity() != meta['capacity']:
        raise ValueError('Declared modest interfaces and internal capacity required')
    model.eval()
    return model, meta
