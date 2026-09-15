"""Immutable first recurrent learning comparison; usable by isolated inference."""
import json
import os

from .common import ROOT, digest, stable
from .comparison_archive_v59 import read, install
from .comparison_contract_v59 import environment, resources

BASE = ROOT/'data/snake-whole-v71'


def precision():
    if os.environ.get('MLX_ENABLE_TF32') != '0':
        raise ValueError('Launch a fresh V71 process with MLX_ENABLE_TF32=0')
    return dict(MLX_ENABLE_TF32='0', dtype='float32', launch_time_configuration=True)


def contract():
    precision()
    path = BASE/'training-contract.json'
    plan = json.loads(path.read_text())
    if plan['status'] != 'allocated_internal_refinement_v71' or plan['environment'] != environment():
        raise ValueError('Exact frozen recurrent comparison and runtime required')
    for p, sha in {**plan['sources'], **plan['evidence']}.items():
        if digest(ROOT/p) != sha:
            raise ValueError('Frozen recurrent evidence changed: '+p)
    return plan, digest(path)


def run_directory(plan, arm, seed):
    if arm != 'internal' or seed != plan['training']['first_seed']:
        raise ValueError('Only the registered internal-only refinement is allocated')
    return BASE/f'runs/{arm}-{seed}'


def load_model(path, binding):
    import mlx.core as mx
    from .refinement_model_v71 import make_model,interfaces
    from .structured_sensor_v69 import SPEC
    precision()
    arrays, meta = read(path)
    cfg = meta['configuration']
    if (meta['format'] != 'recurrent-training-boundary-v64' or cfg['allocation'] != binding
            or meta['binding'] != stable(cfg) or meta.get('comparison_version') != 71 or cfg.get('model_version') != 71 or cfg.get('sensor_spec') != SPEC
            or cfg['precision'] != precision() or cfg['runtime'] != environment()
            or cfg['device'] != str(mx.default_device()) or meta['arm'] != 'internal'
            or cfg['train_internal'] != (meta['arm'] == 'internal')
            or meta['seed'] != cfg['seed'] or meta['completed_epoch'] < 0
            or (meta['completed_epoch'] == 0 and meta['cursor'] != 0)
            or (meta['completed_epoch'] > 0 and (meta['cursor'] != len(arrays['schedule.rows'])
                or meta['epoch'] != meta['completed_epoch']))):
        raise ValueError('Exact complete frozen V71 recurrent checkpoint required')
    for p, sha in cfg['sources'].items():
        if digest(ROOT/p) != sha:
            raise ValueError('Checkpoint model/training source changed: '+p)
    mx.random.seed(meta['seed'])
    model = make_model(meta['seed'])
    if model._physics_identity != cfg['physics'] or model._anatomy_identity != cfg['anatomy']:
        raise ValueError('Exact retained anatomy and derived physics required')
    install(model, arrays)
    if not cfg.get('fixed_interfaces') or interfaces(model)!=cfg['interface_identity']:
        raise ValueError('Exact fixed sensory and output interfaces required')
    if model.capacity() != meta['capacity']:
        raise ValueError('Declared modest interfaces and internal capacity required')
    model.eval()
    return model, meta
