"""Source-bound conventional recurrent diagnostic, safe for independent inference."""
import json
import os
from .common import ROOT,digest,stable
from .comparison_archive_v59 import read,install
from .comparison_contract_v59 import environment,resources

BASE=ROOT/'data/snake-whole-v67'


def precision():
    if os.environ.get('MLX_ENABLE_TF32')!='0':
        raise ValueError('Launch a fresh process with MLX_ENABLE_TF32=0')
    return dict(MLX_ENABLE_TF32='0',dtype='float32',launch_time_configuration=True)


def contract():
    precision()
    path=BASE/'training-contract.json'
    plan=json.loads(path.read_text())
    if plan['status']!='allocated_conventional_gru_diagnostic_v67' or plan['environment']!=environment():
        raise ValueError('Exact frozen conventional GRU diagnostic required')
    for p,sha in {**plan['sources'],**plan['evidence']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Frozen diagnostic evidence changed: '+p)
    return plan,digest(path)


def run_directory(plan,arm,seed):
    if arm!='gru' or seed!=plan['training']['first_seed']:
        raise ValueError('Only one registered conventional comparison seed is allocated')
    return BASE/f'runs/{arm}-{seed}'


def load_model(path,binding):
    import mlx.core as mx
    from .gru_model_v67 import Model,SPEC
    arrays,meta=read(path)
    cfg=meta['configuration']
    if (meta['format']!='conventional-gru-boundary-v67' or cfg['allocation']!=binding
            or meta['binding']!=stable(cfg) or cfg['spec']!=SPEC or cfg['precision']!=precision()
            or cfg['runtime']!=environment() or cfg['device']!=str(mx.default_device())
            or meta['arm']!='gru' or meta['seed']!=cfg['seed'] or meta['completed_epoch']<0
            or (meta['completed_epoch']==0 and meta['cursor']!=0)
            or (meta['completed_epoch']>0 and (meta['cursor']!=len(arrays['schedule.rows']) or meta['epoch']!=meta['completed_epoch']))):
        raise ValueError('Complete frozen conventional checkpoint and exact runtime required')
    for p,sha in cfg['sources'].items():
        if digest(ROOT/p)!=sha:raise ValueError('Checkpoint source changed: '+p)
    mx.random.seed(meta['seed'])
    model=Model()
    install(model,arrays)
    if model.capacity()!=meta['capacity']:raise ValueError('Conventional parameter count changed')
    model.eval()
    return model,meta
