"""Independent fresh-development games for two already frozen anatomical policies."""
import json
from pathlib import Path
from .common import ROOT,digest
from .comparison_contract_v59 import environment,resources
from .structured_contract_v69 import precision

BASE=ROOT/'data/snake-whole-v74'
MODELS=dict(warm=dict(path='data/snake-whole-v69/runs/frozen-91001/checkpoint-10.npz',
    sha256='a56040aec8b503e23a18f0abc2fe959ffdb36e7828f056434f7c1231a3a2118f',
    binding='acb13aa8b13565b4ee8687bc878cb2389b6635b7c31ccf0a12a019c62b9226ae',version=69,source_arm='frozen'),
    refined=dict(path='data/snake-whole-v73/runs/internal-91001/checkpoint-10.npz',
    sha256='498089a85b86ec350b449e15fc587bf0321bad97379f1374064ff13824e38581',
    binding='18d3b0b85c6e7965a7f693ca0d22bc403797f867dfc51b1d79817a09662ed56b',version=73,source_arm='internal'))


def contract():
    precision();path=BASE/'evaluation-contract.json';plan=json.loads(path.read_text())
    if plan['status']!='allocated_fresh_frozen_validation_v74' or plan['environment']!=environment() or plan['models']!=MODELS:
        raise ValueError('Exact frozen fresh-development allocation required')
    for p,sha in {**plan['sources'],**plan['evidence']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Frozen evaluation source/evidence changed: '+p)
    return plan,digest(path)


def run_directory(plan,arm,seed):
    if arm not in MODELS or seed!=91001:raise ValueError('Two fixed source policies only')
    return BASE/f'runs/{arm}-{seed}'


def source_path(arm):
    if arm not in MODELS:raise ValueError('Only declared warm and refined policies are eligible')
    p=ROOT/MODELS[arm]['path']
    if digest(p)!=MODELS[arm]['sha256']:raise ValueError('Selected frozen source checkpoint changed')
    return p


def load_source(arm):
    path=source_path(arm);spec=MODELS[arm]
    if spec['version']==69:from .structured_contract_v69 import load_model as load
    else:from .corrective_contract_v73 import load_model as load
    model,meta=load(path,spec['binding'])
    if meta['arm']!=spec['source_arm'] or meta['seed']!=91001 or meta['completed_epoch']!=10:
        raise ValueError('Only the two exact preselected trained policies are eligible')
    return model,dict(meta,source_arm=meta['arm'],arm=arm)


def load_model(path,binding):
    _,expected=contract()
    if binding!=expected:raise ValueError('Exact fresh-evaluation binding required')
    matches=[arm for arm in MODELS if source_path(arm).resolve()==Path(path).resolve()]
    if len(matches)!=1:raise ValueError('Only frozen preselected source paths are eligible')
    return load_source(matches[0])
