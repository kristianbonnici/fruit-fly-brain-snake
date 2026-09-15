"""Frozen V73 teaching experiment, reusing the qualified V71 inference model."""
import json
from .common import ROOT,digest
from .comparison_contract_v59 import environment,resources
from .refinement_contract_v71 import load_model as qualified_load
from .structured_contract_v69 import precision

BASE=ROOT/'data/snake-whole-v73'


def contract():
    precision();path=BASE/'training-contract.json';plan=json.loads(path.read_text())
    if plan['status']!='allocated_internal_corrections_v73' or plan['environment']!=environment():raise ValueError('Exact corrected teaching allocation required')
    for p,sha in {**plan['sources'],**plan['evidence']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Frozen correction source/evidence changed: '+p)
    return plan,digest(path)


def run_directory(plan,arm,seed):
    if arm!='internal' or seed!=91001:raise ValueError('Only the registered warm-start internal correction is allocated')
    return BASE/f'runs/{arm}-{seed}'


def load_model(path,binding):
    model,meta=qualified_load(path,binding)
    if meta['configuration'].get('data_version')!=73:raise ValueError('Exact versioned correction dataset binding required')
    return model,meta
