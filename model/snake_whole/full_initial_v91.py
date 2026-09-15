"""Explicit efficacy-only stage initialization; source optimizer is not resumed."""
import json
import numpy as np
from .common import ROOT,digest
from .comparison_archive_v59 import read,array_hash

SOURCE=ROOT/'data/snake-whole-v90/runs/internal-91001/final-checkpoint.npz'
SOURCE_SHA='22208e7ce255adf596883750cb7a8539c79441a9611da6be29b5cd70a4aea387'


def initial_efficacy():
    if digest(SOURCE)!=SOURCE_SHA:raise ValueError('Exact preserved V90 final candidate required')
    arrays,meta=read(SOURCE);review=json.loads((ROOT/'data/snake-whole-v90/review.json').read_text())
    gameplay=json.loads((ROOT/'data/snake-whole-v90/gameplay-review.json').read_text())
    if (not review['comparison']['predictive_gate_passed'] or not gameplay['development_progress_passed']
            or meta['context']['phase']!='final' or meta['context']['optimizer_steps']!=478
            or arrays['log_e'].shape!=(14737539,) or array_hash(arrays['log_e'])!=review['phases']['final']['log_e_hash']):
        raise ValueError('Reviewed complete full learner and positive gameplay required')
    return arrays['log_e'].copy(),dict(checkpoint_sha256=SOURCE_SHA,source_binding=meta['binding'],
        source_optimizer_steps=478,new_optimizer='Fresh zero Adam moments and counter for the broader fitting objective',
        training_run='Continuation of91001, not an independent training replicate')


def aligned_initial(old_data,old_predictions,new_data,new_predictions):
    old_rows=np.asarray(old_data['source_row']);new_rows=np.asarray(new_data['source_row'])
    if len(set(new_rows.tolist()))!=len(new_rows):raise ValueError('Unique prepared source rows required')
    lookup={int(row):i for i,row in enumerate(new_rows)}
    if not set(old_rows.tolist())<=set(lookup):raise ValueError('Complete previous prefixes must be preserved')
    selected=np.array([lookup[int(row)] for row in old_rows],np.int32)
    if not np.array_equal(old_predictions,new_predictions[selected]):raise ValueError('Frozen source predictions must reproduce exactly independent of episode packing')
    return len(selected)


def verify_initial_predictions(data,predictions):
    old,_=read(ROOT/'data/snake-whole-v90/inputs.npz');arrays,_=read(SOURCE)
    return aligned_initial(old,arrays['predictions'],data,predictions)
