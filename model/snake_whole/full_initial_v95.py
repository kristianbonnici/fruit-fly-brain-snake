"""V90 efficacy continuation and independently indexed old/live source checks."""
import json
import numpy as np
from .common import ROOT,digest
from .comparison_archive_v59 import read
from .full_initial_v91 import initial_efficacy as predecessor_efficacy
from .full_data_v95 import ROW_OFFSET,EPISODE_OFFSET

OLD=ROOT/'data/snake-whole-v91/runs/internal-91001/initial-checkpoint.npz'
OLD_SHA='e5055e62bc0af8c4004f7f4c6cc4d1d5f768c9a6b13008b312b412bc78f65eb2'
COLLECTION=ROOT/'data/snake-whole-v94/collection'


def initial_efficacy():
    values,source=predecessor_efficacy()
    source['new_optimizer']='Fresh zero Adam moments and counter for the anchored corrective objective'
    return values,source


def align_old(old,old_predictions,data,predictions):
    selected=np.flatnonzero(data['source_dataset']==0)
    ids=np.asarray(old['source_row']);wanted=data['source_row'][selected]
    if len(np.unique(ids))!=len(ids) or len(np.unique(wanted))!=len(wanted):raise ValueError('Unique original source rows required')
    lookup={int(row):i for i,row in enumerate(ids)}
    if not set(wanted.tolist())<=set(lookup):raise ValueError('Every selected old source row required')
    prior=np.array([lookup[int(row)] for row in wanted])
    for key in ('episode','moves','fitting','raw'):
        if not np.array_equal(old[key][prior],data[key][selected]):raise ValueError('Original source identity/chronology changed: '+key)
    if not np.array_equal(old_predictions[prior],predictions[selected]):raise ValueError('Bit-for-bit original source predictions required')
    return len(selected)


def check_live_logits(expected,actual,executed):
    if expected.shape!=actual.shape or expected.shape!=(len(executed),3) or not np.isfinite(actual).all():raise ValueError('Aligned finite live logits required')
    if not np.allclose(actual,expected,rtol=1e-4,atol=3e-5):raise ValueError('Live B1/cached B4 logits exceed registered tolerance')
    if not np.array_equal(actual.argmax(axis=1),executed) or not np.array_equal(expected.argmax(axis=1),executed):raise ValueError('Every source action must reproduce; tolerance cannot hide an argmax flip')
    return float(np.max(np.abs(actual-expected))) if len(actual) else 0.


def live_references(data):
    review=json.loads((ROOT/'data/snake-whole-v94/collection-review.json').read_text())
    selected=np.flatnonzero(data['source_dataset']==1)
    key_to_row={(int(data['episode'][r])-EPISODE_OFFSET,int(data['moves'][r])):int(r) for r in selected}
    if len(key_to_row)!=len(selected):raise ValueError('Unique live prefix/move rows required')
    logits=np.zeros((len(data['episode']),3),np.float64);actions=np.full(len(logits),-1,np.int32);seen=set()
    for name,sha in review['part_artifacts'].items():
        path=COLLECTION/name
        if digest(path)!=sha:raise ValueError('Reviewed source recording changed: '+name)
        a,m=read(path)
        for offset in range(len(a['actions'])):
            key=(m['game_index'],m['start_move']+offset)
            if key not in key_to_row or key in seen:raise ValueError('Exact complete source partition required')
            row=key_to_row[key];seen.add(key)
            if not np.array_equal(data['raw'][row],a['raw_perception'][offset]) or bool(data['fitting'][row])!=m['fitting']:raise ValueError('Source live raw input or role differs')
            logits[row]=a['logits'][offset];actions[row]=a['actions'][offset]
        del a
    if seen!=set(key_to_row):raise ValueError('Every actual source observation required')
    original,_=read(ROOT/'data/snake-whole-v94/inputs.npz')
    original_rows=data['source_row'][selected]-ROW_OFFSET
    if np.any(original_rows<0) or np.any(original_rows>=len(original['episode'])):raise ValueError('Live source namespace differs')
    for key in ('moves','raw','fitting'):
        if not np.array_equal(data[key][selected],original[key][original_rows]):raise ValueError('Live row mapping differs: '+key)
    if not np.array_equal(data['episode'][selected]-EPISODE_OFFSET,original['episode'][original_rows]):raise ValueError('Live episode namespace differs')
    return selected,logits,actions


def verify_initial_predictions(data,predictions):
    if digest(OLD)!=OLD_SHA:raise ValueError('Preserved V91initial V90-efficacy checkpoint required')
    old,_=read(ROOT/'data/snake-whole-v91/inputs.npz');a,_=read(OLD)
    count=align_old(old,a['predictions'],data,predictions)
    selected,logits,actions=live_references(data)
    error=check_live_logits(logits[selected],predictions[selected],actions[selected])
    if count!=6077 or len(selected)!=1798:raise ValueError('Exact selected old/live source coverage required')
    return dict(exact_old_rows=count,tolerance_checked_live_rows=len(selected),identical_live_actions=len(selected),
        maximum_live_logit_absolute_error=error,live_logit_rtol=1e-4,live_logit_atol=3e-5)
