"""Namespaced original-anchor plus full-controller corrective teaching mixture."""
import json,time,resource
import numpy as np
from .common import ROOT,atomic_json,digest
from .comparison_archive_v59 import read,save
from .comparison_contract_v59 import resources
from .full_data_v91 import schedule,get_chunk,episodes
BASE=ROOT/'data/snake-whole-v95'
FIELDS=('targets','valid','moves','episode','origin','cohort','fitting','raw','stratum')
ROW_OFFSET=2**32;EPISODE_OFFSET=2**20


def merge(anchor,old,correction):
    anchor_rows={int(row):i for i,row in enumerate(anchor['source_row'])}
    fit_rows=set(int(row) for row in anchor['source_row'][anchor['fitting']])
    selected=np.flatnonzero(~old['fitting'] | np.isin(old['source_row'],list(fit_rows)))
    previous={k:old[k][selected].copy() for k in FIELDS}
    previous_rows=old['source_row'][selected].astype(np.int64)
    if np.any(previous_rows>=ROW_OFFSET) or np.any(previous['episode']>=EPISODE_OFFSET) or len(np.unique(correction['source_row']))!=len(correction['source_row']):raise ValueError('Distinct explicit old/new row and episode namespaces required')
    old_mass=np.zeros(len(selected),np.float64)
    for i,row in enumerate(previous_rows):
        if previous['fitting'][i]:
            a=anchor_rows[int(row)]
            for key in FIELDS:
                if not np.array_equal(previous[key][i],anchor[key][a]):raise ValueError('Exact original V90fitting anchor required')
            old_mass[i]=.5*anchor['mass'][a]
    new={k:correction[k].copy() for k in FIELDS};new['episode']=(new['episode'].astype(np.int64)+EPISODE_OFFSET).astype(np.int32)
    arrays={k:np.concatenate((previous[k],new[k]),axis=0) for k in FIELDS}
    arrays.update(source_row=np.concatenate((previous_rows,correction['source_row'].astype(np.int64)+ROW_OFFSET)),
        source_dataset=np.concatenate((np.zeros(len(previous_rows),np.int8),np.ones(len(new['episode']),np.int8))),
        mass=np.concatenate((old_mass,.5*correction['mass'].astype(np.float64))))
    if not np.isclose(arrays['mass'].sum(),1,rtol=0,atol=1e-12) or np.any(arrays['mass'][~(arrays['fitting']&arrays['valid'])]):raise ValueError('Only valid fitting rows carry normalized mass')
    if len(np.unique(arrays['source_row']))!=len(arrays['source_row']):raise ValueError('Unique namespaced source rows required')
    for s in range(5):
        expected=.125 if s<4 else .5
        if not np.isclose(arrays['mass'][arrays['stratum']==s].sum(),expected,rtol=0,atol=1e-12):raise ValueError('Exact original-anchor/correction balance required')
    episodes(arrays,None)
    return arrays,selected


def prepare():
    path=BASE/'preparation-contract.json';plan=json.loads(path.read_text());binding=digest(path);started=time.perf_counter()
    if (BASE/'preparation-start.json').exists():raise FileExistsError('Preserve preparation attempt')
    for p,sha in {**plan['sources'],**plan['artifacts']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Frozen corrective preparation source differs: '+p)
    resources(60,started);atomic_json(BASE/'preparation-start.json',dict(binding=binding))
    anchor,am=read(ROOT/'data/snake-whole-v90/inputs.npz');old,om=read(ROOT/'data/snake-whole-v91/inputs.npz');new,nm=read(ROOT/'data/snake-whole-v94/inputs.npz')
    labels,lm=read(ROOT/'data/snake-whole-v94/labels.npz');review=json.loads((ROOT/'data/snake-whole-v94/label-review.json').read_text())
    if review['status']!='independently_verified_full_controller_correction_inputs' or digest(ROOT/'data/snake-whole-v94/inputs.npz')!=review['inputs_sha256']:raise ValueError('Independent actual-controller correction audit required')
    arrays,selected=merge(anchor,old,new)
    if (len(arrays['episode'])!=7875 or arrays['fitting'].sum()!=4191 or len(episodes(arrays,True))!=56 or len(episodes(arrays,False))!=40 or len(selected)!=6077):raise ValueError('Exact prospectively selected source mixture required')
    rng=np.random.default_rng(95001);initial=rng.bit_generator.state;schedules={};slots=[]
    for epoch in range(1,6):
        packed=schedule(arrays,True,rng);slots.append(int(packed['rows'].size));schedules.update({f'epoch_{epoch}.{k}':v for k,v in packed.items()})
    predicted=schedule(arrays,None);schedules.update({'prediction.'+k:v for k,v in predicted.items()})
    meta=dict(format='full-corrective-visual-prefixes-v95',binding=binding,source_anchor_identity=am['identity'],source_old_identity=om['identity'],source_correction_identity=nm['identity'],
        initial_rng=initial,final_rng=rng.bit_generator.state,permutation_seed=95001,source_training_run='Continuation from V90seed91001, not an independent replicate',
        namespaces=dict(old_source_dataset=0,new_source_dataset=1,new_source_row_offset=ROW_OFFSET,new_episode_offset=EPISODE_OFFSET),
        fitting_mass='Half exact original V90fitting masses; half V94equal-prefix fitting masses. Every held-out/invalid row has zero mass.',
        neural_input='Frozen image-derived raw17 only; no true geometry, legality or cached neural state.')
    save(BASE/'inputs.npz',arrays,meta);save(BASE/'schedules.npz',schedules,dict(binding=binding,input_sha256=digest(BASE/'inputs.npz'),initial_rng=initial,final_rng=rng.bit_generator.state,batch=4,chunk=8,epochs=5,augmentation='none'))
    correction_rows=np.flatnonzero(arrays['source_dataset']==1).astype(np.int32)
    if not np.array_equal(arrays['targets'][correction_rows],labels['targets']) or not np.array_equal(arrays['fitting'][correction_rows],labels['fitting']):raise ValueError('Separate held-out legality diagnostic alignment required')
    save(BASE/'evaluation-legal.npz',dict(rows=correction_rows,legal_actions=labels['legal_actions'].copy(),executed=labels['executed'].copy(),fitting=labels['fitting'].copy()),
        dict(binding=binding,source_labels_sha256=digest(ROOT/'data/snake-whole-v94/labels.npz'),scope='Separate post-training development labels only; never full neural forward inputs, training safety filter or runtime action override.'))
    groups=[]
    for fitting in (True,False):
        for s in range(5):
            rows=(arrays['fitting']==fitting)&(arrays['stratum']==s);groups.append(dict(fitting=fitting,stratum=s,rows=int(rows.sum()),episodes=int(len(np.unique(arrays['episode'][rows]))),fitting_mass=float(arrays['mass'][rows].sum())))
    resources(60,started)
    report=dict(status='verified_full_corrective_anchor_mixture',binding=binding,total_rows=len(arrays['episode']),fitting_rows=4191,heldout_rows=3684,selected_old_rows=6077,new_rows=1798,
        groups=groups,fitting_slots_by_epoch=slots,total_fitting_slots=sum(slots),optimizer_updates=sum(slots)//32,prediction_slots=int(predicted['rows'].size),total_initial_final_prediction_slots=2*int(predicted['rows'].size),
        input_sha256=digest(BASE/'inputs.npz'),schedules_sha256=digest(BASE/'schedules.npz'),evaluation_legal_sha256=digest(BASE/'evaluation-legal.npz'),seconds=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,recurrent_observations=0,encoder_executions=0,updates=0,teacher_queries=0,policy_games=0,final_games=0)
    atomic_json(BASE/'data-report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':prepare()
