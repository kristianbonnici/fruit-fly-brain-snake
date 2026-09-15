"""Training-only aligned visual features and fixed teaching-prefix schedules."""
import json
import resource
import time
import numpy as np
from .common import ROOT,atomic_json,digest,stable
from .comparison_archive_v59 import read,save,array_hash
from .sequence_data_v64 import episode_rows,pack

BASE=ROOT/'data/snake-whole-v91'
STRATA=((0,0),(0,1),(1,1),(2,1))


def select(source):
    episodes=episode_rows(source,fitting=None);selection=[]
    for fitting in (True,False):
        for stratum,(cohort,origin) in enumerate(STRATA):
            candidates=[e for e,rows in episodes.items() if bool(source['fitting'][rows[0]])==fitting
                and int(source['cohort'][rows[0]])==cohort and int(source['origin'][rows[0]])==origin]
            count=16 if fitting else 8
            if len(candidates)<count:raise ValueError('Full prospective stratum/split required')
            chosen=sorted(candidates,key=lambda e:stable(['full-learning-v90-subset',int(e),fitting,cohort,origin]))[:count]
            for episode in chosen:
                full=episodes[episode];rows=full[:128]
                selection.append(dict(episode=episode,fitting=fitting,stratum=stratum,cohort=cohort,origin=origin,
                    rows=rows,original_length=len(full),prefix_length=len(rows),truncated=len(full)>len(rows)))
    return selection


def prefix_mass(original,valid):
    weights=np.asarray(original,np.float64).copy();valid=np.asarray(valid,bool)
    if weights.shape!=valid.shape or np.any(weights<0) or not np.isfinite(weights).all() or np.any(weights[~valid]!=0) or weights.sum()<=0:
        raise ValueError('Positive original fitting mass on valid targets only required')
    return weights/weights.sum()/64


def episodes(data,fitting):
    result={}
    for episode in np.unique(data['episode']):
        rows=np.flatnonzero(data['episode']==episode).astype(np.int32)
        if not np.array_equal(data['moves'][rows],np.arange(len(rows))) or np.any(data['fitting'][rows]!=data['fitting'][rows[0]]):
            raise ValueError('Single-split chronological prefixes from true move zero required')
        if fitting is None or bool(data['fitting'][rows[0]])==fitting:result[int(episode)]=rows
    if not result:raise ValueError('Nonempty declared episode split required')
    return result


def schedule(data,fitting,rng=None):
    selected=episodes(data,fitting);order=np.array(sorted(selected),np.int32)
    if rng is not None:order=rng.permutation(order)
    return pack(selected,order,np.zeros(len(order),np.int8),4,8)


def get_chunk(data,packed,cursor):
    if type(cursor) is not int or cursor<0 or cursor%8 or cursor+8>len(packed['rows']):raise ValueError('Aligned complete eight-step chunk required')
    rows=packed['rows'][cursor:cursor+8];active=rows>=0;raw=np.zeros((*rows.shape,17),np.float32)
    targets=np.zeros(rows.shape,np.int32);mass=np.zeros(rows.shape,np.float32)
    raw[active]=data['raw'][rows[active]];targets[active]=data['targets'][rows[active]]
    mass[active]=(data['mass'][rows[active]]*packed['rows'].size).astype(np.float32)
    reset=packed['reset'][cursor:cursor+8].copy()
    if np.any(reset & ~active) or np.any(data['moves'][rows[reset]]!=0):raise ValueError('Reset only at real episode start')
    return dict(raw=raw,targets=targets,mass=mass,active=active.copy(),reset=reset,rows=rows.copy())


def prepare():
    from .comparison_contract_v59 import resources
    started=time.perf_counter();path=BASE/'preparation-contract.json';plan=json.loads(path.read_text());binding=digest(path)
    for p,sha in {**plan['sources'],**plan['artifacts']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Frozen teaching preparation source differs: '+p)
    if (BASE/'preparation-start.json').exists() or (BASE/'data-report.json').exists():raise FileExistsError('Preserve preparation attempt')
    resources(60,started);atomic_json(BASE/'preparation-start.json',dict(binding=binding,unix=time.time()))
    source,sm=read(ROOT/'data/snake-whole-v73/inputs.npz');visual,vm=read(ROOT/'data/snake-whole-v80/inputs.npz')
    prediction,pm=read(ROOT/'data/snake-whole-v84/predictions-60.npz')
    n=len(source['targets'])
    assert n==139185 and prediction['raw'].shape==(161599,17) and pm['epoch']==60
    np.testing.assert_array_equal(source['pixels'],visual['pixels'][:n].astype(np.float32)/np.float32(10))
    for key in ('episode','fitting'):np.testing.assert_array_equal(source[key],visual[key][:n])
    visual_contract=ROOT/'data/snake-whole-v84/evaluation-contract.json'
    # The prediction archive has its own bound contract; the prepared source
    # identity must be the original V80 image corpus, not the teaching geometry.
    prediction_report=json.loads((ROOT/'data/snake-whole-v84/evaluation-report.json').read_text())
    original_plan=json.loads(visual_contract.read_text())
    assert prediction_report['binding']==pm['binding']==digest(visual_contract)
    assert original_plan['evidence']['data/snake-whole-v80/inputs.npz']==digest(ROOT/'data/snake-whole-v80/inputs.npz')
    assert all(digest(ROOT/p)==sha for p,sha in {**original_plan['sources'],**original_plan['evidence']}.items())
    assert prediction_report['predictions_sha256']['60']==digest(ROOT/'data/snake-whole-v84/predictions-60.npz')
    chosen=select(source);source_rows=np.concatenate([item['rows'] for item in chosen])
    arrays={k:source[k][source_rows].copy() for k in ('targets','valid','moves','episode','origin','cohort','fitting')}
    arrays['source_row']=source_rows;arrays['raw']=prediction['raw'][source_rows].copy();arrays['mass']=np.zeros(len(source_rows),np.float64)
    arrays['stratum']=np.empty(len(source_rows),np.int8);cursor=0;descriptions=[]
    for item in chosen:
        length=item['prefix_length'];selected=item['rows'];sl=slice(cursor,cursor+length)
        arrays['stratum'][sl]=item['stratum']
        if item['fitting']:arrays['mass'][sl]=prefix_mass(source['mass'][selected],source['valid'][selected])
        descriptions.append({k:v for k,v in item.items() if k!='rows'});cursor+=length
    old,_=read(ROOT/'data/snake-whole-v90/inputs.npz')
    assert set(old['source_row'].tolist())<=set(arrays['source_row'].tolist())
    assert np.isclose(arrays['mass'].sum(),1,rtol=0,atol=1e-12)
    assert np.all(arrays['mass'][~(arrays['fitting']&arrays['valid'])]==0)
    assert all(np.isclose(arrays['mass'][arrays['stratum']==s].sum(),.25,rtol=0,atol=1e-12) for s in range(4))
    rng=np.random.default_rng(91001);initial_rng=rng.bit_generator.state;schedules={};slots=[]
    for epoch in range(1,6):
        packed=schedule(arrays,True,rng);slots.append(int(packed['rows'].size))
        schedules.update({f'epoch_{epoch}.{k}':v for k,v in packed.items()})
    prediction_schedule=schedule(arrays,None);schedules.update({'prediction.'+k:v for k,v in prediction_schedule.items()})
    metadata=dict(format='full-teaching-visual-prefixes-v91',binding=binding,source_teaching_identity=sm['identity'],
        source_visual_identity=vm['identity'],source_predictions_identity=pm['identity'],selection=descriptions,
        initial_rng=initial_rng,final_rng=rng.bit_generator.state,training_seed=91001,
        source_alignment='All139185original image pairs,episode IDs and splits match exactly; labels are separate training-only fields.',
        cache='Frozen V84 learned17-output image features for training only. No true geometry sensors or neural states included.',
        scope='64fitting and32heldout episode prefixes, maximum128moves from actual game start; not scored complete games.')
    save(BASE/'inputs.npz',arrays,metadata);save(BASE/'schedules.npz',schedules,dict(binding=binding,input_sha256=digest(BASE/'inputs.npz'),
        initial_rng=initial_rng,final_rng=rng.bit_generator.state,chunk=8,batch=4,epochs=5,augmentation='none'))
    resources(60,started)
    groups=[]
    for fitting in (True,False):
        for stratum in range(4):
            mask=(arrays['fitting']==fitting)&(arrays['stratum']==stratum);valid=mask&arrays['valid']
            groups.append(dict(fitting=fitting,stratum=stratum,episodes=len(np.unique(arrays['episode'][mask])),rows=int(mask.sum()),
                valid_rows=int(valid.sum()),targets=np.bincount(arrays['targets'][valid],minlength=3).tolist(),fitting_mass=float(arrays['mass'][mask].sum())))
    report=dict(status='verified_full_visual_teaching_prefixes',binding=binding,total_rows=len(source_rows),groups=groups,
        fitting_rows=int(arrays['fitting'].sum()),heldout_rows=int((~arrays['fitting']).sum()),
        fitting_slots_by_epoch=slots,total_fitting_slots=sum(slots),optimizer_updates=sum(slots)//32,
        prediction_slots=int(prediction_schedule['rows'].size),total_initial_final_prediction_slots=2*int(prediction_schedule['rows'].size),
        input_sha256=digest(BASE/'inputs.npz'),schedules_sha256=digest(BASE/'schedules.npz'),
        source_alignment_exact=True,selection=descriptions,seconds=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,neural_observations=0,encoder_executions=0,updates=0,
        new_teacher_queries=0,games=0,final_games=0)
    atomic_json(BASE/'data-report.json',report);return report


if __name__=='__main__':print(json.dumps(prepare(),indent=2),flush=True)
