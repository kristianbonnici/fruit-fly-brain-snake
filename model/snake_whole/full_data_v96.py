"""Training-only fatal-example weights and source retention targets, no new inputs."""
import json,time,resource
import numpy as np
from .common import ROOT,atomic_json,digest
from .comparison_archive_v59 import read,save
from .comparison_contract_v59 import resources
from .full_data_v91 import get_chunk as original_chunk,episodes
BASE=ROOT/'data/snake-whole-v96'


def objective_arrays(data,source_predictions,fit_labels):
    n=len(data['episode']);fit=data['fitting'] & data['valid'];old=fit & (data['source_dataset']==0)
    expected=np.flatnonzero(data['fitting'] & (data['source_dataset']==1))
    rows=np.asarray(fit_labels['rows']);legal=np.asarray(fit_labels['legal_actions']);executed=np.asarray(fit_labels['executed'])
    if not np.array_equal(rows,expected) or legal.shape!=(len(rows),3) or legal.dtype!=bool or executed.shape!=(len(rows),):raise ValueError('Only exact fitting correction labels may determine weights')
    if np.any(~data['valid'][rows]) or np.any((executed<0)|(executed>2)) or not np.all(legal[np.arange(len(rows)),data['targets'][rows]]):raise ValueError('Valid safe teacher labels and original actions required')
    if source_predictions.shape!=(n,3) or not np.isfinite(source_predictions).all():raise ValueError('Aligned finite source predictions required')
    fatal=np.zeros(n,bool);fatal[rows]=~legal[np.arange(len(rows)),executed]
    ce=np.zeros(n,np.float64);kl=np.zeros(n,np.float64);kl[old]=data['mass'][old]
    targets=np.zeros((n,3),np.float32);targets[old]=source_predictions[old]
    prefixes=np.unique(data['episode'][rows]);per_prefix=.5/len(prefixes)
    for episode in prefixes:
        rr=rows[data['episode'][rows]==episode];yes=rr[fatal[rr]];no=rr[~fatal[rr]]
        if len(yes) and len(no):ce[yes]=per_prefix/2/len(yes);ce[no]=per_prefix/2/len(no)
        else:ce[rr]=per_prefix/len(rr)
    if not np.isclose(ce.sum(),.5,rtol=0,atol=1e-12) or not np.isclose(kl.sum(),.5,rtol=0,atol=1e-12):raise ValueError('Exactly half corrective CE and half original KL mass required')
    if np.any(ce[~fit]) or np.any(kl[~fit]) or np.any(targets[~old]):raise ValueError('No held-out fitting or distillation targets')
    return dict(ce_mass=ce,kl_mass=kl,anchor_logits=targets,source_row=data['source_row'].copy()),fatal


def get_chunk(data,packed,cursor,objective):
    chunk=original_chunk(data,packed,cursor);chunk.pop('mass')
    rows=chunk['rows'];active=chunk['active']
    for key in ('ce_mass','kl_mass'):
        value=np.zeros(rows.shape,np.float32);value[active]=(objective[key][rows[active]]*packed['rows'].size).astype(np.float32);chunk[key]=value
    source=np.zeros((*rows.shape,3),np.float32);source[active]=objective['anchor_logits'][rows[active]];chunk['anchor_logits']=source
    return chunk


def prepare():
    path=BASE/'preparation-contract.json';plan=json.loads(path.read_text());binding=digest(path);started=time.perf_counter()
    if (BASE/'preparation-start.json').exists():raise FileExistsError('Preserve objective preparation')
    for p,sha in {**plan['sources'],**plan['artifacts']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Frozen objective source differs: '+p)
    resources(60,started);atomic_json(BASE/'preparation-start.json',dict(binding=binding))
    old=ROOT/'data/snake-whole-v95';data,dm=read(old/'inputs.npz');schedules,sm=read(old/'schedules.npz');labels,lm=read(old/'evaluation-legal.npz')
    initial,im=read(old/'runs/internal-91001/initial-checkpoint.npz');review=json.loads((old/'review.json').read_text())
    if review['status']!='independently_verified_bounded_full_learning' or review['exact_predecessor_prediction_rows']!=6077 or review['tolerance_checked_live_rows']!=1798:raise ValueError('Verified V90source initial predictions required')
    selection=labels['fitting'];fit_labels={k:labels[k][selection].copy() for k in ('rows','legal_actions','executed')}
    objective,fatal=objective_arrays(data,initial['predictions'],fit_labels)
    if len(data['episode'])!=7875 or len(episodes(data,True))!=56 or len(episodes(data,False))!=40 or fatal.sum()!=21:raise ValueError('Unchanged exact V95mixture required')
    if not np.isclose(objective['ce_mass'][fatal].sum(),.21875,rtol=0,atol=1e-12):raise ValueError('Predeclared21fatal-prefix mass required')
    meta=dict(binding=binding,format='full-safety-objective-inputs-v96',initial_rng=dm['initial_rng'],final_rng=dm['final_rng'],
        predecessor_input_identity=dm['identity'],neural_input='Unchanged frozen image-derived raw17. Objective labels and source logits are separate.',
        mass_field='Unchanged V95diagnostic CE weighting only; training uses objectives.npz ce_mass and kl_mass.')
    save(BASE/'inputs.npz',data,meta)
    save(BASE/'schedules.npz',schedules,dict(binding=binding,input_sha256=digest(BASE/'inputs.npz'),predecessor_schedule_identity=sm['identity'],batch=4,chunk=8,epochs=5,augmentation='none'))
    save(BASE/'objectives.npz',objective,dict(binding=binding,input_sha256=digest(BASE/'inputs.npz'),
        source_initial_checkpoint_sha256=digest(old/'runs/internal-91001/initial-checkpoint.npz'),temperature=1,
        scope='Training labels/importance only. Original fitting rows receive source-to-current KL; corrective fitting rows receive teacher CE; held-out arrays are zero.'))
    save(BASE/'evaluation-legal.npz',labels,dict(binding=binding,source_identity=lm['identity'],scope='Unchanged V95post-training held-out diagnostic; not forwarded or used to fit.'))
    slots=[int(schedules[f'epoch_{i}.rows'].size) for i in range(1,6)];prediction=int(schedules['prediction.rows'].size)
    resources(60,started)
    report=dict(status='verified_source_retention_and_rare_fatal_objective',binding=binding,total_rows=7875,fitting_rows=4191,heldout_rows=3684,
        fitting_prefixes=56,heldout_prefixes=40,fitting_slots_by_epoch=slots,total_fitting_slots=sum(slots),optimizer_updates=sum(slots)//32,
        prediction_slots=prediction,total_initial_final_prediction_slots=2*prediction,ce_mass=float(objective['ce_mass'].sum()),kl_mass=float(objective['kl_mass'].sum()),
        fatal_ce_mass=float(objective['ce_mass'][fatal].sum()),other_ce_mass=float(objective['ce_mass'][~fatal].sum()),fatal_fitting_rows=np.flatnonzero(fatal).tolist(),
        kl_rows=int(np.count_nonzero(objective['kl_mass'])),ce_rows=int(np.count_nonzero(objective['ce_mass'])),
        unchanged_input_arrays=True,unchanged_schedules=True,seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        recurrent_observations=0,encoder_executions=0,updates=0,teacher_queries=0,policy_games=0,final_games=0)
    atomic_json(BASE/'data-report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':prepare()
