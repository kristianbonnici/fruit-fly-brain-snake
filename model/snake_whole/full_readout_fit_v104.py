"""Identical bounded decoder-only fitting for two fixed full-network conditions."""
import argparse,json,resource,time
import numpy as np
from .common import ROOT,atomic_json,digest
from .comparison_archive_v59 import read,save,array_hash
from .comparison_contract_v59 import resources
from .full_decoder_v104 import Decoder,normalize,SOURCE_KEYS
from .full_metrics_v103 import summarize, objective_summary

BASE=ROOT/'data/snake-whole-v104';SOURCE=ROOT/'data/snake-whole-v103';CONDITIONS=('source','learned')


def contract(kind):
    path=BASE/f'decoder-{kind}-contract.json';plan=json.loads(path.read_text())
    for p,sha in {**plan['sources'],**plan['artifacts']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Frozen decoder source differs: '+p)
    return plan,digest(path)


def setup():
    data,_=read(SOURCE/'inputs.npz');order,meta=read(BASE/'decoder-schedules.npz')
    source,_=read(ROOT/'data/snake-whole-v69/runs/frozen-91001/checkpoint-10.npz')
    parameters={k:source[v] for k,v in SOURCE_KEYS.items()}
    if sum(v.size for v in parameters.values())!=2179:raise ValueError('Exact source head capacity required')
    rows=order['rows'];valid=np.flatnonzero(data['fitting']&data['valid']&(data['mass']>0))
    if rows.shape!=(50,len(valid)) or len(valid)!=9540 or any(not np.array_equal(np.sort(r),valid) for r in rows):raise ValueError('Every valid fitting row exactly once per frozen epoch')
    rng=np.random.default_rng(104001)
    if any(not np.array_equal(r,rng.permutation(valid)) for r in rows):raise ValueError('Exact common50seed104001 fitting permutations required')
    if meta['final_rng']!=rng.bit_generator.state:raise ValueError('Bound final RNG required')
    objective,_=read(SOURCE/'objectives.npz')
    fitting=data['fitting']&data['valid']
    if any(np.any(objective[k][~fitting]) for k in ('ce_mass','kl_mass','legal_mass','anchor_logits','legal_actions')):
        raise ValueError('Held rows must have no fitting objective or anchor labels')
    np.testing.assert_array_equal(objective['source_row'],data['source_row'])
    return data,rows,meta,parameters,objective


def context(condition,epoch,batch,schedule_meta,features_sha):
    if condition not in CONDITIONS or type(epoch)is not int or not 0<=epoch<=50 or type(batch)is not int or not 0<=batch<75 or epoch==50 and batch:
        raise ValueError('Exact decoder condition, epoch and batch cursor required')
    return dict(condition=condition,epoch=epoch,batch=batch,updates=epoch*75+batch,
        fitting_exposures=epoch*9540+batch*128,feature_sha256=features_sha,schedule_identity=schedule_meta['identity'],
        rng={k:schedule_meta[k] for k in ('initial_rng','final_rng')},
        neural_state='Frozen training-only output cache from full recurrent extraction; no core runs in decoder fitting.',
        efficacy='Fixed source; zero new internal updates.',pending_events=[],critic=None,eligibility=None)


def paired(before,after,seed):
    a={e['episode']:e for e in before['episodes'] if not e['fitting']};b={e['episode']:e for e in after['episodes'] if not e['fitting']}
    if set(a)!=set(b) or len(a)!=48:raise ValueError('Exactly paired48 development episodes required')
    ids=sorted(a);difference=np.array([a[e]['cross_entropy']-b[e]['cross_entropy'] for e in ids])
    samples=np.random.default_rng(seed).integers(0,48,(10000,48));ci=np.quantile(difference[samples].mean(axis=1),[.025,.975])
    return dict(episode_ids=ids,paired_ce_reductions=difference.tolist(),mean_reduction=float(difference.mean()),
        paired_bootstrap_95_interval=ci.tolist(),bootstrap_seed=seed,bootstrap_draws=10000,
        stratum_reductions={str(s):float(np.mean([difference[i] for i,e in enumerate(ids) if a[e]['stratum']==s])) for s in range(6)})


def comparison(reports):
    from .full_metrics_v100 import compare
    data,_=read(SOURCE/'inputs.npz');labels,_=read(SOURCE/'evaluation-legal.npz')
    original,_=read(BASE/'source-features.npz');initial=original['predictions']
    endpoints={}
    for condition in CONDITIONS:
        archive,_=read(BASE/f'decoder-{condition}-predictions.npz')
        endpoints[condition]=compare(reports['source']['initial'],reports[condition]['final'],data,initial,archive['predictions'],labels)
    own=paired(reports['learned']['initial'],reports['learned']['final'],104031)
    advantage=paired(reports['source']['final'],reports['learned']['final'],104032)
    illegal={c:sum(g['final_illegal'] for g in endpoints[c]['legality']['groups'].values()) for c in CONDITIONS}
    gates=dict(candidate_passes_all_eight=endpoints['learned']['predictive_gate_passed'],
        own_adaptation_mean_positive=own['mean_reduction']>0,own_adaptation_ci_positive=own['paired_bootstrap_95_interval'][0]>0,
        matched_advantage_mean_positive=advantage['mean_reduction']>0,matched_advantage_ci_positive=advantage['paired_bootstrap_95_interval'][0]>0,
        candidate_held_illegal_no_worse_than_adapted_source=illegal['learned']<=illegal['source'])
    return dict(endpoints=endpoints,own_head_improvement=own,matched_internal_advantage=advantage,heldout_illegal=illegal,gates=gates,
        predictive_gate_passed=all(gates.values()),scope='Previously inspected48held prefixes. Identical2179head adaptation on V96/V103 efficacy; zero new internal updates, one continued training seed, no gameplay.')


def batch_objective(objective,rows):
    return {k:objective[k][rows]*9540 if k.endswith('_mass') else objective[k][rows].copy()
            for k in ('ce_mass','kl_mass','anchor_logits','legal_mass','legal_actions')}


def qualify():
    plan,binding=contract('qualification');started=time.perf_counter()
    if (BASE/'decoder-qualification-start.json').exists():raise FileExistsError('Preserve decoder qualification attempt')
    atomic_json(BASE/'decoder-qualification-start.json',dict(binding=binding,decoder_update_executions=3,fitting_sample_exposures=384))
    data,orders,sm,p,objective=setup();source_error=0.
    for condition in CONDITIONS:
        path=BASE/f'{condition}-features.npz';a,_=read(path);x=normalize(a['post']);model=Decoder(p);old=model.logits(x)
        if not np.allclose(old,a['predictions'],rtol=1e-4,atol=3e-5):raise ValueError('Source decoder must match all original vectors')
        source_error=max(source_error,float(np.max(np.abs(old-a['predictions']))))
    timings=[]
    def step(batch):
        rows=orders[0,batch*128:(batch+1)*128];t=time.perf_counter();loss=model.step(x[rows],data['targets'][rows],batch_objective(objective,rows))
        timings.append(time.perf_counter()-t);return loss
    step(0);ctx=context('learned',0,1,sm,digest(path));checkpoint=BASE/'decoder-qualification-discarded.npz'
    t=time.perf_counter();model.save(checkpoint,ctx,binding);save_seconds=time.perf_counter()-t
    reference_loss=step(1);reference={prefix+'.'+k:v.copy() for prefix,values in [('parameter',model.parameters),('m',model.m),('v',model.v)] for k,v in values.items()}
    t=time.perf_counter();model.restore(checkpoint,ctx,binding);restore_seconds=time.perf_counter()-t
    actual_loss=step(1)
    if actual_loss!=reference_loss or any(not np.array_equal(reference[prefix+'.'+k],v) for prefix,values in [('parameter',model.parameters),('m',model.m),('v',model.v)] for k,v in values.items()):
        raise ValueError('Exact decoder parameters/moments/loss after resume required')
    resources(60,started)
    report=dict(status='qualified_exact_decoder_fit',binding=binding,decoder_parameters=2179,decoder_update_executions=3,
        fitting_sample_exposures=384,exact_restore=True,source_head_vectors=29188,maximum_source_head_absolute_error=source_error,
        batch_seconds=timings,save_seconds=save_seconds,restore_seconds=restore_seconds,seconds=time.perf_counter()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,recurrent_observations=0,encoder_executions=0,internal_updates=0,teacher_queries=0,policy_games=0,final_games=0)
    atomic_json(BASE/'decoder-qualification.json',report);print(json.dumps(report,indent=2))


def run(resume=False):
    plan,binding=contract('training');started=time.perf_counter();progress=BASE/'decoder-work.json'
    if (BASE/'decoder-report.json').exists():raise FileExistsError('Preserve completed decoder fit')
    if resume!=progress.exists():raise ValueError('Use explicit resume for existing progress')
    work=json.loads(progress.read_text()) if resume else dict(binding=binding,reserved_updates=0,completed_updates=0,
        reserved_exposures=0,completed_exposures=0,seconds=0.,invocations=0)
    prior=work['seconds'];work['invocations']+=1
    if work['binding']!=binding or work['invocations']>3:raise ValueError('Exact bounded source and invocations required')
    def persist():work.update(seconds=prior+time.perf_counter()-started);atomic_json(progress,work)
    persist();data,orders,sm,p,objective=setup();reports={}
    for condition in CONDITIONS:
        path=BASE/f'{condition}-features.npz';features_sha=digest(path);a,_=read(path);x=normalize(a['post']);model=Decoder(p)
        dest=BASE/f'decoder-{condition}-final.npz';current=BASE/f'decoder-{condition}-current.npz';report_path=BASE/f'decoder-{condition}-report.json'
        if dest.exists():
            model.restore(dest,context(condition,50,0,sm,features_sha),binding)
            if not report_path.exists():raise RuntimeError('Recover interrupted final report explicitly')
            reports[condition]=json.loads(report_path.read_text());continue
        epoch=0
        if current.exists():
            _,meta=read(current,binding);epoch=meta['context']['epoch']
            model.restore(current,context(condition,epoch,0,sm,features_sha),binding)
        for epoch in range(epoch,50):
            losses=[];t=time.perf_counter()
            for batch in range(75):
                rows=orders[epoch,batch*128:(batch+1)*128]
                if work['reserved_updates']+1>plan['maximum_decoder_update_executions'] or work['reserved_exposures']+len(rows)>plan['maximum_fitting_sample_exposures']:
                    raise RuntimeError('Allocated decoder work exhausted')
                work['reserved_updates']+=1;work['reserved_exposures']+=len(rows);persist()
                losses.append(model.step(x[rows],data['targets'][rows],batch_objective(objective,rows)))
                work['completed_updates']+=1;work['completed_exposures']+=len(rows);persist()
            ctx=context(condition,epoch+1,0,sm,features_sha)
            if model.steps!=ctx['updates']:raise ValueError('Decoder update cursor mismatch')
            model.save(current,ctx,binding,replace=current.exists());persist();resources(plan['internal_seconds'],started,prior)
            entry=dict(condition=condition,epoch=epoch+1,updates=model.steps,online_mean_batch_loss=float(np.mean(losses)),seconds=time.perf_counter()-t,invocation=work['invocations'])
            with (BASE/'decoder-epochs.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            if (epoch+1)%10==0:print(json.dumps(entry),flush=True)
        predictions=model.logits(x);seen=np.ones(len(x),bool)
        item=dict(condition=condition,initial=summarize(data,a['predictions'],seen),final=summarize(data,predictions,seen),
            decoder_updates=model.steps,feature_sha256=features_sha,decoder_parameter_hashes={k:array_hash(v) for k,v in model.parameters.items()},
            source_decoder_parameter_hashes={k:array_hash(v) for k,v in p.items()},frozen_log_e_hash=json.loads((BASE/'extraction-review.json').read_text())['conditions'][condition]['log_e_hash'],
            source_internal_training={'source':'Exact V96, V90 plus687V96 full updates after reduced transfer; continuation91001',
                'learned':'Same V96 plus1566V103 full updates; continuation91001'}[condition],
            initial_objective=objective_summary(data,objective,a['predictions'],seen),final_objective=objective_summary(data,objective,predictions,seen))

        # Preserve report before immutable endpoint so a completed checkpoint always has its report.
        if report_path.exists() and json.loads(report_path.read_text())!=item:raise ValueError('Recovered final report differs')
        atomic_json(report_path,item)
        prediction_path=BASE/f'decoder-{condition}-predictions.npz'
        if prediction_path.exists():
            old,meta=read(prediction_path,binding)
            if not np.array_equal(old['predictions'],predictions) or meta['condition']!=condition or meta['feature_sha256']!=features_sha:raise ValueError('Recovered predictions differ')
        else:save(prediction_path,dict(predictions=predictions),dict(binding=binding,condition=condition,feature_sha256=features_sha))
        model.save(dest,context(condition,50,0,sm,features_sha),binding);reports[condition]=item
    outcome=comparison(reports);resources(plan['internal_seconds'],started,prior);persist()
    result=dict(status='complete_bounded_matched_decoder_fit',binding=binding,work=work,comparison=outcome,
        logical_decoder_updates=7500,logical_fitting_sample_exposures=954000,decoder_parameters_per_condition=2179,
        seconds=work['seconds'],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        recurrent_observations=0,encoder_executions=0,internal_updates=0,teacher_queries=0,policy_games=0,final_games=0)
    atomic_json(BASE/'decoder-report.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['qualify','train']);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    qualify() if args.mode=='qualify' else run(args.resume)
