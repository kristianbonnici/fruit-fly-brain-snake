"""Bounded, resumable five-epoch recorded-image learning on the entire graph."""
import argparse
import json
import resource
import time
import numpy as np
from .common import ROOT,atomic_json,digest
from .comparison_archive_v59 import read,save,array_hash,model_arrays
from .comparison_contract_v59 import resources
from .full_data_v96 import get_chunk
from .full_checkpoint_v90b import PHASES,unpack,context,save_checkpoint,restore_checkpoint
from .full_metrics_v96 import collect,summarize,compare,objective_summary

BASE=ROOT/'data/snake-whole-v96';RUN=BASE/'runs/internal-91001'


def populations(graph,states,gradient,log_e,initial,layout,sensory):
    result=[];imposed=np.zeros(graph.n,bool);imposed[sensory]=True
    presynaptic_population=np.asarray(graph.population)[layout.pre[layout.eligible]]
    for code in np.unique(graph.population):
        mask=np.asarray(graph.population)==code;generated=mask & ~imposed
        state=states[...,generated].astype(np.float64);edges=presynaptic_population==code
        values=np.exp(log_e[edges].astype(np.float64));delta=log_e[edges]-initial[edges]
        grad=gradient[edges] if gradient is not None else None
        result.append(dict(population=int(code),neurons=int(mask.sum()),imposed_input_neurons=int(np.sum(mask&imposed)),
            generated_neurons=int(generated.sum()),generated_state_rms=float(np.sqrt(np.mean(state*state))) if state.size else None,
            generated_active_neurons=int(np.sum(np.any(np.abs(state)>1e-6,axis=0))) if state.size else 0,
            generated_max_abs=float(np.max(np.abs(state))) if state.size else None,
            generated_saturated_fraction=float(np.mean(np.abs(state)>.95)) if state.size else None,
            outgoing_eligible_pairs=int(edges.sum()),changed_outgoing_eligible_pairs=int(np.count_nonzero(delta)),
            efficacy_min=float(values.min()) if len(values) else None,efficacy_max=float(values.max()) if len(values) else None,
            gradient_nonzero=int(np.count_nonzero(grad)) if grad is not None else None,
            gradient_l2=float(np.linalg.norm(grad.astype(np.float64))) if grad is not None else None))
    return result


def run(resume=False):
    import mlx.core as mx
    from .full_visual_v89 import load_layout
    from .full_core_v88 import Core
    from .full_training_v96 import forward_chunk,masked_backward
    from .fresh_contract_v74 import load_source
    from .full_initial_v96 import initial_efficacy,verify_initial_predictions
    path=BASE/'training-contract.json';plan=json.loads(path.read_text());binding=digest(path)
    for p,sha in {**plan['sources'],**plan['artifacts']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Frozen main learning source differs: '+p)
    RUN.mkdir(parents=True,exist_ok=True);current=RUN/'current.npz';progress=RUN/'work.json'
    if (RUN/'report.json').exists():raise FileExistsError('Completed main learning preserved')
    if resume!=current.exists():raise ValueError('Use explicit resume only for an existing complete current checkpoint')
    if not resume and progress.exists():raise FileExistsError('Existing attempt needs inspection')
    started=time.perf_counter()
    work=json.loads(progress.read_text()) if resume else dict(binding=binding,reserved_slots=0,completed_slots=0,
        reserved_updates=0,completed_updates=0,completed_active_observations=0,elapsed_seconds=0.,invocations=0)
    if work['binding']!=binding:raise ValueError('Matching durable work ledger required')
    prior_seconds=work['elapsed_seconds'];work['invocations']+=1
    if work['invocations']>3:raise RuntimeError('Prospective maximum three invocations')
    resources(plan['internal_seconds'],started,prior_seconds);atomic_json(progress,work)
    data,dm=read(BASE/'inputs.npz');objective,_=read(BASE/'objectives.npz')
    np.testing.assert_array_equal(objective['source_row'],data['source_row'])
    schedules,_=read(BASE/'schedules.npz');rng={k:dm[k] for k in ('initial_rng','final_rng')}
    graph,layout,embedding,_=load_layout();initial,initial_source=initial_efficacy()
    core=Core(layout,4,'gpu',initial)
    interface,_=load_source('warm');interface.eval();interface_before={k:array_hash(v) for k,v in model_arrays(interface).items()}
    sensory=mx.array(embedding['sensory']);output=mx.array(embedding['output'])
    if resume:ctx,predictions,seen,_=restore_checkpoint(current,core,data,schedules,rng,binding)
    else:
        ctx=context(data,schedules,rng,'initial',0);predictions=np.zeros((len(data['episode']),3),np.float32);seen=np.zeros(len(predictions),bool)
        save_checkpoint(current,core,ctx,binding,predictions,seen)
    def persist():
        save_checkpoint(current,core,ctx,binding,predictions,seen,replace=True)
        work['elapsed_seconds']=prior_seconds+time.perf_counter()-started;atomic_json(progress,work)
    for phase_index in range(PHASES.index(ctx['phase']),len(PHASES)):
        phase=PHASES[phase_index];packed=unpack(schedules,phase);training=phase.startswith('epoch_')
        print(json.dumps(dict(phase=phase,cursor=ctx['cursor'],slots=packed['rows'].size,optimizer_steps=core.updates)),flush=True)
        for cursor in range(ctx['cursor'],len(packed['rows']),8):
            resources(plan['internal_seconds'],started,prior_seconds)
            if work['reserved_slots']+32>plan['maximum_computed_slots'] or work['reserved_updates']+int(training)>plan['maximum_update_executions']:
                raise RuntimeError('Frozen total main work including recovery reserve exhausted')
            work['reserved_slots']+=32;work['reserved_updates']+=int(training);work['elapsed_seconds']=prior_seconds+time.perf_counter()-started
            atomic_json(progress,work);chunk=get_chunk(data,packed,cursor,objective);t=time.perf_counter()
            records,loss,logits,states,objective_parts=forward_chunk(core,interface,chunk,sensory,output,training)
            forward_seconds=time.perf_counter()-t;gradient=None;backward_seconds=optimizer_seconds=0.;update=None
            if training:
                t=time.perf_counter();gradient,_,_=masked_backward(core,records);backward_seconds=time.perf_counter()-t
                t=time.perf_counter();update=core.update(gradient);optimizer_seconds=time.perf_counter()-t
            collect(predictions,seen,chunk['rows'],np.asarray(logits));ctx=context(data,schedules,rng,phase,cursor+8)
            if core.tick!=ctx['neural_steps'] or core.updates!=ctx['optimizer_steps']:raise RuntimeError('Logical clocks diverged')
            diagnostic=None
            if cursor==0 or cursor+8==len(packed['rows']):
                copied=np.asarray(states).copy();active_states=copied[chunk['active']]
                if not np.isfinite(active_states).all() or np.any(np.abs(active_states)>1):raise FloatingPointError('Finite bounded full states required')
                if len(active_states):
                    diagnostic=populations(graph,active_states,np.asarray(gradient) if gradient is not None else None,
                        np.asarray(core.log_e),initial,layout,embedding['sensory'])
                dest=RUN/f'{phase}-states-{cursor:04d}.npz'
                arrays=dict(states=copied,rows=chunk['rows'],active=chunk['active'],reset=chunk['reset'],neuron_ids=np.asarray(graph.ids))
                if not dest.exists():save(dest,arrays,dict(binding=binding,phase=phase,cursor=cursor,dt_ms=100,
                    coverage='Actual full 100ms endpoints for this eight-observation batch chunk; inactive slots explicitly marked. No invented substeps.'))
                else:
                    old,_=read(dest,binding)
                    if any(array_hash(v)!=array_hash(old[k]) for k,v in arrays.items()):raise ValueError('Recovery state recording changed')
            log=dict(phase=phase,cursor=cursor,loss=loss,objective_parts=objective_parts.tolist(),update=update,forward_seconds=forward_seconds,
                backward_seconds=backward_seconds,optimizer_seconds=optimizer_seconds,active_observations=int(chunk['active'].sum()),
                prediction_hash=array_hash(np.asarray(logits)),populations=diagnostic)
            with (RUN/'chunks.jsonl').open('a') as f:f.write(json.dumps(dict(log,invocation=work['invocations']))+'\n')
            work['completed_slots']+=32;work['completed_active_observations']+=int(chunk['active'].sum());work['completed_updates']+=int(training)
            atomic_json(progress,work)
            del records,gradient,states,logits
            if ctx['cursor']%64==0 or ctx['cursor']==len(packed['rows']):
                persist();print(json.dumps(dict(phase=phase,cursor=ctx['cursor'],optimizer_steps=core.updates,
                    loss=loss,gradient_norm=update['norm'] if update else None,elapsed_seconds=work['elapsed_seconds'])),flush=True)
        boundary=RUN/(phase+'-checkpoint.npz')
        if not boundary.exists():save_checkpoint(boundary,core,ctx,binding,predictions,seen)
        matched_source_rows=verify_initial_predictions(data,predictions) if phase=='initial' else None
        report=summarize(data,predictions,seen)
        report['objective']=objective_summary(data,objective,predictions,seen)
        report.update(initial_source=initial_source,source_alignment=matched_source_rows)
        report.update(binding=binding,phase=phase,checkpoint_sha256=digest(boundary),context=ctx,
            learned_log_e_hash=array_hash(np.asarray(core.log_e)),interfaces_unchanged=interface_before=={k:array_hash(v) for k,v in model_arrays(interface).items()})
        if not report['interfaces_unchanged']:raise RuntimeError('Frozen interface changed')
        target=RUN/(phase+'-report.json')
        if target.exists():
            if json.loads(target.read_text())!=report:raise ValueError('Recovery phase report differs')
        else:atomic_json(target,report)
        print(json.dumps(dict(phase=phase,complete=True,groups=report['groups'])),flush=True)
        if phase_index+1<len(PHASES):
            core.state=mx.zeros((4,core.n),mx.float32);core.tick=0
            ctx=context(data,schedules,rng,PHASES[phase_index+1],0);predictions=np.zeros_like(predictions);seen[:]=False;persist()
    first=json.loads((RUN/'initial-report.json').read_text());last=json.loads((RUN/'final-report.json').read_text())
    legal,_=read(BASE/'evaluation-legal.npz')
    initial_predictions,_=read(RUN/'initial-checkpoint.npz',binding)
    outcome=compare(first,last,data,initial_predictions['predictions'],predictions,legal);persist();resources(plan['internal_seconds'],started,prior_seconds)
    result=dict(status='complete_bounded_full_learning_probe',binding=binding,work=work,comparison=outcome,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,logical_context=ctx,logical_optimizer_steps=core.updates,initial_source=initial_source,
        image_encoder_executions=0,teacher_queries=0,policy_games=0,final_games=0,training='Learned with demonstrations; explicit full-graph discrete TBPTT; E-only with rare-fatal teacher CE and source-retention KL; fixed interfaces.',
        checkpoints={p.name:digest(p) for p in RUN.glob('*-checkpoint.npz')})
    atomic_json(RUN/'report.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args();run(args.resume)
