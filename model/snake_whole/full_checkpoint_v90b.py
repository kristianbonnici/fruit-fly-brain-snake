"""Exact recorded-input TBPTT checkpoints; no live game or unsaved random draws."""
import copy
import numpy as np
from .common import stable
from .comparison_archive_v59 import read,save,array_hash

PHASES=('initial','epoch_1','epoch_2','epoch_3','epoch_4','epoch_5','final')


def unpack(schedules,phase):
    prefix=('prediction' if phase in ('initial','final') else phase)+'.'
    result={k[len(prefix):]:v for k,v in schedules.items() if k.startswith(prefix)}
    if phase not in PHASES or set(result)!={'rows','reset','order','symmetry','episode_symmetry'}:
        raise ValueError('Exact declared phase and complete frozen schedule required')
    return result


def context(data,schedules,rng,phase,cursor):
    packed=unpack(schedules,phase);rows=packed['rows'];reset=packed['reset']
    if (rows.ndim!=2 or rows.shape[1]!=4 or rows.dtype!=np.int32 or reset.dtype!=bool or reset.shape!=rows.shape
            or type(cursor) is not int or cursor<0 or cursor%8 or cursor>len(rows) or len(rows)%8):
        raise ValueError('Exact batch4 chronological eight-observation cursor required')
    active=rows>=0
    if np.any(rows[active]>=len(data['episode'])) or np.any(reset & ~active):raise ValueError('Aligned real observations required')
    lane_episode=[];lane_observations=[]
    for lane in range(4):
        history=rows[:cursor,lane];history=history[history>=0]
        lane_episode.append(int(data['episode'][history[-1]]) if len(history) else None)
        lane_observations.append(int(data['moves'][history[-1]])+1 if len(history) else 0)
    previous=[unpack(schedules,p)['rows'] for p in PHASES[:PHASES.index(phase)]]
    fitting=phase.startswith('epoch_')
    updates=sum(unpack(schedules,p)['rows'].size//32 for p in PHASES[:PHASES.index(phase)] if p.startswith('epoch_'))
    return dict(phase=phase,cursor=cursor,batch=4,chunk=8,neural_steps=cursor*10,simulated_ms=cursor*100,
        optimizer_steps=updates+(cursor//8 if fitting else 0),
        computed_slots=sum(p.size for p in previous)+cursor*4,
        active_observations=sum(int(np.count_nonzero(p>=0)) for p in previous)+int(np.count_nonzero(active[:cursor])),
        lane_episode=lane_episode,lane_observations=lane_observations,
        lane_simulated_ms=[v*100 for v in lane_observations],
        schedule_identity=stable({k:array_hash(v) for k,v in packed.items()}),
        schedule_order=packed['order'].tolist(),rng=copy.deepcopy(rng),
        pending_events=[],eligibility=None,critic=None,curriculum='Fixed16x16 original recorded episode prefixes',
        observation_history='Bound frozen image-pair features at source_row; full neural state persists across chunks.',
        game_state='No live environment in recorded-input fitting. Original game histories and RNG are frozen upstream.',
        stochastic_state='No stochastic layers or online draws; all five permutations and their RNG states frozen before fitting.')


def save_checkpoint(path,core,ctx,binding,predictions,seen,replace=False):
    snapshot=core.snapshot()
    if (snapshot['neural_steps']!=ctx['neural_steps'] or snapshot['optimizer_steps']!=ctx['optimizer_steps']
            or core.batch!=ctx['batch']):raise ValueError('Checkpoint core clocks must equal the exact schedule context')
    arrays={k:snapshot[k] for k in ('state','log_e','m','v')}
    predictions=np.asarray(predictions);seen=np.asarray(seen)
    if predictions.dtype!=np.float32 or predictions.shape!=(len(seen),3) or seen.dtype!=bool:
        raise ValueError('Complete prediction progress arrays required')
    arrays.update(predictions=predictions.copy(),seen=seen.copy())
    return save(path,arrays,dict(format='full-recorded-input-learning-v90',binding=binding,
        core_identity=snapshot['identity'],context=copy.deepcopy(ctx)),replace=replace)


def restore_checkpoint(path,core,data,schedules,rng,binding):
    arrays,meta=read(path,binding)
    if (meta.get('format')!='full-recorded-input-learning-v90' or meta.get('core_identity')!=core.identity
            or set(arrays)!={'state','log_e','m','v','predictions','seen'}):
        raise ValueError('Exact full learning checkpoint and core identity required')
    ctx=meta['context'];expected=context(data,schedules,rng,ctx.get('phase'),ctx.get('cursor'))
    if ctx!=expected:raise ValueError('Exact source cursor, schedule, RNG and history required')
    seen=arrays['seen'];predictions=arrays['predictions'];n=len(data['episode'])
    if seen.dtype!=bool or seen.shape!=(n,) or predictions.dtype!=np.float32 or predictions.shape!=(n,3):
        raise ValueError('Complete aligned prediction progress required')
    consumed=unpack(schedules,ctx['phase'])['rows'][:ctx['cursor']];expected_seen=np.zeros(n,bool)
    expected_seen[consumed[consumed>=0]]=True
    if not np.array_equal(seen,expected_seen) or np.any(predictions[~seen]):
        raise ValueError('Prediction progress must match exact consumed observations')
    core.restore(dict(identity=meta['core_identity'],neural_steps=ctx['neural_steps'],simulated_ms=ctx['simulated_ms'],
        optimizer_steps=ctx['optimizer_steps'],**{k:arrays[k] for k in ('state','log_e','m','v')}))
    return copy.deepcopy(ctx),predictions.copy(),seen.copy(),meta
