"""Fresh-process full visual play, chunked actual states and bounded recovery."""
import argparse
import importlib.abc
import json
import resource
import sys
import time
import numpy as np
from snake import Snake
from .common import ROOT,atomic_json,digest,stable
from .comparison_archive_v59 import save,read,model_arrays,array_hash
from .comparison_contract_v59 import resources
from .comparison_metrics_v59 import interval
from .full_policy_v105 import make_policy as FullImagePolicy
from .full_recording_v91 import verify_part,policy_boundary,trial
from .visual_recording_v85 import game_state,image_hash,state_hash

BASE=ROOT/'data/snake-whole-v105'
BLOCKED=('snake_whole.full_readout_fit_', 'snake_whole.full_decoder_v104', 'snake_whole.full_loss_', 'snake_whole.full_features_', 'snake_whole.full_fit_', 'snake_whole.full_data_', 'snake_whole.full_training_','snake_whole.teacher_','snake_whole.perception_training_',
    'snake_whole.perception_data_','snake_whole.perception_run_','snake_whole.structured_training_',
    'snake_whole.structured_data_','snake_whole.structured_inputs_','snake_whole.refinement_training_',
    'snake_whole.corrective_inputs_','snake_whole.corrective_structured_','snake_whole.corrective_run_',
    'snake_whole.comparison_training_','snake_whole.comparison_data_','snake_whole.visual_run_',
    'snake_whole.visual_data_','snake_whole.visual_corpus_','snake_whole.recurrent_training_',
    'snake_whole.sequence_data_','snake_whole.image_run_')


class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(BLOCKED):raise ImportError('Training/teacher modules excluded from full visual play: '+fullname)
        return None


def contract():
    path=BASE/'evaluation-contract.json';plan=json.loads(path.read_text())
    if plan['version']!='full-learned-independent-gameplay-v105':raise ValueError('Exact full gameplay allocation required')
    for p,sha in {**plan['sources'],**plan['artifacts']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Frozen full gameplay evidence changed: '+p)
    return plan,digest(path)


def play(arm):
    if arm not in ('baseline','source','learned'):raise ValueError('Exact frozen efficacy conditions required')
    if any(name.startswith(BLOCKED) for name in sys.modules):raise RuntimeError('Fresh teacher/trainer-free process required')
    sys.meta_path.insert(0,BlockTraining());plan,binding=contract();started=time.perf_counter()
    directory=BASE/'runs'/arm;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():return json.loads((directory/'result.json').read_text())
    if (directory/'failure.json').exists():raise FileExistsError('Preserve a failed allocated comparison')
    attempts=[json.loads(p.read_text()) for p in sorted(directory.glob('attempt-*.json'))]
    for saved_path in sorted(directory.glob('game-*/part-*.npz')):
        _,meta=read(saved_path,binding);index=meta['attempt_index']
        if index>=len(attempts):raise ValueError('Recording part lacks its originating attempt')
        attempts[index]['known_observations']=max(attempts[index]['known_observations'],meta['attempt_known_observations'])
    prior_seconds=sum(max(a['seconds'],time.time()-a['started_unix']) if a['status']=='running' else a['seconds'] for a in attempts)
    prior_work=sum(max(a['known_observations'],a.get('reserved_observations',0))+64*(a['status']=='running') for a in attempts)
    limit=plan['internal_seconds_per_condition'];resources(limit,started,prior_seconds)
    attempt_path=directory/f'attempt-{len(attempts):02d}.json'
    attempt=dict(binding=binding,status='running',started_unix=time.time(),seconds=0.,known_observations=0,reserved_observations=0)
    atomic_json(attempt_path,attempt)
    try:
        policy=FullImagePolicy(arm,record_substeps=False);source_log_e=np.asarray(policy.core.log_e).copy();frozen_interface={k:array_hash(v) for k,v in model_arrays(policy.interface).items()};frozen_encoder={k:array_hash(v) for k,v in model_arrays(policy.visual.encoder).items()};trials=[];part_evidence={}
        for game_index,game_seed in enumerate(plan['game_seeds']):
            game_directory=directory/f'game-{game_index:02d}';game_directory.mkdir(exist_ok=True)
            policy.reset();game=Snake(seed=game_seed,size=16);parts=[];boundary=None
            for part_path in sorted(game_directory.glob('part-*.npz')):
                arrays,meta=read(part_path,binding)
                if (meta['game_seed']!=game_seed or meta['game_index']!=game_index or meta['policy_identity']!=policy.identity
                        or meta['part_index']!=len(parts) or meta['arm']!=arm):raise ValueError('Exact ordered game/part/model identity required')
                np.testing.assert_array_equal(arrays['neuron_ids'],policy.graph.ids)
                verify_part(arrays,meta,game);boundary=policy_boundary(arrays,meta)
                parts.append(part_path);part_evidence[str(part_path.relative_to(directory))]=digest(part_path)
                del arrays
            if boundary is not None:
                policy.restore(boundary)
                # Adopt a fully written part if interruption preceded the small
                # active pointer or completion marker. No neural work is repeated.
                saved_arrays={k:boundary[k] for k in ('neural_state','previous_image')}
                saved_meta={k:v for k,v in boundary.items() if k not in saved_arrays}
                save(game_directory/'active.npz',saved_arrays,dict(saved_meta,binding=binding,saved_game=game_state(game),
                    complete_parts=len(parts),last_part_sha256=digest(parts[-1])),replace=True)
            if game.done:
                completed=trial(game,game_seed);trials.append(completed)
                atomic_json(game_directory/'complete.json',dict(binding=binding,**completed,policy_identity=policy.identity,
                    parts=[p.name for p in parts],full_endpoint_samples=min(game.moves,256) if game_index<2 else 0,
                    full_endpoint_coverage_complete=game_index<2 and game.moves<=256,
                    full_10ms_samples=min(game.moves,32)*10 if game_index==0 else 0))
                continue
            validation_game=Snake(seed=game_seed,size=16)
            if boundary is not None:
                from .visual_recording_v85 import restore_game
                validation_game=restore_game(meta['saved_game'],game_seed)
            histories={k:[] for k in ('actions','logits','images','image_hashes','state_hashes','output_states',
                'sensors','view_logits','raw_perception','sensory_drive')};substeps=[];substep_moves=[];graded_states=[];graded_moves=[]
            chunk_start=game.moves

            def flush():
                nonlocal chunk_start
                if not histories['actions']:return
                arrays={k:np.asarray(v,np.int32 if k=='actions' else np.float64 if k=='logits' else
                    np.uint8 if k in ('image_hashes','state_hashes') else np.float32) for k,v in histories.items()}
                arrays['neuron_ids']=np.asarray(policy.graph.ids).copy()
                arrays['output_indices']=np.asarray(policy.embedding['output'],np.int32).copy()
                arrays['graded_moves']=np.asarray(graded_moves,np.int32)
                arrays['graded_states']=np.stack(graded_states) if graded_states else np.empty((0,165122),np.float32)
                arrays['boundary_state']=np.asarray(policy.state).copy()
                arrays['substep_moves']=np.asarray(substep_moves,np.int32)
                arrays['graded_substeps']=np.stack(substeps) if substeps else np.empty((0,10,165122),np.float32)
                metadata=dict(format='bounded-full-game-recording-v91',binding=binding,arm=arm,game_seed=game_seed,game_index=game_index,part_index=len(parts),
                    start_move=chunk_start,stop_move=game.moves,simulated_ms=game.moves*100,substep_ms=10,
                    neuron_count=165122,pair_count=25563197,synapse_count=124025046,
                    input_frame_order=['previous','current'],view_order=list(range(8)),
                    anatomy_identity=policy.graph.identity,physics_identity=policy.physics['identity'],
                    policy_identity=policy.identity,perception_checkpoint_sha256=policy.perception_checkpoint_sha256,
                    saved_game=game_state(game),attempt_index=len(attempts),attempt_known_observations=attempt['known_observations'],
                    recording='Actual64output states at every decision; full165122node endpoints only for first256moves of games0and1; full state at eachpartboundary; real10ms samples only first32moves of game0. Missing full states remain absent; no interpolated samples/spikes.',
                    initial_state='All165122graded deviations and previous image zero at the original three-segment move-zero start.')
                part_path=game_directory/f'part-{len(parts):04d}.npz';save(part_path,arrays,metadata)
                verify_part(arrays,metadata,validation_game)
                np.testing.assert_array_equal(arrays['boundary_state'],np.asarray(policy.state))
                parts.append(part_path);part_evidence[str(part_path.relative_to(directory))]=digest(part_path)
                saved=policy.snapshot();saved_arrays={k:saved[k] for k in ('neural_state','previous_image')}
                saved_meta={k:v for k,v in saved.items() if k not in saved_arrays}
                save(game_directory/'active.npz',saved_arrays,dict(saved_meta,binding=binding,saved_game=game_state(game),
                    complete_parts=len(parts),last_part_sha256=digest(part_path)),replace=True)
                for values in histories.values():values.clear()
                substeps.clear();substep_moves.clear();graded_states.clear();graded_moves.clear();chunk_start=game.moves
                attempt['seconds']=time.perf_counter()-started;atomic_json(attempt_path,attempt)
                resources(limit,started,prior_seconds)

            while not game.done:
                if prior_work+attempt['reserved_observations']>=plan['maximum_observations_per_condition']:
                    flush();raise RuntimeError('Aggregate full-game decision cap reached; incomplete games cannot qualify')
                if time.perf_counter()-started+prior_seconds>limit or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>8_000_000_000:
                    flush();raise RuntimeError('Frozen full-game time/RSS limit')
                image=game.pixels().copy();policy.record_substeps=game_index==0 and game.moves<32
                attempt['reserved_observations']+=1
                action,logits=policy.choose(image)
                if policy.record_substeps:
                    substeps.append(policy.last_substeps.copy());substep_moves.append(game.moves)
                histories['actions'].append(action);histories['logits'].append(logits.copy());histories['images'].append(image.copy())
                histories['image_hashes'].append(image_hash(image));histories['output_states'].append(np.asarray(policy.state[:,policy.output])[0].copy())
                if game_index<2 and game.moves<256:
                    graded_states.append(np.asarray(policy.state)[0].copy());graded_moves.append(game.moves)
                for key,value in [('sensors',policy.last_sensors),('view_logits',policy.last_views),
                    ('raw_perception',policy.last_raw),('sensory_drive',policy.last_drive)]:histories[key].append(value.copy())
                game.step(action);histories['state_hashes'].append(state_hash(game));attempt['known_observations']+=1
                if game.moves!=policy.observations:raise ValueError('Exact full policy/game clock required')
                if len(histories['actions'])==64:flush()
            flush();result=trial(game,game_seed);trials.append(result)
            atomic_json(game_directory/'complete.json',dict(binding=binding,**result,policy_identity=policy.identity,
                parts=[p.name for p in parts],full_endpoint_samples=min(game.moves,256) if game_index<2 else 0,
                    full_endpoint_coverage_complete=game_index<2 and game.moves<=256,full_10ms_samples=min(game.moves,32)*10 if game_index==0 else 0))
            print(json.dumps(dict(arm=arm,game=game_index+1,**result,seconds=time.perf_counter()-started)),flush=True)
        np.testing.assert_array_equal(np.asarray(policy.core.log_e),source_log_e)
        assert frozen_interface=={k:array_hash(v) for k,v in model_arrays(policy.interface).items()}
        assert frozen_encoder=={k:array_hash(v) for k,v in model_arrays(policy.visual.encoder).items()}
        if policy.core.updates!=0 or any(name.startswith(BLOCKED) for name in sys.modules):raise ValueError('Frozen teacher/trainer-free inference required')
        scores=np.array([t['score'] for t in trials]);unknown=sum(64 for a in attempts if a['status']=='running')
        result=dict(status='complete',binding=binding,arm=arm,policy_identity=policy.identity,games=len(trials),trials=trials,
            mean=float(scores.mean()),median=float(np.median(scores)),maximum=int(scores.max()),fraction_at_least_five=float(np.mean(scores>=5)),
            score_interval=interval(scores),termination_causes={k:sum(t['reason']==k for t in trials) for k in sorted({t['reason'] for t in trials})},
            total_moves=sum(t['moves'] for t in trials),known_attempt_observations=sum(a['known_observations'] for a in attempts)+attempt['known_observations'],
            reserved_attempt_observations=sum(max(a['known_observations'],a.get('reserved_observations',0)) for a in attempts)+attempt['reserved_observations'],
            interrupted_tail_observation_upper_bound=unknown,encoder_executions=8*(sum(a['known_observations'] for a in attempts)+attempt['known_observations']),
            full_endpoint_samples=sum(min(t['moves'],256) for t in trials[:2]),
            complete_output_state_samples=sum(t['moves'] for t in trials),full_10ms_samples=min(trials[0]['moves'],32)*10,
            part_artifacts=part_evidence,body_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            checks=dict(all_games_complete=len(trials)==32,registered_seeds=[t['seed'] for t in trials]==plan['game_seeds'],
                source_efficacy_unchanged=True,interface_encoder_unchanged=True,teacher_trainer_absent=True),updates=0,teacher_queries=0,final_games=0,
            scope='Explicit post-hoc full-network image-only three-condition adaptation/efficacy diagnostic preserving V104 comparative safety failure; comparison on32new independent development seeds; fixed interfaces, bounded labelled full-brain recordings, no inference learning or final acceptance claim.')
        if not all(result['checks'].values()):raise ValueError('Complete original evaluation required')
        atomic_json(directory/'result.json',result);attempt.update(status='complete',seconds=time.perf_counter()-started)
        atomic_json(attempt_path,attempt);return result
    except Exception as error:
        attempt.update(status='failed',seconds=time.perf_counter()-started,error=repr(error));atomic_json(attempt_path,attempt)
        atomic_json(directory/'failure.json',attempt);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=('baseline','source','learned'));args=parser.parse_args()
    result=play(args.arm);print(json.dumps({k:result[k] for k in ('status','mean','median','fraction_at_least_five','body_seconds')}),flush=True)
