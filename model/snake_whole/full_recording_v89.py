"""Pure original-game replay for bounded full-network recording parts."""
import numpy as np
from snake import Snake
from .common import stable
from .visual_recording_v85 import game_state,restore_game,image_hash,state_hash

N=165122


def verify_part(arrays,meta,game):
    count=len(arrays['actions']);start=game.moves;stop=start+count
    if (count<1 or count>64 or meta['start_move']!=start or meta['stop_move']!=stop
            or meta['simulated_ms']!=stop*100 or meta['substep_ms']!=10
            or meta['neuron_count']!=N or meta['pair_count']!=25563197 or meta['synapse_count']!=124025046
            or meta['input_frame_order']!=['previous','current'] or meta['view_order']!=list(range(8))):
        raise ValueError('Exact contiguous full-network recording and clocks required')
    shapes=dict(images=(count,16,16),graded_states=(count,N),sensors=(count,5),view_logits=(count,8,17),
        raw_perception=(count,17),sensory_drive=(count,256))
    for key,shape in shapes.items():
        a=arrays[key]
        if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Complete finite actual recording required: '+key)
    if (arrays['neuron_ids'].shape!=(N,) or arrays['neuron_ids'].dtype!=np.int64
            or len(np.unique(arrays['neuron_ids']))!=N or np.any(np.abs(arrays['graded_states'])>1)
            or arrays['logits'].shape!=(count,3) or not np.isfinite(arrays['logits']).all()
            or arrays['image_hashes'].shape!=(count,32) or arrays['state_hashes'].shape!=(count,32)):
        raise ValueError('Exact node IDs, graded bounds and action evidence required')
    selected=np.arange(start,min(stop,32),dtype=np.int32) if meta['game_index']==0 and start<32 else np.empty(0,np.int32)
    if (not np.array_equal(arrays['substep_moves'],selected)
            or arrays['graded_substeps'].shape!=(len(selected),10,N) or arrays['graded_substeps'].dtype!=np.float32
            or not np.isfinite(arrays['graded_substeps']).all() or np.any(np.abs(arrays['graded_substeps'])>1)):
        raise ValueError('Exactly declared limited 10 ms coverage required')
    if len(selected):np.testing.assert_array_equal(arrays['graded_substeps'][:,-1],arrays['graded_states'][selected-start])
    views=arrays['view_logits'].astype(np.float64);mapped=views.copy();mapped[:,4:,1]*=-1
    mapped[:,4:,2:7]=views[:,4:,12:17];mapped[:,4:,12:17]=views[:,4:,2:7]
    np.testing.assert_allclose(arrays['raw_perception'],mapped.mean(axis=1),rtol=2e-5,atol=2e-6)
    raw=arrays['raw_perception'].astype(np.float64);z=raw[:,2:].reshape(-1,3,5);z-=z.max(axis=-1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=-1,keepdims=True)
    sensors=np.concatenate((np.tanh(raw[:,:2]),np.sum(p*np.array([0,.25,1/3,.5,1]),axis=-1)),axis=1)
    np.testing.assert_allclose(arrays['sensors'],sensors,rtol=2e-5,atol=2e-6)
    for i,action in enumerate(arrays['actions']):
        if (game.done or not np.array_equal(game.pixels(),arrays['images'][i])
                or not np.array_equal(image_hash(game.pixels()),arrays['image_hashes'][i])
                or int(action)!=int(np.argmax(arrays['logits'][i]))):
            raise ValueError('Actual copied current image and unmodified argmax action required')
        game.step(int(action))
        if not np.array_equal(state_hash(game),arrays['state_hashes'][i]):raise ValueError('Original complete transition required')
    if stable(game_state(game))!=stable(meta['saved_game']):raise ValueError('Exact board, termination and food RNG required')
    return game


def policy_boundary(arrays,meta):
    return dict(neural_state=arrays['graded_states'][-1:].copy(),previous_image=arrays['images'][-1].copy(),
        observations=meta['stop_move'],simulated_ms=meta['simulated_ms'],policy_identity=meta['policy_identity'])


def trial(game,seed):
    if not game.done:raise ValueError('A complete original game is required for scores')
    return dict(seed=seed,score=game.score,moves=game.moves,reason=game.reason)
