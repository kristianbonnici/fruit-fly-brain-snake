"""Pure replay checks for actual image-to-anatomy recordings; no MLX or teacher."""
import copy
import hashlib
import numpy as np
from snake import Snake
from .common import stable

def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value, list) else value


def game_state(game):
    return dict(game=copy.deepcopy(game.state()), since_food=game.since_food, food_rng=game.rng.getstate())


def restore_game(saved, seed):
    state = saved['game']
    game = Snake(seed=seed, size=16)
    for key in ('direction', 'score', 'moves', 'done', 'reason'):
        setattr(game, key, state[key])
    game.body = [tuple(v) for v in state['body']]
    game.food = None if state['food'] is None else tuple(state['food'])
    game.since_food = saved['since_food']
    game.rng.setstate(tuples(saved['food_rng']))
    if state['size'] != 16 or stable(game_state(game)) != stable(saved):
        raise ValueError('Exact original game state and food RNG required')
    return game


def image_hash(image):
    return np.frombuffer(hashlib.sha256(np.asarray(image, np.float32).tobytes()).digest(), np.uint8).copy()


def state_hash(game):
    return np.frombuffer(bytes.fromhex(stable(dict(game=game.state(), since_food=game.since_food))), np.uint8).copy()


def verify_game(arrays, meta):
    game = Snake(seed=meta['game_seed'], size=16)
    if len(arrays['actions']) != meta['saved_game']['game']['moves']:
        raise ValueError('Complete real action history required')
    moves = len(arrays['actions'])
    if (arrays['neuron_ids'].shape!=(1055,) or arrays['original_node'].shape!=(1055,)
            or arrays['neuron_ids'].dtype.kind not in 'iu' or arrays['original_node'].dtype.kind not in 'iu'
            or len(np.unique(arrays['neuron_ids']))!=1055 or len(np.unique(arrays['original_node']))!=1055
            or meta['neuron_count']!=1055 or meta['pair_count']!=36342 or meta['synapse_count']!=546836
            or meta['input_frame_order']!=['previous','current'] or meta['view_order']!=list(range(8))):
        raise ValueError('Portable exact node mapping and declared source anatomy required')
    if (arrays['images'].shape != (moves,16,16) or arrays['images'].dtype != np.float32
            or arrays['graded_states'].shape != (moves,1055) or arrays['graded_states'].dtype != np.float32
            or arrays['neural_state'].shape != (1,1055) or arrays['sensors'].shape != (moves,5) or arrays['sensors'].dtype != np.float32
            or not np.isfinite(arrays['graded_states']).all() or np.any(np.abs(arrays['graded_states'])>1.000001)
            or meta['simulated_ms'] != moves*100. or meta['policy_observations'] != moves
            or not np.array_equal(arrays['neural_state'][0], arrays['graded_states'][-1])):
        raise ValueError('Actual complete graded-state recording and neural clock required')
    if (arrays['view_logits'].shape!=(moves,8,17) or arrays['raw_perception'].shape!=(moves,17)
            or arrays['sensory_drive'].shape!=(moves,256) or arrays['graded_substeps'].shape!=(moves,10,1055)
            or arrays['previous_image'].shape!=(16,16) or meta['substep_ms']!=10
            or any(arrays[k].dtype!=np.float32 or not np.isfinite(arrays[k]).all() for k in
                ('view_logits','raw_perception','sensory_drive','graded_substeps','previous_image'))
            or np.any(np.abs(arrays['graded_substeps'])>1.000001)
            or not np.array_equal(arrays['graded_substeps'][:,-1],arrays['graded_states'])
            or not np.array_equal(arrays['previous_image'],arrays['images'][-1])):
        raise ValueError('Complete actual perception/current/substep/image-history recording required')
    views=arrays['view_logits'].astype(np.float64); mapped=views.copy(); mapped[:,4:,1]*=-1
    mapped[:,4:,2:7]=views[:,4:,12:17]; mapped[:,4:,12:17]=views[:,4:,2:7]
    np.testing.assert_allclose(arrays['raw_perception'],mapped.mean(axis=1),rtol=2e-5,atol=2e-6)
    raw=arrays['raw_perception'].astype(np.float64); z=raw[:,2:].reshape(-1,3,5);z-=z.max(axis=-1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=-1,keepdims=True)
    sensors=np.concatenate((np.tanh(raw[:,:2]),(p*np.array([0,.25,1/3,.5,1])).sum(axis=-1)),axis=1)
    np.testing.assert_allclose(arrays['sensors'],sensors,rtol=2e-5,atol=2e-6)
    for i, action in enumerate(arrays['actions']):
        if game.done or not np.array_equal(image_hash(game.pixels()), arrays['image_hashes'][i]):
            raise ValueError('Actual pre-action image or terminal boundary differs')
        if not np.array_equal(arrays['images'][i],game.pixels()):
            raise ValueError('Independently copied actual source image required')
        if int(action) != int(np.argmax(arrays['logits'][i])):
            raise ValueError('Actions must be the recorded policy argmax, without overrides')
        game.step(int(action))  # Original shaped reward is ignored.
        if not np.array_equal(state_hash(game), arrays['state_hashes'][i]):
            raise ValueError('Original movement, food, growth or termination differs')
    if not game.done or stable(game_state(game)) != stable(meta['saved_game']):
        raise ValueError('A complete unchanged game and exact final RNG are required')
    return dict(seed=meta['game_seed'], score=game.score, moves=game.moves, reason=game.reason)

