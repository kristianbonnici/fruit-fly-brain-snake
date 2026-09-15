"""Derived initial physics and identity-bound state for the reduced V62 topology."""
import json

import numpy as np
import mlx.core as mx

from .common import ROOT, digest, stable
from .comparison_archive_v59 import read, array_hash, model_arrays
from .comparison_contract_v59 import environment
from .reduced_model_v62 import AnatomicalCore, ReducedModel as PrototypeModel, ReducedPolicy as PrototypePolicy, load_anatomy


def load_physics():
    base = ROOT/'data/snake-whole-v63'
    report = json.loads((base/'derived-gain.json').read_text())
    path = base/'derived-physics.npz'
    if report['status'] != 'derived_initial_linear_gain' or digest(path) != report['physics_sha256']:
        raise ValueError('Exact separately derived initial physics required')
    arrays, meta = read(path, digest(base/'gain-derivation-plan.json'))
    if (meta['format'] != 'derived-reduced-physics-v63' or meta['sensory_current_gain'] != 3.
            or meta['game_tick_ms'] != 100. or meta['actual_float32_resting_radius'] >= meta['target_resting_radius']):
        raise ValueError('Compatible recorded engineering physics required')
    return arrays, meta


class ReducedModel(PrototypeModel):
    def __init__(self, arrays, train_internal=True):
        _, anatomy_meta = load_anatomy()
        if {k: array_hash(v) for k, v in arrays.items()} != anatomy_meta['array_hashes']:
            raise ValueError('Only the exact frozen1055neuron induced anatomy is supported')
        physics_arrays, physics = load_physics()
        if physics['anatomy_sha256'] != digest(ROOT/'data/snake-whole-v62/reduced-anatomy.npz'):
            raise ValueError('Physics belongs to another anatomical artifact')
        super().__init__(arrays, train_internal)
        # Instance-local replacement, before any forward work; the original
        # gain0.8 prototype and its tests are preserved as a separate version.
        self.core = AnatomicalCore(arrays, train_internal, gain=physics['gain'], dt_ms=physics['dt_ms'],
            tau_ms=physics['tau_ms'], steps=physics['integration_steps'])
        # The derivation froze the actual float32 physiological coefficients.
        # Their integer count factor remains separate in the shared core.
        if not np.allclose(np.asarray(self.core._phi), physics_arrays['phi'], rtol=1e-6, atol=1e-10):
            raise ValueError('Derived coefficient formula differs from the declared physics')
        self.core._phi = mx.array(physics_arrays['phi'])
        self._physics_identity = physics['identity']
        self._anatomy_identity = anatomy_meta['identity']


class ReducedPolicy(PrototypePolicy):
    def __init__(self, model):
        if not isinstance(model, ReducedModel):
            raise ValueError('The separately configured V63 reduced model is required')
        super().__init__(model)
        self.identity = stable(dict(physics=model._physics_identity, anatomy=model._anatomy_identity,
            runtime=environment(), device=str(mx.default_device()),
            parameters={k: array_hash(v) for k, v in model_arrays(model).items()},
            sources={p: digest(ROOT/p) for p in ('snake_whole/reduced_model_v62.py', 'snake_whole/reduced_model_v63.py')}))

    def snapshot(self):
        return dict(super().snapshot(), model_identity=self.identity)

    def restore(self, saved):
        if set(saved) != {'neural_state', 'observations', 'simulated_ms', 'model_identity'} or saved['model_identity'] != self.identity:
            raise ValueError('Recurrent history belongs to another frozen model, anatomy or physics')
        super().restore({k: v for k, v in saved.items() if k != 'model_identity'})
