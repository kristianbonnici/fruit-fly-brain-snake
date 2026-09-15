"""Small anatomy-constrained graded-deviation RNN; no teacher or game-state input."""
import json

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .common import ROOT, digest
from .comparison_archive_v59 import read


def load_anatomy():
    base = ROOT/'data/snake-whole-v62'
    report = json.loads((base/'selection-report.json').read_text())
    path = base/'reduced-anatomy.npz'
    if digest(path) != report['artifact_sha256']:
        raise ValueError('Exact frozen induced anatomical artifact required')
    arrays, meta = read(path, digest(base/'selection-contract.json'))
    if meta['format'] != 'reduced-induced-anatomy-v62':
        raise ValueError('Explicit reduced development anatomy required')
    return arrays, meta


def parameter_count(module, trainable=False):
    values = module.trainable_parameters() if trainable else module.parameters()
    return sum(v.size for _, v in tree_flatten(values))


class AnatomicalCore(nn.Module):
    """Every induced pair transmits a count/sign/physiology/efficacy product."""
    def __init__(self, arrays, train_internal=True, gain=.8, dt_ms=10., tau_ms=20., steps=10):
        super().__init__()
        n = len(arrays['ids'])
        pre, post, counts = (np.asarray(arrays[k]) for k in ('pre', 'post', 'counts'))
        signs = np.asarray(arrays['signs'])
        plastic = np.flatnonzero(arrays['plastic'])
        sensory, output = np.asarray(arrays['sensory']), np.asarray(arrays['output'])
        if (pre.shape != post.shape or pre.shape != counts.shape or counts.dtype.kind not in 'iu'
                or np.any(counts <= 0) or signs.shape != (n,) or np.any(~np.isin(signs, [-1., 1.]))
                or np.any(pre < 0) or np.any(pre >= n) or np.any(post < 0) or np.any(post >= n)
                or len(np.unique(pre.astype(np.int64)*n+post)) != len(pre)
                or np.any(signs[pre[plastic]] != 1) or not 0 < dt_ms/tau_ms <= 1 or gain <= 0
                or type(steps) is not int or steps < 1):
            raise ValueError('Positive integer counts, fixed signs and stable convex integration required')
        if (len(np.unique(sensory)) != len(sensory) or len(np.unique(output)) != len(output)
                or np.any(sensory < 0) or np.any(sensory >= n) or np.any(output < 0) or np.any(output >= n)
                or np.intersect1d(sensory, output).size):
            raise ValueError('Distinct declared sensory and output neurons required')
        self.n, self.edges, self.sensory_count = n, len(counts), len(sensory)
        self.alpha, self.steps, self.tick_ms = dt_ms/tau_ms, steps, steps*dt_ms
        # Leading underscores make fixed arrays non-parameters in MLX Module.
        self._pre, self._post = mx.array(pre.astype(np.int32)), mx.array(post.astype(np.int32))
        self._sensory, self._output = mx.array(sensory.astype(np.int32)), mx.array(output.astype(np.int32))
        self._plastic = mx.array(plastic.astype(np.int32))
        norm = np.sqrt(np.bincount(post, weights=counts.astype(np.float64)**2, minlength=n))
        self._counts = mx.array(counts.astype(np.float32))
        self._phi = mx.array((gain*signs[pre]/np.maximum(norm[post], 1.)).astype(np.float32))
        self.log_e = mx.zeros(len(plastic), mx.float32)
        if not train_internal:
            self.freeze(recurse=False, keys='log_e', strict=True)

    def efficacy(self):
        learned = mx.exp(mx.clip(self.log_e, float(np.log(.05)), float(np.log(4.))))
        return mx.ones(self.edges, mx.float32).at[self._plastic].add(learned-1.)

    def coefficients(self):
        return self._counts*self._phi*self.efficacy()

    def matrix(self):
        return mx.zeros((self.n, self.n), mx.float32).at[self._post, self._pre].add(self.coefficients())

    def project_efficacy(self):
        self.log_e = mx.clip(self.log_e, float(np.log(.05)), float(np.log(4.)))

    def evolve(self, drive, state, matrix=None, record=False):
        if state.ndim != 2 or state.shape[1] != self.n or drive.shape != (state.shape[0], self.sensory_count):
            raise ValueError('Declared sensory currents and complete recurrent state required')
        weights = self.matrix() if matrix is None else matrix
        current = mx.zeros(state.shape, mx.float32).at[:, self._sensory].add(drive)
        recorded = []
        for _ in range(self.steps):
            state = (1-self.alpha)*state+self.alpha*mx.tanh(state@weights.T+current)
            if record:
                recorded.append(state)
        return state, mx.stack(recorded, axis=1) if record else None


class ReducedModel(nn.Module):
    """Current pixels -> local learned sensory currents -> recurrent graph -> logits."""
    def __init__(self, arrays, train_internal=True):
        super().__init__()
        if len(arrays['sensory']) != 256 or len(arrays['output']) != 64:
            raise ValueError('Frozen V62 screen assignment and output capacity required')
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.sensory_weight = mx.full((256, 8), 1/np.sqrt(8), mx.float32)
        self.sensory_bias = mx.zeros(256, mx.float32)
        self.core = AnatomicalCore(arrays, train_internal)
        self.decoder = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

    def encode(self, images):
        if images.ndim != 3 or images.shape[1:] != (16, 16):
            raise ValueError('Only current rendered16x16images enter the visual encoder')
        features = nn.relu(self.conv2(nn.relu(self.conv1(images[..., None])))).reshape(-1, 256, 8)
        # Frozen sensory order is row-major screen order, not anatomical node order.
        return 3.*mx.tanh(mx.sum(features*self.sensory_weight, axis=-1)+self.sensory_bias)

    def __call__(self, images, state, matrix=None):
        state, _ = self.core.evolve(self.encode(images), state, matrix)
        return self.decoder(state[:, self.core._output]), state

    def capacity(self):
        encoder = parameter_count(self.conv1)+parameter_count(self.conv2)+self.sensory_weight.size+self.sensory_bias.size
        return dict(encoder=encoder, decoder=parameter_count(self.decoder), internal_efficacy=parameter_count(self.core),
                    total=parameter_count(self), trainable=parameter_count(self, True))


class ReducedPolicy:
    """Image-only inference; dynamic state lives in actual selected neural neurons."""
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Static materialization of immutable graph coefficients, never a game cache.
        self.matrix = model.core.matrix()
        mx.eval(self.matrix)
        self.reset()

    def reset(self):
        self.state = mx.zeros((1, self.model.core.n), mx.float32)
        self.observations = 0

    def choose(self, image):
        image = np.asarray(image, np.float32)
        if image.shape != (16, 16) or not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
            raise ValueError('A finite rendered standard-board image is required')
        logits, self.state = self.model(mx.array(image[None]), self.state, self.matrix)
        mx.eval(logits, self.state)
        values = np.asarray(logits, np.float64)[0]
        if not np.isfinite(values).all() or not bool(mx.all(mx.isfinite(self.state)).item()):
            raise FloatingPointError('Nonfinite reduced anatomical inference')
        self.observations += 1
        return int(values.argmax()), values.copy()

    def snapshot(self):
        return dict(neural_state=np.asarray(self.state).copy(), observations=self.observations,
                    simulated_ms=self.observations*self.model.core.tick_ms)

    def restore(self, saved):
        state = np.asarray(saved['neural_state'])
        if (set(saved) != {'neural_state', 'observations', 'simulated_ms'} or state.shape != (1, self.model.core.n)
                or state.dtype != np.float32 or not np.isfinite(state).all() or np.any(np.abs(state) > 1.)
                or type(saved['observations']) is not int or saved['observations'] < 0
                or saved['simulated_ms'] != saved['observations']*self.model.core.tick_ms):
            raise ValueError('Complete compatible graded neural state and clock required')
        self.state = mx.array(state.copy())
        self.observations = saved['observations']
