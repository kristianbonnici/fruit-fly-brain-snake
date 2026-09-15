"""Small atomic comparison archives, usable without the teacher or trainer."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .common import stable


def array_hash(value):
    a = np.asarray(value)
    return stable(dict(shape=list(a.shape), dtype=str(a.dtype), bytes=hashlib.sha256(a.tobytes()).hexdigest()))


def save(path, arrays, metadata, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError('Preserve completed comparison archive')
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata, array_hashes={k: array_hash(v) for k, v in arrays.items()})
    meta['identity'] = stable(meta)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez(f, **arrays, metadata=np.array(json.dumps(meta, allow_nan=False)))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return meta


def read(path, binding=None):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f['metadata']))
        arrays = {k: f[k].copy() for k in f.files if k != 'metadata'}
    identity = meta.pop('identity')
    if (stable(meta) != identity or meta['array_hashes'] != {k: array_hash(v) for k, v in arrays.items()}
            or binding is not None and meta.get('binding') != binding
            or any(v.dtype.kind == 'f' and not np.isfinite(v).all() for v in arrays.values())):
        raise ValueError('Unchanged compatible finite comparison archive required')
    meta['identity'] = identity
    return arrays, meta


def model_arrays(model, optimizer=None):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.eval(model.parameters())
    arrays = {'model.'+k: np.asarray(v).copy() for k, v in tree_flatten(model.parameters())}
    if optimizer is not None:
        mx.eval(optimizer.state, mx.random.state)
        arrays.update({'optimizer.'+k: np.asarray(v).copy() for k, v in tree_flatten(optimizer.state)})
        arrays.update({'random.'+str(i): np.asarray(v).copy() for i, v in enumerate(mx.random.state)})
    return arrays


def install(model, arrays, optimizer=None):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    expected = dict(tree_flatten(model.parameters()))
    incoming = {k[6:]: v for k, v in arrays.items() if k.startswith('model.')}
    if (set(expected) != set(incoming)
            or any(v.shape != expected[k].shape or v.dtype != np.dtype(str(expected[k].dtype).split('.')[-1]) for k, v in incoming.items())):
        raise ValueError('Exact comparison parameter names, shapes and dtypes required')
    model.load_weights([(k, mx.array(v)) for k, v in incoming.items()])
    if optimizer is not None:
        state = [(k[10:], mx.array(v)) for k, v in arrays.items() if k.startswith('optimizer.')]
        random = [arrays['random.'+str(i)] for i in range(len(mx.random.state))]
        if not state or any(v.dtype != np.uint32 or v.shape != (2,) for v in random):
            raise ValueError('Complete optimizer and MLX RNG state required')
        # These models have no stochastic layers. MLX RNG is used only during
        # the caller's seeded initialization; regenerate and verify that state.
        # MLX 0.32 exposes the state as read-only, so do not pretend to assign it.
        if any(not np.array_equal(saved, np.asarray(current)) for saved, current in zip(random, mx.random.state)):
            raise ValueError('Recreate the registered seeded model initialization before restoring optimizer state')
        optimizer.state = tree_unflatten(state)
    mx.eval(model.parameters())


def load_model(path, binding):
    from .comparison_policy_v59 import make_model, parameter_count
    arrays, meta = read(path, binding)
    if meta['format'] != 'comparison-training-epoch-v59':
        raise ValueError('A frozen main comparison checkpoint is required')
    model = make_model(meta['kind'])
    install(model, arrays)
    if parameter_count(model) != meta['parameter_count']:
        raise ValueError('Declared comparison capacity differs')
    model.eval()
    return model, meta
