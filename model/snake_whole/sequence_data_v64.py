"""Whole-episode chronological packing; training metadata never enters the encoder."""
import heapq

import numpy as np

from .common import ROOT, digest
from .comparison_archive_v59 import read, array_hash

INPUT_SHA = '8bba9508ea87e06625fd20fd1f7eaabd00af4236eca9d62ab76f2a1e20afe077'
INPUT_ID = 'f22c2042e47dbf89d671924c557c11f6515c8dfc7be226e5f5b55eda665d6283'


def episode_rows(data, fitting=True):
    n = len(data['episode'])
    if (any(data[k].shape != (n,) for k in ('moves', 'targets', 'valid', 'fitting', 'mass'))
            or data['pixels'].shape != (n, 16, 16, 2)
            or data['fitting'].dtype != bool or data['valid'].dtype != bool
            or np.any(data['mass'][~(data['fitting'] & data['valid'])] != 0)
            or np.any(data['mass'] < 0) or not np.isfinite(data['mass']).all()
            or not np.isclose(data['mass'].sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('Aligned chronological data and normalized fitting-only mass required')
    result = {}
    for episode in np.unique(data['episode']):
        rows = np.flatnonzero(data['episode'] == episode)
        split = data['fitting'][rows]
        rows = rows[np.argsort(data['moves'][rows], kind='stable')]
        if np.any(split != split[0]) or not np.array_equal(data['moves'][rows], np.arange(len(rows))):
            raise ValueError('Each whole episode must have one split and consecutive moves from zero')
        if fitting is None or bool(split[0]) == fitting:
            result[int(episode)] = rows.astype(np.int32)
    if not result:
        raise ValueError('At least one complete episode required')
    return result


def load_inputs():
    path = ROOT/'data/snake-whole-v61/inputs.npz'
    if digest(path) != INPUT_SHA:
        raise ValueError('Exact verified V61 image corpus required')
    data, meta = read(path)
    if meta['identity'] != INPUT_ID:
        raise ValueError('Unchanged source episode identities required')
    episode_rows(data)
    return data, meta


def pack(episodes, order, episode_symmetry, batch, chunk):
    """Assign the next episode to the earliest available lane, ties by lane ID."""
    order, episode_symmetry = np.asarray(order), np.asarray(episode_symmetry)
    if (type(batch) is not int or type(chunk) is not int or min(batch, chunk) < 1
            or order.ndim != 1 or order.dtype != np.int32 or episode_symmetry.shape != order.shape
            or episode_symmetry.dtype != np.int8 or np.any(~np.isin(episode_symmetry, np.arange(8)))
            or len(order) != len(episodes) or set(order.tolist()) != set(episodes)):
        raise ValueError('A complete permutation, per-episode symmetry and positive packing dimensions required')
    lanes = [(0, lane) for lane in range(batch)]
    heapq.heapify(lanes)
    assigned = []
    for episode, symmetry in zip(order, episode_symmetry):
        start, lane = heapq.heappop(lanes)
        rows = episodes[int(episode)]
        assigned.append((start, lane, rows, symmetry))
        heapq.heappush(lanes, (start+len(rows), lane))
    time = ((max(end for end, _ in lanes)+chunk-1)//chunk)*chunk
    rows = np.full((time, batch), -1, np.int32)
    reset = np.zeros((time, batch), bool)
    symmetry = np.zeros((time, batch), np.int8)
    for start, lane, selected, sym in assigned:
        rows[start:start+len(selected), lane] = selected
        reset[start, lane] = True
        symmetry[start:start+len(selected), lane] = sym
    return dict(rows=rows, reset=reset, symmetry=symmetry,
                order=order.copy(), episode_symmetry=episode_symmetry.copy())


def schedule(data, batch, chunk, rng=None, fitting=True, augment=True):
    episodes = episode_rows(data, fitting)
    order = np.array(sorted(episodes), np.int32)
    if rng is not None:
        order = rng.permutation(order)
    if augment and rng is None:
        raise ValueError('Explicit saved RNG required for whole-episode augmentation')
    symmetry = rng.integers(0, 8, size=len(order), dtype=np.int8) if augment else np.zeros(len(order), np.int8)
    return pack(episodes, order, symmetry, batch, chunk)


def verify_schedule(data, saved, batch, chunk, fitting=True):
    if set(saved) != {'rows', 'reset', 'symmetry', 'order', 'episode_symmetry'}:
        raise ValueError('Complete packed schedule required')
    expected = pack(episode_rows(data, fitting), saved['order'], saved['episode_symmetry'], batch, chunk)
    if any(array_hash(saved[k]) != array_hash(expected[k]) for k in expected):
        raise ValueError('Packed chronology, reset flags or symmetry changed')


def get_chunk(data, packed, cursor, length):
    if type(cursor) is not int or type(length) is not int or length < 1 or cursor < 0 or cursor % length or cursor+length > len(packed['rows']):
        raise ValueError('One complete aligned chronological chunk required')
    rows = packed['rows'][cursor:cursor+length]
    active = rows >= 0
    images = np.zeros((*rows.shape, 16, 16), np.float32)
    targets = np.zeros(rows.shape, np.int32)
    mass = np.zeros(rows.shape, np.float32)
    # Read only actual active current-image rows. No held-out or row-zero filler.
    images[active] = data['pixels'][rows[active], ..., 1]
    targets[active] = data['targets'][rows[active]]
    fitting_count = int(np.sum(data['fitting'] & data['valid']))
    mass[active] = data['mass'][rows[active]]*fitting_count
    symmetries = packed['symmetry'][cursor:cursor+length]
    for sym in range(8):
        where = active & (symmetries == sym)
        changed = np.rot90(images[where], sym % 4, axes=(1, 2))
        if sym >= 4:
            changed = changed[:, :, ::-1]
            targets[where] = 2-targets[where]
        images[where] = changed
    return dict(images=images, targets=targets, mass=mass,
                reset=packed['reset'][cursor:cursor+length].copy(), active=active.copy(), rows=rows.copy())
