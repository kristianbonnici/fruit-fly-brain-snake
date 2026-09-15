"""Lossless incoming-order view of the frozen integer MaleCNS graph."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from conditioning.prepare import topology_hash, verify
from .common import ROOT, OUT, digest, stable, atomic_json

CACHE = OUT / 'anatomy'


def prepare():
    if (CACHE / 'manifest.json').exists():
        return validate()
    source = verify()
    original = json.loads((ROOT / 'data/manifest.json').read_text())
    if source['source_hashes'] != original['sha256']:
        raise ValueError('Anatomical source manifests disagree')
    with np.load(ROOT / 'data/conditioning/anatomy-v1.npz') as f:
        a = {k: f[k] for k in ('ids', 'indptr', 'indices', 'counts')}
    if topology_hash(a) != source['topology_hash']:
        raise ValueError('Integer topology hash mismatch')
    n = len(a['ids']); m = len(a['counts'])
    if (n, m, int(a['counts'].sum())) != (165122, 25563197, 124025046):
        raise ValueError('Full-network contract differs')
    if np.any(a['counts'] <= 0) or a['counts'].dtype.kind not in 'iu':
        raise ValueError('Anatomical counts must be positive integers')
    # Transpose edge indices, not weights. The mapping is a permutation of all
    # original pairs; counts are gathered from the immutable integer artifact.
    ordering = csr_matrix((np.arange(m, dtype=np.int32), a['indices'], a['indptr']),
                          shape=(n, n)).transpose().tocsr()
    order = ordering.data
    if not np.array_equal(np.sort(order), np.arange(m, dtype=np.int32)):
        raise ValueError('Incoming reordering lost or duplicated a pair')
    ann = pd.read_feather(ROOT / 'raw/annotations.feather')
    ann = ann[ann.status == 'Traced'].sort_values('bodyId').reset_index(drop=True)
    if not np.array_equal(ann.bodyId.to_numpy(), a['ids']):
        raise ValueError('Annotations no longer align with integer identities')
    labels = np.array(json.loads((ROOT/'data/conditioning/physiology-v1.json').read_text())['labels'])
    signs = np.where(np.isin(labels, ['gaba', 'glutamate', 'histamine']), -1, 1).astype(np.float32)
    # Only consensus acetylcholine, not predicted-only/uncertain labels, learns.
    nt = pd.read_feather(ROOT/'raw/neurotransmitters.feather').set_index('body').reindex(a['ids'])
    confident = (nt.consensus_nt.fillna('').to_numpy() == 'acetylcholine')
    pre = ordering.indices.astype(np.int32)
    post = np.repeat(np.arange(n, dtype=np.int32), np.diff(ordering.indptr))
    classes, population = np.unique(ann.superclass.fillna('unclassified').astype(str), return_inverse=True)
    with np.load(ROOT/'data/interfaces.npz') as f:
        interface = {k: f[k] for k in f.files}
    sensory = interface['sensory']; output = interface['readout']
    if np.intersect1d(sensory, output).size:
        raise ValueError('Sensory/output overlap')
    # Reproducible balanced assignment within each annotated motor population.
    # No soma side, game outcomes, targets, or trained readout is used.
    decoder = np.zeros((n, 3), np.float32)
    assignments = []
    for cls in ('descending_neuron', 'vnc_motor', 'cb_motor'):
        members = np.flatnonzero(ann.superclass.to_numpy() == cls)
        members = sorted(members, key=lambda i: stable(['whole-output-v1', int(a['ids'][i])]))
        for action in range(3):
            group = np.array(members[action::3], dtype=np.int32)
            decoder[group, action] = 1 / np.sqrt(len(group))
            assignments += [dict(index=int(i), body_id=int(a['ids'][i]), population=cls,
                                 action=['left','straight','right'][action]) for i in group]
    # Centering removes a common motor activation component; remains fixed.
    decoder -= decoder.mean(axis=1, keepdims=True)
    arrays = dict(ids=a['ids'], ptr=ordering.indptr.astype(np.int32), pre=pre, post=post,
                  counts=a['counts'][order], original_edge=order,
                  signs=signs, plastic=confident[pre].astype(np.uint8),
                  population=population.astype(np.int32), sensory=sensory,
                  xy=interface['retina_xy'], output=output, decoder=decoder,
                  graded=np.isin(ann.type.fillna('').to_numpy(), ['L1','L2','L3','APL']).astype(np.uint8),
                  apl=np.flatnonzero(ann.type.to_numpy() == 'APL').astype(np.int32))
    CACHE.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(CACHE / (k+'.npy'), v, allow_pickle=False)
    table = []
    for code, name in enumerate(classes):
        idx = np.flatnonzero(population == code)
        table.append(dict(name=str(name), code=code, neurons=len(idx),
                          incoming=int(np.sum(population[post] == code)),
                          plastic_incoming=int(np.sum((population[post] == code) & confident[pre]))))
    manifest = dict(version='whole-incoming-anatomy-v1', source_manifest=source,
        neurons=n, edges=m, synapses=int(a['counts'].sum()),
        topology_hash=source['topology_hash'], populations=table,
        plastic_edges=int(confident[pre].sum()),
        learning_mask='Every existing pair from a consensus-acetylcholine presynaptic neuron',
        uncertain_channels='All pairs retained; inherited transmitter sign, fixed efficacy for non-consensus ACh',
        files={p.name: digest(p) for p in sorted(CACHE.glob('*.npy'))},
        output_assignment=assignments,
        output_interpretation='Balanced hashed engineering assignment within annotated populations; '
                              'no claim of biological left/straight/right identity')
    manifest['identity'] = stable(manifest)
    atomic_json(CACHE/'manifest.json', manifest)
    return manifest


def validate():
    m = json.loads((CACHE/'manifest.json').read_text())
    copy = dict(m); identity = copy.pop('identity')
    if stable(copy) != identity:
        raise ValueError('Anatomical manifest identity mismatch')
    for name, expected in m['files'].items():
        if digest(CACHE/name) != expected:
            raise ValueError('Changed anatomical/interface artifact: '+name)
    return m


class Graph:
    def __init__(self, check=True):
        self.manifest = validate() if check else json.loads((CACHE/'manifest.json').read_text())
        self.identity = self.manifest['identity']
        for name in self.manifest['files']:
            setattr(self, Path(name).stem, np.load(CACHE/name, mmap_mode='r'))
        self.n = len(self.ids); self.m = len(self.counts)


if __name__ == '__main__':
    m = prepare()
    print(json.dumps({k:v for k,v in m.items() if k not in ('output_assignment','files')}, indent=2))
