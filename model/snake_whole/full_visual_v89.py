"""Frozen image interfaces feeding every measured pair of the full graded graph."""
import json
import time
import numpy as np
import mlx.core as mx
from .common import ROOT,digest,stable
from .comparison_archive_v59 import read,array_hash,model_arrays
from .anatomy import Graph
from .full_core_v88 import Layout,Core
from .fresh_contract_v74 import load_source
from .perception_symmetry_v84 import SymmetricPerception,load_encoder,CHECKPOINTS
from .perception_model_v80 import values


def load_layout():
    report=json.loads((ROOT/'data/snake-whole-v88/report.json').read_text())
    review=json.loads((ROOT/'data/snake-whole-v88/review.json').read_text())
    if (report['status']!='qualified_full_signed_sparse_solver' or review['binding']!=report['binding']
            or review['status']!='independently_verified_full_solver_artifacts'):
        raise ValueError('Qualified full solver and independently checked artifacts required')
    for version,artifact,expected in (
        (86,'embedding.npz','1bcbadd253a7e361ade143cf356f802b82d77893eec2845f67fad4186bf9249c'),
        (87,'physics.npz','471f9439f4f55a72176dc67cc414ccf777e07746c43de07342526346b126f08b')):
        if digest(ROOT/f'data/snake-whole-v{version}'/artifact)!=expected:raise ValueError('Exact full transfer artifact required')
    graph=Graph();embedding,em=read(ROOT/'data/snake-whole-v86/embedding.npz')
    physics,pm=read(ROOT/'data/snake-whole-v87/physics.npz')
    if em['full_anatomy_identity']!=graph.identity or pm['full_anatomy_identity']!=graph.identity:
        raise ValueError('Full measured anatomy must align')
    layout=Layout(graph,physics['phi'])
    if layout.identity!=report['layout_identity']:raise ValueError('Exact qualified full sparse layout required')
    return graph,layout,embedding,pm


class FullImagePolicy:
    def __init__(self,arm,loaded=None,record_substeps=True):
        if arm not in ('initial','transferred'):raise ValueError('Exact prospective efficacy conditions required')
        mx.set_default_device(mx.gpu)
        self.graph,self.layout,self.embedding,self.physics=load_layout() if loaded is None else loaded
        if self.layout.n!=165122 or self.layout.m!=25563197:raise ValueError('Complete selected network required')
        log_e=np.zeros(len(self.layout.eligible),np.float32)
        if arm=='transferred':log_e[self.embedding['transferred_eligible_slots']]=self.embedding['transferred_log_e']
        self.core=Core(self.layout,1,'gpu',log_e)
        # Only these source encoder/head modules execute. Its reduced core is
        # retained by the loader for provenance and never enters this path.
        self.interface,_=load_source('warm');self.interface.eval()
        encoder,_=load_encoder(60);self.visual=SymmetricPerception(encoder)
        self.sensory=mx.array(self.embedding['sensory']);self.output=mx.array(self.embedding['output'])
        self.perception_checkpoint_sha256=CHECKPOINTS[60];self.arm=arm
        interface={k:array_hash(v) for k,v in model_arrays(self.interface).items() if k!='model.core.log_e'}
        self.identity=stable(dict(version=89,arm=arm,core_identity=self.core.identity,
            log_e=array_hash(log_e),interface_parameters=interface,perception_checkpoint=CHECKPOINTS[60],
            anatomy=self.graph.identity,physics=self.physics['identity'],
            observation='Current rendered image and previous rendered image, blank only at game start',
            sources={p:digest(ROOT/p) for p in ('snake_whole/full_visual_v89.py','snake_whole/full_core_v88.py',
                'snake_whole/perception_symmetry_v84.py','snake_whole/structured_model_v69.py')}))
        self.record_substeps=record_substeps;self.reset()

    @property
    def state(self):return self.core.state

    @property
    def observations(self):return self.core.tick//10

    def reset(self):
        self.core.state=mx.zeros((1,self.layout.n),mx.float32);self.core.tick=0
        self.previous=np.zeros((16,16),np.float32)
        self.last_raw=self.last_views=self.last_sensors=self.last_drive=self.last_substeps=None

    def choose(self,image):
        image=np.asarray(image,np.float32)
        if image.shape!=(16,16) or not np.isfinite(image).all() or np.any((image<0)|(image>1)):
            raise ValueError('Only the finite current rendered image is accepted')
        pair=np.stack((self.previous,image),axis=-1)[None];t=time.perf_counter()
        raw,views=self.visual.predict(pair);sensor_values=np.asarray(values(raw))[0].copy()
        perception_seconds=time.perf_counter()-t;t=time.perf_counter()
        drive=self.interface.encode(mx.array(sensor_values[None]))
        current=mx.zeros((1,self.layout.n),mx.float32).at[:,self.sensory].add(drive);mx.eval(drive,current)
        sensory_seconds=time.perf_counter()-t;t=time.perf_counter()
        records=self.core.evolve(current,record=self.record_substeps)
        dynamics_seconds=time.perf_counter()-t;t=time.perf_counter()
        logits=self.interface.decoder(self.state[:,self.output]);mx.eval(logits)
        head_seconds=time.perf_counter()-t;t=time.perf_counter()
        output=np.asarray(logits,np.float64)[0].copy()
        if not np.isfinite(output).all() or not bool(mx.all(mx.isfinite(self.state)).item()):raise FloatingPointError('Finite full visual dynamics required')
        self.last_raw=np.asarray(raw)[0].copy();self.last_views=np.asarray(views)[:,0].copy()
        self.last_sensors=sensor_values;self.last_drive=np.asarray(drive)[0].copy()
        self.last_substeps=np.stack([np.asarray(r['state'])[0].copy() for r in records]) if self.record_substeps else None
        self.previous=image.copy()
        self.last_timing=dict(perception=perception_seconds,sensory=sensory_seconds,dynamics=dynamics_seconds,
            head=head_seconds,copy_and_checks=time.perf_counter()-t)
        return int(output.argmax()),output

    def snapshot(self):
        return dict(neural_state=np.asarray(self.state).copy(),previous_image=self.previous.copy(),
            observations=self.observations,simulated_ms=self.observations*100,policy_identity=self.identity)

    def restore(self,saved):
        if (set(saved)!={'neural_state','previous_image','observations','simulated_ms','policy_identity'}
                or saved['policy_identity']!=self.identity or type(saved['observations']) is not int
                or saved['observations']<0 or saved['simulated_ms']!=saved['observations']*100):
            raise ValueError('Exact full visual identity and 100 ms clock required')
        state=np.asarray(saved['neural_state']);previous=np.asarray(saved['previous_image'])
        if (state.shape!=(1,self.layout.n) or state.dtype!=np.float32 or not np.isfinite(state).all()
                or np.any(np.abs(state)>1) or previous.shape!=(16,16) or previous.dtype!=np.float32
                or not np.isfinite(previous).all() or np.any((previous<0)|(previous>1))
                or saved['observations']==0 and (np.any(previous) or np.any(state))):
            raise ValueError('Complete finite copied image and full neural history required')
        if self.core.updates!=0:raise ValueError('Inference restoration requires the frozen source efficacy')
        self.core.state=mx.array(state.copy());self.core.tick=saved['observations']*10;self.previous=previous.copy()
        self.last_raw=self.last_views=self.last_sensors=self.last_drive=self.last_substeps=None
