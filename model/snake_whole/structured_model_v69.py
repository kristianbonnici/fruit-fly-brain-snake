"""Structured-input anatomical control; the existing recurrent graph chooses actions."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from .common import ROOT,digest,stable
from .comparison_archive_v59 import model_arrays,array_hash
from .comparison_contract_v59 import environment
from .gru_contract_v67 import precision
from .reduced_model_v66 import ReducedModel as QualifiedTemplate,load_anatomy
from .reduced_model_v62 import ReducedPolicy as QualifiedState,parameter_count
from .structured_sensor_v69 import SPEC


class Model(nn.Module):
    def __init__(self,arrays,train_internal=True):
        super().__init__()
        # Reuse the exact qualified graph/physics/head construction. The template's
        # image encoder is discarded; it is absent from this model's parameters.
        template=QualifiedTemplate(arrays,train_internal)
        self.core,self.decoder=template.core,template.decoder
        self._physics_identity,self._anatomy_identity=template._physics_identity,template._anatomy_identity
        self.encoder=nn.Linear(5,256)

    def encode(self,sensors):
        if sensors.ndim!=2 or sensors.shape[1]!=5:raise ValueError('Five declared structured sensory values required')
        return 3.*mx.tanh(self.encoder(sensors))

    def __call__(self,sensors,state,matrix=None):
        state,_=self.core.evolve(self.encode(sensors),state,matrix)
        return self.decoder(state[:,self.core._output]),state

    def capacity(self):
        return dict(encoder=parameter_count(self.encoder),decoder=parameter_count(self.decoder),
            internal_efficacy=parameter_count(self.core),total=parameter_count(self),trainable=parameter_count(self,True))


class Policy(QualifiedState):
    def __init__(self,model):
        precision()
        if not isinstance(model,Model):raise ValueError('Explicit structured-input anatomical model required')
        super().__init__(model)
        self.identity=stable(dict(sensor=SPEC,physics=model._physics_identity,anatomy=model._anatomy_identity,
            runtime=environment(),device=str(mx.default_device()),precision=precision(),
            parameters={k:array_hash(v) for k,v in model_arrays(model).items()},
            sources={p:digest(ROOT/p) for p in ('snake_whole/structured_model_v69.py','snake_whole/structured_sensor_v69.py',
                'snake_whole/reduced_model_v62.py','snake_whole/reduced_model_v63.py','snake_whole/reduced_model_v66.py')}))

    def choose(self,sensors):
        sensors=np.asarray(sensors,np.float32)
        if (sensors.shape!=(5,) or not np.isfinite(sensors).all() or np.any(np.abs(sensors[:2])>1)
                or np.any((sensors[2:]<0)|(sensors[2:]>1))):raise ValueError('Finite declared structured sensor range required')
        logits,self.state=self.model(mx.array(sensors[None]),self.state,self.matrix);mx.eval(logits,self.state)
        values=np.asarray(logits,np.float64)[0]
        if not np.isfinite(values).all() or not bool(mx.all(mx.isfinite(self.state)).item()):raise FloatingPointError('Nonfinite structured anatomical inference')
        self.observations+=1;return int(values.argmax()),values.copy()

    def snapshot(self):return dict(super().snapshot(),model_identity=self.identity)

    def restore(self,saved):
        if set(saved)!={'neural_state','observations','simulated_ms','model_identity'} or saved['model_identity']!=self.identity:
            raise ValueError('Exact structured sensory/anatomical identity required')
        super().restore({k:v for k,v in saved.items() if k!='model_identity'})
