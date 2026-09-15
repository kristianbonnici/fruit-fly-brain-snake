"""Conventional GRU diagnostic for the chronological pipeline; contains no anatomy."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .common import ROOT,stable,digest
from .comparison_archive_v59 import model_arrays,array_hash
from .comparison_contract_v59 import environment
from .gru_contract_v67 import precision

SPEC=dict(kind='conventional_GRU_pipeline_control',hidden_size=64,input_features=256,
    encoder='Two local3x3convolutions1->8->8 plus256per-cell8channel mixtures',
    decoder='Non-affine LayerNorm64 eps1e-5 ->Linear64to32 ->LeakyReLU.1 ->Linear32to3',
    recurrent_steps_per_image=1,game_tick_ms=100.,anatomical_neurons=0,anatomical_edges=0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,8,3,padding=1)
        self.conv2=nn.Conv2d(8,8,3,padding=1)
        self.sensory_weight=mx.full((256,8),1/np.sqrt(8),mx.float32)
        self.sensory_bias=mx.zeros(256,mx.float32)
        # Construct interfaces before the GRU to preserve their seeded values
        # relative to V66. The learned GRU is explicitly a comparison policy.
        self.decoder=nn.Sequential(nn.LayerNorm(64,eps=1e-5,affine=False),nn.Linear(64,32),
            nn.LeakyReLU(.1),nn.Linear(32,3))
        self.gru=nn.GRU(256,64)

    def encode(self,images):
        if images.ndim!=3 or images.shape[1:]!=(16,16):
            raise ValueError('Current16x16rendered images only')
        f=nn.relu(self.conv2(nn.relu(self.conv1(images[...,None])))).reshape(-1,256,8)
        return 3.*mx.tanh(mx.sum(f*self.sensory_weight,axis=-1)+self.sensory_bias)

    def sequence(self,images,hidden,reset,active):
        t,b=images.shape[:2]
        if (images.shape!=(t,b,16,16) or hidden.shape!=(b,64)
                or reset.shape!=(t,b) or active.shape!=(t,b)):
            raise ValueError('Aligned chronological images/masks and conventional hidden state required')
        x=self.encode(images.reshape(t*b,16,16)).reshape(t,b,256)
        projected=mx.addmm(self.gru.b,x,self.gru.Wx.T)
        states=[]
        for i in range(t):
            hidden=mx.where(reset[i,:,None],mx.zeros_like(hidden),hidden)
            recurrent=hidden@self.gru.Wh.T
            r,z=mx.split(mx.sigmoid(projected[i,:,:128]+recurrent[:,:128]),2,axis=-1)
            proposal=mx.tanh(projected[i,:,128:]+r*(recurrent[:,128:]+self.gru.bhn))
            updated=(1-z)*proposal+z*hidden
            hidden=mx.where(active[i,:,None],updated,hidden)
            states.append(hidden)
        states=mx.stack(states)
        logits=self.decoder(states.reshape(t*b,64)).reshape(t,b,3)
        return logits,hidden,states

    def __call__(self,images,hidden):
        b=len(images)
        logits,hidden,_=self.sequence(images[None],hidden,mx.zeros((1,b),mx.bool_),mx.ones((1,b),mx.bool_))
        return logits[0],hidden

    def capacity(self):
        count=lambda module:sum(v.size for _,v in tree_flatten(module.parameters()))
        return dict(encoder=count(self.conv1)+count(self.conv2)+self.sensory_weight.size+self.sensory_bias.size,
            decoder=count(self.decoder),recurrent=count(self.gru),total=count(self),anatomical_parameters=0)


class Policy:
    def __init__(self,model):
        precision()
        if not isinstance(model,Model):
            raise ValueError('Explicit conventional GRU comparison required')
        self.model=model
        self.model.eval()
        self.identity=stable(dict(spec=SPEC,runtime=environment(),precision=precision(),device=str(mx.default_device()),
            parameters={k:array_hash(v) for k,v in model_arrays(model).items()},
            source=digest(ROOT/'snake_whole/gru_model_v67.py')))
        self.reset()

    def reset(self):
        self.hidden=mx.zeros((1,64),mx.float32)
        self.observations=0

    def choose(self,image):
        image=np.asarray(image,np.float32)
        if image.shape!=(16,16) or not np.isfinite(image).all() or np.any((image<0)|(image>1)):
            raise ValueError('Finite current rendered image required')
        logits,self.hidden=self.model(mx.array(image[None]),self.hidden)
        mx.eval(logits,self.hidden)
        values=np.asarray(logits,np.float64)[0]
        if not np.isfinite(values).all() or not bool(mx.all(mx.isfinite(self.hidden)).item()):
            raise FloatingPointError('Nonfinite conventional GRU inference')
        self.observations+=1
        return int(values.argmax()),values.copy()

    def snapshot(self):
        return dict(hidden_state=np.asarray(self.hidden).copy(),observations=self.observations,
            simulated_ms=self.observations*100.,model_identity=self.identity)

    def restore(self,saved):
        h=np.asarray(saved['hidden_state'])
        if (set(saved)!={'hidden_state','observations','simulated_ms','model_identity'}
                or saved['model_identity']!=self.identity or h.shape!=(1,64) or h.dtype!=np.float32
                or not np.isfinite(h).all() or np.any(np.abs(h)>1.)
                or type(saved['observations']) is not int or saved['observations']<0
                or saved['simulated_ms']!=saved['observations']*100.):
            raise ValueError('Exact conventional policy identity, copied history and clock required')
        self.hidden=mx.array(h.copy())
        self.observations=saved['observations']
