"""Small existing output decoder, trained on frozen neural features in NumPy.

Nonaffine LayerNorm64 -> Linear64x32 -> LeakyReLU(.1) -> Linear32x3.
This module never receives game images or game state and never runs the core.
"""
import copy
import numpy as np
from .comparison_archive_v59 import read,save

SHAPES={'w1':(32,64),'b1':(32,),'w2':(3,32),'b2':(3,)}
SOURCE_KEYS={'w1':'model.decoder.layers.1.weight','b1':'model.decoder.layers.1.bias',
             'w2':'model.decoder.layers.3.weight','b2':'model.decoder.layers.3.bias'}


def normalize(features):
    x=np.asarray(features)
    if x.ndim!=2 or x.shape[1]!=64 or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Finite64 FP32 neural output features required')
    centered=x-x.mean(axis=1,keepdims=True)
    return centered/np.sqrt(np.mean(centered*centered,axis=1,keepdims=True)+np.float32(1e-5))


class Decoder:
    def __init__(self,parameters):
        if set(parameters)!=set(SHAPES):raise ValueError('Exact small decoder required')
        self.parameters={k:np.asarray(v).copy() for k,v in parameters.items()}
        for k,v in self.parameters.items():
            if v.shape!=SHAPES[k] or v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError('Exact2179 finite FP32 decoder parameters required')
        self.m={k:np.zeros_like(v) for k,v in self.parameters.items()};self.v=copy.deepcopy(self.m);self.steps=0

    def logits(self,x):
        p=self.parameters;hidden=x@p['w1'].T+p['b1'];hidden=np.where(hidden>=0,hidden,np.float32(.1)*hidden)
        return hidden@p['w2'].T+p['b2']

    def gradient(self,x,targets,mass):
        if (x.ndim!=2 or x.shape[1]!=64 or targets.shape!=(len(x),) or mass.shape!=(len(x),)
                or targets.dtype.kind not in 'iu' or np.any(targets<0) or np.any(targets>2)
                or not np.isfinite(x).all() or not np.isfinite(mass).all() or np.any(mass<0)):
            raise ValueError('Separate finite neural features, action labels and fitting mass required')
        mass=np.asarray(mass,np.float32)
        p=self.parameters;pre=x@p['w1'].T+p['b1'];hidden=np.where(pre>=0,pre,np.float32(.1)*pre)
        z=hidden@p['w2'].T+p['b2'];z=z-z.max(axis=1,keepdims=True);ex=np.exp(z);prob=ex/ex.sum(axis=1,keepdims=True)
        ce=np.log(ex.sum(axis=1))-z[np.arange(len(x)),targets]
        loss=np.mean(ce*mass)
        error=prob.copy();error[np.arange(len(x)),targets]-=1;error*=mass[:,None]/len(x)
        back=(error@p['w2'])*np.where(pre>=0,np.float32(1),np.float32(.1))
        gradients=dict(w2=error.T@hidden,b2=error.sum(axis=0),w1=back.T@x,b1=back.sum(axis=0))
        return float(loss),gradients

    def step(self,x,targets,mass):
        loss,grads=self.gradient(x,targets,mass);self.steps+=1
        for k,g in grads.items():
            self.m[k]=np.float32(.9)*self.m[k]+np.float32(.1)*g
            self.v[k]=np.float32(.999)*self.v[k]+np.float32(.001)*(g*g)
            delta=(self.m[k]/np.float32(1-.9**self.steps))/(np.sqrt(self.v[k]/np.float32(1-.999**self.steps))+np.float32(1e-8))
            self.parameters[k]-=np.float32(.001)*delta
            if not np.isfinite(self.parameters[k]).all() or not np.isfinite(self.m[k]).all() or not np.isfinite(self.v[k]).all():raise FloatingPointError('Finite decoder Adam state required')
        return loss

    def save(self,path,context,binding,replace=False):
        arrays={prefix+'.'+k:v.copy() for prefix,values in [('parameter',self.parameters),('m',self.m),('v',self.v)] for k,v in values.items()}
        return save(path,arrays,dict(format='frozen-full-decoder-fit-v92',binding=binding,context=copy.deepcopy(context),optimizer_steps=self.steps),replace=replace)

    def restore(self,path,expected_context,binding):
        arrays,meta=read(path,binding)
        if meta.get('format')!='frozen-full-decoder-fit-v92' or meta['context']!=expected_context or meta['optimizer_steps']!=expected_context['updates']:
            raise ValueError('Exact decoder source/schedule/cursor/RNG context required')
        if set(arrays)!={prefix+'.'+k for prefix in ('parameter','m','v') for k in SHAPES}:raise ValueError('Complete decoder and optimizer state required')
        for name,v in arrays.items():
            prefix,k=name.split('.')
            if v.shape!=SHAPES[k] or v.dtype!=np.float32 or not np.isfinite(v).all() or prefix=='v' and np.any(v<0):raise ValueError('Finite complete optimizer arrays required')
        self.parameters={k:arrays['parameter.'+k].copy() for k in SHAPES}
        self.m={k:arrays['m.'+k].copy() for k in SHAPES};self.v={k:arrays['v.'+k].copy() for k in SHAPES};self.steps=meta['optimizer_steps']
        return meta
