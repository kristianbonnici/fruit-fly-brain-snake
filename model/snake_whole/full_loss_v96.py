"""Mixed training-only CE and source-to-current KL; fixed-head feature gradients."""
import numpy as np


def numpy_loss_gradient(logits,targets,ce_mass,kl_mass,source):
    x=np.array(logits,dtype=np.float64,copy=True);s=np.array(source,dtype=np.float64,copy=True)
    if x.ndim!=2 or x.shape[1]!=3 or s.shape!=x.shape or targets.shape!=ce_mass.shape or targets.shape!=kl_mass.shape or targets.shape!=(len(x),):raise ValueError('Aligned three-action objective required')
    if not all(np.isfinite(v).all() for v in (x,s,ce_mass,kl_mass)) or np.any(ce_mass<0) or np.any(kl_mass<0) or np.any((targets<0)|(targets>2)):raise ValueError('Finite nonnegative weights and valid targets required')
    x-=x.max(axis=1,keepdims=True);logq=x-np.log(np.exp(x).sum(axis=1,keepdims=True))
    s-=s.max(axis=1,keepdims=True);logp=s-np.log(np.exp(s).sum(axis=1,keepdims=True));p=np.exp(logp);q=np.exp(logq)
    ce=-logq[np.arange(len(x)),targets];kl=np.sum(p*(logp-logq),axis=1)
    components=np.array([np.mean(ce*ce_mass)/8,np.mean(kl*kl_mass)/8],np.float64)
    teacher=q.copy();teacher[np.arange(len(x)),targets]-=1
    gradient=(teacher*ce_mass[:,None]+(q-p)*kl_mass[:,None])/(len(x)*8)
    return float(components.sum()),gradient,components


def components(logits,targets,ce_mass,kl_mass,source):
    import mlx.core as mx
    logq=logits-mx.logsumexp(logits,axis=1,keepdims=True)
    logp=source-mx.logsumexp(source,axis=1,keepdims=True);p=mx.stop_gradient(mx.exp(logp))
    ce=-mx.take_along_axis(logq,targets[:,None],axis=1).squeeze(1)
    kl=mx.sum(p*(mx.stop_gradient(logp)-logq),axis=1)
    return mx.stack([mx.mean(ce*ce_mass)/8,mx.mean(kl*kl_mass)/8])


def head_gradient(decoder,features,targets,ce_mass,kl_mass,source):
    import mlx.core as mx
    if targets.shape!=ce_mass.shape or targets.shape!=kl_mass.shape or targets.shape!=(features.shape[0],) or source.shape!=(features.shape[0],3):raise ValueError('Separate aligned objective labels and weights required')
    def objective(x):return mx.sum(components(decoder(x),targets,ce_mass,kl_mass,source))
    loss,gradient=mx.value_and_grad(objective)(features)
    parts=components(decoder(features),targets,ce_mass,kl_mass,source);mx.eval(loss,gradient,parts)
    return loss,gradient,parts
