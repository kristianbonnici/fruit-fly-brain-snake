"""V103's three-term fitting loss for the existing small frozen-feature decoder."""
import numpy as np
from .full_decoder_v92 import Decoder as SourceDecoder, normalize, SHAPES, SOURCE_KEYS
from .full_loss_v103 import numpy_loss_gradient

OBJECTIVE_KEYS={'ce_mass','kl_mass','anchor_logits','legal_mass','legal_actions'}


class Decoder(SourceDecoder):
    # Inherited step/save/restore retain V92's qualified FP32 Adam arithmetic.
    # The mass argument is a separate fitting-objective bundle, never features.
    def gradient(self,x,targets,mass):
        if (x.ndim!=2 or x.shape[1]!=64 or x.dtype!=np.float32
                or not np.isfinite(x).all() or not isinstance(mass,dict)
                or set(mass)!=OBJECTIVE_KEYS):
            raise ValueError('Only finite64 neural features and a separate complete fitting objective')
        p=self.parameters
        pre=x@p['w1'].T+p['b1'];hidden=np.where(pre>=0,pre,np.float32(.1)*pre)
        z=hidden@p['w2'].T+p['b2']
        loss,error,_=numpy_loss_gradient(z,targets,mass['ce_mass'],mass['kl_mass'],
            mass['anchor_logits'],mass['legal_mass'],mass['legal_actions'])
        # V103 divided by eight time positions. These shuffled row batches have
        # no time axis, so undo that factor. Caller scales masses by fitting N.
        error*=8
        back=(error@p['w2'])*np.where(pre>=0,np.float32(1),np.float32(.1))
        grads=dict(w2=error.T@hidden,b2=error.sum(axis=0),w1=back.T@x,b1=back.sum(axis=0))
        return float(loss*8),{k:v.astype(np.float32) for k,v in grads.items()}
