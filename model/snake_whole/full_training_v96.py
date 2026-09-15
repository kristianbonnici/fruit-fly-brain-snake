"""Unchanged full forward/adjoint path, with qualified mixed objective errors."""
import numpy as np
from .full_training_v90 import masked_evolve,masked_backward
from .full_loss_v96 import head_gradient


def forward_chunk(core,interface,chunk,sensory,output,training=True):
    import mlx.core as mx
    from .perception_model_v80 import values
    records=[];loss=0.;logits=[];states=[];objective_parts=np.zeros(2,np.float64)
    for index in range(8):
        # Cached17values are the frozen learned visual encoder output. The true
        # geometry sensors from the source teaching artifact are not present.
        predicted=np.asarray(values(mx.array(chunk['raw'][index]))).copy()
        drive=interface.encode(mx.array(predicted))
        current=mx.zeros((core.batch,core.n),mx.float32).at[:,sensory].add(drive);mx.eval(current)
        steps=masked_evolve(core,current,chunk['reset'][index],chunk['active'][index])
        features=core.state[:,output];value=interface.decoder(features);mx.eval(value);logits.append(value);states.append(core.state)
        if training:
            current_loss,error,parts=head_gradient(interface.decoder,features,mx.array(chunk['targets'][index]),mx.array(chunk['ce_mass'][index]),mx.array(chunk['kl_mass'][index]),mx.array(chunk['anchor_logits'][index]))
            objective_parts+=np.asarray(parts).astype(np.float64)
            steps[-1]['error']=mx.zeros_like(core.state).at[:,output].add(error);mx.eval(steps[-1]['error']);loss+=float(current_loss.item())
            records+=steps
    stacked=mx.stack(states);predictions=mx.stack(logits);mx.eval(stacked,predictions)
    return records,loss,predictions,stacked,objective_parts
