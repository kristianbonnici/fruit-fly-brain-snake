"""Training-only masked full-graph TBPTT, with fixed learned visual interfaces."""
import numpy as np


def masked_evolve(core,current,reset,active):
    reset=np.asarray(reset);active=np.asarray(active)
    if (reset.shape!=(core.batch,) or active.shape!=reset.shape or reset.dtype!=bool or active.dtype!=bool
            or np.any(reset & ~active)):
        raise ValueError('Aligned active slots and real episode starts required')
    xp=core.xp;q=xp.array(active[:,None],dtype=xp.float32);keep=xp.array((~reset)[:,None],dtype=xp.float32)
    core.state=core.state*keep;records=[]
    for step in range(10):
        previous=core.state;record=core.step(current,True)
        core.state=xp.where(xp.array(active[:,None]),core.state,previous)
        record.update(state=core.state,active=q)
        if step==0:record['keep']=keep
        records.append(record)
    core.sync(core.state);return records


def masked_backward(core,records,terminal=None,return_inputs=False):
    xp=core.xp;adj=xp.zeros_like(core.state) if terminal is None else xp.array(terminal,dtype=xp.float32)
    gradient=xp.zeros_like(core.log_e);inputs=[]
    if core.backend=='gpu':
        outgoing=xp.contiguous(core.weight[core.outedge]);eligible=xp.contiguous(core.weight[core.eligible]);core.sync(outgoing,eligible)
    for record in reversed(records):
        if record['error'] is not None:adj=adj+record['error']
        q=record['active'];drive=.5*q*(1-record['activation']*record['activation'])*adj
        if return_inputs:inputs.append(drive)
        if core.backend=='gpu':
            if core.p:
                gradient=core.gradient_kernel(inputs=[core.epre,core.epost,eligible,record['previous'],drive,gradient],
                    template=[('N',core.n),('B',core.batch),('P',core.p)],grid=(core.p,1,1),threadgroup=(256,1,1),
                    output_shapes=[(core.p,)],output_dtypes=[xp.float32])[0]
            propagated=core.transpose_kernel(inputs=[core.outptr,core.outpost,outgoing,drive],template=[('N',core.n)],
                grid=(core.n*32,1,1),threadgroup=(256,1,1),output_shapes=[core.state.shape],output_dtypes=[xp.float32])[0]
        else:
            edge=core.layout.eligible
            for start in range(0,core.p,500_000):
                e=edge[start:start+500_000]
                gradient[start:start+len(e)]+=core.weight[e]*np.sum(record['previous'][:,core.layout.pre[e]]*drive[:,core.layout.post[e]],axis=0)
            propagated=(core.matrix.T@drive.T).T
        adj=((1-.5*q)*adj+propagated)*record['keep']
    core.sync(gradient,adj,*inputs)
    return gradient,adj,list(reversed(inputs))


def weighted_head_gradient(decoder,features,targets,mass):
    import mlx.core as mx
    import mlx.nn as nn
    if targets.shape!=mass.shape or targets.shape!=(features.shape[0],):raise ValueError('Separate aligned targets and loss weights required')
    def objective(x):return mx.mean(nn.losses.cross_entropy(decoder(x),targets,reduction='none')*mass)/8
    loss,gradient=mx.value_and_grad(objective)(features);mx.eval(loss,gradient)
    return loss,gradient


def forward_chunk(core,interface,chunk,sensory,output,training=True):
    import mlx.core as mx
    from .perception_model_v80 import values
    records=[];loss=0.;logits=[];states=[]
    for index in range(8):
        # Cached17values are the frozen learned visual encoder output. The true
        # geometry sensors from the source teaching artifact are not present.
        predicted=np.asarray(values(mx.array(chunk['raw'][index]))).copy()
        drive=interface.encode(mx.array(predicted))
        current=mx.zeros((core.batch,core.n),mx.float32).at[:,sensory].add(drive);mx.eval(current)
        steps=masked_evolve(core,current,chunk['reset'][index],chunk['active'][index])
        features=core.state[:,output];value=interface.decoder(features);mx.eval(value);logits.append(value);states.append(core.state)
        if training:
            current_loss,error=weighted_head_gradient(interface.decoder,features,mx.array(chunk['targets'][index]),mx.array(chunk['mass'][index]))
            steps[-1]['error']=mx.zeros_like(core.state).at[:,output].add(error);mx.eval(steps[-1]['error']);loss+=float(current_loss.item())
            records+=steps
    stacked=mx.stack(states);predictions=mx.stack(logits);mx.eval(stacked,predictions)
    return records,loss,predictions,stacked
