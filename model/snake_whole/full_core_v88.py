"""All-edge signed graded dynamics and explicit discrete TBPTT adjoint.

The reverse pass is optimizer machinery. Only measured forward edges transmit;
only declared eligible log-efficacies receive updates. No dense full matrix.
"""
import importlib.metadata
import platform
import numpy as np
from scipy.sparse import csr_matrix
from .common import ROOT,digest,stable
from .comparison_archive_v59 import array_hash

LOW=np.float32(np.log(.05));HIGH=np.float32(np.log(4.))


class Layout:
    def __init__(self,graph,phi):
        self.n=graph.n;self.m=graph.m
        self.ptr=np.asarray(graph.ptr,np.int32);self.pre=np.asarray(graph.pre,np.int32)
        self.post=np.asarray(graph.post,np.int32);self.counts=np.asarray(graph.counts)
        self.phi=np.asarray(phi,np.float32);self.eligible=np.flatnonzero(graph.plastic).astype(np.int32)
        if (self.ptr.shape!=(self.n+1,) or self.ptr[0]!=0 or self.ptr[-1]!=self.m
                or np.any(np.diff(self.ptr)<0) or self.pre.shape!=(self.m,) or self.post.shape!=(self.m,)
                or self.counts.shape!=(self.m,) or self.counts.dtype.kind not in 'iu'
                or np.any(self.counts<=0) or np.any(self.pre<0) or np.any(self.pre>=self.n)
                or not np.array_equal(self.post,np.repeat(np.arange(self.n),np.diff(self.ptr)))
                or self.phi.shape!=(self.m,) or not np.isfinite(self.phi).all() or np.any(self.phi==0)
                or not np.array_equal(np.sign(self.phi),graph.signs[self.pre])
                or np.any(graph.signs[self.pre[self.eligible]]!=1)):
            raise ValueError('Complete count/sign/physiology/eligible CSR contract required')
        same=self.post[1:]==self.post[:-1]
        if np.any(same & (self.pre[1:]<=self.pre[:-1])):raise ValueError('Unique sorted incoming pairs required')
        outgoing=csr_matrix((np.arange(self.m,dtype=np.int32),self.pre,self.ptr),shape=(self.n,self.n)).T.tocsr()
        self.outptr=outgoing.indptr.astype(np.int32);self.outpost=outgoing.indices.astype(np.int32)
        self.outedge=outgoing.data.astype(np.int32)
        self.base=self.counts.astype(np.float32)*self.phi
        if not np.isfinite(self.base).all() or np.any(self.base==0):raise ValueError('Every pair must transmit')
        self.identity=stable({k:array_hash(getattr(self,k)) for k in ('ptr','pre','post','counts','phi','eligible')})


class Core:
    """Owns persistent neural state, efficacy and projected Adam state."""
    def __init__(self,layout,batch=1,backend='cpu',log_e=None):
        if batch not in (1,2,4) or backend not in ('cpu','gpu'):raise ValueError('Bounded supported backend/batch required')
        self.layout=layout;self.n=layout.n;self.p=len(layout.eligible);self.batch=batch;self.backend=backend
        if backend=='gpu':
            import mlx.core as mx
            from .gru_contract_v67 import precision
            precision();mx.set_default_device(mx.gpu);self.xp=mx
        else:self.xp=np
        xp=self.xp
        runtime={name:importlib.metadata.version(name) for name in ('numpy','scipy')+(('mlx',) if backend=='gpu' else ())}
        self.identity=stable(dict(layout=layout.identity,backend=backend,batch=batch,runtime=runtime,python=platform.python_version(),
            source=digest(ROOT/'snake_whole/full_core_v88.py'),equation='half-old-plus-half-tanh',dt_ms=10))
        self.log_e=xp.zeros(self.p,xp.float32) if log_e is None else xp.array(log_e,dtype=xp.float32)
        if self.log_e.shape!=(self.p,) or not np.all(np.isfinite(np.asarray(self.log_e))) or np.any(np.asarray(self.log_e)<LOW) or np.any(np.asarray(self.log_e)>HIGH):
            raise ValueError('Finite bounded eligible log efficacy required')
        self.m=xp.zeros_like(self.log_e);self.v=xp.zeros_like(self.log_e);self.updates=0
        self.state=xp.zeros((batch,self.n),xp.float32);self.tick=0
        if backend=='gpu':
            for k in ('ptr','pre','outptr','outpost','outedge','eligible'):
                setattr(self,k,xp.array(getattr(layout,k)))
            self.epre=xp.array(layout.pre[layout.eligible]);self.epost=xp.array(layout.post[layout.eligible])
            self.base=xp.array(layout.base);self._kernels()
        self.materialize()

    def sync(self,*values):
        if self.backend=='gpu':self.xp.eval(*values)

    def materialize(self):
        xp=self.xp
        if self.backend=='gpu':
            efficacy=xp.ones(self.layout.m,xp.float32).at[self.eligible].add(xp.exp(self.log_e)-1.)
            self.weight=self.base*efficacy;self.sync(self.weight)
        else:
            self.weight=self.layout.base.copy();self.weight[self.layout.eligible]*=np.exp(self.log_e)
            self.matrix=csr_matrix((self.weight,self.layout.pre,self.layout.ptr),shape=(self.n,self.n))

    def _kernels(self):
        xp=self.xp;b=self.batch
        declarations='\n'.join(f'float sum{j}=0.0f;' for j in range(b))
        products='\n'.join(f'sum{j}+=w*h[{j}*N+pre[e]];' for j in range(b))
        reduction='\n'.join(f'sum{j}=simd_sum(sum{j});' for j in range(b))
        write='\n'.join(f'{{ uint f={j}*N+i;float t=metal::tanh(sum{j}+input[f]);activation[f]=t;next_h[f]=0.5f*h[f]+0.5f*t; }}' for j in range(b))
        self.forward_kernel=xp.fast.metal_kernel(name=f'signed_graded_forward_v88_b{b}',
            input_names=['ptr','pre','weight','h','input'],output_names=['next_h','activation'],source=f'''
            uint i=thread_position_in_grid.x/32,lane=thread_index_in_simdgroup;
            if(i>=N)return;
            {declarations}
            for(int e=ptr[i]+lane;e<ptr[i+1];e+=32){{float w=weight[e];{products}}}
            {reduction}
            if(lane==0){{{write}}}
            ''')
        transpose_product='\n'.join(f'sum{j}+=weight[e]*drive[{j}*N+post[e]];' for j in range(b))
        transpose_write='\n'.join(f'out[{j}*N+i]=sum{j};' for j in range(b))
        self.transpose_kernel=xp.fast.metal_kernel(name=f'signed_graded_transpose_v88_b{b}',
            input_names=['ptr','post','weight','drive'],output_names=['out'],source=f'''
            uint i=thread_position_in_grid.x/32,lane=thread_index_in_simdgroup;
            if(i>=N)return;
            {declarations}
            for(int e=ptr[i]+lane;e<ptr[i+1];e+=32){{{transpose_product}}}
            {reduction}
            if(lane==0){{{transpose_write}}}
            ''')
        self.gradient_kernel=xp.fast.metal_kernel(name=f'signed_graded_loge_gradient_v88_b{b}',
            input_names=['pre','post','weight','previous','drive','grad'],output_names=['out'],source='''
            uint e=thread_position_in_grid.x;if(e>=P)return;
            float value=0.0f;
            for(int b=0;b<B;b++)value+=previous[b*N+pre[e]]*drive[b*N+post[e]];
            out[e]=grad[e]+weight[e]*value;
            ''')

    def step(self,current,record=False):
        xp=self.xp
        if current.shape!=(self.batch,self.n):raise ValueError('Complete declared current shape required')
        previous=self.state
        if self.backend=='gpu':
            if not isinstance(current,xp.array):current=xp.array(current,dtype=xp.float32)
            self.state,activation=self.forward_kernel(inputs=[self.ptr,self.pre,self.weight,previous,current],
                template=[('N',self.n)],grid=(self.n*32,1,1),threadgroup=(256,1,1),
                output_shapes=[previous.shape]*2,output_dtypes=[xp.float32]*2)
        else:
            activation=np.tanh((self.matrix@previous.T).T+current).astype(np.float32)
            self.state=(np.float32(.5)*previous+np.float32(.5)*activation).astype(np.float32)
        self.tick+=1
        return dict(previous=previous,activation=activation,state=self.state,error=None,
            keep=xp.ones((self.batch,1),xp.float32)) if record else None

    def evolve(self,current,record=False):
        records=[]
        for _ in range(10):
            rec=self.step(current,record)
            if record:records.append(rec)
        self.sync(self.state)
        return records

    def backward(self,records,terminal=None):
        xp=self.xp;adj=xp.zeros_like(self.state) if terminal is None else xp.array(terminal,dtype=xp.float32)
        grad=xp.zeros_like(self.log_e);inputs=[]
        if self.backend=='gpu':
            outgoing=xp.contiguous(self.weight[self.outedge]);eligible=xp.contiguous(self.weight[self.eligible])
            self.sync(outgoing,eligible)
        for rec in reversed(records):
            if rec['error'] is not None:adj=adj+rec['error']
            drive=.5*(1-rec['activation']*rec['activation'])*adj
            inputs.append(drive)
            if self.backend=='gpu':
                if self.p:
                    grad=self.gradient_kernel(inputs=[self.epre,self.epost,eligible,rec['previous'],drive,grad],
                        template=[('N',self.n),('B',self.batch),('P',self.p)],grid=(self.p,1,1),threadgroup=(256,1,1),
                        output_shapes=[(self.p,)],output_dtypes=[xp.float32])[0]
                propagated=self.transpose_kernel(inputs=[self.outptr,self.outpost,outgoing,drive],template=[('N',self.n)],
                    grid=(self.n*32,1,1),threadgroup=(256,1,1),output_shapes=[self.state.shape],output_dtypes=[xp.float32])[0]
            else:
                edge=self.layout.eligible
                # Chunk eligible pairs to bound temporary gather memory for B=4.
                for start in range(0,self.p,500_000):
                    e=edge[start:start+500_000]
                    grad[start:start+len(e)]+=self.weight[e]*np.sum(rec['previous'][:,self.layout.pre[e]]*drive[:,self.layout.post[e]],axis=0)
                propagated=(self.matrix.T@drive.T).T
            adj=(.5*adj+propagated)*rec['keep']
        self.sync(grad,adj,*inputs)
        return grad,adj,list(reversed(inputs))

    def update(self,gradient,enabled=True,lr=.001):
        if not np.isfinite(lr) or lr<=0:raise ValueError('Positive finite learning rate required')
        xp=self.xp;gradient=xp.array(gradient,dtype=xp.float32)
        if gradient.shape!=(self.p,):raise ValueError('Exactly eligible efficacy gradients required')
        norm=xp.sqrt(xp.sum(gradient*gradient));self.sync(norm)
        value=float(norm.item())
        if not np.isfinite(value):raise FloatingPointError('Finite internal gradient required')
        if not enabled:return dict(norm=value,updated=False)
        gradient=gradient*xp.minimum(1.,1./xp.maximum(norm,1e-12))
        self.updates+=1;self.m=.9*self.m+.1*gradient;self.v=.999*self.v+.001*gradient*gradient
        delta=(self.m/(1-.9**self.updates))/(xp.sqrt(self.v/(1-.999**self.updates))+1e-8)
        self.log_e=xp.clip(self.log_e-lr*delta,float(LOW),float(HIGH));self.sync(self.log_e,self.m,self.v)
        self.materialize();return dict(norm=value,updated=True)

    def snapshot(self):
        return dict(identity=self.identity,neural_steps=self.tick,simulated_ms=self.tick*10,
            optimizer_steps=self.updates,**{k:np.asarray(getattr(self,k)).copy() for k in ('state','log_e','m','v')})

    def restore(self,saved):
        expected={'identity','neural_steps','simulated_ms','optimizer_steps','state','log_e','m','v'}
        if (set(saved)!=expected or saved['identity']!=self.identity or type(saved['neural_steps']) is not int
                or saved['neural_steps']<0 or saved['simulated_ms']!=saved['neural_steps']*10
                or type(saved['optimizer_steps']) is not int or saved['optimizer_steps']<0):
            raise ValueError('Exact full-core identity and clocks required')
        for key in ('state','log_e','m','v'):
            a=np.asarray(saved[key]);shape=(self.batch,self.n) if key=='state' else (self.p,)
            if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Finite complete FP32 checkpoint required')
        if (np.any(np.abs(saved['state'])>1) or np.any(saved['log_e']<LOW) or np.any(saved['log_e']>HIGH) or np.any(saved['v']<0)):
            raise ValueError('Bounded neural/efficacy and valid Adam state required')
        for key in ('state','log_e','m','v'):setattr(self,key,self.xp.array(saved[key].copy()))
        self.tick=saved['neural_steps'];self.updates=saved['optimizer_steps'];self.materialize()
