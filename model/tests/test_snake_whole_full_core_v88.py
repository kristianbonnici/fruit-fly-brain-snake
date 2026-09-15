import copy
import unittest
from types import SimpleNamespace
import numpy as np
from snake_whole.full_core_v88 import Layout,Core
try:
    import mlx.core as mx
    HAS_MLX=True
except ImportError:HAS_MLX=False


def fixture():
    n=11;signs=np.where(np.arange(n)%3==0,-1,1).astype(np.float32)
    pairs=sorted({(j,i) for j in range(n-1) for i in (j,(j+1)%n,(j+4)%n)})
    post,pre=np.asarray(pairs,np.int32).T;counts=(1+np.arange(len(pre))%5).astype(np.int64)
    ptr=np.r_[0,np.cumsum(np.bincount(post,minlength=n))].astype(np.int32)
    graph=SimpleNamespace(n=n,m=len(pre),ptr=ptr,pre=pre,post=post,counts=counts,
        signs=signs,plastic=(signs[pre]>0).astype(np.uint8))
    norm=np.sqrt(np.bincount(post,weights=counts.astype(float)**2,minlength=n))
    phi=(.3*signs[pre]/np.maximum(norm[post],1)).astype(np.float32)
    return Layout(graph,phi)


def independent_dense(layout,initial,currents,log_e,terminal):
    # Independent FP64 objective, not the implementation's CSR or adjoint.
    weights=layout.base.astype(np.float64).copy();weights[layout.eligible]*=np.exp(np.asarray(log_e,np.float64))
    matrix=np.zeros((layout.n,layout.n));matrix[layout.post,layout.pre]=weights
    state=np.asarray(initial,np.float64).copy();states=[]
    for current in currents:
        state=.5*state+.5*np.tanh(state@matrix.T+current);states.append(state.copy())
    return float(np.sum(state*terminal)),np.array(states)


def check_backend(test,backend):
    layout=fixture();rng=np.random.default_rng(88001)
    for batch in (1,2,4):
        core=Core(layout,batch,backend);initial=rng.uniform(-.2,.2,(batch,layout.n)).astype(np.float32)
        currents=rng.uniform(-.4,.4,(3,batch,layout.n)).astype(np.float32)
        terminal=rng.normal(size=(batch,layout.n)).astype(np.float32)/batch
        initial_log=rng.uniform(-.1,.1,core.p).astype(np.float32)
        saved=core.snapshot();saved['state']=initial;saved['log_e']=initial_log;core.restore(saved)
        records=[core.step(c,True) for c in currents];core.sync(core.state)
        gradient,initial_gradient,input_gradient=core.backward(records,terminal)
        _,reference=independent_dense(layout,initial,currents,initial_log,terminal)
        np.testing.assert_allclose(np.asarray(core.state),reference[-1],rtol=3e-5,atol=3e-6)
        if backend=='cpu':
            epsilon=1e-5;estimate=[]
            for index in range(core.p):
                plus=initial_log.astype(np.float64);minus=plus.copy();plus[index]+=epsilon;minus[index]-=epsilon
                estimate.append((independent_dense(layout,initial,currents,plus,terminal)[0]-
                    independent_dense(layout,initial,currents,minus,terminal)[0])/(2*epsilon))
            np.testing.assert_allclose(gradient,estimate,rtol=3e-4,atol=3e-6)
            plus=initial.astype(np.float64);minus=plus.copy();plus[0,0]+=epsilon;minus[0,0]-=epsilon
            difference=(independent_dense(layout,plus,currents,initial_log,terminal)[0]-
                independent_dense(layout,minus,currents,initial_log,terminal)[0])/(2*epsilon)
            test.assertAlmostEqual(float(initial_gradient[0,0]),difference,places=5)
            plus=currents.astype(np.float64);minus=plus.copy();plus[0,0,0]+=epsilon;minus[0,0,0]-=epsilon
            difference=(independent_dense(layout,initial,plus,initial_log,terminal)[0]-
                independent_dense(layout,initial,minus,initial_log,terminal)[0])/(2*epsilon)
            test.assertAlmostEqual(float(input_gradient[0][0,0]),difference,places=5)
        else:
            cpu=Core(layout,batch,'cpu');reference_saved=cpu.snapshot()
            reference_saved['state']=initial;reference_saved['log_e']=initial_log;cpu.restore(reference_saved)
            cpu_records=[cpu.step(c,True) for c in currents];cpu_gradient,cpu_initial,cpu_inputs=cpu.backward(cpu_records,terminal)
            for a,b in [(gradient,cpu_gradient),(initial_gradient,cpu_initial),*zip(input_gradient,cpu_inputs)]:
                np.testing.assert_allclose(np.asarray(a),b,rtol=3e-4,atol=3e-6)
        before=core.snapshot();core.update(gradient,False)
        for key in ('state','log_e','m','v'):np.testing.assert_array_equal(before[key],core.snapshot()[key])
        test.assertEqual(core.updates,0)
        core.update(gradient);boundary=core.snapshot()
        test.assertTrue(np.any(boundary['log_e']!=before['log_e']))
        after_records=[core.step(c,True) for c in currents];after_gradient,_,_=core.backward(after_records,terminal)
        core.update(after_gradient);expected=core.snapshot()
        restored=Core(layout,batch,backend);restored.restore(boundary);boundary['state'][:]=0
        repeated=[restored.step(c,True) for c in currents];repeated_gradient,_,_=restored.backward(repeated,terminal)
        np.testing.assert_array_equal(np.asarray(after_gradient),np.asarray(repeated_gradient))
        restored.update(repeated_gradient);actual=restored.snapshot()
        for key in ('state','log_e','m','v'):np.testing.assert_array_equal(expected[key],actual[key])
        test.assertEqual(actual['neural_steps'],6);test.assertEqual(actual['simulated_ms'],60);test.assertEqual(actual['optimizer_steps'],2)


class FullCoreCPUTests(unittest.TestCase):
    def test_independent_dense_delayed_gradients_and_exact_resume(self):check_backend(self,'cpu')

    def test_incompatible_history_and_count_sign_contract_rejected(self):
        layout=fixture();core=Core(layout);saved=core.snapshot()
        for key,value in [('identity','other'),('simulated_ms',10),('optimizer_steps',-1),('state',np.zeros((1,10),np.float32))]:
            bad=copy.deepcopy(saved);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):core.restore(bad)
        other=Core(layout,2)
        with self.assertRaises(ValueError):other.restore(saved)


@unittest.skipUnless(HAS_MLX,'Optional native Metal full-core kernels')
class FullCoreGPUTests(unittest.TestCase):
    def test_sparse_metal_numerics_gradients_and_exact_resume(self):check_backend(self,'gpu')


if __name__=='__main__':unittest.main()
