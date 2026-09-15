import tempfile,unittest
from pathlib import Path
import numpy as np
from snake_whole.full_decoder_v104 import Decoder,normalize,SHAPES


class LegalDecoderTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(104011)
        p={k:rng.normal(0,.1,shape).astype(np.float32) for k,shape in SHAPES.items()}
        x=normalize(rng.normal(size=(7,64)).astype(np.float32));y=np.array([0,1,2,1,0,2,1],np.int32)
        obj=dict(ce_mass=np.array([.5,1,2,0,1,0,0]),kl_mass=np.array([.1,.3,0,1,.3,0,0]),
            anchor_logits=rng.normal(size=(7,3)).astype(np.float32),legal_mass=np.array([1,.5,0,1,0,1,0]),
            legal_actions=np.array([[1,0,1],[0,1,0],[0,0,0],[1,1,1],[0,0,0],[1,0,1],[0,0,0]],bool))
        return p,x,y,obj

    def test_mixed_parameter_gradient_and_zero_mass(self):
        p,x,y,obj=self.fixture();model=Decoder(p);_,g=model.gradient(x,y,obj)
        for k,indices in {'w1':[(0,0),(4,30)],'b1':[(0,),(17,)],'w2':[(0,3),(2,7)],'b2':[(0,),(2,)]}.items():
            for ix in indices:
                plus={a:b.copy() for a,b in p.items()};minus={a:b.copy() for a,b in p.items()}
                plus[k][ix]+=.001;minus[k][ix]-=.001
                numerical=(Decoder(plus).gradient(x,y,obj)[0]-Decoder(minus).gradient(x,y,obj)[0])/.002
                self.assertAlmostEqual(float(g[k][ix]),numerical,delta=2e-4)
        empty={k:v.copy() for k,v in obj.items()}
        for k in ('ce_mass','kl_mass','legal_mass'):empty[k][:]=0
        empty['legal_actions'][:]=False
        loss,zero=model.gradient(x,y,empty);self.assertEqual(loss,0)
        self.assertTrue(all(v.dtype==np.float32 and not v.any() for v in zero.values()))
        empty['legal_mass'][0]=1
        with self.assertRaises(ValueError):model.gradient(x,y,empty)

    def test_exact_mixed_adam_restoration(self):
        p,x,y,obj=self.fixture();model=Decoder(p);model.step(x,y,obj)
        ctx=dict(updates=1,epoch=0,batch=1,source='legal-fixture',rng='frozen')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head.npz';model.save(path,ctx,'bound')
            expected=model.step(x,y,obj);resumed=Decoder(p);resumed.restore(path,ctx,'bound')
            self.assertEqual(expected,resumed.step(x,y,obj))
            for k in p:
                for first,second in ((model.parameters,resumed.parameters),(model.m,resumed.m),(model.v,resumed.v)):
                    np.testing.assert_array_equal(first[k],second[k]);self.assertEqual(first[k].dtype,np.float32)
            with self.assertRaises(ValueError):resumed.restore(path,dict(ctx,source='wrong'),'bound')

if __name__=='__main__':unittest.main()
