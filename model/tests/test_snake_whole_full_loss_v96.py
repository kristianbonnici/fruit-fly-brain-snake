import unittest
import numpy as np
from snake_whole.full_loss_v96 import numpy_loss_gradient


class MixedInternalLossTests(unittest.TestCase):
    def test_analytic_logit_gradient_matches_finite_difference_and_zero_mass(self):
        x=np.array([[1.,.2,-.3],[-.4,.8,.3],[.1,.2,.3]])
        source=np.array([[.2,.3,.4],[.8,-.1,.2],[1.,2.,3.]])
        labels=np.array([2,0,1]);ce=np.array([2.,0.,0.]);kl=np.array([0.,1.5,0.])
        original=x.copy();original_source=source.copy();loss,gradient,parts=numpy_loss_gradient(x,labels,ce,kl,source)
        np.testing.assert_array_equal(x,original);np.testing.assert_array_equal(source,original_source)
        self.assertAlmostEqual(loss,parts.sum());self.assertGreater(parts[0],0);self.assertGreater(parts[1],0)
        fd=np.zeros_like(x);h=1e-6
        for i in range(x.size):
            plus=x.copy();minus=x.copy();plus.flat[i]+=h;minus.flat[i]-=h
            fd.flat[i]=(numpy_loss_gradient(plus,labels,ce,kl,source)[0]-numpy_loss_gradient(minus,labels,ce,kl,source)[0])/(2*h)
        np.testing.assert_allclose(gradient,fd,rtol=1e-7,atol=1e-10);np.testing.assert_array_equal(gradient[2],0)
        self.assertLess(gradient[0,2],0);self.assertGreater(gradient[0,0],0)
        self.assertLess(gradient[1,0],0)  # KL pulls current toward source, not the reverse.

    def test_zero_kl_at_source_and_masked_held_row_has_no_effect(self):
        x=np.array([[.2,.1,.8],[9.,-4.,2.]])
        zero=np.zeros(2);kl=np.array([1.,0.]);labels=np.zeros(2,int)
        loss,grad,_=numpy_loss_gradient(x,labels,zero,kl,x)
        self.assertEqual(loss,0);np.testing.assert_array_equal(grad,0)
        other=x.copy();other[1]=[-20.,10.,100.]
        self.assertEqual(numpy_loss_gradient(other,labels,zero,kl,x)[0],0)


if __name__=='__main__':unittest.main()
