"""Image-only full network with exact separately adapted frozen decoder weights."""
import json
import numpy as np
import mlx.core as mx
from .common import ROOT,digest,stable
from .comparison_archive_v59 import read,array_hash,model_arrays
from .full_visual_v89 import FullImagePolicy
from .full_diagnostic_v105 import verify_selection,HEADS,EFFICACY


class AdaptedImagePolicy(FullImagePolicy):
    def __init__(self,arm,loaded=None,record_substeps=True):
        if arm not in HEADS:raise ValueError('Exact adapted source or learned condition required')
        verify_selection();version,sha=EFFICACY[arm];base=ROOT/f'data/snake-whole-v{version}'
        path=base/'runs/internal-91001/final-checkpoint.npz'
        if digest(path)!=sha:raise ValueError('Exact efficacy checkpoint required')
        e,em=read(path,digest(base/'training-contract.json'));review=json.loads((base/'review.json').read_text())
        if em['context']['phase']!='final' or array_hash(e['log_e'])!=review['phases']['final']['log_e_hash']:raise ValueError('Exact reviewed final efficacy required')
        head_base=ROOT/'data/snake-whole-v104';head_path=head_base/f'decoder-{arm}-final.npz'
        if digest(head_path)!=HEADS[arm]:raise ValueError('Exact adapted head checkpoint required')
        h,hm=read(head_path,digest(head_base/'decoder-training-contract.json'))
        if hm['context']['condition']!=arm or hm['context']['epoch']!=50 or hm['optimizer_steps']!=3750:raise ValueError('Exact completed head required')
        super().__init__('transferred',loaded,record_substeps)
        if e['log_e'].shape!=(self.core.p,) or np.min(e['log_e'])<np.float32(np.log(.05)) or np.max(e['log_e'])>np.float32(np.log(4)):
            raise ValueError('Complete positive bounded efficacy required')
        parent=self.identity;self.core.log_e=mx.array(e['log_e']);self.core.materialize()
        keys={'w1':'layers.1.weight','b1':'layers.1.bias','w2':'layers.3.weight','b2':'layers.3.bias'}
        self.interface.decoder.load_weights([(name,mx.array(h['parameter.'+k])) for k,name in keys.items()]);self.interface.eval()
        mx.eval(self.interface.parameters());installed=model_arrays(self.interface)
        for k,name in keys.items():np.testing.assert_array_equal(installed['model.decoder.'+name],h['parameter.'+k])
        self.arm=arm;self.identity=stable(dict(version=105,arm=arm,parent_image_policy=parent,core_identity=self.core.identity,
            efficacy_checkpoint=sha,log_e=array_hash(e['log_e']),head_checkpoint=HEADS[arm],
            interface={k:array_hash(v) for k,v in installed.items()},selection=digest(ROOT/'data/snake-whole-v105/eligibility-contract.json'),
            source=digest(ROOT/'snake_whole/full_policy_v105.py'),learning='Frozen V104 adapted head and exact full efficacy; post-hoc diagnostic preserves failed six-versus-five safety comparison'))
        self.reset()


def make_policy(arm,loaded=None,record_substeps=True):
    if arm=='baseline':
        from .full_policy_v97 import LearnedImagePolicy
        return LearnedImagePolicy(loaded=loaded,record_substeps=record_substeps)
    return AdaptedImagePolicy(arm,loaded,record_substeps)
