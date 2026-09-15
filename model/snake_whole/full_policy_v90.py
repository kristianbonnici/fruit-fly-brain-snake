"""Learned full efficacy with the exact original frozen image-only interfaces."""
import json
import numpy as np
import mlx.core as mx
from .common import ROOT,digest,stable
from .comparison_archive_v59 import read,array_hash
from .full_visual_v89 import FullImagePolicy

BASE=ROOT/'data/snake-whole-v90'


class LearnedImagePolicy(FullImagePolicy):
    def __init__(self,arm='learned',loaded=None,record_substeps=True):
        if arm!='learned':raise ValueError('Exact final full learned efficacy required')
        review=json.loads((BASE/'review.json').read_text());report=json.loads((BASE/'runs/internal-91001/report.json').read_text())
        plan=json.loads((BASE/'training-contract.json').read_text());training_binding=digest(BASE/'training-contract.json')
        if (review['status']!='independently_verified_bounded_full_learning' or review['training_binding']!=training_binding
                or report['binding']!=training_binding or not report['comparison']['predictive_gate_passed']
                or not review['comparison']['predictive_gate_passed'] or report['logical_optimizer_steps']!=478):
            raise ValueError('Complete reviewed five-epoch candidate and passing predictive gate required')
        for p,sha in {**plan['sources'],**plan['artifacts']}.items():
            if digest(ROOT/p)!=sha:raise ValueError('Changed source learning provenance: '+p)
        checkpoint=BASE/'runs/internal-91001/final-checkpoint.npz';checkpoint_sha=digest(checkpoint)
        if report['checkpoints']['final-checkpoint.npz']!=checkpoint_sha:raise ValueError('Exact frozen final full checkpoint required')
        arrays,meta=read(checkpoint,training_binding)
        if meta['context']['phase']!='final' or meta['context']['optimizer_steps']!=478 or meta['context']['computed_slots']!=24512:
            raise ValueError('Complete logical final checkpoint required')
        super().__init__('transferred',loaded,record_substeps)
        if arrays['log_e'].shape!=(self.core.p,) or array_hash(arrays['log_e'])!=review['phases']['final']['log_e_hash']:
            raise ValueError('Exact reviewed full eligible efficacy vector required')
        if np.any(arrays['log_e']<np.float32(np.log(.05))) or np.any(arrays['log_e']>np.float32(np.log(4.))):
            raise ValueError('Positive bounded learned efficacy required')
        source_identity=self.identity;self.core.log_e=mx.array(arrays['log_e']);self.core.materialize();self.arm=arm
        self.identity=stable(dict(version=90,arm=arm,source_image_policy=source_identity,core_identity=self.core.identity,
            log_e=array_hash(arrays['log_e']),training_binding=training_binding,checkpoint_sha256=checkpoint_sha,
            source=digest(ROOT/'snake_whole/full_policy_v90.py'),learning='Demonstration-trained full internal efficacy; fixed image interfaces'))
        self.reset()
