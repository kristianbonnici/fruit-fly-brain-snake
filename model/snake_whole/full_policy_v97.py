"""Learned full efficacy with the exact original frozen image-only interfaces."""
import json
import numpy as np
import mlx.core as mx
from .common import ROOT,digest,stable
from .comparison_archive_v59 import read,array_hash
from .full_visual_v89 import FullImagePolicy

BASE=ROOT/'data/snake-whole-v96'
DIAGNOSTIC=ROOT/'data/snake-whole-v97'


class LearnedImagePolicy(FullImagePolicy):
    def __init__(self,arm='learned',loaded=None,record_substeps=True):
        if arm!='learned':raise ValueError('Exact final full learned efficacy required')
        from .full_diagnostic_v97 import evidence
        selection=json.loads((DIAGNOSTIC/'eligibility-contract.json').read_text())
        if selection['status']!='eligible_for_explicit_posthoc_game_diagnostic_only':raise ValueError('Explicit diagnostic admission required')
        for p,sha in {**selection['sources'],**selection['artifacts']}.items():
            if digest(ROOT/p)!=sha:raise ValueError('Changed diagnostic eligibility evidence: '+p)

        review=json.loads((BASE/'review.json').read_text());report=json.loads((BASE/'runs/internal-91001/report.json').read_text())
        plan=json.loads((BASE/'training-contract.json').read_text());training_binding=digest(BASE/'training-contract.json')
        if (review['status']!='independently_verified_bounded_full_learning' or review['training_binding']!=training_binding
                or report['binding']!=training_binding or report['logical_optimizer_steps']!=687):
            raise ValueError('Complete independently reviewed frozen V96candidate required')
        if evidence(review,json.loads((BASE/'old-legality-report.json').read_text()))!=selection['evidence']:raise ValueError('Preserved offline failure and explicit diagnostic evidence required')
        for p,sha in {**plan['sources'],**plan['artifacts']}.items():
            if digest(ROOT/p)!=sha:raise ValueError('Changed source learning provenance: '+p)
        checkpoint=BASE/'runs/internal-91001/final-checkpoint.npz';checkpoint_sha=digest(checkpoint)
        if report['checkpoints']['final-checkpoint.npz']!=checkpoint_sha or checkpoint_sha!=selection['candidate_checkpoint_sha256']:raise ValueError('Exact frozen final full checkpoint required')
        arrays,meta=read(checkpoint,training_binding)
        if meta['context']['phase']!='final' or meta['context']['optimizer_steps']!=687 or meta['context']['computed_slots']!=37984:
            raise ValueError('Complete logical final checkpoint required')
        super().__init__('transferred',loaded,record_substeps)
        if arrays['log_e'].shape!=(self.core.p,) or array_hash(arrays['log_e'])!=review['phases']['final']['log_e_hash']:
            raise ValueError('Exact reviewed full eligible efficacy vector required')
        if np.any(arrays['log_e']<np.float32(np.log(.05))) or np.any(arrays['log_e']>np.float32(np.log(4.))):
            raise ValueError('Positive bounded learned efficacy required')
        source_identity=self.identity;self.core.log_e=mx.array(arrays['log_e']);self.core.materialize();self.arm=arm
        self.identity=stable(dict(version=97,arm=arm,source_image_policy=source_identity,core_identity=self.core.identity,
            log_e=array_hash(arrays['log_e']),training_binding=training_binding,checkpoint_sha256=checkpoint_sha,
            source=digest(ROOT/'snake_whole/full_policy_v97.py'),diagnostic_eligibility=digest(DIAGNOSTIC/'eligibility-contract.json'),learning='Frozen V96efficacy in a post-hoc game diagnostic; V96offline gate remains failed; fixed image interfaces'))
        self.reset()


def make_policy(arm,loaded=None,record_substeps=True):
    if arm=='learned':return LearnedImagePolicy(arm,loaded,record_substeps)
    if arm=='source':
        from .full_policy_v90 import LearnedImagePolicy as Predecessor
        return Predecessor(loaded=loaded,record_substeps=record_substeps)
    raise ValueError('Exact source or learned efficacy condition required')
