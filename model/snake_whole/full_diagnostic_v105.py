"""Explicit post-hoc admission preserving V104's failed safety comparison."""
import json
from .common import ROOT,digest

HEADS={'source':'b2d65045ecd21abc7a896ac6b12b13722ad857c2e17bd09376980050281f3b08',
       'learned':'94a6e7bd642f9bf11e0651eadda9042b97b73dcbfa3f362c469c0023cbf0e26c'}
EFFICACY={'source':(96,'de381ff7e64e615489c34fe6925173dbd5d116386d45d1787e92c0c328feffe0'),
          'learned':(103,'199b63a20db82959113f6b9dc81156856b061fd3be16c06f10149a362cfd6a1d')}
GATES=dict(candidate_passes_all_eight=True,own_adaptation_mean_positive=True,own_adaptation_ci_positive=True,
    matched_advantage_mean_positive=True,matched_advantage_ci_positive=True,candidate_held_illegal_no_worse_than_adapted_source=False)
ENDPOINT_GATES={'corrective_mean_improvement','corrective_ci_lower_positive','at_least_four_fatal_choices_corrected',
    'at_most_two_new_heldout_illegal','every_heldout_cycle_agreement_gain_at_least_point1','old_heldout_illegal_at_most_eight',
    'old_newly_illegal_at_most_two','fitting_illegal_no_worse_than_source'}


def evidence(report,review,binding):
    c=report['comparison']
    if (report['status']!='complete_bounded_matched_decoder_fit' or review['status']!='independently_verified_matched_decoder_fit'
        or report['binding']!=binding or review['binding']!=binding or report['work']!=review['work']
        or report['logical_decoder_updates']!=7500 or report['logical_fitting_sample_exposures']!=954000
        or report['decoder_parameters_per_condition']!=2179 or report['internal_updates']!=0
        or c['predictive_gate_passed'] is not False or review['predictive_gate_passed'] is not False
        or set(c['gates'])!=set(GATES) or any(c['gates'][k] is not v for k,v in GATES.items()) or review['gates']!=c['gates']
        or c['heldout_illegal']!={'source':5,'learned':6} or review['independent_head_vectors']!=29188):
        raise ValueError('Exact audited V104 fit with its original safety failure preserved required')
    for arm in HEADS:
        ep=c['endpoints'][arm];audit=review['endpoints'][arm]
        if (set(ep['gates'])!=ENDPOINT_GATES or any(v is not True for v in ep['gates'].values()) or audit['gates']!=ep['gates']
            or audit['held_illegal']!=c['heldout_illegal'][arm] or ep['fitting_safety']['final_illegal']!=11
            or review['checkpoint_identities'][arm]['checkpoint_sha256']!=HEADS[arm]):
            raise ValueError('Both exact adapted heads must retain all eight passed endpoint gates')
    for name in ('own_head_improvement','matched_internal_advantage'):
        if c[name]['mean_reduction']<=0 or c[name]['paired_bootstrap_95_interval'][0]<=0:raise ValueError('Recorded positive own/matched CE evidence required')
    return dict(diagnostic_only=True,original_overall_gate_passed=False,original_gates=dict(c['gates']),
        head_checkpoints=dict(HEADS),heldout_illegal=dict(c['heldout_illegal']),
        scope='Explicit post-hoc three-condition game diagnostic, no V104 gate waiver or final eligibility.')


def verify_selection():
    path=ROOT/'data/snake-whole-v105/eligibility-contract.json';selection=json.loads(path.read_text())
    if selection['status']!='eligible_for_explicit_posthoc_game_diagnostic_only':raise ValueError('Explicit admission required')
    for p,sha in {**selection['sources'],**selection['artifacts']}.items():
        if digest(ROOT/p)!=sha:raise ValueError('Changed post-hoc evidence: '+p)
    prior=ROOT/'data/snake-whole-v104'
    r=json.loads((prior/'decoder-report.json').read_text());a=json.loads((prior/'decoder-review.json').read_text())
    if evidence(r,a,digest(prior/'decoder-training-contract.json'))!=selection['evidence']:raise ValueError('Changed preserved outcome')
    return selection
