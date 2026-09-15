"""Recorded-input prediction summaries and predeclared paired development gate."""
import numpy as np


def collect(predictions,seen,rows,values):
    if rows.dtype.kind not in 'iu':raise ValueError('Integer scheduled row indices required')
    active=rows>=0;selected=rows[active]
    if len(np.unique(selected))!=len(selected) or np.any(seen[selected]):raise ValueError('Each real row is predicted once per phase')
    values=np.asarray(values)
    if values.shape!=(*rows.shape,3) or not np.isfinite(values).all():raise ValueError('Finite complete three-action logits required')
    predictions[selected]=values[active];seen[selected]=True


def summarize(data,predictions,seen):
    if predictions.shape!=(len(seen),3) or not np.isfinite(predictions).all():raise ValueError('Aligned finite predictions required')
    x=predictions.astype(np.float64);x-=x.max(axis=1,keepdims=True)
    ce=np.log(np.exp(x).sum(axis=1))-x[np.arange(len(x)),data['targets']];action=predictions.argmax(axis=1)
    episodes=[]
    for episode in np.unique(data['episode'][seen]):
        rows=np.flatnonzero((data['episode']==episode)&data['valid'])
        if not np.all(seen[rows]):raise ValueError('Complete selected prefix required')
        episodes.append(dict(episode=int(episode),fitting=bool(data['fitting'][rows[0]]),stratum=int(data['stratum'][rows[0]]),
            observations=len(rows),cross_entropy=float(np.mean(ce[rows])),accuracy=float(np.mean(action[rows]==data['targets'][rows]))))
    groups=[]
    for fitting in (True,False):
        for stratum in (None,0,1,2,3,4):
            rows=seen & data['valid'] & (data['fitting']==fitting)
            if stratum is not None:rows &= data['stratum']==stratum
            selected=[e for e in episodes if e['fitting']==fitting and (stratum is None or e['stratum']==stratum)]
            if not selected:continue
            confusion=np.bincount(data['targets'][rows]*3+action[rows],minlength=9).reshape(3,3)
            recall=[float(confusion[i,i]/confusion[i].sum()) if confusion[i].sum() else None for i in range(3)]
            groups.append(dict(fitting=fitting,stratum=stratum,episodes=len(selected),observations=int(rows.sum()),
                equal_episode_cross_entropy=float(np.mean([e['cross_entropy'] for e in selected])),
                row_accuracy=float(np.mean(action[rows]==data['targets'][rows])),confusion=confusion.tolist(),class_recall=recall,
                action_counts=np.bincount(action[rows],minlength=3).tolist()))
    fit=seen & data['fitting'];mass=data['mass'][fit]
    return dict(episodes=episodes,groups=groups,fitting_mass_weighted_cross_entropy=float(np.sum(ce[fit]*mass)/mass.sum()) if mass.sum() else None)


def legality(data,initial,final,labels):
    rows=labels['rows'];legal=labels['legal_actions'];executed=labels['executed']
    expected=np.flatnonzero(data['source_dataset']==1)
    if not np.array_equal(rows,expected) or legal.shape!=(len(rows),3) or legal.dtype!=bool:raise ValueError('Exact separate correction legality rows required')
    if not np.array_equal(labels['fitting'],data['fitting'][rows]) or np.any(~data['valid'][rows]):raise ValueError('Verified correction role and valid labels required')
    if not np.array_equal(initial[rows].argmax(axis=1),executed):raise ValueError('Initial predictions must reproduce every original action')
    held=~labels['fitting'];held_rows=rows[held];allowed=legal[held]
    first=executed[held];last=final[held_rows].argmax(axis=1);index=np.arange(len(first))
    fatal=~allowed[index,first];illegal=~allowed[index,last]
    if len(first)!=468 or len(np.unique(data['episode'][held_rows]))!=8 or fatal.sum()!=8:raise ValueError('Exactly468 held-out correction rows and8 original fatal choices required')
    errors=[dict(row=int(r),episode=int(data['episode'][r]),move=int(data['moves'][r]),source_action=int(a),final_action=int(b),
        legal_actions=allowed[i].tolist(),source_fatal=bool(fatal[i])) for i,(r,a,b) in enumerate(zip(held_rows,first,last)) if illegal[i]]
    return dict(heldout_rows=len(first),source_illegal_actions=int(fatal.sum()),final_illegal_actions=int(illegal.sum()),
        corrected_source_fatal_choices=int(np.sum(fatal & ~illegal)),remaining_errors=errors,
        source_fatal_rows=held_rows[fatal].tolist(),corrected_source_fatal_rows=held_rows[fatal & ~illegal].tolist(),
        scope='Separate post-training diagnostic labels; no runtime filter or privileged neural inputs.')


def compare(initial,final,data,initial_predictions,final_predictions,labels):
    before={e['episode']:e for e in initial['episodes'] if not e['fitting']}
    after={e['episode']:e for e in final['episodes'] if not e['fitting']}
    if set(before)!=set(after) or len(before)!=40:raise ValueError('Exactly40 paired held-out prefixes required')
    ids=sorted(before)
    if any(before[e]['stratum']!=after[e]['stratum'] for e in ids):raise ValueError('Unchanged episode strata required')
    difference=np.array([before[e]['cross_entropy']-after[e]['cross_entropy'] for e in ids])
    if not np.isfinite(difference).all():raise ValueError('Finite paired losses required')
    draws=np.random.default_rng(95031).integers(0,40,(10000,40));ci=np.quantile(difference[draws].mean(axis=1),[.025,.975])
    strata={str(s):float(np.mean([difference[i] for i,e in enumerate(ids) if before[e]['stratum']==s])) for s in range(5)}
    new=[i for i,e in enumerate(ids) if before[e]['stratum']==4]
    if len(new)!=8 or any(sum(before[e]['stratum']==s for e in ids)!=8 for s in range(5)):raise ValueError('Eight held-out prefixes in each of five strata required')
    delta=difference[new];draws=np.random.default_rng(95032).integers(0,8,(10000,8));new_ci=np.quantile(delta[draws].mean(axis=1),[.025,.975])
    diagnostic=legality(data,initial_predictions,final_predictions,labels)
    gates=dict(heldout_mean_improvement=float(difference.mean())>=.05,paired_ci_lower_positive=float(ci[0])>0,
        no_old_stratum_regression_over_point02=all(strata[str(s)]>=-.02 for s in range(4)),
        corrective_mean_improvement=float(delta.mean())>=.1,corrective_ci_lower_positive=float(new_ci[0])>0,
        at_most_four_illegal_choices=diagnostic['final_illegal_actions']<=4,at_least_six_fatal_choices_corrected=diagnostic['corrected_source_fatal_choices']>=6)
    return dict(heldout_episode_ids=ids,paired_cross_entropy_reductions=difference.tolist(),mean_reduction=float(difference.mean()),
        paired_bootstrap_95_interval=ci.tolist(),bootstrap_seed=95031,bootstrap_draws=10000,stratum_reductions=strata,gates=gates,
        predictive_gate_passed=all(gates.values()),new_heldout=dict(episode_ids=[ids[i] for i in new],mean_reduction=float(delta.mean()),
        paired_bootstrap_95_interval=new_ci.tolist(),bootstrap_seed=95032,bootstrap_draws=10000),legality=diagnostic,
        scope='Previously inspected development prefixes, one continued training seed. Not gameplay, final evaluation or training-run uncertainty.')
