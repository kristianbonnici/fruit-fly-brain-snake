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
        for stratum in (None,0,1,2,3,4,5):
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


from .full_metrics_v96 import objective_summary as previous_objective_summary


def objective_summary(data,objective,predictions,seen):
    result=previous_objective_summary(data,objective,predictions,seen)
    if result is not None:result['scope']='Fitting-only fatal/cycle CE and V96source KL; diagnostic CE masses are half old V96 and half new V98. Epoch predictions span updates; initial/final checkpoints are fixed.'
    return result


def legality(data,initial,final,labels):
    n=len(data['episode']);legal=labels['legal_actions'];rows=labels['rows'];first=initial.argmax(axis=1);last=final.argmax(axis=1)
    if not np.array_equal(rows,np.arange(n)) or legal.shape!=(n,3) or legal.dtype!=bool or not np.array_equal(labels['fitting'],data['fitting']):raise ValueError('Complete aligned diagnostic-only legality required')
    if not np.array_equal(first,labels['source_actions']):raise ValueError('Every original source action must reproduce')
    before=~legal[rows,first];after=~legal[rows,last];groups={}
    for name,mask in [('old',(data['source_dataset']!=2)&~data['fitting']),('new',(data['source_dataset']==2)&~data['fitting'])]:
        rr=rows[mask]
        groups[name]=dict(rows=len(rr),source_illegal=int(before[rr].sum()),final_illegal=int(after[rr].sum()),
            corrected_source_fatal=int(np.sum(before[rr]&~after[rr])),newly_illegal=int(np.sum(~before[rr]&after[rr])),
            remaining_errors=[dict(row=int(r),episode=int(data['episode'][r]),move=int(data['moves'][r]),source_action=int(first[r]),final_action=int(last[r]),
                legal_actions=legal[r].tolist(),source_fatal=bool(before[r])) for r in rr[after[rr]]])
    if groups['old']['rows']!=3684 or groups['new']['rows']!=1370 or groups['old']['source_illegal']!=6 or groups['new']['source_illegal']!=6:raise ValueError('Exact predeclared old/new source legality baselines required')
    selected=(data['source_dataset']==2)&~data['fitting']&labels['cycle_membership'];cycles=[]
    for episode in np.unique(data['episode'][selected]):
        rr=rows[selected&(data['episode']==episode)];a=float(np.mean(first[rr]==data['targets'][rr]));b=float(np.mean(last[rr]==data['targets'][rr]))
        cycles.append(dict(episode=int(episode),rows=len(rr),source_teacher_agreement=a,final_teacher_agreement=b,agreement_gain=b-a))
    if len(cycles)!=2 or sum(c['rows'] for c in cycles)!=720:raise ValueError('Exactly two held-out source cycles and720rows required')
    return dict(groups=groups,source_cycle_prefixes=cycles,scope='Recorded-state assessment only, no proof of actual cycle escape or runtime filtering.')


def eligibility(new_gain,new_lower,diagnostic):
    old=diagnostic['groups']['old'];new=diagnostic['groups']['new'];cycles=diagnostic['source_cycle_prefixes']
    return dict(corrective_mean_improvement=new_gain>=.1,corrective_ci_lower_positive=new_lower>0,
        at_least_four_fatal_choices_corrected=new['corrected_source_fatal']>=4,at_most_two_new_heldout_illegal=new['final_illegal']<=2,
        every_heldout_cycle_agreement_gain_at_least_point1=len(cycles)==2 and all(c['agreement_gain']>=.1 for c in cycles),
        old_heldout_illegal_at_most_eight=old['final_illegal']<=8,old_newly_illegal_at_most_two=old['newly_illegal']<=2)


def compare(initial,final,data,initial_predictions,final_predictions,labels):
    before={e['episode']:e for e in initial['episodes'] if not e['fitting']};after={e['episode']:e for e in final['episodes'] if not e['fitting']}
    if set(before)!=set(after) or len(before)!=48:raise ValueError('Exactly48 paired held-out prefixes required')
    ids=sorted(before)
    if any(before[e]['stratum']!=after[e]['stratum'] for e in ids) or any(sum(before[e]['stratum']==s for e in ids)!=8 for s in range(6)):raise ValueError('Eight held-out prefixes in each of six fixed strata required')
    difference=np.array([before[e]['cross_entropy']-after[e]['cross_entropy'] for e in ids]);new=[i for i,e in enumerate(ids) if before[e]['stratum']==5];delta=difference[new]
    if not np.isfinite(difference).all():raise ValueError('Finite paired loss differences required')
    draws=np.random.default_rng(99031).integers(0,48,(10000,48));ci=np.quantile(difference[draws].mean(axis=1),[.025,.975])
    draws=np.random.default_rng(99032).integers(0,8,(10000,8));new_ci=np.quantile(delta[draws].mean(axis=1),[.025,.975])
    diagnostic=legality(data,initial_predictions,final_predictions,labels);gates=eligibility(float(delta.mean()),float(new_ci[0]),diagnostic)
    strata={str(s):float(np.mean([difference[i] for i,e in enumerate(ids) if before[e]['stratum']==s])) for s in range(6)}
    return dict(heldout_episode_ids=ids,paired_cross_entropy_reductions=difference.tolist(),mean_reduction=float(difference.mean()),paired_bootstrap_95_interval=ci.tolist(),
        bootstrap_seed=99031,bootstrap_draws=10000,stratum_reductions=strata,gates=gates,predictive_gate_passed=all(gates.values()),
        new_heldout=dict(episode_ids=[ids[i] for i in new],mean_reduction=float(delta.mean()),paired_bootstrap_95_interval=new_ci.tolist(),bootstrap_seed=99032,bootstrap_draws=10000),
        legality=diagnostic,scope='Recorded old/current-source held-out training prefixes; one continued training seed. CE/legality/cycle diagnostics are not gameplay, final acceptance or training-run uncertainty.')
