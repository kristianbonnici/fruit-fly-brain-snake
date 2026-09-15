"""Preserved V99 held-out checks plus a fitting-data safety requirement."""
import numpy as np
from .full_metrics_v99 import collect,summarize,compare as previous_compare
from .full_metrics_v96 import objective_summary as previous_objective_summary


def fitting_safety(data,initial,final,labels):
    rr=np.flatnonzero(data['fitting']&data['valid']);legal=labels['legal_actions'];first=initial.argmax(axis=1);last=final.argmax(axis=1)
    before=~legal[rr,first[rr]];after=~legal[rr,last[rr]]
    return dict(rows=len(rr),source_illegal=int(before.sum()),final_illegal=int(after.sum()),newly_illegal=int(np.sum(~before&after)))


def compare(initial,final,data,initial_predictions,final_predictions,labels):
    result=previous_compare(initial,final,data,initial_predictions,final_predictions,labels)
    safety=fitting_safety(data,initial_predictions,final_predictions,labels)
    if safety['rows']!=9540 or safety['source_illegal']!=28:raise ValueError('Exact source fitting safety baseline required')
    result['fitting_safety']=safety;result['gates']['fitting_illegal_no_worse_than_source']=safety['final_illegal']<=safety['source_illegal']
    result['predictive_gate_passed']=all(result['gates'].values());return result


def objective_summary(data,objective,predictions,seen):
    result=previous_objective_summary(data,objective,predictions,seen)
    if result is not None:result['scope']='Fitting-only all-risk and complete-cycle teacher CE plus unchanged V96-source KL; fixed initial/final endpoints, epoch predictions span updates'
    return result
