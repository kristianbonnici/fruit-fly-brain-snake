"""Unchanged predictive gates plus explicitly separated mixed-loss reporting."""
import numpy as np
from .full_metrics_v95 import collect,summarize,compare
from .full_loss_v96 import numpy_loss_gradient


def objective_summary(data,objective,predictions,seen):
    rows=np.flatnonzero(seen & data['fitting'] & data['valid'])
    if not len(rows):return None
    _,_,parts=numpy_loss_gradient(predictions[rows],data['targets'][rows],objective['ce_mass'][rows],objective['kl_mass'][rows],objective['anchor_logits'][rows])
    parts*=8*len(rows)
    return dict(corrective_ce_weighted_sum=float(parts[0]),source_kl_weighted_sum=float(parts[1]),mixed_objective=float(parts.sum()),
        ce_mass=float(objective['ce_mass'][rows].sum()),kl_mass=float(objective['kl_mass'][rows].sum()),
        scope='Mixed fitting objective only; teacher CE groups retain the unchanged V95diagnostic weighting. Epoch predictions span updates; initial/final predictions use fixed checkpoints.')
