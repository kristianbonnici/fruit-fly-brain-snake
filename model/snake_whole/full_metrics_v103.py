"""Unchanged eight eligibility gates; three separately reported loss terms."""
import numpy as np
from .full_metrics_v100 import collect, summarize, compare
from .full_loss_v103 import numpy_loss_gradient


def objective_summary(data, objective, predictions, seen):
    rows = np.flatnonzero(seen & data['fitting'] & data['valid'])
    if not len(rows): return None
    _, _, parts = numpy_loss_gradient(predictions[rows], data['targets'][rows],
        objective['ce_mass'][rows], objective['kl_mass'][rows], objective['anchor_logits'][rows],
        objective['legal_mass'][rows], objective['legal_actions'][rows])
    parts *= 8 * len(rows)
    return dict(corrective_ce_weighted_sum=float(parts[0]), source_kl_weighted_sum=float(parts[1]),
        legal_set_weighted_sum=float(parts[2]), mixed_objective=float(parts.sum()),
        ce_mass=float(objective['ce_mass'][rows].sum()), kl_mass=float(objective['kl_mass'][rows].sum()),
        legal_mass=float(objective['legal_mass'][rows].sum()),
        scope='Fitting-only teacher CE .25, source KL .5, legal-set loss .25; initial/final endpoints fixed, epoch predictions span updates')
