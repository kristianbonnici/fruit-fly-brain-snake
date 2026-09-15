"""Admission to an explicit game diagnostic, never an action safety filter."""


def evidence(review,old_legality):
    if review['status']!='independently_verified_bounded_full_learning' or old_legality['status']!='verified_posthoc_old_prefix_legality':raise ValueError('Independent learning and original-image replay audits required')
    comparison=review['comparison'];gates=comparison['gates']
    required=('corrective_mean_improvement','corrective_ci_lower_positive','at_most_four_illegal_choices','at_least_six_fatal_choices_corrected')
    if not all(gates[k] for k in required):raise ValueError('Audited corrective-prediction and legality evidence required')
    group=next(g for g in old_legality['groups'] if not g['fitting'] and g['stratum'] is None)
    if group['rows']!=3216 or group['arms']['v96']['newly_illegal']!=0 or group['arms']['v96']['illegal_actions']>=group['arms']['source']['illegal_actions']:raise ValueError('Old-state safety improvement without new illegal actions required')
    return dict(diagnostic_only=True,candidate='Exact V96final efficacy with its original failed offline gate preserved',
        original_offline_gate_passed=bool(comparison['predictive_gate_passed']),original_offline_gates=gates,
        diagnostic_evidence_gates={k:bool(gates[k]) for k in required},old_source_illegal=group['arms']['source']['illegal_actions'],
        old_candidate_illegal=group['arms']['v96']['illegal_actions'],old_newly_illegal=group['arms']['v96']['newly_illegal'])
