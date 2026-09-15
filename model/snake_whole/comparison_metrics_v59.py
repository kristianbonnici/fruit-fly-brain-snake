"""Episode-separated predictive and paired-game learning evidence."""
import numpy as np


def losses(logits, labels):
    z = np.asarray(logits, np.float64)
    if z.shape != (len(labels), 3) or not np.isfinite(z).all():
        raise ValueError('Finite aligned three-action predictions required')
    z = z-z.max(axis=1, keepdims=True)
    return -z[np.arange(len(z)), labels]+np.log(np.exp(z).sum(axis=1))


def interval(values, seed=59019):
    values = np.asarray(values, np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two independent finite units required')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.percentile(means, [2.5, 97.5]).tolist(), units=len(values))


def predictive(logits, baseline, data):
    current, old = losses(logits, data['targets']), losses(baseline, data['targets'])
    prediction, old_prediction = np.argmax(logits, axis=1), np.argmax(baseline, axis=1)
    metrics, gains = {}, {}
    for origin, name in enumerate(('expert', 'learner')):
        for fitting, split in ((True, 'fitting'), (False, 'validation')):
            mask = data['valid'] & (data['fitting'] == fitting) & (data['origin'] == origin)
            recalls = [float(np.mean(prediction[mask & (data['targets'] == a)] == a)) for a in range(3)]
            previous = [float(np.mean(old_prediction[mask & (data['targets'] == a)] == a)) for a in range(3)]
            metrics[split+'/'+name] = dict(cross_entropy=float(current[mask].mean()), accuracy=float(np.mean(prediction[mask] == data['targets'][mask])),
                action_recall=recalls, turn_macro_recall=float(np.mean([recalls[0], recalls[2]])),
                turn_macro_recall_gain=float(np.mean([recalls[0]-previous[0], recalls[2]-previous[2]])),
                labels=np.bincount(data['targets'][mask], minlength=3).tolist())
            if not fitting:
                values = [float(np.mean((old-current)[mask & (data['episode'] == e)])) for e in np.unique(data['episode'][mask])]
                gains[name] = dict(**interval(values), episode_improvements=values)
    return dict(metrics=metrics, heldout_episode_CE_improvement=gains, fitting_weighted_CE=float(data['mass']@current))


def image_gate(game, predictive_report, thresholds):
    loss = predictive_report['heldout_episode_CE_improvement']['learner']
    return dict(mean=game['mean'] >= thresholds['mean_at_least'], median=game['median'] >= thresholds['median_at_least'],
        five=game['fraction_at_least_five'] >= thresholds['fraction_five_food_at_least'],
        paired_score=game['paired_improvement']['ci95'][0] > thresholds['paired_score_ci95_lower_above'],
        learner_CE=loss['mean'] >= thresholds['learner_equal_episode_CE_improvement_at_least'],
        learner_CI=loss['ci95'][0] > thresholds['learner_episode_ci95_lower_above'])


def neural_gate(report, thresholds):
    gain = report['heldout_episode_CE_improvement']
    return dict(learner_CE=gain['learner']['mean'] >= thresholds['learner_equal_episode_CE_improvement_over_original_linear_at_least'],
        learner_CI=gain['learner']['ci95'][0] > thresholds['learner_episode_ci95_lower_above'],
        learner_turn_recall=report['metrics']['validation/learner']['turn_macro_recall_gain'] >= thresholds['learner_turn_macro_recall_gain_at_least'],
        expert_nonregression=gain['expert']['mean'] >= -thresholds['maximum_expert_equal_episode_CE_regression'])
