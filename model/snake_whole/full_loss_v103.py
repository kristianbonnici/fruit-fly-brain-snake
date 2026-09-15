"""Training-only legal-set probability, teacher CE and source-retention KL."""
import numpy as np


def numpy_loss_gradient(logits, targets, ce_mass, kl_mass, source, legal_mass, legal_actions):
    x = np.array(logits, dtype=np.float64, copy=True)
    anchor = np.array(source, dtype=np.float64, copy=True)
    weights = (ce_mass, kl_mass, legal_mass)
    if (x.ndim != 2 or x.shape[1] != 3 or not len(x) or anchor.shape != x.shape
            or targets.shape != (len(x),) or any(w.shape != targets.shape for w in weights)
            or legal_actions.shape != x.shape or legal_actions.dtype != bool):
        raise ValueError('Aligned three-action logits, weights and Boolean loss labels required')
    if (not all(np.isfinite(v).all() for v in (x, anchor, *weights))
            or any(np.any(w < 0) for w in weights)
            or targets.dtype.kind not in 'iu' or np.any((targets < 0) | (targets > 2))
            or np.any((legal_mass > 0) & ~legal_actions.any(axis=1))):
        raise ValueError('Finite weights and a nonempty legal set on every weighted row required')

    x -= x.max(axis=1, keepdims=True)
    normalizer = np.log(np.exp(x).sum(axis=1, keepdims=True))
    logq = x - normalizer
    anchor -= anchor.max(axis=1, keepdims=True)
    logp = anchor - np.log(np.exp(anchor).sum(axis=1, keepdims=True))
    p, q = np.exp(logp), np.exp(logq)
    # Empty stored labels on held/inactive rows become a benign all-legal set
    # only when their loss mass is zero. Avoid zero times log(0), without using
    # a finite sentinel that could leak probability to an illegal action.
    mask = np.where(legal_mass[:, None] > 0, legal_actions, True)
    masked = np.where(mask, x, -np.inf)
    legal_max = masked.max(axis=1, keepdims=True)
    legal_logsum = legal_max + np.log(np.exp(masked - legal_max).sum(axis=1, keepdims=True))
    conditional = np.exp(masked - legal_logsum)
    legal_loss = (normalizer - legal_logsum)[:, 0]
    ce = -logq[np.arange(len(x)), targets]
    kl = np.sum(p * (logp - logq), axis=1)
    parts = np.array([np.mean(ce * ce_mass), np.mean(kl * kl_mass),
                      np.mean(legal_loss * legal_mass)], np.float64) / 8
    teacher = q.copy()
    teacher[np.arange(len(x)), targets] -= 1
    gradient = (teacher * ce_mass[:, None] + (q - p) * kl_mass[:, None]
                + (q - conditional) * legal_mass[:, None]) / (len(x) * 8)
    return float(parts.sum()), gradient, parts


def components(logits, targets, ce_mass, kl_mass, source, legal_mass, legal_actions):
    import mlx.core as mx
    normalizer = mx.logsumexp(logits, axis=1, keepdims=True)
    logq = logits - normalizer
    logp = source - mx.logsumexp(source, axis=1, keepdims=True)
    p = mx.stop_gradient(mx.exp(logp))
    ce = -mx.take_along_axis(logq, targets[:, None], axis=1).squeeze(1)
    kl = mx.sum(p * (mx.stop_gradient(logp) - logq), axis=1)
    mask = mx.where(legal_mass[:, None] > 0, legal_actions, mx.ones_like(legal_actions))
    legal_logsum = mx.logsumexp(mx.where(mask, logits, -float('inf')), axis=1, keepdims=True)
    legal_loss = (normalizer - legal_logsum).squeeze(1)
    return mx.stack([mx.mean(ce * ce_mass) / 8, mx.mean(kl * kl_mass) / 8,
                     mx.mean(legal_loss * legal_mass) / 8])


def head_gradient(decoder, features, targets, ce_mass, kl_mass, source, legal_mass, legal_actions):
    import mlx.core as mx
    if (targets.shape != (features.shape[0],) or ce_mass.shape != targets.shape
            or kl_mass.shape != targets.shape or legal_mass.shape != targets.shape
            or source.shape != (features.shape[0], 3) or legal_actions.shape != source.shape):
        raise ValueError('Separate aligned fitting-loss labels and weights required')
    if bool(mx.any((legal_mass > 0) & ~mx.any(legal_actions, axis=1)).item()):
        raise ValueError('Positive legal-set loss mass requires a legal action')
    def objective(x):
        return mx.sum(components(decoder(x), targets, ce_mass, kl_mass, source, legal_mass, legal_actions))
    loss, gradient = mx.value_and_grad(objective)(features)
    parts = components(decoder(features), targets, ce_mass, kl_mass, source, legal_mass, legal_actions)
    mx.eval(loss, gradient, parts)
    return loss, gradient, parts
