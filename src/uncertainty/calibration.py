"""Calibration + uncertainty metrics (numpy only).

Conventions: logits arrays are (S, N, C): S stochastic members (ensemble members or MC-dropout passes),
N samples, C classes. The binary transition head is represented as 2-class logits [0, z].
"""
import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def predictive(logits, T=1.0):
    """Mean of member softmaxes -> (N,C)."""
    return softmax(logits / T, -1).mean(0)


def entropy(p, eps=1e-12):
    return -(p * np.log(p + eps)).sum(-1)


def mutual_information(logits, T=1.0):
    """BALD epistemic term: H[E p] - E H[p]. Zero for a single deterministic model."""
    P = softmax(logits / T, -1)
    return entropy(P.mean(0)) - entropy(P).mean(0)


def accuracy(p, y):
    return float((p.argmax(1) == y).mean())


def nll(p, y, eps=1e-12):
    return float(-np.log(p[np.arange(len(y)), y] + eps).mean())


def brier(p, y):
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(1).mean())


def macro_f1(p, y):
    C = p.shape[1]
    pred = p.argmax(1)
    f1 = []
    for c in range(C):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        d = 2 * tp + fp + fn
        if (y == c).sum() > 0:
            f1.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1))


def confusion(p, y):
    C = p.shape[1]
    m = np.zeros((C, C), int)
    for t, q in zip(y, p.argmax(1)):
        m[t, q] += 1
    return m


def reliability(p, y, n_bins=15):
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append((lo, hi, int(m.sum()), float(conf[m].mean()) if m.any() else 0.0,
                    float(correct[m].mean()) if m.any() else 0.0))
    return out


def ece(p, y, n_bins=15):
    n = len(y)
    return float(sum(c / n * abs(a - cf) for _, _, c, cf, a in reliability(p, y, n_bins) if c > 0))


def mce(p, y, n_bins=15):
    v = [abs(a - cf) for _, _, c, cf, a in reliability(p, y, n_bins) if c > 0]
    return float(max(v)) if v else 0.0


def auroc(score, positive):
    """Rank-based AUROC; `score` higher => more likely positive. Returns nan if one class missing."""
    positive = np.asarray(positive).astype(bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); s = np.asarray(score)[order]
    i = 0
    while i < len(s):                       # average ranks for ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(score, positive):
    positive = np.asarray(positive).astype(bool)
    if positive.sum() == 0:
        return float("nan")
    o = np.argsort(-np.asarray(score), kind="mergesort")
    tp = np.cumsum(positive[o]); prec = tp / np.arange(1, len(o) + 1)
    return float((prec * positive[o]).sum() / positive.sum())


def aurc(uncertainty, wrong):
    """Area under risk-coverage curve: keep the most certain samples first. Lower is better."""
    o = np.argsort(uncertainty, kind="mergesort")
    risk = np.cumsum(np.asarray(wrong, float)[o]) / np.arange(1, len(o) + 1)
    return float(risk.mean())


def selective_accuracy(uncertainty, wrong, coverages=(1.0, 0.9, 0.8, 0.7, 0.5)):
    o = np.argsort(uncertainty, kind="mergesort"); w = np.asarray(wrong, float)[o]
    return {f"acc@{int(c*100)}%cov": float(1 - w[: max(1, int(len(w) * c))].mean()) for c in coverages}


def fit_temperature(logits, y, grid=None):
    """1-D search for T minimising NLL of the member-averaged predictive on (val) data."""
    grid = grid if grid is not None else np.exp(np.linspace(np.log(0.2), np.log(8.0), 121))
    best = min(grid, key=lambda T: nll(predictive(logits, T), y))
    fine = np.exp(np.linspace(np.log(best / 1.1), np.log(best * 1.1), 41))
    return float(min(fine, key=lambda T: nll(predictive(logits, T), y)))


def summarize(logits, y, T=1.0, n_bins=15):
    """All posture-style metrics for one method (logits (S,N,C))."""
    p = predictive(logits, T)
    wrong = p.argmax(1) != y
    ent, mi = entropy(p), mutual_information(logits, T)
    d = dict(acc=accuracy(p, y), macro_f1=macro_f1(p, y), nll=nll(p, y), brier=brier(p, y),
             ece=ece(p, y, n_bins), mce=mce(p, y, n_bins), T=float(T),
             auroc_err_entropy=auroc(ent, wrong), auroc_err_maxprob=auroc(1 - p.max(1), wrong),
             aurc_entropy=aurc(ent, wrong), mean_entropy=float(ent.mean()))
    if logits.shape[0] > 1:
        d["auroc_err_mi"] = auroc(mi, wrong)
    d.update(selective_accuracy(ent, wrong))
    return d, p
