"""Sequence windows: real (stable posture, incl. real small body shifts) + synthetic posture transitions.

PoPu contains no recordings of people rolling from one posture to another, so transitions are SIMULATED:
real frames of posture A, then a smoothed cross-fade into real frames of posture B (same subject).
Real windows that span a 'shift' file boundary (subject slightly re-positioned, same posture) act as
hard negatives for the transition detector.
"""
import numpy as np
import cv2


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def make_synthetic(data, subjects, T, per_pair, pre_len, trans_len, seed):
    rng = np.random.RandomState(seed)
    frames, meta = [], []
    seqs = {}
    for sid in np.unique(data["seq_id"]):
        idx = np.where(data["seq_id"] == sid)[0]
        seqs[(int(data["subject"][idx[0]]), int(data["pose"][idx[0]]))] = seqs.get(
            (int(data["subject"][idx[0]]), int(data["pose"][idx[0]])), []) + [idx]
    for s in subjects:
        for a in range(4):
            for b in range(4):
                if a == b or (s, a) not in seqs or (s, b) not in seqs:
                    continue
                for _ in range(per_pair):
                    ia = seqs[(s, a)][rng.randint(len(seqs[(s, a)]))]
                    ib = seqs[(s, b)][rng.randint(len(seqs[(s, b)]))]
                    n_pre = rng.randint(pre_len[0], pre_len[1] + 1)
                    n_tr = rng.randint(trans_len[0], trans_len[1] + 1)
                    n_post = T - n_pre - n_tr
                    if n_post < 2:
                        n_post = 2; n_pre = T - n_tr - n_post
                    sa = rng.randint(0, len(ia) - n_pre + 1)
                    sb = rng.randint(0, len(ib) - n_post + 1)
                    A = data["frames"][ia[sa:sa + n_pre]].astype(np.float32)
                    B = data["frames"][ib[sb:sb + n_post]].astype(np.float32)
                    a_last, b_first = A[-1], B[0]
                    mid = []
                    for j in range(n_tr):
                        al = _smoothstep((j + 1) / (n_tr + 1))
                        f = (1 - al) * a_last + al * b_first
                        f = cv2.GaussianBlur(f, (3, 3), rng.uniform(0.6, 1.2))     # moving body smears
                        f *= rng.uniform(0.9, 1.05)
                        mid.append(f)
                    clip = np.concatenate([A, np.stack(mid), B], 0)[:T]
                    frames.append(clip)
                    meta.append((s, a, b, n_pre))
    if not frames:
        return dict(frames=np.zeros((0, T, 64, 27), np.float32), subject=np.zeros(0, np.int16),
                    src=np.zeros(0, np.int8), dst=np.zeros(0, np.int8), t_start=np.zeros(0, np.int8))
    m = np.array(meta)
    return dict(frames=np.stack(frames).astype(np.float32), subject=m[:, 0].astype(np.int16),
                src=m[:, 1].astype(np.int8), dst=m[:, 2].astype(np.int8), t_start=m[:, 3].astype(np.int8))


def build_windows(data, synth, T, stride):
    """Return dict of aligned arrays. Token index space = [real frames (N)] + [synthetic frames (M*T)]."""
    N = len(data["frames"])
    idxs, ys, trs, subs, kinds, vrs, spans, lasts = [], [], [], [], [], [], [], []
    for sid in np.unique(data["seq_id"]):
        fi = np.where(data["seq_id"] == sid)[0]
        for st in range(0, len(fi) - T + 1, stride):
            w = fi[st:st + T]
            idxs.append(w); ys.append(int(data["pose"][w[-1]])); trs.append(0)
            subs.append(int(data["subject"][w[0]])); kinds.append(0); vrs.append(int(data["var"][w[0]]))
            spans.append(int(data["shift"][w[0]] != data["shift"][w[-1]])); lasts.append(int(w[-1]))
    M = len(synth["frames"])
    for m in range(M):
        idxs.append(N + m * T + np.arange(T)); ys.append(int(synth["dst"][m])); trs.append(1)
        subs.append(int(synth["subject"][m])); kinds.append(1); vrs.append(0); spans.append(0); lasts.append(-1)
    return dict(idx=np.array(idxs, np.int64), y=np.array(ys, np.int64), trans=np.array(trs, np.int64),
                subject=np.array(subs, np.int16), kind=np.array(kinds, np.int8), var=np.array(vrs, np.int8),
                spans_shift=np.array(spans, np.int8), last_frame=np.array(lasts, np.int64))
