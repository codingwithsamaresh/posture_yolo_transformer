"""Per-frame token construction (numpy only).

token(frame) = [ boxes (6 classes x [present, conf, cx, cy, w, h]) | PCA(pooled YOLO features) | pressure stats ]
"""
import numpy as np

N_CLS = 6
STAT_NAMES = ["mean_pressure", "contact_frac", "cop_row", "cop_col", "peak", "row_spread"]


def boxes_token(dets, R, n_cls=N_CLS):
    """dets: list (one per frame) of (n,6) arrays [x1,y1,x2,y2,conf,cls] in canvas pixels.
    Keeps the most confident box per body-part class."""
    out = np.zeros((len(dets), n_cls * 6), np.float32)
    for i, d in enumerate(dets):
        d = np.asarray(d, np.float32).reshape(-1, 6)
        for c in range(n_cls):
            m = d[:, 5] == c
            if m.any():
                row = d[m][np.argmax(d[m, 4])]
                g = R.canvas_xyxy_to_grid_norm(row[:4])[0]
                out[i, c * 6:(c + 1) * 6] = [1.0, row[4], g[0], g[1], g[2], g[3]]
    return out


def pressure_stats(frames, vmax=100.0, thr=3.0):
    """frames (N,64,27) -> (N,6)."""
    x = np.clip(frames.astype(np.float32), 0, None)
    N, H, W = x.shape
    tot = x.sum((1, 2)) + 1e-6
    rows = np.arange(H, dtype=np.float32)[None, :]
    cols = np.arange(W, dtype=np.float32)[None, :]
    pr, pc = x.sum(2), x.sum(1)
    cop_r = (pr * rows).sum(1) / tot
    cop_c = (pc * cols).sum(1) / tot
    spread = np.sqrt(np.maximum((pr * (rows - cop_r[:, None]) ** 2).sum(1) / tot, 0))
    return np.stack([x.mean((1, 2)) / vmax, (x > thr).mean((1, 2)), cop_r / H, cop_c / W,
                     x.max((1, 2)) / vmax, spread / H], 1).astype(np.float32)


class Pca:
    """Tiny PCA on standardised features (fit on training-subject frames only)."""

    def fit(self, X, k, max_rows=10000):
        X = X[:: max(1, len(X) // max_rows)].astype(np.float64)       # subsample first (memory/speed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        self.comp = vt[:k].T.astype(np.float32)
        return self

    def transform(self, X):
        return (((X.astype(np.float64) - self.mu) / self.sd).astype(np.float32)) @ self.comp


def assemble(tok, groups, pcas=None):
    """tok: dict with boxes/feat/stats/pix -> (Ntot, D) float32 token matrix (un-standardised).
    groups subset of: boxes (YOLO boxes+conf), feat (PCA of pooled YOLO features), stats (pressure summary),
    pix (PCA of raw log-pressure pixels; NO YOLO involved -> baseline)."""
    parts = []
    if "boxes" in groups:
        parts.append(tok["boxes"])
    if "feat" in groups:
        parts.append(pcas["feat"].transform(tok["feat"]))
    if "pix" in groups:
        parts.append(pcas["pix"].transform(tok["pix"]))
    if "stats" in groups:
        parts.append(tok["stats"])
    return np.concatenate(parts, 1).astype(np.float32)
