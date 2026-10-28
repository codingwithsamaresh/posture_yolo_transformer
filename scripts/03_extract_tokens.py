"""Step 3: run each fold's detector on every real + synthetic frame -> per-frame tokens."""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, work
from src.data.render import Renderer
from src.data.tokens import boxes_token, pressure_stats
from src.models.yolo_stage import TokenExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None); ap.add_argument("--debug", action="store_true")
    ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(debug=a.debug, work_dir=a.work, overrides=a.set)
    W = work(cfg)
    R = Renderer.from_cfg(cfg)
    real = np.load(W / "frames.npz")["frames"]
    syn = np.load(W / "synth.npz")["frames"]
    allf = np.concatenate([real.astype(np.float32), syn.reshape(-1, 64, 27)], 0)   # token index space
    print(f"[tokens] {len(real)} real + {len(syn.reshape(-1,64,27))} synthetic frames")
    stats = pressure_stats(allf, cfg.data.vmax)
    for f in (a.folds if a.folds is not None else range(cfg.split.n_folds)):
        t0 = time.time()
        ext = TokenExtractor(W / "yolo_runs" / f"fold{f}_best.pt", R, cfg.yolo.det_conf, cfg.yolo.det_iou)
        dets, feat = ext.run(allf, cfg.yolo.extract_batch)
        boxes = boxes_token(dets, R)
        np.savez_compressed(W / f"tokens_fold{f}.npz", boxes=boxes, feat=feat, stats=stats)
        print(f"[tokens] fold {f}: boxes {boxes.shape}, feat {feat.shape}, "
              f"mean #parts present {boxes[:, ::6].sum(1).mean():.2f}  ({time.time()-t0:.0f}s)")
        del ext


if __name__ == "__main__":
    main()
