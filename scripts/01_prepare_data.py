"""Step 1: extract PoPu, cache frames, make leave-subjects-out folds, YOLO dataset, synthetic transitions, windows."""
import argparse, sys, json, time
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, work
from src.data import popu_io as io
from src.data.render import Renderer
from src.data.splits import make_folds, save_folds
from src.data.windows import make_synthetic, build_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="PoPu-main.zip | PoPu_data.zip | extracted folder")
    ap.add_argument("--work", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(debug=a.debug, work_dir=a.work, overrides=a.set)
    W = work(cfg)
    root = io.ensure_extracted(a.data, W / "raw")
    subjects = io.list_subjects(root)
    if cfg.data.max_subjects:
        subjects = subjects[: cfg.data.max_subjects]
    print(f"[prep] {len(subjects)} subjects")
    t0 = time.time()
    data = io.load_all_frames(root, subjects)
    N = len(data["frames"])
    print(f"[prep] {N} frames in {time.time()-t0:.0f}s, max value {data['frames'].max()}")
    np.savez_compressed(W / "frames.npz", **data)

    # ---- folds
    folds = make_folds(subjects, cfg.split.n_folds, cfg.split.n_val_subjects, cfg.seed)
    save_folds(folds, W / "folds.json")
    print("[prep] folds:", [(f["fold"], len(f["train"]), len(f["val"]), len(f["test"])) for f in folds])

    # ---- YOLO dataset (explicit canvas; labels only for the annotated shift file of each sequence)
    R = Renderer.from_cfg(cfg)
    img_dir, lab_dir = W / "yolo_data" / "images", W / "yolo_data" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lab_dir.mkdir(parents=True, exist_ok=True)
    rows = []      # (path, subject, frame_idx)
    n_noann = 0
    stride = cfg.yolo.frame_stride
    for sid in np.unique(data["seq_id"]):
        fi = np.where(data["seq_id"] == sid)[0]
        s, pi, var = int(data["subject"][fi[0]]), int(data["pose"][fi[0]]), int(data["var"][fi[0]])
        pose = io.POSES[pi]
        boxes = io.load_boxes(root, s, pose, var)
        ref = io.reference_image(root, s, pose, var)
        if boxes is None or ref is None:
            n_noann += 1
            continue
        k = io.find_anchor_shift(data["frames"][fi], data["shift"][fi], ref)
        sel = fi[data["shift"][fi] == k][::stride]
        lines = []
        for cls, x, y, w, h in boxes:
            cx, cy, bw, bh = R.grid_box_to_yolo(x, y, w, h)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            n_noann += 1
            continue
        for j in sel:
            name = f"s{s:02d}_{pose}{var}_k{k}_n{int(data['snap'][j])}"
            cv2.imwrite(str(img_dir / f"{name}.png"), R.one(data["frames"][j]))
            (lab_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
            rows.append((str((img_dir / f"{name}.png").resolve()), s, int(j)))
    print(f"[prep] YOLO images: {len(rows)} ({n_noann} sequences without usable annotation skipped)")
    names = {i: n for i, n in enumerate(io.BODY_PARTS)}
    for fo in folds:
        f = fo["fold"]
        d = W / "yolo_data" / f"fold{f}"
        d.mkdir(exist_ok=True)
        for split in ("train", "val", "test"):
            S = set(fo[split])
            (d / f"{split}.txt").write_text("\n".join(p for p, s, _ in rows if s in S) + "\n")
        yaml_txt = f"path: {d.resolve()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n" + \
                   "".join(f"  {i}: {n}\n" for i, n in names.items())
        (d / "data.yaml").write_text(yaml_txt)

    # ---- synthetic transitions + windows (all subjects; the fold decides who uses what)
    T = cfg.seq.window
    synth = make_synthetic(data, subjects, T, cfg.seq.synth_per_pair, cfg.seq.pre_len, cfg.seq.trans_len, cfg.seed)
    np.savez_compressed(W / "synth.npz", **synth)
    win = build_windows(data, synth, T, cfg.seq.stride)
    np.savez_compressed(W / "windows.npz", **win)
    print(f"[prep] synthetic clips: {len(synth['frames'])}, windows: {len(win['y'])} "
          f"(real {int((win['kind']==0).sum())}, synthetic {int((win['kind']==1).sum())})")
    json.dump(dict(n_frames=int(N), n_subjects=len(subjects), n_yolo_images=len(rows)), open(W / "prep_info.json", "w"))
    print("[prep] done")


if __name__ == "__main__":
    main()
