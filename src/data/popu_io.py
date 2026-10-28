"""Loading the PoPu pressure-map dataset (Tactilus sheet placed over the mattress).

Each frame is a 64 (rows, head->feet) x 27 (cols) pressure grid.
Layout: <root>/tactilus_data/<subject>/<pose><variation>_<shift>.json with 10 snapshots each
        pose in {supine,left,right,prone}, variation 1..7, shift 0..2
        -> 60 subjects * 28 poses * 3 shifts * 10 snapshots = 50,400 frames.
"""
from __future__ import annotations
import json, zipfile
from pathlib import Path
import numpy as np

POSES = ["supine", "left", "right", "prone"]
POSE_TO_ID = {p: i for i, p in enumerate(POSES)}
N_VAR, N_SHIFT, N_SNAP = 7, 3, 10
BODY_PARTS = ["Head", "Lower_Arm", "Lower_Leg", "Torso", "Upper_Arm", "Upper_Leg"]
# COCO category ids in the PoPu annotation files -> 0-based YOLO class ids
COCO_CAT_TO_CLS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
ROWS, COLS = 64, 27


# --------------------------------------------------------------------------- extraction
def find_data_root(path) -> Path | None:
    """Return the directory that contains tactilus_data/ (searching a few levels)."""
    path = Path(path)
    if not path.exists():
        return None
    if (path / "tactilus_data").is_dir():
        return path
    for p in sorted(path.rglob("tactilus_data")):
        if p.is_dir():
            return p.parent
    return None


def ensure_extracted(src, out_dir) -> Path:
    """`src` may be: PoPu-main.zip (outer), PoPu_data.zip, or an already extracted folder."""
    src, out_dir = Path(src), Path(out_dir)
    root = find_data_root(out_dir)
    if root is not None and (root / "segmentation_data").is_dir():
        return root
    if src.is_dir():
        root = find_data_root(src)
        if root is None:
            raise FileNotFoundError(f"No tactilus_data/ under {src}")
        return root
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] extracting {src} -> {out_dir}")
    with zipfile.ZipFile(src) as zf:
        names = zf.namelist()
        inner = [n for n in names if n.endswith("PoPu_data.zip")]  # NOT *_no_segments.zip
        if inner:  # outer repo zip
            zf.extract(inner[0], out_dir)
            inner_path = out_dir / inner[0]
            with zipfile.ZipFile(inner_path) as zf2:
                zf2.extractall(out_dir)
        else:
            zf.extractall(out_dir)
    root = find_data_root(out_dir)
    if root is None or not (root / "segmentation_data").is_dir():
        raise FileNotFoundError(
            "Could not find tactilus_data/ and segmentation_data/ after extraction. "
            "Pass PoPu-main.zip (or the extracted PoPu_data folder) via --data.")
    return root


# --------------------------------------------------------------------------- loading
def list_subjects(root: Path) -> list:
    subs = [int(p.name) for p in (root / "tactilus_data").iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(subs)


def load_all_frames(root: Path, subjects=None, verbose: bool = True) -> dict:
    """Load every labelled frame. Returns dict of aligned arrays (N = number of frames).

    frames   (N,64,27) uint16     raw sensor values
    subject  (N,) int16
    pose     (N,) int8            0..3 index in POSES
    var      (N,) int8            1..7  (pose variation)
    shift    (N,) int8            0..2  (file index: initial pose, 1st shift, 2nd shift)
    snap     (N,) int8            0..9  (snapshot order inside file = time order)
    seq_id   (N,) int32           unique id of (subject,pose,var); frames ordered by shift, snap
    meta_subjects (S,), meta_values (S,3) = sex(1=male), height cm, weight kg
    """
    subjects = subjects or list_subjects(root)
    F, S, P, V, K, N_, SEQ = [], [], [], [], [], [], []
    meta = {}
    seq_counter = 0
    missing = 0
    for s in subjects:
        sdir = root / "tactilus_data" / str(s)
        for pi, pose in enumerate(POSES):
            for var in range(1, N_VAR + 1):
                got_any = False
                for k in range(N_SHIFT):
                    f = sdir / f"{pose}{var}_{k}.json"
                    if not f.exists():
                        missing += 1
                        continue
                    with open(f) as fh:
                        d = json.load(fh)
                    meta[s] = (1 if str(d.get("sex", "")).lower().startswith("m") else 0,
                               float(d.get("height", np.nan)), float(d.get("weight", np.nan)))
                    rows, cols = int(d.get("tactilus_rows", ROWS)), int(d.get("tactilus_columns", COLS))
                    snaps = d["snapshots"]
                    for key in sorted(snaps, key=lambda x: int(x)):
                        a = np.asarray(snaps[key]["tactilus_readings"], dtype=np.float32).reshape(rows, cols)
                        F.append(np.clip(a, 0, 65535).astype(np.uint16))
                        S.append(s); P.append(pi); V.append(var); K.append(k); N_.append(int(key)); SEQ.append(seq_counter)
                    got_any = True
                if got_any:
                    seq_counter += 1
        if verbose:
            print(f"[data] subject {s:>2} loaded ({len(F)} frames so far)", end="\r")
    if verbose:
        print()
        if missing:
            print(f"[data] WARNING: {missing} expected files were missing")
    subj_sorted = sorted(meta)
    return dict(
        frames=np.stack(F), subject=np.array(S, np.int16), pose=np.array(P, np.int8),
        var=np.array(V, np.int8), shift=np.array(K, np.int8), snap=np.array(N_, np.int8),
        seq_id=np.array(SEQ, np.int32),
        meta_subjects=np.array(subj_sorted, np.int16),
        meta_values=np.array([meta[s] for s in subj_sorted], np.float32),
    )


# --------------------------------------------------------------------------- annotations
def load_boxes(root: Path, subject: int, pose: str, var: int):
    """List of (cls, x, y, w, h) in pixel units of the 27x64 grid, or None if missing/empty."""
    f = root / "segmentation_data" / str(subject) / f"{subject}_{pose}{var}_annotations.coco.json"
    if not f.exists():
        return None
    with open(f) as fh:
        d = json.load(fh)
    out = []
    for a in d.get("annotations", []):
        cls = COCO_CAT_TO_CLS.get(a["category_id"])
        if cls is None:
            continue
        x, y, w, h = a["bbox"]
        out.append((cls, x, y, w, h))
    return out or None


def reference_image(root: Path, subject: int, pose: str, var: int):
    """Colourised preview the annotators labelled (grey, 64x27), or None."""
    f = root / "segmentation_data" / str(subject) / "images" / f"{subject}_{pose}{var}.jpg"
    if not f.exists():
        return None
    from PIL import Image
    return np.asarray(Image.open(f).convert("L"), dtype=np.float32)


def find_anchor_shift(frames_seq: np.ndarray, shifts: np.ndarray, ref: np.ndarray) -> int:
    """Annotators labelled ONE image per (subject,pose,variation). Find which shift file (0/1/2)
    it was rendered from by correlating the colourised image with every frame."""
    best_k, best_c = 0, -2.0
    r = ref.ravel() - ref.mean()
    for k in np.unique(shifts):
        fr = frames_seq[shifts == k].reshape(-1, ref.size).astype(np.float32)
        fr = fr - fr.mean(1, keepdims=True)
        c = (fr @ r) / (np.linalg.norm(fr, axis=1) * np.linalg.norm(r) + 1e-8)
        m = float(np.mean(c))
        if m > best_c:
            best_c, best_k = m, int(k)
    return best_k
