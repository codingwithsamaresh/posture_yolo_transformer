"""Step 2: fine-tune YOLO for body-part detection, one detector per leave-subjects-out fold."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, work
from src.models.yolo_stage import train_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None); ap.add_argument("--debug", action="store_true")
    ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(debug=a.debug, work_dir=a.work, overrides=a.set)
    W = work(cfg)
    for f in (a.folds if a.folds is not None else range(cfg.split.n_folds)):
        print(f"\n===== YOLO fold {f} =====")
        train_fold(cfg, f, W)


if __name__ == "__main__":
    main()
