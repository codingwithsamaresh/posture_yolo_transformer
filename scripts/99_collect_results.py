"""Bundle everything worth sending back (metrics, figures, predictions, small weights) into results_bundle.zip."""
import argparse, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, work


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", default=None); ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    W = work(load_config(debug=a.debug, work_dir=a.work))
    out = W / "results_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for pat in ("results/*.json", "results/*.md", "results/*.npz", "results/figures/*.png", "results/*.pt",
                    "yolo_runs/*_test_metrics.json", "folds.json", "prep_info.json"):
            for p in sorted(W.glob(pat)):
                z.write(p, p.relative_to(W))
        # YOLO training curves (small csv) if present
        for p in sorted(W.glob("yolo_runs/fold*/results.csv")):
            z.write(p, p.relative_to(W))
    print(f"[collect] wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
