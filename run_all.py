"""Run the whole pipeline.  Example (Colab):
    python run_all.py --data /content/PoPu-main.zip --work /content/work            # full 5-fold run
    python run_all.py --data /content/PoPu-main.zip --work /content/work_dbg --debug  # ~10 min smoke test
    python run_all.py --data ... --folds 0                                           # single fold only
"""
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(HERE / "scripts" / script), *map(str, args)]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--work", default="/content/work")
    ap.add_argument("--debug", action="store_true"); ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--skip_prep", action="store_true"); ap.add_argument("--skip_yolo", action="store_true")
    ap.add_argument("--no_ablation", action="store_true"); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    common = ["--work", a.work] + (["--debug"] if a.debug else [])
    fold_args = (["--folds", *map(str, a.folds)] if a.folds is not None else [])
    sets = (["--set", *a.set] if a.set else [])
    if not a.skip_prep:
        run("01_prepare_data.py", "--data", a.data, *common, *sets)
    if not a.skip_yolo:
        run("02_train_yolo.py", *common, *fold_args, *sets)
    run("03_extract_tokens.py", *common, *fold_args, *sets)
    run("04_train_eval.py", *common, *fold_args, *sets, *(["--no_ablation"] if a.no_ablation else []))
    run("06_rl_demo.py", *common, *fold_args)
    run("05_aggregate.py", *common)
    run("99_collect_results.py", "--work", a.work)


if __name__ == "__main__":
    main()
