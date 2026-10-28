"""Step 5: aggregate per-fold results -> summary.json / summary.md / figures (mean +- std over folds)."""
import argparse, sys, json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, work
from src.uncertainty import calibration as C

POSES = ["supine", "left", "right", "prone"]
METHODS = ["single", "mc_dropout", "ensemble"]


def ms(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))


def fmt(t, d=3):
    return f"{t[0]:.{d}f} ± {t[1]:.{d}f}"


def pool_reliability(rels):
    nb = len(rels[0]); out = []
    for b in range(nb):
        cnt = sum(r[b][2] for r in rels)
        conf = sum(r[b][2] * r[b][3] for r in rels) / cnt if cnt else 0
        acc = sum(r[b][2] * r[b][4] for r in rels) / cnt if cnt else 0
        out.append((rels[0][b][0], rels[0][b][1], cnt, conf, acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None); ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    cfg = load_config(debug=a.debug, work_dir=a.work)
    W = work(cfg); R = W / "results"; FIG = R / "figures"; FIG.mkdir(parents=True, exist_ok=True)
    fs = sorted(glob.glob(str(R / "fold[0-9]*.json")))
    folds = [json.load(open(f)) for f in fs]
    if not folds:
        sys.exit("no fold results found")
    print(f"[aggregate] {len(folds)} folds: {[f['fold'] for f in folds]}")
    S = dict(n_folds=len(folds), folds=[f["fold"] for f in folds], posture={}, transition={})
    md = [f"# Results summary ({len(folds)} leave-subjects-out folds, mean ± std over folds)\n"]

    # ---------------- YOLO detection
    ym = [json.load(open(p)) for p in sorted(glob.glob(str(W / "yolo_runs" / "fold*_test_metrics.json")))]
    if ym:
        S["yolo"] = dict(map50=ms([m["map50"] for m in ym]), map50_95=ms([m["map50_95"] for m in ym]),
                         precision=ms([m["precision"] for m in ym]), recall=ms([m["recall"] for m in ym]))
        md += ["## Stage 1 - YOLO body-part detection (held-out subjects)\n", "| mAP50 | mAP50-95 | precision | recall |", "|---|---|---|---|",
               f"| {fmt(S['yolo']['map50'])} | {fmt(S['yolo']['map50_95'])} | {fmt(S['yolo']['precision'])} | {fmt(S['yolo']['recall'])} |\n"]
        cls = {}
        for m in ym:
            for k, v in m.get("per_class_map50_95", {}).items():
                cls.setdefault(k, []).append(v)
        if cls:
            md += ["| class | mAP50-95 |", "|---|---|"] + [f"| {k} | {fmt(ms(v))} |" for k, v in cls.items()] + [""]

    # ---------------- posture
    keys = ["acc", "macro_f1", "nll", "brier", "ece", "mce", "auroc_err_entropy", "aurc_entropy", "acc@80%cov"]
    md += ["## Stage 2 - posture classification (supine / left / right / prone)\n",
           "| method | calib. | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for m in METHODS:
        for v in ("raw", "ts"):
            row = {k: ms([f["posture"][m][v].get(k) for f in folds]) for k in keys}
            S["posture"][f"{m}_{v}"] = row
            md.append(f"| {m} | {'temp-scaled' if v=='ts' else 'raw'} | " + " | ".join(fmt(row[k]) for k in keys) + " |")
    md.append("")
    tkeys = ["acc", "auroc", "ap", "nll", "ece", "tpr_synth", "fpr_real_stable", "fpr_real_shift"]
    md += ["## Posture-transition detection (synthetic transitions vs. real stable / real shifted windows)\n",
           "| method | calib. | " + " | ".join(tkeys) + " |", "|---|---|" + "---|" * len(tkeys)]
    for m in METHODS:
        for v in ("raw", "ts"):
            row = {k: ms([f["transition"][m][v].get(k) for f in folds]) for k in tkeys}
            S["transition"][f"{m}_{v}"] = row
            md.append(f"| {m} | {'temp-scaled' if v=='ts' else 'raw'} | " + " | ".join(fmt(row[k]) for k in tkeys) + " |")
    md.append("")

    # ---------------- breakdowns
    bd = {k: ms([f["breakdown"].get(k) for f in folds]) for k in
          ("acc_real_windows", "acc_synthetic_windows", "acc_real_spanning_shift")}
    S["breakdown"] = bd
    byc = {c: ms([f["breakdown"]["acc_by_class"].get(c) for f in folds]) for c in POSES}
    byv = {v: ms([f["breakdown"]["acc_by_variation"].get(str(v)) for f in folds]) for v in range(1, 8)}
    S["breakdown"]["by_class"], S["breakdown"]["by_variation"] = byc, byv
    md += ["## Breakdown (ensemble, temperature-scaled)\n", "| subset | accuracy |", "|---|---|"]
    md += [f"| {k} | {fmt(v)} |" for k, v in bd.items() if isinstance(v, tuple)]
    md += [f"| class: {c} | {fmt(v)} |" for c, v in byc.items()] + [f"| pose variation {v} | {fmt(x)} |" for v, x in byv.items()] + [""]
    subj_acc = {}
    for f in folds:
        subj_acc.update({int(k): v for k, v in f["breakdown"]["acc_by_subject"].items()})
    sa = np.array(list(subj_acc.values()))
    S["per_subject_acc"] = dict(n=len(sa), mean=float(sa.mean()), std=float(sa.std()), min=float(sa.min()),
                                worst_subjects={int(k): float(v) for k, v in sorted(subj_acc.items(), key=lambda kv: kv[1])[:5]})
    md.append(f"Per-subject accuracy over {len(sa)} unseen subjects: mean {sa.mean():.3f}, std {sa.std():.3f}, min {sa.min():.3f}; "
              f"worst: {S['per_subject_acc']['worst_subjects']}\n")

    # ---------------- ablation
    if all("ablation" in f for f in folds):
        names = list(folds[0]["ablation"].keys())
        full = {k: ms([f["posture"]["single"]["ts"][k] for f in folds]) for k in ("acc", "macro_f1", "nll", "ece")}
        full["trans_auroc"] = ms([f["transition"]["single"]["ts"]["auroc"] for f in folds])
        S["ablation"] = {"FULL (boxes+feat+stats, L=10, single model)": full}
        for n in names:
            S["ablation"][n] = {k: ms([f["ablation"][n][k] for f in folds]) for k in ("acc", "macro_f1", "nll", "ece", "trans_auroc")}
        md += ["## Ablation (single model, temperature-scaled)\n", "| variant | acc | macro-F1 | NLL | ECE | transition AUROC |", "|---|---|---|---|---|---|"]
        for n, r in S["ablation"].items():
            md.append(f"| {n} | {fmt(r['acc'])} | {fmt(r['macro_f1'])} | {fmt(r['nll'])} | {fmt(r['ece'])} | {fmt(r['trans_auroc'])} |")
        md.append("")
        fig, ax = plt.subplots(figsize=(9, 4.5)); nn_ = list(S["ablation"].keys())
        ax.barh(nn_, [S["ablation"][n]["acc"][0] for n in nn_], xerr=[S["ablation"][n]["acc"][1] for n in nn_], color="#4c72b0")
        ax.set_xlabel("posture accuracy (held-out subjects)"); ax.set_xlim(min(0.5, min(S["ablation"][n]["acc"][0] for n in nn_) - .05), 1.0)
        plt.tight_layout(); plt.savefig(FIG / "ablation_accuracy.png", dpi=150); plt.close()

    # ---------------- figures: reliability, confusion, per-subject, ECE by method, risk-coverage
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, key, title in zip(axs, ("reliability_raw", "reliability_ts"), ("deep ensemble - raw", "deep ensemble - temperature scaled")):
        rel = pool_reliability([f["posture"]["ensemble"][key] for f in folds])
        cs = [r[3] for r in rel if r[2] > 0]; ac = [r[4] for r in rel if r[2] > 0]; ct = [r[2] for r in rel if r[2] > 0]
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.scatter(cs, ac, s=[10 + 200 * c / sum(ct) for c in ct], c="#c44e52")
        ax.plot(cs, ac, c="#c44e52", lw=1); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(FIG / "reliability_posture.png", dpi=150); plt.close()

    cm = sum(np.array(f["confusion_ensemble_ts"]) for f in folds)
    fig, ax = plt.subplots(figsize=(4.8, 4.2)); cmn = cm / cm.sum(1, keepdims=True).clip(1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(POSES); ax.set_yticks(range(4)); ax.set_yticklabels(POSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); plt.tight_layout(); plt.savefig(FIG / "confusion_matrix.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 3.5)); ax.hist(sa, bins=15, color="#55a868"); ax.set_xlabel("per-subject accuracy (unseen subjects)")
    ax.set_ylabel("# subjects"); plt.tight_layout(); plt.savefig(FIG / "per_subject_accuracy.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4)); w = 0.35
    for i, v in enumerate(("raw", "ts")):
        vals = [S["posture"][f"{m}_{v}"]["ece"] for m in METHODS]
        ax.bar(np.arange(3) + i * w, [x[0] for x in vals], w, yerr=[x[1] for x in vals], label="temperature scaled" if v == "ts" else "raw")
    ax.set_xticks(np.arange(3) + w / 2); ax.set_xticklabels(["single", "MC-dropout", "deep ensemble"]); ax.set_ylabel("ECE (lower is better)")
    ax.legend(); plt.tight_layout(); plt.savefig(FIG / "ece_by_method.png", dpi=150); plt.close()

    ent, wrong = [], []
    for p in sorted(glob.glob(str(R / "preds_fold*.npz"))):
        d = np.load(p); ent.append(d["entropy"]); wrong.append(d["probs"].argmax(1) != d["y"])
    if ent:
        ent, wrong = np.concatenate(ent), np.concatenate(wrong); o = np.argsort(ent)
        cov = np.arange(1, len(o) + 1) / len(o); acc = 1 - np.cumsum(wrong[o]) / np.arange(1, len(o) + 1)
        fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(cov, acc); ax.set_xlabel("coverage (most confident first)"); ax.set_ylabel("accuracy on kept")
        ax.set_title("selective prediction by predictive entropy"); plt.tight_layout(); plt.savefig(FIG / "risk_coverage.png", dpi=150); plt.close()
        S["selective"] = C.selective_accuracy(ent, wrong)
        md.append("Selective prediction (pooled, ensemble+TS): " + ", ".join(f"{k}={v:.3f}" for k, v in S["selective"].items()) + "\n")

    # ---------------- RL (if present)
    rp = R / "rl_results.json"
    if rp.exists():
        rl = json.load(open(rp)); S["rl"] = rl
        md += ["## Stretch - toy RL surface policy (simulated environment)\n", "| policy | mean reward | peak pressure (norm.) | peak reduction vs static | adjustments/episode |", "|---|---|---|---|---|"]
        for k, v in rl["policies"].items():
            md.append(f"| {k} | {v['reward']:.3f} | {v['peak']:.3f} | {100*v['peak_reduction']:.1f}% | {v['adjust']:.2f} |")
        md.append("")
    (R / "summary.json").write_text(json.dumps(S, indent=1)); (R / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
