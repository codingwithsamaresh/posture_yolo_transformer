"""Step 4: per fold -> train transformer ensemble on YOLO tokens, MC-dropout, temperature scaling,
metrics (accuracy, F1, NLL, Brier, ECE, error-detection AUROC, transition detection), ablations."""
import argparse, sys, json, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, work
from src.data.splits import load_folds
from src.data.tokens import Pca, assemble
from src.models.transformer import train_one, predict_logits
from src.uncertainty import calibration as C

POSES = ["supine", "left", "right", "prone"]


def log2(lt):                                   # binary logit z -> 2-class logits [0, z]
    return np.stack([np.zeros_like(lt), lt], -1)


def posture_report(lg, y, T, nb):
    d, p = C.summarize(lg, y, T, nb)
    return d, p


def trans_report(lt, ytr, kind, spans, T, nb):
    d, p = C.summarize(log2(lt), ytr, T, nb)
    p1 = p[:, 1]
    pred = p1 > 0.5
    d["auroc"] = C.auroc(p1, ytr); d["ap"] = C.average_precision(p1, ytr)
    d["tpr_synth"] = float(pred[ytr == 1].mean()) if (ytr == 1).any() else float("nan")
    real = kind == 0
    d["fpr_real_stable"] = float(pred[real & (spans == 0)].mean()) if (real & (spans == 0)).any() else float("nan")
    d["fpr_real_shift"] = float(pred[real & (spans == 1)].mean()) if (real & (spans == 1)).any() else float("nan")
    return d, p1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None); ap.add_argument("--debug", action="store_true")
    ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no_ablation", action="store_true")
    a = ap.parse_args()
    cfg = load_config(debug=a.debug, work_dir=a.work, overrides=a.set)
    W = work(cfg)
    (W / "results").mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tc, nb = cfg.transformer, cfg.eval.ece_bins
    fr = np.load(W / "frames.npz")
    real_subj = fr["subject"]; N = len(real_subj)
    syn = np.load(W / "synth.npz")["frames"].reshape(-1, 64, 27)
    allf = np.concatenate([fr["frames"].astype(np.float32), syn], 0)
    pix = (np.log1p(np.clip(allf, 0, None)) / np.log1p(cfg.data.vmax)).reshape(len(allf), -1).astype(np.float32)
    del syn
    win = {k: v for k, v in np.load(W / "windows.npz").items()}
    folds = load_folds(W / "folds.json")
    T_win = cfg.seq.window

    for f in (a.folds if a.folds is not None else range(cfg.split.n_folds)):
        t0 = time.time()
        fo = folds[f]
        tk = np.load(W / f"tokens_fold{f}.npz")
        tok = dict(boxes=tk["boxes"], feat=tk["feat"], stats=tk["stats"], pix=pix)
        tr_mask = np.isin(real_subj, fo["train"])
        pcas = {"feat": Pca().fit(tok["feat"][:N][tr_mask], cfg.tokens.pca_dim),
                "pix": Pca().fit(pix[:N][tr_mask], cfg.tokens.pca_dim)}
        ids = {s: np.where(np.isin(win["subject"], fo[s]))[0] for s in ("train", "val", "test")}
        print(f"\n===== fold {f}: windows train/val/test = {len(ids['train'])}/{len(ids['val'])}/{len(ids['test'])} =====")

        def make_tok(groups):
            X = assemble(tok, groups, pcas)
            mu, sd = X[:N][tr_mask].mean(0), X[:N][tr_mask].std(0) + 1e-6      # train-subject statistics only
            return torch.from_numpy((X - mu) / sd).float().to(dev)

        def run(groups, L, members, seed0=0):
            tt = make_tok(groups)
            models = [train_one(tt, win, ids["train"], ids["val"], tc, cfg.seed + seed0 + m, L, dev) for m in range(members)]
            return tt, models

        # ------------------------------------------------------------ main model
        groups, L = cfg.tokens.groups, T_win
        tt, models = run(groups, L, tc.ensemble)
        for m, mod in enumerate(models):
            torch.save(mod.state_dict(), W / "results" / f"transformer_fold{f}_m{m}.pt")

        def logits(split, kind):
            if kind == "single":
                return predict_logits(models[0], tt, win, ids[split], L, dev)
            if kind == "mc_dropout":
                return predict_logits(models[0], tt, win, ids[split], L, dev, mc=True, S=tc.mc_samples)
            outs = [predict_logits(m, tt, win, ids[split], L, dev) for m in models]      # deep ensemble
            return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])

        yv, yt = win["y"][ids["val"]], win["y"][ids["test"]]
        trv, trt = win["trans"][ids["val"]], win["trans"][ids["test"]]
        kind_t, span_t = win["kind"][ids["test"]], win["spans_shift"][ids["test"]]
        res = dict(fold=f, test_subjects=fo["test"], val_subjects=fo["val"], train_subjects=fo["train"],
                   n_test_windows=int(len(ids["test"])), token_dim=int(tt.shape[1]),
                   posture={}, transition={})
        store = {}
        for meth in ("single", "mc_dropout", "ensemble"):
            pv, tv = logits("val", meth); pt, tt_ = logits("test", meth)
            Tp, Tt = C.fit_temperature(pv, yv), C.fit_temperature(log2(tv), trv)
            raw, _ = posture_report(pt, yt, 1.0, nb); ts, pts = posture_report(pt, yt, Tp, nb)
            res["posture"][meth] = dict(raw=raw, ts=ts)
            traw, _ = trans_report(tt_, trt, kind_t, span_t, 1.0, nb); tts, p1 = trans_report(tt_, trt, kind_t, span_t, Tt, nb)
            res["transition"][meth] = dict(raw=traw, ts=tts)
            print(f"  [{meth:>10}] posture acc {ts['acc']:.4f} F1 {ts['macro_f1']:.4f} ECE {raw['ece']:.4f}->{ts['ece']:.4f} "
                  f"| transition AUROC {tts['auroc']:.4f} ECE {traw['ece']:.4f}->{tts['ece']:.4f}")
            store[meth] = (pv, pt, Tp, pts, p1)
            if meth == "ensemble":
                res["posture"][meth]["reliability_raw"] = C.reliability(C.predictive(pt, 1.0), yt, nb)
                res["posture"][meth]["reliability_ts"] = C.reliability(pts, yt, nb)
                res["confusion_ensemble_ts"] = C.confusion(pts, yt).tolist()
                res["transition"][meth]["reliability_ts"] = C.reliability(C.predictive(log2(tt_), Tt), trt, nb)

        # ------------------------------------------------------------ breakdowns (ensemble + TS)
        pv, pt, Tp, pts, _ = store["ensemble"]
        corr = pts.argmax(1) == yt
        res["breakdown"] = dict(
            acc_real_windows=float(corr[kind_t == 0].mean()) if (kind_t == 0).any() else None,
            acc_synthetic_windows=float(corr[kind_t == 1].mean()) if (kind_t == 1).any() else None,
            acc_real_spanning_shift=float(corr[(kind_t == 0) & (span_t == 1)].mean()) if ((kind_t == 0) & (span_t == 1)).any() else None,
            acc_by_variation={int(v): float(corr[(kind_t == 0) & (win["var"][ids["test"]] == v)].mean())
                              for v in range(1, 8) if ((kind_t == 0) & (win["var"][ids["test"]] == v)).any()},
            acc_by_subject={int(s): float(corr[win["subject"][ids["test"]] == s].mean()) for s in fo["test"]
                            if (win["subject"][ids["test"]] == s).any()},
            acc_by_class={POSES[c]: float(corr[yt == c].mean()) for c in range(4) if (yt == c).any()})

        # ------------------------------------------------------------ save predictions for the RL stage
        mi = C.mutual_information(pt, Tp)
        ent_val = C.entropy(C.predictive(store["ensemble"][0], Tp))
        np.savez_compressed(W / "results" / f"preds_fold{f}.npz", win_ids=ids["test"], probs=pts,
                            probs_raw=C.predictive(pt, 1.0), entropy=C.entropy(pts), mi=mi, y=yt,
                            kind=kind_t, last_frame=win["last_frame"][ids["test"]], subject=win["subject"][ids["test"]],
                            trans_prob=store["ensemble"][4], val_entropy=ent_val)

        # ------------------------------------------------------------ ablations (1 model, deterministic + TS)
        if cfg.eval.ablations and not a.no_ablation:
            variants = [("pix_only (no YOLO)", ["pix"], T_win), ("pix+stats (no YOLO)", ["pix", "stats"], T_win),
                        ("stats_only", ["stats"], T_win), ("yolo_boxes_only", ["boxes"], T_win),
                        ("yolo_feat_only", ["feat"], T_win), ("yolo_boxes+feat", ["boxes", "feat"], T_win),
                        ("full_L=1 (single frame)", list(groups), 1), ("full_L=5", list(groups), max(1, T_win // 2))]
            ab = {}
            for name, g, Lv in variants:
                t2, ms = run(g, Lv, 1, seed0=100)
                pv_, tv_ = predict_logits(ms[0], t2, win, ids["val"], Lv, dev)
                pt_, ttr_ = predict_logits(ms[0], t2, win, ids["test"], Lv, dev)
                Tp_, Tt_ = C.fit_temperature(pv_, yv), C.fit_temperature(log2(tv_), trv)
                d1, _ = posture_report(pt_, yt, Tp_, nb); d2, _ = trans_report(ttr_, trt, kind_t, span_t, Tt_, nb)
                ab[name] = dict(acc=d1["acc"], macro_f1=d1["macro_f1"], nll=d1["nll"], ece=d1["ece"],
                                trans_auroc=d2["auroc"], trans_ece=d2["ece"], token_dim=int(t2.shape[1]))
                print(f"  [ablation] {name:<26} acc {d1['acc']:.4f} F1 {d1['macro_f1']:.4f} ECE {d1['ece']:.4f} | trans AUROC {d2['auroc']:.4f}")
                del t2, ms
            res["ablation"] = ab
        res["runtime_s"] = round(time.time() - t0)
        (W / "results" / f"fold{f}.json").write_text(json.dumps(res, indent=1))
        print(f"[fold {f}] done in {res['runtime_s']}s")


if __name__ == "__main__":
    main()
