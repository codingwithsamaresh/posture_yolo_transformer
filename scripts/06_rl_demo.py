"""Step 6 (stretch): toy RL policy on a simulated adjustable surface, driven by the calibrated posture classifier."""
import argparse, sys, json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, work
from src.data.splits import load_folds
from src.rl.surface_env import SurfaceEnv, QPolicy, zone_pressure, rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None); ap.add_argument("--debug", action="store_true")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    a = ap.parse_args()
    cfg = load_config(debug=a.debug, work_dir=a.work); rc = cfg.rl
    W = work(cfg); R = W / "results"
    fr = np.load(W / "frames.npz")
    frames, subj, pose = fr["frames"], fr["subject"], fr["pose"]
    P_all = zone_pressure(frames, rc.zone_edges, cfg.data.vmax)
    folds = load_folds(W / "folds.json")
    fold_ids = a.folds if a.folds is not None else [int(Path(p).stem.replace("preds_fold", "")) for p in sorted(glob.glob(str(R / "preds_fold*.npz")))]
    if not fold_ids:
        sys.exit('no preds_fold*.npz found - run 04_train_eval.py first')
    env = SurfaceEnv(rc.n_levels, rc.horizon)
    names = ["static (no-op)", "random", "heuristic (soften hottest zone)", "RL, posture-agnostic",
             "RL, true posture (oracle)", "RL, predicted posture", "RL, predicted posture + uncertainty gate"]
    acc = {n: dict(reward=[], peak=[], peak0=[], adjust=[]) for n in names}
    extra = dict(posture_acc=[], gate_rate=[], acc_kept=[], acc_gated=[])
    for f in fold_ids:
        rng = np.random.RandomState(cfg.seed + f)
        d = np.load(R / f"preds_fold{f}.npz")
        real = d["kind"] == 0
        probs, ent, lf, ytrue = d["probs"][real], d["entropy"][real], d["last_frame"][real], d["y"][real]
        tau = np.quantile(d["val_entropy"], rc.uncertainty_quantile)
        test_set = set(folds[f]["test"])
        tr_idx = np.where(~np.isin(subj, list(test_set)))[0]
        bins = np.quantile(P_all[tr_idx].max(1), [1 / 3, 2 / 3])
        print(f"[rl] fold {f}: training Q-policy on {len(tr_idx)} frames of {len(set(subj[tr_idx]))} non-test subjects")
        pol = QPolicy(env, rc.alpha, rc.gamma)
        pol.train(P_all[tr_idx], pose[tr_idx], rc.episodes, bins, rc.eps_start, rc.eps_end, rc.mask_posture_p, rng)
        greedy = lambda s, p, lv: pol.act(s)
        noop = lambda s, p, lv: 0
        rnd = lambda s, p, lv, r=np.random.RandomState(cfg.seed): int(r.randint(env.n_actions))
        def heur(s, p, lv):
            z = int(np.argmax(env.pressure(p, lv, 0)))
            return 2 * z + 1 if lv[z] > 0 else 0                 # action 2z+1 = soften zone z
        pred = probs.argmax(1); gated = ent > tau
        extra["posture_acc"].append(float((pred == ytrue).mean())); extra["gate_rate"].append(float(gated.mean()))
        extra["acc_kept"].append(float((pred == ytrue)[~gated].mean()) if (~gated).any() else float("nan"))
        extra["acc_gated"].append(float((pred == ytrue)[gated].mean()) if gated.any() else float("nan"))
        init_rng = np.random.RandomState(1000 + f)
        runs = {n: [] for n in names}
        for k in range(len(lf)):
            p = P_all[lf[k]]; lv0 = init_rng.randint(1, rc.n_levels - 1, size=4); yt = int(ytrue[k])
            obs = {"static (no-op)": (noop, yt), "random": (rnd, yt), "heuristic (soften hottest zone)": (heur, yt),
                   "RL, posture-agnostic": (greedy, 4), "RL, true posture (oracle)": (greedy, yt),
                   "RL, predicted posture": (greedy, int(pred[k])),
                   "RL, predicted posture + uncertainty gate": (greedy, 4 if gated[k] else int(pred[k]))}
            for n, (fn, o) in obs.items():
                c0, c1, pk1, pk0, adj = rollout(env, fn, p, yt, o, lv0, bins)
                runs[n].append((c0 - c1 - 0.01 * adj, pk1, pk0, adj))
        for n in names:
            r = np.array(runs[n]); acc[n]["reward"].append(r[:, 0].mean()); acc[n]["peak"].append(r[:, 1].mean())
            acc[n]["peak0"].append(r[:, 2].mean()); acc[n]["adjust"].append(r[:, 3].mean())
    out = dict(policies={}, posture_acc_on_windows=float(np.mean(extra["posture_acc"])), gate_rate=float(np.mean(extra["gate_rate"])),
               acc_when_kept=float(np.nanmean(extra["acc_kept"])), acc_when_gated=float(np.nanmean(extra["acc_gated"])),
               note="simulated, hand-designed environment; not clinically validated")
    p0 = np.mean(acc["static (no-op)"]["peak"])
    for n in names:
        out["policies"][n] = dict(reward=float(np.mean(acc[n]["reward"])), peak=float(np.mean(acc[n]["peak"])),
                                  peak_reduction=float(1 - np.mean(acc[n]["peak"]) / p0), adjust=float(np.mean(acc[n]["adjust"])),
                                  reward_std_over_folds=float(np.std(acc[n]["reward"])))
        print(f"  {n:<44} reward {out['policies'][n]['reward']:+.4f}  peak {out['policies'][n]['peak']:.3f}  "
              f"reduction {100*out['policies'][n]['peak_reduction']:5.1f}%  adjust/ep {out['policies'][n]['adjust']:.2f}")
    print(f"  posture acc {out['posture_acc_on_windows']:.3f}; gate rate {out['gate_rate']:.2f}; acc kept {out['acc_when_kept']:.3f} vs gated {out['acc_when_gated']:.3f}")
    (R / "rl_results.json").write_text(json.dumps(out, indent=1))
    (R / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.barh(names, [out["policies"][n]["peak_reduction"] * 100 for n in names], color="#8172b2")
    ax.set_xlabel("simulated peak-pressure reduction vs. no-op (%)"); plt.tight_layout(); plt.savefig(R / "figures" / "rl_peak_reduction.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
