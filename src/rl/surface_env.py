"""Toy simulated adjustable support surface (stretch goal).

NOT a validated biomechanical model: a hand-designed environment to show how a calibrated posture
prediction (+ its uncertainty) can drive a control policy.

* The mat is split into Z zones along the body axis (head/shoulders, torso/lumbar, hips/thighs, legs/heels).
* Each zone has a firmness level 0 (soft) .. n_levels-1 (firm). Action = soften/firm one zone, or no-op.
* Zone peak pressure p_z comes from the REAL measured frame (95th percentile inside the zone band).
* Softening a zone lets the body sink in -> its peak pressure drops by `gain[posture]`*levels, a quarter of the
  relieved load is pushed onto each neighbouring zone (load is partly conserved). Side-lying (small contact area)
  gains more from immersion than supine/prone (larger area).
* Cost = max zone pressure + 0.5*mean zone pressure + penalty for bottoming-out / over-firm zones
  + penalty for abrupt firmness steps between neighbouring zones.
"""
import numpy as np

GAIN = np.array([0.10, 0.16, 0.16, 0.08])      # supine, left, right, prone
N_POSE_OBS = 5                                   # 4 postures + 'unknown'


def zone_pressure(frames, edges, vmax):
    """frames (N,64,27) -> (N,Z) normalised 95th-percentile pressure per zone."""
    x = frames.astype(np.float32)
    return np.stack([np.percentile(x[:, a:b].reshape(len(x), -1), 95, axis=1)
                     for a, b in zip(edges[:-1], edges[1:])], 1) / vmax


class SurfaceEnv:
    def __init__(self, n_levels=5, horizon=6):
        self.L, self.H, self.Z = n_levels, horizon, 4
        self.n_actions = 1 + 2 * self.Z
        self.mid = (n_levels - 1) / 2

    # ---- physics
    def pressure(self, p, levels, posture):
        g = GAIN[posture]
        p1 = p * (1 + g * (levels - self.mid))
        delta = p - p1
        spread = np.zeros_like(p)
        spread[1:] += 0.25 * delta[:-1]
        spread[:-1] += 0.25 * delta[1:]
        return np.clip(p1 + spread, 0, None)

    def cost(self, p, levels, posture):
        q = self.pressure(p, levels, posture)
        extreme = ((levels == 0) | (levels == self.L - 1)).sum()
        steps = np.maximum(np.abs(np.diff(levels)) - 1, 0).sum()
        return float(q.max() + 0.5 * q.mean() + 0.03 * extreme + 0.04 * steps), float(q.max())

    def apply(self, levels, a):
        lv = levels.copy()
        if a > 0:
            z, up = (a - 1) // 2, (a - 1) % 2            # odd: soften (-1), even: firm (+1)
            lv[z] = int(np.clip(lv[z] + (1 if up else -1), 0, self.L - 1))
        return lv

    # ---- observation -> discrete state
    def encode(self, pose_obs, p, levels, bins):
        hot = int(np.argmax(p)); b = int(np.searchsorted(bins, p.max()))
        code = 0
        for l in levels:
            code = code * self.L + int(l)
        return ((int(pose_obs) * self.Z + hot) * 3 + b) * (self.L ** self.Z) + code

    @property
    def n_states(self):
        return N_POSE_OBS * self.Z * 3 * (self.L ** self.Z)


class QPolicy:
    def __init__(self, env, alpha=0.2, gamma=0.9):
        self.env, self.alpha, self.gamma = env, alpha, gamma
        self.Q = np.zeros((env.n_states, env.n_actions), np.float32)

    def act(self, s, eps=0.0, rng=None):
        if rng is not None and rng.rand() < eps:
            return int(rng.randint(self.env.n_actions))
        q = self.Q[s]
        return int(rng.choice(np.flatnonzero(q == q.max()))) if rng is not None else int(np.argmax(q))

    def train(self, P, poses, episodes, bins, eps_start, eps_end, mask_p, rng, log_every=0):
        """P (N,Z) zone pressures of training frames, poses (N,) true postures."""
        env = self.env
        for ep in range(episodes):
            eps = eps_start + (eps_end - eps_start) * ep / max(1, episodes - 1)
            i = rng.randint(len(P)); p, pose = P[i], int(poses[i])
            obs = 4 if rng.rand() < mask_p else pose
            lv = rng.randint(1, env.L - 1, size=env.Z)
            c, _ = env.cost(p, lv, pose)
            for t in range(env.H):
                s = env.encode(obs, p, lv, bins)
                a = self.act(s, eps, rng)
                lv2 = env.apply(lv, a)
                c2, _ = env.cost(p, lv2, pose)
                r = (c - c2) - 0.01 * (a != 0)
                s2 = env.encode(obs, p, lv2, bins)
                self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[s2].max() - self.Q[s, a])
                lv, c = lv2, c2
            if log_every and (ep + 1) % log_every == 0:
                print(f"    [rl] episode {ep+1}/{episodes}")


def rollout(env, policy_fn, p, pose_true, pose_obs, lv0, bins):
    """Run one episode; returns (initial_cost, final_cost, final_peak, initial_peak, n_adjust)."""
    lv = lv0.copy()
    c0, pk0 = env.cost(p, lv, pose_true)
    adj = 0
    for _ in range(env.H):
        a = policy_fn(env.encode(pose_obs, p, lv, bins), p, lv)
        adj += int(a != 0)
        lv = env.apply(lv, a)
    c1, pk1 = env.cost(p, lv, pose_true)
    return c0, c1, pk1, pk0, adj
