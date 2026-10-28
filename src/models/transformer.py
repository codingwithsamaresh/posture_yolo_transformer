"""Stage 2: small transformer encoder over a window of per-frame tokens.
Heads: posture class (supine/left/right/prone) and 'posture transition inside this window' (binary)."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PostureTransformer(nn.Module):
    def __init__(self, d_in, n_cls=4, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.2, max_len=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head_pose = nn.Linear(d_model, n_cls)
        self.head_trans = nn.Linear(d_model, 1)

    def forward(self, x):                       # x: (B, T, d_in)
        h = self.inp(x)
        B, T, _ = h.shape
        h = torch.cat([self.cls.expand(B, -1, -1), h], 1) + self.pos[:, : T + 1]
        z = self.norm(self.enc(h))[:, 0]
        return self.head_pose(z), self.head_trans(z).squeeze(-1)


def _loss(lp, lt, y, t, w):
    return F.cross_entropy(lp, y) + w * F.binary_cross_entropy_with_logits(lt, t.float())


def train_one(tok, win, tr_ids, va_ids, tc, seed, L, device, verbose=False):
    """tok: (Ntot, D) float tensor on device (standardised). win: dict of numpy arrays. Returns best model."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = PostureTransformer(tok.shape[1], 4, tc.d_model, tc.n_heads, tc.n_layers, tc.d_ff, tc.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, tc.epochs))
    idx = torch.from_numpy(win["idx"][:, -L:]).to(device)
    y = torch.from_numpy(win["y"]).to(device); t = torch.from_numpy(win["trans"]).to(device)
    tr = torch.from_numpy(np.asarray(tr_ids)).to(device); va = torch.from_numpy(np.asarray(va_ids)).to(device)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(tc.epochs):
        model.train()
        ids = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(ids), tc.batch):
            b = ids[i:i + tc.batch]
            x = tok[idx[b]]
            if tc.noise_std > 0:
                x = x + tc.noise_std * torch.randn_like(x)
            if tc.frame_drop > 0 and x.shape[1] > 1:
                x = x * (torch.rand(x.shape[0], x.shape[1], 1, device=device) > tc.frame_drop).float()
            lp, lt = model(x)
            loss = _loss(lp, lt, y[b], t[b], tc.trans_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            for i in range(0, len(va), 1024):
                b = va[i:i + 1024]
                lp, lt = model(tok[idx[b]])
                vl += _loss(lp, lt, y[b], t[b], tc.trans_weight).item() * len(b); n += len(b)
            vl /= max(n, 1)
        if verbose:
            print(f"    ep {ep+1:>2}/{tc.epochs} val_loss {vl:.4f}")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc.patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def predict_logits(model, tok, win, ids, L, device, mc=False, S=1, bs=1024):
    """Returns posture logits (S,N,4) and transition logits (S,N). mc=True keeps dropout active
    (the network has no batch-norm, so train() only switches dropout on)."""
    idx = torch.from_numpy(win["idx"][:, -L:]).to(device)
    ids = torch.from_numpy(np.asarray(ids)).to(device)
    model.train() if mc else model.eval()
    P, Tr = [], []
    for _ in range(S):
        lp_all, lt_all = [], []
        for i in range(0, len(ids), bs):
            lp, lt = model(tok[idx[ids[i:i + bs]]])
            lp_all.append(lp.float().cpu().numpy()); lt_all.append(lt.float().cpu().numpy())
        P.append(np.concatenate(lp_all)); Tr.append(np.concatenate(lt_all))
    model.eval()
    return np.stack(P), np.stack(Tr)
