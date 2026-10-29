"""修订 A4：按范文 StRoMA 的模型设计复现 TCGA 内部结果（hemorph 环境，MPS）。规则见 config/fig1.yaml 的 A4（已冻结）。

两个版本：honest（训练折内留 10% 病例做早停）与 leaky_as_reported（用测试折早停，只作诊断）。
用法：python he_08_stroma_model.py [--quick]
输出：results/he/A4/{perfold.tsv, oof.tsv, summary.tsv}
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import he_07_stroma_replication as R

A = R.A
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_num_threads(8)
OUT = A.HE / "A4"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
MAX_EP, PATIENCE = (3, 2) if QUICK else (200, 15)
D, N_TILES, ACCUM, MIX_ALPHA, LAMBDA_RANK = 64, 1024, 8, 0.2, 0.5


# ---------------------------------------------------------------- data: one bag per slide
def load_slides(pats):
    st = pd.read_csv(R.CACHE_OWN / "extraction_status.tsv", sep="\t")
    st = st[(st.status == "ok") & st.patient.isin(pats) & (st.file_name.str[13:15] == "01")]
    slides = []
    for r in st.itertuples():
        z = np.load(R.CACHE_OWN / "slides" / f"{r.file_id}.npz")
        g = np.c_[z["x"] / z["tile_w"], z["y"] / z["tile_h"]].astype(np.float32)   # tile-grid coordinates
        order = np.lexsort((g[:, 0], g[:, 1]))                                    # spatial order (row-major)
        slides.append({"patient": r.patient, "X": torch.from_numpy(z["features"][order].astype(np.float32)),
                       "G": torch.from_numpy(g[order])})
    return slides


def posenc(G, d=D):
    """2-D sinusoidal encoding: half the dims for x, half for y."""
    h = d // 4
    freq = torch.exp(torch.arange(h, dtype=torch.float32) * (-math.log(10000.0) / h))
    out = []
    for c in (0, 1):
        a = G[:, c:c + 1] * freq
        out += [torch.sin(a), torch.cos(a)]
    return torch.cat(out, 1)


class StRoMA(nn.Module):
    def __init__(self, din=1536, d=D, p=0.5):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(din, d), nn.ReLU(), nn.Dropout(p))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 512, p, batch_first=True, norm_first=True), 1)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 8, 512, p, batch_first=True, norm_first=True), 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(d, 1))

    def tokens(self, x, g):
        return self.proj(x) + posenc(g).to(x.device)

    def forward_tokens(self, t):
        h = self.enc(t[None])
        return self.head(self.dec(self.cls, h)[0, 0]).squeeze(-1)


def sample(s, train, rng):
    n = len(s["X"])
    if n <= N_TILES:
        idx = np.arange(n)
    elif train:
        idx = np.sort(rng.choice(n, N_TILES, replace=False))
    else:
        idx = np.linspace(0, n - 1, N_TILES).round().astype(int)
    return s["X"][idx].to(DEV), s["G"][idx]


def rank_loss(pred, y, margin=0.1):
    dy = y[:, None] - y[None, :]
    mask = dy.abs() > 0.05
    if mask.sum() == 0:
        return pred.sum() * 0
    sign = torch.sign(dy)
    return torch.clamp(margin - sign * (pred[:, None] - pred[None, :]), min=0)[mask].mean()


@torch.no_grad()
def predict(m, slides, idx):
    m.eval()
    return np.array([m.forward_tokens(m.tokens(*sample(slides[i], False, None))).item() for i in idx])


def case_mean(slides, idx, p):
    return pd.Series(p, index=[slides[i]["patient"] for i in idx]).groupby(level=0).mean()


def train_fold(slides, y_slide, tr, es_idx, seed, y_case_es):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = StRoMA().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EP)
    best, best_state, best_ep, wait = -np.inf, None, 0, 0
    for ep in range(MAX_EP):
        m.train()
        order = rng.permutation(tr)
        for b in range(0, len(order), ACCUM):
            batch = order[b:b + ACCUM]
            preds, ys = [], []
            for i in batch:
                t = m.tokens(*sample(slides[i], True, rng))
                y = torch.tensor(y_slide[i], device=DEV)
                if MIX_ALPHA > 0:                     # manifold mixup with another training slide
                    j = rng.choice(tr); lam = float(rng.beta(MIX_ALPHA, MIX_ALPHA))
                    tj = m.tokens(*sample(slides[j], True, rng))
                    k = min(len(t), len(tj))
                    t = lam * t[:k] + (1 - lam) * tj[:k]
                    y = lam * y + (1 - lam) * torch.tensor(y_slide[j], device=DEV)
                preds.append(m.forward_tokens(t)); ys.append(y)
            P, Y = torch.stack(preds), torch.stack(ys)
            loss = nn.functional.smooth_l1_loss(P, Y, beta=1.0) + LAMBDA_RANK * rank_loss(P, Y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        # early stopping on case-level Spearman of the early-stopping set
        pe = case_mean(slides, es_idx, predict(m, slides, es_idx))
        score = stats.spearmanr(pe.values, y_case_es.reindex(pe.index).values)[0]
        if score > best:
            best, best_ep, wait = score, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.load_state_dict(best_state)
    return m, best_ep


def main():
    t0 = time.time()
    E, cdr = R.all_paad()
    pats = [p for p in E.columns if (R.CACHE_OWN / f"{p}.npy").exists()]
    y_case = R.composite(E[pats], R.GENES)
    slides = load_slides(pats)
    sp = np.array([s["patient"] for s in slides])
    print(f"{len(pats)} cases, {len(slides)} slides; device {DEV}", flush=True)
    strata = (y_case > y_case.median()).astype(int).values
    rows, oof = [], []
    folds = list(StratifiedKFold(10, shuffle=True, random_state=A.SEED).split(pats, strata))
    for variant in ["honest", "leaky_as_reported"]:
        for f, (trc, tec) in enumerate(folds[:2] if QUICK else folds):
            tr_p, te_p = np.array(pats)[trc], np.array(pats)[tec]
            if variant == "honest":
                fit_p, es_p = train_test_split(tr_p, test_size=0.1, random_state=A.SEED + f,
                                               stratify=strata[trc])
            else:
                fit_p, es_p = tr_p, te_p
            mu, sd = y_case[fit_p].mean(), y_case[fit_p].std()
            yz_case = (y_case - mu) / sd
            y_slide = yz_case.reindex(sp).to_numpy(np.float32)
            tr = np.where(np.isin(sp, fit_p))[0]; es = np.where(np.isin(sp, es_p))[0]; te = np.where(np.isin(sp, te_p))[0]
            m, best_ep = train_fold(slides, y_slide, tr, es, A.SEED + f, yz_case[es_p])
            pc = case_mean(slides, te, predict(m, slides, te))
            yt = yz_case.reindex(pc.index)
            rows.append({"variant": variant, "fold": f, "n_test_cases": len(pc), "best_epoch": best_ep,
                         "spearman": stats.spearmanr(pc, yt)[0],
                         "auc_z0": roc_auc_score(yt > 0, pc) if 0 < (yt > 0).sum() < len(yt) else np.nan})
            oof.append(pd.DataFrame({"variant": variant, "fold": f, "patient": pc.index, "pred": pc.values,
                                     "y_global": y_case.reindex(pc.index).values}))
            print(f"  {variant} fold {f}: rho {rows[-1]['spearman']:.3f} auc {rows[-1]['auc_z0']:.3f} "
                  f"best_ep {best_ep} ({time.time() - t0:.0f}s)", flush=True)
    pf = pd.DataFrame(rows); pf.to_csv(OUT / "perfold.tsv", sep="\t", index=False)
    oof = pd.concat(oof); oof.to_csv(OUT / "oof.tsv", sep="\t", index=False)
    summ = []
    for v, d in pf.groupby("variant"):
        o = oof[oof.variant == v]
        yz = (o.y_global - o.y_global.mean()) / o.y_global.std()
        summ.append({"variant": v, "spearman_perfold": f"{d.spearman.mean():.3f} ± {d.spearman.std():.3f}",
                     "auc_perfold": f"{d.auc_z0.mean():.3f} ± {d.auc_z0.std():.3f}",
                     "spearman_pooled": stats.spearmanr(o.pred, o.y_global)[0],
                     "auc_pooled": roc_auc_score(yz > 0, o.pred), "mean_best_epoch": d.best_epoch.mean()})
    s = pd.DataFrame(summ); s.to_csv(OUT / "summary.tsv", sep="\t", index=False)
    print(s.to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
