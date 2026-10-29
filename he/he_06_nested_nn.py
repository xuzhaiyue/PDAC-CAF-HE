"""Amendment A2: nested tuning with equal budgets for I1 (additive decoupling) and B2 (attention MIL) (hemorph environment, MPS).
Rules in he_amendments.A2 of config/fig1.yaml (fixed); TCGA features self-extracted per A3, CPTAC pre-extracted.

Labels as in A1 (gene z-scores fitted in training folds, averaged; state = residual / subtraction).
I1: 5 heads fit the gene-set scores (program, Moffitt, Dominguez, StRoMA 3-gene, pan-CAF);
    predicted state = head_geneset − head_panCAF (subtraction) or head_geneset − β·head_panCAF (residual, β = regression coefficient on training labels).
B2: direct regression of all 11 A1 targets.
Grid: lr {1e-4, 3e-4, 1e-3} × weight_decay {1e-4, 1e-2} × epochs {20, 40, 80} (checkpoints of the same run).
Outer: 5 folds (rep 0 folds of round one); 3-fold tuning within each outer training set, criterion = inner Spearman ρ of program_resid.
CPTAC: 3-fold tuning on all TCGA → final configuration averaged over 5 seeds.
Outputs: results/he/A2/{selection.tsv, oof_tcga.tsv, metrics_tcga.tsv, pred_cptac.tsv, metrics_cptac.tsv, loss_curves.tsv}
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import KFold

import he_03_ridge_gc_protocol as A

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_num_threads(8)
OUT = A.HE / "A2"
OUT.mkdir(parents=True, exist_ok=True)
SEED = A.SEED
HEADS = ["program", "Moffitt_activated", "Dominguez_LRRC15", "StRoMA_3gene", "pan_CAF"]
GRID = list(itertools.product([1e-4, 3e-4, 1e-3], [1e-4, 1e-2]))
CHECK = [20, 40, 80]
TILES_PER_STEP = 1024


class I1(nn.Module):
    def __init__(self, k=len(HEADS), h=256, p=0.25):
        super().__init__()
        self.tile = nn.Sequential(nn.Linear(1536, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, k))
        self.log_a = nn.Parameter(torch.zeros(k)); self.b = nn.Parameter(torch.zeros(k))

    def forward(self, x):
        P = (nn.functional.softplus(self.tile(x)) + 1e-4).mean(0)
        return torch.exp(self.log_a) * torch.log(P) + self.b


class B2(nn.Module):
    def __init__(self, k, h=256, p=0.25):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1536, h), nn.ReLU(), nn.Dropout(p))
        self.att = nn.Sequential(nn.Linear(h, 128), nn.Tanh(), nn.Linear(128, 1)); self.out = nn.Linear(h, k)

    def forward(self, x):
        z = self.fc(x); w = torch.softmax(self.att(z).squeeze(-1), 0)
        return self.out((w[:, None] * z).sum(0))


def tile_stats(bags, idx):
    rng = np.random.default_rng(SEED)
    X = torch.cat([bags[i][rng.choice(len(bags[i]), min(len(bags[i]), 2000), replace=False)] for i in idx]).float()
    return X.mean(0), X.std(0) + 1e-6


@torch.no_grad()
def predict(m, bags, idx, mu, sd):
    m.eval()
    return np.vstack([m((bags[i].to(DEV).float() - mu) / sd).cpu().numpy() for i in idx])


def train(kind, bags, Y, tr, lr, wd, seed, eval_sets=(), log=None):
    """Train up to max(CHECK) epochs; return {epoch: [predictions for each eval set]}."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = (I1() if kind == "I1" else B2(Y.shape[1])).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    mu, sd = (t.to(DEV) for t in tile_stats(bags, tr))
    Yt = torch.tensor(Y, dtype=torch.float32, device=DEV)
    if kind == "I1":
        with torch.no_grad():
            m.b.copy_(Yt[tr].mean(0))
    out = {}
    for ep in range(1, max(CHECK) + 1):
        m.train(); tot = 0.0
        for i in rng.permutation(tr):
            b = bags[i]; sel = rng.choice(len(b), min(len(b), TILES_PER_STEP), replace=False)
            loss = ((m((b[sel].to(DEV).float() - mu) / sd) - Yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if log is not None:
            log.append({"kind": kind, "lr": lr, "wd": wd, "seed": seed, "epoch": ep, "train_mse": tot / len(tr)})
        if ep in CHECK:
            out[ep] = [predict(m, eb, range(len(eb)), mu, sd) for eb in eval_sets]
    return out


def state_preds(kind, P, cols, beta):
    """Convert raw network outputs into the 11 A1 targets."""
    if kind == "B2":
        return pd.DataFrame(P, columns=cols)
    d = pd.DataFrame(P, columns=HEADS)
    for k in ["program", "Moffitt_activated", "Dominguez_LRRC15"]:
        d[f"{k}_sub"] = d[k] - d["pan_CAF"]
        d[f"{k}_resid"] = d[k] - beta[k] * d["pan_CAF"]
    return d.reindex(columns=cols)


def resid_beta(y):
    return {k: np.polyfit(y["pan_CAF"], y[k], 1)[0] for k in ["program", "Moffitt_activated", "Dominguez_LRRC15"]}


def select(kind, bags, E, idx, seed):
    """3-fold inner CV over the grid on patients idx; returns best (lr, wd, epochs) and the table."""
    rows = []
    for lr, wd in GRID:
        scores = {e: [] for e in CHECK}
        for f, (a, b) in enumerate(KFold(3, shuffle=True, random_state=seed).split(idx)):
            ia, ib = idx[a], idx[b]
            ytr, yte = A.make_targets(E.iloc[ia], E.iloc[ib])
            cols = list(ytr.columns)
            Y = np.zeros((len(bags), len(HEADS) if kind == "I1" else len(cols)))
            Y[ia] = ytr[HEADS].to_numpy() if kind == "I1" else ytr[cols].to_numpy()
            res = train(kind, bags, Y, ia, lr, wd, seed + f, eval_sets=([bags[i] for i in ib],))
            for ep, (P,) in res.items():
                d = state_preds(kind, P, cols, resid_beta(ytr))
                scores[ep].append(stats.spearmanr(d["program_resid"], yte["program_resid"])[0])
        for ep in CHECK:
            rows.append({"kind": kind, "lr": lr, "wd": wd, "epochs": ep, "inner_rho": float(np.mean(scores[ep]))})
    t = pd.DataFrame(rows)
    best = t.sort_values("inner_rho", ascending=False).iloc[0]
    return (best.lr, best.wd, int(best.epochs)), t


def main():
    t0 = time.time()
    T = pd.read_csv(A.HE / "tcga_targets.tsv", sep="\t", index_col=0)
    pats = [p for p in T.index if (A.HE / "cache_own" / "tcga" / f"{p}.npy").exists()]
    Et = pd.read_csv(A.PROJ / "results" / "fig2" / "tcga_expr.tsv.gz", sep="\t", index_col=0)[pats].T
    C = pd.read_csv(A.HE / "cptac_targets.tsv", sep="\t", index_col=0)
    cp = [p for p in C.index if (A.CACHE / "cptac" / f"{p}.npy").exists()]
    Ec = pd.read_csv(A.PROJ / "results" / "fig2" / "cptac_expr.tsv.gz", sep="\t", index_col=0)[cp].T
    bt = [torch.from_numpy(np.load(A.HE / "cache_own" / "tcga" / f"{p}.npy")) for p in pats]
    bc = [torch.from_numpy(np.load(A.CACHE / "cptac" / f"{p}.npy")) for p in cp]
    yc, _ = A.make_targets(Ec, Ec.iloc[:2]); yc.index = cp
    print(f"TCGA {len(pats)} (own features), CPTAC {len(cp)}; device {DEV}", flush=True)

    sel_rows, oof, logs, preds_c, mc = [], [], [], [], []
    for kind in ["I1", "B2"]:
        # ---- outer 5-fold (rep 0 folds), inner 3-fold selection
        for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(pats)):
            (lr, wd, ep), tab = select(kind, bt, Et, tr, SEED + 100 * fold)
            sel_rows.append(tab.assign(outer_fold=fold, chosen=lambda d: (d.lr == lr) & (d.wd == wd) & (d.epochs == ep)))
            ytr, yte = A.make_targets(Et.iloc[tr], Et.iloc[te])
            cols = list(ytr.columns)
            Y = np.zeros((len(bt), len(HEADS) if kind == "I1" else len(cols)))
            Y[tr] = ytr[HEADS].to_numpy() if kind == "I1" else ytr[cols].to_numpy()
            res = train(kind, bt, Y, tr, lr, wd, SEED + fold, eval_sets=([bt[i] for i in te],), log=logs)
            d = state_preds(kind, res[ep][0], cols, resid_beta(ytr))
            for t in cols:
                oof.append(pd.DataFrame({"model": kind, "fold": fold, "patient": np.array(pats)[te], "target": t,
                                         "y_true": yte[t].to_numpy(), "y_pred": d[t].to_numpy()}))
            print(f"  {kind} outer fold {fold}: chosen lr={lr:g} wd={wd:g} epochs={ep} ({time.time() - t0:.0f}s)", flush=True)
        # ---- CPTAC: select on all TCGA, then 5 seeds
        allidx = np.arange(len(pats))
        (lr, wd, ep), tab = select(kind, bt, Et, allidx, SEED + 999)
        sel_rows.append(tab.assign(outer_fold="all_TCGA", chosen=lambda d: (d.lr == lr) & (d.wd == wd) & (d.epochs == ep)))
        ytr, _ = A.make_targets(Et, Et.iloc[:2]); cols = list(ytr.columns)
        Y = ytr[HEADS].to_numpy() if kind == "I1" else ytr[cols].to_numpy()
        P = [state_preds(kind, train(kind, bt, Y, allidx, lr, wd, SEED + 1000 + s, eval_sets=(bc,), log=logs)[ep][0],
                         cols, resid_beta(ytr)) for s in range(5)]
        p = sum(P) / 5; p.index = cp
        preds_c.append(p.assign(model=kind))
        for t in cols:
            mc.append({"model": kind, "target": t, "spearman": stats.spearmanr(p[t], yc[t])[0],
                       "pearson": stats.pearsonr(p[t], yc[t])[0], "chosen": f"lr={lr:g},wd={wd:g},ep={ep}"})
        print(f"  {kind} CPTAC done: lr={lr:g} wd={wd:g} epochs={ep} ({time.time() - t0:.0f}s)", flush=True)

    pd.concat(sel_rows).to_csv(OUT / "selection.tsv", sep="\t", index=False)
    oof = pd.concat(oof); oof.to_csv(OUT / "oof_tcga.tsv", sep="\t", index=False, float_format="%.5f")
    pd.DataFrame(logs).to_csv(OUT / "loss_curves.tsv", sep="\t", index=False)
    pd.concat(preds_c).to_csv(OUT / "pred_cptac.tsv", sep="\t", float_format="%.5f")
    mt = oof.groupby(["model", "target"]).apply(lambda d: pd.Series({
        "spearman": stats.spearmanr(d.y_true, d.y_pred)[0], "pearson": stats.pearsonr(d.y_true, d.y_pred)[0]})).reset_index()
    mt.to_csv(OUT / "metrics_tcga.tsv", sep="\t", index=False)
    mc = pd.DataFrame(mc); mc.to_csv(OUT / "metrics_cptac.tsv", sep="\t", index=False)
    print("\n== TCGA (outer 5-fold, Spearman)\n" + mt.pivot(index="target", columns="model", values="spearman").round(3).to_string())
    print("\n== CPTAC external (Spearman)\n" + mc.pivot(index="target", columns="model", values="spearman").round(3).to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
