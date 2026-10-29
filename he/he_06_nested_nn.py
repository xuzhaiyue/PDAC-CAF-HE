"""修订 A2：I1（加和式解耦）与 B2（注意力 MIL）同等预算的嵌套调参（hemorph 环境，MPS）。
规则见 config/fig1.yaml 的 he_amendments.A2（已冻结）；特征按 A3 用自己提取的 TCGA 全切片特征，CPTAC 为 Mahmood 特征。

标签与 A1 相同（基因在训练折内 z 标准化后取平均；状态 = 残差 / 减法）。
I1：5 个头拟合各基因集得分（program、Moffitt、Dominguez、StRoMA 3 基因、泛 CAF），
    预测状态 = 头_基因集 − 头_泛CAF（减法），或 头_基因集 − β·头_泛CAF（残差，β 取训练标签的回归系数）。
B2：直接回归 A1 的全部 11 个目标。
网格：lr {1e-4, 3e-4, 1e-3} × weight_decay {1e-4, 1e-2} × 轮数 {20, 40, 80}（同一次训练的检查点）。
外层：5 折（第一轮 rep 0 的折）；每个外层训练集内 3 折选参，标准 = 内层 program_resid 的 Spearman ρ。
CPTAC：全部 TCGA 上用 3 折选参 → 最终配置 5 个种子取均值。
输出：results/he/A2/{selection.tsv, oof_tcga.tsv, metrics_tcga.tsv, pred_cptac.tsv, metrics_cptac.tsv, loss_curves.tsv}
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
