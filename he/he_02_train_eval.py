"""H&E 模型：TCGA 交叉验证 + CPTAC 外部验证（hemorph 环境，MPS）。规则见 config/fig1.yaml 的 he_model（已冻结）。

I1 加和式解耦模型：图块 MLP → 5 个非负局部量（程序、Moffitt 活化、Dominguez、StRoMA 3 基因、泛 CAF），
   切片层面 P_k = 图块均值，输出_k = a_k·log(P_k) + b_k（a_k > 0）；损失只拟合 bulk 的原始得分（log2 尺度）。
   相对得分 = 输出_基因集 − 输出_泛CAF（结构上分开状态与数量）。
B1 均值池化 + 岭回归：每个目标单独，alpha 由内层 5 折选择。
B2 注意力 MIL（范文式）：切片层面直接回归全部 8 个目标（z 标准化）。
TCGA：患者层面 5 折 × 5 次重复（三个模型用相同折）；CPTAC：全部 TCGA 训练（神经网络 5 个种子取均值）。
输出：results/he/{oof_tcga.tsv, pred_cptac.tsv, metrics_tcga.tsv, metrics_cptac.tsv, decoupling_cptac.tsv}
用法：python he_02_train_eval.py [--quick]   （--quick：1 次重复、少轮数，只用于检查代码能跑通）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((PROJ / "config" / "fig1.yaml").read_text())
OUT = PROJ / "results" / "he"
CACHE = OUT / "cache"
SEED = CFG["random_seed"]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_num_threads(8)

HEADS = ["program", "Moffitt_activated", "Dominguez_LRRC15", "StRoMA_3gene", "pan_CAF"]   # I1 的非负局部量
REL = {"program_rel": "program", "Moffitt_activated_rel": "Moffitt_activated", "Dominguez_LRRC15_rel": "Dominguez_LRRC15"}
TARGETS = ["program_rel", "pan_CAF", "program", "Moffitt_activated_rel", "Moffitt_activated",
           "Dominguez_LRRC15_rel", "Dominguez_LRRC15", "StRoMA_3gene"]
QUICK = "--quick" in sys.argv
N_REP = 1 if QUICK else 5
EPOCHS_I1, EPOCHS_B2 = (3, 3) if QUICK else (40, 30)
TILES_PER_STEP = 1024
N_SEEDS_EXT = 1 if QUICK else 5


# ---------------------------------------------------------------- data
def load_bags(cohort: str, patients: list[str]) -> list[torch.Tensor]:
    return [torch.from_numpy(np.load(CACHE / cohort / f"{p}.npy")) for p in patients]


def tile_stats(bags, idx, n=200_000):
    rng = np.random.default_rng(SEED)
    X = torch.cat([b[rng.choice(len(b), min(len(b), n // max(len(idx), 1) + 1), replace=False)] for b in (bags[i] for i in idx)])
    X = X.float()
    return X.mean(0), X.std(0) + 1e-6


# ---------------------------------------------------------------- models
class AdditiveDecoupled(nn.Module):
    """I1: tile -> non-negative local amounts; slide = mean over tiles; output = a*log(mean) + b."""

    def __init__(self, k: int = len(HEADS), h: int = 256, p: float = 0.25):
        super().__init__()
        self.tile = nn.Sequential(nn.Linear(1536, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, k))
        self.log_a = nn.Parameter(torch.zeros(k))
        self.b = nn.Parameter(torch.zeros(k))

    def local(self, x):
        return nn.functional.softplus(self.tile(x)) + 1e-4

    def forward(self, x):
        P = self.local(x).mean(0)
        return torch.exp(self.log_a) * torch.log(P) + self.b


class ABMIL(nn.Module):
    """B2: attention MIL with a slide-level regression head (StRoMA-style: attention weights as heatmap)."""

    def __init__(self, k: int = len(TARGETS), h: int = 256, p: float = 0.25):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1536, h), nn.ReLU(), nn.Dropout(p))
        self.att = nn.Sequential(nn.Linear(h, 128), nn.Tanh(), nn.Linear(128, 1))
        self.out = nn.Linear(h, k)

    def forward(self, x):
        z = self.fc(x)
        w = torch.softmax(self.att(z).squeeze(-1), 0)
        return self.out((w[:, None] * z).sum(0))


def train_nn(kind, bags, Y, tr, mu, sd, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = (AdditiveDecoupled() if kind == "I1" else ABMIL()).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=2e-4, weight_decay=1e-4)
    mu_d, sd_d = mu.to(DEV), sd.to(DEV)
    Yt = torch.tensor(Y, dtype=torch.float32, device=DEV)
    if kind == "I1":                                   # init b at the training mean of each head
        with torch.no_grad():
            m.b.copy_(Yt[tr].mean(0))
    for _ in range(EPOCHS_I1 if kind == "I1" else EPOCHS_B2):
        m.train()
        for i in rng.permutation(tr):
            b = bags[i]
            sel = rng.choice(len(b), min(len(b), TILES_PER_STEP), replace=False)
            x = (b[sel].to(DEV).float() - mu_d) / sd_d
            loss = ((m(x) - Yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


@torch.no_grad()
def predict_nn(m, bags, idx, mu, sd):
    mu_d, sd_d = mu.to(DEV), sd.to(DEV)
    out = []
    for i in idx:
        x = (bags[i].to(DEV).float() - mu_d) / sd_d
        out.append(m(x).cpu().numpy())
    return np.vstack(out)


def i1_to_targets(P: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame(P, columns=HEADS)
    for rel, raw in REL.items():
        d[rel] = d[raw] - d["pan_CAF"]
    return d[TARGETS]


def meanpool(bags):
    return np.vstack([b.float().mean(0).numpy() for b in bags])


def ridge_fit_predict(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    # alpha chosen per target by inner 5-fold CV (sklearn's alpha_per_target only works with LOO)
    return np.column_stack([RidgeCV(alphas=np.logspace(-2, 6, 17), cv=5).fit(A, Ytr[:, j]).predict(B)
                            for j in range(Ytr.shape[1])])


# ---------------------------------------------------------------- one train/predict round for all three models
def run_round(bags_tr_all, T, tr, bags_te, seed, mp_tr, mp_te):
    preds = {}
    mu, sd = tile_stats(bags_tr_all, tr)
    # I1 on raw head scores (log2 scale, unstandardised so that relative = difference)
    m = train_nn("I1", bags_tr_all, T[HEADS].to_numpy(), tr, mu, sd, seed)
    preds["I1"] = i1_to_targets(predict_nn(m, bags_te, range(len(bags_te)), mu, sd))
    # B2 on z-scored targets (training statistics)
    Ym, Ys = T[TARGETS].iloc[tr].mean(), T[TARGETS].iloc[tr].std()
    m = train_nn("B2", bags_tr_all, ((T[TARGETS] - Ym) / Ys).to_numpy(), tr, mu, sd, seed)
    preds["B2"] = pd.DataFrame(predict_nn(m, bags_te, range(len(bags_te)), mu, sd), columns=TARGETS)
    # B1 ridge on mean-pooled features
    preds["B1"] = pd.DataFrame(ridge_fit_predict(mp_tr, T[TARGETS].iloc[tr].to_numpy(), mp_te), columns=TARGETS)
    return preds


def perm_p(x, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    r0 = stats.spearmanr(x, y)[0]
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n)])
    return r0, (np.sum(null >= r0) + 1) / (n + 1)


def main():
    t0 = time.time()
    T = pd.read_csv(OUT / "tcga_targets.tsv", sep="\t", index_col=0)
    T = T[[(CACHE / "tcga" / f"{p}.npy").exists() for p in T.index]]
    C = pd.read_csv(OUT / "cptac_targets.tsv", sep="\t", index_col=0)
    C = C[[(CACHE / "cptac" / f"{p}.npy").exists() for p in C.index]]
    bags_t, bags_c = load_bags("tcga", list(T.index)), load_bags("cptac", list(C.index))
    mp_t, mp_c = meanpool(bags_t), meanpool(bags_c)
    print(f"TCGA {len(T)} patients, CPTAC {len(C)} patients; device {DEV}", flush=True)

    # ---- TCGA cross-validation
    oof = []
    for rep in range(N_REP):
        kf = KFold(5, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(kf.split(T)):
            p = run_round(bags_t, T, tr, [bags_t[i] for i in te], seed=SEED + 10 * rep + fold,
                          mp_tr=mp_t[tr], mp_te=mp_t[te])
            for model, d in p.items():
                oof.append(d.assign(patient=T.index[te].values, model=model, rep=rep))
            print(f"  rep {rep} fold {fold} done ({time.time() - t0:.0f}s)", flush=True)
    oof = pd.concat(oof, ignore_index=True)
    oof.to_csv(OUT / "oof_tcga.tsv", sep="\t", index=False, float_format="%.5f")
    rows = []
    for (model, rep), d in oof.groupby(["model", "rep"]):
        y = T.loc[d.patient]
        for t in TARGETS:
            rows.append({"model": model, "rep": rep, "target": t, "rho": stats.spearmanr(d[t], y[t])[0]})
    mt = pd.DataFrame(rows).groupby(["model", "target"]).rho.agg(["mean", "std"]).reset_index()
    mt.to_csv(OUT / "metrics_tcga.tsv", sep="\t", index=False)
    print(mt.pivot(index="target", columns="model", values="mean").round(3).reindex(TARGETS).to_string(), flush=True)

    # ---- CPTAC external: train on all TCGA
    all_tr = np.arange(len(T))
    ext = {m: [] for m in ["I1", "B2", "B1"]}
    for s in range(N_SEEDS_EXT):
        p = run_round(bags_t, T, all_tr, bags_c, seed=SEED + 1000 + s, mp_tr=mp_t, mp_te=mp_c)
        for m in ext:
            ext[m].append(p[m])
    pred = pd.concat({m: sum(v) / len(v) for m, v in ext.items()}, names=["model"]).reset_index(level=0)
    pred["patient"] = np.tile(C.index.values, 3)
    pred.to_csv(OUT / "pred_cptac.tsv", sep="\t", index=False, float_format="%.5f")
    rows = []
    for model, d in pred.groupby("model"):
        y = C.loc[d.patient]
        for t in TARGETS:
            r, p = perm_p(d[t].to_numpy(), y[t].to_numpy(), n=200 if QUICK else 1000)
            rows.append({"model": model, "target": t, "rho": r, "perm_p": p, "n": len(d)})
    mc = pd.DataFrame(rows)
    mc.to_csv(OUT / "metrics_cptac.tsv", sep="\t", index=False)
    print(mc.pivot(index="target", columns="model", values="rho").round(3).reindex(TARGETS).to_string())

    # ---- decoupling in CPTAC (predicted state vs bulk state | bulk abundance; predicted abundance vs stroma)
    cl = pd.read_csv(PROJ / "results" / "fig2" / "cptac_clinical.tsv", sep="\t", index_col=0)
    es = pd.read_csv(PROJ / "results" / "fig2" / "cptac_estimate.tsv", sep="\t", index_col=0)
    rows = []
    for model, d in pred.groupby("model"):
        d = d.set_index("patient")
        y = C.loc[d.index]
        rx, ry, rz = (stats.rankdata(v) for v in (d.program_rel, y.program_rel, y.pan_CAF))
        res = lambda a, z: a - np.polyval(np.polyfit(z, a, 1), z)  # noqa: E731
        pr, pp = stats.pearsonr(res(rx, rz), res(ry, rz))
        rows.append({"model": model, "test": "partial rho(pred state, bulk state | bulk pan-CAF)", "rho": pr, "p": pp})
        for col, src in [("StromalScore", es), ("Stromal_fraction", cl)]:
            v = src[col].reindex(d.index)
            ok = v.notna()
            r = stats.spearmanr(d.pan_CAF[ok], v[ok])
            rows.append({"model": model, "test": f"rho(pred abundance, {col})", "rho": r[0], "p": r[1]})
            r = stats.spearmanr(d.program_rel[ok], v[ok])
            rows.append({"model": model, "test": f"rho(pred state, {col})", "rho": r[0], "p": r[1]})
    dc = pd.DataFrame(rows)
    dc.to_csv(OUT / "decoupling_cptac.tsv", sep="\t", index=False)
    print(dc.round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
