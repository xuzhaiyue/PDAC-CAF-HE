"""修订 A1：胃癌项目流程的岭回归（hemorph 环境）。规则见 config/fig1.yaml 的 he_amendments.A1_gc_protocol（已冻结）。

标签：基因在训练折内 z 标准化后取平均；状态 = 对泛 CAF 的残差（主要）或减法（对照）。
特征：切片内图块均值 → 患者内切片等权平均。模型：StandardScaler + Ridge，GridSearchCV（内层 4 折）。
TCGA 5 折 × 5 次重复（与第一轮相同的折）；CPTAC：全部 TCGA 训练后外部预测（标签用 CPTAC 自身的基因统计量）。
用法：python he_03_ridge_gc_protocol.py [--own-tcga]
输出：results/he/A1（或 A1_ownTCGA）/{oof_tcga.tsv, metrics_tcga.tsv, pred_cptac.tsv, metrics_cptac.tsv, decoupling_cptac.tsv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((PROJ / "config" / "fig1.yaml").read_text())
SEED = CFG["random_seed"]
HE = PROJ / "results" / "he"
OWN = "--own-tcga" in sys.argv               # 修订 A3：TCGA 用自己提取的全切片特征（CPTAC 仍为 Mahmood 特征）
OUT = HE / ("A1_ownTCGA" if OWN else "A1")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = HE / "cache"
CACHE_T = HE / "cache_own" if OWN else CACHE
PROG = yaml.safe_load((PROJ / "results" / "fig1" / "signature" / "programs.yaml").read_text())
PUB = yaml.safe_load((PROJ / "config" / "published_signatures.yaml").read_text())
SETS = {"program": PROG["CTHRC1_matrix_CAF"], "pan_CAF": PROG["pan_CAF"],
        "Moffitt_activated": PUB["Moffitt_activated_stroma"]["genes"],
        "Dominguez_LRRC15": PUB["Dominguez_LRRC15_CAF"]["genes"],
        "StRoMA_3gene": CFG["he_model"]["targets"]["benchmark"]["StRoMA_3gene"]}
STATE_OF = ["program", "Moffitt_activated", "Dominguez_LRRC15"]


def patient_features(cohort, patients):
    root = CACHE_T if cohort == "tcga" else CACHE
    rows = []
    for p in patients:
        X = np.load(root / cohort / f"{p}.npy").astype(np.float32)
        s = np.load(root / cohort / f"{p}_coords.npy")[:, 2]
        rows.append(np.mean([X[s == k].mean(0) for k in np.unique(s)], axis=0))   # slides equally weighted
    return np.vstack(rows)


def make_targets(E_tr, E_te, pan_name="pan_CAF"):
    """E_*: samples x genes (log scale). Gene stats and residual betas from the training samples only."""
    raw_tr, raw_te = {}, {}
    for k, g in SETS.items():
        g = [x for x in g if x in E_tr.columns]
        mu, sd = E_tr[g].mean(), E_tr[g].std(ddof=0).replace(0, 1)
        a, b = ((E_tr[g] - mu) / sd).mean(1), ((E_te[g] - mu) / sd).mean(1)
        m, s = a.mean(), a.std(ddof=0) or 1.0
        raw_tr[k], raw_te[k] = (a - m) / s, (b - m) / s
    tr, te = pd.DataFrame(raw_tr), pd.DataFrame(raw_te)
    for k in STATE_OF:
        X = np.c_[np.ones(len(tr)), tr[pan_name]]
        beta, *_ = np.linalg.lstsq(X, tr[k], rcond=None)
        r_tr = tr[k] - X @ beta
        r_te = te[k] - np.c_[np.ones(len(te)), te[pan_name]] @ beta
        m, s = r_tr.mean(), r_tr.std(ddof=0) or 1.0
        tr[f"{k}_resid"], te[f"{k}_resid"] = (r_tr - m) / s, (r_te - m) / s
        d_tr, d_te = tr[k] - tr[pan_name], te[k] - te[pan_name]
        m, s = d_tr.mean(), d_tr.std(ddof=0) or 1.0
        tr[f"{k}_sub"], te[f"{k}_sub"] = (d_tr - m) / s, (d_te - m) / s
    return tr, te


def ridge(seed):
    pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    return GridSearchCV(pipe, {"model__alpha": np.logspace(-2, 4, 13)}, scoring="neg_mean_squared_error",
                        cv=KFold(4, shuffle=True, random_state=seed), n_jobs=4, refit=True)


def boot_pearson(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v.append(stats.pearsonr(y[i], p[i])[0])
    return np.quantile(v, [0.025, 0.975])


def main():
    T = pd.read_csv(HE / "tcga_targets.tsv", sep="\t", index_col=0)
    T = T[[(CACHE_T / "tcga" / f"{p}.npy").exists() for p in T.index]]
    pats = list(T.index)
    Et = pd.read_csv(PROJ / "results" / "fig2" / "tcga_expr.tsv.gz", sep="\t", index_col=0)[pats].T
    Ct = pd.read_csv(HE / "cptac_targets.tsv", sep="\t", index_col=0)
    cp = [p for p in Ct.index if (CACHE / "cptac" / f"{p}.npy").exists()]
    Ec = pd.read_csv(PROJ / "results" / "fig2" / "cptac_expr.tsv.gz", sep="\t", index_col=0)[cp].T
    Xt, Xc = patient_features("tcga", pats), patient_features("cptac", cp)
    _, tnames = make_targets(Et, Et.iloc[:2])
    TARGETS = list(tnames.columns)
    print(f"TCGA {len(pats)}, CPTAC {len(cp)}; targets: {TARGETS}", flush=True)

    rows = []
    for rep in range(5):
        for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED + rep).split(pats)):
            ytr, yte = make_targets(Et.iloc[tr], Et.iloc[te])
            for t in TARGETS:
                g = ridge(SEED + rep * 100 + fold).fit(Xt[tr], ytr[t].to_numpy())
                rows.append(pd.DataFrame({"patient": np.array(pats)[te], "rep": rep, "fold": fold, "target": t,
                                          "y_true": yte[t].to_numpy(), "y_pred": g.predict(Xt[te]),
                                          "alpha": g.best_params_["model__alpha"]}))
        print(f"  rep {rep} done", flush=True)
    oof = pd.concat(rows, ignore_index=True)
    oof.to_csv(OUT / "oof_tcga.tsv", sep="\t", index=False, float_format="%.5f")
    avg = oof.groupby(["patient", "target"], as_index=False)[["y_true", "y_pred"]].mean()
    met = []
    for t, d in avg.groupby("target"):
        y, p = d.y_true.to_numpy(), d.y_pred.to_numpy()
        lo, hi = boot_pearson(y, p)
        per_rep = oof[oof.target == t].groupby("rep").apply(lambda x: stats.spearmanr(x.y_true, x.y_pred)[0])
        met.append({"target": t, "pearson": stats.pearsonr(y, p)[0], "pearson_lo": lo, "pearson_hi": hi,
                    "spearman": stats.spearmanr(y, p)[0], "spearman_per_rep_mean": per_rep.mean(),
                    "R2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
                    "median_alpha": oof[oof.target == t].alpha.median()})
    met = pd.DataFrame(met).set_index("target").reindex(TARGETS)
    met.to_csv(OUT / "metrics_tcga.tsv", sep="\t")
    print("\n== TCGA 5x5 CV (patient-averaged)\n" + met.round(3).to_string(), flush=True)

    # ---- CPTAC external
    ytr_all, _ = make_targets(Et, Et.iloc[:2])
    yc, _ = make_targets(Ec, Ec.iloc[:2])          # CPTAC labels from CPTAC's own gene statistics
    pred = {}
    for t in TARGETS:
        pred[t] = ridge(SEED).fit(Xt, ytr_all[t].to_numpy()).predict(Xc)
    pred = pd.DataFrame(pred, index=cp)
    pred.to_csv(OUT / "pred_cptac.tsv", sep="\t", float_format="%.5f")
    yc.index = cp
    yc.to_csv(OUT / "labels_cptac.tsv", sep="\t", float_format="%.5f")
    rng = np.random.default_rng(SEED)
    mc = []
    for t in TARGETS:
        r = stats.spearmanr(pred[t], yc[t])[0]
        null = [stats.spearmanr(pred[t], rng.permutation(yc[t].to_numpy()))[0] for _ in range(1000)]
        mc.append({"target": t, "spearman": r, "pearson": stats.pearsonr(pred[t], yc[t])[0],
                   "perm_p": (np.sum(np.array(null) >= r) + 1) / 1001})
    mc = pd.DataFrame(mc).set_index("target")
    mc.to_csv(OUT / "metrics_cptac.tsv", sep="\t")
    print("\n== CPTAC external\n" + mc.round(3).to_string())

    cl = pd.read_csv(PROJ / "results" / "fig2" / "cptac_clinical.tsv", sep="\t", index_col=0)
    es = pd.read_csv(PROJ / "results" / "fig2" / "cptac_estimate.tsv", sep="\t", index_col=0)
    dec = []
    for st in ["program_resid", "program_sub"]:
        rx, ry, rz = (stats.rankdata(v) for v in (pred[st], yc[st], yc.pan_CAF))
        res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
        r, p = stats.pearsonr(res(rx), res(ry))
        dec.append({"test": f"partial rho(pred {st}, bulk {st} | bulk pan-CAF)", "rho": r, "p": p})
    for col, src in [("StromalScore", es), ("Stromal_fraction", cl)]:
        v = src[col].reindex(cp)
        ok = v.notna().to_numpy()
        for nm in ["pan_CAF", "program_resid", "program"]:
            r = stats.spearmanr(pred[nm][ok], v[ok])
            dec.append({"test": f"rho(pred {nm}, {col})", "rho": r[0], "p": r[1]})
    dec = pd.DataFrame(dec)
    dec.to_csv(OUT / "decoupling_cptac.tsv", sep="\t", index=False)
    print("\n== CPTAC decoupling\n" + dec.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
