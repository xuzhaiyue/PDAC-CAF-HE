"""Diagnostic: replication following the StRoMA definitions (hemorph environment). Does not change the main analysis; separates pipeline/data issues from case-selection/metric differences.

Definitions (following the StRoMA Methods):
  cohort  all TCGA PAAD primary tumours (sample type 01, non-ductal not excluded), one expression profile per patient; 178 in the original study
  target  COL12A1, LOX, LOXL2: log2 expression z-scored per gene within the cohort, then averaged (EB++ is already log2(norm+1))
  cross-validation  StratifiedKFold 10 folds (stratified on the median split), targets re-standardised within training folds
  metrics  per-fold Spearman mean ± SD (as reported in the original study); Spearman of pooled out-of-fold predictions also reported;
        AUC for Z > 0 (per-fold mean and pooled)
Model: ridge regression (mean pooling, self-extracted whole-slide features). Comparison: same settings on the 147 ductal adenocarcinomas only.
Sanity checks: expression and slide barcodes agree; quantities H&E should predict easily (endocrine / acinar content, non-ductal histology).
Output: results/he/stroma_replication/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import he_03_ridge_gc_protocol as A

A.CACHE_T = A.HE / "cache_own"          # self-extracted TCGA whole-slide features
OUT = A.HE / "stroma_replication"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_OWN = A.HE / "cache_own" / "tcga"
F2 = A.CFG["fig2"]["data"]
GENES = ["COL12A1", "LOX", "LOXL2"]


def all_paad():
    ph = pd.read_csv(F2["tcga_pheno"], sep="\t")
    ph = ph[(ph._primary_disease == "pancreatic adenocarcinoma") & (ph["sample"].str[13:15] == "01")].copy()
    ph["patient"] = ph["sample"].str[:12]
    ph = ph.drop_duplicates("patient")
    hdr = pd.read_csv(F2["tcga_expr"], sep="\t", nrows=0).columns
    cols = ["sample"] + [c for c in hdr if c in set(ph["sample"])]
    E = pd.read_csv(F2["tcga_expr"], sep="\t", usecols=cols, index_col=0)
    E = E[~E.index.astype(str).str.fullmatch(r"\d+")]
    E = E[~E.index.duplicated()]
    E.columns = E.columns.str[:12]
    cdr = pd.read_csv(A.HE.parent / "fig2" / "tcga_cdr_paad.tsv", sep="\t").set_index("bcr_patient_barcode")
    return E, cdr


def composite(E, genes):
    Z = E.loc[genes].sub(E.loc[genes].mean(1), axis=0).div(E.loc[genes].std(1), axis=0)
    return Z.mean(0)


def cv_eval(X, y, label, seed=A.SEED):
    """StRoMA-style 10-fold stratified CV; returns per-fold metrics and pooled out-of-fold predictions."""
    strata = (y > np.median(y)).astype(int)
    oof = np.zeros(len(y)); rows = []
    for f, (tr, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=seed).split(X, strata)):
        m, s = y[tr].mean(), y[tr].std()
        g = A.ridge(seed + f).fit(X[tr], (y[tr] - m) / s)
        p = g.predict(X[te]); oof[te] = p
        yz = (y[te] - m) / s
        rows.append({"fold": f, "spearman": stats.spearmanr(yz, p)[0],
                     "auc_z0": roc_auc_score(yz > 0, p) if 0 < (yz > 0).sum() < len(yz) else np.nan})
    pf = pd.DataFrame(rows)
    yz_all = (y - y.mean()) / y.std()
    return {"label": label, "n": len(y),
            "spearman_perfold_mean": pf.spearman.mean(), "spearman_perfold_sd": pf.spearman.std(),
            "auc_perfold_mean": pf.auc_z0.mean(),
            "spearman_pooled": stats.spearmanr(y, oof)[0], "auc_pooled_z0": roc_auc_score(yz_all > 0, oof)}


def main():
    E, cdr = all_paad()
    pats = [p for p in E.columns if (CACHE_OWN / f"{p}.npy").exists()]
    print(f"TCGA PAAD primary tumours with expression: {E.shape[1]}; with own slide features: {len(pats)}")
    # ---- barcode alignment check
    st = pd.read_csv(CACHE_OWN / "extraction_status.tsv", sep="\t")
    st = st[st.status == "ok"]
    mism = [(p, f) for p, f in zip(st.patient, st.file_name) if f[:12] != p]
    print(f"slide filename vs patient key mismatches: {len(mism)}")
    hist = cdr.histological_type.reindex(pats).fillna("NA")
    print("histology:", hist.value_counts().to_dict())
    E = E[pats]
    X = A.patient_features("tcga", pats)
    ductal = (hist == "Pancreas-Adenocarcinoma Ductal Type").to_numpy()

    # ---- sanity targets (should be easy if image and RNA belong together)
    sanity = {"endocrine (INS,GCG,CHGA,SST)": E.loc[["INS", "GCG", "CHGA", "SST"]].mean(),
              "acinar (PRSS1,CPA1,CELA3A,CTRB2)": E.loc[["PRSS1", "CPA1", "CELA3A", "CTRB2"]].mean(),
              "non-ductal histology (0/1)": pd.Series((~ductal).astype(float), index=pats)}
    rows = []
    for k, y in sanity.items():
        rows.append(cv_eval(X, y.to_numpy(float), f"sanity: {k} [all PAAD]"))
    # ---- StRoMA target, StRoMA cohort vs our ductal cohort
    y_all = composite(E, GENES).to_numpy()
    rows.append(cv_eval(X, y_all, "StRoMA 3-gene [all PAAD, StRoMA cohort]"))
    Ed = E.loc[:, ductal]
    rows.append(cv_eval(X[ductal], composite(Ed, GENES).to_numpy(), "StRoMA 3-gene [ductal only, our cohort]"))
    for k, y in list(sanity.items())[:2]:
        rows.append(cv_eval(X[ductal], y.to_numpy(float)[ductal], f"sanity: {k} [ductal only]"))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "replication_metrics.tsv", sep="\t", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
