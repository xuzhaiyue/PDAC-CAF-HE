"""诊断：按范文 StRoMA 的口径复现（hemorph 环境）。不改变主分析，只用来区分"流程/数据问题"与"病例纳入/指标算法差异"。

口径（照范文 Methods）：
  队列  TCGA 全部 PAAD 原发肿瘤（样本类型 01，不排除非导管型），每名患者一个表达谱；范文为 178 例
  目标  COL12A1、LOX、LOXL2：log2 表达在队列内逐基因 z 标准化后取平均（EB++ 已是 log2(norm+1)）
  交叉验证  StratifiedKFold 10 折（按中位数二分分层），训练折内再标准化目标
  指标  每折 Spearman 取平均 ± SD（范文的汇报方式）；同时报告合并折外预测的 Spearman；
        AUC 按 Z > 0 二分（每折平均与合并两种）
模型：岭回归（胃癌流程，均值池化，自提全切片特征）。对照：同样设置只用 147 例导管腺癌。
对号检查：表达条形码与切片条形码一致性；H&E 应能轻松预测的量（内分泌 / 腺泡含量、非导管组织学类型）。
输出：results/he/stroma_replication/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import he_03_ridge_gc_protocol as A

A.CACHE_T = A.HE / "cache_own"          # 用自己提取的 TCGA 全切片特征
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
