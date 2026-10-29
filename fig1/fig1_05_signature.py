"""Fig 1 step 05 — 提炼 CTHRC1+ CAF 程序与泛 CAF 参考程序（hemorph 环境）。

规则全部来自 config/fig1.yaml 的 signature 段，在接触任何预后数据之前冻结：
  1. 状态差异：每个数据集内，按患者做伪 bulk（CTHRC1+ 亚群 vs 其他 CAF），患者配对 log2FC，
     三个数据集都需满足 log2FC / FDR / 患者一致性阈值。
  2. 细胞特异性：用本研究的全局谱系注释重算伪 bulk，基因在成纤维细胞中需显著高于非间质谱系与周细胞/平滑肌，
     三个数据集都需满足 —— 保证程序在 bulk RNA-seq / Visium HD 中主要反映 CAF。
  3. 泛 CAF 参考：成纤维细胞特异、但在 CAF 亚群之间差异小的高表达基因（相对状态评分的分母）。
  4. 留一数据集：用两个数据集提炼，在第三个中检验（细胞水平 AUROC、患者水平评分差异、基因重合度）。

Outputs: results/fig1/signature/
  de_per_dataset.tsv, lineage_cpm.tsv, specificity.tsv,
  cthrc1_program.tsv, pancaf_program.tsv, lodo.tsv, programs.yaml
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fig1.yaml").read_text())
S = {**CFG["signature"], **CFG.get("signature_v2", {}), **CFG.get("signature_v3", {}),
     **CFG.get("signature_v4", {})}
ROOT = Path(CFG["paths"]["out_dir"])
OUT = ROOT / "signature"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = list(CFG["datasets"])


def group_sum(X: sp.csr_matrix, keys: np.ndarray) -> tuple[np.ndarray, sp.csr_matrix]:
    uniq, inv = np.unique(keys, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    return uniq, (G @ X).tocsr()


def log2cpm(M) -> np.ndarray:
    M = np.asarray(M.todense() if sp.issparse(M) else M, dtype=float)
    return np.log2(M / np.maximum(M.sum(1, keepdims=True), 1) * 1e6 + 1)


# ---------------------------------------------------------------- 1. state DE
PATIENT_LFC: dict[str, pd.DataFrame] = {}


def pooled_test(datasets: list[str]) -> pd.DataFrame:
    """One-sample t-test of per-patient log2FC pooled across datasets (paired design), BH-FDR."""
    M = pd.concat([PATIENT_LFC[d] for d in datasets if d in PATIENT_LFC])
    t, p = stats.ttest_1samp(M.to_numpy(), 0.0, axis=0)
    p = np.nan_to_num(p, nan=1.0)
    return pd.DataFrame({"pooled_log2fc": M.mean(0).to_numpy(), "pooled_p": p,
                         "pooled_fdr": multipletests(p, method="fdr_bh")[1], "pooled_n": len(M)},
                        index=M.columns)
def paired_de(caf: ad.AnnData, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        sub = caf[(caf.obs.dataset == ds).to_numpy()]
        grp = np.where(sub.obs.is_target, "T", "R")
        keys = sub.obs.patient_key.astype(str).to_numpy() + "|" + grp
        uniq, P = group_sum(sub.layers["counts"], keys)
        n = pd.Series(keys).value_counts().reindex(uniq).to_numpy()
        pats = pd.Series([u.split("|")[0] for u in uniq])
        ok = pd.DataFrame({"p": pats, "g": [u.split("|")[1] for u in uniq], "n": n})
        good = ok[ok.n >= S["min_cells_per_group"]].groupby("p").g.nunique()
        good = good[good == 2].index
        if len(good) < 3:
            print(f"  {ds}: only {len(good)} paired patients — skipped")
            continue
        L = log2cpm(P)
        iT = [np.where(uniq == f"{p}|T")[0][0] for p in good]
        iR = [np.where(uniq == f"{p}|R")[0][0] for p in good]
        lfc = L[iT] - L[iR]                                     # patients x genes
        PATIENT_LFC[ds] = pd.DataFrame(lfc, columns=caf.var_names, index=list(good))
        t, pval = stats.ttest_rel(L[iT], L[iR], axis=0)
        pval = np.nan_to_num(pval, nan=1.0)
        cpmT = (2 ** L[iT] - 1).mean(0)
        rows.append(pd.DataFrame({
            "dataset": ds, "gene": caf.var_names, "n_patients": len(good),
            "log2fc": lfc.mean(0), "frac_up": (lfc > 0).mean(0), "p": pval,
            "fdr": multipletests(pval, method="fdr_bh")[1], "target_cpm": cpmT}))
        print(f"  {ds}: {len(good)} paired patients")
    return pd.concat(rows, ignore_index=True)


def select_state_genes(de: pd.DataFrame, datasets: list[str], spec_ok: set[str],
                       pooled: pd.DataFrame) -> pd.DataFrame:
    """Effect size + patient consistency required in every dataset; significance from pooled patients."""
    d = de[de.dataset.isin(datasets)]
    pass_ = ((d.log2fc > S["min_log2fc"]) & (d.frac_up >= S["min_frac_patients_up"]) &
             (d.target_cpm >= S["min_target_cpm"]))
    d = d.assign(pass_=pass_)
    agg = d.groupby("gene").agg(n_pass=("pass_", "sum"), min_log2fc=("log2fc", "min"),
                                mean_log2fc=("log2fc", "mean"), min_frac_up=("frac_up", "min"),
                                mean_target_cpm=("target_cpm", "mean"))
    agg = agg.join(pooled[["pooled_fdr", "pooled_n"]])
    n_ds = d.dataset.nunique()
    sel = agg[(agg.n_pass == n_ds) & (agg.pooled_fdr < S["max_fdr"]) & agg.index.isin(spec_ok)]
    return sel.sort_values("min_log2fc", ascending=False).head(S["max_genes"])


# ---------------------------------------------------------------- 2. lineage specificity
def lineage_cpm(conditions: list[str] | None = None) -> pd.DataFrame:
    ann = pd.read_csv(ROOT / "annotate_global" / "global_full_annotation.tsv.gz", sep="\t", index_col=0)
    if conditions:
        ann = ann[ann.condition.isin(conditions)]
    full = ad.read_h5ad(ROOT / "global_full_raw.h5ad")
    full = full[full.obs_names.isin(ann.index)]
    ann = ann.reindex(full.obs_names)
    keep = ~ann.lineage.isin(["Ambiguous", "Low_quality", "Hepatocyte_like"]).to_numpy()
    keys = (ann.dataset.astype(str) + "|" + ann.lineage.astype(str) + "|" + ann.patient_key.astype(str)).to_numpy()
    uniq, P = group_sum(sp.csr_matrix(full.X)[keep], keys[keep])
    n = pd.Series(keys[keep]).value_counts().reindex(uniq).to_numpy()
    ok = n >= 20                                              # patient-level pseudobulk needs >= 20 cells
    L = 2 ** log2cpm(P[np.where(ok)[0]]) - 1
    meta = pd.Series(uniq[ok]).str.split("|", expand=True)
    df = pd.DataFrame(L, columns=full.var_names)
    df["dataset"], df["lineage"] = meta[0].values, meta[1].values
    out = df.groupby(["dataset", "lineage"]).mean()          # mean CPM over patients
    return out


def contribution(lcpm_pdac: pd.DataFrame, frac: pd.DataFrame) -> pd.DataFrame:
    """Share of each gene's transcripts in PDAC tissue that come from fibroblasts (abundance-weighted)."""
    rows = []
    for ds in DATASETS:
        t, f = lcpm_pdac.loc[ds], frac.loc[ds]
        f = f.reindex(t.index).fillna(0)
        tot = (t.mul(f, axis=0)).sum(0)
        fib = t.loc["Fibroblast"] * f.loc["Fibroblast"]
        rows.append(pd.DataFrame({"dataset": ds, "gene": t.columns, "fib_contribution": (fib / tot.replace(0, np.nan)).values}))
    c = pd.concat(rows, ignore_index=True)
    c["pass"] = c.fib_contribution >= S["min_fib_contribution"]
    return c


def specificity(lcpm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        t = lcpm.loc[ds]
        if "Fibroblast" not in t.index:
            continue
        fib = t.loc["Fibroblast"]
        non = t.loc[t.index.intersection(S["nonmesenchymal_lineages"])].max(0)
        per = t.loc["Pericyte_SMC"] if "Pericyte_SMC" in t.index else 0 * fib
        rows.append(pd.DataFrame({"dataset": ds, "gene": fib.index, "fib_cpm": fib.values,
                                  "ratio_vs_nonmes": ((fib + 1) / (non + 1)).values,
                                  "ratio_vs_pericyte": ((fib + 1) / (per + 1)).values,
                                  "top_nonmes_lineage": t.loc[t.index.intersection(S["nonmesenchymal_lineages"])].idxmax(0).values}))
    sp_ = pd.concat(rows, ignore_index=True)
    sp_["pass"] = (sp_.ratio_vs_nonmes >= S["specificity_vs_nonmesenchymal"]) & \
                  (sp_.ratio_vs_pericyte >= S["specificity_vs_pericyte"])
    return sp_


# ---------------------------------------------------------------- scoring / LODO
def score(caf: ad.AnnData, genes: list[str]) -> np.ndarray:
    """Mean of log-normalised expression (X is log1p CP10k)."""
    idx = [caf.var_names.get_loc(g) for g in genes if g in caf.var_names]
    X = caf.X[:, idx]
    return np.asarray(X.mean(1)).ravel()


def main():
    caf = ad.read_h5ad(ROOT / S["caf_h5ad"])
    cl = caf.obs[S["cluster_key"]].astype(str)
    caf = caf[(~cl.isin(S["exclude_from_reference"])).to_numpy() &
              caf.obs.condition.isin(CFG["discovery_conditions"]).to_numpy()].copy()
    caf.obs["is_target"] = caf.obs[S["cluster_key"]].astype(str).isin(S["target_clusters"]).to_numpy()
    caf.layers["counts"] = sp.csr_matrix(caf.layers["counts"])
    print(f"CAFs used: {caf.n_obs} (target {caf.obs.is_target.sum()}), patients {caf.obs.patient_key.nunique()}")

    print("1. state DE")
    de = paired_de(caf, DATASETS)
    de.to_csv(OUT / "de_per_dataset.tsv", sep="\t", index=False)

    print("2. lineage specificity")
    lcpm = lineage_cpm()
    lcpm.to_csv(OUT / "lineage_cpm.tsv", sep="\t")
    spec = specificity(lcpm)                                  # kept for reporting (max-lineage ratios)
    spec.to_csv(OUT / "specificity.tsv", sep="\t", index=False)
    ann = pd.read_csv(ROOT / "annotate_global" / "global_full_annotation.tsv.gz", sep="\t", index_col=0)
    ann = ann[ann.condition.isin(S["contribution_conditions"]) &
              ~ann.lineage.isin(["Ambiguous", "Low_quality", "Hepatocyte_like"])]
    frac = pd.crosstab(ann.dataset, ann.lineage, normalize="index")
    lcpm_pdac = lineage_cpm(S["contribution_conditions"])
    contrib = contribution(lcpm_pdac, frac)
    contrib.to_csv(OUT / "fibroblast_contribution.tsv", sep="\t", index=False)
    spec_ok = set(contrib.groupby("gene")["pass"].all().loc[lambda s: s].index)
    spec_ok -= set(S.get("exclude_genes", []))   # v4: platform-unreliable genes (Visium HD slide dropout)
    print(f"   genes with >= {S['min_fib_contribution']:.0%} fibroblast contribution in all datasets: {len(spec_ok)}")

    pooled = pooled_test(DATASETS)
    pooled.to_csv(OUT / "pooled_test.tsv", sep="\t")
    prog = select_state_genes(de, DATASETS, spec_ok, pooled)
    prog.to_csv(OUT / "cthrc1_program.tsv", sep="\t")
    print(f"CTHRC1+ program: {len(prog)} genes\n{prog.round(2).to_string()}")

    # pan-CAF reference: fibroblast-specific, flat across CAF states, highly expressed
    flat = de.groupby("gene").log2fc.apply(lambda v: v.abs().max())
    fib_cpm = spec.groupby("gene").fib_cpm.mean()
    pan = pd.DataFrame({"max_abs_log2fc": flat, "mean_fib_cpm": fib_cpm}).dropna()
    pan = pan[pan.index.isin(spec_ok) & (pan.max_abs_log2fc < S["pancaf_max_abs_log2fc"]) &
              (pan.mean_fib_cpm >= S.get("pancaf_min_fib_cpm", 0))]
    pan = pan.sort_values("mean_fib_cpm", ascending=False).head(S["pancaf_n"])
    pan.to_csv(OUT / "pancaf_program.tsv", sep="\t")
    print(f"pan-CAF reference: {list(pan.index)}")

    # leave-one-dataset-out
    lodo, sigs = [], {}
    for held in DATASETS:
        train = [d for d in DATASETS if d != held]
        g = list(select_state_genes(de, train, spec_ok, pooled_test(train)).index)
        sigs[held] = set(g)
        m = (caf.obs.dataset == held).to_numpy()
        s = score(caf[m], g) if g else np.zeros(m.sum())
        auc = roc_auc_score(caf.obs.is_target[m], s) if g else np.nan
        cth = score(caf[m], ["CTHRC1"])
        lodo.append({"held_out": held, "n_genes": len(g), "auroc_program": auc,
                     "auroc_CTHRC1_alone": roc_auc_score(caf.obs.is_target[m], cth),
                     "overlap_with_full": len(set(g) & set(prog.index)) / max(len(prog), 1)})
    for a_, b_ in combinations(DATASETS, 2):
        j = len(sigs[a_] & sigs[b_]) / max(len(sigs[a_] | sigs[b_]), 1)
        lodo.append({"held_out": f"jaccard {a_} vs {b_}", "n_genes": np.nan, "auroc_program": j})
    lodo = pd.DataFrame(lodo)
    lodo.to_csv(OUT / "lodo.tsv", sep="\t", index=False)
    print(lodo.round(3).to_string(index=False))

    (OUT / "programs.yaml").write_text(yaml.safe_dump({
        "CTHRC1_matrix_CAF": list(prog.index), "pan_CAF": list(pan.index),
        "rules": S, "frozen_before_outcome_data": True}, sort_keys=False))


if __name__ == "__main__":
    main()
