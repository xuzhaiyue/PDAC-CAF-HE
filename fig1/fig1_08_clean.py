"""Fig 1 step 08 — cleaning (hemorph environment). Two subcommands:

  cells   cell-level filtering of mesenchymal cells: Scrublet doublets + within-dataset MAD QC + mitochondrial fraction
          → results/fig1/mesenchymal_clean_raw.h5ad, results/fig1/qc/removed_cells.tsv.gz
  flag    read an integrated h5ad and flag non-CAF / low-quality clusters with the numerical rules in config.cluster_exclusion
          → <integrate_dir>/cluster_flags.tsv, and print the --exclude argument

Every removed cell and the reason are recorded (supplementary table).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fig1.yaml").read_text())
ROOT = Path(CFG["paths"]["out_dir"])
Q, X_RULES = CFG["cell_qc"], CFG["cluster_exclusion"]


def mad(x):
    return np.median(np.abs(x - np.median(x))) * 1.4826


def cmd_cells():
    mes = ad.read_h5ad(ROOT / "mesenchymal_raw.h5ad")
    qc = pd.read_csv(ROOT / "qc" / "cell_qc.tsv.gz", sep="\t", index_col=0)
    o = mes.obs.join(qc[["doublet_score", "predicted_doublet", "n_genes", "n_umi", "pct_mt"]], how="left")
    reason = pd.Series("", index=o.index)
    for ds, d in o.groupby("dataset", observed=True):
        lg, lu = np.log(d.n_genes.clip(lower=1)), np.log(d.n_umi.clip(lower=1))
        g_min = max(Q["min_genes"], float(np.exp(np.median(lg) - Q["mad_k"] * mad(lg))))
        u_min = float(np.exp(np.median(lu) - Q["mad_k"] * mad(lu)))
        mt_max = max(Q["max_mt_floor"], float(np.median(d.pct_mt) + Q["mad_k"] * mad(d.pct_mt)))
        r = pd.Series("", index=d.index)
        r[d.n_genes < g_min] += "low_genes;"
        r[d.n_umi < u_min] += "low_umi;"
        r[d.pct_mt > mt_max] += "high_mt;"
        if Q["remove_predicted_doublets"]:
            r[d.predicted_doublet.fillna(False).astype(bool)] += "doublet;"
        reason.loc[d.index] = r
        print(f"{ds}: thresholds genes>={g_min:.0f}, umi>={u_min:.0f}, mt<={mt_max:.1f}% | "
              f"removed {(r != '').sum()} / {len(d)}")
    removed = o.loc[reason != "", ["dataset", "patient_key", "condition", "lineage", "n_genes", "n_umi",
                                   "pct_mt", "doublet_score"]].assign(reason=reason[reason != ""])
    removed.to_csv(ROOT / "qc" / "removed_cells.tsv.gz", sep="\t")
    summ = removed.assign(reason=removed.reason.str.rstrip(";")).groupby(["dataset", "reason"]).size()
    print(summ.to_string())

    keep = reason == ""
    clean = mes[keep.reindex(mes.obs_names).to_numpy()].copy()
    rng = np.random.default_rng(CFG["random_seed"])
    clean.obs["in_cap"] = False
    cap = CFG["max_cells_per_patient"]
    for _, idx in clean.obs.groupby(["patient_key", "condition"], observed=True).indices.items():
        take = idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)
        clean.obs.iloc[take, clean.obs.columns.get_loc("in_cap")] = True
    clean.write_h5ad(ROOT / "mesenchymal_clean_raw.h5ad", compression="gzip")
    print(f"clean mesenchymal: {clean.n_obs} cells ({clean.obs.lineage.value_counts().to_dict()}); in cap {clean.obs.in_cap.sum()}")


def cmd_flag(integrated: Path, key: str):
    a = ad.read_h5ad(integrated)
    C = sp.csr_matrix(a.layers["counts"])
    v = list(a.var_names)
    pos = {g: (C[:, v.index(g)].toarray().ravel() > 0) for g in ["RGS5", "NOTCH3", "MYH11", "PDGFRA"] if g in v}
    df = pd.DataFrame({"cl": a.obs[key].astype(str).values,
                       "n_genes": np.asarray((C > 0).sum(1)).ravel(), **pos,
                       "epi": a.obs.get("contam_epithelial", pd.Series(0, index=a.obs_names)).values,
                       "imm": a.obs.get("contam_immune", pd.Series(0, index=a.obs_names)).values,
                       "patient": a.obs.patient_key.astype(str).values})
    overall = df.n_genes.median()
    g = df.groupby("cl")
    f = pd.DataFrame({
        "n_cells": g.size(), "n_patients": g.patient.nunique(),
        "pericyte_smc_mean_frac": g[["RGS5", "NOTCH3", "MYH11"]].mean().mean(1),
        "PDGFRA_frac": g.PDGFRA.mean(),
        "median_genes_ratio": g.n_genes.median() / overall,
        "epithelial_score": g.epi.mean(), "immune_score": g.imm.mean()}).round(3)
    reasons = []
    for cl, r in f.iterrows():
        why = []
        if r.pericyte_smc_mean_frac >= X_RULES["pericyte_smc_mean_frac"]:
            why.append("pericyte/SMC markers")
        if r.median_genes_ratio < X_RULES["low_complexity_ratio"]:
            why.append("low complexity")
        if r.epithelial_score > X_RULES["epithelial_score"]:
            why.append("epithelial contamination")
        if r.immune_score > X_RULES["immune_score"]:
            why.append("immune contamination")
        reasons.append("; ".join(why))
    f["exclude_reason"] = reasons
    f.to_csv(integrated.parent / "cluster_flags.tsv", sep="\t")
    print(f.to_string())
    ex = [str(c) for c in f.index[f.exclude_reason != ""]]
    print(f"\n--exclude {','.join(ex) if ex else '(none)'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cells")
    fl = sub.add_parser("flag")
    fl.add_argument("integrated")
    fl.add_argument("--key", default="leiden_0.5")
    args = ap.parse_args()
    if args.cmd == "cells":
        cmd_cells()
    else:
        cmd_flag(Path(args.integrated), args.key)


if __name__ == "__main__":
    main()
