"""Fig 1 step 01 — build inputs (hemorph environment).

Outputs (results/fig1/):
  fibro_raw.h5ad          all fibroblast-like cells (raw counts, genes intersected across the three datasets)
                          obs: dataset, patient_key, condition, author_label, in_cap (within the per-patient subsampling cap)
  lineage_pseudobulk.h5ad summed raw counts per dataset × author lineage × patient (tests whether the programme is high only in CAFs)
  inputs_summary.tsv      cell and patient numbers per dataset / condition / lineage
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import yaml

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fig1.yaml").read_text())
PANIN = Path(CFG["paths"]["panin"])
OUT = Path(CFG["paths"]["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(CFG["random_seed"])


def counts_rows(a: ad.AnnData, idx: np.ndarray, chunk: int = 20000) -> sp.csr_matrix:
    """Raw counts for selected rows of a backed AnnData, in chunks."""
    src = a.layers["counts"] if "counts" in a.layers else a.X
    parts = []
    for i in range(0, len(idx), chunk):
        block = src[np.sort(idx[i:i + chunk])]
        parts.append(sp.csr_matrix(block))
    return sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, a.n_vars))


def pseudobulk(X: sp.csr_matrix, keys: pd.Series) -> tuple[pd.Index, sp.csr_matrix]:
    codes, uniq = pd.factorize(keys, sort=True)
    G = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
    return pd.Index(uniq), (G @ X).tocsr()


def load_h5ad_dataset(name: str, c: dict):
    a = ad.read_h5ad(PANIN / c["h5ad"], backed="r")
    obs = a.obs
    lin = obs[c["lineage_col"]].astype(str)
    patient = obs[c["patient_col"]].astype(str)
    cond = obs[c["condition_col"]].astype(str).map(c["condition_map"]).fillna("other")
    meta = pd.DataFrame({"dataset": name, "patient_key": name + ":" + patient.values,
                         "condition": cond.values, "author_label": lin.values}, index=obs.index.astype(str))
    # pseudobulk over all lineages, chunked to keep memory flat
    pb_keys = meta.dataset + "|" + meta.author_label + "|" + meta.condition + "|" + meta.patient_key
    keys_all, sums = None, None
    n = a.n_obs
    acc = {}
    for i in range(0, n, 20000):
        rows = np.arange(i, min(i + 20000, n))
        X = counts_rows(a, rows)
        k, S = pseudobulk(X, pb_keys.iloc[rows].reset_index(drop=True))
        for j, key in enumerate(k):
            acc[key] = acc[key] + S[j] if key in acc else S[j]
        print(f"   {name}: pseudobulk {min(i + 20000, n)}/{n}", flush=True)
    keys_all = pd.Index(sorted(acc))
    sums = sp.vstack([acc[k] for k in keys_all]).tocsr()
    fib_idx = np.where(lin.isin(c["fibro_labels"]).to_numpy())[0]
    Xf = counts_rows(a, fib_idx)
    return meta.iloc[fib_idx], Xf, pd.Index(a.var_names), keys_all, sums


def load_mtx_dataset(name: str, c: dict):
    d = PANIN / c["mtx_dir"]
    feats = pd.read_csv(d / f"{c['prefix']}features.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(d / f"{c['prefix']}barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    md = pd.read_csv(d / f"{c['prefix']}metadata.csv.gz", index_col=0)
    M = scipy.io.mmread(d / f"{c['prefix']}matrix.mtx.gz").tocsr()
    if M.shape[0] == len(feats) and M.shape[1] == len(barcodes):
        M = M.T.tocsr()
    assert M.shape == (len(barcodes), len(feats)), M.shape
    md = md.loc[barcodes.values]
    lin = md[c["lineage_col"]].astype(str)
    cond = md[c["condition_col"]].astype(str).map(c["condition_map"]).fillna(c.get("default_condition", "other"))
    patient = md[c["patient_col"]].astype(str)
    meta = pd.DataFrame({"dataset": name, "patient_key": name + ":" + patient.values,
                         "condition": cond.values, "author_label": lin.values}, index=barcodes.values)
    genes = pd.Index(feats[1].astype(str))
    # duplicated symbols: keep first occurrence
    keep = ~genes.duplicated()
    M, genes = M[:, np.where(keep)[0]], genes[keep]
    pb_keys = (meta.dataset + "|" + meta.author_label + "|" + meta.condition + "|" + meta.patient_key).reset_index(drop=True)
    keys_all, sums = pseudobulk(M, pb_keys)
    fib_idx = np.where(lin.isin(c["fibro_labels"]).to_numpy())[0]
    return meta.iloc[fib_idx], M[fib_idx], genes, keys_all, sums


def main():
    fibro_parts, pb_parts = [], []
    for name, c in CFG["datasets"].items():
        print(f"== {name}", flush=True)
        loader = load_mtx_dataset if "mtx_dir" in c else load_h5ad_dataset
        meta, Xf, genes, pk, ps = loader(name, c)
        gi = ~pd.Index(genes).duplicated()
        fibro_parts.append(ad.AnnData(sp.csr_matrix(Xf[:, np.where(gi)[0]]), obs=meta,
                                      var=pd.DataFrame(index=pd.Index(genes)[gi])))
        pb = ad.AnnData(sp.csr_matrix(ps[:, np.where(gi)[0]]), var=pd.DataFrame(index=pd.Index(genes)[gi]))
        parts = pd.Series(pk).str.split("|", expand=True)
        pb.obs = pd.DataFrame({"dataset": parts[0].values, "author_label": parts[1].values,
                               "condition": parts[2].values, "patient_key": parts[3].values},
                              index=pd.Index(pk))
        pb.obs["n_cells"] = np.nan
        pb_parts.append(pb)
        print(f"   fibroblast-lineage cells: {Xf.shape[0]}; genes: {gi.sum()}", flush=True)

    common = sorted(set.intersection(*[set(p.var_names) for p in fibro_parts]))
    print(f"common genes across datasets: {len(common)}")
    fib = ad.concat([p[:, common] for p in fibro_parts], join="inner", merge="same")
    fib.obs_names_make_unique()
    fib.layers["counts"] = fib.X.copy()
    fib.obs["n_counts"] = np.asarray(fib.X.sum(1)).ravel()
    fib.obs["n_genes"] = np.asarray((fib.X > 0).sum(1)).ravel()

    # per patient x condition cap (balanced input for integration/clustering)
    cap = CFG["max_cells_per_patient"]
    fib.obs["in_cap"] = False
    for pk, idx in fib.obs.groupby(["patient_key", "condition"], observed=True).indices.items():
        take = idx if len(idx) <= cap else RNG.choice(idx, cap, replace=False)
        fib.obs.iloc[take, fib.obs.columns.get_loc("in_cap")] = True
    fib.obs["discovery"] = fib.obs.condition.isin(CFG["discovery_conditions"])
    for col in ["dataset", "patient_key", "condition", "author_label"]:
        fib.obs[col] = fib.obs[col].astype("category")
    fib.write_h5ad(OUT / "fibro_raw.h5ad", compression="gzip")

    pb = ad.concat([p[:, common] for p in pb_parts], join="inner")
    pb.write_h5ad(OUT / "lineage_pseudobulk.h5ad", compression="gzip")

    s = (fib.obs.groupby(["dataset", "condition", "author_label"], observed=True)
         .agg(cells=("patient_key", "size"), patients=("patient_key", "nunique")).reset_index())
    s["cells_in_cap"] = fib.obs[fib.obs.in_cap].groupby(["dataset", "condition", "author_label"], observed=True).size().values
    s.to_csv(OUT / "inputs_summary.tsv", sep="\t", index=False)
    print(s.to_string(index=False))
    print(f"fibro: {fib.shape}; discovery in cap: {(fib.obs.discovery & fib.obs.in_cap).sum()}; "
          f"pseudobulk groups: {pb.n_obs}")


if __name__ == "__main__":
    sys.exit(main())
