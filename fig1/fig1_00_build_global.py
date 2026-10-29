"""Fig 1 step 00 — 全局图谱输入（hemorph 环境）。

三个单细胞数据集的所有细胞（GSE278688 排除 PBMC），每患者×条件最多 global_max_cells_per_patient 个，
原始计数，基因取交集。作者标注保留在 author_label，只作对照；正式大类注释由全局聚类决定。

Output: results/fig1/global_raw.h5ad（抽样，用于整合与注释）
        results/fig1/global_full_raw.h5ad（--cap 0：全部细胞，用于标签传递）
"""
from __future__ import annotations

import argparse
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
CAP = CFG["global_max_cells_per_patient"]  # overridden by --cap


def capped_index(meta: pd.DataFrame) -> np.ndarray:
    take = []
    for _, idx in meta.reset_index(drop=True).groupby(["patient_key", "condition"], observed=True).indices.items():
        take.append(idx if CAP <= 0 or len(idx) <= CAP else RNG.choice(idx, CAP, replace=False))
    return np.sort(np.concatenate(take))


def from_h5ad(name: str, c: dict) -> ad.AnnData:
    a = ad.read_h5ad(PANIN / c["h5ad"], backed="r")
    o = a.obs
    meta = pd.DataFrame({
        "dataset": name,
        "patient_key": name + ":" + o[c["patient_col"]].astype(str).values,
        "condition": o[c["condition_col"]].astype(str).map(c["condition_map"]).fillna("other").values,
        "author_label": o[c["lineage_col"]].astype(str).values}, index=o.index.astype(str))
    meta = meta[~meta.condition.isin(c.get("exclude_conditions", []))]
    pos = np.where(o.index.astype(str).isin(meta.index))[0]
    sel = pos[capped_index(meta)]
    src = a.layers["counts"] if "counts" in a.layers else a.X
    X = sp.vstack([sp.csr_matrix(src[sel[i:i + 20000]]) for i in range(0, len(sel), 20000)]).tocsr()
    return ad.AnnData(X, obs=meta.loc[o.index[sel].astype(str)], var=pd.DataFrame(index=a.var_names.astype(str)))


def from_mtx(name: str, c: dict) -> ad.AnnData:
    d = PANIN / c["mtx_dir"]
    feats = pd.read_csv(d / f"{c['prefix']}features.tsv.gz", sep="\t", header=None)
    bc = pd.read_csv(d / f"{c['prefix']}barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    md = pd.read_csv(d / f"{c['prefix']}metadata.csv.gz", index_col=0).loc[bc.values]
    M = scipy.io.mmread(d / f"{c['prefix']}matrix.mtx.gz").tocsr()
    if M.shape == (len(feats), len(bc)):
        M = M.T.tocsr()
    meta = pd.DataFrame({
        "dataset": name,
        "patient_key": name + ":" + md[c["patient_col"]].astype(str).values,
        "condition": md[c["condition_col"]].astype(str).map(c["condition_map"])
                     .fillna(c.get("default_condition", "other")).values,
        "author_label": md[c["lineage_col"]].astype(str).values}, index=bc.values)
    keep_rows = np.where(~meta.condition.isin(c.get("exclude_conditions", [])).to_numpy())[0]
    meta = meta.iloc[keep_rows]
    sel = keep_rows[capped_index(meta)]
    genes = pd.Index(feats[1].astype(str))
    gk = np.where(~genes.duplicated())[0]
    obs = meta.loc[bc.values[sel]]
    obs.index.name = None
    return ad.AnnData(M[sel][:, gk], obs=obs, var=pd.DataFrame(index=pd.Index(genes[gk].values, name=None)))


def main():
    global CAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=CFG["global_max_cells_per_patient"], help="0 = keep all cells")
    ap.add_argument("--name", default="global_raw.h5ad")
    args = ap.parse_args()
    CAP = args.cap
    parts = []
    for name, c in CFG["datasets"].items():
        a = from_mtx(name, c) if "mtx_dir" in c else from_h5ad(name, c)
        a.var_names_make_unique()
        print(f"{name}: {a.n_obs} cells (cap {CAP}/patient×condition), {a.obs.patient_key.nunique()} patients; "
              f"conditions {a.obs.condition.value_counts().to_dict()}", flush=True)
        parts.append(a)
    common = sorted(set.intersection(*[set(p.var_names) for p in parts]))
    g = ad.concat([p[:, common] for p in parts], join="inner")
    g.obs_names_make_unique()
    g.layers["counts"] = g.X.copy()
    for col in ["dataset", "patient_key", "condition", "author_label"]:
        g.obs[col] = g.obs[col].astype("category")
    g.obs["in_cap"] = True
    g.write_h5ad(OUT / args.name, compression="gzip")
    print(f"global: {g.shape}")
    print(pd.crosstab(g.obs.author_label, g.obs.dataset).to_string())


if __name__ == "__main__":
    sys.exit(main())
