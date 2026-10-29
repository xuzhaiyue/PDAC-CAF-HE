"""Fig 1 step 02 — 成纤维细胞整合与聚类（pdac_scvi 环境）。

Input : results/fig1/fibro_raw.h5ad（只用 in_cap 细胞，每患者×条件 ≤ 500）
        第二轮：--exclude 去掉上一轮的非 CAF 亚群（周细胞、平滑肌、低质量、污染）后，只用 CAF 重新整合
Method: seurat_v3 HVG（按数据集）→ scVI（batch = 患者，协变量 = 数据集）→ kNN → Leiden（多分辨率）→ UMAP
Output: results/fig1/integrate/
  fibro_integrated.h5ad       latent (X_scVI), UMAP, leiden_*，log1p 表达
  scvi_model/                 训练好的 scVI 模型（之后投射其余细胞）
  cluster_profile_<res>.tsv   每个亚群：大小、数据集/条件构成、污染标志评分、CAF 分型评分、CTHRC1 表达
  markers_<res>.tsv           每个亚群 top 标志基因（Wilcoxon）
  umap_*.png, dotplot_<res>.png
"""
from __future__ import annotations

import os

# numba's default OpenMP threading layer segfaults in scanpy's sparse mean/variance kernel in this env
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
from pathlib import Path  # noqa: E402

import argparse

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("OMP_NUM_THREADS", "8")
import scvi  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)
CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fig1.yaml").read_text())
RES = [0.3, 0.5, 0.8, 1.2]
MAIN_RES = 0.5
scvi.settings.seed = CFG["random_seed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="integrate")
    ap.add_argument("--input", default="fibro_raw.h5ad", help="file in results/fig1/")
    ap.add_argument("--mode", default="caf", choices=["caf", "global"],
                    help="caf: contaminant + CAF-subtype scores; global: major-lineage marker scores")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--exclude-from", default=None, help="previous fibro_integrated.h5ad")
    ap.add_argument("--exclude-key", default="leiden_0.5")
    ap.add_argument("--exclude", default="", help="comma-separated cluster ids to drop")
    args = ap.parse_args()
    global OUT
    OUT = Path(CFG["paths"]["out_dir"]) / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(OUT)

    a = sc.read_h5ad(Path(CFG["paths"]["out_dir"]) / args.input)
    a = a[a.obs.in_cap].copy()
    if args.exclude_from:
        prev = sc.read_h5ad(args.exclude_from, backed="r").obs[[args.exclude_key]]
        drop = set(args.exclude.split(","))
        keep = prev.index[~prev[args.exclude_key].astype(str).isin(drop)]
        a = a[a.obs_names.isin(keep)].copy()
        a.obs["prev_cluster"] = prev.loc[a.obs_names, args.exclude_key].astype(str).values
        print(f"excluded clusters {sorted(drop)} from {args.exclude_from}", flush=True)
    print(f"cells: {a.n_obs}; patients: {a.obs.patient_key.nunique()}", flush=True)
    sc.pp.filter_genes(a, min_cells=20)
    a.layers["counts"] = a.X.copy()
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=3000, layer="counts",
                                batch_key="dataset", subset=False)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    hv = a[:, a.var.highly_variable].copy()
    scvi.model.SCVI.setup_anndata(hv, layer="counts", batch_key="patient_key",
                                  categorical_covariate_keys=["dataset"])
    model = scvi.model.SCVI(hv, n_latent=20, n_layers=2, gene_likelihood="nb")
    model.train(max_epochs=args.max_epochs, early_stopping=True, accelerator="cpu", batch_size=512,
                plan_kwargs={"lr": 1e-3})
    model.save(OUT / "scvi_model", overwrite=True)
    a.obsm["X_scVI"] = model.get_latent_representation()

    sc.pp.neighbors(a, use_rep="X_scVI", n_neighbors=15, random_state=0)
    for r in RES:
        sc.tl.leiden(a, resolution=r, key_added=f"leiden_{r:g}", flavor="igraph", n_iterations=2, random_state=0)
    sc.tl.umap(a, random_state=0)

    # module scores
    var = set(a.var_names)
    if args.mode == "global":
        score_sets = {f"lin_{k}": v for k, v in CFG["lineage_markers"].items()}
    else:
        score_sets = {**{f"contam_{k}": v for k, v in CFG["contaminant_markers"].items()},
                      **{f"ref_{k}": v for k, v in CFG["caf_reference"].items()}}
    for k, genes in score_sets.items():
        g = [x for x in genes if x in var]
        sc.tl.score_genes(a, g, score_name=k, random_state=0)
    a.obs["CTHRC1_expr"] = np.asarray(a[:, "CTHRC1"].X.todense()).ravel()

    for r in RES:
        key = f"leiden_{r:g}"
        grp = a.obs.groupby(key, observed=True)
        prof = pd.concat([
            grp.size().rename("n_cells"),
            grp.patient_key.nunique().rename("n_patients"),
            pd.crosstab(a.obs[key], a.obs.dataset, normalize="index").add_prefix("ds_"),
            pd.crosstab(a.obs[key], a.obs.condition, normalize="index").add_prefix("cond_"),
            grp[[c for c in a.obs.columns if c.startswith(("contam_", "ref_", "lin_"))]].mean(),
            grp.CTHRC1_expr.mean().rename("CTHRC1_mean"),
            grp.CTHRC1_expr.apply(lambda v: (v > 0).mean()).rename("CTHRC1_frac_pos"),
        ], axis=1).round(3)
        prof.to_csv(OUT / f"cluster_profile_{r:g}.tsv", sep="\t")
        if r == MAIN_RES:
            print(prof.to_string())
            sc.tl.rank_genes_groups(a, key, method="wilcoxon", n_genes=50, use_raw=False)
            mk = sc.get.rank_genes_groups_df(a, None)
            mk.to_csv(OUT / f"markers_{r:g}.tsv", sep="\t", index=False)
            top = mk.groupby("group").head(8).groupby("group").names.apply(lambda s: ", ".join(s))
            print(top.to_string())

    if args.mode == "global":
        keys = ["dataset", "condition", "author_label", f"leiden_{MAIN_RES:g}", "CTHRC1_expr"] + \
               [f"lin_{k}" for k in CFG["lineage_markers"]]
    else:
        keys = ["dataset", "condition", "author_label", f"leiden_{MAIN_RES:g}", "CTHRC1_expr",
                "ref_myCAF", "ref_iCAF", "ref_apCAF", "contam_pericyte", "contam_smooth_muscle",
                "contam_epithelial", "contam_immune"]
    fig = sc.pl.umap(a, color=keys, ncols=4, frameon=False, show=False, return_fig=True, size=4)
    fig.savefig(OUT / "umap_overview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    marker_src = CFG["lineage_markers"] if args.mode == "global" else {
        **CFG["caf_reference"], "pericyte": CFG["contaminant_markers"]["pericyte"],
        "SMC": CFG["contaminant_markers"]["smooth_muscle"]}
    dot_genes = {k: [g for g in v if g in var][:6] for k, v in marker_src.items()}
    dp = sc.pl.dotplot(a, dot_genes, groupby=f"leiden_{MAIN_RES:g}", show=False, return_fig=True, standard_scale="var")
    dp.savefig(OUT / f"dotplot_{MAIN_RES:g}.png", dpi=110, bbox_inches="tight")

    a.write_h5ad(OUT / "fibro_integrated.h5ad", compression="gzip")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
