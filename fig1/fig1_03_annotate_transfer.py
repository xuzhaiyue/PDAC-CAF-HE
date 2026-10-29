"""Fig 1 step 03 — global lineage annotation → label transfer to all cells → extraction of mesenchymal lineages (pdac_scvi environment).

1. In the subsampled global atlas (integrate_global), each Leiden cluster is annotated automatically from lin_* marker scores:
   the top-scoring lineage; top score < min_score or margin to the second < min_margin is called Ambiguous;
   global_annotation_overrides in the config can correct clusters manually (recorded in the table).
2. The same scVI model projects all 418,000 cells into the latent space; kNN (k=15) transfers labels from annotated subsampled cells, with confidence.
3. Fibroblast / Pericyte / Smooth_muscle cells with confidence >= min_conf → mesenchymal_raw.h5ad
   (subsampled to max_cells_per_patient per patient × condition, for the CAF layer).

Outputs: results/fig1/annotate_global/
  cluster_annotation.tsv        per cluster: lineage scores, automatic annotation, manual correction flag
  global_full_annotation.tsv.gz   all cells: dataset, patient, condition, author label, study lineage, confidence
  lineage_vs_author.tsv         study annotation versus author labels (supplementary)
  lineage_composition.tsv       lineage composition per dataset / condition
results/fig1/mesenchymal_raw.h5ad
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import yaml  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

import scvi  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)
CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fig1.yaml").read_text())
ROOT = Path(CFG["paths"]["out_dir"])
OUT = ROOT / "annotate_global"
OUT.mkdir(parents=True, exist_ok=True)
MESENCHYMAL = ["Fibroblast", "Pericyte", "Smooth_muscle", "Pericyte_SMC"]


def auto_annotate(a, key, min_score, min_margin):
    lin_cols = [c for c in a.obs.columns if c.startswith("lin_") and c != "lin_Cycling"]
    m = a.obs.groupby(key, observed=True)[lin_cols + ["lin_Cycling"]].mean()
    rows = []
    for cl, r in m.iterrows():
        s = r[lin_cols].sort_values(ascending=False)
        top, second = s.index[0][4:], s.index[1][4:]
        label = top if (s.iloc[0] >= min_score and s.iloc[0] - s.iloc[1] >= min_margin) else "Ambiguous"
        rows.append({"cluster": str(cl), "auto_label": label, "top": top, "top_score": round(s.iloc[0], 3),
                     "second": second, "second_score": round(s.iloc[1], 3),
                     "cycling_score": round(r["lin_Cycling"], 3)})
    tab = pd.DataFrame(rows).set_index("cluster")
    tab = tab.join(m.round(3))
    tab["n_cells"] = a.obs[key].value_counts().reindex(tab.index.astype(str)).values
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="leiden_0.5")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--min-margin", type=float, default=0.2)
    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--annotate-only", action="store_true", help="write cluster_annotation.tsv and stop")
    args = ap.parse_args()

    g = sc.read_h5ad(ROOT / "integrate_global" / "fibro_integrated.h5ad")
    g.obs[args.key] = g.obs[args.key].astype(str)
    tab = auto_annotate(g, args.key, args.min_score, args.min_margin)
    overrides = {str(k): v for k, v in (CFG.get("global_annotation_overrides") or {}).items()}
    tab["override"] = tab.index.map(lambda c: overrides.get(c, ""))
    tab["final_label"] = np.where(tab.override != "", tab.override, tab.auto_label)
    tab.to_csv(OUT / "cluster_annotation.tsv", sep="\t")
    print(tab[["n_cells", "auto_label", "top_score", "second", "second_score", "cycling_score", "override",
               "final_label"]].to_string())
    if args.annotate_only:
        return

    g.obs["lineage"] = g.obs[args.key].map(tab.final_label).values
    # --- label transfer to all cells through the same scVI latent space
    full = sc.read_h5ad(ROOT / "global_full_raw.h5ad")
    model_dir = ROOT / "integrate_global" / "scvi_model"
    # scvi saves the training genes inside model.pt; the AnnData itself was not saved
    hv_genes = list(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)["var_names"])
    full_hv = full[:, hv_genes].copy()
    full_hv.layers["counts"] = full_hv.X.copy()
    lat = scvi.model.SCVI.load(model_dir, adata=full_hv).get_latent_representation(full_hv, batch_size=4096)

    ref_ok = ~g.obs.lineage.isin(["Ambiguous", "Low_quality"])
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=8)
    knn.fit(g.obsm["X_scVI"][ref_ok.to_numpy()], g.obs.lineage[ref_ok].to_numpy())
    proba = knn.predict_proba(lat)
    full.obs["lineage"] = knn.classes_[proba.argmax(1)]
    full.obs["lineage_conf"] = proba.max(1)
    in_ref = full.obs_names.isin(g.obs_names)
    full.obs.loc[in_ref, "lineage"] = g.obs.lineage.reindex(full.obs_names[in_ref]).values
    full.obs.loc[in_ref, "lineage_conf"] = 1.0

    ann = full.obs[["dataset", "patient_key", "condition", "author_label", "lineage", "lineage_conf"]].copy()
    ann.to_csv(OUT / "global_full_annotation.tsv.gz", sep="\t")  # pdac_scvi env has no parquet engine
    pd.crosstab([ann.dataset, ann.author_label], ann.lineage).to_csv(OUT / "lineage_vs_author.tsv", sep="\t")
    comp = pd.crosstab([ann.dataset, ann.condition], ann.lineage, normalize="index").round(4)
    comp.to_csv(OUT / "lineage_composition.tsv", sep="\t")
    print(pd.crosstab(ann.dataset, ann.lineage).to_string())
    print(f"label confidence: median {ann.lineage_conf.median():.2f}; < {args.min_conf}: "
          f"{(ann.lineage_conf < args.min_conf).mean():.1%}")

    # --- mesenchymal lineage → input for the CAF layer
    rng = np.random.default_rng(CFG["random_seed"])
    mes = full[full.obs.lineage.isin(MESENCHYMAL) & (full.obs.lineage_conf >= args.min_conf)].copy()
    mes.obs["in_cap"] = False
    cap = CFG["max_cells_per_patient"]
    for _, idx in mes.obs.groupby(["patient_key", "condition"], observed=True).indices.items():
        take = idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)
        mes.obs.iloc[take, mes.obs.columns.get_loc("in_cap")] = True
    mes.obs["discovery"] = mes.obs.condition.isin(CFG["discovery_conditions"])
    mes.layers["counts"] = mes.X.copy()
    mes.write_h5ad(ROOT / "mesenchymal_raw.h5ad", compression="gzip")
    print(f"mesenchymal cells: {mes.n_obs} ({mes.obs.lineage.value_counts().to_dict()}); in cap: {mes.obs.in_cap.sum()}")


if __name__ == "__main__":
    main()
