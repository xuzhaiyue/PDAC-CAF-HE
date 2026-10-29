"""Fig 1 step 07 — cell-level QC and doublet detection (pdac_scvi environment; scikit-image needed for Scrublet automatic thresholds).

Doublets must be detected on all cells of each sample (simulating mixtures of CAFs with epithelial/immune cells), so this runs on global_full_raw.
Sample unit = patient × condition (library IDs were not retained in the raw data; cross-library differences within a patient are slightly underestimated).

Output: results/fig1/qc/
  cell_qc.tsv.gz        per cell: sample, doublet_score, predicted_doublet, n_genes, n_umi, pct_mt
  doublet_summary.tsv   doublet fraction per dataset / lineage
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fig1.yaml").read_text())
ROOT = Path(CFG["paths"]["out_dir"])
OUT = ROOT / "qc"
OUT.mkdir(parents=True, exist_ok=True)
MIN_CELLS_FOR_SCRUBLET = 200


def main():
    a = sc.read_h5ad(ROOT / "global_full_raw.h5ad")
    a.X = sp.csr_matrix(a.X)
    a.obs["sample"] = (a.obs.patient_key.astype(str) + "|" + a.obs.condition.astype(str)).values
    v = a.var_names.astype(str)
    tot = np.asarray(a.X.sum(1)).ravel()
    a.obs["n_umi"] = tot
    a.obs["n_genes"] = np.asarray((a.X > 0).sum(1)).ravel()
    a.obs["pct_mt"] = np.asarray(a.X[:, np.where(v.str.startswith("MT-"))[0]].sum(1)).ravel() / np.maximum(tot, 1) * 100

    a.obs["doublet_score"] = np.nan
    a.obs["predicted_doublet"] = False
    sizes = a.obs["sample"].value_counts()
    for i, (s, n) in enumerate(sizes.items(), 1):
        if n < MIN_CELLS_FOR_SCRUBLET:
            continue
        sub = a[a.obs["sample"] == s].copy()
        try:
            sc.pp.scrublet(sub, expected_doublet_rate=min(0.008 * n / 1000, 0.1), random_state=0, verbose=False)
        except Exception as e:  # tiny / degenerate samples
            print(f"  scrublet failed for {s} (n={n}): {e}")
            continue
        a.obs.loc[sub.obs_names, "doublet_score"] = sub.obs["doublet_score"].values
        a.obs.loc[sub.obs_names, "predicted_doublet"] = sub.obs["predicted_doublet"].fillna(False).values
        if i % 20 == 0:
            print(f"  {i}/{len(sizes)} samples", flush=True)

    qc = a.obs[["dataset", "patient_key", "condition", "sample", "author_label", "doublet_score",
                "predicted_doublet", "n_genes", "n_umi", "pct_mt"]]
    qc.to_csv(OUT / "cell_qc.tsv.gz", sep="\t")
    ann = pd.read_csv(ROOT / "annotate_global" / "global_full_annotation.tsv.gz", sep="\t", index_col=0)
    qc = qc.join(ann[["lineage"]])
    summ = qc.groupby(["dataset", "lineage"]).agg(n=("predicted_doublet", "size"),
                                                  doublet_rate=("predicted_doublet", "mean"),
                                                  median_score=("doublet_score", "median")).round(3)
    summ.to_csv(OUT / "doublet_summary.tsv", sep="\t")
    print(qc.groupby("dataset").predicted_doublet.mean().round(3).to_string())
    print(summ.xs("Fibroblast", level="lineage").to_string())


if __name__ == "__main__":
    main()
