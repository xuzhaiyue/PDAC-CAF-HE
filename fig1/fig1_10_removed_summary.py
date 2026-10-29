"""Fig 1 supplementary table — which cells were removed at each step from mesenchymal cells to the final CAF atlas, and why (hemorph environment).

Steps
  1. Cell QC (all mesenchymal cells): qc/removed_cells.tsv.gz (low gene count / low UMI / high mitochondrial / doublet)
  2. Subsampling: <= max_cells_per_patient per patient × condition (cells not selected are not counted as removed)
  3. Cluster exclusion: cluster_flags.tsv in each integration round (rules in config.cluster_exclusion)

Outputs: results/fig1/qc/
  removal_by_step.tsv     cell counts per step × dataset × reason
  removed_cells_all.tsv.gz  removal step and reason for every removed cell
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fig1.yaml").read_text())
ROOT = Path(CFG["paths"]["out_dir"])
ROUNDS = ["integrate_mes_clean", "integrate_caf_clean_r1", "integrate_caf_clean_r2"]


def main():
    rows = []
    qc = pd.read_csv(ROOT / "qc" / "removed_cells.tsv.gz", sep="\t", index_col=0)
    qc["step"] = "1_cell_qc"
    qc["reason"] = qc.reason.str.rstrip(";")
    rows.append(qc[["dataset", "patient_key", "condition", "step", "reason"]])

    for prev, nxt in zip(ROUNDS[:-1], ROUNDS[1:]):
        a = ad.read_h5ad(ROOT / prev / "fibro_integrated.h5ad", backed="r")
        b = ad.read_h5ad(ROOT / nxt / "fibro_integrated.h5ad", backed="r")
        flags = pd.read_csv(ROOT / prev / "cluster_flags.tsv", sep="\t", index_col=0)
        flags.index = flags.index.astype(str)
        gone = a.obs.loc[~a.obs_names.isin(b.obs_names)].copy()
        gone["cluster"] = gone["leiden_0.5"].astype(str)
        gone["step"] = f"3_cluster_exclusion_{prev}"
        gone["reason"] = gone.cluster.map(flags.exclude_reason).fillna("") + " (cluster " + gone.cluster + ")"
        rows.append(gone[["dataset", "patient_key", "condition", "step", "reason"]])

    allr = pd.concat(rows)
    allr.to_csv(ROOT / "qc" / "removed_cells_all.tsv.gz", sep="\t")
    summ = allr.assign(reason_short=allr.reason.str.replace(r" \(cluster \d+\)", "", regex=True)) \
               .groupby(["step", "reason_short", "dataset"]).size().unstack(fill_value=0)
    summ["total"] = summ.sum(1)
    summ.to_csv(ROOT / "qc" / "removal_by_step.tsv", sep="\t")
    print(summ.to_string())
    final = ad.read_h5ad(ROOT / ROUNDS[-1] / "fibro_integrated.h5ad", backed="r")
    print(f"\nfinal CAF atlas: {final.n_obs} cells, {final.obs.patient_key.nunique()} patients; "
          f"by dataset {final.obs.dataset.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
