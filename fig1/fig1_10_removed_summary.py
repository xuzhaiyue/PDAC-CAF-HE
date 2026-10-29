"""Fig 1 补充表 — 从间质细胞到最终 CAF 图谱，每一步去掉了哪些细胞、为什么（hemorph 环境）。

Steps
  1. 细胞质控（全部间质细胞）：qc/removed_cells.tsv.gz（低基因数 / 低 UMI / 高线粒体 / 双联体）
  2. 抽样：每患者×条件 ≤ max_cells_per_patient（未入选的细胞不计为"去除"）
  3. 亚群排除：各轮整合目录中的 cluster_flags.tsv（规则见 config.cluster_exclusion）

Outputs: results/fig1/qc/
  removal_by_step.tsv     每一步 × 数据集 × 原因 的细胞数
  removed_cells_all.tsv.gz  每个被去除细胞的去除步骤和原因
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
