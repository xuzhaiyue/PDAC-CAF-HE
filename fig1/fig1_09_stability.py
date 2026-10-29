"""Fig 1 step 09 — 最终 CAF 图谱的稳定性检查（pdac_scvi 环境）。

1. 聚类随机种子：leiden_0.5 换 5 个种子，整体 ARI + CTHRC1+ 亚群的 Jaccard（与每次结果中重合最多的亚群比较）
2. 聚类分辨率：0.3 / 0.5 / 0.8 / 1.2 下，CTHRC1+ 亚群的最佳匹配 Jaccard，以及它被拆到几个亚群里
3. 整合模型：scVI 用 2 个不同种子重新训练 → 重新建图聚类 → CTHRC1+ 亚群 Jaccard

Output: results/fig1/integrate_caf_final/stability.tsv
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import scvi  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)
CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fig1.yaml").read_text())
D = Path(CFG["paths"]["out_dir"]) / "integrate_caf_final"
TARGET = {**CFG["signature"], **CFG.get("signature_v3", {})}["target_clusters"][0]


def best_match(ref_mask: np.ndarray, labels: pd.Series) -> tuple[float, str, int]:
    """Jaccard of the reference cell set with the best-overlapping cluster; and #clusters holding >=10% of it."""
    best, best_c = 0.0, ""
    for c in labels.unique():
        m = (labels == c).to_numpy()
        j = (ref_mask & m).sum() / (ref_mask | m).sum()
        if j > best:
            best, best_c = j, c
    share = labels[ref_mask].value_counts(normalize=True)
    return round(best, 3), best_c, int((share >= 0.10).sum())


def main():
    a = sc.read_h5ad(D / "fibro_integrated.h5ad")
    main_lab = a.obs["leiden_0.5"].astype(str)
    ref = (main_lab == TARGET).to_numpy()
    rows = []
    for seed in range(1, 6):
        lab = sc.tl.leiden(a, resolution=0.5, flavor="igraph", n_iterations=2, random_state=seed,
                           key_added="_s", copy=True).obs["_s"].astype(str)
        j, c, k = best_match(ref, lab)
        rows.append({"test": f"leiden seed {seed}", "ARI_vs_main": round(adjusted_rand_score(main_lab, lab), 3),
                     "CTHRC1_cluster_jaccard": j, "split_into": k})
    for r in [0.3, 0.5, 0.8, 1.2]:
        lab = a.obs[f"leiden_{r:g}"].astype(str)
        j, c, k = best_match(ref, lab)
        rows.append({"test": f"resolution {r:g}", "ARI_vs_main": round(adjusted_rand_score(main_lab, lab), 3),
                     "CTHRC1_cluster_jaccard": j, "split_into": k})

    hv = a[:, a.var.highly_variable].copy()
    hv.X = hv.layers["counts"].copy()
    for seed in [101, 202]:
        scvi.settings.seed = seed
        scvi.model.SCVI.setup_anndata(hv, layer="counts", batch_key="patient_key",
                                      categorical_covariate_keys=["dataset"])
        m = scvi.model.SCVI(hv, n_latent=20, n_layers=2, gene_likelihood="nb")
        m.train(max_epochs=300, early_stopping=True, accelerator="cpu", batch_size=512)
        b = sc.AnnData(obs=a.obs[["patient_key"]].copy())
        b.obsm["X_scVI"] = m.get_latent_representation()
        sc.pp.neighbors(b, use_rep="X_scVI", n_neighbors=15, random_state=0)
        lab = sc.tl.leiden(b, resolution=0.5, flavor="igraph", n_iterations=2, random_state=0,
                           key_added="_m", copy=True).obs["_m"].astype(str)
        j, c, k = best_match(ref, lab)
        rows.append({"test": f"scVI retrained seed {seed}", "ARI_vs_main": round(adjusted_rand_score(main_lab, lab), 3),
                     "CTHRC1_cluster_jaccard": j, "split_into": k})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(D / "stability.tsv", sep="\t", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
