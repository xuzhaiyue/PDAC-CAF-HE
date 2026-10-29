"""External single-cell validation (pdac_scvi environment). Rules in the external_validation section of config/fig1.yaml (fixed before running).

Each external dataset is processed independently:
  1. Read counts (subsampled to <= cap per sample) → within-dataset MAD QC
  2. Global scVI (batch = patient) + Leiden 0.5 → automatic annotation with lineage_markers (top lineage with lead >= margin)
  3. Fibroblast clusters → second scVI + Leiden 0.5 → one round of cleaning with the same cluster-exclusion rules as discovery
  4. External CAF clusters matched to the 8 discovery clusters by centroid correlation (programme and pan-CAF genes excluded to avoid circularity)
  5. Metrics: presence of a cluster matched to CTHRC1+; programme AUROC (compared with CTHRC1 alone, Moffitt activated stroma and
     Dominguez LRRC15+); CTHRC1+ fraction per sample; patient-paired log2FC of programme genes
Usage: python fig1_15_external_validation.py run GSE253429   |   python fig1_15_external_validation.py summary
Output: results/fig1_external/<dataset>/, results/fig1_external/summary.tsv
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
import glob  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scvi  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((PROJ / "config" / "fig1.yaml").read_text())
EV = CFG["external_validation"]
ROOT = Path(CFG["paths"]["out_dir"])
OUT = PROJ / "results" / "fig1_external"
PROG = yaml.safe_load((ROOT / "signature" / "programs.yaml").read_text())
PUB = yaml.safe_load((PROJ / "config" / "published_signatures.yaml").read_text())
CAF_NAMES = {"2": "CTHRC1+ matrix myCAF", "1": "HOPX+ F2R+ myCAF", "5": "Glycolytic CAF",
             "0": "C7+ C3+ iCAF", "4": "MFAP5+ IGF1+ iCAF", "3": "PI16+ universal",
             "6": "SPP1+ IBSP+ CAF", "7": "Chemo-associated CAF"}
TARGET = "CTHRC1+ matrix myCAF"
COMPARE = {"program": PROG["CTHRC1_matrix_CAF"], "CTHRC1_alone": ["CTHRC1"],
           "Moffitt_activated": PUB["Moffitt_activated_stroma"]["genes"],
           "Dominguez_LRRC15": PUB["Dominguez_LRRC15_CAF"]["genes"]}
RNG = np.random.default_rng(CFG["random_seed"])
scvi.settings.seed = 0


def mad(x):
    return np.median(np.abs(x - np.median(x))) * 1.4826


# ---------------------------------------------------------------- 1. load + QC
def load(ds: str) -> ad.AnnData:
    spec = EV["datasets"][ds]
    files = sorted(glob.glob(str(Path(EV["bundle"]) / spec["files"])))
    parts = []
    for f in files:
        a = ad.read_h5ad(f)
        if "keep_tissue" in spec and "tissue_type" in a.obs:
            a = a[a.obs.tissue_type.astype(str) == spec["keep_tissue"]]
        X = a.layers["counts"] if "counts" in a.layers else a.X
        b = ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame(index=a.obs_names.astype(str)), var=pd.DataFrame(index=a.var_names.astype(str)))
        b.obs["patient_key"] = a.obs["patient_key"].astype(str).values
        b.obs["condition"] = a.obs["condition"].astype(str).values
        b.obs["author_label"] = (a.obs["ann_L2"].astype(str).values if "ann_L2" in a.obs else "NA")
        b.var_names_make_unique()
        parts.append(b)
    a = ad.concat(parts, join="inner", merge=None) if len(parts) > 1 else parts[0]
    a.obs_names_make_unique()
    # per-sample cap
    keep = np.zeros(a.n_obs, bool)
    for _, idx in a.obs.groupby("patient_key", observed=True).indices.items():
        keep[idx if len(idx) <= EV["cap_per_sample"] else RNG.choice(idx, EV["cap_per_sample"], replace=False)] = True
    a = a[keep].copy()
    # QC within dataset
    q, mod = EV["qc"], spec["modality"]
    C = a.X
    n_genes = np.asarray((C > 0).sum(1)).ravel()
    n_umi = np.asarray(C.sum(1)).ravel()
    mt = a.var_names.str.upper().str.startswith("MT-")
    pct_mt = np.asarray(C[:, mt].sum(1)).ravel() / np.maximum(n_umi, 1) * 100
    lg, lu = np.log(np.maximum(n_genes, 1)), np.log(np.maximum(n_umi, 1))
    g_min = max(q["min_genes"][mod], float(np.exp(np.median(lg) - q["mad_k"] * mad(lg))))
    u_min = float(np.exp(np.median(lu) - q["mad_k"] * mad(lu)))
    mt_max = max(q["max_mt_floor"][mod], float(np.median(pct_mt) + q["mad_k"] * mad(pct_mt)))
    ok = (n_genes >= g_min) & (n_umi >= u_min) & (pct_mt <= mt_max)
    print(f"{ds}: {a.n_obs} cells after cap; QC genes>={g_min:.0f}, umi>={u_min:.0f}, mt<={mt_max:.1f}% "
          f"-> keep {ok.sum()}", flush=True)
    a = a[ok].copy()
    a.obs["n_genes"] = n_genes[ok]
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def integrate(a: ad.AnnData, max_epochs: int) -> ad.AnnData:
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=3000, layer="counts", subset=False)
    hv = a[:, a.var.highly_variable].copy()
    scvi.model.SCVI.setup_anndata(hv, layer="counts", batch_key="patient_key")
    m = scvi.model.SCVI(hv, n_latent=20, n_layers=2, gene_likelihood="nb")
    m.train(max_epochs=max_epochs, early_stopping=True, accelerator="cpu", batch_size=512, enable_progress_bar=False)
    a.obsm["X_scVI"] = m.get_latent_representation()
    sc.pp.neighbors(a, use_rep="X_scVI", n_neighbors=15, random_state=0)
    sc.tl.leiden(a, resolution=EV["caf_resolution"], key_added="leiden", flavor="igraph", n_iterations=2,
                 random_state=0)
    sc.tl.umap(a, random_state=0)
    return a


def gscore(a, genes, name):
    sc.tl.score_genes(a, [g for g in genes if g in a.var_names], score_name=name, random_state=0)


# ---------------------------------------------------------------- 2. global lineage
def annotate_lineage(a: ad.AnnData, out: Path) -> pd.DataFrame:
    for k, g in CFG["lineage_markers"].items():
        gscore(a, g, f"lin_{k}")
    cols = [f"lin_{k}" for k in CFG["lineage_markers"] if k != "Cycling"]
    m = a.obs.groupby("leiden", observed=True)[cols].mean()
    rows = []
    for cl, r in m.iterrows():
        s = r.sort_values(ascending=False)
        top = s.index[0][4:]
        rows.append({"cluster": str(cl), "top": top, "top_score": s.iloc[0], "second": s.index[1][4:],
                     "margin": s.iloc[0] - s.iloc[1],
                     "label": top if s.iloc[0] - s.iloc[1] >= EV["lineage_min_margin"] else "Ambiguous",
                     "n_cells": int((a.obs.leiden == cl).sum())})
    tab = pd.DataFrame(rows).set_index("cluster")
    tab.to_csv(out / "global_cluster_annotation.tsv", sep="\t")
    a.obs["lineage"] = a.obs.leiden.astype(str).map(tab.label)
    print(tab.round(2).to_string(), flush=True)
    return tab


# ---------------------------------------------------------------- 3. CAF clustering + exclusion
def flag_clusters(c: ad.AnnData) -> pd.DataFrame:
    X = CFG["cluster_exclusion"]
    C = sp.csr_matrix(c.layers["counts"])
    v = list(c.var_names)
    pos = {g: (C[:, v.index(g)].toarray().ravel() > 0) for g in ["RGS5", "NOTCH3", "MYH11"] if g in v}
    for k, g in CFG["contaminant_markers"].items():
        if k in ("epithelial", "immune"):
            gscore(c, g, f"contam_{k}")
    df = pd.DataFrame({"cl": c.obs.leiden.astype(str).values, "n_genes": c.obs.n_genes.values, **pos,
                       "epi": c.obs.contam_epithelial.values, "imm": c.obs.contam_immune.values})
    g = df.groupby("cl")
    f = pd.DataFrame({"n_cells": g.size(), "pericyte_smc_mean_frac": g[list(pos)].mean().mean(1),
                      "median_genes_ratio": g.n_genes.median() / df.n_genes.median(),
                      "epithelial_score": g.epi.mean(), "immune_score": g.imm.mean()})
    f["exclude"] = ((f.pericyte_smc_mean_frac >= X["pericyte_smc_mean_frac"]) |
                    (f.median_genes_ratio < X["low_complexity_ratio"]) |
                    (f.epithelial_score > X["epithelial_score"]) | (f.immune_score > X["immune_score"]))
    return f


# ---------------------------------------------------------------- 4. mapping to discovery subtypes
def discovery_centroids():
    d = ad.read_h5ad(ROOT / "integrate_caf_final" / "fibro_integrated.h5ad")
    mk = pd.read_csv(ROOT / "integrate_caf_final" / "markers_0.5.tsv", sep="\t")
    mk["group"] = mk.group.astype(str)
    excl = set(PROG["CTHRC1_matrix_CAF"]) | set(PROG["pan_CAF"])
    genes = sorted({g for _, s in mk.groupby("group") for g in s.names.head(EV["mapping"]["markers_per_subtype"])} - excl)
    return d, genes


def centroids(a, key, genes):
    C = sp.csr_matrix(a.layers["counts"])
    lab = a.obs[key].astype(str).to_numpy()
    rows = {}
    for g in sorted(set(lab)):
        s = np.asarray(C[lab == g].sum(0)).ravel()
        rows[g] = np.log2(s / s.sum() * 1e6 + 1)
    M = pd.DataFrame(rows, index=a.var_names).T
    return M.reindex(columns=genes)


def map_clusters(c, out):
    d, genes = discovery_centroids()
    genes = [g for g in genes if g in c.var_names]
    D = centroids(d, "leiden_0.5", genes).rename(index=CAF_NAMES)
    E = centroids(c, "caf_cluster", genes)
    z = lambda M: (M - M.mean()) / M.std().replace(0, np.nan)  # noqa: E731
    Dz, Ez = z(D).dropna(axis=1), z(E).dropna(axis=1)
    common = Dz.columns.intersection(Ez.columns)
    R = pd.DataFrame(np.corrcoef(Ez[common].to_numpy(), Dz[common].to_numpy())[:len(Ez), len(Ez):],
                     index=Ez.index, columns=Dz.index)
    best = R.idxmax(1)
    mapped = best.where(R.max(1) >= EV["mapping"]["min_r"], "unmapped")
    R.assign(best=best, best_r=R.max(1), mapped=mapped).round(3).to_csv(out / "cluster_mapping.tsv", sep="\t")
    print(f"mapping on {len(common)} genes (program / pan-CAF genes excluded)")
    print(R.round(2).assign(mapped=mapped).to_string(), flush=True)
    return mapped


# ---------------------------------------------------------------- 5. metrics
def auroc(score, y):
    pos, neg = score[y], score[~y]
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def paired_lfc(c, y, genes):
    C = sp.csr_matrix(c.layers["counts"])
    rows = []
    for p, idx in c.obs.groupby("patient_key", observed=True).indices.items():
        t, r = idx[y[idx]], idx[~y[idx]]
        if len(t) < 10 or len(r) < 10:
            continue
        cpm = lambda ii: np.log2(np.asarray(C[ii].sum(0)).ravel() / max(C[ii].sum(), 1) * 1e6 + 1)  # noqa: E731
        lt, lr = cpm(t), cpm(r)
        rows.append(pd.Series({g: lt[c.var_names.get_loc(g)] - lr[c.var_names.get_loc(g)]
                               for g in genes if g in c.var_names}, name=p))
    return pd.DataFrame(rows)


def run(ds: str):
    out = OUT / ds
    out.mkdir(parents=True, exist_ok=True)
    a = load(ds)
    a = integrate(a, max_epochs=80)
    annotate_lineage(a, out)
    c = a[a.obs.lineage == "Fibroblast"].copy()
    print(f"fibroblast-lineage cells: {c.n_obs} from {c.obs.patient_key.nunique()} samples", flush=True)
    c = integrate(c, max_epochs=200)
    flags = flag_clusters(c)
    flags.round(3).to_csv(out / "caf_cluster_flags.tsv", sep="\t")
    print(flags.round(2).to_string(), flush=True)
    c = c[~c.obs.leiden.astype(str).isin(flags.index[flags.exclude])].copy()
    c.obs["caf_cluster"] = "E" + c.obs.leiden.astype(str)
    mapped = map_clusters(c, out)
    c.obs["mapped"] = c.obs.caf_cluster.map(mapped)
    y = (c.obs.mapped == TARGET).to_numpy()
    for k, g in COMPARE.items():
        idx = [c.var_names.get_loc(x) for x in g if x in c.var_names]
        c.obs[f"score_{k}"] = np.asarray(c.X[:, idx].mean(1)).ravel()     # same score as the LODO analysis
    res = {"dataset": ds, "modality": EV["datasets"][ds]["modality"], "n_cafs": int(c.n_obs),
           "n_samples": int(c.obs.patient_key.nunique()), "n_mapped_cthrc1": int(y.sum()),
           "mapped_clusters": ",".join(sorted(mapped.index[mapped == TARGET]))}
    if y.sum() > 0 and (~y).sum() > 0:
        for k in COMPARE:
            res[f"auroc_{k}"] = auroc(c.obs[f"score_{k}"].to_numpy(), y)
        lfc = paired_lfc(c, y, PROG["CTHRC1_matrix_CAF"])
        lfc.to_csv(out / "program_paired_log2fc.tsv", sep="\t")
        med = lfc.median()
        res.update({"n_paired_patients": len(lfc), "genes_up": int((med > 0).sum()),
                    "genes_up_list": ",".join(med.index[med > 0])})
    # per-sample presence
    st = c.obs.groupby("patient_key", observed=True).agg(n=("mapped", "size"),
                                                          frac=("mapped", lambda s: (s == TARGET).mean()),
                                                          condition=("condition", "first"))
    st.to_csv(out / "per_sample_cthrc1_fraction.tsv", sep="\t")
    st = st[st.n >= EV["presence_min_cafs"]]
    res.update({"samples_evaluable": len(st), "samples_present": int((st.frac >= EV["presence_frac"]).sum()),
                "median_frac": float(st.frac.median()) if len(st) else np.nan})
    s = EV["success"]
    res["validated"] = bool(y.sum() >= EV["min_mapped_cells_for_test"] and
                            res.get("auroc_program", 0) >= s["min_auroc"] and res.get("genes_up", 0) >= s["min_genes_up"])
    res["descriptive_only"] = bool(y.sum() < EV["min_mapped_cells_for_test"])
    # author labels as secondary check
    pd.crosstab(c.obs.caf_cluster.astype(str) + " → " + c.obs.mapped, c.obs.author_label).to_csv(
        out / "author_label_crosstab.tsv", sep="\t")
    sc.tl.rank_genes_groups(c, "caf_cluster", method="wilcoxon", n_genes=30)
    sc.get.rank_genes_groups_df(c, None).to_csv(out / "caf_cluster_markers.tsv", sep="\t", index=False)
    c.obs.to_csv(out / "caf_obs.tsv.gz", sep="\t")
    pd.DataFrame(c.obsm["X_umap"], index=c.obs_names, columns=["UMAP1", "UMAP2"]).to_csv(out / "caf_umap.tsv.gz", sep="\t")
    pd.Series(res).to_csv(out / "result.tsv", sep="\t", header=False)
    print(pd.Series(res).to_string(), flush=True)


def summary():
    rows = [pd.read_csv(f, sep="\t", header=None, index_col=0)[1] for f in sorted(OUT.glob("*/result.tsv"))]
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.tsv", sep="\t", index=False)
    print(s.T.to_string())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        summary()
