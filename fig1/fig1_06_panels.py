"""Fig 1 正式面板（pdac_scvi 环境）。每个面板单独导出 PDF（文字可编辑）+ PNG（300 dpi）。

b  global_umap         全局图谱：成纤维细胞 / 周细胞-平滑肌上色，其余谱系浅灰，谱系名直接标注
c  caf_umap            CAF 细分：≤3 主色（散点图所有颜色两两可分的上限）+ 灰，亚群名直接标注
   caf_umap_facets     补充：每个亚群单独一格
d  score_lineage       CTHRC1+ 程序得分（患者水平均值）在全部谱系中的分布
e  score_caf_subtype   CTHRC1+ 程序得分在各 CAF 亚群中的分布（按数据集着形状）
f  contribution        PDAC 组织中各基因转录本来自成纤维细胞的比例（丰度加权）
g  lodo                留一数据集验证：程序 vs CTHRC1 单基因的 AUROC

配色来自 dataviz 参考调色板（validate_palette.js 校验：前 3 槽在散点图全两两组合下通过）。
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import yaml  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "fig1.yaml").read_text())
ROOT = Path(CFG["paths"]["out_dir"])
OUT = ROOT / "figure_panels"
OUT.mkdir(parents=True, exist_ok=True)
PROG = yaml.safe_load((ROOT / "signature" / "programs.yaml").read_text())

# ---- palette (reference instance, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY_LIGHT, GRAY_MID, GRAY_DARK = "#d3d1c7", "#b4b2a9", "#888780"
INK, INK2 = "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DS_MARKERS = {"CRA001160": "o", "GSE205013": "s", "GSE278688": "^"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
})
OUTLINE = [patheffects.withStroke(linewidth=1.6, foreground="white")]

CAF_LABEL_OFFSETS: dict = {}       # 看图后按需填写，避免标签重叠
CAF_ATLAS = "integrate_caf_final"   # 清理后（细胞质控 + 双联体 + 规则化亚群排除）的 CAF 图谱
CAF_NAMES = {"2": "CTHRC1+ matrix myCAF", "1": "HOPX+ F2R+ myCAF", "5": "Glycolytic CAF",
             "0": "C7+ C3+ iCAF", "4": "MFAP5+ IGF1+ iCAF", "3": "PI16+ universal",
             "6": "SPP1+ IBSP+ CAF (8 patients)", "7": "Chemo-associated CAF"}
CAF_COLOR = {"CTHRC1+ matrix myCAF": BLUE, "HOPX+ F2R+ myCAF": ORANGE, "Glycolytic CAF": ORANGE,
             "C7+ C3+ iCAF": AQUA, "MFAP5+ IGF1+ iCAF": AQUA, "PI16+ universal": GRAY_DARK,
             "SPP1+ IBSP+ CAF (8 patients)": GRAY_LIGHT, "Chemo-associated CAF": GRAY_LIGHT}
LINEAGE_COLOR = {"Fibroblast": BLUE, "Pericyte_SMC": AQUA}
LINEAGE_LABEL = {"Ductal_tumor": "Ductal (tumour)", "Ductal_normal": "Ductal (normal)", "T_NK": "T/NK",
                 "Pericyte_SMC": "Pericyte/SMC", "Neutrophil_RBC": "Neutrophil/RBC", "Fibroblast": "Fibroblast"}
DROP_LINEAGES = ["Ambiguous", "Low_quality", "Hepatocyte_like"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def label_centroids(ax, xy, labels, min_n=150, fontsize=6.5, skip=(), offsets=None):
    """Direct labels at cluster medians; `offsets` (label -> (dx, dy) as fraction of the data range)
    moves crowded labels off the points with a thin leader line."""
    offsets = offsets or {}
    span = xy.max(0) - xy.min(0)
    for lab in pd.unique(labels):
        m = labels == lab
        if m.sum() < min_n or lab in skip:
            continue
        cx, cy = np.median(xy[m], axis=0)
        text = LINEAGE_LABEL.get(lab, lab.replace("_", " "))
        if lab in offsets:
            dx, dy = offsets[lab]
            tx, ty = cx + dx * span[0], cy + dy * span[1]
            ax.annotate(text, (cx, cy), (tx, ty), fontsize=fontsize, ha="center", va="center", color=INK,
                        path_effects=OUTLINE, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5,
                                                              shrinkA=2, shrinkB=0))
        else:
            ax.text(cx, cy, text, fontsize=fontsize, ha="center", va="center", color=INK, path_effects=OUTLINE)


def umap_axes(ax, title, axis_label=True):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", color=INK)
    if axis_label:
        ax.text(0.0, 0.0, "UMAP", transform=ax.transAxes, fontsize=6, color=INK2, va="top")


def program_score(a, genes, name):
    sc.tl.score_genes(a, [g for g in genes if g in a.var_names], score_name=name, random_state=0)


def patient_means(obs, score, group):
    return (obs.groupby([group, "dataset", "patient_key"], observed=True)[score]
            .agg(["mean", "size"]).reset_index().query("size >= 20"))


# ------------------------------------------------------------------ b. global atlas
def panel_global():
    g = sc.read_h5ad(ROOT / "integrate_global" / "fibro_integrated.h5ad")
    ann = pd.read_csv(ROOT / "annotate_global" / "cluster_annotation.tsv", sep="\t", index_col=0)
    ann.index = ann.index.astype(str)
    g.obs["lineage"] = g.obs["leiden_0.5"].astype(str).map(ann.final_label)
    g = g[~g.obs.lineage.isin(DROP_LINEAGES)].copy()
    xy = g.obsm["X_umap"]
    lin = g.obs.lineage.astype(str).to_numpy()
    col = np.array([LINEAGE_COLOR.get(l, GRAY_LIGHT) for l in lin])
    order = np.argsort([l in LINEAGE_COLOR for l in lin], kind="stable")  # coloured on top
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.scatter(xy[order, 0], xy[order, 1], c=col[order], s=0.15, linewidths=0, rasterized=True)
    label_centroids(ax, xy, lin, min_n=300, offsets={
        "Fibroblast": (0.02, 0.30), "Pericyte_SMC": (-0.16, -0.20), "Endocrine": (0.10, 0.22),
        "Schwann": (0.02, -0.17)})
    umap_axes(ax, f"Integrated PDAC atlas ({g.n_obs:,} cells, {g.obs.patient_key.nunique()} patients, 3 datasets)")
    save(fig, "b_global_umap")
    comp = pd.crosstab([g.obs.dataset, g.obs.condition], g.obs.lineage)
    comp.to_csv(OUT / "b_global_lineage_counts.tsv", sep="\t")
    program_score(g, PROG["CTHRC1_matrix_CAF"], "cthrc1_prog")
    return g


# ------------------------------------------------------------------ c. CAF subtypes
def panel_caf():
    a = sc.read_h5ad(ROOT / CAF_ATLAS / "fibro_integrated.h5ad")
    a.obs["subtype"] = a.obs["leiden_0.5"].astype(str).map(CAF_NAMES)
    xy = a.obsm["X_umap"]
    st = a.obs.subtype.to_numpy()
    col = np.array([CAF_COLOR[s] for s in st])
    rank = {BLUE: 3, ORANGE: 2, AQUA: 2, GRAY_DARK: 1, GRAY_LIGHT: 1}
    order = np.argsort([rank[c] for c in col], kind="stable")
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.scatter(xy[order, 0], xy[order, 1], c=col[order], s=1.2, linewidths=0, rasterized=True)
    label_centroids(ax, xy, st, min_n=90, fontsize=6, offsets=CAF_LABEL_OFFSETS)
    umap_axes(ax, f"CAF subtypes ({a.n_obs:,} CAFs, {a.obs.patient_key.nunique()} patients)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=4, label=l) for c, l in
               [(BLUE, "CTHRC1+ matrix myCAF"), (ORANGE, "Other myCAF"), (AQUA, "iCAF"),
                (GRAY_DARK, "Universal fibroblast"), (GRAY_LIGHT, "Small / skewed subsets")]]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6, handletextpad=0.2,
              borderaxespad=0)
    save(fig, "c_caf_umap")

    # supplementary facets: each subtype in its own panel (exact identity without relying on colour)
    names = list(CAF_NAMES.values())
    fig, axs = plt.subplots(3, 3, figsize=(6.8, 6.6))
    axs.ravel()[-1].axis("off")
    for ax, n in zip(axs.ravel(), names):
        m = st == n
        ax.scatter(xy[~m, 0], xy[~m, 1], c=GRAY_LIGHT, s=0.6, linewidths=0, rasterized=True)
        ax.scatter(xy[m, 0], xy[m, 1], c=BLUE, s=0.8, linewidths=0, rasterized=True)
        umap_axes(ax, f"{n.split(' (')[0]}\n(n={m.sum():,}, {a.obs.patient_key[m].nunique()} patients)",
                  axis_label=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.1)
    save(fig, "c_caf_umap_facets_supp")

    program_score(a, PROG["CTHRC1_matrix_CAF"], "cthrc1_prog")
    return a


# ------------------------------------------------------------------ d/e. program score
def panel_scores(g, a):
    pm = patient_means(g.obs, "cthrc1_prog", "lineage")
    order = pm.groupby("lineage")["mean"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(2.6, 3.2))
    for i, l in enumerate(order):
        v = pm[pm.lineage == l]
        c = LINEAGE_COLOR.get(l, GRAY_MID)
        ax.boxplot(v["mean"], positions=[i], vert=False, widths=0.6, showfliers=False,
                   medianprops=dict(color=INK, lw=0.8), boxprops=dict(color=c, lw=0.8),
                   whiskerprops=dict(color=c, lw=0.6), capprops=dict(color=c, lw=0.6))
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
        ax.scatter(v["mean"], i + jit, s=4, c=c, linewidths=0, alpha=0.8)
    ax.set_yticks(range(len(order)), [LINEAGE_LABEL.get(l, l.replace("_", " ")) for l in order])
    ax.set_xlabel("CTHRC1+ program score\n(patient mean, ≥20 cells)")
    ax.set_title("Program is fibroblast-restricted", loc="left")
    save(fig, "d_score_lineage")
    pm.to_csv(OUT / "d_score_lineage_patient_means.tsv", sep="\t", index=False)

    a.obs["subtype"] = a.obs["leiden_0.5"].astype(str).map(CAF_NAMES)
    pm2 = patient_means(a.obs, "cthrc1_prog", "subtype")
    order2 = [n for n in CAF_NAMES.values() if n in set(pm2.subtype)]
    order2 = sorted(order2, key=lambda n: pm2[pm2.subtype == n]["mean"].median())
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    for i, n in enumerate(order2):
        v = pm2[pm2.subtype == n]
        c = CAF_COLOR[n] if CAF_COLOR[n] != GRAY_LIGHT else GRAY_MID
        ax.boxplot(v["mean"], positions=[i], vert=False, widths=0.6, showfliers=False,
                   medianprops=dict(color=INK, lw=0.8), boxprops=dict(color=c, lw=0.8),
                   whiskerprops=dict(color=c, lw=0.6), capprops=dict(color=c, lw=0.6))
        for ds, mk in DS_MARKERS.items():
            w = v[v.dataset == ds]
            jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(w))
            ax.scatter(w["mean"], i + jit, s=6, marker=mk, facecolors="none", edgecolors=c, linewidths=0.5)
    ax.set_yticks(range(len(order2)), order2)
    ax.set_xlabel("CTHRC1+ program score (patient mean)")
    ax.set_title("Highest in CTHRC1+ matrix myCAF", loc="left")
    handles = [plt.Line2D([], [], marker=m, ls="", mfc="none", mec=INK2, ms=4, label=d) for d, m in DS_MARKERS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=6, handletextpad=0.2)
    save(fig, "e_score_caf_subtype")
    pm2.to_csv(OUT / "e_score_subtype_patient_means.tsv", sep="\t", index=False)


# ------------------------------------------------------------------ f. fibroblast contribution
def panel_contribution():
    """Horizontal heatmap (genes on x, datasets on y) so it can share a row with panel g."""
    c = pd.read_csv(ROOT / "signature" / "fibroblast_contribution.tsv", sep="\t")
    # 与已发表基因集的比较见 fig1_14_published_signatures.py（StRoMA 不作为生物学参照，已移除）
    groups = [("CTHRC1+ program", PROG["CTHRC1_matrix_CAF"]), ("Pan-CAF ref.", PROG["pan_CAF"])]
    genes = [g for _, gs in groups for g in gs]
    m = c.pivot(index="gene", columns="dataset", values="fib_contribution").reindex(genes).T
    fig, ax = plt.subplots(figsize=(3.8, 2.0))
    im = ax.imshow(m.to_numpy(), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=5.2,
                    color="white" if v > 0.6 else INK)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=6)
    x = 0
    for name, gs in groups:
        ax.text(x + (len(gs) - 1) / 2, -0.6, name, fontsize=6, color=INK2, ha="center", va="bottom")
        x += len(gs)
        if x < len(genes):
            ax.axvline(x - 0.5, color="white", lw=2)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Share of transcripts from fibroblasts\n(PDAC tissue, abundance-weighted)",
                 fontsize=5.5)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    ax.set_title("Fibroblast origin of program genes", loc="left", pad=16)
    save(fig, "f_fibroblast_contribution")


# ------------------------------------------------------------------ g. leave-one-dataset-out
def panel_lodo():
    d = pd.read_csv(ROOT / "signature" / "lodo.tsv", sep="\t")
    d = d[~d.held_out.str.startswith("jaccard")]
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    x = np.arange(len(d))
    for i, r in enumerate(d.itertuples()):
        ax.plot([i, i], [r.auroc_CTHRC1_alone, r.auroc_program], color=GRAY_MID, lw=0.8, zorder=1)
    ax.scatter(x, d.auroc_CTHRC1_alone, s=22, c=ORANGE, zorder=2, label="CTHRC1 alone", edgecolors="white", linewidths=0.6)
    ax.scatter(x, d.auroc_program, s=22, c=BLUE, zorder=3, label=f"{len(PROG['CTHRC1_matrix_CAF'])}-gene program", edgecolors="white", linewidths=0.6)
    for i, r in enumerate(d.itertuples()):
        ax.text(i + 0.12, r.auroc_program, f"{r.auroc_program:.2f}", fontsize=6, color=INK, va="center")
    ax.set_xticks(x, d.held_out, rotation=30, ha="right")
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("AUROC in held-out dataset\n(CTHRC1+ matrix myCAF vs other CAF)")
    ax.set_xlim(-0.5, len(d) - 0.3)
    ax.grid(axis="y", color="#e8e7e2", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=6, handletextpad=0.1)
    ax.set_title("Leave-one-dataset-out", loc="left")
    save(fig, "g_lodo")


def main():
    import sys
    only = sys.argv[1:]
    if only == ["caf"]:
        panel_caf()
        return
    if only == ["f"]:
        panel_contribution()
        return
    g = panel_global()
    a = panel_caf()
    panel_scores(g, a)
    panel_contribution()
    panel_lodo()
    print(f"wrote panels to {OUT}")


if __name__ == "__main__":
    main()
