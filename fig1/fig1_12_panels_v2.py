"""Fig 1 面板 v2（审稿人友好版，pdac_scvi 环境）。旧版 fig1_06_panels.py 保留，其 d/e/f 面板转为补充图。

b  global_umap        全局图谱：全部谱系上色（按细胞区室分色系，区室内深浅区分），每个谱系直接标注
c  lineage_dotplot    大类标志基因 dotplot（注释依据）
d  caf_umap           CAF 细分：8 个亚群上色（myCAF 橙系 / iCAF 绿系 / 其他灰系），CTHRC1+ 为蓝色
e  subtype_dotplot    亚群标志基因 + 经典 CAF 标志（ACTA2、LRRC15、FAP、PDGFRA、CXCL12、IL6、HLA-DRA、CD74）
f  composition        按组织类型（正常/癌旁、未治疗、化疗后、治疗未知）的各样本 CAF 亚群组成
g  program_dotplot    程序基因与泛 CAF 参考在全部谱系和 CAF 亚群中的表达（替代旧 d/e/f）
h  lodo               留一数据集验证（沿用旧版）

dotplot：点大小 = 表达细胞比例，颜色 = 平均 log 表达，每个基因在行间缩放到 0–1（g 中谱系与 CAF 亚群两块一起缩放）。
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
import sys  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fig1_06_panels import (AQUA, BLUE, CAF_ATLAS, CAF_NAMES, DROP_LINEAGES, DS_MARKERS, GRAY_DARK,  # noqa: E402
                            GRAY_LIGHT, GRAY_MID, INK, INK2, LINEAGE_LABEL, ORANGE, OUT, PROG, ROOT, SEQ,
                            label_centroids, panel_lodo, save, umap_axes)

# ---- colours: hue = compartment / CAF family, lightness separates members; identity always carried by labels
LIN_COLOR = {
    "Fibroblast": BLUE, "Pericyte_SMC": AQUA,
    "Ductal_tumor": "#eb6834", "Ductal_normal": "#f4a27f", "Acinar": "#b9481c", "Endocrine": "#f7c5ad",
    "Tuft": "#d9622c",
    "T_NK": "#4a3aa7", "B": "#8379cf", "Plasma": "#b2abe6", "Myeloid": "#6a5bc0", "Mast": "#3a2d85",
    "pDC": "#9d95dc", "Neutrophil_RBC": "#c9c4ef",
    "Endothelial": "#eda100", "Schwann": GRAY_DARK,
}
COMPARTMENTS = [("Fibroblast", BLUE), ("Pericyte / SMC", AQUA), ("Epithelial", "#eb6834"),
                ("Immune", "#6a5bc0"), ("Endothelial", "#eda100"), ("Schwann", GRAY_DARK)]
LIN_ORDER = ["Ductal_tumor", "Ductal_normal", "Acinar", "Endocrine", "Tuft", "Fibroblast", "Pericyte_SMC",
             "Endothelial", "Schwann", "T_NK", "B", "Plasma", "Myeloid", "Mast", "pDC", "Neutrophil_RBC"]
LIN_MARKERS = {
    "Ductal_tumor": ["EPCAM", "KRT19"], "Ductal_normal": ["CFTR", "SLC4A4"], "Acinar": ["PRSS1", "CPA1"],
    "Endocrine": ["CHGA", "INS"], "Tuft": ["SH2D6", "POU2F3"], "Fibroblast": ["DCN", "LUM"],
    "Pericyte_SMC": ["RGS5", "MYH11"], "Endothelial": ["VWF", "CDH5"], "Schwann": ["S100B", "PLP1"],
    "T_NK": ["CD3D", "NKG7"], "B": ["MS4A1", "CD79A"], "Plasma": ["JCHAIN", "MZB1"], "Myeloid": ["CD68", "LYZ"],
    "Mast": ["TPSAB1", "CPA3"], "pDC": ["LILRA4", "IRF7"], "Neutrophil_RBC": ["S100A8", "HBB"],
}
SUB_COLOR = {"CTHRC1+ matrix myCAF": BLUE, "HOPX+ F2R+ myCAF": "#eb6834", "Glycolytic CAF": "#f4a27f",
             "C7+ C3+ iCAF": AQUA, "MFAP5+ IGF1+ iCAF": "#8fd8b9", "PI16+ universal": GRAY_DARK,
             "SPP1+ IBSP+ CAF (8 patients)": GRAY_MID, "Chemo-associated CAF": GRAY_LIGHT}
SUB_ORDER = ["CTHRC1+ matrix myCAF", "HOPX+ F2R+ myCAF", "Glycolytic CAF", "C7+ C3+ iCAF", "MFAP5+ IGF1+ iCAF",
             "PI16+ universal", "SPP1+ IBSP+ CAF (8 patients)", "Chemo-associated CAF"]
SUB_SHORT = {s: s.replace(" (8 patients)", "") for s in SUB_ORDER}
SUB_MARKERS = {"CTHRC1+ matrix myCAF": ["CTHRC1", "COL11A1", "COL10A1"], "HOPX+ F2R+ myCAF": ["HOPX", "F2R", "COL4A1"],
               "Glycolytic CAF": ["ENO1", "LDHA", "PGK1"], "C7+ C3+ iCAF": ["C7", "C3", "SPARCL1"],
               "MFAP5+ IGF1+ iCAF": ["MFAP5", "IGF1", "MEDAG"], "PI16+ universal": ["PI16", "CLDN1", "ABCA8"],
               "SPP1+ IBSP+ CAF (8 patients)": ["SPP1", "IBSP", "MMP13"],
               "Chemo-associated CAF": ["OGN", "KCNN3", "THBS4"]}
CLASSIC = [("myCAF", ["ACTA2", "LRRC15", "FAP"]), ("iCAF", ["PDGFRA", "CXCL12", "IL6"]),
           ("apCAF", ["HLA-DRA", "CD74"])]
CAF_OFFSETS = {"SPP1+ IBSP+ CAF (8 patients)": (0.0, 0.09), "Chemo-associated CAF": (0.10, 0.06),
               "PI16+ universal": (0.04, 0.10), "Glycolytic CAF": (-0.16, 0.10), "MFAP5+ IGF1+ iCAF": (0.10, 0.12)}


def load_global():
    g = sc.read_h5ad(ROOT / "integrate_global" / "fibro_integrated.h5ad")
    ann = pd.read_csv(ROOT / "annotate_global" / "cluster_annotation.tsv", sep="\t", index_col=0)
    ann.index = ann.index.astype(str)
    g.obs["lineage"] = g.obs["leiden_0.5"].astype(str).map(ann.final_label)
    return g[~g.obs.lineage.isin(DROP_LINEAGES)].copy()


def load_caf():
    a = sc.read_h5ad(ROOT / CAF_ATLAS / "fibro_integrated.h5ad")
    a.obs["subtype"] = a.obs["leiden_0.5"].astype(str).map(CAF_NAMES)
    return a


def group_stats(adata, key, order, genes):
    """mean log-normalised expression and fraction of expressing cells per group (rows) x gene (cols)."""
    genes = [x for x in genes if x in adata.var_names]
    X = adata[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    lab = adata.obs[key].astype(str).to_numpy()
    mean = pd.DataFrame([X[lab == o].mean(0) for o in order], index=order, columns=genes)
    frac = pd.DataFrame([(X[lab == o] > 0).mean(0) for o in order], index=order, columns=genes)
    return mean, frac


def dotplot(ax, mean, frac, scale=True, vmax=None, smax=42):
    m = mean.copy()
    if scale:
        m = (m - m.min()) / (m.max() - m.min()).replace(0, 1)
    vmax = vmax if vmax is not None else (1.0 if scale else float(m.to_numpy().max()))
    yy, xx = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij")
    sc_ = ax.scatter(xx.ravel(), yy.ravel(), s=frac.to_numpy().ravel() * smax, c=m.to_numpy().ravel(),
                     cmap=SEQ, vmin=0, vmax=vmax, linewidths=0.25, edgecolors=GRAY_MID)
    ax.set_xlim(-0.6, m.shape[1] - 0.4); ax.set_ylim(m.shape[0] - 0.4, -0.6)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=90, fontsize=5.6, fontstyle="italic")
    ax.tick_params(length=0)
    ax.tick_params(axis="y", pad=9)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    return sc_


def dot_legends(fig, ax, mappable, color_label, smax=42, anchor=(1.02, 0.0), cb_dx=0.03):
    """size legend (fraction expressing) + slim colour bar, placed to the right of `ax`."""
    hs = [ax.scatter([], [], s=f * smax, c="white", edgecolors=INK2, linewidths=0.4) for f in (0.25, 0.5, 1.0)]
    leg = ax.legend(hs, ["25%", "50%", "100%"], title="Cells\nexpressing", fontsize=5.5, title_fontsize=5.5,
                    loc="lower left", bbox_to_anchor=anchor, labelspacing=0.6, handletextpad=0.3, borderaxespad=0)
    leg._legend_box.align = "left"
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + cb_dx, pos.y1 - 0.22 * pos.height, 0.012, 0.18 * pos.height])
    cb = fig.colorbar(mappable, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.2, length=1.5)
    cb.set_label(color_label, fontsize=5.5)


def group_bracket(ax, groups, y, text_y, horizontal=True, fontsize=5.6):
    """thin brackets with a group name over consecutive columns (horizontal) or beside rows."""
    x = 0
    for name, n in groups:
        if horizontal:
            ax.plot([x - 0.3, x + n - 0.7], [y, y], color=INK2, lw=0.5, clip_on=False)
            ax.text(x + (n - 1) / 2, text_y, name, ha="center", va="bottom", fontsize=fontsize if n > 2 else fontsize - 0.6,
                    color=INK2)
        x += n


# ------------------------------------------------------------------ b. global atlas, all lineages coloured
def panel_global(g):
    xy = g.obsm["X_umap"]
    lin = g.obs.lineage.astype(str).to_numpy()
    col = np.array([LIN_COLOR[l] for l in lin])
    order = np.random.default_rng(0).permutation(len(lin))
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.scatter(xy[order, 0], xy[order, 1], c=col[order], s=0.15, linewidths=0, rasterized=True)
    label_centroids(ax, xy, lin, min_n=100, fontsize=6, offsets={
        "Endocrine": (-0.10, 0.12), "Schwann": (0.02, -0.17), "pDC": (-0.07, -0.08), "Tuft": (-0.04, 0.10),
        "Neutrophil_RBC": (0.0, 0.10)})
    umap_axes(ax, f"Integrated PDAC atlas ({g.n_obs:,} cells, {g.obs.patient_key.nunique()} patients)")
    hs = [plt.Line2D([], [], marker="o", ls="", color=c, ms=4, label=l) for l, c in COMPARTMENTS]
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, -0.04), fontsize=5.8, handletextpad=0.1,
              ncol=3, columnspacing=0.6, borderaxespad=0)
    save(fig, "v2_b_global_umap")


# ------------------------------------------------------------------ c. lineage marker dotplot
def panel_lineage_dotplot(g):
    order = [l for l in LIN_ORDER if l in set(g.obs.lineage)]
    genes = [x for l in order for x in LIN_MARKERS[l]]
    mean, frac = group_stats(g, "lineage", order, genes)
    fig, ax = plt.subplots(figsize=(3.0, 2.9))
    m = dotplot(ax, mean, frac)
    ax.set_yticks(range(len(order)), [LINEAGE_LABEL.get(l, l.replace("_", " ")) for l in order], fontsize=6)
    for i, l in enumerate(order):  # colour key next to each lineage name, matching panel b
        ax.scatter(-0.95, i, s=10, c=LIN_COLOR[l], clip_on=False, linewidths=0, transform=ax.transData)
    ax.set_title("Lineage markers", loc="left")
    fig.subplots_adjust(right=0.8)
    dot_legends(fig, ax, m, "Scaled mean\nexpression")
    save(fig, "v2_c_lineage_dotplot")
    pd.concat({"mean": mean, "frac": frac}, axis=1).to_csv(OUT / "v2_c_lineage_dotplot.tsv", sep="\t")


# ------------------------------------------------------------------ d. CAF subtypes
def panel_caf_umap(a):
    xy = a.obsm["X_umap"]
    st = a.obs.subtype.astype(str).to_numpy()
    col = np.array([SUB_COLOR[s] for s in st])
    order = np.random.default_rng(0).permutation(len(st))
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.scatter(xy[order, 0], xy[order, 1], c=col[order], s=1.0, linewidths=0, rasterized=True)
    label_centroids(ax, xy, st, min_n=50, fontsize=6, offsets=CAF_OFFSETS)
    umap_axes(ax, f"CAF subtypes ({a.n_obs:,} CAFs, {a.obs.patient_key.nunique()} patients)")
    save(fig, "v2_d_caf_umap")


# ------------------------------------------------------------------ e. subtype markers + classical CAF markers
def panel_subtype_dotplot(a):
    genes = [x for s in SUB_ORDER for x in SUB_MARKERS[s]] + [x for _, gs in CLASSIC for x in gs]
    mean, frac = group_stats(a, "subtype", SUB_ORDER, genes)
    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    m = dotplot(ax, mean, frac, smax=36)
    ax.set_yticks(range(len(SUB_ORDER)), [SUB_SHORT[s] for s in SUB_ORDER], fontsize=6)
    for i, s in enumerate(SUB_ORDER):
        ax.scatter(-0.95, i, s=10, c=SUB_COLOR[s], clip_on=False, linewidths=0, transform=ax.transData)
    n_sub = sum(len(SUB_MARKERS[s]) for s in SUB_ORDER)
    ax.axvline(n_sub - 0.5, color=GRAY_MID, lw=0.5)
    group_bracket(ax, [("Subtype markers", n_sub)] + [(n, len(gs)) for n, gs in CLASSIC],
                  -0.95, -1.15)
    ax.text(n_sub + 3.5, -1.75, "Classical CAF markers", ha="center", fontsize=5.6, color=INK2)
    ax.set_title("CAF subtype markers", loc="left", pad=20)
    fig.subplots_adjust(right=0.82)
    dot_legends(fig, ax, m, "Scaled mean\nexpression", smax=36)
    save(fig, "v2_e_subtype_dotplot")
    pd.concat({"mean": mean, "frac": frac}, axis=1).to_csv(OUT / "v2_e_subtype_dotplot.tsv", sep="\t")


# ------------------------------------------------------------------ f. composition per tissue / sample
TISSUE = {"normal_pancreas": "Normal/adj.", "adjacent_normal": "Normal/adj.", "PDAC_naive": "PDAC naive",
          "PDAC_chemo": "Post-chemo", "PDAC_unknown_tx": "Tx unknown"}
TISSUE_ORDER = ["Normal/adj.", "PDAC naive", "Post-chemo", "Tx unknown"]


def panel_composition(a, min_cafs=30, present_at=0.05):
    """样本 = 患者 × 条件（同一患者的肿瘤与癌旁分开计）；只画 ≥min_cafs 个 CAF 的样本。"""
    obs = a.obs[["dataset", "patient_key", "condition", "subtype"]].astype(str)
    obs["tissue"] = obs.condition.map(TISSUE)
    obs["sample"] = obs.patient_key + "|" + obs.condition
    ct = pd.crosstab(obs["sample"], obs.subtype).reindex(columns=SUB_ORDER, fill_value=0)
    meta = obs.groupby("sample")[["dataset", "tissue"]].first()
    ct = ct[ct.sum(1) >= min_cafs]
    fr = ct.div(ct.sum(1), axis=0)
    cth = SUB_ORDER[0]
    present = fr[cth] >= present_at
    widths = [int((meta.tissue[fr.index] == t).sum()) for t in TISSUE_ORDER]
    fig, axs = plt.subplots(1, 1 + len(TISSUE_ORDER), figsize=(7.0, 1.9),
                            gridspec_kw=dict(width_ratios=[5] + [max(w, 7) for w in widths], wspace=0.12))
    # pooled per tissue type (all CAFs)
    pooled = pd.crosstab(obs.tissue, obs.subtype, normalize="index").reindex(index=TISSUE_ORDER, columns=SUB_ORDER)
    ax = axs[0]
    bottom = np.zeros(len(TISSUE_ORDER))
    for s in SUB_ORDER:
        ax.bar(range(len(TISSUE_ORDER)), pooled[s], bottom=bottom, color=SUB_COLOR[s], width=0.75,
               edgecolor="white", linewidth=0.4)
        bottom += pooled[s].to_numpy()
    ax.set_xticks(range(len(TISSUE_ORDER)), TISSUE_ORDER, fontsize=5.6, rotation=60, ha="right")
    ax.set_ylabel("Fraction of CAFs")
    ax.set_ylim(0, 1)
    ax.set_title("Pooled", loc="left", fontsize=6)
    for ax, t in zip(axs[1:], TISSUE_ORDER):
        sub = fr[meta.tissue[fr.index] == t].sort_values(cth, ascending=False)
        bottom = np.zeros(len(sub))
        for s in SUB_ORDER:
            ax.bar(range(len(sub)), sub[s], bottom=bottom, color=SUB_COLOR[s], width=0.9, linewidth=0)
            bottom += sub[s].to_numpy()
        for i, d in enumerate(meta.dataset[sub.index]):  # dataset of each sample, shape-coded under the bar
            ax.scatter(i, -0.07, s=5, marker=DS_MARKERS[d], facecolors="none", edgecolors=INK2, linewidths=0.4,
                       clip_on=False)
        ax.set_xlim(-0.6, max(len(sub), 7) - 0.4); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        p = present[sub.index]
        ax.set_title(f"{t}\n{p.sum()}/{len(p)}", loc="left", fontsize=6)
    axs[1].text(0, -0.17, f"Samples with ≥{min_cafs} CAFs, sorted by CTHRC1+ share; "
                f"x/y = samples in which CTHRC1+ cells are ≥{present_at:.0%} of CAFs", transform=axs[1].transAxes, fontsize=5.6,
                color=INK2, va="top")
    hs = [plt.Rectangle((0, 0), 1, 1, color=SUB_COLOR[s]) for s in SUB_ORDER]
    hs += [plt.Line2D([], [], marker=m, ls="", mfc="none", mec=INK2, ms=3.5) for m in DS_MARKERS.values()]
    fig.legend(hs, [SUB_SHORT[s] for s in SUB_ORDER] + list(DS_MARKERS), loc="upper center",
               bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=5.6, handlelength=0.9, handletextpad=0.3,
               columnspacing=0.9)
    fig.suptitle("CTHRC1+ matrix myCAFs are tumour-specific", x=0.125, ha="left", y=1.1, fontsize=8)
    save(fig, "v2_f_composition")
    fr.assign(dataset=meta.dataset[fr.index], tissue=meta.tissue[fr.index].str.replace("\n", " "),
              n_cafs=ct.sum(1)).to_csv(OUT / "v2_f_composition_per_sample.tsv", sep="\t")
    for t in TISSUE_ORDER:
        p = present[meta.tissue[fr.index] == t]
        print(f"{t}: {p.sum()}/{len(p)} samples with CTHRC1+ ≥{present_at:.0%}; "
              f"median share {fr.loc[p.index, cth].median():.3f}")


# ------------------------------------------------------------------ g. program genes across lineages and CAF subtypes
def panel_program_dotplot(g, a):
    genes = PROG["CTHRC1_matrix_CAF"] + PROG["pan_CAF"]
    order = [l for l in LIN_ORDER if l in set(g.obs.lineage)]
    m1, f1 = group_stats(g, "lineage", order, genes)
    m2, f2 = group_stats(a, "subtype", SUB_ORDER, genes)
    mean = pd.concat([m1, m2]); frac = pd.concat([f1, f2])
    fig, ax = plt.subplots(figsize=(2.9, 4.2))
    m = dotplot(ax, mean, frac, smax=34)
    rows = [LINEAGE_LABEL.get(l, l.replace("_", " ")) for l in order] + [SUB_SHORT[s] for s in SUB_ORDER]
    ax.set_yticks(range(len(rows)), rows, fontsize=5.8)
    for i, c in enumerate([LIN_COLOR[l] for l in order] + [SUB_COLOR[s] for s in SUB_ORDER]):
        ax.scatter(-0.95, i, s=9, c=c, clip_on=False, linewidths=0, transform=ax.transData)
    ax.axhline(len(order) - 0.5, color=INK2, lw=0.6)
    ax.axvline(len(PROG["CTHRC1_matrix_CAF"]) - 0.5, color=GRAY_MID, lw=0.5)
    ax.text(len(genes) - 0.3, len(order) / 2 - 0.5, "All lineages", rotation=270, va="center", fontsize=5.8,
            color=INK2)
    ax.text(len(genes) - 0.3, len(order) + len(SUB_ORDER) / 2 - 0.5, "CAF subtypes", rotation=270, va="center",
            fontsize=5.8, color=INK2)
    group_bracket(ax, [(f"CTHRC1+ program ({len(PROG['CTHRC1_matrix_CAF'])})", len(PROG["CTHRC1_matrix_CAF"])),
                       (f"Pan-CAF ref. ({len(PROG['pan_CAF'])})", len(PROG["pan_CAF"]))], -0.95, -1.15)
    ax.set_title("Program genes are expressed by fibroblasts,\nhighest in CTHRC1+ matrix myCAF", loc="left", pad=14)
    fig.subplots_adjust(right=0.78)
    dot_legends(fig, ax, m, "Scaled mean\nexpression", smax=34, anchor=(1.12, 0.0), cb_dx=0.09)
    save(fig, "v2_g_program_dotplot")
    pd.concat({"mean": mean, "frac": frac}, axis=1).to_csv(OUT / "v2_g_program_dotplot.tsv", sep="\t")


def main():
    only = set(sys.argv[1:])
    want = (lambda k: not only or k in only)
    g = load_global() if (want("b") or want("c") or want("g")) else None
    a = load_caf() if (want("d") or want("e") or want("f") or want("g")) else None
    if want("b"):
        panel_global(g)
    if want("c"):
        panel_lineage_dotplot(g)
    if want("d"):
        panel_caf_umap(a)
    if want("e"):
        panel_subtype_dotplot(a)
    if want("f"):
        panel_composition(a)
    if want("g"):
        panel_program_dotplot(g, a)
    if want("h"):
        panel_lodo()
    print(f"wrote v2 panels to {OUT}")


if __name__ == "__main__":
    main()
