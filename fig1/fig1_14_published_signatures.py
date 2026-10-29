"""Comparison of the CTHRC1+ programme with published PDAC stromal / CAF gene sets (Figure 1 supplement, pdac_scvi environment).

A  overlap        membership of programme genes and the pan-CAF reference in each published gene set
B  subtype_scores score of each published gene set in the 8 CAF clusters (cluster means, z-scored across clusters per gene set)
C  contribution   fraction of each gene set's member transcripts in PDAC tissue coming from fibroblasts (mean of three datasets)
                  note: the programme was selected at >= 0.6, so its high value in C follows from the selection rule and is not independent evidence.
Gene sets: config/published_signatures.yaml (all traceable to the original paper or the authors' code).
Output: results/fig1/figure_panels/supp_published_signatures.{pdf,png}, results/fig1/published_signatures/*.tsv
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fig1_06_panels import BLUE, GRAY_LIGHT, GRAY_MID, INK, INK2, OUT, PROG, ROOT, save  # noqa: E402
from fig1_12_panels_v2 import SUB_COLOR, SUB_ORDER, SUB_SHORT, load_caf  # noqa: E402

PUB = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "published_signatures.yaml").read_text())
PUB = {k: v for k, v in PUB.items() if k not in ("Moffitt_basal", "Moffitt_classical", "MPAS")}   # stromal / CAF gene sets only
TAB = ROOT / "published_signatures"
TAB.mkdir(parents=True, exist_ok=True)
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948"])
OURS = {"CTHRC1_program": ("CTHRC1+ program (this study)", PROG["CTHRC1_matrix_CAF"]),
        "pan_CAF": ("Pan-CAF ref. (this study)", PROG["pan_CAF"])}


SHORT = {"Moffitt_activated_stroma": "Moffitt act.", "Moffitt_normal_stroma": "Moffitt norm.",
         "Puleo_stroma_activated": "Puleo act.", "Elyada_myCAF": "Elyada my", "Elyada_iCAF": "Elyada i",
         "Elyada_apCAF": "Elyada ap", "Dominguez_LRRC15_CAF": "LRRC15+"}


def signature_table(var_names):
    sets = {k: (v[0], v[1]) for k, v in OURS.items()}
    sets.update({k: (v["label"], v["genes"]) for k, v in PUB.items()})
    rows = []
    for k, (lab, genes) in sets.items():
        present = [g for g in genes if g in var_names]
        rows.append({"key": k, "label": lab, "n_genes": len(genes), "n_in_data": len(present),
                     "missing": ",".join(sorted(set(genes) - set(present))), "genes": present})
    return pd.DataFrame(rows).set_index("key")


def panel_overlap(ax, sig):
    genes = PROG["CTHRC1_matrix_CAF"] + PROG["pan_CAF"]
    cols = list(PUB)
    M = np.array([[g in PUB[c]["genes"] for c in cols] for g in genes])
    yy, xx = np.nonzero(M)
    ax.scatter(xx, yy, s=26, c=INK2, linewidths=0)
    ax.scatter(*np.meshgrid(range(len(cols)), range(len(genes))), s=26, facecolors="none", edgecolors=GRAY_LIGHT,
               linewidths=0.5, zorder=0)
    ax.axhline(len(PROG["CTHRC1_matrix_CAF"]) - 0.5, color=GRAY_MID, lw=0.6)
    ax.set_xticks(range(len(cols)), [SHORT[c] for c in cols], rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(genes)), genes, fontsize=6, fontstyle="italic")
    ax.set_xlim(-0.6, len(cols) - 0.4); ax.set_ylim(len(genes) - 0.5, -1.4)
    n_prog = len(PROG["CTHRC1_matrix_CAF"])
    for j in range(len(cols)):
        ax.text(j, -1.0, f"{M[:n_prog, j].sum()}/{n_prog}", ha="center", fontsize=5.6, color=INK)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("Gene overlap\n(top: program genes found in each set)", loc="left", fontsize=7)
    return pd.DataFrame(M, index=genes, columns=cols)


def panel_scores(ax, fig, a, sig):
    for k in sig.index:
        sc.tl.score_genes(a, sig.loc[k, "genes"], score_name=f"s_{k}", random_state=0)
    st = a.obs.subtype.astype(str)
    mean = pd.DataFrame({k: a.obs[f"s_{k}"].groupby(st).mean() for k in sig.index}).reindex(SUB_ORDER)
    z = (mean - mean.mean()) / mean.std()
    im = ax.imshow(z.T.to_numpy(), cmap=DIV, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(SUB_ORDER)), [SUB_SHORT[s] for s in SUB_ORDER], rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(sig)), sig.label, fontsize=6)
    for i, s in enumerate(SUB_ORDER):
        ax.scatter(i, len(sig) - 0.2, s=10, c=SUB_COLOR[s], clip_on=False, linewidths=0)
    ax.axhline(1.5, color="white", lw=2)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cax = fig.add_axes([0.56, 0.035, 0.17, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Score, z across subtypes", fontsize=5.6)
    cb.ax.tick_params(labelsize=5.4, length=1.5)
    cb.outline.set_visible(False)
    ax.set_title("Published signatures across CAF subtypes", loc="left", fontsize=7)
    return mean, z


def panel_contribution(ax, sig):
    c = pd.read_csv(ROOT / "signature" / "fibroblast_contribution.tsv", sep="\t")
    fc = c.groupby("gene").fib_contribution.mean()
    rows = []
    for i, k in enumerate(sig.index):
        v = fc.reindex(sig.loc[k, "genes"]).dropna()
        col = BLUE if k == "CTHRC1_program" else GRAY_MID
        ax.boxplot(v, positions=[i], vert=False, widths=0.55, showfliers=False,
                   medianprops=dict(color=INK, lw=0.8), boxprops=dict(color=col, lw=0.7),
                   whiskerprops=dict(color=col, lw=0.6), capprops=dict(color=col, lw=0.6))
        ax.scatter(v, i + np.random.default_rng(i).uniform(-0.17, 0.17, len(v)), s=4, c=col, linewidths=0)
        rows.append({"key": k, "label": sig.loc[k, "label"], "n": len(v), "median": v.median(),
                     "frac_ge_0.6": (v >= 0.6).mean()})
    ax.axvline(0.6, color=GRAY_MID, lw=0.5, ls="--")
    ax.set_yticks(range(len(sig)), [])          # rows aligned with the heatmap to the left
    ax.set_ylim(len(sig) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of transcripts from fibroblasts\n(PDAC tissue, mean of 3 datasets)", fontsize=6)
    ax.set_title("Fibroblast origin of each set's genes", loc="left", fontsize=7, pad=14)
    ax.text(0.59, -0.45, "selection threshold\nof this study", fontsize=5.2, color=INK2, va="bottom", ha="right")
    return pd.DataFrame(rows)


def main():
    a = load_caf()
    sig = signature_table(a.var_names)
    sig.drop(columns="genes").to_csv(TAB / "signature_genes_in_data.tsv", sep="\t")
    fig = plt.figure(figsize=(7.1, 3.6))
    # manual placement: the heatmap's row labels need the gap between the first two panels
    ov = panel_overlap(fig.add_axes([0.09, 0.30, 0.22, 0.56]), sig)
    mean, z = panel_scores(fig.add_axes([0.52, 0.30, 0.25, 0.56]), fig, a, sig)
    contrib = panel_contribution(fig.add_axes([0.79, 0.30, 0.19, 0.56]), sig)
    save(fig, "supp_published_signatures")
    ov.to_csv(TAB / "overlap.tsv", sep="\t")
    mean.to_csv(TAB / "subtype_mean_scores.tsv", sep="\t")
    contrib.to_csv(TAB / "fibroblast_contribution_by_set.tsv", sep="\t", index=False)
    n_prog = len(PROG["CTHRC1_matrix_CAF"])
    print(sig[["label", "n_genes", "n_in_data", "missing"]].to_string())
    print("\nprogram genes in each set:", ov.iloc[:n_prog].sum().to_dict())
    print("\ntop subtype per signature:", z.idxmax().to_dict())
    print(contrib.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
