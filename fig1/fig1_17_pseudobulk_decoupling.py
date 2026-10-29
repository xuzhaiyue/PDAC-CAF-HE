"""模拟 bulk 拆分检验（hemorph 环境）。规则见 config/fig1.yaml 的 decoupling（运行前已冻结）。

A. 真实样本：每个 患者×条件 的全部细胞加总为模拟 bulk；真值 = CAF 占比（全数据谱系注释）
   与 CTHRC1+ 占 CAF 的比例（CAF 图谱，随机抽样的清理后 CAF）
B. 受控混合：每个数据集内，从肿瘤样本的 CTHRC1+ CAF / 其他 CAF / 非 CAF 细胞池按网格抽样（有放回），
   CAF 占比与 CTHRC1+ 比例正交变化
得分（bulk 常用写法）：基因集内 log2(CPM+1) 的均值；相对得分 = 该得分 − 泛 CAF 得分。
比较对象：本程序、Moffitt 活化基质、Dominguez LRRC15+，各自的原始得分与相对得分；以及泛 CAF 本身。
Output: results/fig1/decoupling/*.tsv，results/fig1/figure_panels/decoupling.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((PROJ / "config" / "fig1.yaml").read_text())
DC = CFG["decoupling"]
ROOT = Path(CFG["paths"]["out_dir"])
OUT = ROOT / "decoupling"
OUT.mkdir(parents=True, exist_ok=True)
PANELS = ROOT / "figure_panels"
PROG = yaml.safe_load((ROOT / "signature" / "programs.yaml").read_text())
PUB = yaml.safe_load((PROJ / "config" / "published_signatures.yaml").read_text())
SETS = {"program": PROG["CTHRC1_matrix_CAF"], "Moffitt_activated": PUB["Moffitt_activated_stroma"]["genes"],
        "Dominguez_LRRC15": PUB["Dominguez_LRRC15_CAF"]["genes"], "pan_CAF": PROG["pan_CAF"]}
LABEL = {"program": "9-gene program", "Moffitt_activated": "Moffitt activated stroma",
         "Dominguez_LRRC15": "Dominguez LRRC15+ CAF", "pan_CAF": "Pan-CAF ref."}
DROP = ["Ambiguous", "Low_quality", "Hepatocyte_like"]
TARGET = "2"                                   # CTHRC1+ matrix myCAF (integrate_caf_final, leiden_0.5)
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#888780", "#0b0b0b", "#52514e"
SEQ5 = ["#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
RNG = np.random.default_rng(CFG["random_seed"])


def scores(P: np.ndarray, var: pd.Index) -> pd.DataFrame:
    """P: bulk x genes counts -> raw and relative signature scores."""
    L = np.log2(P / P.sum(1, keepdims=True) * 1e6 + 1)
    s = {}
    for k, g in SETS.items():
        idx = [var.get_loc(x) for x in g if x in var]
        s[k] = L[:, idx].mean(1)
    s = pd.DataFrame(s)
    for k in DC["compare"]:
        s[f"{k}_rel"] = s[k] - s["pan_CAF"]
    return s


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return stats.pearsonr(res(rx), res(ry))[0]


def load():
    full = ad.read_h5ad(ROOT / "global_full_raw.h5ad")
    ann = pd.read_csv(ROOT / "annotate_global" / "global_full_annotation.tsv.gz", sep="\t", index_col=0)
    full = full[full.obs_names.isin(ann.index[~ann.lineage.isin(DROP)])].copy()
    full.obs["lineage"] = ann.lineage.reindex(full.obs_names).values
    caf = ad.read_h5ad(ROOT / "integrate_caf_final" / "fibro_integrated.h5ad", backed="r").obs
    caf = caf.assign(is_target=caf["leiden_0.5"].astype(str) == TARGET)
    return full, caf


# ---------------------------------------------------------------- A. natural pseudo-bulk
def natural(full, caf):
    X = sp.csr_matrix(full.X)
    key = (full.obs.patient_key.astype(str) + "|" + full.obs.condition.astype(str)).to_numpy()
    uniq, inv = np.unique(key, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    P = np.asarray((G @ X).todense(), float)
    meta = pd.DataFrame({"sample": uniq})
    meta["dataset"] = full.obs.dataset.astype(str).groupby(key).first().reindex(uniq).values
    meta["condition"] = [u.split("|")[1] for u in uniq]
    meta["n_cells"] = np.bincount(inv)
    meta["caf_frac"] = pd.Series(full.obs.lineage.eq("Fibroblast").to_numpy()).groupby(inv).mean().values
    ck = caf.patient_key.astype(str) + "|" + caf.condition.astype(str)
    t = caf.groupby(ck).is_target.agg(["mean", "size"]).reindex(uniq)
    meta["cthrc1_frac"], meta["n_caf_atlas"] = t["mean"].values, t["size"].values
    d = pd.concat([meta, scores(P, full.var_names)], axis=1)
    d = d[d.n_caf_atlas >= DC["natural"]["min_caf_atlas_cells"]].reset_index(drop=True)
    d.to_csv(OUT / "natural_pseudobulk.tsv", sep="\t", index=False)
    rows = []
    for col in [c for c in d.columns if c in SETS or c.endswith("_rel")]:
        rows.append({"score": col, "n": len(d),
                     "rho_state": stats.spearmanr(d[col], d.cthrc1_frac)[0],
                     "rho_caf": stats.spearmanr(d[col], d.caf_frac)[0],
                     "partial_state": partial_spearman(d[col], d.cthrc1_frac, d.caf_frac),
                     "partial_caf": partial_spearman(d[col], d.caf_frac, d.cthrc1_frac)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "natural_correlations.tsv", sep="\t", index=False)
    print(f"natural pseudo-bulk: {len(d)} samples; rho(caf_frac, cthrc1_frac) = "
          f"{stats.spearmanr(d.caf_frac, d.cthrc1_frac)[0]:.2f}")
    print(r.round(2).to_string(index=False))
    return d, r


# ---------------------------------------------------------------- B. controlled mixtures
def mixtures(full, caf):
    M = DC["mixtures"]
    X = sp.csr_matrix(full.X)
    pos = pd.Series(np.arange(full.n_obs), index=full.obs_names)
    tumour = full.obs.condition.isin(M["conditions"]).to_numpy()
    caf_t = caf[caf.condition.isin(M["conditions"])]
    out = []
    for ds in sorted(full.obs.dataset.unique()):
        cd = caf_t[caf_t.dataset == ds]
        pools = {"cth": pos[cd.index[cd.is_target]].to_numpy(), "oth": pos[cd.index[~cd.is_target]].to_numpy(),
                 "non": np.where(tumour & (full.obs.dataset == ds).to_numpy() &
                                 (full.obs.lineage != "Fibroblast").to_numpy())[0]}
        print(f"{ds}: pools CTHRC1+ {len(pools['cth'])}, other CAF {len(pools['oth'])}, non-CAF {len(pools['non'])}")
        rows, mats = [], []
        for cf in M["caf_frac"]:
            for tf in M["cthrc1_frac"]:
                for rep in range(M["reps"]):
                    n_caf = int(round(M["n_cells"] * cf))
                    n_cth = int(round(n_caf * tf))
                    idx = np.concatenate([RNG.choice(pools["cth"], n_cth), RNG.choice(pools["oth"], n_caf - n_cth),
                                          RNG.choice(pools["non"], M["n_cells"] - n_caf)])
                    mats.append(np.asarray(X[idx].sum(0)).ravel())   # duplicated rows are summed twice
                    rows.append({"dataset": ds, "caf_frac": cf, "cthrc1_frac": tf, "rep": rep})
        d = pd.concat([pd.DataFrame(rows), scores(np.vstack(mats), full.var_names)], axis=1)
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d.to_csv(OUT / "mixtures.tsv", sep="\t", index=False)
    rows = []
    for ds, g in d.groupby("dataset"):
        for col in [c for c in d.columns if c in SETS or c.endswith("_rel")]:
            rows.append({"dataset": ds, "score": col, "rho_state": stats.spearmanr(g[col], g.cthrc1_frac)[0],
                         "rho_caf": stats.spearmanr(g[col], g.caf_frac)[0]})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "mixture_correlations.tsv", sep="\t", index=False)
    print(r.pivot(index="score", columns="dataset", values=["rho_state", "rho_caf"]).round(2).to_string())
    return d, r


def verdict(rn, rm):
    s = DC["success"]
    m = rm[rm.score == "program_rel"]
    n = rn[rn.score == "program_rel"].iloc[0]
    checks = {
        "mixtures: rho_state >= %.1f in every dataset" % s["mixtures_min_rho_state"]: bool((m.rho_state >= s["mixtures_min_rho_state"]).all()),
        "mixtures: |rho_caf| <= %.1f in every dataset" % s["mixtures_max_abs_rho_caf"]: bool((m.rho_caf.abs() <= s["mixtures_max_abs_rho_caf"]).all()),
        "natural: partial rho_state >= %.1f" % s["natural_min_rho_state"]: bool(n.partial_state >= s["natural_min_rho_state"]),
        "natural: |partial rho_caf| <= %.1f" % s["natural_max_abs_rho_caf"]: bool(abs(n.partial_caf) <= s["natural_max_abs_rho_caf"]),
    }
    pd.Series(checks).to_csv(OUT / "verdict.tsv", sep="\t", header=False)
    print("\nVERDICT (program_rel):")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")


# ---------------------------------------------------------------- figure
def figure(dm, rm, dn, rn):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig = plt.figure(figsize=(7.1, 2.3))
    axs = [fig.add_axes(r) for r in ([0.06, 0.25, 0.15, 0.62], [0.26, 0.25, 0.15, 0.62],
                                     [0.585, 0.25, 0.12, 0.62], [0.79, 0.25, 0.15, 0.62])]
    g = dm.groupby(["caf_frac", "cthrc1_frac"])[["program", "program_rel"]].mean().reset_index()
    for ax, col, title in [(axs[0], "program", "Raw program score"), (axs[1], "program_rel", "Relative state score")]:
        for c, cf in zip(SEQ5, DC["mixtures"]["caf_frac"]):
            h = g[g.caf_frac == cf]
            ax.plot(h.cthrc1_frac, h[col], color=c, lw=1.2, marker="o", ms=2.5, label=f"{cf:.0%}")
        ax.set_xlabel("CTHRC1+ share of CAFs")
        ax.set_title(title, loc="left", fontsize=7)
    axs[0].set_ylabel("Score (mean log2 CPM)")
    axs[0].legend(title="CAF share", fontsize=5.4, title_fontsize=5.4, loc="upper left", handlelength=1.0,
                  bbox_to_anchor=(0.0, 1.02), labelspacing=0.25, borderaxespad=0)
    axs[0].set_ylim(top=axs[0].get_ylim()[1] + 1.2)

    ax = axs[2]
    order = ["program", "program_rel", "Moffitt_activated", "Moffitt_activated_rel", "Dominguez_LRRC15",
             "Dominguez_LRRC15_rel", "pan_CAF"]
    m = rm.groupby("score")[["rho_state", "rho_caf"]].mean().reindex(order)
    y = np.arange(len(order))
    ax.scatter(m.rho_state, y - 0.13, s=14, c=BLUE, label="vs CTHRC1+ share", zorder=3)
    ax.scatter(m.rho_caf, y + 0.13, s=14, c=ORANGE, label="vs CAF share", zorder=3)
    ax.axvline(0, color=GRAY, lw=0.5)
    names = [LABEL.get(k.replace("_rel", ""), k) + (" (relative)" if k.endswith("_rel") else "") for k in order]
    ax.set_yticks(y, names, fontsize=5.6)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ (controlled mixtures,\nmean of 3 datasets)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.2, -0.32), ncol=2, fontsize=5.4, handletextpad=0.1)
    ax.set_title("What each score tracks", loc="left", fontsize=7)

    ax = axs[3]
    sca = ax.scatter(dn.cthrc1_frac, dn.program_rel, c=dn.caf_frac, cmap="Blues", s=10, edgecolors=INK2,
                     linewidths=0.2, vmin=0, vmax=dn.caf_frac.quantile(0.95))
    rr = rn.set_index("score").loc["program_rel"]
    ax.text(0.98, 0.03, f"partial ρ, state = {rr.partial_state:.2f}\npartial ρ, CAF share = {rr.partial_caf:.2f}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=5.4)
    ax.set_xlabel("True CTHRC1+ share of CAFs")
    ax.set_ylabel("Relative state score")
    cb = fig.colorbar(sca, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("CAF share", fontsize=5.6)
    cb.ax.tick_params(labelsize=5.2)
    cb.outline.set_visible(False)
    ax.set_title(f"Real samples (n={len(dn)})", loc="left", fontsize=7)
    fig.savefig(PANELS / "decoupling.png")
    fig.savefig(PANELS / "decoupling.pdf")


def main():
    full, caf = load()
    dn, rn = natural(full, caf)
    dm, rm = mixtures(full, caf)
    verdict(rn, rm)
    figure(dm, rm, dn, rn)


if __name__ == "__main__":
    main()
