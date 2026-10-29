"""修订 A17（探索性，2026-10-28 用户要求）第 2 部分：D1 中 CAF 状态高的区域周围是什么。

区域、相对状态 rel、CAF-rich 掩膜按 he_10.region_labels 重建，并与 A10 labels_D1_150 逐项断言一致；
细胞类型用用户已有注释（VISIUMHDdata/HDD1_count/outs/results/cell_type.loupe_category.csv），不重新注释。
邻域窗口 = 3×3 个 150 µm 区域；CAF-rich 区域内 Spearman(rel, 指标) 与 Spearman(CAF 数量, 指标)，空间块 bootstrap 95% CI。
在 he/ 目录内运行：nice -n 19 ~/miniforge3/envs/hemorph/bin/python he_30_a17_d1_neighbourhood.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from hemorph import config as C  # noqa: E402
from hemorph import expression as E  # noqa: E402
from hemorph import io_visiumhd as IO  # noqa: E402

OUT = PROJ / "results/he/A17/2026-10-28_run01"
ANN = Path("/Users/Zhaiyue/VISIUMHDdata/HDD1_count/outs/results/cell_type.loupe_category.csv")
A10_LAB = PROJ / "results/he/A10/2026-10-21_run01/labels_D1_150.tsv"
SIZE, BLOCK, NBOOT, SEED, MIN_MAL = 150, 4, 2000, 20261028, 20
MALIGNANT = ["Basal-like Malignant", "Classical Malignant", "Proliferating Malignant"]


def main():
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    assert not (OUT / "correlations.tsv").exists(), "不覆盖已有结果"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = C.load_config(); smp = next(x for x in C.samples(cfg) if x.sample_id == "D1")
    a = IO.read_cell_counts(smp); nuc = IO.read_polygons(smp, "nucleus")
    a = a[a.obs_names.isin(nuc.index)]
    X = sp.csr_matrix(a.X); var = list(a.var_names); vix = {g: i for i, g in enumerate(var)}

    # ---- 重建 A10 区域标签（与 he_10.region_labels 相同）
    PROG = yaml.safe_load((PROJ / "results/fig1/signature/programs.yaml").read_text())
    PUB = yaml.safe_load((PROJ / "config/published_signatures.yaml").read_text())
    progs = C.load_programs(cfg, a.var_names)
    excl = set(progs["acinar"]) | set(progs["endocrine"]) | set(E.ACINAR_EXTRA)
    nonac = np.array([g not in excl for g in var]); acinar_genes = sorted(excl & set(var))
    xy = nuc.loc[a.obs_names, ["x", "y"]].to_numpy()
    ix, iy = E.region_index(xy[:, 0], xy[:, 1], smp.mpp, SIZE)
    key = ix.astype(np.int64) * 1_000_000 + iy
    keys, P = E.region_pseudobulk(X, key)
    ncell = np.bincount(np.unique(key, return_inverse=True)[1])
    u = E.program_umis(P, var, {"prog": PROG["CTHRC1_matrix_CAF"], "pan": PROG["pan_CAF"], "aci": acinar_genes})
    tot = np.asarray(P[:, nonac].sum(1)).ravel()
    aci = u["aci"].to_numpy() / np.maximum(np.asarray(P.sum(1)).ravel(), 1)
    s_ = E.log_sum_cp10k(u[["prog", "pan"]], tot)
    caf = (ncell >= 5) & (aci <= 0.2) & (u["pan"].to_numpy() >= 10 * (SIZE / 100) ** 2)
    reg = pd.DataFrame({"key": keys, "rel": (s_.prog - s_.pan).to_numpy(), "amount": s_.pan.to_numpy(), "caf_rich": caf}).set_index("key")
    ref = pd.read_csv(A10_LAB, sep="\t").set_index("key")
    assert reg.index.equals(ref.index) and np.allclose(reg.rel, ref.rel) and (reg.caf_rich == ref.caf_rich).all(), "与 A10 标签不一致"

    # ---- 用户注释
    ann = pd.read_csv(ANN).set_index("Barcode").Category
    ct = pd.Series(a.obs_names, index=a.obs_names).map(ann)
    cover = float(ct.notna().mean()); assert cover > .95, cover
    ct = ct.fillna("unannotated").to_numpy()
    types = sorted(set(ct) - {"Low_quality", "unannotated"})

    # ---- 3×3 窗口
    reg["ix"] = reg.index // 1_000_000; reg["iy"] = reg.index % 1_000_000
    cells = pd.DataFrame({"ix": ix, "iy": iy, "ct": ct})
    counts = cells.groupby(["ix", "iy", "ct"]).size().unstack(fill_value=0)
    mal = np.isin(ct, MALIGNANT)
    mkeys, MP = E.region_pseudobulk(X[mal], key[mal])
    mpos = {k: i for i, k in enumerate(mkeys)}
    sig = {"MPAS": PUB["MPAS"]["genes"], "basal": PUB["Moffitt_basal"]["genes"], "classical": PUB["Moffitt_classical"]["genes"]}
    genes = sorted({g for v in sig.values() for g in v if g in vix})
    gcols = [vix[g] for g in genes]
    rows, gmat = [], []
    for k, r in reg.iterrows():
        nb = [(r.ix + dx, r.iy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        c = counts.reindex(nb).fillna(0).sum()
        n_ok = c.drop([t for t in ["Low_quality", "unannotated"] if t in c]).sum()
        d = {"key": k, "n_cells_window": int(n_ok)}
        for t in types:
            d[f"frac_{t}"] = c.get(t, 0) / n_ok if n_ok else np.nan
        nb_b = c.get("Basal-like Malignant", 0); nb_c = c.get("Classical Malignant", 0); n_mal = sum(c.get(t, 0) for t in MALIGNANT)
        d["n_malignant_window"] = int(n_mal)
        d["basal_fraction_of_malignant"] = nb_b / (nb_b + nb_c) if (n_mal >= MIN_MAL and nb_b + nb_c > 0) else np.nan
        idx = [mpos[x * 1_000_000 + y] for x, y in nb if x * 1_000_000 + y in mpos]
        if n_mal >= MIN_MAL and idx:
            v = np.asarray(MP[idx].sum(0)).ravel(); totm = v.sum()
            gmat.append(np.log1p(1e4 * v[gcols] / max(totm, 1)))
        else:
            gmat.append(np.full(len(gcols), np.nan))
        rows.append(d)
    ctx = pd.DataFrame(rows).set_index("key")
    G = pd.DataFrame(np.vstack(gmat), index=ctx.index, columns=genes)
    Z = (G - G.mean()) / G.std(ddof=0).replace(0, np.nan)
    for name, gs in sig.items():
        gs = [g for g in gs if g in Z and Z[g].notna().any()]
        ctx[f"malignant_{name}"] = Z[gs].mean(axis=1, skipna=False)
        ctx[f"n_genes_{name}"] = len(gs)
    ctx["malignant_basal_minus_classical"] = ctx.malignant_basal - ctx.malignant_classical
    d = reg.join(ctx); d.to_csv(OUT / "regions.tsv", sep="\t")

    # ---- 相关与空间块 bootstrap
    q = d[d.caf_rich].copy(); q["block"] = (q.ix // BLOCK).astype(str) + "_" + (q.iy // BLOCK).astype(str)
    metrics = [f"frac_{t}" for t in types] + ["basal_fraction_of_malignant", "malignant_MPAS", "malignant_basal_minus_classical"]
    rng = np.random.default_rng(SEED); blocks = q.block.unique(); members = {b: np.where(q.block.to_numpy() == b)[0] for b in blocks}
    boot_idx = [np.concatenate([members[b] for b in rng.choice(blocks, len(blocks))]) for _ in range(NBOOT)]
    res = []
    for m in metrics:
        for xname in ["rel", "amount"]:
            ok = q[m].notna().to_numpy(); x, y = q[xname].to_numpy(), q[m].to_numpy()
            rho = stats.spearmanr(x[ok], y[ok]).statistic
            bs = []
            for bi in boot_idx:
                bi = bi[ok[bi]]
                if len(bi) > 10:
                    bs.append(stats.spearmanr(x[bi], y[bi]).statistic)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            res.append(dict(metric=m, predictor="state (rel)" if xname == "rel" else "amount (pan-CAF)", n_regions=int(ok.sum()),
                            spearman=rho, boot_lo=lo, boot_hi=hi, n_blocks=len(blocks)))
    res = pd.DataFrame(res); res.to_csv(OUT / "correlations.tsv", sep="\t", index=False)
    q["state_tertile"] = pd.qcut(q.rel, 3, labels=["low", "mid", "high"])
    q.groupby("state_tertile", observed=True)[metrics].median().T.to_csv(OUT / "tertile_medians.tsv", sep="\t")
    meta = dict(n_cells_nucleus_matched=int(a.n_obs), annotation_coverage=cover, n_regions=len(d), n_caf_rich=int(d.caf_rich.sum()),
                types=types, celltype_counts=pd.Series(ct).value_counts().to_dict(), sig_genes_used={k: int(ctx[f"n_genes_{k}"].iloc[0]) for k in sig})
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2, default=int))
    print(json.dumps(meta, indent=1, default=int)); pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
