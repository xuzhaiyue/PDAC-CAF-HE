"""GEO 生存队列构建（hemorph 环境）。规则见 config/fig1.yaml 的 fig2_meta（计算生存关联前已冻结）。

每个队列：读取 series matrix（只读）→ 探针映射到基因（同基因取平均表达最高的探针）→ 原发肿瘤样本 →
解析 OS（统一为月）→ 与 fig2 相同的得分定义。
输出：results/fig2/geo/{cohort}_expr.tsv.gz、_scores.tsv、_clinical.tsv
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fig2_01_build_cohorts import scores

PROJ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((PROJ / "config" / "fig1.yaml").read_text())
FM = CFG["fig2_meta"]
GEO = Path(FM["geo_dir"])
OUT = PROJ / "results" / "fig2" / "geo"
OUT.mkdir(parents=True, exist_ok=True)


def read_matrix(gse: str, channel: int):
    lines = gzip.open(GEO / f"{gse}_series_matrix.txt.gz", "rt", errors="replace").read().splitlines()
    ids = next(l for l in lines if l.startswith("!Sample_geo_accession")).split("\t")[1:]
    ids = [i.strip('"') for i in ids]
    char = {i: {} for i in ids}
    raw = {i: [] for i in ids}
    tag = f"!Sample_characteristics_ch{channel}"
    for l in lines:
        if l.startswith(tag) or (channel == 1 and l.startswith("!Sample_characteristics_ch1")):
            for i, v in zip(ids, l.split("\t")[1:]):
                v = v.strip('"')
                raw[i].append(v)
                if ": " in v:
                    k, val = v.split(": ", 1)
                    char[i][k.strip()] = val.strip()
    b = lines.index("!series_matrix_table_begin")
    e = lines.index("!series_matrix_table_end")
    X = pd.read_csv(io.StringIO("\n".join(lines[b + 1:e])), sep="\t", index_col=0)
    X.index = X.index.astype(str).str.strip('"')
    return X.apply(pd.to_numeric, errors="coerce"), pd.DataFrame(char).T, raw


def to_genes(X, probe_map):
    if probe_map is None:
        G = X.copy()
    else:
        pm = Path(probe_map) if Path(probe_map).is_absolute() else GEO / probe_map
        m = pd.read_csv(pm, sep="\t", dtype=str).dropna()
        m = m[m.symbol.str.len() > 0].set_index("probe").symbol
        G = X.loc[X.index.intersection(m.index)].copy()
        G["gene"] = m.reindex(G.index).values
        G["mean"] = G.drop(columns="gene").mean(1)
        G = G.sort_values("mean", ascending=False).drop_duplicates("gene").set_index("gene").drop(columns="mean")
    G = G[~G.index.duplicated()]
    return G.dropna(how="all")


def main():
    for gse, spec in FM["cohorts"].items():
        X, ch, raw = read_matrix(gse, spec.get("channel", 1))
        keep = ch.index
        if "keep" in spec:
            keep = [i for i in ch.index if spec["keep"] in raw[i]]
        c = ch.loc[keep]
        t = pd.to_numeric(c[spec["time"]], errors="coerce")
        if spec["time_unit"] == "year":
            t = t * 12
        ev_raw = c[spec["event"]].astype(str).str.upper()
        ev = ev_raw.map({"1": 1, "0": 0, "DEAD": 1, "ALIVE": 0, "DECEASED": 1, "LIVING": 0}).astype(float)
        clin = pd.DataFrame({"OS_time": t, "OS": ev}, index=keep)
        if "Stage" in c:
            clin["stage_raw"] = c["Stage"]
        clin = clin.dropna(subset=["OS_time", "OS"])
        G = to_genes(X[clin.index], spec["probe_map"])
        G.to_csv(OUT / f"{gse}_expr.tsv.gz", sep="\t", float_format="%.4f")
        s, cov = scores(G, gse)
        s.to_csv(OUT / f"{gse}_scores.tsv", sep="\t", float_format="%.5f")
        clin.to_csv(OUT / f"{gse}_clinical.tsv", sep="\t")
        miss = cov[cov.n_present < cov.n_genes]
        print(f"{gse}: {len(clin)} tumours with OS ({int(clin.OS.sum())} events), {G.shape[0]} genes; "
              f"event values {ev_raw.value_counts().to_dict()}")
        if len(miss):
            print("   missing genes:", "; ".join(f"{r.set}: {r.missing}" for r in miss.itertuples()))


if __name__ == "__main__":
    main()
