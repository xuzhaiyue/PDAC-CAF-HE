"""Assemble archived Figure 5 tables and verify key results."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRICS = ["frac_Basal-like Malignant", "frac_TAMs (SPP1+)"]
TABLES = (
    "panel_a_decoupling", "panel_b_per_cohort", "panel_c_pooled",
    "panel_d_biology", "panel_e_prognosis", "panel_f_at_risk",
    "panel_f_km_CPTAC", "panel_f_km_TCGA", "panel_g_correlations", "panel_g_maps",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True, help="A17 regions.tsv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: args.tables / f"{name}.csv" for name in TABLES}
    data = {name: pd.read_csv(path) for name, path in paths.items()}
    pooled = data["panel_c_pooled"].query("set == 'program' and model == 'joint'").set_index("term")
    cohorts = data["panel_b_per_cohort"].query("score == 'program_rel' and model == 'joint' and term == 'state'")
    state = pooled.loc["state"]
    if cohorts.cohort.nunique() != 8 or not cohorts.HR.gt(1).all():
        raise ValueError("Expected eight cohort estimates above one")
    if [round(float(state[key]), 2) for key in ("HR", "lo", "hi")] != [1.15, 1.04, 1.27]:
        raise ValueError("Pooled programme estimate differs from the manuscript")

    regions = pd.read_csv(args.regions, sep="\t")
    maps = regions.loc[regions.caf_rich.eq(True)]
    correlations = data["panel_g_correlations"].query("predictor == 'state (rel)'").set_index("metric")
    spatial = {}
    for metric in METRICS:
        pair = maps[["rel", metric]].dropna()
        rho = spearmanr(pair.rel, pair[metric]).statistic
        saved = correlations.loc[metric]
        np.testing.assert_allclose(rho, saved.spearman, atol=1e-12, rtol=1e-12)
        if len(pair) != saved.n_regions:
            raise ValueError(f"Region count differs for {metric}")
        spatial[metric] = float(rho)
    if [round(spatial[key], 2) for key in METRICS] != [0.80, 0.65]:
        raise ValueError("Spatial correlations differ from the manuscript")

    for row in data["panel_f_at_risk"].itertuples(index=False):
        km = data[f"panel_f_km_{row.cohort}"]
        count = (km.high.eq(row.group == "High") & km.months.ge(row.months)).sum()
        if count != row.at_risk:
            raise ValueError(f"Risk count differs for {row.cohort} at {row.months} months")

    args.out.mkdir(parents=True, exist_ok=False)
    for name, path in paths.items():
        (args.out / f"{name}.csv").write_bytes(path.read_bytes())
    summary = dict(pooled_hr=float(state.HR), ci=[float(state.lo), float(state.hi)],
                   cohort_count=int(cohorts.cohort.nunique()), d1_spearman=spatial,
                   input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [*paths.values(), args.regions]})
    (args.out / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
