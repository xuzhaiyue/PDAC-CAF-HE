"""Recompute the Figure 2f paired S0 comparison from archived A15 slide predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides", type=Path, required=True, help="A15 slide_rho.tsv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    slides = pd.read_csv(args.slides, sep="\t")
    columns = ["S0_orig", "S0_norm"]
    if slides["patient"].isna().any() or not np.isfinite(slides[columns].to_numpy()).all():
        raise ValueError("Expected complete patient IDs and paired section correlations")
    patients = slides.groupby("patient")[columns].mean()
    if len(patients) != 21:
        raise ValueError("Expected 21 paired public patient groups")
    delta = (patients["S0_norm"] - patients["S0_orig"]).to_numpy()
    indices = np.random.default_rng(20261021).integers(0, len(delta), (2000, len(delta)))
    lo, hi = np.quantile(np.median(delta[indices], axis=1), [0.025, 0.975])
    pvalue = 1.0 if np.all(delta == 0) else wilcoxon(
        delta, alternative="two-sided", method="approx", zero_method="wilcox"
    ).pvalue
    summary = pd.DataFrame([dict(contrast="S0_intervention", n_patients=len(delta),
                                 median_delta=np.median(delta), lo=lo, hi=hi,
                                 p_wilcoxon=pvalue, n_improved=int((delta > 0).sum()))])
    args.out.mkdir(parents=True, exist_ok=False)
    patients.to_csv(args.out / "panel_f_patients.tsv", sep="\t")
    summary.to_csv(args.out / "panel_f_paired.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
