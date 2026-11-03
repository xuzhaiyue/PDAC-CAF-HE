"""Reconstruct Figure 3 numerical tables from archived spatial predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ARMS = ["S0", "S1", "S2"]
CASES = ["D1", "GSM6505134", "GSM8452849"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a10", "a12", "a15", "regions", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def read(path, **kwargs):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", **kwargs)

    slides = read(args.a10 / "slide_rho.tsv")
    patients = read(args.a10 / "patient_rho.tsv").set_index("patient")
    if len(slides) != 70 or len(patients) != 25 or not patients["complete"].eq(True).all():
        raise ValueError("Expected 70 sections and 25 complete patient groups")
    pd.testing.assert_frame_equal(slides.groupby("patient")[ARMS].mean(),
                                  patients[ARMS].sort_index(), atol=1e-12, rtol=1e-12)
    tests = read(args.a10 / "paired_tests.tsv").set_index("contrast")
    field = read(args.a12 / "patient_rho.tsv").set_index("patient")
    field_tests = read(args.a12 / "paired_tests.tsv").set_index(["scope", "contrast"])
    stain = read(args.a15 / "patient_rho.tsv").set_index("patient")
    stain_tests = read(args.a15 / "paired_tests.tsv").set_index("contrast")
    public = field.loc[stain.index]
    comparisons = [
        ("Original inputs", "A10", patients.S1 - patients.S0, tests.loc["S1-S0"]),
        ("Larger tile field", "A12", field.S1_128 - field.S0_128,
         field_tests.loc[("all70_25patients", "S1_128-S0_128")]),
        ("Public, original stain", "A12", public.S1_128 - public.S0_128,
         field_tests.loc[("public66_21patients", "S1_128-S0_128")]),
        ("Public, TCGA stain", "A15", stain.S1_norm - stain.S0_norm,
         stain_tests.loc["norm_S1_minus_S0"]),
        ("PDAC − pan-cancer", "A10", patients.S2 - patients.S1, tests.loc["S2-S1"]),
        ("PDAC − bulk", "A10", patients.S2 - patients.S0, tests.loc["S2-S0"]),
    ]
    forest = []
    for label, analysis, delta, saved in comparisons:
        median = float(saved.get("median_paired_delta", saved.get("median_delta")))
        np.testing.assert_allclose(delta.median(), median, atol=1e-12, rtol=1e-12)
        forest.append(dict(row=label, analysis=analysis, n_patients=len(delta),
                           median_delta=median, lo=saved.lo, hi=saved.hi,
                           p_wilcoxon=saved.p_wilcoxon, n_improved=int((delta > 0).sum())))

    maps = []
    for case in CASES:
        regions = read(args.regions / f"panel_c_{case}_regions.csv.gz")
        if not regions.caf_rich.isin([True, False]).all():
            raise ValueError(f"Invalid CAF-rich mask for {case}")
        eligible = regions.loc[regions.caf_rich.eq(True)]
        for arm in ARMS:
            rho = spearmanr(eligible.truth, eligible[arm], nan_policy="raise").statistic
            maps.append(dict(sample=case, arm=arm, rho=rho, n_regions=len(eligible)))

    args.out.mkdir(parents=True, exist_ok=False)
    patients.to_csv(args.out / "panel_b_patients.csv")
    patients[ARMS].median().rename("median_rho").to_csv(args.out / "panel_b_medians.csv")
    pd.DataFrame(maps).to_csv(args.out / "panel_c_map_rho.csv", index=False)
    pd.DataFrame(forest).to_csv(args.out / "panel_d_paired_deltas.csv", index=False)
    slides.to_csv(args.out / "panel_e_sections.csv", index=False)
    slides.groupby("source")[ARMS].median().to_csv(args.out / "panel_e_source_medians.csv")
    (args.out / "input_hashes.json").write_text(json.dumps(inputs, indent=2) + "\n")


if __name__ == "__main__":
    main()
