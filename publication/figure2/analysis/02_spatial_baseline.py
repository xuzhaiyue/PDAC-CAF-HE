"""Calculate Figure 2d example correlations and Figure 2e source summaries."""
import argparse
import gzip
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

EXAMPLES = ("D1", "GSM6505134", "GSM8452849")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", type=Path, required=True, help="A12 slide_rho.tsv")
    parser.add_argument("--regions", type=Path, required=True, help="Directory containing the Figure 2d region tables")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sections = pd.read_csv(args.sections, sep="\t")
    if len(sections) != 70 or sections["sample"].nunique() != 70 or sections["patient"].nunique() != 25:
        raise ValueError("Expected 70 unique sections across 25 patient groups")
    examples = []
    for sample in EXAMPLES:
        with gzip.open(args.regions / f"panel_d_{sample}_regions.csv.gz", "rt") as stream:
            regions = pd.read_csv(stream)
        mask = regions["caf_rich"].astype(str).str.lower()
        if not mask.isin(["true", "false"]).all():
            raise ValueError(f"Invalid CAF-rich mask in {sample}")
        eligible = regions.loc[mask.eq("true"), ["truth", "pred"]]
        rho = spearmanr(eligible["truth"], eligible["pred"], nan_policy="raise").statistic
        examples.append({"sample": sample, "n_regions": len(eligible), "rho": rho})
    source = sections.groupby("source")["S0_128"].agg(n_sections="size", median_rho="median")
    args.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(examples).to_csv(args.out / "panel_d.csv", index=False)
    sections.to_csv(args.out / "panel_e_sections.csv", index=False)
    source.to_csv(args.out / "panel_e_source_summary.csv")


if __name__ == "__main__":
    main()
