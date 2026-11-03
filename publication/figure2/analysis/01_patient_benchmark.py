"""Assemble Figure 2b and 2c from the archived benchmark tables."""
import argparse
import csv
from pathlib import Path
from itertools import product

MODELS = ("Ridge (own TCGA feats)", "I1 additive (ours)", "B2 attention MIL")
COHORTS = ("TCGA CV", "CPTAC ext")
TARGETS = ("program_resid", "program_sub", "Moffitt_activated_sub", "Dominguez_LRRC15_sub")


def read_table(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return reader.fieldnames, list(reader)


def write_table(path, fields, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auc", type=Path, required=True)
    parser.add_argument("--proxies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fields, rows = read_table(args.auc)
    keys = set(product(MODELS, COHORTS, TARGETS))
    selected = [row for row in rows if (row["model"], row["cohort"], row["target"]) in keys]
    actual = {(row["model"], row["cohort"], row["target"]) for row in selected}
    if len(selected) != 24 or actual != keys:
        raise ValueError("Expected one AUC per model, cohort and target")
    proxy_fields, proxies = read_table(args.proxies)
    prefixes = ("acinar content", "stromal content", "immune content", "epithelial content")
    if any(sum(row["target"].startswith(prefix) for row in proxies) != 1 for prefix in prefixes):
        raise ValueError("Expected one row for each of the four tissue proxies")
    args.out.mkdir(parents=True, exist_ok=False)
    write_table(args.out / "panel_b.csv", fields, selected)
    write_table(args.out / "panel_c_source.csv", proxy_fields, proxies)
    write_table(args.out / "panel_c_displayed.csv", proxy_fields,
                [row for row in proxies if row["target"].startswith(prefixes)])


if __name__ == "__main__":
    main()
