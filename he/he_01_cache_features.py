"""H&E feature cache (hemorph environment). Rules in the he_model section of config/fig1.yaml (fixed).

Reads pre-extracted UNI2-h features (read only) and randomly samples <= tiles_per_slide tiles per slide (fixed seed),
merged per patient (a patient may have several slides), stored as float16: results/he/cache/{cohort}/{patient}.npy (tiles × 1536)
Also stores coords (tile coordinates, for plotting) and index.tsv (patient, slide, tile count).
Bulk targets: the bulk score table plus the StRoMA 3-gene score (mean log expression of COL12A1, LOX, LOXL2).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

PROJ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((PROJ / "config" / "fig1.yaml").read_text())
HM = CFG["he_model"]
OUT = PROJ / "results" / "he"
CACHE = OUT / "cache"
SEED = CFG["random_seed"]


def patient_of(name: str, cohort: str) -> str:
    return name[:12] if cohort == "tcga" else name[:9]


def cache_cohort(cohort: str, patients: set[str]):
    src = Path(HM["features"][cohort])
    d = CACHE / cohort
    d.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.glob("*.h5") if not f.name.startswith("._"))
    by_pat: dict[str, list[Path]] = {}
    for f in files:
        p = patient_of(f.name, cohort)
        if p in patients:
            by_pat.setdefault(p, []).append(f)
    rng = np.random.default_rng(SEED)
    rows = []
    for p, fs in sorted(by_pat.items()):
        feats, coords = [], []
        for f in fs:
            with h5py.File(f, "r") as h:
                F = h["features"]
                n = F.shape[1]
                take = np.sort(rng.choice(n, min(n, HM["features"]["tiles_per_slide"]), replace=False))
                feats.append(F[0][take].astype(np.float16))
                coords.append(np.c_[h["coords"][0][take], np.full(len(take), len(coords))])
                rows.append({"cohort": cohort, "patient": p, "slide": f.name, "n_tiles_total": n, "n_tiles_used": len(take)})
        np.save(d / f"{p}.npy", np.concatenate(feats))
        np.save(d / f"{p}_coords.npy", np.concatenate(coords))
    idx = pd.DataFrame(rows)
    idx.to_csv(d / "index.tsv", sep="\t", index=False)
    print(f"{cohort}: {idx.patient.nunique()} patients, {len(idx)} slides, "
          f"{idx.n_tiles_used.sum():,} tiles cached (median per patient "
          f"{idx.groupby('patient').n_tiles_used.sum().median():.0f})")


def targets(cohort: str) -> pd.DataFrame:
    s = pd.read_csv(PROJ / "results" / "fig2" / f"{cohort}_scores.tsv", sep="\t", index_col=0)
    e = pd.read_csv(PROJ / "results" / "fig2" / f"{cohort}_expr.tsv.gz", sep="\t", index_col=0)
    g = [x for x in HM["targets"]["benchmark"]["StRoMA_3gene"] if x in e.index]
    s["StRoMA_3gene"] = e.loc[g].mean(0).reindex(s.index)
    return s


def main():
    for cohort in ["tcga", "cptac"]:
        t = targets(cohort)
        t.to_csv(OUT / f"{cohort}_targets.tsv", sep="\t", float_format="%.5f")
        cache_cohort(cohort, set(t.index))


if __name__ == "__main__":
    main()
