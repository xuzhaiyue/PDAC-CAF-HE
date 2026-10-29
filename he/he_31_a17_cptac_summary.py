"""Amendment A17 part 1 (descriptive): TCGA / CPTAC follow-up, power and between-cohort HR difference, tabulated for the supplementary tables.
Patient matching and filtering as in he_19_a9_prognosis.R. Run inside he/: nice -n 19 ~/miniforge3/envs/hemorph/bin/python he_31_a17_cptac_summary.py
"""
from __future__ import annotations

from math import erf, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
A9 = PROJ / "results/he/A9"
OUT = PROJ / "results/he/A17/2026-10-28_run01"
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))  # noqa: E731


def power(hr, deaths):
    z = log(hr) * sqrt(deaths)                # continuous per-SD predictor: SE(logHR) ≈ 1/√D
    return Phi(z - 1.96) + Phi(-z - 1.96)


def main():
    per = pd.read_csv(A9 / "prognosis_per_cohort.tsv", sep="\t")
    per = per[(per.source == "A9_all70") & (per.model == "univariable")].set_index("cohort")
    ci = pd.read_csv(A9 / "ci_paired.tsv", sep="\t")
    rows = []
    for co in ["TCGA", "CPTAC"]:
        cl = pd.read_csv(PROJ / f"results/fig2/{co.lower()}_clinical.tsv", sep="\t", index_col=0)
        pr = pd.read_csv(A9 / f"patient_preds_{co}_all70.tsv", sep="\t").set_index("patient")
        d = cl.join(pr.pred_state, how="inner"); d = d[d.OS_time.notna() & (d.OS_time > 0) & d.OS.notna()]
        m = (d.OS_time / 30.4375).to_numpy(); e = d.OS.astype(int).to_numpy()
        s, med = 1.0, np.nan                  # reverse KM: censoring as the event
        for t in np.sort(np.unique(m[e == 0])):
            s *= 1 - ((m == t) & (e == 0)).sum() / (m >= t).sum()
            if np.isnan(med) and s <= .5:
                med = t
        r = per.loc[co]; assert int(r.n) == len(d) and int(r.events) == e.sum(), co
        rho = ci[(ci.cohort == co) & (ci.target == "program_resid") & (ci.model == "A9_all70")].rho.iloc[0]
        rows.append(dict(cohort=co, n=len(d), deaths=int(e.sum()), death_fraction=e.mean(), median_os_time_months=np.median(m),
                         max_time_months=m.max(), reverse_km_median_followup_months=med, stage_late_fraction=d.stage_late.mean(),
                         A9_pred_vs_measured_rho=rho, A9_HR=r.HR, A9_HR_lo=r.lo, A9_HR_hi=r.hi, A9_p=r.p,
                         power_HR_1_15=power(1.15, e.sum()), power_HR_1_22=power(1.22, e.sum())))
    t = pd.DataFrame(rows)
    z = (per.loc["TCGA", "logHR"] - per.loc["CPTAC", "logHR"]) / sqrt(per.loc["TCGA", "se"] ** 2 + per.loc["CPTAC", "se"] ** 2)
    t["TCGA_vs_CPTAC_z"] = z; t["TCGA_vs_CPTAC_p"] = 2 * (1 - Phi(abs(z)))
    t.to_csv(OUT / "part1_summary.tsv", sep="\t", index=False); print(t.round(3).T)


if __name__ == "__main__":
    main()
