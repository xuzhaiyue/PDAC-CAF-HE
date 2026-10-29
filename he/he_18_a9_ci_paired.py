"""Amendment A9: bootstrap 95% CIs and paired comparisons of patient-level AUC / Spearman (hemorph environment). Rules in A9 of config/fig1.yaml (fixed).

Reference model: A9 all70 (fine-tuned on all 70 PDAC spatial sections); comparators: A9 ablations (own4_only, primary_only), A8 finetuned_all8,
A8 zero-shot, A2 I1, A2 B2, A1 ridge. Method as in he_15 (one set of 2000 patient resamples; DeLong paired test for AUC).
Output: results/he/A9/ci_paired.tsv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import he_15_auc_ci_paired as C

HE, B, SEED = C.HE, C.B, C.SEED
REF = "A9_all70"


def preds(cohort, target):
    out = {f"A9_{t}": pd.read_csv(HE / f"A9/patient_preds_{cohort}_{t}.tsv", sep="\t").set_index("patient").pred_state
           for t in ["all70", "own4_only", "primary_only"]}
    return out | C.preds(cohort, target)


def main():
    os.nice(19)
    rows = []
    for cohort in ["CPTAC", "TCGA"]:
        y_all = C.labels(cohort)
        for target in ["program_resid", "program_sub"]:
            P = preds(cohort, target)
            ids = sorted(set.intersection(*[set(v.index) for v in P.values()]) & set(y_all.index))
            y = y_all.loc[ids, target].to_numpy(); yb = y > np.median(y)
            pv = {k: v.loc[ids].to_numpy() for k, v in P.items()}
            rng = np.random.default_rng(SEED)
            boots = [b for b in (rng.integers(0, len(ids), len(ids)) for _ in range(B)) if 0 < yb[b].sum() < len(b)]
            for k, p in pv.items():
                aucs = [roc_auc_score(yb[b], p[b]) for b in boots]
                rhos = [stats.spearmanr(y[b], p[b])[0] for b in boots]
                r = {"cohort": cohort, "target": target, "model": k, "n": len(ids),
                     "AUC": roc_auc_score(yb, p), "AUC_lo": np.quantile(aucs, .025), "AUC_hi": np.quantile(aucs, .975),
                     "rho": stats.spearmanr(y, p)[0], "rho_lo": np.quantile(rhos, .025), "rho_hi": np.quantile(rhos, .975)}
                if k != REF:
                    d_auc = [roc_auc_score(yb[b], pv[REF][b]) - roc_auc_score(yb[b], p[b]) for b in boots]
                    d_rho = [stats.spearmanr(y[b], pv[REF][b])[0] - stats.spearmanr(y[b], p[b])[0] for b in boots]
                    dd, p_delong = C.delong(yb, pv[REF], p)
                    r.update({"dAUC_all70_minus_this": dd, "dAUC_lo": np.quantile(d_auc, .025), "dAUC_hi": np.quantile(d_auc, .975),
                              "p_delong": p_delong, "drho_lo": np.quantile(d_rho, .025), "drho_hi": np.quantile(d_rho, .975)})
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(HE / "A9/ci_paired.tsv", sep="\t", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    show = t.assign(AUC_CI=t.apply(lambda r: f"{r.AUC:.3f} ({r.AUC_lo:.3f}-{r.AUC_hi:.3f})", axis=1),
                    rho_CI=t.apply(lambda r: f"{r.rho:.3f} ({r.rho_lo:.3f}-{r.rho_hi:.3f})", axis=1),
                    all70_minus=t.apply(lambda r: "" if pd.isna(r.get("p_delong")) else
                                        f"{r.dAUC_all70_minus_this:+.3f} ({r.dAUC_lo:+.3f} to {r.dAUC_hi:+.3f}) p={r.p_delong:.3f}", axis=1))
    print(show[["cohort", "target", "model", "n", "AUC_CI", "rho_CI", "all70_minus"]].to_string(index=False))


if __name__ == "__main__":
    main()
