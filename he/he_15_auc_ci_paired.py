"""Bootstrap CIs and paired comparisons of patient-level AUC / Spearman (hemorph environment). Descriptive; does not change any fixed conclusion.

Models (CPTAC external predictions; TCGA out-of-fold, A1 / A2 / pan-cancer models only):
  pancancer_finetuned_all8 (A8), pancancer_zero_shot (A8), A2 I1, A2 B2, A1 ridge (self-extracted features)
Labels: A1 definition (gene z mean; residual / subtraction); CPTAC uses its own gene statistics.
Method: one set of 2000 patient resamples; 95% CIs of AUC and Spearman per model and CIs of paired differences against pancancer_finetuned_all8;
      DeLong paired test for AUC.
Output: results/he/A8/ci_paired.tsv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import he_03_ridge_gc_protocol as A

HE = A.HE
B, SEED = 2000, A.SEED


def delong(y, p1, p2):
    """Paired DeLong test for two correlated AUCs (Sun & Xu 2014 fast algorithm)."""
    y = np.asarray(y).astype(bool)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    def v(p):
        P, Nn = p[pos], p[neg]
        V10 = np.array([(np.sum(x > Nn) + 0.5 * np.sum(x == Nn)) / len(Nn) for x in P])
        V01 = np.array([(np.sum(P > x) + 0.5 * np.sum(P == x)) / len(P) for x in Nn])
        return V10.mean(), V10, V01
    a1, v10a, v01a = v(np.asarray(p1)); a2, v10b, v01b = v(np.asarray(p2))
    S10 = np.cov(np.vstack([v10a, v10b])); S01 = np.cov(np.vstack([v01a, v01b]))
    S = S10 / len(pos) + S01 / len(neg)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return a1 - a2, 2 * stats.norm.sf(abs(z))


def labels(cohort):
    if cohort == "CPTAC":
        C = pd.read_csv(HE / "cptac_targets.tsv", sep="\t", index_col=0)
        cp = [p for p in C.index if (A.CACHE / "cptac" / f"{p}.npy").exists()]
        E = pd.read_csv(A.PROJ / "results/fig2/cptac_expr.tsv.gz", sep="\t", index_col=0)[cp].T
    else:
        T = pd.read_csv(HE / "tcga_targets.tsv", sep="\t", index_col=0)
        cp = [p for p in T.index if (HE / "cache_own/tcga" / f"{p}.npy").exists()]
        E = pd.read_csv(A.PROJ / "results/fig2/tcga_expr.tsv.gz", sep="\t", index_col=0)[cp].T
    y, _ = A.make_targets(E, E.iloc[:2]); y.index = cp
    return y


def preds(cohort, target):
    out = {}
    f = pd.read_csv(HE / f"A8/patient_preds_finetuned_{cohort}.tsv", sep="\t").set_index("patient").pred_state
    out["pancancer_finetuned_all8"] = f
    z = pd.read_csv(HE / "A8/patient_preds.tsv", sep="\t")
    out["pancancer_zero_shot"] = z[z.cohort == cohort].set_index("patient").pred_state
    if cohort == "CPTAC":
        a2 = pd.read_csv(HE / "A2/pred_cptac.tsv", sep="\t", index_col=0)
        for m in ["I1", "B2"]:
            out[f"A2 {m}"] = a2[a2.model == m][target]
        out["A1 ridge"] = pd.read_csv(HE / "A1_ownTCGA/pred_cptac.tsv", sep="\t", index_col=0)[target]
    else:
        a2 = pd.read_csv(HE / "A2/oof_tcga.tsv", sep="\t")
        for m in ["I1", "B2"]:
            out[f"A2 {m}"] = a2[(a2.model == m) & (a2.target == target)].set_index("patient").y_pred
        a1 = pd.read_csv(HE / "A1_ownTCGA/oof_tcga.tsv", sep="\t")
        out["A1 ridge"] = a1[a1.target == target].groupby("patient").y_pred.mean()
    return out


def main():
    rows = []
    for cohort in ["CPTAC", "TCGA"]:
        y_all = labels(cohort)
        for target in ["program_resid", "program_sub"]:
            P = preds(cohort, target)
            ids = sorted(set.intersection(*[set(v.index) for v in P.values()]) & set(y_all.index))
            y = y_all.loc[ids, target].to_numpy(); yb = y > np.median(y)
            pv = {k: v.loc[ids].to_numpy() for k, v in P.items()}
            rng = np.random.default_rng(SEED)
            boots = [rng.integers(0, len(ids), len(ids)) for _ in range(B)]
            ref = "pancancer_finetuned_all8"
            for k, p in pv.items():
                aucs = [roc_auc_score(yb[b], p[b]) for b in boots if 0 < yb[b].sum() < len(b)]
                rhos = [stats.spearmanr(y[b], p[b])[0] for b in boots]
                r = {"cohort": cohort, "target": target, "model": k, "n": len(ids),
                     "AUC": roc_auc_score(yb, p), "AUC_lo": np.quantile(aucs, .025), "AUC_hi": np.quantile(aucs, .975),
                     "rho": stats.spearmanr(y, p)[0], "rho_lo": np.quantile(rhos, .025), "rho_hi": np.quantile(rhos, .975)}
                if k != ref:
                    d_auc = [roc_auc_score(yb[b], pv[ref][b]) - roc_auc_score(yb[b], p[b]) for b in boots if 0 < yb[b].sum() < len(b)]
                    d_rho = [stats.spearmanr(y[b], pv[ref][b])[0] - stats.spearmanr(y[b], p[b])[0] for b in boots]
                    dd, p_delong = delong(yb, pv[ref], p)
                    r.update({"dAUC_vs_finetuned": -dd, "dAUC_finetuned_minus_this_lo": np.quantile(d_auc, .025),
                              "dAUC_finetuned_minus_this_hi": np.quantile(d_auc, .975), "p_delong": p_delong,
                              "drho_finetuned_minus_this_lo": np.quantile(d_rho, .025),
                              "drho_finetuned_minus_this_hi": np.quantile(d_rho, .975),
                              "p_boot_drho_le0": float(np.mean(np.array(d_rho) <= 0))})
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(HE / "A8/ci_paired.tsv", sep="\t", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    show = t.assign(AUC_CI=t.apply(lambda r: f"{r.AUC:.3f} ({r.AUC_lo:.3f}-{r.AUC_hi:.3f})", axis=1),
                    rho_CI=t.apply(lambda r: f"{r.rho:.3f} ({r.rho_lo:.3f}-{r.rho_hi:.3f})", axis=1),
                    dAUC_CI=t.apply(lambda r: "" if pd.isna(r.get("p_delong")) else
                                    f"{-r.dAUC_vs_finetuned:+.3f} ({r.dAUC_finetuned_minus_this_lo:+.3f} to {r.dAUC_finetuned_minus_this_hi:+.3f}) p={r.p_delong:.3f}", axis=1),
                    drho_CI=t.apply(lambda r: "" if pd.isna(r.get("p_delong")) else
                                    f"({r.drho_finetuned_minus_this_lo:+.3f} to {r.drho_finetuned_minus_this_hi:+.3f})", axis=1))
    print(show[["cohort", "target", "model", "n", "AUC_CI", "rho_CI", "dAUC_CI", "drho_CI"]].to_string(index=False))


if __name__ == "__main__":
    main()
