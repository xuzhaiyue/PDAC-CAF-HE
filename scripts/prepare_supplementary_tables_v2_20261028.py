"""Supplementary tables v2: reorganised by Figures 1–3 and the supplementary figures; transcribes existing fixed results only, no new tests.

Differences from v1 (results/tables/2026-10-22_run01):
- S1–S4 (single cell, genes, published gene sets, survival cohorts) reuse the v1 tables after checking that the source hashes recorded by v1 are unchanged
- S5 patient level: AUC summary, tissue proxies (Fig 2c), StRoMA replication (A4), A8 / A9 CIs
- S6 one row per section: preservation, site, corrected µm/px (GEO ×65/55 under the 65 µm convention), A10 / A12 / A15 / A14 columns and use in figures
- S7 new: patient-level ρ (25 patients) and all paired comparisons / correlation tests (A10, A12, A13, A15, A14; A16 merged automatically when complete)
Output results/tables/2026-10-28_run02/: xlsx + one TSV per table + input_manifest + verification.json.
scRNA environment (pandas + xlsxwriter + openpyxl); nice 19; reads TSVs only, no npz / raw data.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
V1 = PROJ / "results/tables/2026-10-22_run01"
OUT = Path(os.environ.get("T4_OUT", PROJ / "results/tables/2026-10-28_run02"))   # re-run into a new directory after A16 completes
HE = PROJ / "results/he"
A10, A12, A13 = HE / "A10/2026-10-21_run01", HE / "A12/2026-10-21_run01", HE / "A13/2026-10-22_run01"
A14, A15, A16 = HE / "A14/2026-10-27_run01", HE / "A15/2026-10-27_run01", HE / "A16/2026-10-28_run02"
A17 = HE / "A17/2026-10-28_run01"
K = 65.0 / 55.0                       # true GEO µm / old µm (Space Ranger spot_diameter_fullres = 65 µm)
GEO = {"GSE211895", "GSE274557", "GSE327056"}
SITE = {"Primary PDAC": "Primary", "Pancreatic cancer": "Primary", "pancreatic cancer tussue": "Primary", "tumor stroma": "Primary",
        "Peritoneal metastasis": "Peritoneal metastasis", "Liver metastasis": "Liver metastasis", "Lung metastasis": "Lung metastasis",
        "adjacent tumor tissue": "Non-tumour", "normal pancreatic tissue": "Non-tumour"}
FIG_USE = {"D1": "Fig 2d, 3c", "GSM6505134": "Fig 2d, 3c", "GSM8452849": "Fig 2d, 3c", "GSM9647220": "Supp Fig (transfer limits) a",
           "10x": "Supp Fig (transfer limits) b"}
SHEETS, SOURCES = [], {}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size < 50_000_000 else "metadata_only_large_file"


def read(p, **kw) -> pd.DataFrame:
    p = Path(p); SOURCES[str(p)] = {"path": str(p.relative_to(PROJ)), "bytes": p.stat().st_size, "sha256": sha(p)}
    return pd.read_csv(p, **{"sep": "\t", **kw})


def add(name, title, df, figure, source, note=""):
    assert len(name) <= 31 and df.columns.is_unique, name
    df = df.reset_index(drop=True); df.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)
    SHEETS.append(dict(sheet=name, title=title, figure=figure, n_rows=len(df), source=source, note=note, df=df))
    print(f"{name:24s} {len(df):5d} rows", flush=True)


def main():
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    assert not (OUT / "Supplementary_Tables.xlsx").exists(), "do not overwrite existing results"
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- S1–S4: reuse v1 after checking that the recorded sources are unchanged
    man1 = pd.read_csv(V1 / "input_manifest.tsv", sep="\t")
    def same(r):   # config only appends amendments: the bytes recorded by v1 must be a prefix of the current file
        b = Path(r.path).read_bytes()
        return hashlib.sha256(b).hexdigest() == r.sha256 or (r.path.endswith("config/fig1.yaml") and hashlib.sha256(b[:r.bytes]).hexdigest() == r.sha256)
    changed = [r.path for r in man1.itertuples() if r.sha256 != "metadata_only_large_file" and Path(r.path).exists() and not same(r)]
    missing = [r.path for r in man1.itertuples() if not Path(r.path).exists()]
    assert not changed and not missing, (changed, missing)
    v1 = [("S1_discovery", "Discovery datasets: final CAF reference", "Fig 1c–f",
           "Final cleaned/capped CAF reference (9,067 cells, 73 patients); patients can occur in more than one condition."),
          ("S1_input_scope", "Author-labelled CAF input by condition", "Fig 1",
           "Input scope before cleaning; counts differ from the final reference."),
          ("S1_external", "External single-cell validation", "Supp Fig S1–S5",
           "1 of 4 datasets met the frozen criteria; n_samples is not the number of unique patients."),
          ("S2_genes", "Nine-gene programme and four-gene CAF reference", "Fig 1g",
           "Frozen 9 + 4 genes; blank = statistic not applicable."),
          ("S2_dataset_DE", "Per-dataset screening statistics of selected genes", "Fig 1 (methods)",
           "Patient-matched pseudo-bulk statistics; the full candidate screen remains in results/fig1/signature/de_per_dataset.tsv."),
          ("S3_signatures", "Published gene sets and recorded sources", "Fig 2b, Supp",
           "Transcribed from config/published_signatures.yaml."),
          ("S4_cohorts", "Eight bulk survival cohorts", "Supp (survival)", "749 patients, 483 deaths."),
          ("S4_meta", "Eight-cohort meta-analysis", "Supp (survival)", "Added post hoc after TCGA/CPTAC alone were not significant; REML."),
          ("S4_meta_cohort", "Per-cohort meta-analysis inputs", "Supp (survival)", ""),
          ("S4_A11_pooled", "A11 state-within-abundance pooled results", "Supp (survival)",
           "Post hoc A11; attenuated after ESTIMATE and basal/classical adjustment."),
          ("S4_A11_cohort", "A11 per-cohort results", "Supp (survival)", "Includes low-event flags.")]
    for name, title, fig, note in v1:
        add(name, title, read(V1 / f"{name}.tsv"), fig, f"v1 table {V1.relative_to(PROJ)}/{name}.tsv (sources hash-checked)", note)

    # ---- S5: patient level
    auc = read(HE / "auc_summary.tsv")
    fig2b = read(PROJ / "results/manuscript_figures/2026-10-28_Figure2_v9/data/panel_b_24_auc_and_CI.csv", sep=",")
    key = ["model", "cohort", "target"]
    shown = set(map(tuple, fig2b[key].to_numpy()))
    auc["shown_in_Fig2b"] = [tuple(r) in shown for r in auc[key].to_numpy()]
    assert auc.shown_in_Fig2b.sum() == 24
    chk = auc[auc.shown_in_Fig2b].merge(fig2b, on=key, suffixes=("", "_fig"))
    assert np.allclose(chk.AUC, chk.AUC_fig) and np.allclose(chk.lo, chk.lo_fig) and np.allclose(chk.hi, chk.hi_fig)
    auc["prediction_source"] = np.where(auc.model.str.contains("pancancer", case=False, na=False), "spatial-only training; direct patient evaluation",
                                        np.where(auc.cohort.str.contains("TCGA", case=False, na=False), "TCGA out-of-fold", "CPTAC external"))
    add("S5_patient_AUC", "Patient-level histology model performance", auc, "Fig 2b", "results/he/auc_summary.tsv",
        "Historical consolidated summary with original rounding; shown_in_Fig2b marks the 24 estimates plotted.")
    add("S5_tissue_proxy", "Ridge-model accuracy for tissue-associated RNA proxies", read(HE / "positive_control_ceiling.tsv"), "Fig 2c",
        "results/he/positive_control_ceiling.tsv", "Point estimates only; the source has no confidence intervals. RNA-derived proxies, not pathology counts.")
    a4 = read(HE / "A4/perfold.tsv"); a4s = read(HE / "A4/summary.tsv")
    add("S5_StRoMA_folds", "StRoMA replication (A4), per fold", a4, "Supp Fig (benchmark) a, b", "results/he/A4/perfold.tsv",
        "honest = epoch chosen on inner validation; leaky_as_reported = epoch chosen on the test fold. auc_z0 = AUC for target > 0.")
    add("S5_StRoMA_summary", "StRoMA replication (A4), summary", a4s, "Supp Fig (benchmark) a, b", "results/he/A4/summary.tsv",
        "Published TCGA values: Spearman 0.655, AUC 0.811.")
    for tag in ["A8", "A9"]:
        d = read(HE / f"{tag}/ci_paired.tsv")
        add(f"S5_{tag}_CI", f"{tag}: spatial-only models evaluated on TCGA/CPTAC patients", d, "Supp",
            f"results/he/{tag}/ci_paired.tsv", "No model selection on TCGA/CPTAC; comparison direction follows the column names.")

    # ---- S5 (cont.): H&E prognosis and tumour-biology associations
    add("S5_prognosis_A7", "Prognosis of bulk-trained H&E predictions (A7)", read(HE / "A7/per_cohort.tsv"), "Fig 4e",
        "results/he/A7/per_cohort.tsv", "Per SD; univariable and age + stage multivariable; TCGA out-of-fold, CPTAC external.")
    add("S5_prognosis_A7_pooled", "Prognosis of bulk-trained H&E predictions, pooled (A7)", read(HE / "A7/pooled.tsv"), "Fig 4e",
        "results/he/A7/pooled.tsv", "TCGA + CPTAC fixed effect (primary) and random effects.")
    add("S5_prognosis_A9", "Prognosis of PDAC spatially trained predictions (A9)", read(HE / "A9/prognosis_per_cohort.tsv"), "Fig 4e, f",
        "results/he/A9/prognosis_per_cohort.tsv", "Models never saw TCGA/CPTAC data; all70 is shown in Fig 4.")
    add("S5_prognosis_A9_pooled", "Prognosis of PDAC spatially trained predictions, pooled (A9)", read(HE / "A9/prognosis_pooled.tsv"), "Fig 4e",
        "results/he/A9/prognosis_pooled.tsv", "High heterogeneity (I2) for all70; report random-effects P alongside fixed effect.")
    add("S4_biology", "Relative score versus tumour characteristics", read(PROJ / "results/fig2/D_biology.tsv"), "Fig 4d",
        "results/fig2/D_biology.tsv", "Exploratory; Spearman for continuous items, Wilcoxon / Kruskal for KRAS.")

    # ---- S6: sections
    s1 = read(V1 / "S6_spatial_samples.tsv")
    s10 = read(A10 / "slide_rho.tsv"); s12 = read(A12 / "slide_rho.tsv"); s15 = read(A15 / "slide_rho.tsv")
    m15 = read(A15 / "image_metrics_70.tsv"); d14 = read(A14 / "slide_domain.tsv")
    assert len(s10) == len(s12) == len(m15) == len(d14) == 70 and len(s15) == 66
    sec = s10.rename(columns={"S0": "rho_S0_bulk", "S1": "rho_S1_pancancer", "S2": "rho_S2_PDAC_LOPO"}).drop(columns=["status"])
    sec = sec.merge(s12[["sample", "S0_128", "S1_128", "S2_historical", "n_regions"]], on="sample", suffixes=("", "_A12"))
    assert (sec.n_regions == sec.n_regions_A12).all() and np.allclose(sec.rho_S2_PDAC_LOPO, sec.S2_historical)
    v1s = s1[s1["sample"] != "10x"].set_index("sample")
    assert np.allclose(sec.set_index("sample").rho_S2_PDAC_LOPO, v1s.rho.loc[sec["sample"]])  # v1 A9 ρ equals A10 S2
    sec = sec.drop(columns=["n_regions_A12", "S2_historical"]).rename(columns={"S0_128": "rho_S0_A12_128", "S1_128": "rho_S1_A12_128"})
    sec = sec.merge(s15[["sample", "S0_orig", "S0_norm", "S1_norm", "S1_orig"]], on="sample", how="left")
    pub = sec.S0_orig.notna()
    assert np.allclose(sec.loc[pub, "S0_orig"], sec.loc[pub, "rho_S0_A12_128"]) and np.allclose(sec.loc[pub, "S1_orig"], sec.loc[pub, "rho_S1_A12_128"])
    sec = sec.drop(columns=["S0_orig", "S1_orig"]).rename(columns={"S0_norm": "rho_S0_A15_stainnorm", "S1_norm": "rho_S1_A15_stainnorm"})
    sec = sec.merge(m15[["sample", "preservation", "stain_difference", "laplacian_orig"]], on="sample")
    sec = sec.merge(d14[["sample", "preservation", "distance"]].rename(columns={"distance": "A14_UNI2h_distance_to_TCGA"}), on="sample", suffixes=("", "_A14"))
    assert (sec.preservation == sec.preservation_A14).all(); sec = sec.drop(columns=["preservation_A14"])
    sec["site"] = sec.tissue.map(SITE); assert sec.site.notna().all()
    meta = v1s[["technology", "treatment_recorded", "n_spots", "n_image_tiles", "um_per_px", "region_definition", "clinical_patient_id",
                "patient_identity_note"]].rename(columns={"um_per_px": "um_per_px_legacy"})
    sec = sec.merge(meta, left_on="sample", right_index=True)
    geo = sec.source.isin(GEO); assert geo.sum() == 62
    sec["um_scale_factor"] = np.where(geo, K, 1.0)
    sec["um_per_px"] = sec.um_per_px_legacy * sec.um_scale_factor
    sec["A12_field_true_um"] = np.where(geo, 128 * K, 128.0)
    for sid, mpp in [("GSM6505134", 0.220), ("GSM8452849", 0.362), ("D1", 0.362)]:     # values in the Figure 2d legend
        assert round(float(sec.loc[sec["sample"] == sid, "um_per_px"].iloc[0]), 3) == mpp, sid
    sec["figure_use"] = sec["sample"].map(FIG_USE).fillna("Fig 2e, 3b/e (summary)")
    if (A16 / "COMPLETE").exists():
        s16 = read(A16 / "slide_rho.tsv")
        keep = [c for c in ["S0_true112", "S0_true128", "S1_true112", "S1_true128"] if c in s16]
        sec = sec.merge(s16[["sample"] + keep].rename(columns={c: f"rho_{c}_A16" for c in keep}), on="sample", how="left")
    # Independent 10x section
    t = s1[s1["sample"] == "10x"].iloc[0]
    tenx = dict(sample="10x", patient="10x_independent", source="10x public", tissue="PDAC", site=None, preservation=None,
                technology=t.technology, treatment_recorded=t.treatment_recorded, n_image_tiles=t.n_image_tiles, um_per_px_legacy=t.um_per_px,
                um_scale_factor=1.0, um_per_px=t.um_per_px, region_definition=t.region_definition, n_regions=t.n_regions,
                patient_identity_note=t.patient_identity_note, figure_use=FIG_USE["10x"],
                note_10x="rho in S6_10x_models (A6); not one of the 70 evaluated sections; same tissue source as HEST TENX179")
    sec = pd.concat([sec, pd.DataFrame([tenx])], ignore_index=True)
    front = ["sample", "patient", "clinical_patient_id", "source", "tissue", "site", "preservation", "technology", "treatment_recorded",
             "n_spots", "n_image_tiles", "um_per_px", "um_per_px_legacy", "um_scale_factor", "region_definition", "n_regions", "treated",
             "rho_S0_bulk", "rho_S1_pancancer", "rho_S2_PDAC_LOPO", "rho_S0_A12_128", "rho_S1_A12_128", "A12_field_true_um",
             "rho_S0_A15_stainnorm", "rho_S1_A15_stainnorm", "stain_difference", "laplacian_orig", "A14_UNI2h_distance_to_TCGA"]
    sec = sec[front + [c for c in sec if c not in front]]
    add("S6_sections", "Spatial sections: 70 evaluated plus independent 10x", sec, "Fig 2d–f, 3b–e; Supp Figs",
        "A10/A12/A15/A14 slide tables; v1 S6 (spots, tiles, legacy µm/px)",
        "rho = local Spearman between predicted and measured programme-minus-reference index in CAF-rich regions. "
        "S0 bulk, S1 pan-cancer spatial, S2 PDAC spatial leave-one-patient-out (A10 original inputs); A12 = both models at the nominal 128-µm field; "
        "A15 = after TCGA-reference stain normalisation (public sections only). GEO µm/px corrected to the 65-µm spot convention "
        "(um_per_px = legacy × 65/55); nominal 128-µm GEO fields were ~151 µm. not_recorded treatment is not confirmed untreated.")
    add("S6_10x_models", "Independent 10x Visium HD section: all models", read(HE / "A6/spatial_10x.tsv"), "Supp Fig (transfer limits) b",
        "results/he/A6/spatial_10x.tsv", "Adding PDAC Visium HD supervision lowered accuracy here (bulk 0.50 → mixed 0.29).")

    # ---- S7: patient level and tests
    p10 = read(A10 / "patient_rho.tsv").rename(columns={"S0": "rho_S0_bulk", "S1": "rho_S1_pancancer", "S2": "rho_S2_PDAC_LOPO"})
    p12 = read(A12 / "patient_rho.tsv")[["patient", "S0_128", "S1_128"]].rename(columns={"S0_128": "rho_S0_A12_128", "S1_128": "rho_S1_A12_128"})
    p15 = read(A15 / "patient_rho.tsv")[["patient", "S0_norm", "S1_norm"]].rename(columns={"S0_norm": "rho_S0_A15_stainnorm", "S1_norm": "rho_S1_A15_stainnorm"})
    p14 = read(A14 / "patient_domain.tsv")
    pat = p10.merge(p12, on="patient").merge(p15, on="patient", how="left")
    pat = pat.merge(p14[["patient", "preservation", "distance"]].rename(columns={"distance": "A14_UNI2h_distance_to_TCGA"}), on="patient", how="left")
    src = sec[sec["sample"] != "10x"].groupby("patient").source.agg(lambda s: ";".join(sorted(set(s))))
    pat.insert(1, "source", pat.patient.map(src))
    assert len(pat) == 25 and pat.preservation.notna().all()
    if (A16 / "COMPLETE").exists():
        p16 = read(A16 / "patient_rho.tsv"); keep = [c for c in p16 if c.startswith(("S0_true", "S1_true"))]
        pat = pat.merge(p16[["patient"] + keep].rename(columns={c: f"rho_{c}_A16" for c in keep}), on="patient", how="left")
    add("S7_patient_rho", "Patient-level local ρ (25 patient groups)", pat, "Fig 2f, 3b, 3d", "A10/A12/A15/A14 patient tables",
        "Sections averaged within patient. A15 columns are blank for the four PDAC Visium HD patients (not normalised).")

    tests = []

    def T(analysis, role, scope, contrast, n, delta, lo, hi, p, fig, src_, n_improved=None, stat="median paired Δρ"):
        tests.append(dict(analysis=analysis, role=role, scope=scope, contrast=contrast, n_patients=int(n), statistic=stat, estimate=delta,
                          ci_low=lo, ci_high=hi, p_two_sided=p, n_improved=n_improved, figure=fig, source=src_))
    lab = {"S1-S0": "pan-cancer spatial − bulk", "S2-S0": "PDAC spatial − bulk", "S2-S1": "PDAC spatial − pan-cancer spatial"}
    figs10 = {"S1-S0": "Fig 3b, 3d row 1", "S2-S1": "Fig 3b, 3d row 5", "S2-S0": "Fig 3b, 3d row 6"}
    for r in read(A10 / "paired_tests.tsv").itertuples():
        T("A10", "primary" if r.contrast == "S1-S0" else "secondary", "all70_25patients", f"{r.contrast} ({lab[r.contrast]})", r.n_patients,
          r.median_paired_delta, r.lo, r.hi, r.p_wilcoxon, figs10[r.contrast], "A10/paired_tests.tsv")
    for r in read(A12 / "paired_tests.tsv").itertuples():
        primary = r.scope == "all70_25patients" and r.contrast == "S1_128-S0_128"
        fig = {("all70_25patients", "S1_128-S0_128"): "Fig 3d row 2", ("public66_21patients", "S1_128-S0_128"): "Fig 3d row 3"}.get((r.scope, r.contrast), "")
        T("A12", "primary" if primary else "secondary/descriptive", r.scope, r.contrast, r.n_patients, r.median_paired_delta, r.lo, r.hi,
          r.p_wilcoxon, fig, "A12/paired_tests.tsv")
    for r in read(A13 / "primary_contrasts.tsv").itertuples():
        T("A13", "primary (sequential: B−C then B−A)", "all70_25patients",
          {"B-C": "local paired − shuffled pairing", "B-A": "local paired − aggregated"}[r.contrast], r.n_common_patients, r.delta_mean,
          r.ci_low, r.ci_high, r.p_two_sided, "", "A13/primary_contrasts.tsv", r.patients_improved,
          stat="mean paired Δρ (sign-flip P)")
    for r in read(A15 / "paired_tests.tsv").itertuples():
        T("A15 (post hoc)", "primary" if r.contrast == "S0_intervention" else "secondary", "public66_21patients",
          {"S0_intervention": "bulk: stain-normalised − original", "S1_intervention": "pan-cancer spatial: stain-normalised − original",
           "norm_S1_minus_S0": "stain-normalised: pan-cancer spatial − bulk"}[r.contrast], r.n_patients, r.median_delta, r.lo, r.hi,
          r.p_wilcoxon, {"S0_intervention": "Fig 2f", "norm_S1_minus_S0": "Fig 3d row 4"}.get(r.contrast, ""), "A15/paired_tests.tsv", r.n_improved)
    for r in read(A15 / "image_quality_correlations.tsv").itertuples():
        T("A15 (post hoc)", "prespecified correlation", "all70_25patients", f"{r.metric} vs ρ {r.model}", r.n_patients,
          r.spearman, None, None, r.p_two_sided,
          {"stain_difference": "Supp Fig (transfer limits) e", "laplacian_orig": "Supp Fig (transfer limits) f"}[r.metric] if r.model == "S0_128" else "",
          "A15/image_quality_correlations.tsv", stat="Spearman (cross-patient)")
    for r in read(A14 / "correlations.tsv").itertuples():
        T("A14 (post hoc, exploratory)", "primary (expected negative; not supported)" if r.model == "S0_128" else "secondary", "all70_25patients",
          f"UNI2-h distance to TCGA vs ρ {r.model}", r.n_patients, r.spearman, None, None, r.p_two_sided,
          "Supp Fig (benchmark) c" if r.model == "S0_128" else "Supp Fig (benchmark) d", "A14/correlations.tsv", stat="Spearman (cross-patient)")
    if (A16 / "COMPLETE").exists():
        for r in read(A16 / "paired_tests.tsv").itertuples():
            T("A16 (post hoc, scale correction)", "primary" if "primary" in r.scope else "secondary/sensitivity", r.scope, r.contrast,
              r.n_patients, r.median_paired_delta, r.lo, r.hi, r.p_wilcoxon, "", "A16/paired_tests.tsv")
    tt = pd.DataFrame(tests)
    a10 = tt[(tt.analysis == "A10") & tt.contrast.str.startswith("S1-S0")].iloc[0]
    assert abs(a10.estimate - 0.10727468737091984) < 1e-12 and abs(a10.p_two_sided - 0.0006021785797492559) < 1e-15
    add("S7_tests", "All paired comparisons and correlation tests (H&E spatial readout)", tt, "Fig 2f, 3b, 3d; Supp Figs",
        "A10/A12/A13/A14/A15(/A16) result tables",
        "Copied from the frozen analysis outputs; no new tests. CIs are patient-bootstrap 95% CIs; P two-sided Wilcoxon signed-rank unless stated. "
        "A10–A16 are amendments added after A9 failed its section-level criterion, each frozen before it was run; "
        "A14 and A15 were added post hoc. A16 rows appear once that run is complete.")

    # ---- S8: A17 (exploratory; survival legend and D1 neighbourhood)
    if (A17 / "correlations.tsv").exists():
        add("S8_A17_cptac_summary", "TCGA vs CPTAC: follow-up, power and heterogeneity (A17 part 1)", read(A17 / "part1_summary.tsv"), "Fig 4e legend",
            "results/he/A17/2026-10-28_run01/part1_summary.tsv",
            "Descriptive, computed post hoc. Power for a per-SD continuous predictor with SE(logHR) ≈ 1/sqrt(deaths), two-sided α = 0.05.")
        add("S8_A17_D1_correlations", "D1 neighbourhood: window metrics vs CAF state and amount (A17 part 2)", read(A17 / "correlations.tsv"), "Fig 4g",
            "results/he/A17/2026-10-28_run01/correlations.tsv",
            "552 CAF-rich 150-µm regions; 3×3-region (450 µm) windows; 95% spatial block-bootstrap CI (600-µm blocks). Single section, exploratory.")
        add("S8_A17_D1_tertiles", "D1 neighbourhood: medians by CAF-state tertile (A17 part 2)",
            read(A17 / "tertile_medians.tsv").rename(columns={"Unnamed: 0": "metric"}), "Fig 4g", "results/he/A17/2026-10-28_run01/tertile_medians.tsv",
            "Descriptive.")
        reg = read(A17 / "regions.tsv")
        add("S8_A17_D1_regions", "D1 neighbourhood: all 150-µm regions (A17 part 2)", reg, "Fig 4g", "results/he/A17/2026-10-28_run01/regions.tsv",
            "Region key = ix × 1,000,000 + iy (150-µm grid in D1 full-resolution pixels). rel and caf_rich identical to A10. Cell types from the existing user annotation.")

    # ---- Write out
    pd.DataFrame(SOURCES.values()).to_csv(OUT / "input_manifest.tsv", sep="\t", index=False)
    index = pd.DataFrame([{k: s[k] for k in ["sheet", "title", "figure", "n_rows", "source", "note"]} for s in SHEETS])
    index.to_csv(OUT / "README_index.tsv", sep="\t", index=False)
    (OUT / "sheets.json").write_text(json.dumps([s["sheet"] for s in SHEETS]))
    print("TOTAL", len(SHEETS), "sheets", int(index.n_rows.sum()), "rows; A16 included:", (A16 / "COMPLETE").exists())


if __name__ == "__main__":
    main()
