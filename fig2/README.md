# Figure 2 release inventory

| Panel | Numerical source | Proposed release entry point |
| --- | --- | --- |
| a | Workflow schematic | Patient-level benchmark |
| b | `results/he/auc_summary.tsv` | Patient-level Ridge, additive and attention benchmark |
| c | `results/he/positive_control_ceiling.tsv` | RNA-derived tissue proxy benchmark |
| d | A10 D1 regions and A12 public-section regions | Bulk spatial baseline |
| e | A12 section-level correlations | Bulk spatial baseline |
| f | A15 patient-level paired stain intervention | Stain normalisation |

The proposed public layout contains three analysis entry points: patient benchmark, spatial baseline and stain normalisation. Shared extraction, feature and inference routines should be reused across later figures. StRoMA replication supports Supplementary Figure S6 and remains a distinct benchmark path.

`SOURCE_MAP.json` identifies the candidate analysis sources and their SHA-256 hashes. The inventory is a staging document. The exact writer of the Figure 2c historical proxy table remains to be established. The existing StRoMA replication script evaluates different targets and does not directly produce the four-row Figure 2c table. The Figure 2b consolidated table also requires a documented assembly step before release. Raw data, trained weights and layout scripts are outside this inventory.

## Verified table-level entry points

Run from the repository root with the hemorph environment. Give each command a new output directory.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
FIG2_TABLES=results/manuscript_figures/2026-10-29_Figure2_v10/data
FIG2_OUTPUT=results/code_release/figure2_new_run
nice -n 19 python fig2/01_patient_benchmark.py \
  --auc results/he/auc_summary.tsv --proxies results/he/positive_control_ceiling.tsv \
  --out "$FIG2_OUTPUT/patient"
nice -n 19 python fig2/02_spatial_baseline.py \
  --sections results/he/A12/2026-10-21_run01/slide_rho.tsv --regions "$FIG2_TABLES" \
  --out "$FIG2_OUTPUT/spatial"
nice -n 19 python fig2/03_stain_normalisation.py \
  --slides results/he/A15/2026-10-27_run01/slide_rho.tsv --out "$FIG2_OUTPUT/stain"
```

The patient command selects the 24 archived AUC/CI rows and the four displayed RNA proxies, preserving the six-row proxy source table. The spatial command recomputes example Spearman correlations on the archived CAF-rich masks and summarises section-level S0 correlations. The stain command averages paired section correlations within patient and reproduces the original bootstrap confidence interval and Wilcoxon test.

All five reference table comparisons passed at 1e-12 tolerance. The three scripts are table-level reproduction tools. Upstream model training, feature extraction and prediction are a separate release-preparation step.
