# Figure 3 numerical reproduction

Run from the repository root in the hemorph environment.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 19 python fig3/figure3_tables.py \
  --a10 results/he/A10/2026-10-21_run01 \
  --a12 results/he/A12/2026-10-21_run01 \
  --a15 results/he/A15/2026-10-27_run01 \
  --regions results/manuscript_figures/2026-10-29_Figure3_v3/data \
  --out results/code_release/figure3_new_run
```

The entry point checks A10 patient averages against section predictions, computes the three model medians, recalculates example Spearman correlations on the CAF-rich masks, and checks the six paired median differences against their archived tests. The paired table carries forward the original confidence intervals and P values. It also exports section correlations and medians by source.

Panel b uses the original A10 comparison, including the S0 median of 0.064. Figure 2e uses the A12 128-µm evaluation. The inputs are archived predictions and statistical tables. The output includes input SHA-256 hashes.

Validated against the Figure 3 v3 tables on 2026-11-03. The four reference comparisons for panels b–e passed at a tolerance of 1e-12.
