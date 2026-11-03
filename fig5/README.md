# Figure 5 table reproduction

Run from the project root in the hemorph environment.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 19 python fig5/figure5_tables.py \
  --tables results/manuscript_figures/2026-10-29_Figure5_v2/data \
  --regions results/he/A17/2026-10-28_run01/regions.tsv \
  --out results/code_release/figure5_new_run
```

The script exports the ten archived panel tables, checks the pooled HR and confidence interval against the manuscript, checks the direction of the eight cohort estimates, and recalculates the Kaplan–Meier risk counts. It also recalculates the two reported D1 Spearman correlations using the A17 CAF-rich regions. The panel map table contains display values and is retained for visualisation. Cox estimates, pooled estimates and spatial bootstrap intervals are taken from the archived tables. The verification JSON records source SHA-256 hashes.

The six upstream bulk cohort and survival scripts retain their historical filenames and write to the original `results/fig2/` paths used by the shared H&E workflows.
