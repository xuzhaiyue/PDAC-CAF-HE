# Release validation

Validated on 2 November 2026 against the archived analysis and existing result tables.

- Exactly three public Python files; all compile and expose `--help` without training or creating output directories.
- 49 retained function bodies match the archived numerical implementations after reversing namespace/path changes and the explicitly documented plotting/thread edits.
- Programme selection from the saved patient-paired statistics reproduces all nine selected genes and their selection statistics within 1e-12 tolerance.
- Reference selection reproduces DCN, LOX, OMD and CILP and their selection statistics within 1e-12 tolerance.
- All 28 natural-sample correlation statistics reproduce the original table within 1e-12 tolerance.
- The published-programme command was executed on the archived 9,067-cell atlas. All four shared output tables match the archived tables exactly, including the eight-group mean scores.
- The external-summary command reproduces all four cohort records exactly, including the original one-of-four validation verdict.
- Sparse count aggregation and tied-score AUROC checks pass.
- Repeating an already attempted stage is rejected before writing its outputs.
- Input paths are supplied through configuration/environment variables. No historical plotting module, dynamic source loader, data download or raw-data mutation is used.

The consolidation does not include a full atlas regeneration, scVI retraining, complete external reprocessing or regeneration of the controlled count mixtures. Their retained computational functions were checked against the archived implementation. Thread counts are two; random seeds, model capacities, training batch sizes and analysis thresholds are retained. S5 belongs to a separate sensitivity branch and is not represented as a rerun of the main eight-group analysis.
