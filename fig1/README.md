# Figure 1 analysis

Three standalone Python files reproduce the numerical analysis for the discovery CAF atlas, fixed molecular targets, published-programme comparisons, external evaluation and pseudo-bulk experiments. They do not import the historical plotting or assembly scripts.

| File | Analysis |
| --- | --- |
| `01_build_caf_atlas.py` | Input preparation, global integration, annotation and transfer, doublet/QC assessment, bounded CAF cleaning, stability and cell-removal summaries |
| `02_define_matrix_programme.py` | Patient-paired comparisons, lineage contribution estimates, fixed programme/reference selection and leave-one-dataset-out analysis |
| `03_validate_matrix_programme.py` | Published programmes, independently processed external cohorts, controlled mixtures and natural pseudo-bulk correlations |

## Inputs and configuration

Discovery datasets: CRA001160, GSE205013 and GSE278688. External datasets: GSE242230, GSE202051, GSE253429 and GSE291124. The configuration describes the input H5AD/MTX paths, metadata columns and condition mappings. The input H5AD resources are standardized upstream datasets, not unprocessed sequencing reads. Supply those files with the documented schema before running the analysis. No data or models are downloaded by these scripts.

Set the two input locations and a new output location:

```bash
export FIG1_DATA_ROOT=/path/to/discovery_data_root
export FIG1_EXTERNAL_BUNDLE=/path/to/standardized_external_bundle
export FIG1_RUN_DIR=/path/to/new_figure1_run
export NUMBA_THREADING_LAYER=workqueue
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

`config/figure1.yaml` contains the discovery metadata and frozen numerical rules. `config/published_signatures.yaml` contains the published gene lists and their provenance. Local absolute input paths are supplied through the environment; the scripts set output paths from `--run-dir`. `requirements.txt` and `ENVIRONMENT.json` record the relevant versions from the archived scVI environment. A compatible Scanpy/scVI environment is required for the atlas and external-processing stages; `--help` does not load the models.

## 1. Discovery atlas

Run the following from the directory containing these three Python files. Each command is a separate stage and can be inspected before running. No stage is started automatically.

```bash
python 01_build_caf_atlas.py --run-dir "$FIG1_RUN_DIR" --stage build-global
python 01_build_caf_atlas.py --run-dir "$FIG1_RUN_DIR" --stage build-full
python 01_build_caf_atlas.py --run-dir "$FIG1_RUN_DIR" --stage integrate-global
python 01_build_caf_atlas.py --run-dir "$FIG1_RUN_DIR" --stage annotate
python 01_build_caf_atlas.py --run-dir "$FIG1_RUN_DIR" --stage qc
python 01_build_caf_atlas.py --run-dir "$FIG1_RUN_DIR" --stage clean-cells
python 01_build_caf_atlas.py --run-dir "$FIG1_RUN_DIR" --stage clean-loop
python 01_build_caf_atlas.py --run-dir "$FIG1_RUN_DIR" --stage removal-summary
```

The clean loop starts with the clean mesenchymal scope, writes the cluster flags for every round and permits the historical maximum of four exclusion rounds. `integrate_caf_final` points to the final round only when no flagged clusters remain. The cell-removal ledger follows the actual recorded rounds. Counts layers, original cell identifiers and the exclusion records are retained. Integration uses the original model settings and batch sizes, with numerical threads limited to two.

The following stage includes two additional scVI training runs and should be invoked separately:

```bash
python 01_build_caf_atlas.py --run-dir "$FIG1_RUN_DIR" --stage stability
```

`--stage legacy-inputs` retains the historical author-label input preparation route for provenance. It is not an additional prerequisite for the global-to-CAF route above.

## 2. Fixed molecular targets

```bash
python 02_define_matrix_programme.py --run-dir "$FIG1_RUN_DIR"
```

This writes `fig1/signature/`, including patient-paired differential statistics, the pooled test, lineage estimates, selected genes, leave-one-dataset-out results and `programs.yaml`. Rules from the original signature revisions are combined in their original order. The current target has nine programme genes and four reference genes; no outcome-driven gene reselection is added.

## 3. Comparisons and external evaluation

```bash
python 03_validate_matrix_programme.py --run-dir "$FIG1_RUN_DIR" --stage published
python 03_validate_matrix_programme.py --run-dir "$FIG1_RUN_DIR" --stage external --dataset GSE242230
python 03_validate_matrix_programme.py --run-dir "$FIG1_RUN_DIR" --stage external --dataset GSE202051
python 03_validate_matrix_programme.py --run-dir "$FIG1_RUN_DIR" --stage external --dataset GSE253429
python 03_validate_matrix_programme.py --run-dir "$FIG1_RUN_DIR" --stage external --dataset GSE291124
python 03_validate_matrix_programme.py --run-dir "$FIG1_RUN_DIR" --stage external-summary
python 03_validate_matrix_programme.py --run-dir "$FIG1_RUN_DIR" --stage pseudobulk
```

The published-programme stage writes membership, coverage, eight-group mean scores and contribution summaries without plotting. The external stage trains each dataset independently and retains the original matching and validation criteria. The pseudo-bulk stage writes raw/relative scores, correlations and the original PASS/FAIL verdicts for both controlled mixtures and natural samples.

## Reusing archived intermediate files

All three scripts accept `--input-root /path/to/archived/results`. That directory must contain `fig1/` and, when needed, `fig1_external/` with their original file layout. A missing input in the new run is read from the archived root. No input is copied or changed. New outputs always stay in the dedicated run directory; existing output files or a previously attempted stage are rejected. `--input-root` refers to intermediate analysis files and does not replace the raw dataset paths in the configuration.

For example, recompute published-programme tables from an archived atlas into a fresh output directory:

```bash
python 03_validate_matrix_programme.py --run-dir /path/to/new_comparison_run --input-root /path/to/archived/results --stage published
```

## Output and provenance

Outputs are numerical TSV/YAML/H5AD/model files under `fig1/` or `fig1_external/`. Run records under `provenance/` contain the command, effective configuration, seeds, package versions, source hashes and observed inputs. Small input files are SHA-256 checked; large inputs are recorded by path, size and modification time to avoid an extra full read. Failed attempts remain recorded; use a new run directory rather than overwriting them.

`SOURCE_MAP.json` maps the consolidated routines to the archived analysis files. `VALIDATION.md` describes the checks performed for this release. This package covers the main eight-group analysis and its S1–S4 analyses. The separate ten-cluster sensitivity branch shown in S5 is outside this consolidation. Publication layout and panel rendering are outside this analysis-only package.
