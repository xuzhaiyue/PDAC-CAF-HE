# PDAC-CAF-HE

Analysis code for *"Spatially supervised histology mapping of a matrix-associated fibroblast programme in pancreatic ductal adenocarcinoma"*.

The repository contains the analyses behind every number reported in the paper. Figures were drawn from the saved outputs of these analyses, and the plotted values are provided in the Supplementary Tables. Data are not included.

## Data
| Input | Accessions |
|---|---|
| Single-cell discovery | CRA001160, GSE205013, GSE278688 |
| Single-cell external | GSE242230, GSE202051, GSE253429, GSE291124 |
| Bulk survival | TCGA-PAAD, CPTAC-PDA, GSE28735, GSE62452, GSE57495, GSE71729, GSE85916, GSE21501 |
| Histology | TCGA-PAAD diagnostic slides; CPTAC UNI2-h features |
| Spatial | HEST-1k, GSE211895, GSE274557, GSE327056, 10x Visium HD 3′ PDAC, PDAC Visium HD (SRA BioProject PRJNA1390721) |

Local data paths are set in `config/fig1.yaml` and `config/aim1.yaml`.

## Code
| Path | Analysis |
|---|---|
| `fig1/` | Single-cell atlas (scVI integration, QC, doublets, clustering stability), nine-gene programme and four-gene CAF reference, published-signature comparison, external validation, pseudo-bulk decoupling |
| `fig2/` | Bulk cohorts, ESTIMATE, programme scores, multi-cohort meta-analysis, joint state–abundance Cox models |
| `he/` | H&E analyses A1–A17: UNI2-h features, bulk-supervised models, StRoMA replication, pan-cancer pretraining and PDAC fine-tuning, bulk vs spatial supervision, matched-field, matched supervision experiment, domain distance, stain normalisation, corrected physical fields, survival of predictions, D1 neighbourhood |
| `src/hemorph/` | Shared functions |
| `scripts/` | Supplementary-table assembly, A12 summary, public-data download |
| `config/` | Analysis rules (`fig1.yaml`), gene sets, published signatures |
| `env/` | Conda environments and R session info |

Analysis rules and their order are recorded in `config/fig1.yaml`. Identifiers A1–A17 map to the analysis names in the paper through Supplementary Table S9.

## Running
- Environments: `hemorph` (general analyses and H&E), `pdac_scvi` (scVI; set `NUMBA_THREADING_LAYER=workqueue`), `scRNA` and R.
- Scripts in `he/` import each other; run them from inside `he/`.
- Developed on Apple silicon with PyTorch MPS.
- Some scripts verify SHA-256 hashes of the authors' original intermediate files; when re-running from scratch, regenerate those files and update the recorded hashes.
