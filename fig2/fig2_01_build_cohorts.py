"""Figure 2 队列构建（hemorph 环境）。规则见 config/fig1.yaml 的 fig2（读取生存数据前已冻结）。

TCGA-PAAD：原发肿瘤（样本类型 01）、组织学 = 导管腺癌（TCGA-CDR），每名患者一个样本；表达 = EB++ log2(norm+1)
CPTAC-PDA：肿瘤 mRNA（RSEM UQ log2），组织学 = PDAC
输出（results/fig2/）：
  {cohort}_expr.tsv.gz      基因 × 样本（供 ESTIMATE）
  {cohort}_scores.tsv       各基因集得分（基因集均值；相对 = 减去泛 CAF；Moffitt basal − classical；MPAS）
  {cohort}_clinical.tsv     年龄、分期、生存、KRAS 状态（CPTAC 另含病理估计的基质比例、肿瘤细胞比例）
原始文件只读，不复制。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((PROJ / "config" / "fig1.yaml").read_text())
F2 = CFG["fig2"]
OUT = PROJ / "results" / "fig2"
OUT.mkdir(parents=True, exist_ok=True)
PROG = yaml.safe_load((PROJ / "results" / "fig1" / "signature" / "programs.yaml").read_text())
PUB = yaml.safe_load((PROJ / "config" / "published_signatures.yaml").read_text())
SETS = {"program": PROG["CTHRC1_matrix_CAF"], "pan_CAF": PROG["pan_CAF"], "CTHRC1_alone": ["CTHRC1"],
        "Moffitt_activated": PUB["Moffitt_activated_stroma"]["genes"],
        "Dominguez_LRRC15": PUB["Dominguez_LRRC15_CAF"]["genes"],
        "Moffitt_basal": PUB["Moffitt_basal"]["genes"], "Moffitt_classical": PUB["Moffitt_classical"]["genes"]}
STAGE_EARLY = {"Stage I", "Stage IA", "Stage IB", "Stage IIA"}


def scores(E: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """E: genes x samples, log2 scale."""
    s, cover = {}, []
    for k, g in SETS.items():
        present = [x for x in g if x in E.index]
        cover.append({"cohort": cohort, "set": k, "n_genes": len(g), "n_present": len(present),
                      "missing": ",".join(sorted(set(g) - set(present)))})
        s[k] = E.loc[present].mean(0)
    s = pd.DataFrame(s)
    for k in ["program", "Moffitt_activated", "Dominguez_LRRC15"]:
        s[f"{k}_rel"] = s[k] - s["pan_CAF"]
    s["Moffitt_basal_minus_classical"] = s.Moffitt_basal - s.Moffitt_classical
    mp = [x for x in PUB["MPAS"]["genes"] if x in E.index]
    z = E.loc[mp].sub(E.loc[mp].mean(1), axis=0).div(E.loc[mp].std(1), axis=0)
    s["MPAS"] = z.sum(0) / np.sqrt(len(mp))
    cover.append({"cohort": cohort, "set": "MPAS", "n_genes": 10, "n_present": len(mp),
                  "missing": ",".join(sorted(set(PUB["MPAS"]["genes"]) - set(mp)))})
    return s, pd.DataFrame(cover)


def kras_from(df, sample_col, aa_col):
    k = df[df.gene_col == "KRAS"]
    hot = k[k[aa_col].astype(str).str.contains(r"p\.[GQ](12|13|61)", regex=True)]
    allele = hot.groupby(sample_col)[aa_col].first().str.replace("p.", "", regex=False)
    return allele


def tcga():
    D = F2["data"]
    ph = pd.read_csv(D["tcga_pheno"], sep="\t")
    ph = ph[(ph._primary_disease == "pancreatic adenocarcinoma") & (ph["sample"].str[13:15] == F2["cohorts"]["tcga"]["sample_type"])]
    cdr = pd.read_csv(OUT / "tcga_cdr_paad.tsv", sep="\t")
    cdr = cdr[cdr.histological_type == F2["cohorts"]["tcga"]["histology"]]
    ph["patient"] = ph["sample"].str[:12]
    ph = ph[ph.patient.isin(cdr.bcr_patient_barcode)].drop_duplicates("patient")
    hdr = pd.read_csv(D["tcga_expr"], sep="\t", nrows=0).columns
    cols = ["sample"] + [c for c in hdr if c in set(ph["sample"])]
    E = pd.read_csv(D["tcga_expr"], sep="\t", usecols=cols, index_col=0)
    E = E[~E.index.astype(str).str.fullmatch(r"\d+")]
    E = E[~E.index.duplicated()]
    E.columns = E.columns.str[:12]
    print(f"TCGA: {E.shape[1]} ductal PDAC primary tumours with expression; {E.shape[0]} genes")
    mut = pd.read_csv(D["tcga_mut"], sep="\t", usecols=["sample", "gene", "Amino_Acid_Change"]).rename(columns={"gene": "gene_col"})
    mut = mut[mut["sample"].str[:12].isin(E.columns)]
    sequenced = set(mut["sample"].str[:12])
    allele = kras_from(mut.assign(patient=mut["sample"].str[:12]), "patient", "Amino_Acid_Change")
    c = cdr.set_index("bcr_patient_barcode").loc[E.columns]
    clin = pd.DataFrame({
        "age": c.age_at_initial_pathologic_diagnosis.values,
        "stage_raw": c.ajcc_pathologic_tumor_stage.values,
        "stage_late": np.where(c.ajcc_pathologic_tumor_stage.isin(STAGE_EARLY), 0,
                               np.where(c.ajcc_pathologic_tumor_stage.str.startswith("Stage"), 1, np.nan)),
        "OS": c.OS.values, "OS_time": c["OS.time"].values, "PFI": c.PFI.values, "PFI_time": c["PFI.time"].values,
        "KRAS_sequenced": [p in sequenced for p in E.columns],
        "KRAS_allele": [allele.get(p, "WT") if p in sequenced else np.nan for p in E.columns]}, index=E.columns)
    return E, clin


def cptac():
    D = F2["data"]
    E = pd.read_csv(D["cptac_expr"], sep="\t", index_col=0)
    E = E[~E.index.duplicated()]
    cl = pd.read_csv(D["cptac_clin"], sep="\t").set_index("case_id")
    keep = [s for s in E.columns if s in cl.index and cl.loc[s, "histology_diagnosis"] == F2["cohorts"]["cptac"]["histology"]]
    print(f"CPTAC: {E.shape[1]} tumours with expression; {len(keep)} PDAC histology "
          f"(excluded: {sorted(set(E.columns) - set(keep))})")
    E = E[keep]
    maf = pd.read_csv(D["cptac_mut"], sep="\t", usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "HGVSp_Short"],
                      low_memory=False).rename(columns={"Hugo_Symbol": "gene_col"})
    sequenced = set(maf.Tumor_Sample_Barcode)
    allele = kras_from(maf, "Tumor_Sample_Barcode", "HGVSp_Short")
    c = cl.loc[keep]
    stage = c.tumor_stage_pathological.astype(str)
    clin = pd.DataFrame({
        "age": pd.to_numeric(c.age, errors="coerce").values, "stage_raw": stage.values,
        "stage_late": np.where(stage.isin(STAGE_EARLY), 0, np.where(stage.str.startswith("Stage"), 1, np.nan)),
        "OS": c.vital_status.map({"Deceased": 1, "Living": 0}).values,
        "OS_time": pd.to_numeric(c.follow_up_days, errors="coerce").values,
        "Stromal_fraction": pd.to_numeric(c.Stromal_fraction, errors="coerce").values,
        "Neoplastic_cellularity": pd.to_numeric(c.Neoplastic_cellularity, errors="coerce").values,
        "KRAS_sequenced": [s in sequenced for s in keep],
        "KRAS_allele": [allele.get(s, "WT") if s in sequenced else np.nan for s in keep]}, index=keep)
    return E, clin


def main():
    covers = []
    for name, fn in [("tcga", tcga), ("cptac", cptac)]:
        E, clin = fn()
        E.to_csv(OUT / f"{name}_expr.tsv.gz", sep="\t", float_format="%.4f")
        s, cov = scores(E, name)
        s.to_csv(OUT / f"{name}_scores.tsv", sep="\t", float_format="%.5f")
        clin.to_csv(OUT / f"{name}_clinical.tsv", sep="\t")
        covers.append(cov)
        print(f"  stage: {clin.stage_raw.value_counts().to_dict()}")
        print(f"  KRAS alleles: {clin.KRAS_allele.value_counts(dropna=False).to_dict()}")
        print(f"  OS events: {int(np.nansum(clin.OS))} / {clin.OS.notna().sum()}")
    cov = pd.concat(covers)
    cov.to_csv(OUT / "gene_coverage.tsv", sep="\t", index=False)
    print(cov[cov.n_present < cov.n_genes].to_string(index=False))


if __name__ == "__main__":
    main()
