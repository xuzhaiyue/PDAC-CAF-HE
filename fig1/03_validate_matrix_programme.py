"""Figure 1 published-programme comparison, external validation and pseudo-bulk checks. Numerical analysis only; original results are never overwritten."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
try:
    os.nice(19)
except (AttributeError, PermissionError):
    pass

_RUN_DIR = None
_INPUT_ROOT = None
_CONFIG_PATH = None
_READS = set()


def _read_path(path):
    """Resolve a run-relative missing input against a read-only results root."""
    path = Path(path)
    if not path.exists() and _INPUT_ROOT is not None:
        try:
            relative = path.relative_to(_RUN_DIR)
        except ValueError:
            relative = None
        if relative is not None:
            candidate = _INPUT_ROOT / relative
            if candidate.exists():
                path = candidate
    if path.exists() and path.is_file():
        _READS.add(str(path.resolve()))
    return path


def _load_config():
    import yaml
    text = os.path.expandvars(_CONFIG_PATH.read_text())
    if "${" in text:
        raise ValueError("Set all environment variables used by the configuration.")
    config = yaml.safe_load(text)
    config["paths"]["out_dir"] = str(_RUN_DIR / "fig1")
    return config


def _load_published():
    import yaml
    path = _CONFIG_PATH.parent / "published_signatures.yaml"
    _READS.add(str(path.resolve()))
    return yaml.safe_load(path.read_text())


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _start(args, stage, targets):
    global _RUN_DIR, _INPUT_ROOT, _CONFIG_PATH
    _RUN_DIR = Path(args.run_dir).expanduser().resolve()
    _INPUT_ROOT = Path(args.input_root).expanduser().resolve() if args.input_root else None
    _CONFIG_PATH = Path(args.config).expanduser().resolve()
    if _INPUT_ROOT is not None and (_RUN_DIR == _INPUT_ROOT or _INPUT_ROOT.is_relative_to(_RUN_DIR) or any(_RUN_DIR.is_relative_to(_INPUT_ROOT / name) for name in ("fig1", "fig1_external"))):
        raise ValueError("Run directory must not contain or overlap either cached analysis tree.")
    config = _load_config()
    raw_roots = [Path(config["paths"]["panin"]).resolve(), Path(config["external_validation"]["bundle"]).resolve()]
    if any(_RUN_DIR == p or _RUN_DIR.is_relative_to(p) for p in raw_roots):
        raise ValueError("Run directory must be outside raw-data roots.")
    marker = _RUN_DIR / ".figure1_analysis_run"
    if _RUN_DIR.exists() and not marker.exists():
        raise FileExistsError("Refusing an existing directory not created by this release.")
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)
    token = f"{Path(__file__).stem}_{stage}"
    record = _RUN_DIR / "provenance" / f"{token}.json"
    if record.exists():
        raise FileExistsError(f"Stage already attempted: {record}; use a new run directory.")
    for relative in targets:
        path = _RUN_DIR / relative
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    record.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    config_snapshot = record.with_suffix(".yaml")
    config_snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    payload = {"stage": stage, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "command": sys.argv, "python": platform.python_version(), "script_sha256": _sha256(__file__),
               "config_sha256": _sha256(_CONFIG_PATH), "input_root": str(_INPUT_ROOT), "threads": 2,
               "state": "STARTED", "run_config": str(config_snapshot)}
    record.write_text(json.dumps(payload, indent=2))
    return record, payload


def _finish(record, payload, error=None):
    import importlib.metadata
    payload["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload["state"] = "FAILED" if error is not None else "COMPLETE"
    payload["input_files"] = [{"path": p, "bytes": Path(p).stat().st_size,
                               "mtime_ns": Path(p).stat().st_mtime_ns,
                               "sha256": _sha256(p) if Path(p).stat().st_size <= 16*1024*1024 else None}
                              for p in sorted(_READS)]
    payload["large_input_hash_policy"] = "Size/mtime recorded; large input files are not rehashed during execution."
    payload["versions"] = {}
    for package in ("numpy", "pandas", "scipy", "scanpy", "anndata", "scvi-tools", "torch", "scikit-learn"):
        try:
            payload["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if error is not None:
        payload["error"] = repr(error)
    record.write_text(json.dumps(payload, indent=2))


import yaml


def _external_result_files():
    files = {}
    if _INPUT_ROOT is not None:
        files.update({p.parent.name:p for p in (_INPUT_ROOT / 'fig1_external').glob('*/result.tsv')})
    files.update({p.parent.name:p for p in (_RUN_DIR / 'fig1_external').glob('*/result.tsv')})
    return [files[key] for key in sorted(files)]

def published_main():
    """Numerical outputs underlying S2; no plotting module is required."""
    import numpy as np
    import pandas as pd
    import scanpy as sc
    root = _RUN_DIR / "fig1"
    config = _load_config()
    programmes = yaml.safe_load(_read_path(root / "signature/programs.yaml").read_text())
    published = {k:v for k,v in _load_published().items() if k not in ("Moffitt_basal", "Moffitt_classical", "MPAS")}
    names = config["release"]["caf_names"]
    order = config["release"]["caf_order"]
    a = sc.read_h5ad(_read_path(root / "integrate_caf_final/fibro_integrated.h5ad"))
    a.obs["subtype"] = a.obs["leiden_0.5"].astype(str).map(names)
    sets = {"CTHRC1_program": ("CTHRC1+ program (this study)", programmes["CTHRC1_matrix_CAF"]),
            "pan_CAF": ("Pan-CAF ref. (this study)", programmes["pan_CAF"])}
    sets.update({k:(v["label"], v["genes"]) for k,v in published.items()})
    rows = []
    for key, (label, genes) in sets.items():
        present = [g for g in genes if g in a.var_names]
        rows.append({"key":key, "label":label, "n_genes":len(genes), "n_in_data":len(present),
                     "missing":",".join(sorted(set(genes)-set(present))), "genes":present})
    sig = pd.DataFrame(rows).set_index("key")
    out = root / "published_signatures"
    out.mkdir(parents=True)
    sig.drop(columns="genes").to_csv(out / "signature_genes_in_data.tsv", sep="\t")
    genes = programmes["CTHRC1_matrix_CAF"] + programmes["pan_CAF"]
    pd.DataFrame([[g in published[k]["genes"] for k in published] for g in genes], index=genes, columns=list(published)).to_csv(out / "overlap.tsv", sep="\t")
    for key in sig.index:
        sc.tl.score_genes(a, sig.loc[key,"genes"], score_name=f"s_{key}", random_state=0)
    st = a.obs.subtype.astype(str)
    means = pd.DataFrame({k:a.obs[f"s_{k}"].groupby(st).mean() for k in sig.index}).reindex(order)
    means.to_csv(out / "subtype_mean_scores.tsv", sep="\t")
    ((means-means.mean())/means.std()).to_csv(out / "subtype_score_z.tsv", sep="\t")
    c = pd.read_csv(_read_path(root / "signature/fibroblast_contribution.tsv"), sep="\t")
    fc = c.groupby("gene").fib_contribution.mean()
    rows = []
    for key in sig.index:
        v = fc.reindex(sig.loc[key,"genes"]).dropna()
        rows.append({"key":key,"label":sig.loc[key,"label"],"n":len(v),"median":v.median(),"frac_ge_0.6":(v>=0.6).mean()})
    pd.DataFrame(rows).to_csv(out / "fibroblast_contribution_by_set.tsv", sep="\t", index=False)


# fig1_15_external_validation.py: retained numerical analysis.
def _init_external():
    global Path, ad, external_CAF_NAMES, external_CFG, external_COMPARE, external_EV, external_OUT, external_PROG, external_PROJ, external_PUB, external_RNG, external_ROOT, external_TARGET, glob, np, os, pd, sc, scvi, sp, stats, sys, yaml
    import os
    os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
    import glob
    import sys
    from pathlib import Path
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    import scvi
    import yaml
    from scipy import stats
    external_PROJ = _RUN_DIR.parent
    external_CFG = _load_config()
    external_EV = external_CFG['external_validation']
    external_ROOT = Path(external_CFG['paths']['out_dir'])
    external_OUT = _RUN_DIR / 'fig1_external'
    external_PROG = yaml.safe_load(_read_path(external_ROOT / 'signature' / 'programs.yaml').read_text())
    external_PUB = _load_published()
    external_CAF_NAMES = {'2': 'CTHRC1+ matrix myCAF', '1': 'HOPX+ F2R+ myCAF', '5': 'Glycolytic CAF', '0': 'C7+ C3+ iCAF', '4': 'MFAP5+ IGF1+ iCAF', '3': 'PI16+ universal', '6': 'SPP1+ IBSP+ CAF', '7': 'Chemo-associated CAF'}
    external_TARGET = 'CTHRC1+ matrix myCAF'
    external_COMPARE = {'program': external_PROG['CTHRC1_matrix_CAF'], 'CTHRC1_alone': ['CTHRC1'], 'Moffitt_activated': external_PUB['Moffitt_activated_stroma']['genes'], 'Dominguez_LRRC15': external_PUB['Dominguez_LRRC15_CAF']['genes']}
    external_RNG = np.random.default_rng(external_CFG['random_seed'])
    scvi.settings.seed = 0

def external_mad(x):
    return np.median(np.abs(x - np.median(x))) * 1.4826

def external_load(ds: str) -> ad.AnnData:
    spec = external_EV['datasets'][ds]
    files = sorted(glob.glob(str(Path(external_EV['bundle']) / spec['files'])))
    parts = []
    for f in files:
        a = ad.read_h5ad(_read_path(f))
        if 'keep_tissue' in spec and 'tissue_type' in a.obs:
            a = a[a.obs.tissue_type.astype(str) == spec['keep_tissue']]
        X = a.layers['counts'] if 'counts' in a.layers else a.X
        b = ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame(index=a.obs_names.astype(str)), var=pd.DataFrame(index=a.var_names.astype(str)))
        b.obs['patient_key'] = a.obs['patient_key'].astype(str).values
        b.obs['condition'] = a.obs['condition'].astype(str).values
        b.obs['author_label'] = a.obs['ann_L2'].astype(str).values if 'ann_L2' in a.obs else 'NA'
        b.var_names_make_unique()
        parts.append(b)
    a = ad.concat(parts, join='inner', merge=None) if len(parts) > 1 else parts[0]
    a.obs_names_make_unique()
    keep = np.zeros(a.n_obs, bool)
    for _, idx in a.obs.groupby('patient_key', observed=True).indices.items():
        keep[idx if len(idx) <= external_EV['cap_per_sample'] else external_RNG.choice(idx, external_EV['cap_per_sample'], replace=False)] = True
    a = a[keep].copy()
    q, mod = (external_EV['qc'], spec['modality'])
    C = a.X
    n_genes = np.asarray((C > 0).sum(1)).ravel()
    n_umi = np.asarray(C.sum(1)).ravel()
    mt = a.var_names.str.upper().str.startswith('MT-')
    pct_mt = np.asarray(C[:, mt].sum(1)).ravel() / np.maximum(n_umi, 1) * 100
    lg, lu = (np.log(np.maximum(n_genes, 1)), np.log(np.maximum(n_umi, 1)))
    g_min = max(q['min_genes'][mod], float(np.exp(np.median(lg) - q['mad_k'] * external_mad(lg))))
    u_min = float(np.exp(np.median(lu) - q['mad_k'] * external_mad(lu)))
    mt_max = max(q['max_mt_floor'][mod], float(np.median(pct_mt) + q['mad_k'] * external_mad(pct_mt)))
    ok = (n_genes >= g_min) & (n_umi >= u_min) & (pct_mt <= mt_max)
    print(f'{ds}: {a.n_obs} cells after cap; QC genes>={g_min:.0f}, umi>={u_min:.0f}, mt<={mt_max:.1f}% -> keep {ok.sum()}', flush=True)
    a = a[ok].copy()
    a.obs['n_genes'] = n_genes[ok]
    a.layers['counts'] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=10000.0)
    sc.pp.log1p(a)
    return a

def external_integrate(a: ad.AnnData, max_epochs: int) -> ad.AnnData:
    sc.pp.highly_variable_genes(a, flavor='seurat_v3', n_top_genes=3000, layer='counts', subset=False)
    hv = a[:, a.var.highly_variable].copy()
    scvi.model.SCVI.setup_anndata(hv, layer='counts', batch_key='patient_key')
    m = scvi.model.SCVI(hv, n_latent=20, n_layers=2, gene_likelihood='nb')
    m.train(max_epochs=max_epochs, early_stopping=True, accelerator='cpu', batch_size=512, enable_progress_bar=False)
    a.obsm['X_scVI'] = m.get_latent_representation()
    sc.pp.neighbors(a, use_rep='X_scVI', n_neighbors=15, random_state=0)
    sc.tl.leiden(a, resolution=external_EV['caf_resolution'], key_added='leiden', flavor='igraph', n_iterations=2, random_state=0)
    sc.tl.umap(a, random_state=0)
    return a

def external_gscore(a, genes, name):
    sc.tl.score_genes(a, [g for g in genes if g in a.var_names], score_name=name, random_state=0)

def external_annotate_lineage(a: ad.AnnData, out: Path) -> pd.DataFrame:
    for k, g in external_CFG['lineage_markers'].items():
        external_gscore(a, g, f'lin_{k}')
    cols = [f'lin_{k}' for k in external_CFG['lineage_markers'] if k != 'Cycling']
    m = a.obs.groupby('leiden', observed=True)[cols].mean()
    rows = []
    for cl, r in m.iterrows():
        s = r.sort_values(ascending=False)
        top = s.index[0][4:]
        rows.append({'cluster': str(cl), 'top': top, 'top_score': s.iloc[0], 'second': s.index[1][4:], 'margin': s.iloc[0] - s.iloc[1], 'label': top if s.iloc[0] - s.iloc[1] >= external_EV['lineage_min_margin'] else 'Ambiguous', 'n_cells': int((a.obs.leiden == cl).sum())})
    tab = pd.DataFrame(rows).set_index('cluster')
    tab.to_csv(out / 'global_cluster_annotation.tsv', sep='\t')
    a.obs['lineage'] = a.obs.leiden.astype(str).map(tab.label)
    print(tab.round(2).to_string(), flush=True)
    return tab

def external_flag_clusters(c: ad.AnnData) -> pd.DataFrame:
    X = external_CFG['cluster_exclusion']
    C = sp.csr_matrix(c.layers['counts'])
    v = list(c.var_names)
    pos = {g: C[:, v.index(g)].toarray().ravel() > 0 for g in ['RGS5', 'NOTCH3', 'MYH11'] if g in v}
    for k, g in external_CFG['contaminant_markers'].items():
        if k in ('epithelial', 'immune'):
            external_gscore(c, g, f'contam_{k}')
    df = pd.DataFrame({'cl': c.obs.leiden.astype(str).values, 'n_genes': c.obs.n_genes.values, **pos, 'epi': c.obs.contam_epithelial.values, 'imm': c.obs.contam_immune.values})
    g = df.groupby('cl')
    f = pd.DataFrame({'n_cells': g.size(), 'pericyte_smc_mean_frac': g[list(pos)].mean().mean(1), 'median_genes_ratio': g.n_genes.median() / df.n_genes.median(), 'epithelial_score': g.epi.mean(), 'immune_score': g.imm.mean()})
    f['exclude'] = (f.pericyte_smc_mean_frac >= X['pericyte_smc_mean_frac']) | (f.median_genes_ratio < X['low_complexity_ratio']) | (f.epithelial_score > X['epithelial_score']) | (f.immune_score > X['immune_score'])
    return f

def external_discovery_centroids():
    d = ad.read_h5ad(_read_path(external_ROOT / 'integrate_caf_final' / 'fibro_integrated.h5ad'))
    mk = pd.read_csv(_read_path(external_ROOT / 'integrate_caf_final' / 'markers_0.5.tsv'), sep='\t')
    mk['group'] = mk.group.astype(str)
    excl = set(external_PROG['CTHRC1_matrix_CAF']) | set(external_PROG['pan_CAF'])
    genes = sorted({g for _, s in mk.groupby('group') for g in s.names.head(external_EV['mapping']['markers_per_subtype'])} - excl)
    return (d, genes)

def external_centroids(a, key, genes):
    C = sp.csr_matrix(a.layers['counts'])
    lab = a.obs[key].astype(str).to_numpy()
    rows = {}
    for g in sorted(set(lab)):
        s = np.asarray(C[lab == g].sum(0)).ravel()
        rows[g] = np.log2(s / s.sum() * 1000000.0 + 1)
    M = pd.DataFrame(rows, index=a.var_names).T
    return M.reindex(columns=genes)

def external_map_clusters(c, out):
    d, genes = external_discovery_centroids()
    genes = [g for g in genes if g in c.var_names]
    D = external_centroids(d, 'leiden_0.5', genes).rename(index=external_CAF_NAMES)
    E = external_centroids(c, 'caf_cluster', genes)
    z = lambda M: (M - M.mean()) / M.std().replace(0, np.nan)
    Dz, Ez = (z(D).dropna(axis=1), z(E).dropna(axis=1))
    common = Dz.columns.intersection(Ez.columns)
    R = pd.DataFrame(np.corrcoef(Ez[common].to_numpy(), Dz[common].to_numpy())[:len(Ez), len(Ez):], index=Ez.index, columns=Dz.index)
    best = R.idxmax(1)
    mapped = best.where(R.max(1) >= external_EV['mapping']['min_r'], 'unmapped')
    R.assign(best=best, best_r=R.max(1), mapped=mapped).round(3).to_csv(out / 'cluster_mapping.tsv', sep='\t')
    print(f'mapping on {len(common)} genes (program / pan-CAF genes excluded)')
    print(R.round(2).assign(mapped=mapped).to_string(), flush=True)
    return mapped

def external_auroc(score, y):
    pos, neg = (score[y], score[~y])
    u = stats.mannwhitneyu(pos, neg, alternative='two-sided').statistic
    return float(u / (len(pos) * len(neg)))

def external_paired_lfc(c, y, genes):
    C = sp.csr_matrix(c.layers['counts'])
    rows = []
    for p, idx in c.obs.groupby('patient_key', observed=True).indices.items():
        t, r = (idx[y[idx]], idx[~y[idx]])
        if len(t) < 10 or len(r) < 10:
            continue
        cpm = lambda ii: np.log2(np.asarray(C[ii].sum(0)).ravel() / max(C[ii].sum(), 1) * 1000000.0 + 1)
        lt, lr = (cpm(t), cpm(r))
        rows.append(pd.Series({g: lt[c.var_names.get_loc(g)] - lr[c.var_names.get_loc(g)] for g in genes if g in c.var_names}, name=p))
    return pd.DataFrame(rows)

def external_run(ds: str):
    out = external_OUT / ds
    out.mkdir(parents=True, exist_ok=True)
    a = external_load(ds)
    a = external_integrate(a, max_epochs=80)
    external_annotate_lineage(a, out)
    c = a[a.obs.lineage == 'Fibroblast'].copy()
    print(f'fibroblast-lineage cells: {c.n_obs} from {c.obs.patient_key.nunique()} samples', flush=True)
    c = external_integrate(c, max_epochs=200)
    flags = external_flag_clusters(c)
    flags.round(3).to_csv(out / 'caf_cluster_flags.tsv', sep='\t')
    print(flags.round(2).to_string(), flush=True)
    c = c[~c.obs.leiden.astype(str).isin(flags.index[flags.exclude])].copy()
    c.obs['caf_cluster'] = 'E' + c.obs.leiden.astype(str)
    mapped = external_map_clusters(c, out)
    c.obs['mapped'] = c.obs.caf_cluster.map(mapped)
    y = (c.obs.mapped == external_TARGET).to_numpy()
    for k, g in external_COMPARE.items():
        idx = [c.var_names.get_loc(x) for x in g if x in c.var_names]
        c.obs[f'score_{k}'] = np.asarray(c.X[:, idx].mean(1)).ravel()
    res = {'dataset': ds, 'modality': external_EV['datasets'][ds]['modality'], 'n_cafs': int(c.n_obs), 'n_samples': int(c.obs.patient_key.nunique()), 'n_mapped_cthrc1': int(y.sum()), 'mapped_clusters': ','.join(sorted(mapped.index[mapped == external_TARGET]))}
    if y.sum() > 0 and (~y).sum() > 0:
        for k in external_COMPARE:
            res[f'auroc_{k}'] = external_auroc(c.obs[f'score_{k}'].to_numpy(), y)
        lfc = external_paired_lfc(c, y, external_PROG['CTHRC1_matrix_CAF'])
        lfc.to_csv(out / 'program_paired_log2fc.tsv', sep='\t')
        med = lfc.median()
        res.update({'n_paired_patients': len(lfc), 'genes_up': int((med > 0).sum()), 'genes_up_list': ','.join(med.index[med > 0])})
    st = c.obs.groupby('patient_key', observed=True).agg(n=('mapped', 'size'), frac=('mapped', lambda s: (s == external_TARGET).mean()), condition=('condition', 'first'))
    st.to_csv(out / 'per_sample_cthrc1_fraction.tsv', sep='\t')
    st = st[st.n >= external_EV['presence_min_cafs']]
    res.update({'samples_evaluable': len(st), 'samples_present': int((st.frac >= external_EV['presence_frac']).sum()), 'median_frac': float(st.frac.median()) if len(st) else np.nan})
    s = external_EV['success']
    res['validated'] = bool(y.sum() >= external_EV['min_mapped_cells_for_test'] and res.get('auroc_program', 0) >= s['min_auroc'] and (res.get('genes_up', 0) >= s['min_genes_up']))
    res['descriptive_only'] = bool(y.sum() < external_EV['min_mapped_cells_for_test'])
    pd.crosstab(c.obs.caf_cluster.astype(str) + ' → ' + c.obs.mapped, c.obs.author_label).to_csv(out / 'author_label_crosstab.tsv', sep='\t')
    sc.tl.rank_genes_groups(c, 'caf_cluster', method='wilcoxon', n_genes=30)
    sc.get.rank_genes_groups_df(c, None).to_csv(out / 'caf_cluster_markers.tsv', sep='\t', index=False)
    c.obs.to_csv(out / 'caf_obs.tsv.gz', sep='\t')
    pd.DataFrame(c.obsm['X_umap'], index=c.obs_names, columns=['UMAP1', 'UMAP2']).to_csv(out / 'caf_umap.tsv.gz', sep='\t')
    pd.Series(res).to_csv(out / 'result.tsv', sep='\t', header=False)
    print(pd.Series(res).to_string(), flush=True)

def external_summary():
    rows = [pd.read_csv(_read_path(f), sep='\t', header=None, index_col=0)[1] for f in _external_result_files()]
    if not rows:
        raise FileNotFoundError('No external result.tsv files found')
    s = pd.DataFrame(rows)
    s.to_csv(external_OUT / 'summary.tsv', sep='\t', index=False)
    print(s.T.to_string())

# fig1_17_pseudobulk_decoupling.py: retained numerical analysis.
def _init_pseudo():
    global Path, ad, np, pd, pseudo_CFG, pseudo_DC, pseudo_DROP, pseudo_OUT, pseudo_PROG, pseudo_PROJ, pseudo_PUB, pseudo_RNG, pseudo_ROOT, pseudo_SETS, pseudo_TARGET, sp, stats, yaml
    from pathlib import Path
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import yaml
    from scipy import stats
    pseudo_PROJ = _RUN_DIR.parent
    pseudo_CFG = _load_config()
    pseudo_DC = pseudo_CFG['decoupling']
    pseudo_ROOT = Path(pseudo_CFG['paths']['out_dir'])
    pseudo_OUT = pseudo_ROOT / 'decoupling'
    pseudo_OUT.mkdir(parents=True, exist_ok=True)
    pseudo_PROG = yaml.safe_load(_read_path(pseudo_ROOT / 'signature' / 'programs.yaml').read_text())
    pseudo_PUB = _load_published()
    pseudo_SETS = {'program': pseudo_PROG['CTHRC1_matrix_CAF'], 'Moffitt_activated': pseudo_PUB['Moffitt_activated_stroma']['genes'], 'Dominguez_LRRC15': pseudo_PUB['Dominguez_LRRC15_CAF']['genes'], 'pan_CAF': pseudo_PROG['pan_CAF']}
    pseudo_DROP = ['Ambiguous', 'Low_quality', 'Hepatocyte_like']
    pseudo_TARGET = '2'
    pseudo_RNG = np.random.default_rng(pseudo_CFG['random_seed'])

def pseudo_scores(P: np.ndarray, var: pd.Index) -> pd.DataFrame:
    """P: bulk x genes counts -> raw and relative signature scores."""
    L = np.log2(P / P.sum(1, keepdims=True) * 1000000.0 + 1)
    s = {}
    for k, g in pseudo_SETS.items():
        idx = [var.get_loc(x) for x in g if x in var]
        s[k] = L[:, idx].mean(1)
    s = pd.DataFrame(s)
    for k in pseudo_DC['compare']:
        s[f'{k}_rel'] = s[k] - s['pan_CAF']
    return s

def pseudo_partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return stats.pearsonr(res(rx), res(ry))[0]

def pseudo_load():
    full = ad.read_h5ad(_read_path(pseudo_ROOT / 'global_full_raw.h5ad'))
    ann = pd.read_csv(_read_path(pseudo_ROOT / 'annotate_global' / 'global_full_annotation.tsv.gz'), sep='\t', index_col=0)
    full = full[full.obs_names.isin(ann.index[~ann.lineage.isin(pseudo_DROP)])].copy()
    full.obs['lineage'] = ann.lineage.reindex(full.obs_names).values
    caf = ad.read_h5ad(_read_path(pseudo_ROOT / 'integrate_caf_final' / 'fibro_integrated.h5ad'), backed='r').obs
    caf = caf.assign(is_target=caf['leiden_0.5'].astype(str) == pseudo_TARGET)
    return (full, caf)

def pseudo_natural(full, caf):
    X = sp.csr_matrix(full.X)
    key = (full.obs.patient_key.astype(str) + '|' + full.obs.condition.astype(str)).to_numpy()
    uniq, inv = np.unique(key, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    P = np.asarray((G @ X).todense(), float)
    meta = pd.DataFrame({'sample': uniq})
    meta['dataset'] = full.obs.dataset.astype(str).groupby(key).first().reindex(uniq).values
    meta['condition'] = [u.split('|')[1] for u in uniq]
    meta['n_cells'] = np.bincount(inv)
    meta['caf_frac'] = pd.Series(full.obs.lineage.eq('Fibroblast').to_numpy()).groupby(inv).mean().values
    ck = caf.patient_key.astype(str) + '|' + caf.condition.astype(str)
    t = caf.groupby(ck).is_target.agg(['mean', 'size']).reindex(uniq)
    meta['cthrc1_frac'], meta['n_caf_atlas'] = (t['mean'].values, t['size'].values)
    d = pd.concat([meta, pseudo_scores(P, full.var_names)], axis=1)
    d = d[d.n_caf_atlas >= pseudo_DC['natural']['min_caf_atlas_cells']].reset_index(drop=True)
    d.to_csv(pseudo_OUT / 'natural_pseudobulk.tsv', sep='\t', index=False)
    rows = []
    for col in [c for c in d.columns if c in pseudo_SETS or c.endswith('_rel')]:
        rows.append({'score': col, 'n': len(d), 'rho_state': stats.spearmanr(d[col], d.cthrc1_frac)[0], 'rho_caf': stats.spearmanr(d[col], d.caf_frac)[0], 'partial_state': pseudo_partial_spearman(d[col], d.cthrc1_frac, d.caf_frac), 'partial_caf': pseudo_partial_spearman(d[col], d.caf_frac, d.cthrc1_frac)})
    r = pd.DataFrame(rows)
    r.to_csv(pseudo_OUT / 'natural_correlations.tsv', sep='\t', index=False)
    print(f'natural pseudo-bulk: {len(d)} samples; rho(caf_frac, cthrc1_frac) = {stats.spearmanr(d.caf_frac, d.cthrc1_frac)[0]:.2f}')
    print(r.round(2).to_string(index=False))
    return (d, r)

def pseudo_mixtures(full, caf):
    M = pseudo_DC['mixtures']
    X = sp.csr_matrix(full.X)
    pos = pd.Series(np.arange(full.n_obs), index=full.obs_names)
    tumour = full.obs.condition.isin(M['conditions']).to_numpy()
    caf_t = caf[caf.condition.isin(M['conditions'])]
    out = []
    for ds in sorted(full.obs.dataset.unique()):
        cd = caf_t[caf_t.dataset == ds]
        pools = {'cth': pos[cd.index[cd.is_target]].to_numpy(), 'oth': pos[cd.index[~cd.is_target]].to_numpy(), 'non': np.where(tumour & (full.obs.dataset == ds).to_numpy() & (full.obs.lineage != 'Fibroblast').to_numpy())[0]}
        print(f"{ds}: pools CTHRC1+ {len(pools['cth'])}, other CAF {len(pools['oth'])}, non-CAF {len(pools['non'])}")
        rows, mats = ([], [])
        for cf in M['caf_frac']:
            for tf in M['cthrc1_frac']:
                for rep in range(M['reps']):
                    n_caf = int(round(M['n_cells'] * cf))
                    n_cth = int(round(n_caf * tf))
                    idx = np.concatenate([pseudo_RNG.choice(pools['cth'], n_cth), pseudo_RNG.choice(pools['oth'], n_caf - n_cth), pseudo_RNG.choice(pools['non'], M['n_cells'] - n_caf)])
                    mats.append(np.asarray(X[idx].sum(0)).ravel())
                    rows.append({'dataset': ds, 'caf_frac': cf, 'cthrc1_frac': tf, 'rep': rep})
        d = pd.concat([pd.DataFrame(rows), pseudo_scores(np.vstack(mats), full.var_names)], axis=1)
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d.to_csv(pseudo_OUT / 'mixtures.tsv', sep='\t', index=False)
    rows = []
    for ds, g in d.groupby('dataset'):
        for col in [c for c in d.columns if c in pseudo_SETS or c.endswith('_rel')]:
            rows.append({'dataset': ds, 'score': col, 'rho_state': stats.spearmanr(g[col], g.cthrc1_frac)[0], 'rho_caf': stats.spearmanr(g[col], g.caf_frac)[0]})
    r = pd.DataFrame(rows)
    r.to_csv(pseudo_OUT / 'mixture_correlations.tsv', sep='\t', index=False)
    print(r.pivot(index='score', columns='dataset', values=['rho_state', 'rho_caf']).round(2).to_string())
    return (d, r)

def pseudo_verdict(rn, rm):
    s = pseudo_DC['success']
    m = rm[rm.score == 'program_rel']
    n = rn[rn.score == 'program_rel'].iloc[0]
    checks = {'mixtures: rho_state >= %.1f in every dataset' % s['mixtures_min_rho_state']: bool((m.rho_state >= s['mixtures_min_rho_state']).all()), 'mixtures: |rho_caf| <= %.1f in every dataset' % s['mixtures_max_abs_rho_caf']: bool((m.rho_caf.abs() <= s['mixtures_max_abs_rho_caf']).all()), 'natural: partial rho_state >= %.1f' % s['natural_min_rho_state']: bool(n.partial_state >= s['natural_min_rho_state']), 'natural: |partial rho_caf| <= %.1f' % s['natural_max_abs_rho_caf']: bool(abs(n.partial_caf) <= s['natural_max_abs_rho_caf'])}
    pd.Series(checks).to_csv(pseudo_OUT / 'verdict.tsv', sep='\t', header=False)
    print('\nVERDICT (program_rel):')
    for k, v in checks.items():
        print(f"  {('PASS' if v else 'FAIL')}  {k}")

def pseudo_main():
    full, caf = pseudo_load()
    dn, rn = pseudo_natural(full, caf)
    dm, rm = pseudo_mixtures(full, caf)
    pseudo_verdict(rn, rm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "figure1.yaml"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-root", help="Optional read-only results root containing fig1/ and fig1_external/.")
    parser.add_argument("--stage", choices=["published", "external", "external-summary", "pseudobulk"], required=True)
    parser.add_argument("--dataset", help="Required only for external; must be a configured accession.")
    args = parser.parse_args()
    targets = {"published":["fig1/published_signatures"], "pseudobulk":["fig1/decoupling"],
               "external-summary":["fig1_external/summary.tsv"], "external":[]}
    config_datasets = None
    if args.stage == "external":
        if not args.dataset or not args.dataset.replace("_", "").isalnum():
            parser.error("Provide a configured accession with --dataset.")
        targets["external"] = [f"fig1_external/{args.dataset}"]
    record, payload = _start(args, args.stage + ("_"+args.dataset if args.stage == "external" else ""), targets[args.stage])
    try:
        if args.stage == "published":
            published_main()
        elif args.stage == "pseudobulk":
            _init_pseudo()
            pseudo_main()
        else:
            _init_external()
            if args.stage == "external":
                if args.dataset not in external_EV["datasets"]:
                    raise ValueError("Accession is not present in the frozen external-validation configuration.")
                external_run(args.dataset)
            else:
                external_summary()
    except BaseException as error:
        _finish(record, payload, error)
        raise
    _finish(record, payload)


if __name__ == "__main__":
    main()
