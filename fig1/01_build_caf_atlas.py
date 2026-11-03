"""Figure 1 discovery atlas, QC and clustering. Numerical analysis only; original results are never overwritten."""
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



# fig1_00_build_global.py: retained numerical analysis.
def _init_inputs():
    global Path, ad, argparse, inputs_CAP, inputs_CFG, inputs_OUT, inputs_PANIN, inputs_RNG, np, pd, scipy, sp, sys, yaml
    import argparse
    import sys
    from pathlib import Path
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.io
    import scipy.sparse as sp
    import yaml
    inputs_CFG = _load_config()
    inputs_PANIN = Path(inputs_CFG['paths']['panin'])
    inputs_OUT = Path(inputs_CFG['paths']['out_dir'])
    inputs_OUT.mkdir(parents=True, exist_ok=True)
    inputs_RNG = np.random.default_rng(inputs_CFG['random_seed'])
    inputs_CAP = inputs_CFG['global_max_cells_per_patient']

def inputs_capped_index(meta: pd.DataFrame) -> np.ndarray:
    take = []
    for _, idx in meta.reset_index(drop=True).groupby(['patient_key', 'condition'], observed=True).indices.items():
        take.append(idx if inputs_CAP <= 0 or len(idx) <= inputs_CAP else inputs_RNG.choice(idx, inputs_CAP, replace=False))
    return np.sort(np.concatenate(take))

def inputs_from_h5ad(name: str, c: dict) -> ad.AnnData:
    a = ad.read_h5ad(_read_path(inputs_PANIN / c['h5ad']), backed='r')
    o = a.obs
    meta = pd.DataFrame({'dataset': name, 'patient_key': name + ':' + o[c['patient_col']].astype(str).values, 'condition': o[c['condition_col']].astype(str).map(c['condition_map']).fillna('other').values, 'author_label': o[c['lineage_col']].astype(str).values}, index=o.index.astype(str))
    meta = meta[~meta.condition.isin(c.get('exclude_conditions', []))]
    pos = np.where(o.index.astype(str).isin(meta.index))[0]
    sel = pos[inputs_capped_index(meta)]
    src = a.layers['counts'] if 'counts' in a.layers else a.X
    X = sp.vstack([sp.csr_matrix(src[sel[i:i + 20000]]) for i in range(0, len(sel), 20000)]).tocsr()
    return ad.AnnData(X, obs=meta.loc[o.index[sel].astype(str)], var=pd.DataFrame(index=a.var_names.astype(str)))

def inputs_from_mtx(name: str, c: dict) -> ad.AnnData:
    d = inputs_PANIN / c['mtx_dir']
    feats = pd.read_csv(_read_path(d / f"{c['prefix']}features.tsv.gz"), sep='\t', header=None)
    bc = pd.read_csv(_read_path(d / f"{c['prefix']}barcodes.tsv.gz"), sep='\t', header=None)[0].astype(str)
    md = pd.read_csv(_read_path(d / f"{c['prefix']}metadata.csv.gz"), index_col=0).loc[bc.values]
    M = scipy.io.mmread(_read_path(d / f"{c['prefix']}matrix.mtx.gz")).tocsr()
    if M.shape == (len(feats), len(bc)):
        M = M.T.tocsr()
    meta = pd.DataFrame({'dataset': name, 'patient_key': name + ':' + md[c['patient_col']].astype(str).values, 'condition': md[c['condition_col']].astype(str).map(c['condition_map']).fillna(c.get('default_condition', 'other')).values, 'author_label': md[c['lineage_col']].astype(str).values}, index=bc.values)
    keep_rows = np.where(~meta.condition.isin(c.get('exclude_conditions', [])).to_numpy())[0]
    meta = meta.iloc[keep_rows]
    sel = keep_rows[inputs_capped_index(meta)]
    genes = pd.Index(feats[1].astype(str))
    gk = np.where(~genes.duplicated())[0]
    obs = meta.loc[bc.values[sel]]
    obs.index.name = None
    return ad.AnnData(M[sel][:, gk], obs=obs, var=pd.DataFrame(index=pd.Index(genes[gk].values, name=None)))

def inputs_main():
    global inputs_CAP
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=int, default=inputs_CFG['global_max_cells_per_patient'], help='0 = keep all cells')
    ap.add_argument('--name', default='global_raw.h5ad')
    args = ap.parse_args()
    inputs_CAP = args.cap
    parts = []
    for name, c in inputs_CFG['datasets'].items():
        a = inputs_from_mtx(name, c) if 'mtx_dir' in c else inputs_from_h5ad(name, c)
        a.var_names_make_unique()
        print(f'{name}: {a.n_obs} cells (cap {inputs_CAP}/patient×condition), {a.obs.patient_key.nunique()} patients; conditions {a.obs.condition.value_counts().to_dict()}', flush=True)
        parts.append(a)
    common = sorted(set.intersection(*[set(p.var_names) for p in parts]))
    g = ad.concat([p[:, common] for p in parts], join='inner')
    g.obs_names_make_unique()
    g.layers['counts'] = g.X.copy()
    for col in ['dataset', 'patient_key', 'condition', 'author_label']:
        g.obs[col] = g.obs[col].astype('category')
    g.obs['in_cap'] = True
    g.write_h5ad(inputs_OUT / args.name, compression='gzip')
    print(f'global: {g.shape}')
    print(pd.crosstab(g.obs.author_label, g.obs.dataset).to_string())

# fig1_01_build_inputs.py: retained numerical analysis.
def _init_legacy():
    global Path, ad, legacy_CFG, legacy_OUT, legacy_PANIN, legacy_RNG, np, pd, scipy, sp, sys, yaml
    import sys
    from pathlib import Path
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.io
    import scipy.sparse as sp
    import yaml
    legacy_CFG = _load_config()
    legacy_PANIN = Path(legacy_CFG['paths']['panin'])
    legacy_OUT = Path(legacy_CFG['paths']['out_dir'])
    legacy_OUT.mkdir(parents=True, exist_ok=True)
    legacy_RNG = np.random.default_rng(legacy_CFG['random_seed'])

def legacy_counts_rows(a: ad.AnnData, idx: np.ndarray, chunk: int=20000) -> sp.csr_matrix:
    """Raw counts for selected rows of a backed AnnData, in chunks."""
    src = a.layers['counts'] if 'counts' in a.layers else a.X
    parts = []
    for i in range(0, len(idx), chunk):
        block = src[np.sort(idx[i:i + chunk])]
        parts.append(sp.csr_matrix(block))
    return sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, a.n_vars))

def legacy_pseudobulk(X: sp.csr_matrix, keys: pd.Series) -> tuple[pd.Index, sp.csr_matrix]:
    codes, uniq = pd.factorize(keys, sort=True)
    G = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
    return (pd.Index(uniq), (G @ X).tocsr())

def legacy_load_h5ad_dataset(name: str, c: dict):
    a = ad.read_h5ad(_read_path(legacy_PANIN / c['h5ad']), backed='r')
    obs = a.obs
    lin = obs[c['lineage_col']].astype(str)
    patient = obs[c['patient_col']].astype(str)
    cond = obs[c['condition_col']].astype(str).map(c['condition_map']).fillna('other')
    meta = pd.DataFrame({'dataset': name, 'patient_key': name + ':' + patient.values, 'condition': cond.values, 'author_label': lin.values}, index=obs.index.astype(str))
    pb_keys = meta.dataset + '|' + meta.author_label + '|' + meta.condition + '|' + meta.patient_key
    keys_all, sums = (None, None)
    n = a.n_obs
    acc = {}
    for i in range(0, n, 20000):
        rows = np.arange(i, min(i + 20000, n))
        X = legacy_counts_rows(a, rows)
        k, S = legacy_pseudobulk(X, pb_keys.iloc[rows].reset_index(drop=True))
        for j, key in enumerate(k):
            acc[key] = acc[key] + S[j] if key in acc else S[j]
        print(f'   {name}: pseudobulk {min(i + 20000, n)}/{n}', flush=True)
    keys_all = pd.Index(sorted(acc))
    sums = sp.vstack([acc[k] for k in keys_all]).tocsr()
    fib_idx = np.where(lin.isin(c['fibro_labels']).to_numpy())[0]
    Xf = legacy_counts_rows(a, fib_idx)
    return (meta.iloc[fib_idx], Xf, pd.Index(a.var_names), keys_all, sums)

def legacy_load_mtx_dataset(name: str, c: dict):
    d = legacy_PANIN / c['mtx_dir']
    feats = pd.read_csv(_read_path(d / f"{c['prefix']}features.tsv.gz"), sep='\t', header=None)
    barcodes = pd.read_csv(_read_path(d / f"{c['prefix']}barcodes.tsv.gz"), sep='\t', header=None)[0].astype(str)
    md = pd.read_csv(_read_path(d / f"{c['prefix']}metadata.csv.gz"), index_col=0)
    M = scipy.io.mmread(_read_path(d / f"{c['prefix']}matrix.mtx.gz")).tocsr()
    if M.shape[0] == len(feats) and M.shape[1] == len(barcodes):
        M = M.T.tocsr()
    assert M.shape == (len(barcodes), len(feats)), M.shape
    md = md.loc[barcodes.values]
    lin = md[c['lineage_col']].astype(str)
    cond = md[c['condition_col']].astype(str).map(c['condition_map']).fillna(c.get('default_condition', 'other'))
    patient = md[c['patient_col']].astype(str)
    meta = pd.DataFrame({'dataset': name, 'patient_key': name + ':' + patient.values, 'condition': cond.values, 'author_label': lin.values}, index=barcodes.values)
    genes = pd.Index(feats[1].astype(str))
    keep = ~genes.duplicated()
    M, genes = (M[:, np.where(keep)[0]], genes[keep])
    pb_keys = (meta.dataset + '|' + meta.author_label + '|' + meta.condition + '|' + meta.patient_key).reset_index(drop=True)
    keys_all, sums = legacy_pseudobulk(M, pb_keys)
    fib_idx = np.where(lin.isin(c['fibro_labels']).to_numpy())[0]
    return (meta.iloc[fib_idx], M[fib_idx], genes, keys_all, sums)

def legacy_main():
    fibro_parts, pb_parts = ([], [])
    for name, c in legacy_CFG['datasets'].items():
        print(f'== {name}', flush=True)
        loader = legacy_load_mtx_dataset if 'mtx_dir' in c else legacy_load_h5ad_dataset
        meta, Xf, genes, pk, ps = loader(name, c)
        gi = ~pd.Index(genes).duplicated()
        fibro_parts.append(ad.AnnData(sp.csr_matrix(Xf[:, np.where(gi)[0]]), obs=meta, var=pd.DataFrame(index=pd.Index(genes)[gi])))
        pb = ad.AnnData(sp.csr_matrix(ps[:, np.where(gi)[0]]), var=pd.DataFrame(index=pd.Index(genes)[gi]))
        parts = pd.Series(pk).str.split('|', expand=True)
        pb.obs = pd.DataFrame({'dataset': parts[0].values, 'author_label': parts[1].values, 'condition': parts[2].values, 'patient_key': parts[3].values}, index=pd.Index(pk))
        pb.obs['n_cells'] = np.nan
        pb_parts.append(pb)
        print(f'   fibroblast-lineage cells: {Xf.shape[0]}; genes: {gi.sum()}', flush=True)
    common = sorted(set.intersection(*[set(p.var_names) for p in fibro_parts]))
    print(f'common genes across datasets: {len(common)}')
    fib = ad.concat([p[:, common] for p in fibro_parts], join='inner', merge='same')
    fib.obs_names_make_unique()
    fib.layers['counts'] = fib.X.copy()
    fib.obs['n_counts'] = np.asarray(fib.X.sum(1)).ravel()
    fib.obs['n_genes'] = np.asarray((fib.X > 0).sum(1)).ravel()
    cap = legacy_CFG['max_cells_per_patient']
    fib.obs['in_cap'] = False
    for pk, idx in fib.obs.groupby(['patient_key', 'condition'], observed=True).indices.items():
        take = idx if len(idx) <= cap else legacy_RNG.choice(idx, cap, replace=False)
        fib.obs.iloc[take, fib.obs.columns.get_loc('in_cap')] = True
    fib.obs['discovery'] = fib.obs.condition.isin(legacy_CFG['discovery_conditions'])
    for col in ['dataset', 'patient_key', 'condition', 'author_label']:
        fib.obs[col] = fib.obs[col].astype('category')
    fib.write_h5ad(legacy_OUT / 'fibro_raw.h5ad', compression='gzip')
    pb = ad.concat([p[:, common] for p in pb_parts], join='inner')
    pb.write_h5ad(legacy_OUT / 'lineage_pseudobulk.h5ad', compression='gzip')
    s = fib.obs.groupby(['dataset', 'condition', 'author_label'], observed=True).agg(cells=('patient_key', 'size'), patients=('patient_key', 'nunique')).reset_index()
    s['cells_in_cap'] = fib.obs[fib.obs.in_cap].groupby(['dataset', 'condition', 'author_label'], observed=True).size().values
    s.to_csv(legacy_OUT / 'inputs_summary.tsv', sep='\t', index=False)
    print(s.to_string(index=False))
    print(f'fibro: {fib.shape}; discovery in cap: {(fib.obs.discovery & fib.obs.in_cap).sum()}; pseudobulk groups: {pb.n_obs}')

# fig1_02_integrate_cluster.py: retained numerical analysis.
def _init_integration():
    global Path, argparse, integration_CFG, integration_MAIN_RES, integration_OUT, integration_RES, np, os, pd, sc, scvi, torch, yaml
    import os
    os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
    from pathlib import Path
    import argparse
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import yaml
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    import scvi
    import torch
    torch.set_num_threads(2)
    integration_CFG = _load_config()
    integration_RES = [0.3, 0.5, 0.8, 1.2]
    integration_MAIN_RES = 0.5
    scvi.settings.seed = integration_CFG['random_seed']

def integration_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='integrate')
    ap.add_argument('--input', default='fibro_raw.h5ad', help='file in results/fig1/')
    ap.add_argument('--mode', default='caf', choices=['caf', 'global'], help='caf: contaminant + CAF-subtype scores; global: major-lineage marker scores')
    ap.add_argument('--max-epochs', type=int, default=300)
    ap.add_argument('--exclude-from', default=None, help='previous fibro_integrated.h5ad')
    ap.add_argument('--exclude-key', default='leiden_0.5')
    ap.add_argument('--exclude', default='', help='comma-separated cluster ids to drop')
    args = ap.parse_args()
    global integration_OUT
    integration_OUT = Path(integration_CFG['paths']['out_dir']) / args.out
    integration_OUT.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(integration_OUT)
    a = sc.read_h5ad(_read_path(Path(integration_CFG['paths']['out_dir']) / args.input))
    a = a[a.obs.in_cap].copy()
    if args.exclude_from:
        prev = sc.read_h5ad(_read_path(args.exclude_from), backed='r').obs[[args.exclude_key]]
        drop = set(args.exclude.split(','))
        keep = prev.index[~prev[args.exclude_key].astype(str).isin(drop)]
        a = a[a.obs_names.isin(keep)].copy()
        a.obs['prev_cluster'] = prev.loc[a.obs_names, args.exclude_key].astype(str).values
        print(f'excluded clusters {sorted(drop)} from {args.exclude_from}', flush=True)
    print(f'cells: {a.n_obs}; patients: {a.obs.patient_key.nunique()}', flush=True)
    sc.pp.filter_genes(a, min_cells=20)
    a.layers['counts'] = a.X.copy()
    sc.pp.highly_variable_genes(a, flavor='seurat_v3', n_top_genes=3000, layer='counts', batch_key='dataset', subset=False)
    sc.pp.normalize_total(a, target_sum=10000.0)
    sc.pp.log1p(a)
    hv = a[:, a.var.highly_variable].copy()
    scvi.model.SCVI.setup_anndata(hv, layer='counts', batch_key='patient_key', categorical_covariate_keys=['dataset'])
    model = scvi.model.SCVI(hv, n_latent=20, n_layers=2, gene_likelihood='nb')
    model.train(max_epochs=args.max_epochs, early_stopping=True, accelerator='cpu', batch_size=512, plan_kwargs={'lr': 0.001})
    model.save(integration_OUT / 'scvi_model', overwrite=True)
    a.obsm['X_scVI'] = model.get_latent_representation()
    sc.pp.neighbors(a, use_rep='X_scVI', n_neighbors=15, random_state=0)
    for r in integration_RES:
        sc.tl.leiden(a, resolution=r, key_added=f'leiden_{r:g}', flavor='igraph', n_iterations=2, random_state=0)
    sc.tl.umap(a, random_state=0)
    var = set(a.var_names)
    if args.mode == 'global':
        score_sets = {f'lin_{k}': v for k, v in integration_CFG['lineage_markers'].items()}
    else:
        score_sets = {**{f'contam_{k}': v for k, v in integration_CFG['contaminant_markers'].items()}, **{f'ref_{k}': v for k, v in integration_CFG['caf_reference'].items()}}
    for k, genes in score_sets.items():
        g = [x for x in genes if x in var]
        sc.tl.score_genes(a, g, score_name=k, random_state=0)
    a.obs['CTHRC1_expr'] = np.asarray(a[:, 'CTHRC1'].X.todense()).ravel()
    for r in integration_RES:
        key = f'leiden_{r:g}'
        grp = a.obs.groupby(key, observed=True)
        prof = pd.concat([grp.size().rename('n_cells'), grp.patient_key.nunique().rename('n_patients'), pd.crosstab(a.obs[key], a.obs.dataset, normalize='index').add_prefix('ds_'), pd.crosstab(a.obs[key], a.obs.condition, normalize='index').add_prefix('cond_'), grp[[c for c in a.obs.columns if c.startswith(('contam_', 'ref_', 'lin_'))]].mean(), grp.CTHRC1_expr.mean().rename('CTHRC1_mean'), grp.CTHRC1_expr.apply(lambda v: (v > 0).mean()).rename('CTHRC1_frac_pos')], axis=1).round(3)
        prof.to_csv(integration_OUT / f'cluster_profile_{r:g}.tsv', sep='\t')
        if r == integration_MAIN_RES:
            print(prof.to_string())
            sc.tl.rank_genes_groups(a, key, method='wilcoxon', n_genes=50, use_raw=False)
            mk = sc.get.rank_genes_groups_df(a, None)
            mk.to_csv(integration_OUT / f'markers_{r:g}.tsv', sep='\t', index=False)
            top = mk.groupby('group').head(8).groupby('group').names.apply(lambda s: ', '.join(s))
            print(top.to_string())
    a.write_h5ad(integration_OUT / 'fibro_integrated.h5ad', compression='gzip')
    print(f'wrote {integration_OUT}')

# fig1_03_annotate_transfer.py: retained numerical analysis.
def _init_annotation():
    global KNeighborsClassifier, Path, annotation_CFG, annotation_MESENCHYMAL, annotation_OUT, annotation_ROOT, argparse, np, os, pd, sc, scvi, torch, yaml
    import argparse
    import os
    os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import yaml
    from sklearn.neighbors import KNeighborsClassifier
    import scvi
    import torch
    torch.set_num_threads(2)
    annotation_CFG = _load_config()
    annotation_ROOT = Path(annotation_CFG['paths']['out_dir'])
    annotation_OUT = annotation_ROOT / 'annotate_global'
    annotation_OUT.mkdir(parents=True, exist_ok=True)
    annotation_MESENCHYMAL = ['Fibroblast', 'Pericyte', 'Smooth_muscle', 'Pericyte_SMC']

def annotation_auto_annotate(a, key, min_score, min_margin):
    lin_cols = [c for c in a.obs.columns if c.startswith('lin_') and c != 'lin_Cycling']
    m = a.obs.groupby(key, observed=True)[lin_cols + ['lin_Cycling']].mean()
    rows = []
    for cl, r in m.iterrows():
        s = r[lin_cols].sort_values(ascending=False)
        top, second = (s.index[0][4:], s.index[1][4:])
        label = top if s.iloc[0] >= min_score and s.iloc[0] - s.iloc[1] >= min_margin else 'Ambiguous'
        rows.append({'cluster': str(cl), 'auto_label': label, 'top': top, 'top_score': round(s.iloc[0], 3), 'second': second, 'second_score': round(s.iloc[1], 3), 'cycling_score': round(r['lin_Cycling'], 3)})
    tab = pd.DataFrame(rows).set_index('cluster')
    tab = tab.join(m.round(3))
    tab['n_cells'] = a.obs[key].value_counts().reindex(tab.index.astype(str)).values
    return tab

def annotation_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='leiden_0.5')
    ap.add_argument('--min-score', type=float, default=0.3)
    ap.add_argument('--min-margin', type=float, default=0.2)
    ap.add_argument('--min-conf', type=float, default=0.7)
    ap.add_argument('--annotate-only', action='store_true', help='write cluster_annotation.tsv and stop')
    args = ap.parse_args()
    g = sc.read_h5ad(_read_path(annotation_ROOT / 'integrate_global' / 'fibro_integrated.h5ad'))
    g.obs[args.key] = g.obs[args.key].astype(str)
    tab = annotation_auto_annotate(g, args.key, args.min_score, args.min_margin)
    overrides = {str(k): v for k, v in (annotation_CFG.get('global_annotation_overrides') or {}).items()}
    tab['override'] = tab.index.map(lambda c: overrides.get(c, ''))
    tab['final_label'] = np.where(tab.override != '', tab.override, tab.auto_label)
    tab.to_csv(annotation_OUT / 'cluster_annotation.tsv', sep='\t')
    print(tab[['n_cells', 'auto_label', 'top_score', 'second', 'second_score', 'cycling_score', 'override', 'final_label']].to_string())
    if args.annotate_only:
        return
    g.obs['lineage'] = g.obs[args.key].map(tab.final_label).values
    full = sc.read_h5ad(_read_path(annotation_ROOT / 'global_full_raw.h5ad'))
    model_dir = annotation_ROOT / 'integrate_global' / 'scvi_model'
    hv_genes = list(torch.load(_read_path(model_dir / 'model.pt'), map_location='cpu', weights_only=False)['var_names'])
    full_hv = full[:, hv_genes].copy()
    full_hv.layers['counts'] = full_hv.X.copy()
    lat = scvi.model.SCVI.load(_read_path(model_dir), adata=full_hv).get_latent_representation(full_hv, batch_size=4096)
    ref_ok = ~g.obs.lineage.isin(['Ambiguous', 'Low_quality'])
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=2)
    knn.fit(g.obsm['X_scVI'][ref_ok.to_numpy()], g.obs.lineage[ref_ok].to_numpy())
    proba = knn.predict_proba(lat)
    full.obs['lineage'] = knn.classes_[proba.argmax(1)]
    full.obs['lineage_conf'] = proba.max(1)
    in_ref = full.obs_names.isin(g.obs_names)
    full.obs.loc[in_ref, 'lineage'] = g.obs.lineage.reindex(full.obs_names[in_ref]).values
    full.obs.loc[in_ref, 'lineage_conf'] = 1.0
    ann = full.obs[['dataset', 'patient_key', 'condition', 'author_label', 'lineage', 'lineage_conf']].copy()
    ann.to_csv(annotation_OUT / 'global_full_annotation.tsv.gz', sep='\t')
    pd.crosstab([ann.dataset, ann.author_label], ann.lineage).to_csv(annotation_OUT / 'lineage_vs_author.tsv', sep='\t')
    comp = pd.crosstab([ann.dataset, ann.condition], ann.lineage, normalize='index').round(4)
    comp.to_csv(annotation_OUT / 'lineage_composition.tsv', sep='\t')
    print(pd.crosstab(ann.dataset, ann.lineage).to_string())
    print(f'label confidence: median {ann.lineage_conf.median():.2f}; < {args.min_conf}: {(ann.lineage_conf < args.min_conf).mean():.1%}')
    rng = np.random.default_rng(annotation_CFG['random_seed'])
    mes = full[full.obs.lineage.isin(annotation_MESENCHYMAL) & (full.obs.lineage_conf >= args.min_conf)].copy()
    mes.obs['in_cap'] = False
    cap = annotation_CFG['max_cells_per_patient']
    for _, idx in mes.obs.groupby(['patient_key', 'condition'], observed=True).indices.items():
        take = idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)
        mes.obs.iloc[take, mes.obs.columns.get_loc('in_cap')] = True
    mes.obs['discovery'] = mes.obs.condition.isin(annotation_CFG['discovery_conditions'])
    mes.layers['counts'] = mes.X.copy()
    mes.write_h5ad(annotation_ROOT / 'mesenchymal_raw.h5ad', compression='gzip')
    print(f'mesenchymal cells: {mes.n_obs} ({mes.obs.lineage.value_counts().to_dict()}); in cap: {mes.obs.in_cap.sum()}')

# fig1_07_doublets_qc.py: retained numerical analysis.
def _init_qc():
    global Path, np, os, pd, qc_CFG, qc_MIN_CELLS_FOR_SCRUBLET, qc_OUT, qc_ROOT, sc, sp, yaml
    import os
    os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    import yaml
    qc_CFG = _load_config()
    qc_ROOT = Path(qc_CFG['paths']['out_dir'])
    qc_OUT = qc_ROOT / 'qc'
    qc_OUT.mkdir(parents=True, exist_ok=True)
    qc_MIN_CELLS_FOR_SCRUBLET = 200

def qc_main():
    a = sc.read_h5ad(_read_path(qc_ROOT / 'global_full_raw.h5ad'))
    a.X = sp.csr_matrix(a.X)
    a.obs['sample'] = (a.obs.patient_key.astype(str) + '|' + a.obs.condition.astype(str)).values
    v = a.var_names.astype(str)
    tot = np.asarray(a.X.sum(1)).ravel()
    a.obs['n_umi'] = tot
    a.obs['n_genes'] = np.asarray((a.X > 0).sum(1)).ravel()
    a.obs['pct_mt'] = np.asarray(a.X[:, np.where(v.str.startswith('MT-'))[0]].sum(1)).ravel() / np.maximum(tot, 1) * 100
    a.obs['doublet_score'] = np.nan
    a.obs['predicted_doublet'] = False
    sizes = a.obs['sample'].value_counts()
    for i, (s, n) in enumerate(sizes.items(), 1):
        if n < qc_MIN_CELLS_FOR_SCRUBLET:
            continue
        sub = a[a.obs['sample'] == s].copy()
        try:
            sc.pp.scrublet(sub, expected_doublet_rate=min(0.008 * n / 1000, 0.1), random_state=0, verbose=False)
        except Exception as e:
            print(f'  scrublet failed for {s} (n={n}): {e}')
            continue
        a.obs.loc[sub.obs_names, 'doublet_score'] = sub.obs['doublet_score'].values
        a.obs.loc[sub.obs_names, 'predicted_doublet'] = sub.obs['predicted_doublet'].fillna(False).values
        if i % 20 == 0:
            print(f'  {i}/{len(sizes)} samples', flush=True)
    qc = a.obs[['dataset', 'patient_key', 'condition', 'sample', 'author_label', 'doublet_score', 'predicted_doublet', 'n_genes', 'n_umi', 'pct_mt']]
    qc.to_csv(qc_OUT / 'cell_qc.tsv.gz', sep='\t')
    ann = pd.read_csv(_read_path(qc_ROOT / 'annotate_global' / 'global_full_annotation.tsv.gz'), sep='\t', index_col=0)
    qc = qc.join(ann[['lineage']])
    summ = qc.groupby(['dataset', 'lineage']).agg(n=('predicted_doublet', 'size'), doublet_rate=('predicted_doublet', 'mean'), median_score=('doublet_score', 'median')).round(3)
    summ.to_csv(qc_OUT / 'doublet_summary.tsv', sep='\t')
    print(qc.groupby('dataset').predicted_doublet.mean().round(3).to_string())
    print(summ.xs('Fibroblast', level='lineage').to_string())

# fig1_08_clean.py: retained numerical analysis.
def _init_clean():
    global Path, ad, argparse, clean_CFG, clean_Q, clean_ROOT, clean_X_RULES, np, pd, sp, yaml
    import argparse
    from pathlib import Path
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import yaml
    clean_CFG = _load_config()
    clean_ROOT = Path(clean_CFG['paths']['out_dir'])
    clean_Q, clean_X_RULES = (clean_CFG['cell_qc'], clean_CFG['cluster_exclusion'])

def clean_mad(x):
    return np.median(np.abs(x - np.median(x))) * 1.4826

def clean_cmd_cells():
    mes = ad.read_h5ad(_read_path(clean_ROOT / 'mesenchymal_raw.h5ad'))
    qc = pd.read_csv(_read_path(clean_ROOT / 'qc' / 'cell_qc.tsv.gz'), sep='\t', index_col=0)
    o = mes.obs.join(qc[['doublet_score', 'predicted_doublet', 'n_genes', 'n_umi', 'pct_mt']], how='left')
    reason = pd.Series('', index=o.index)
    for ds, d in o.groupby('dataset', observed=True):
        lg, lu = (np.log(d.n_genes.clip(lower=1)), np.log(d.n_umi.clip(lower=1)))
        g_min = max(clean_Q['min_genes'], float(np.exp(np.median(lg) - clean_Q['mad_k'] * clean_mad(lg))))
        u_min = float(np.exp(np.median(lu) - clean_Q['mad_k'] * clean_mad(lu)))
        mt_max = max(clean_Q['max_mt_floor'], float(np.median(d.pct_mt) + clean_Q['mad_k'] * clean_mad(d.pct_mt)))
        r = pd.Series('', index=d.index)
        r[d.n_genes < g_min] += 'low_genes;'
        r[d.n_umi < u_min] += 'low_umi;'
        r[d.pct_mt > mt_max] += 'high_mt;'
        if clean_Q['remove_predicted_doublets']:
            r[d.predicted_doublet.fillna(False).astype(bool)] += 'doublet;'
        reason.loc[d.index] = r
        print(f"{ds}: thresholds genes>={g_min:.0f}, umi>={u_min:.0f}, mt<={mt_max:.1f}% | removed {(r != '').sum()} / {len(d)}")
    removed = o.loc[reason != '', ['dataset', 'patient_key', 'condition', 'lineage', 'n_genes', 'n_umi', 'pct_mt', 'doublet_score']].assign(reason=reason[reason != ''])
    removed.to_csv(clean_ROOT / 'qc' / 'removed_cells.tsv.gz', sep='\t')
    summ = removed.assign(reason=removed.reason.str.rstrip(';')).groupby(['dataset', 'reason']).size()
    print(summ.to_string())
    keep = reason == ''
    clean = mes[keep.reindex(mes.obs_names).to_numpy()].copy()
    rng = np.random.default_rng(clean_CFG['random_seed'])
    clean.obs['in_cap'] = False
    cap = clean_CFG['max_cells_per_patient']
    for _, idx in clean.obs.groupby(['patient_key', 'condition'], observed=True).indices.items():
        take = idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)
        clean.obs.iloc[take, clean.obs.columns.get_loc('in_cap')] = True
    clean.write_h5ad(clean_ROOT / 'mesenchymal_clean_raw.h5ad', compression='gzip')
    print(f'clean mesenchymal: {clean.n_obs} cells ({clean.obs.lineage.value_counts().to_dict()}); in cap {clean.obs.in_cap.sum()}')

def clean_cmd_flag(integrated: Path, key: str):
    a = ad.read_h5ad(_read_path(integrated))
    C = sp.csr_matrix(a.layers['counts'])
    v = list(a.var_names)
    pos = {g: C[:, v.index(g)].toarray().ravel() > 0 for g in ['RGS5', 'NOTCH3', 'MYH11', 'PDGFRA'] if g in v}
    df = pd.DataFrame({'cl': a.obs[key].astype(str).values, 'n_genes': np.asarray((C > 0).sum(1)).ravel(), **pos, 'epi': a.obs.get('contam_epithelial', pd.Series(0, index=a.obs_names)).values, 'imm': a.obs.get('contam_immune', pd.Series(0, index=a.obs_names)).values, 'patient': a.obs.patient_key.astype(str).values})
    overall = df.n_genes.median()
    g = df.groupby('cl')
    f = pd.DataFrame({'n_cells': g.size(), 'n_patients': g.patient.nunique(), 'pericyte_smc_mean_frac': g[['RGS5', 'NOTCH3', 'MYH11']].mean().mean(1), 'PDGFRA_frac': g.PDGFRA.mean(), 'median_genes_ratio': g.n_genes.median() / overall, 'epithelial_score': g.epi.mean(), 'immune_score': g.imm.mean()}).round(3)
    reasons = []
    for cl, r in f.iterrows():
        why = []
        if r.pericyte_smc_mean_frac >= clean_X_RULES['pericyte_smc_mean_frac']:
            why.append('pericyte/SMC markers')
        if r.median_genes_ratio < clean_X_RULES['low_complexity_ratio']:
            why.append('low complexity')
        if r.epithelial_score > clean_X_RULES['epithelial_score']:
            why.append('epithelial contamination')
        if r.immune_score > clean_X_RULES['immune_score']:
            why.append('immune contamination')
        reasons.append('; '.join(why))
    f['exclude_reason'] = reasons
    f.to_csv(integrated.parent / 'cluster_flags.tsv', sep='\t')
    print(f.to_string())
    ex = [str(c) for c in f.index[f.exclude_reason != '']]
    print(f"\n--exclude {(','.join(ex) if ex else '(none)')}")

def clean_main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('cells')
    fl = sub.add_parser('flag')
    fl.add_argument('integrated')
    fl.add_argument('--key', default='leiden_0.5')
    args = ap.parse_args()
    if args.cmd == 'cells':
        clean_cmd_cells()
    else:
        clean_cmd_flag(Path(args.integrated), args.key)

# fig1_09_stability.py: retained numerical analysis.
def _init_stability():
    global Path, adjusted_rand_score, np, os, pd, sc, scvi, stability_CFG, stability_D, stability_TARGET, torch, yaml
    import os
    os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import yaml
    from sklearn.metrics import adjusted_rand_score
    import scvi
    import torch
    torch.set_num_threads(2)
    stability_CFG = _load_config()
    stability_D = Path(stability_CFG['paths']['out_dir']) / 'integrate_caf_final'
    stability_TARGET = {**stability_CFG['signature'], **stability_CFG.get('signature_v3', {})}['target_clusters'][0]

def stability_best_match(ref_mask: np.ndarray, labels: pd.Series) -> tuple[float, str, int]:
    """Jaccard of the reference cell set with the best-overlapping cluster; and #clusters holding >=10% of it."""
    best, best_c = (0.0, '')
    for c in labels.unique():
        m = (labels == c).to_numpy()
        j = (ref_mask & m).sum() / (ref_mask | m).sum()
        if j > best:
            best, best_c = (j, c)
    share = labels[ref_mask].value_counts(normalize=True)
    return (round(best, 3), best_c, int((share >= 0.1).sum()))

def stability_main():
    a = sc.read_h5ad(_read_path(stability_D / 'fibro_integrated.h5ad'))
    main_lab = a.obs['leiden_0.5'].astype(str)
    ref = (main_lab == stability_TARGET).to_numpy()
    rows = []
    for seed in range(1, 6):
        lab = sc.tl.leiden(a, resolution=0.5, flavor='igraph', n_iterations=2, random_state=seed, key_added='_s', copy=True).obs['_s'].astype(str)
        j, c, k = stability_best_match(ref, lab)
        rows.append({'test': f'leiden seed {seed}', 'ARI_vs_main': round(adjusted_rand_score(main_lab, lab), 3), 'CTHRC1_cluster_jaccard': j, 'split_into': k})
    for r in [0.3, 0.5, 0.8, 1.2]:
        lab = a.obs[f'leiden_{r:g}'].astype(str)
        j, c, k = stability_best_match(ref, lab)
        rows.append({'test': f'resolution {r:g}', 'ARI_vs_main': round(adjusted_rand_score(main_lab, lab), 3), 'CTHRC1_cluster_jaccard': j, 'split_into': k})
    hv = a[:, a.var.highly_variable].copy()
    hv.X = hv.layers['counts'].copy()
    for seed in [101, 202]:
        scvi.settings.seed = seed
        scvi.model.SCVI.setup_anndata(hv, layer='counts', batch_key='patient_key', categorical_covariate_keys=['dataset'])
        m = scvi.model.SCVI(hv, n_latent=20, n_layers=2, gene_likelihood='nb')
        m.train(max_epochs=300, early_stopping=True, accelerator='cpu', batch_size=512)
        b = sc.AnnData(obs=a.obs[['patient_key']].copy())
        b.obsm['X_scVI'] = m.get_latent_representation()
        sc.pp.neighbors(b, use_rep='X_scVI', n_neighbors=15, random_state=0)
        lab = sc.tl.leiden(b, resolution=0.5, flavor='igraph', n_iterations=2, random_state=0, key_added='_m', copy=True).obs['_m'].astype(str)
        j, c, k = stability_best_match(ref, lab)
        rows.append({'test': f'scVI retrained seed {seed}', 'ARI_vs_main': round(adjusted_rand_score(main_lab, lab), 3), 'CTHRC1_cluster_jaccard': j, 'split_into': k})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(stability_D / 'stability.tsv', sep='\t', index=False)
    print(out.to_string(index=False))

# fig1_10_removed_summary.py: retained numerical analysis.
def _init_removal():
    global Path, ad, pd, removal_CFG, removal_ROOT, removal_ROUNDS, yaml
    from pathlib import Path
    import anndata as ad
    import pandas as pd
    import yaml
    removal_CFG = _load_config()
    removal_ROOT = Path(removal_CFG['paths']['out_dir'])
    removal_ROUNDS = json.loads(_read_path(_RUN_DIR / 'fig1' / 'clean_rounds.json').read_text())

def removal_main():
    rows = []
    qc = pd.read_csv(_read_path(removal_ROOT / 'qc' / 'removed_cells.tsv.gz'), sep='\t', index_col=0)
    qc['step'] = '1_cell_qc'
    qc['reason'] = qc.reason.str.rstrip(';')
    rows.append(qc[['dataset', 'patient_key', 'condition', 'step', 'reason']])
    for prev, nxt in zip(removal_ROUNDS[:-1], removal_ROUNDS[1:]):
        a = ad.read_h5ad(_read_path(removal_ROOT / prev / 'fibro_integrated.h5ad'), backed='r')
        b = ad.read_h5ad(_read_path(removal_ROOT / nxt / 'fibro_integrated.h5ad'), backed='r')
        flags = pd.read_csv(_read_path(removal_ROOT / prev / 'cluster_flags.tsv'), sep='\t', index_col=0)
        flags.index = flags.index.astype(str)
        gone = a.obs.loc[~a.obs_names.isin(b.obs_names)].copy()
        gone['cluster'] = gone['leiden_0.5'].astype(str)
        gone['step'] = f'3_cluster_exclusion_{prev}'
        gone['reason'] = gone.cluster.map(flags.exclude_reason).fillna('') + ' (cluster ' + gone.cluster + ')'
        rows.append(gone[['dataset', 'patient_key', 'condition', 'step', 'reason']])
    allr = pd.concat(rows)
    allr.to_csv(removal_ROOT / 'qc' / 'removed_cells_all.tsv.gz', sep='\t')
    summ = allr.assign(reason_short=allr.reason.str.replace(' \\(cluster \\d+\\)', '', regex=True)).groupby(['step', 'reason_short', 'dataset']).size().unstack(fill_value=0)
    summ['total'] = summ.sum(1)
    summ.to_csv(removal_ROOT / 'qc' / 'removal_by_step.tsv', sep='\t')
    print(summ.to_string())
    final = ad.read_h5ad(_read_path(removal_ROOT / removal_ROUNDS[-1] / 'fibro_integrated.h5ad'), backed='r')
    print(f'\nfinal CAF atlas: {final.n_obs} cells, {final.obs.patient_key.nunique()} patients; by dataset {final.obs.dataset.value_counts().to_dict()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "figure1.yaml"))
    parser.add_argument("--run-dir", required=True, help="Dedicated new output workspace; stages may share it.")
    parser.add_argument("--input-root", help="Optional read-only results root containing fig1/.")
    parser.add_argument("--stage", required=True, choices=["build-global", "build-full", "legacy-inputs", "integrate-global", "annotate", "qc", "clean-cells", "clean-loop", "stability", "removal-summary"])
    parser.add_argument("stage_args", nargs=argparse.REMAINDER, help="Original stage options after --.")
    args = parser.parse_args()
    extra = args.stage_args[1:] if args.stage_args[:1] == ["--"] else args.stage_args
    if args.stage in {"build-full", "integrate-global", "integrate-caf"} and extra:
        parser.error("Use the fixed stage defaults; no extra options for this stage.")
    if args.stage == "build-global" and extra:
        parser.error("The discovery cap and output name are fixed by the configuration.")
    defaults = {"build-full": ["--cap", "0", "--name", "global_full_raw.h5ad"],
                "integrate-global": ["--input", "global_raw.h5ad", "--mode", "global", "--out", "integrate_global"],
                "clean-cells": ["cells"]}
    target_map = {
        "build-global": ["fig1/global_raw.h5ad"], "build-full": ["fig1/global_full_raw.h5ad"],
        "legacy-inputs": ["fig1/fibro_raw.h5ad", "fig1/lineage_pseudobulk.h5ad", "fig1/inputs_summary.tsv"],
        "integrate-global": ["fig1/integrate_global"],
        "annotate": ["fig1/annotate_global", "fig1/mesenchymal_raw.h5ad"],
        "qc": ["fig1/qc/cell_qc.tsv.gz", "fig1/qc/doublet_summary.tsv"],
        "clean-cells": ["fig1/mesenchymal_clean_raw.h5ad", "fig1/qc/removed_cells.tsv.gz"],
        "stability": ["fig1/integrate_caf_final/stability.tsv"],
        "removal-summary": ["fig1/qc/removed_cells_all.tsv.gz", "fig1/qc/removal_by_step.tsv"],
        "clean-loop": ["fig1/integrate_mes_clean", "fig1/integrate_caf_clean_r1", "fig1/integrate_caf_clean_r2", "fig1/integrate_caf_clean_r3", "fig1/integrate_caf_clean_r4", "fig1/integrate_caf_final"]}
    record, payload = _start(args, args.stage, target_map[args.stage])
    dispatch = {"build-global": ("inputs", inputs_main), "build-full": ("inputs", inputs_main),
                "legacy-inputs": ("legacy", legacy_main), "integrate-global": ("integration", integration_main),
                "annotate": ("annotation", annotation_main), "qc": ("qc", qc_main),
                "clean-cells": ("clean", clean_main), "stability": ("stability", stability_main),
                "removal-summary": ("removal", removal_main)}
    try:
        if args.stage == "clean-loop":
            if extra:
                raise ValueError("The clean-loop follows fixed historical settings and accepts no extra arguments.")
            _init_integration()
            _init_clean()
            previous = None
            exclude = []
            rounds = []
            for i in range(5):
                name = "integrate_mes_clean" if i == 0 else f"integrate_caf_clean_r{i}"
                command = ["--input", "mesenchymal_clean_raw.h5ad", "--mode", "caf", "--out", name]
                if previous is not None:
                    command += ["--exclude-from", str(_RUN_DIR / "fig1" / previous / "fibro_integrated.h5ad"), "--exclude", ",".join(exclude)]
                sys.argv = [sys.argv[0]] + command
                _init_integration()
                integration_main()
                clean_cmd_flag(_RUN_DIR / "fig1" / name / "fibro_integrated.h5ad", "leiden_0.5")
                flags = pd.read_csv(_RUN_DIR / "fig1" / name / "cluster_flags.tsv", sep="\t", index_col=0).fillna("")
                exclude = [str(k) for k in flags.index[flags.exclude_reason != ""]]
                previous = name
                rounds.append(name)
                if not exclude:
                    break
            if exclude:
                raise RuntimeError("Suspect clusters remain after the historical maximum of four exclusion rounds; no final atlas assigned.")
            (_RUN_DIR / "fig1" / "integrate_caf_final").symlink_to(previous, target_is_directory=True)
            (_RUN_DIR / "fig1" / "clean_rounds.json").write_text(json.dumps(rounds, indent=2))
            _finish(record, payload)
            return
        prefix, function = dispatch[args.stage]
        globals()[f"_init_{prefix}"]()
        sys.argv = [sys.argv[0]] + defaults.get(args.stage, []) + extra
        function()
    except BaseException as error:
        _finish(record, payload, error)
        raise
    _finish(record, payload)


if __name__ == "__main__":
    main()
