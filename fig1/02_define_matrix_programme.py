"""Figure 1 fixed nine-gene programme and four-gene reference selection. Numerical analysis only; original results are never overwritten."""
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



# fig1_05_signature.py: retained numerical analysis.
def _init_signature():
    global Path, ad, combinations, multipletests, np, pd, roc_auc_score, signature_CFG, signature_DATASETS, signature_OUT, signature_PATIENT_LFC, signature_ROOT, signature_S, sp, stats, yaml
    from itertools import combinations
    from pathlib import Path
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import yaml
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    from statsmodels.stats.multitest import multipletests
    signature_CFG = _load_config()
    signature_S = {**signature_CFG['signature'], **signature_CFG.get('signature_v2', {}), **signature_CFG.get('signature_v3', {}), **signature_CFG.get('signature_v4', {})}
    signature_ROOT = Path(signature_CFG['paths']['out_dir'])
    signature_OUT = signature_ROOT / 'signature'
    signature_OUT.mkdir(parents=True, exist_ok=True)
    signature_DATASETS = list(signature_CFG['datasets'])
    signature_PATIENT_LFC = {}

def signature_group_sum(X: sp.csr_matrix, keys: np.ndarray) -> tuple[np.ndarray, sp.csr_matrix]:
    uniq, inv = np.unique(keys, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    return (uniq, (G @ X).tocsr())

def signature_log2cpm(M) -> np.ndarray:
    M = np.asarray(M.todense() if sp.issparse(M) else M, dtype=float)
    return np.log2(M / np.maximum(M.sum(1, keepdims=True), 1) * 1000000.0 + 1)

def signature_pooled_test(datasets: list[str]) -> pd.DataFrame:
    """One-sample t-test of per-patient log2FC pooled across datasets (paired design), BH-FDR."""
    M = pd.concat([signature_PATIENT_LFC[d] for d in datasets if d in signature_PATIENT_LFC])
    t, p = stats.ttest_1samp(M.to_numpy(), 0.0, axis=0)
    p = np.nan_to_num(p, nan=1.0)
    return pd.DataFrame({'pooled_log2fc': M.mean(0).to_numpy(), 'pooled_p': p, 'pooled_fdr': multipletests(p, method='fdr_bh')[1], 'pooled_n': len(M)}, index=M.columns)

def signature_paired_de(caf: ad.AnnData, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        sub = caf[(caf.obs.dataset == ds).to_numpy()]
        grp = np.where(sub.obs.is_target, 'T', 'R')
        keys = sub.obs.patient_key.astype(str).to_numpy() + '|' + grp
        uniq, P = signature_group_sum(sub.layers['counts'], keys)
        n = pd.Series(keys).value_counts().reindex(uniq).to_numpy()
        pats = pd.Series([u.split('|')[0] for u in uniq])
        ok = pd.DataFrame({'p': pats, 'g': [u.split('|')[1] for u in uniq], 'n': n})
        good = ok[ok.n >= signature_S['min_cells_per_group']].groupby('p').g.nunique()
        good = good[good == 2].index
        if len(good) < 3:
            print(f'  {ds}: only {len(good)} paired patients — skipped')
            continue
        L = signature_log2cpm(P)
        iT = [np.where(uniq == f'{p}|T')[0][0] for p in good]
        iR = [np.where(uniq == f'{p}|R')[0][0] for p in good]
        lfc = L[iT] - L[iR]
        signature_PATIENT_LFC[ds] = pd.DataFrame(lfc, columns=caf.var_names, index=list(good))
        t, pval = stats.ttest_rel(L[iT], L[iR], axis=0)
        pval = np.nan_to_num(pval, nan=1.0)
        cpmT = (2 ** L[iT] - 1).mean(0)
        rows.append(pd.DataFrame({'dataset': ds, 'gene': caf.var_names, 'n_patients': len(good), 'log2fc': lfc.mean(0), 'frac_up': (lfc > 0).mean(0), 'p': pval, 'fdr': multipletests(pval, method='fdr_bh')[1], 'target_cpm': cpmT}))
        print(f'  {ds}: {len(good)} paired patients')
    return pd.concat(rows, ignore_index=True)

def signature_select_state_genes(de: pd.DataFrame, datasets: list[str], spec_ok: set[str], pooled: pd.DataFrame) -> pd.DataFrame:
    """Effect size + patient consistency required in every dataset; significance from pooled patients."""
    d = de[de.dataset.isin(datasets)]
    pass_ = (d.log2fc > signature_S['min_log2fc']) & (d.frac_up >= signature_S['min_frac_patients_up']) & (d.target_cpm >= signature_S['min_target_cpm'])
    d = d.assign(pass_=pass_)
    agg = d.groupby('gene').agg(n_pass=('pass_', 'sum'), min_log2fc=('log2fc', 'min'), mean_log2fc=('log2fc', 'mean'), min_frac_up=('frac_up', 'min'), mean_target_cpm=('target_cpm', 'mean'))
    agg = agg.join(pooled[['pooled_fdr', 'pooled_n']])
    n_ds = d.dataset.nunique()
    sel = agg[(agg.n_pass == n_ds) & (agg.pooled_fdr < signature_S['max_fdr']) & agg.index.isin(spec_ok)]
    return sel.sort_values('min_log2fc', ascending=False).head(signature_S['max_genes'])

def signature_lineage_cpm(conditions: list[str] | None=None) -> pd.DataFrame:
    ann = pd.read_csv(_read_path(signature_ROOT / 'annotate_global' / 'global_full_annotation.tsv.gz'), sep='\t', index_col=0)
    if conditions:
        ann = ann[ann.condition.isin(conditions)]
    full = ad.read_h5ad(_read_path(signature_ROOT / 'global_full_raw.h5ad'))
    full = full[full.obs_names.isin(ann.index)]
    ann = ann.reindex(full.obs_names)
    keep = ~ann.lineage.isin(['Ambiguous', 'Low_quality', 'Hepatocyte_like']).to_numpy()
    keys = (ann.dataset.astype(str) + '|' + ann.lineage.astype(str) + '|' + ann.patient_key.astype(str)).to_numpy()
    uniq, P = signature_group_sum(sp.csr_matrix(full.X)[keep], keys[keep])
    n = pd.Series(keys[keep]).value_counts().reindex(uniq).to_numpy()
    ok = n >= 20
    L = 2 ** signature_log2cpm(P[np.where(ok)[0]]) - 1
    meta = pd.Series(uniq[ok]).str.split('|', expand=True)
    df = pd.DataFrame(L, columns=full.var_names)
    df['dataset'], df['lineage'] = (meta[0].values, meta[1].values)
    out = df.groupby(['dataset', 'lineage']).mean()
    return out

def signature_contribution(lcpm_pdac: pd.DataFrame, frac: pd.DataFrame) -> pd.DataFrame:
    """Share of each gene's transcripts in PDAC tissue that come from fibroblasts (abundance-weighted)."""
    rows = []
    for ds in signature_DATASETS:
        t, f = (lcpm_pdac.loc[ds], frac.loc[ds])
        f = f.reindex(t.index).fillna(0)
        tot = t.mul(f, axis=0).sum(0)
        fib = t.loc['Fibroblast'] * f.loc['Fibroblast']
        rows.append(pd.DataFrame({'dataset': ds, 'gene': t.columns, 'fib_contribution': (fib / tot.replace(0, np.nan)).values}))
    c = pd.concat(rows, ignore_index=True)
    c['pass'] = c.fib_contribution >= signature_S['min_fib_contribution']
    return c

def signature_specificity(lcpm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in signature_DATASETS:
        t = lcpm.loc[ds]
        if 'Fibroblast' not in t.index:
            continue
        fib = t.loc['Fibroblast']
        non = t.loc[t.index.intersection(signature_S['nonmesenchymal_lineages'])].max(0)
        per = t.loc['Pericyte_SMC'] if 'Pericyte_SMC' in t.index else 0 * fib
        rows.append(pd.DataFrame({'dataset': ds, 'gene': fib.index, 'fib_cpm': fib.values, 'ratio_vs_nonmes': ((fib + 1) / (non + 1)).values, 'ratio_vs_pericyte': ((fib + 1) / (per + 1)).values, 'top_nonmes_lineage': t.loc[t.index.intersection(signature_S['nonmesenchymal_lineages'])].idxmax(0).values}))
    sp_ = pd.concat(rows, ignore_index=True)
    sp_['pass'] = (sp_.ratio_vs_nonmes >= signature_S['specificity_vs_nonmesenchymal']) & (sp_.ratio_vs_pericyte >= signature_S['specificity_vs_pericyte'])
    return sp_

def signature_score(caf: ad.AnnData, genes: list[str]) -> np.ndarray:
    """Mean of log-normalised expression (X is log1p CP10k)."""
    idx = [caf.var_names.get_loc(g) for g in genes if g in caf.var_names]
    X = caf.X[:, idx]
    return np.asarray(X.mean(1)).ravel()

def signature_main():
    caf = ad.read_h5ad(_read_path(signature_ROOT / signature_S['caf_h5ad']))
    cl = caf.obs[signature_S['cluster_key']].astype(str)
    caf = caf[(~cl.isin(signature_S['exclude_from_reference'])).to_numpy() & caf.obs.condition.isin(signature_CFG['discovery_conditions']).to_numpy()].copy()
    caf.obs['is_target'] = caf.obs[signature_S['cluster_key']].astype(str).isin(signature_S['target_clusters']).to_numpy()
    caf.layers['counts'] = sp.csr_matrix(caf.layers['counts'])
    print(f'CAFs used: {caf.n_obs} (target {caf.obs.is_target.sum()}), patients {caf.obs.patient_key.nunique()}')
    print('1. state DE')
    de = signature_paired_de(caf, signature_DATASETS)
    de.to_csv(signature_OUT / 'de_per_dataset.tsv', sep='\t', index=False)
    print('2. lineage specificity')
    lcpm = signature_lineage_cpm()
    lcpm.to_csv(signature_OUT / 'lineage_cpm.tsv', sep='\t')
    spec = signature_specificity(lcpm)
    spec.to_csv(signature_OUT / 'specificity.tsv', sep='\t', index=False)
    ann = pd.read_csv(_read_path(signature_ROOT / 'annotate_global' / 'global_full_annotation.tsv.gz'), sep='\t', index_col=0)
    ann = ann[ann.condition.isin(signature_S['contribution_conditions']) & ~ann.lineage.isin(['Ambiguous', 'Low_quality', 'Hepatocyte_like'])]
    frac = pd.crosstab(ann.dataset, ann.lineage, normalize='index')
    lcpm_pdac = signature_lineage_cpm(signature_S['contribution_conditions'])
    contrib = signature_contribution(lcpm_pdac, frac)
    contrib.to_csv(signature_OUT / 'fibroblast_contribution.tsv', sep='\t', index=False)
    spec_ok = set(contrib.groupby('gene')['pass'].all().loc[lambda s: s].index)
    spec_ok -= set(signature_S.get('exclude_genes', []))
    print(f"   genes with >= {signature_S['min_fib_contribution']:.0%} fibroblast contribution in all datasets: {len(spec_ok)}")
    pooled = signature_pooled_test(signature_DATASETS)
    pooled.to_csv(signature_OUT / 'pooled_test.tsv', sep='\t')
    prog = signature_select_state_genes(de, signature_DATASETS, spec_ok, pooled)
    prog.to_csv(signature_OUT / 'cthrc1_program.tsv', sep='\t')
    print(f'CTHRC1+ program: {len(prog)} genes\n{prog.round(2).to_string()}')
    flat = de.groupby('gene').log2fc.apply(lambda v: v.abs().max())
    fib_cpm = spec.groupby('gene').fib_cpm.mean()
    pan = pd.DataFrame({'max_abs_log2fc': flat, 'mean_fib_cpm': fib_cpm}).dropna()
    pan = pan[pan.index.isin(spec_ok) & (pan.max_abs_log2fc < signature_S['pancaf_max_abs_log2fc']) & (pan.mean_fib_cpm >= signature_S.get('pancaf_min_fib_cpm', 0))]
    pan = pan.sort_values('mean_fib_cpm', ascending=False).head(signature_S['pancaf_n'])
    pan.to_csv(signature_OUT / 'pancaf_program.tsv', sep='\t')
    print(f'pan-CAF reference: {list(pan.index)}')
    lodo, sigs = ([], {})
    for held in signature_DATASETS:
        train = [d for d in signature_DATASETS if d != held]
        g = list(signature_select_state_genes(de, train, spec_ok, signature_pooled_test(train)).index)
        sigs[held] = set(g)
        m = (caf.obs.dataset == held).to_numpy()
        s = signature_score(caf[m], g) if g else np.zeros(m.sum())
        auc = roc_auc_score(caf.obs.is_target[m], s) if g else np.nan
        cth = signature_score(caf[m], ['CTHRC1'])
        lodo.append({'held_out': held, 'n_genes': len(g), 'auroc_program': auc, 'auroc_CTHRC1_alone': roc_auc_score(caf.obs.is_target[m], cth), 'overlap_with_full': len(set(g) & set(prog.index)) / max(len(prog), 1)})
    for a_, b_ in combinations(signature_DATASETS, 2):
        j = len(sigs[a_] & sigs[b_]) / max(len(sigs[a_] | sigs[b_]), 1)
        lodo.append({'held_out': f'jaccard {a_} vs {b_}', 'n_genes': np.nan, 'auroc_program': j})
    lodo = pd.DataFrame(lodo)
    lodo.to_csv(signature_OUT / 'lodo.tsv', sep='\t', index=False)
    print(lodo.round(3).to_string(index=False))
    (signature_OUT / 'programs.yaml').write_text(yaml.safe_dump({'CTHRC1_matrix_CAF': list(prog.index), 'pan_CAF': list(pan.index), 'rules': signature_S, 'frozen_before_outcome_data': True}, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "figure1.yaml"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-root", help="Optional read-only results root containing fig1/.")
    args = parser.parse_args()
    record, payload = _start(args, "select", ["fig1/signature"])
    try:
        _init_signature()
        signature_main()
    except BaseException as error:
        _finish(record, payload, error)
        raise
    _finish(record, payload)


if __name__ == "__main__":
    main()
