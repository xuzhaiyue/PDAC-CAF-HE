"""Program scores at cell and micro-region level.

Primary unit = micro-region: sum the counts of *malignant epithelial* cells whose nucleus
centroid falls in a square region (epithelium-restricted pseudobulk, reduces CAF spill-over
compared with summing all 2 µm bins). Cell level is auxiliary only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Highly abundant acinar transcripts beyond config `acinar` (top of A1/D2 UMI ranking). In these
# sections they make up 35–62 % of UMIs and diffuse into neighbouring cells, so they are dropped
# from every normalisation denominator.
ACINAR_EXTRA = ["CLPS", "CELA3B", "GP2", "REG1A", "REG1B", "SYCN", "CELA2A", "CELA2B", "PLA2G1B",
                "CTRB1", "CPA2", "PNLIPRP1", "PRSS3", "SPINK1"]


def program_umis(X, var_names, programs: dict[str, list[str]]) -> pd.DataFrame:
    """Raw UMI sums per program for each row of X (cells or regions)."""
    idx = {g: i for i, g in enumerate(var_names)}
    X = sp.csr_matrix(X)
    out = {}
    for name, genes in programs.items():
        cols = [idx[g] for g in genes if g in idx]
        out[name] = np.asarray(X[:, cols].sum(1)).ravel() if cols else np.zeros(X.shape[0])
    return pd.DataFrame(out)


def log_sum_cp10k(prog_umi: pd.DataFrame, total_umi: np.ndarray) -> pd.DataFrame:
    """log1p(1e4 * program UMIs / total UMIs). Sum-based: robust when single genes are ~0."""
    tot = np.maximum(np.asarray(total_umi, float), 1.0)
    return np.log1p(prog_umi.mul(1e4 / tot, axis=0))


def region_index(x_px: np.ndarray, y_px: np.ndarray, mpp: float, size_um: float):
    """Assign points to a square grid anchored at the image origin. Returns (ix, iy)."""
    step = size_um / mpp
    return np.floor(x_px / step).astype(int), np.floor(y_px / step).astype(int)


def region_pseudobulk(X, region_key: np.ndarray):
    """Sum rows of X (cells x genes) by region key. Returns (unique_keys, regions x genes csr)."""
    keys, inv = np.unique(region_key, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(keys), len(inv)))
    return keys, (G @ sp.csr_matrix(X)).tocsr()


def matched_random_genesets(var_names, mean_expr: np.ndarray, genes: list[str],
                            n_sets: int, rng: np.random.Generator, n_bins: int = 20) -> list[list[str]]:
    """Random gene sets matched to `genes` on expression-level bins (control for 'any gene set')."""
    var_names = np.asarray(var_names)
    ok = mean_expr > 0
    bins = pd.qcut(pd.Series(mean_expr[ok]).rank(method="first"), n_bins, labels=False).to_numpy()
    pool_names, pool_bins = var_names[ok], bins
    name2bin = dict(zip(pool_names, pool_bins))
    target_bins = [name2bin[g] for g in genes if g in name2bin]
    gene_set = set(genes)
    by_bin = {b: [g for g in pool_names[pool_bins == b] if g not in gene_set] for b in set(target_bins)}
    return [[rng.choice(by_bin[b]) for b in target_bins] for _ in range(n_sets)]
