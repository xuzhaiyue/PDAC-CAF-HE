"""Amendment A6: public 10x PDAC Visium HD section (3′, fresh frozen) as a fully independent spatial test (hemorph environment, MPS). Rules in A6 of config/fig1.yaml.

1) Tile features: tissue_image.btf, 128 µm grid, tissue mask, all tissue tiles, UNI2-h (as he_09)
2) Labels: 8 µm bins aggregated by centre into tiles / regions (no nucleus segmentation; non-acinar UMI >= 500 replaces cell count for region filtering)
3) Models: trained on TCGA only (plus the 4 PDAC Visium HD sections for I1_I2), 3 seeds, evaluated once on this section after fixing
Outputs: results/he/hd/10x_PDAC.npz, results/he/A6/{spatial_10x.tsv, spatial_tiles_10x.tsv.gz}
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import tifffile
import torch
from PIL import Image
from scipy import stats

import he_10_spatial_mixed as M
from he_05_extract_tcga_uni2h import load_model
from he_09_hd_tiles import tissue_fraction

A, N, E, C = M.A, M.N, M.E, M.C
ROOT = Path("/Volumes/INTENSO_BIG/10x_VisiumHD_3prime_Human_Pancreatic_Cancer")
BIN = ROOT / "binned_outputs" / "square_008um"
NPZ = A.HE / "hd" / "10x_PDAC.npz"
OUT = A.HE / "A6"
OUT.mkdir(parents=True, exist_ok=True)
FIELD_UM, BATCH = 128.0, 16


def load_bins():
    a = sc.read_10x_h5(BIN / "filtered_feature_bc_matrix.h5")
    a.var_names_make_unique()
    pos = pd.read_parquet(BIN / "spatial" / "tissue_positions.parquet").set_index("barcode")
    pos = pos.loc[a.obs_names]
    mpp = float(pd.read_json(BIN / "spatial" / "scalefactors_json.json", typ="series")["microns_per_pixel"])
    xy = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
    cfg = C.load_config()
    progs = C.load_programs(cfg, a.var_names)
    excl = set(progs["acinar"]) | set(progs["endocrine"]) | set(E.ACINAR_EXTRA)
    return sp.csr_matrix(a.X), list(a.var_names), xy, mpp, excl


def pseudobulk(X, var, xy, mpp, size, excl):
    ix, iy = E.region_index(xy[:, 0], xy[:, 1], mpp, size)
    key = ix.astype(np.int64) * 1_000_000 + iy
    keys, P = E.region_pseudobulk(X, key)
    nonac = np.array([g not in excl for g in var])
    u = E.program_umis(P, var, {"prog": A.SETS["program"], "pan": A.SETS["pan_CAF"], "aci": sorted(excl & set(var))})
    tot = np.asarray(P[:, nonac].sum(1)).ravel()
    aci = u["aci"].to_numpy() / np.maximum(np.asarray(P.sum(1)).ravel(), 1)
    return keys, u, tot, aci


def build_tiles():
    if NPZ.exists():
        return
    t0 = time.time()
    os.nice(10)
    img = tifffile.imread(ROOT / "Visium_HD_3prime_Human_Pancreatic_Cancer_tissue_image.btf")
    X, var, xy, mpp, excl = load_bins()
    step = FIELD_UM / mpp
    gx, gy, frac = tissue_fraction(img, step)
    keep = frac >= 0.5
    gx, gy = gx[keep], gy[keep]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, prep = load_model(dev)
    feats = []
    for i in range(0, len(gx), BATCH):
        ims = [Image.fromarray(img[int(round(y * step)):int(round(y * step)) + int(round(step)),
                                   int(round(x * step)):int(round(x * step)) + int(round(step))])
               for x, y in zip(gx[i:i + BATCH], gy[i:i + BATCH])]
        with torch.inference_mode():
            feats.append(model(torch.stack([prep(im) for im in ims]).to(dev)).float().cpu().numpy())
    keys, u, tot, _ = pseudobulk(X, var, xy, mpp, FIELD_UM, excl)
    nb = np.bincount(np.unique(E.region_index(xy[:, 0], xy[:, 1], mpp, FIELD_UM)[0].astype(np.int64) * 1_000_000
                               + E.region_index(xy[:, 0], xy[:, 1], mpp, FIELD_UM)[1], return_inverse=True)[1])
    lut = {k: i for i, k in enumerate(keys)}
    j = np.array([lut.get(k, -1) for k in gx.astype(np.int64) * 1_000_000 + gy])
    g = lambda v: np.where(j >= 0, np.asarray(v)[np.maximum(j, 0)], 0)  # noqa: E731
    np.savez(NPZ, features=np.concatenate(feats).astype(np.float16), ix=gx, iy=gy,
             x_px=(gx * step).astype(np.float32), y_px=(gy * step).astype(np.float32), mpp=mpp,
             prog_umi=g(u["prog"].to_numpy()), pan_umi=g(u["pan"].to_numpy()), total_umi=g(tot), n_cells=g(nb),
             tissue_frac=frac[keep])
    print(f"10x tiles: {len(gx)} tissue tiles, {(j >= 0).sum()} with bins, mpp {mpp:.3f}, {time.time() - t0:.0f}s", flush=True)


def region_labels_10x(size):
    X, var, xy, mpp, excl = load_bins()
    keys, u, tot, aci = pseudobulk(X, var, xy, mpp, size, excl)
    s_ = E.log_sum_cp10k(u[["prog", "pan"]], tot)
    caf = (tot >= 500) & (aci <= 0.2) & (u["pan"].to_numpy() >= 10 * (size / 100) ** 2)
    return pd.DataFrame({"key": keys, "rel": (s_.prog - s_.pan).to_numpy(), "caf_rich": caf}).set_index("key")


def main():
    t0 = time.time()
    build_tiles()
    z = dict(np.load(NPZ))
    z["X"] = torch.from_numpy(z["features"])
    hd = M.load_hd()
    hd["10x"] = z
    pats, Et, bt = M.load_tcga()
    allidx = np.arange(len(pats))
    ytr, _ = A.make_targets(Et, Et.iloc[:2]); cols = list(ytr.columns)
    Y1 = ytr[N.HEADS].to_numpy()
    mu, sd = (t.to(M.DEV) for t in N.tile_stats(bt, allidx))
    store = {}
    for seed in M.SEEDS:
        for name, hd_train in [("I1_bulk", []), ("I1_I2_mixed", M.HD_SAMPLES)]:
            m = M.train_i1(bt, Y1, allidx, hd, hd_train, seed, mu, sd)
            L = M.i1_local(m, z["X"], mu, sd)
            store.setdefault(f"{name}__p", []).append(L[:, M.HEAD_PROG]); store.setdefault(f"{name}__c", []).append(L[:, M.HEAD_PAN])
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        mb = N.B2(len(cols)).to(M.DEV)
        opt = torch.optim.Adam(mb.parameters(), lr=M.B2_LR, weight_decay=M.B2_WD)
        Yt = torch.tensor(ytr[cols].to_numpy(), dtype=torch.float32, device=M.DEV)
        for _ in range(M.B2_EP):
            mb.train()
            for i in rng.permutation(allidx):
                b = bt[i]; sel = rng.choice(len(b), min(len(b), N.TILES_PER_STEP), replace=False)
                loss = ((mb((b[sel].to(M.DEV).float() - mu) / sd) - Yt[i]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        mb.eval()
        store.setdefault("B2_attention", []).append(M.b2_attention(mb, z["X"], mu, sd))
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    Xt = A.patient_features("tcga", pats)
    store["ridge_tile"] = [A.ridge(A.SEED).fit(Xt, ytr["program_sub"].to_numpy()).predict(z["X"].float().numpy())]
    tp = pd.DataFrame({k: np.mean(v, axis=0) for k, v in store.items()})
    tp.assign(x_px=z["x_px"], y_px=z["y_px"]).to_csv(OUT / "spatial_tiles_10x.tsv.gz", sep="\t", index=False)
    labels = {sz: region_labels_10x(sz) for sz in (150, 100)}
    rows = M.region_eval(tp, hd, "10x", labels)
    # bootstrap CI over regions for each model / size
    rng = np.random.default_rng(A.SEED)
    step_px = FIELD_UM / float(z["mpp"])
    for r in rows:
        ix, iy = E.region_index(z["x_px"] + step_px / 2, z["y_px"] + step_px / 2, float(z["mpp"]), r["size_um"])
        df = tp.assign(key=ix.astype(np.int64) * 1_000_000 + iy).groupby("key")
        nm = r["model"]
        pred = (np.log(df[f"{nm}__p"].mean()) - np.log(df[f"{nm}__c"].mean())) if f"{nm}__p" in tp else df[nm].mean()
        d = pd.DataFrame({"pred": pred}).join(labels[r["size_um"]], how="inner")
        d = d[d.caf_rich]
        bs = [stats.spearmanr(*d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))[["pred", "rel"]].T.values)[0]
              for _ in range(1000)]
        r["ci_lo"], r["ci_hi"] = np.quantile(bs, [0.025, 0.975])
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "spatial_10x.tsv", sep="\t", index=False)
    print(res.round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
