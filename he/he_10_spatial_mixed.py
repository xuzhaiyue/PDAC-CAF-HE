"""Amendment A5: mixed supervision (I1+I2), stroma_only and spatial validation (hemorph environment, MPS). Rules in A5 of config/fig1.yaml (fixed).

1) Spatial validation: I1_bulk / B2 / ridge_tile trained on TCGA only (3 seeds), I1_I2 leave-one-patient-out (TCGA + the other 3 HD sections);
   region-level (150 µm primary, 100 µm sensitivity) Spearman ρ between predicted and measured relative state in the 4 Visium HD sections.
2) Bulk evaluation: I1_I2 (TCGA + all 4 HD sections) and stroma_only, Spearman and AUC in TCGA 5-fold and CPTAC external.
Usage: python he_10_spatial_mixed.py [--quick]
Outputs: results/he/A5/{spatial.tsv, spatial_tiles_<sample>.tsv.gz, bulk.tsv}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import he_03_ridge_gc_protocol as A
import he_06_nested_nn as N

sys.path.insert(0, str(A.PROJ / "src"))
from hemorph import config as C  # noqa: E402
from hemorph import expression as E  # noqa: E402
from hemorph import io_visiumhd as IO  # noqa: E402

A.CACHE_T = A.HE / "cache_own"
DEV = N.DEV
OUT = A.HE / "A5"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
EPOCHS = 2 if QUICK else 80
SEEDS = [0] if QUICK else [0, 1, 2]
LR, WD = 3e-4, 1e-2                          # I1 config chosen on all TCGA in A2
B2_LR, B2_WD, B2_EP = 1e-4, 1e-2, (2 if QUICK else 20)
HD_SAMPLES = ["A1", "D1", "A2", "D2"]
HEAD_PROG, HEAD_PAN = N.HEADS.index("program"), N.HEADS.index("pan_CAF")
FIELD_UM = 128.0


# ---------------------------------------------------------------- data
def load_tcga():
    T = pd.read_csv(A.HE / "tcga_targets.tsv", sep="\t", index_col=0)
    pats = [p for p in T.index if (A.CACHE_T / "tcga" / f"{p}.npy").exists()]
    Et = pd.read_csv(A.PROJ / "results" / "fig2" / "tcga_expr.tsv.gz", sep="\t", index_col=0)[pats].T
    bags = [torch.from_numpy(np.load(A.CACHE_T / "tcga" / f"{p}.npy")) for p in pats]
    return pats, Et, bags


def load_cptac():
    C_ = pd.read_csv(A.HE / "cptac_targets.tsv", sep="\t", index_col=0)
    cp = [p for p in C_.index if (A.CACHE / "cptac" / f"{p}.npy").exists()]
    Ec = pd.read_csv(A.PROJ / "results" / "fig2" / "cptac_expr.tsv.gz", sep="\t", index_col=0)[cp].T
    yc, _ = A.make_targets(Ec, Ec.iloc[:2]); yc.index = cp
    return cp, yc, [torch.from_numpy(np.load(A.CACHE / "cptac" / f"{p}.npy")) for p in cp]


def load_hd():
    hd = {}
    for s in HD_SAMPLES:
        z = dict(np.load(A.HE / "hd" / f"{s}.npz"))
        ok = (z["n_cells"] >= 5) & (z["total_umi"] >= 500) & (z["pan_umi"] >= 5)
        lp = np.log1p(1e4 * z["prog_umi"] / np.maximum(z["total_umi"], 1))
        lc = np.log1p(1e4 * z["pan_umi"] / np.maximum(z["total_umi"], 1))
        zz = lambda v: (v - v[ok].mean()) / (v[ok].std() + 1e-9)  # noqa: E731
        z.update(X=torch.from_numpy(z["features"]), ok=ok, yloc=np.c_[zz(lp), zz(lc)].astype(np.float32))
        hd[s] = z
    return hd


def region_labels(s, size_um):
    """Region pseudobulk from nucleus-matched cells: relative state and the CAF-rich filter (same rules as hd_measurability)."""
    cfg = C.load_config()
    smp = next(x for x in C.samples(cfg) if x.sample_id == s)
    a = IO.read_cell_counts(smp); nuc = IO.read_polygons(smp, "nucleus")
    a = a[a.obs_names.isin(nuc.index)]
    X = sp.csr_matrix(a.X); var = list(a.var_names)
    progs = C.load_programs(cfg, a.var_names)
    excl = set(progs["acinar"]) | set(progs["endocrine"]) | set(E.ACINAR_EXTRA)
    nonac = np.array([g not in excl for g in var])
    acinar_genes = sorted(excl & set(var))
    xy = nuc.loc[a.obs_names, ["x", "y"]].to_numpy()
    ix, iy = E.region_index(xy[:, 0], xy[:, 1], smp.mpp, size_um)
    key = ix.astype(np.int64) * 1_000_000 + iy
    keys, P = E.region_pseudobulk(X, key)
    ncell = np.bincount(np.unique(key, return_inverse=True)[1])
    u = E.program_umis(P, var, {"prog": N.A.SETS["program"], "pan": N.A.SETS["pan_CAF"], "aci": acinar_genes})
    tot = np.asarray(P[:, nonac].sum(1)).ravel()
    aci = u["aci"].to_numpy() / np.maximum(np.asarray(P.sum(1)).ravel(), 1)
    s_ = E.log_sum_cp10k(u[["prog", "pan"]], tot)
    caf = (ncell >= 5) & (aci <= 0.2) & (u["pan"].to_numpy() >= 10 * (size_um / 100) ** 2)
    return pd.DataFrame({"key": keys, "rel": (s_.prog - s_.pan).to_numpy(), "caf_rich": caf}).set_index("key")


# ---------------------------------------------------------------- I1 with optional mixed supervision
def train_i1(bags, Y, tr, hd, hd_train, seed, mu, sd):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = N.I1().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    Yt = torch.tensor(Y, dtype=torch.float32, device=DEV)
    with torch.no_grad():
        m.b.copy_(Yt[tr].mean(0))
    if hd_train:
        HX = torch.cat([hd[s]["X"][hd[s]["ok"]] for s in hd_train]).float()
        HY = torch.from_numpy(np.concatenate([hd[s]["yloc"][hd[s]["ok"]] for s in hd_train]))
    for _ in range(EPOCHS):
        m.train()
        for i in rng.permutation(tr):
            b = bags[i]; sel = rng.choice(len(b), min(len(b), N.TILES_PER_STEP), replace=False)
            loss = ((m((b[sel].to(DEV).float() - mu) / sd) - Yt[i]) ** 2).mean()
            if hd_train:
                j = rng.choice(len(HX), 256, replace=False)
                loc = nn.functional.softplus(m.tile((HX[j].to(DEV) - mu) / sd)) + 1e-4
                pred = torch.exp(m.log_a) * torch.log(loc) + m.b
                loss = loss + ((pred[:, [HEAD_PROG, HEAD_PAN]] - HY[j].to(DEV)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


@torch.no_grad()
def i1_local(m, X, mu, sd):
    return (nn.functional.softplus(m.tile((X.to(DEV).float() - mu) / sd)) + 1e-4).cpu().numpy()


@torch.no_grad()
def b2_attention(m, X, mu, sd):
    z = m.fc((X.to(DEV).float() - mu) / sd)
    return torch.softmax(m.att(z).squeeze(-1), 0).cpu().numpy()


def region_eval(tile_pred: pd.DataFrame, hd, s, labels):
    """tile_pred: columns per model; for I1-type models two columns <name>__p / <name>__c (local amounts)."""
    z = hd[s]
    step_px = FIELD_UM / float(z["mpp"])
    rows = []
    for size, lab in labels.items():
        cx = (z["x_px"] + step_px / 2); cy = (z["y_px"] + step_px / 2)
        ix, iy = E.region_index(cx, cy, float(z["mpp"]), size)
        key = ix.astype(np.int64) * 1_000_000 + iy
        df = tile_pred.assign(key=key)
        names = sorted({c.split("__")[0] for c in tile_pred.columns})
        g = df.groupby("key")
        for nm in names:
            if f"{nm}__p" in df:
                r = np.log(g[f"{nm}__p"].mean()) - np.log(g[f"{nm}__c"].mean())
            else:
                r = g[nm].mean()
            d = pd.DataFrame({"pred": r}).join(lab, how="inner")
            d = d[d.caf_rich]
            rows.append({"sample": s, "size_um": size, "model": nm, "n_regions": len(d),
                         "spearman": stats.spearmanr(d.pred, d.rel)[0] if len(d) > 10 else np.nan})
    return rows


def auc(y, p):
    return roc_auc_score(y > np.median(y), p)


def main():
    t0 = time.time()
    pats, Et, bt = load_tcga()
    hd = load_hd()
    labels = {s: {sz: region_labels(s, sz) for sz in (150, 100)} for s in HD_SAMPLES}
    print(f"TCGA {len(pats)}; HD labelled tiles: " + ", ".join(f"{s} {hd[s]['ok'].sum()}" for s in HD_SAMPLES), flush=True)
    allidx = np.arange(len(pats))
    ytr, _ = A.make_targets(Et, Et.iloc[:2]); cols = list(ytr.columns)
    Y1 = ytr[N.HEADS].to_numpy()
    mu, sd = (t.to(DEV) for t in N.tile_stats(bt, allidx))

    # ---------------- spatial validation
    sp_rows = []
    tile_store = {s: {} for s in HD_SAMPLES}
    for seed in SEEDS:
        m = train_i1(bt, Y1, allidx, hd, [], seed, mu, sd)
        for s in HD_SAMPLES:
            L = i1_local(m, hd[s]["X"], mu, sd)
            tile_store[s].setdefault("I1_bulk__p", []).append(L[:, HEAD_PROG]); tile_store[s].setdefault("I1_bulk__c", []).append(L[:, HEAD_PAN])
        print(f"  seed {seed}: I1_bulk done ({time.time() - t0:.0f}s)", flush=True)
    # B2 (A2 config) trained on all TCGA; its attention weights are the spatial map (StRoMA-style)
    for seed in SEEDS:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        mb = N.B2(len(cols)).to(DEV)
        opt = torch.optim.Adam(mb.parameters(), lr=B2_LR, weight_decay=B2_WD)
        Yt = torch.tensor(ytr[cols].to_numpy(), dtype=torch.float32, device=DEV)
        for _ in range(B2_EP):
            mb.train()
            for i in rng.permutation(allidx):
                b = bt[i]; sel = rng.choice(len(b), min(len(b), N.TILES_PER_STEP), replace=False)
                loss = ((mb((b[sel].to(DEV).float() - mu) / sd) - Yt[i]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        mb.eval()
        for s in HD_SAMPLES:
            tile_store[s].setdefault("B2_attention", []).append(b2_attention(mb, hd[s]["X"], mu, sd))
    # ridge on single tiles (trained on patient-level mean features, program_sub target)
    Xt = A.patient_features("tcga", pats)
    g = A.ridge(A.SEED).fit(Xt, ytr["program_sub"].to_numpy())
    for s in HD_SAMPLES:
        tile_store[s]["ridge_tile"] = [g.predict(hd[s]["X"].float().numpy())]
    # I1 + I2, leave-one-patient-out
    for held in HD_SAMPLES:
        tr_hd = [s for s in HD_SAMPLES if s != held]
        for seed in SEEDS:
            m = train_i1(bt, Y1, allidx, hd, tr_hd, seed, mu, sd)
            L = i1_local(m, hd[held]["X"], mu, sd)
            tile_store[held].setdefault("I1_I2_mixed__p", []).append(L[:, HEAD_PROG])
            tile_store[held].setdefault("I1_I2_mixed__c", []).append(L[:, HEAD_PAN])
        print(f"  I1_I2 LOPO held-out {held} done ({time.time() - t0:.0f}s)", flush=True)
    for s in HD_SAMPLES:
        tp = pd.DataFrame({k: np.mean(v, axis=0) for k, v in tile_store[s].items()})
        tp.assign(x_px=hd[s]["x_px"], y_px=hd[s]["y_px"]).to_csv(OUT / f"spatial_tiles_{s}.tsv.gz", sep="\t", index=False)
        sp_rows += region_eval(tp, hd, s, labels[s])
    spt = pd.DataFrame(sp_rows); spt.to_csv(OUT / "spatial.tsv", sep="\t", index=False)
    print("\n== spatial (Spearman, CAF-rich regions)\n" +
          spt.pivot_table(index=["size_um", "sample"], columns="model", values="spearman").round(3).to_string(), flush=True)

    # ---------------- bulk: I1_I2 (all HD) and stroma_only
    cp, yc, bc = load_cptac()
    rows = []
    folds = list(KFold(5, shuffle=True, random_state=A.SEED).split(pats))
    oof = {"I1_I2_mixed": {}, "stroma_only": {}}
    for f, (tr, te) in enumerate(folds):
        ytr_f, yte_f = A.make_targets(Et.iloc[tr], Et.iloc[te])
        mu_f, sd_f = (t.to(DEV) for t in N.tile_stats(bt, tr))
        Yf = np.zeros((len(bt), len(N.HEADS))); Yf[tr] = ytr_f[N.HEADS].to_numpy()
        beta = N.resid_beta(ytr_f)
        m = train_i1(bt, Yf, tr, hd, HD_SAMPLES, A.SEED + f, mu_f, sd_f)
        P = N.state_preds("I1", N.predict(m, bt, te, mu_f, sd_f), list(ytr_f.columns), beta)
        for t in ["program_resid", "program_sub", "Dominguez_LRRC15_sub", "pan_CAF"]:
            oof["I1_I2_mixed"].setdefault(t, []).append(pd.DataFrame({"y": yte_f[t].to_numpy(), "p": P[t].to_numpy()}))
        # stroma_only: I1_bulk trained on the fold selects the top-30% pan-CAF tiles; ridge on their mean
        mb = train_i1(bt, Yf, tr, hd, [], A.SEED + f, mu_f, sd_f)
        feats = []
        for b in bt:
            c = i1_local(mb, b, mu_f, sd_f)[:, HEAD_PAN]
            k = np.argsort(c)[-max(1, int(0.3 * len(c))):]
            feats.append(b[k].float().mean(0).numpy())
        Xs = np.vstack(feats)
        for t in ["program_resid", "program_sub", "Dominguez_LRRC15_sub", "pan_CAF"]:
            p = A.ridge(A.SEED + f).fit(Xs[tr], ytr_f[t].to_numpy()).predict(Xs[te])
            oof["stroma_only"].setdefault(t, []).append(pd.DataFrame({"y": yte_f[t].to_numpy(), "p": p}))
        print(f"  bulk fold {f} done ({time.time() - t0:.0f}s)", flush=True)
    for mname, d in oof.items():
        for t, parts in d.items():
            x = pd.concat(parts)
            rows.append({"model": mname, "cohort": "TCGA 5-fold", "target": t, "spearman": stats.spearmanr(x.y, x.p)[0],
                         "auc": auc(x.y.to_numpy(), x.p.to_numpy())})
    # CPTAC external
    beta = N.resid_beta(ytr)
    preds = []
    for seed in SEEDS:
        m = train_i1(bt, Y1, allidx, hd, HD_SAMPLES, A.SEED + 1000 + seed, mu, sd)
        preds.append(N.state_preds("I1", N.predict(m, bc, range(len(bc)), mu, sd), cols, beta))
    pc = sum(preds) / len(preds); pc.index = cp
    mb = train_i1(bt, Y1, allidx, hd, [], A.SEED + 2000, mu, sd)
    def stroma_feats(bags):
        out = []
        for b in bags:
            c = i1_local(mb, b, mu, sd)[:, HEAD_PAN]
            k = np.argsort(c)[-max(1, int(0.3 * len(c))):]
            out.append(b[k].float().mean(0).numpy())
        return np.vstack(out)
    Xs_t, Xs_c = stroma_feats(bt), stroma_feats(bc)
    for t in ["program_resid", "program_sub", "Dominguez_LRRC15_sub", "pan_CAF"]:
        rows.append({"model": "I1_I2_mixed", "cohort": "CPTAC external", "target": t,
                     "spearman": stats.spearmanr(pc[t], yc[t])[0], "auc": auc(yc[t].to_numpy(), pc[t].to_numpy())})
        p = A.ridge(A.SEED).fit(Xs_t, ytr[t].to_numpy()).predict(Xs_c)
        rows.append({"model": "stroma_only", "cohort": "CPTAC external", "target": t,
                     "spearman": stats.spearmanr(p, yc[t])[0], "auc": auc(yc[t].to_numpy(), p)})
    bk = pd.DataFrame(rows); bk.to_csv(OUT / "bulk.tsv", sep="\t", index=False)
    print("\n== bulk\n" + bk.pivot_table(index=["cohort", "target"], columns="model", values=["spearman", "auc"]).round(3).to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
