"""修订 A8：泛癌空间数据预训练 CAF 状态读取模型 → 胰腺癌零样本 / 微调测试 → 病人层面（hemorph 环境，MPS，低负载）。
规则见 config/fig1.yaml 的 A8（已冻结）。

区域：每个斑点 + 半径 1.1 × 最近邻距离内的斑点；区域预测 = a·log(mean 局部输出) + b（程序、泛 CAF 两个头），状态 = 两头之差。
标签：区域伪 bulk 的 log1p CP10k(程序 / 泛 CAF)，样本内 z 标准化；有效区域：总 UMI ≥ 500；评估只看富含 CAF 区域。
输出：results/he/A8/{qc.tsv, within_pretrain.tsv, pdac.tsv, learning_curve.tsv, resolution.tsv, patient_preds.tsv, patient_level.tsv}
用法：python he_14_pancancer_pretrain.py [--quick]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import he_10_spatial_mixed as M

A, N, E = M.A, M.N, M.E
DEV = M.DEV
OUT = A.HE / "A8"
FEAT = OUT / "hest"
QUICK = "--quick" in sys.argv
EPOCHS, SEEDS, BATCH = (1, [0], 128) if QUICK else (40, [0, 1, 2], 256)
FT_EPOCHS, FT_LR = (1, 1e-4) if QUICK else (10, 1e-4)
LR, WD = 3e-4, 1e-2
PDAC_HEST = ["NCBI569", "NCBI570", "NCBI571", "NCBI572"]
OWN = ["A1", "D1", "A2", "D2"]
EXPLORATORY = ["TENX179"]


# ---------------------------------------------------------------- data
def load_sample(sid):
    z = np.load(FEAT / f"{sid}.npz", allow_pickle=True)
    c = z["coords"].astype(float) * float(z["pixel_size"])
    tree = cKDTree(c)
    nn_d = np.median(tree.query(c, k=2)[0][:, 1])
    nb = tree.query_ball_point(c, r=1.1 * nn_d)
    K = max(len(x) for x in nb)
    idx = np.full((len(c), K), -1, np.int64)
    for i, x in enumerate(nb):
        idx[i, :len(x)] = x
    pr, pa, tot = (np.nan_to_num(z[k]) for k in ("prog_umi", "pan_umi", "total_umi"))
    m = idx >= 0
    agg = lambda v: np.where(m, v[np.maximum(idx, 0)], 0).sum(1)  # noqa: E731
    P, Pa, T = agg(pr), agg(pa), agg(tot)
    valid = T >= 500
    lp, lc = np.log1p(1e4 * P / np.maximum(T, 1)), np.log1p(1e4 * Pa / np.maximum(T, 1))
    zz = lambda v: (v - v[valid].mean()) / (v[valid].std() + 1e-9)  # noqa: E731
    yp, yc = zz(lp), zz(lc)
    caf = valid & (lc > np.median(lc[valid])) if valid.any() else valid
    return {"id": sid, "X": torch.from_numpy(z["features"]), "idx": torch.from_numpy(idx), "valid": valid, "caf": caf,
            "y": np.c_[yp, yc].astype(np.float32), "rel": (lp - lc), "organ": str(z["organ"]), "dataset": str(z["dataset"]),
            "tech": str(z["st_technology"]), "px": float(z["pixel_size"]), "gp": int(z["genes_prog_present"]),
            "gc": int(z["genes_pan_present"]), "n_spots": len(c), "nn_um": nn_d}


class Head(nn.Module):
    """I1 tile structure with two non-negative local outputs (program, pan-CAF)."""

    def __init__(self, h=256, p=0.25):
        super().__init__()
        self.tile = nn.Sequential(nn.Linear(1536, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 2))
        self.log_a = nn.Parameter(torch.zeros(2)); self.b = nn.Parameter(torch.zeros(2))

    def local(self, x):
        return nn.functional.softplus(self.tile(x)) + 1e-4

    def region(self, loc, idx):
        m = (idx >= 0).float()[..., None]
        mean = (loc[idx.clamp(min=0)] * m).sum(1) / m.sum(1)
        return torch.exp(self.log_a) * torch.log(mean) + self.b


def feat_stats(samples):
    X = torch.cat([s["X"][torch.randperm(len(s["X"]))[:500]] for s in samples]).float()
    return X.mean(0).to(DEV), (X.std(0) + 1e-6).to(DEV)


def train(samples, seed, mu, sd, model=None, epochs=EPOCHS, lr=LR):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = model or Head().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=WD)
    valid = [np.where(s["valid"])[0] for s in samples]
    w = np.array([len(v) for v in valid], float); w /= w.sum()
    steps = max(1, int(sum(len(v) for v in valid) // BATCH))
    for _ in range(epochs):
        m.train()
        for _st in range(steps):
            k = rng.choice(len(samples), p=w)                  # one sample per step (labels are z-scored per sample)
            s = samples[k]
            rows = rng.choice(valid[k], min(BATCH, len(valid[k])), replace=False)
            sub = s["idx"][rows]                                # regions x K spot indices (-1 = padding)
            u, inv = torch.unique(sub[sub >= 0], return_inverse=True)
            remap = torch.full_like(sub, -1); remap[sub >= 0] = inv
            loc = m.local((s["X"][u].to(DEV).float() - mu) / sd)  # only the tiles this step needs
            pred = m.region(loc, remap.to(DEV))
            loss = ((pred - torch.from_numpy(s["y"][rows]).to(DEV)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        time.sleep(0.5)                                   # low-load pause between epochs
    m.eval()
    return m


@torch.no_grad()
def local_out(m, X, mu, sd):
    return m.local((X.to(DEV).float() - mu) / sd).cpu().numpy()


@torch.no_grad()
def eval_hest(models, s, mu, sd):
    """Spearman of predicted vs measured state over CAF-rich regions (ensemble = mean of local outputs)."""
    L = np.mean([local_out(m, s["X"], mu, sd) for m in models], axis=0)
    idx = s["idx"].numpy(); msk = idx >= 0
    mean = lambda k: np.where(msk, L[np.maximum(idx, 0), k], 0).sum(1) / msk.sum(1)  # noqa: E731
    pred = np.log(mean(0)) - np.log(mean(1))
    c = s["caf"]
    return stats.spearmanr(pred[c], s["rel"][c])[0], int(c.sum())


def eval_own(models, hd, labels, mu, sd):
    rows = []
    for o in OWN:
        L = np.mean([local_out(m, hd[o]["X"], mu, sd) for m in models], axis=0)
        tp = pd.DataFrame({"m__p": L[:, 0], "m__c": L[:, 1]})
        r = [x for x in M.region_eval(tp, hd, o, labels[o]) if x["size_um"] == 150][0]
        rows.append((o, r["spearman"], r["n_regions"]))
    return rows


def pdac_eval(models, pd_samples, hd, labels, mu, sd, tag):
    rows = []
    for s in pd_samples:
        r, n = eval_hest(models, s, mu, sd)
        rows.append({"model": tag, "sample": s["id"], "source": "HEST", "rho": r, "n_regions": n})
    for o, r, n in eval_own(models, hd, labels, mu, sd):
        rows.append({"model": tag, "sample": o, "source": "own HD (150 um)", "rho": r, "n_regions": n})
    return rows


def main():
    os.nice(19); torch.set_num_threads(2)
    t0 = time.time()
    sel = pd.read_csv("/Volumes/INTENSO_BIG/HEST1k/_manifest/selection.tsv", sep="\t")
    samples, qc = [], []
    organ_csv = sel.set_index("id").organ                # HEST main table (per-sample json differs for 24 kidney samples)
    for sid in sel.id:
        s = load_sample(sid)
        s["organ"] = str(organ_csv[sid])
        ok = s["gp"] >= 7 and s["gc"] == 4 and s["caf"].sum() >= 100
        qc.append({"id": sid, "organ": s["organ"], "tech": s["tech"], "pixel_um": s["px"], "n_spots": s["n_spots"],
                   "nn_um": s["nn_um"], "genes_prog": s["gp"], "genes_pan": s["gc"], "caf_regions": int(s["caf"].sum()),
                   "passed_qc": ok, "dataset": s["dataset"]})
        if ok:
            samples.append(s)
    qc = pd.DataFrame(qc); qc.to_csv(OUT / "qc.tsv", sep="\t", index=False)
    pre = [s for s in samples if s["organ"] != "Pancreas"]
    pdh = [s for s in samples if s["id"] in PDAC_HEST]
    print(f"QC passed {qc.passed_qc.sum()}/{len(qc)}; pretrain {len(pre)} samples from {len({s['dataset'] for s in pre})} studies; "
          f"HEST PDAC {[s['id'] for s in pdh]} ({time.time() - t0:.0f}s)", flush=True)
    hd = M.load_hd()
    labels = {o: {150: M.region_labels(o, 150)} for o in OWN}
    mu, sd = feat_stats(pre)

    # ---- 1) within-pretraining: grouped by study
    groups = np.array([s["dataset"] for s in pre])
    wrows = []
    for f, (tr, te) in enumerate(GroupKFold(5).split(pre, groups=groups)):
        ms = [train([pre[i] for i in tr], A.SEED + 10 * f + sd_, mu, sd) for sd_ in SEEDS]
        for i in te:
            r, n = eval_hest(ms, pre[i], mu, sd)
            wrows.append({"fold": f, "sample": pre[i]["id"], "organ": pre[i]["organ"], "tech": pre[i]["tech"],
                          "pixel_um": pre[i]["px"], "rho": r, "n_regions": n})
        print(f"  within-pretrain fold {f} done ({time.time() - t0:.0f}s)", flush=True)
    w = pd.DataFrame(wrows); w.to_csv(OUT / "within_pretrain.tsv", sep="\t", index=False)
    print(w.groupby("organ").rho.agg(["median", "count"]).round(3).to_string(), flush=True)

    # ---- 2) PDAC zero-shot (pretrain on all non-pancreas), plus TCGA-trained comparators on HEST PDAC
    pre_models = [train(pre, A.SEED + 500 + s_, mu, sd) for s_ in SEEDS]
    prow = pdac_eval(pre_models, pdh, hd, labels, mu, sd, "pancancer_zero_shot")
    print(f"  zero-shot done ({time.time() - t0:.0f}s)", flush=True)
    # comparators trained on TCGA bulk (A5 configuration), evaluated on HEST PDAC with the same region rule
    pats, Et, bt = M.load_tcga()
    allidx = np.arange(len(pats)); ytr, _ = A.make_targets(Et, Et.iloc[:2])
    mu_t, sd_t = (t.to(DEV) for t in N.tile_stats(bt, allidx))
    i1 = [M.train_i1(bt, ytr[N.HEADS].to_numpy(), allidx, hd, [], s_, mu_t, sd_t) for s_ in SEEDS]
    for s in pdh:
        L = np.mean([M.i1_local(m, s["X"], mu_t, sd_t)[:, [M.HEAD_PROG, M.HEAD_PAN]] for m in i1], axis=0)
        idx = s["idx"].numpy(); msk = idx >= 0
        mean = lambda k: np.where(msk, L[np.maximum(idx, 0), k], 0).sum(1) / msk.sum(1)  # noqa: E731
        pred = np.log(mean(0)) - np.log(mean(1)); c = s["caf"]
        prow.append({"model": "I1_bulk (TCGA)", "sample": s["id"], "source": "HEST", "rho": stats.spearmanr(pred[c], s["rel"][c])[0],
                     "n_regions": int(c.sum())})
    a5 = pd.read_csv(A.HE / "A5" / "spatial.tsv", sep="\t")
    for _, r in a5[(a5.size_um == 150) & a5.model.isin(["I1_bulk", "I1_I2_mixed", "B2_attention"])].iterrows():
        prow.append({"model": f"{r.model} (A5)", "sample": r["sample"], "source": "own HD (150 um)", "rho": r.spearman, "n_regions": r.n_regions})

    # ---- 3) PDAC fine-tuning, leave-one-patient-out over the 8 PDAC samples
    for held in [s["id"] for s in pdh] + OWN:
        ft_hest = [s for s in pdh if s["id"] != held]
        ft_own = []
        for o in OWN:
            if o == held:
                continue
            z = hd[o]; ok = z["ok"]
            ft_own.append({"X": z["X"][ok], "idx": torch.arange(int(ok.sum()))[:, None], "valid": np.ones(int(ok.sum()), bool),
                           "y": z["yloc"][ok]})
        ms = []
        for k, s_ in enumerate(SEEDS):
            base = Head().to(DEV); base.load_state_dict(pre_models[k].state_dict())
            ms.append(train(ft_hest + ft_own, A.SEED + 900 + s_, mu, sd, model=base, epochs=FT_EPOCHS, lr=FT_LR))
        if held in PDAC_HEST:
            r, n = eval_hest(ms, next(s for s in pdh if s["id"] == held), mu, sd)
            prow.append({"model": "pancancer_finetuned", "sample": held, "source": "HEST", "rho": r, "n_regions": n})
        else:
            o, r, n = [x for x in eval_own(ms, hd, labels, mu, sd) if x[0] == held][0]
            prow.append({"model": "pancancer_finetuned", "sample": held, "source": "own HD (150 um)", "rho": r, "n_regions": n})
        print(f"  fine-tune held-out {held} done ({time.time() - t0:.0f}s)", flush=True)
    pdt = pd.DataFrame(prow); pdt.to_csv(OUT / "pdac.tsv", sep="\t", index=False)
    print(pdt.pivot_table(index="sample", columns="model", values="rho").round(3).to_string(), flush=True)

    # ---- 4) learning curve (zero-shot PDAC median rho vs number of pretraining studies)
    studies = sorted({s["dataset"] for s in pre})
    lrows = []
    for n_st in ([5] if QUICK else [5, 10, 20, len(studies)]):
        for rep in ([0] if QUICK or n_st == len(studies) else [0, 1, 2]):
            pick = set(np.random.default_rng(A.SEED + rep + 100 * n_st).choice(studies, n_st, replace=False))
            sub = [s for s in pre if s["dataset"] in pick]
            m = [train(sub, A.SEED + 700 + rep, mu, sd)]
            rr = [x["rho"] for x in pdac_eval(m, pdh, hd, labels, mu, sd, "lc")]
            lrows.append({"n_studies": n_st, "rep": rep, "n_samples": len(sub), "median_rho_pdac": float(np.nanmedian(rr))})
        print(f"  learning curve n={n_st} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(lrows).to_csv(OUT / "learning_curve.tsv", sep="\t", index=False)

    # ---- 5) resolution sensitivity: pretrain without samples coarser than 0.55 um/px
    hi = [s for s in pre if s["px"] <= 0.55]
    m_hi = [train(hi, A.SEED + 800 + s_, mu, sd) for s_ in SEEDS]
    res = pd.DataFrame(pdac_eval(m_hi, pdh, hd, labels, mu, sd, "pancancer_zero_shot_highres_only"))
    res.to_csv(OUT / "resolution.tsv", sep="\t", index=False)

    # ---- 6) patient level: TCGA (own features) and CPTAC
    cp, yc, bc = M.load_cptac()
    rows = []
    for tag, models in [("pancancer_zero_shot", pre_models)]:
        for cohort, bags, ids in [("TCGA", bt, pats), ("CPTAC", bc, cp)]:
            st_ = []
            for b in bags:
                L = np.mean([local_out(m, b, mu, sd) for m in models], axis=0)
                st_.append(np.log(L[:, 0].mean()) - np.log(L[:, 1].mean()))
            rows.append(pd.DataFrame({"model": tag, "cohort": cohort, "patient": ids, "pred_state": st_,
                                      "pred_abundance": [np.log(np.mean([local_out(m, b, mu, sd) for m in models], 0)[:, 1].mean()) for b in bags]}))
    pp = pd.concat(rows); pp.to_csv(OUT / "patient_preds.tsv", sep="\t", index=False)
    ytr_t, _ = A.make_targets(Et, Et.iloc[:2]); ytr_t.index = pats
    lv = []
    for (tag, cohort), d in pp.groupby(["model", "cohort"]):
        lab = ytr_t if cohort == "TCGA" else yc
        d = d.set_index("patient")
        for t in ["program_resid", "program_sub"]:
            y = lab.loc[d.index, t].to_numpy(); p = d.pred_state.to_numpy()
            lv.append({"model": tag, "cohort": cohort, "target": t, "spearman": stats.spearmanr(y, p)[0],
                       "auc": roc_auc_score(y > np.median(y), p)})
    lv = pd.DataFrame(lv); lv.to_csv(OUT / "patient_level.tsv", sep="\t", index=False)
    print(lv.round(3).to_string(index=False))
    print(f"total {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
