"""Amendment A9: fine-tune the pan-cancer pretrained model on ~70 PDAC spatial sections (hemorph environment, MPS, low load). Rules in A9 of config/fig1.yaml (fixed).

1) Retrain the A8 pan-cancer model (156 samples, 3 seeds, same seeds as A8), this time saving weights
2) Leave one patient out: hold out each patient group, fine-tune on all other PDAC spatial sections (3 seeds), test on all sections of the held-out patient
3) Fine-tune on all sections → TCGA / CPTAC patient level (Spearman, AUC), per-patient predictions saved
4) Ablation: (a) the 4 PDAC Visium HD sections only; (b) primary tumours only (no liver / lung / peritoneal metastases); (c) all (= 3)
Outputs: results/he/A9/{models/, lopo_spatial.tsv, patient_preds_{cohort}_{tag}.tsv, patient_level.tsv, ablation_spatial.tsv}
Load: nice 19, 2 threads (as A8).
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

import he_14_pancancer_pretrain as P

M, A = P.M, P.A
GEO = A.HE / "A9" / "geo"
OUT = A.HE / ("A9_quick_discard" if P.QUICK else "A9")       # --quick: pipeline check only, written elsewhere and deleted
(OUT / "models").mkdir(parents=True, exist_ok=True)
METASTASIS = ("Liver metastasis", "Lung metastasis", "Peritoneal metastasis")


def load_any(path: Path, sid: str):
    P.FEAT = path
    s = P.load_sample(sid)
    z = np.load(path / f"{sid}.npz", allow_pickle=True)
    s["patient"] = str(z["patient"]) if "patient" in z.files and str(z["patient"]) not in ("", "nan", "None") else sid
    s["patient"] = re.sub(r"_Pri\d+$", "", s["patient"])   # additional GSE274557 primaries (Pt-6_Pri2 etc.) grouped with the same patient (fixed rule)
    s["tissue"] = str(z["tissue"]) if "tissue" in z.files else "Primary"
    s["treatment"] = str(z["treatment"]) if "treatment" in z.files else ""
    return s


def passes(s):
    return s["gp"] >= 7 and s["gc"] == 4 and s["caf"].sum() >= 100


def own_as_samples(hd, names):
    out = []
    for o in names:
        z = hd[o]; ok = z["ok"]
        out.append({"id": o, "X": z["X"][ok], "idx": torch.arange(int(ok.sum()))[:, None],
                    "valid": np.ones(int(ok.sum()), bool), "y": z["yloc"][ok]})
    return out


def finetune(base_states, ft_samples, mu, sd, seed0):
    ms = []
    for k, st in enumerate(base_states):
        m = P.Head().to(P.DEV); m.load_state_dict(st)
        ms.append(P.train(ft_samples, seed0 + k, mu, sd, model=m, epochs=P.FT_EPOCHS, lr=P.FT_LR))
    return ms


def eval_samples(ms, samples, hd, labels, mu, sd):
    rows = []
    for s in samples:
        if s.get("own"):
            o, r, n = [x for x in P.eval_own(ms, hd, labels, mu, sd) if x[0] == s["id"]][0]
        else:
            r, n = P.eval_hest(ms, s, mu, sd)
        rows.append({"sample": s["id"], "patient": s["patient"], "source": s["source"], "tissue": s["tissue"],
                     "treated": s["treatment"] not in ("", "No treatment", "nan"), "rho": r, "n_regions": n})
    return rows


def patient_level(ms, mu, sd, tag, bt, pats, bc, cp, ytr, yc):
    rows = []
    for cohort, bags, ids, lab in [("TCGA", bt, pats, ytr), ("CPTAC", bc, cp, yc)]:
        st = []
        for b in bags:
            L = np.mean([P.local_out(m, b, mu, sd) for m in ms], axis=0)
            st.append(np.log(L[:, 0].mean()) - np.log(L[:, 1].mean()))
        st = np.array(st)
        pd.DataFrame({"model": tag, "cohort": cohort, "patient": ids, "pred_state": st}).to_csv(
            OUT / f"patient_preds_{cohort}_{tag}.tsv", sep="\t", index=False)
        for t in ["program_resid", "program_sub"]:
            y = lab.loc[ids, t].to_numpy()
            rows.append({"model": tag, "cohort": cohort, "target": t, "spearman": stats.spearmanr(y, st)[0],
                         "auc": roc_auc_score(y > np.median(y), st)})
    return rows


def main():
    os.nice(19); torch.set_num_threads(2)
    t0 = time.time()
    sel = pd.read_csv("/Volumes/INTENSO_BIG/HEST1k/_manifest/selection.tsv", sep="\t"); oc = sel.set_index("id").organ
    hest = []
    for sid in sel.id:
        s = load_any(A.HE / "A8" / "hest", sid); s["organ"] = str(oc[sid])
        if passes(s):
            hest.append(s)
    pre = [s for s in hest if s["organ"] != "Pancreas"]
    pdac = []
    for s in [s for s in hest if s["id"] in P.PDAC_HEST]:
        s.update(source="HEST", tissue="Primary PDAC", patient=s["id"]); pdac.append(s)
    geo_qc = []
    for f in sorted(GEO.glob("*.npz")):
        s = load_any(GEO, f.stem); s["source"] = str(np.load(f, allow_pickle=True)["dataset"])
        geo_qc.append({"gsm": f.stem, "passed": passes(s), "caf_regions": int(s["caf"].sum())})
        if passes(s):
            pdac.append(s)
    pd.DataFrame(geo_qc).to_csv(OUT / "geo_sample_qc_A8rules.tsv", sep="\t", index=False)
    hd = M.load_hd()
    labels = {o: {150: M.region_labels(o, 150)} for o in P.OWN}
    own = [{"id": o, "own": True, "patient": o, "source": "own Visium HD", "tissue": "Primary PDAC", "treatment": ""} for o in P.OWN]
    print(f"pretrain {len(pre)}; PDAC spatial: {len(pdac)} public + {len(own)} own = {len(pdac) + len(own)} samples, "
          f"{len({s['patient'] for s in pdac}) + len(own)} patients ({time.time() - t0:.0f}s)", flush=True)

    # ---- 1) pretraining (same seeds as A8), weights saved
    mu, sd = P.feat_stats(pre)
    torch.save({"mu": mu.cpu(), "sd": sd.cpu()}, OUT / "models" / "feature_stats.pt")
    base = []
    for k, s_ in enumerate(P.SEEDS):
        f = OUT / "models" / f"pretrain_seed{s_}.pt"
        if f.exists():
            base.append(torch.load(f))
        else:
            m = P.train(pre, A.SEED + 500 + s_, mu, sd); base.append(m.state_dict()); torch.save(m.state_dict(), f)
    print(f"  pretraining done ({time.time() - t0:.0f}s)", flush=True)

    # ---- 2) leave-one-patient-out
    groups = sorted({s["patient"] for s in pdac} | set(P.OWN))
    lopo = []
    for g in groups:
        ft = [s for s in pdac if s["patient"] != g] + own_as_samples(hd, [o for o in P.OWN if o != g])
        ms = finetune(base, ft, mu, sd, A.SEED + 1200)
        held = [s for s in pdac if s["patient"] == g] + [s for s in own if s["patient"] == g]
        lopo += [dict(r, held_out_group=g) for r in eval_samples(ms, held, hd, labels, mu, sd)]
        pd.DataFrame(lopo).to_csv(OUT / "lopo_spatial.tsv", sep="\t", index=False)
        print(f"  LOPO {g}: {[round(r['rho'], 2) for r in lopo if r['held_out_group'] == g]} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 3) all samples → patient level; 4) ablations
    pats, Et, bt = M.load_tcga(); cp, yc, bc = M.load_cptac()
    ytr, _ = A.make_targets(Et, Et.iloc[:2]); ytr.index = pats
    sets = {"all70": [s for s in pdac] + own_as_samples(hd, P.OWN),
            "own4_only": own_as_samples(hd, P.OWN),
            "primary_only": [s for s in pdac if s["tissue"] not in METASTASIS] + own_as_samples(hd, P.OWN)}
    lv, abl = [], []
    for tag, ft in sets.items():
        fs = [OUT / "models" / f"finetune_{tag}_seed{k}.pt" for k in range(len(base))]
        if all(f.exists() for f in fs):     # re-run after the grouping correction: models that do not depend on grouping are reused
            ms = []
            for f in fs:
                m = P.Head().to(P.DEV); m.load_state_dict(torch.load(f)); m.eval(); ms.append(m)
        else:
            ms = finetune(base, ft, mu, sd, A.SEED + 1500)
            for f, m in zip(fs, ms):
                torch.save(m.state_dict(), f)
        lv += patient_level(ms, mu, sd, tag, bt, pats, bc, cp, ytr, yc)
        if tag == "own4_only":            # spatial on all public PDAC samples (none used in this model)
            abl += [dict(r, model=tag) for r in eval_samples(ms, pdac, hd, labels, mu, sd)]
        if tag == "primary_only":         # spatial on metastases (none used in this model)
            abl += [dict(r, model=tag) for r in eval_samples(ms, [s for s in pdac if s["tissue"] in METASTASIS], hd, labels, mu, sd)]
        print(f"  fine-tune {tag} done ({time.time() - t0:.0f}s)", flush=True)
    lv = pd.DataFrame(lv); lv.to_csv(OUT / "patient_level.tsv", sep="\t", index=False)
    pd.DataFrame(abl).to_csv(OUT / "ablation_spatial.tsv", sep="\t", index=False)

    lo = pd.DataFrame(lopo)
    pd.set_option("display.width", 200)
    print("\n== LOPO spatial rho by source / tissue\n" + lo.groupby(["source", "tissue"]).rho.agg(["median", "count"]).round(3).to_string())
    orig8 = lo[lo["sample"].isin(P.OWN + P.PDAC_HEST)]
    print(f"\noriginal 8 samples: median rho {orig8.rho.median():.3f}; all samples: {(lo.rho >= 0.3).mean():.0%} with rho >= 0.3 "
          f"(n={len(lo)}), median {lo.rho.median():.3f}")
    print("\n== patient level\n" + lv.round(3).to_string(index=False))
    print(f"total {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
