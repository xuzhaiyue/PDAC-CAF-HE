"""Figure 3 supplement: reproduce the A9 leave-one-patient-out (S2) model for D1 and export 150 µm region predictions (hemorph environment, low load, run inside he/).

The archived A9 results only stored the summary ρ for D1 (0.749), not region predictions; T1 reproduced this only for public samples.
Method identical to T1 (he_20 public stage): read only the A9/models feature_stats and the 3 pretrained weights,
fine-tuning set = 66 public PDAC samples + the other 3 PDAC Visium HD sections (D1 excluded), seeds A.SEED+1200+k, FT_EPOCHS / BATCH / LR unchanged;
D1 evaluation as he_14.eval_own (all tile local outputs → 150 µm grid → Spearman in caf_rich regions).
Acceptance: D1 ρ within 0.03 of A9/lopo_spatial.tsv (T1 rule). he_17 main is not run; no fixed result is changed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

import he_20_spatial_maps as T1

P, M, A = T1.P, T1.M, T1.A
SID = "D1"
OUT = A.HE / "figures/spatial_maps/2026-10-28_D1_A9_LOPO"


def main():
    import os
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    torch.set_num_threads(2)
    for sub in ["models", "tables", "source_snapshot"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    (OUT / "source_snapshot" / Path(__file__).name).write_text(Path(__file__).read_text())
    protected = [A.HE / "A9/models/feature_stats.pt"] + sorted((A.HE / "A9/models").glob("pretrain_seed*.pt"))
    (OUT / "protected_before.json").write_text(json.dumps({str(p): T1.sha(p) for p in protected}, indent=2))

    # Fine-tuning set: the same 66 public samples as the he_20 public stage + the PDAC Visium HD sections other than D1
    pdac = []
    hest_root = Path(yaml.safe_load((T1.ROOT / "config/fig1.yaml").read_text())["he_amendments"]
                     ["A12_a10_patch_scale_robustness"]["paths"]["hest_root"])
    hest_order = pd.read_csv(hest_root / "_manifest/selection.tsv", sep="\t").id.tolist()
    for sid in [s for s in hest_order if s in P.PDAC_HEST]:
        s = T1.load_any(A.HE / "A8/hest", sid); s["patient"] = sid; pdac.append(s)
    for f in sorted((A.HE / "A9/geo").glob("*.npz")):
        s = T1.load_any(f.parent, f.stem)
        if s["gp"] >= 7 and s["gc"] == 4 and s["caf"].sum() >= 100:
            pdac.append(s)
    assert len(pdac) == 66
    hd = M.load_hd()
    own = []
    for o in [o for o in P.OWN if o != SID]:
        z = hd[o]; ok = z["ok"]
        own.append({"id": o, "X": z["X"][ok], "idx": torch.arange(int(ok.sum()))[:, None],
                    "valid": np.ones(int(ok.sum()), bool), "y": z["yloc"][ok]})
    ft = pdac + own
    pd.DataFrame([{"sample": v["id"], "patient": v.get("patient", v["id"])} for v in ft]).to_csv(
        OUT / "tables" / f"{SID}_training_members.tsv", sep="\t", index=False)

    st = torch.load(protected[0], map_location="cpu"); mu, sd = st["mu"].to(P.DEV), st["sd"].to(P.DEV)
    base = [torch.load(p, map_location="cpu") for p in protected[1:]]; assert len(base) == 3
    log, models = [], []
    for k, state in enumerate(base):
        f = OUT / "models" / f"{SID}_seed{k}.pt"; m = P.Head().to(P.DEV)
        if f.exists():
            m.load_state_dict(torch.load(f, map_location=P.DEV)); m.eval()
        else:
            m.load_state_dict(state)
            m = T1.train_same(ft, A.SEED + 1200 + k, mu, sd, m, log, SID, OUT / "training_heartbeat.json")
            torch.save(m.state_dict(), f)
        models.append(m)

    # D1 evaluation: all tiles → 150 µm grid (as he_14.eval_own / he_10.region_eval)
    z = hd[SID]
    L = T1.local(models, z["X"], mu, sd)
    tp = pd.DataFrame({"A9_LOPO__p": L[:, 0], "A9_LOPO__c": L[:, 1]})
    d = T1.hd_table(SID, z, tp, M.region_labels(SID, 150), "A9_LOPO")
    mask = d.caf_rich
    rho = stats.spearmanr(d.loc[mask, "rel"], d.loc[mask, "pred"]).statistic
    hist = pd.read_csv(A.HE / "A9/lopo_spatial.tsv", sep="\t").set_index("sample").loc[SID]
    ref, n_ref = float(hist.rho), int(hist.n_regions)
    passed = abs(rho - ref) <= 0.03 and int(mask.sum()) == n_ref
    T1.save_table(T1.ranks(d), OUT / "tables" / f"{SID}_A9_LOPO.tsv.gz")
    res = dict(sample=SID, model="A9_LOPO", rho=float(rho), reference_rho=ref, delta=float(rho - ref),
               n_regions=int(mask.sum()), reference_n=n_ref, passed=bool(passed),
               seeds=[A.SEED + 1200 + k for k in range(3)], epochs=P.FT_EPOCHS, batch=P.BATCH, lr=P.FT_LR,
               finished=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version, torch=str(torch.__version__))
    (OUT / "reproduction.json").write_text(json.dumps(res, indent=2))
    (OUT / "protected_after.json").write_text(json.dumps({str(p): T1.sha(p) for p in protected}, indent=2))
    print("D1 LOPO REPRODUCED", res, flush=True)
    if not passed:
        raise RuntimeError("D1 reproduction outside tolerance; see reproduction.json")


if __name__ == "__main__":
    main()
