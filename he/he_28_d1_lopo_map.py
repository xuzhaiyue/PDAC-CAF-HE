"""Figure 3 补图：为 D1 重现 A9 留一患者（S2）模型并导出 150 µm 区域预测（hemorph 环境，低负载，在 he/ 内运行）。

A9 历史结果只存了 D1 的汇总 ρ（0.749），没有区域预测；T1 只为公开样本做过同样的重现。
做法与 T1（he_20 public 阶段）完全相同：只读 A9/models 的 feature_stats 与 3 个预训练权重，
微调集合 = 66 个公开 PDAC 样本 + 其余 3 个自有 HD（不含 D1），种子 A.SEED+1200+k，FT_EPOCHS / BATCH / LR 不变；
D1 评估 = he_14.eval_own 口径（全部图块局部输出 → 150 µm 网格 → caf_rich 区域 Spearman）。
验收：与 A9/lopo_spatial.tsv 的 D1 ρ 相差 ≤ 0.03（T1 规则）。不运行 he_17 main，不改任何冻结结果。
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

    # 微调集合：与 he_20 public 阶段相同的 66 个公开样本 + 除 D1 外的自有 HD
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

    # D1 评估：全部图块 → 150 µm 网格（he_14.eval_own / he_10.region_eval 口径）
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
