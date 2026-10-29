"""冻结修订 A16：62 张 GEO 公开切片按 65 µm 口径（µm/px = 65 / spot_diameter_fullres / ratio）重新提取真实 112 / 128 µm 视野的
UNI2-h 特征，S0 / S1 只读推理，与 A12 旧尺度（实际约 132 / 151 µm）配对比较。规则见 config/fig1.yaml 的 A16_geo_true_scale。

流程与 A12（he_22）相同，只改 GEO 的 µm/px：读图、中心、条码、裁切、resize、编码、区域汇总全部复用。
HEST 4 张与自有 HD 4 张沿用 A12 / A10 原值。不训练、不下载；nice 19、2 线程、批量 32、负载 > 8 暂停。
分阶段：preflight（邻距 + 旧尺度重建）→ extract（可续跑）→ evaluate。在 he/ 内运行。
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(name, "2")
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from he_22_a10_scale_robustness import atomic_json, digest, pair_test  # noqa: E402

# 用户 2026-10-28 再次要求降低电脑压力：负载阈值与批间暂停可调得更保守（只影响调度，不影响特征数值）
LOAD_GATE = float(os.environ.get("A16_LOAD_GATE", 8)); PAUSE = float(os.environ.get("A16_PAUSE", .05))
SECTION_PAUSE = float(os.environ.get("A16_SECTION_PAUSE", 0))


def gate():
    while os.getloadavg()[0] > LOAD_GATE:
        print(f"{dt.datetime.now().isoformat()} load {os.getloadavg()[0]:.2f}>{LOAD_GATE}; paused 30s", flush=True)
        time.sleep(30)

Image.MAX_IMAGE_PIXELS = None
SPOT_UM, OLD_SPOT_UM = 65.0, 55.0
FIELDS = (112, 128)


def log(msg):
    print(f"{dt.datetime.now().isoformat()} {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage", choices=["all", "preflight", "extract", "evaluate"], default="all")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfgpath = root / "config/fig1.yaml"
    rule = yaml.safe_load(cfgpath.read_text())["he_amendments"]["A16_geo_true_scale"]
    assert rule["status"].startswith("frozen_")
    out = args.output.resolve(); assert out.is_relative_to(root / rule["paths"]["output"])
    for sub in ["features", "regions", "preflight", "source_snapshot"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    if (out / "run_config.yaml").exists():
        assert yaml.safe_load((out / "run_config.yaml").read_text())["rule"] == rule, "Frozen rule changed"
    else:
        (out / "run_config.yaml").write_text(yaml.safe_dump(dict(rule=rule, date=dt.datetime.now().isoformat(), command=sys.argv,
                                                                 python=sys.version, torch=str(torch.__version__)), sort_keys=False, allow_unicode=True))
        (out / "source_snapshot" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        (out / "source_snapshot/fig1.yaml").write_bytes(cfgpath.read_bytes())
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    torch.set_num_threads(2); torch.manual_seed(20261021)
    import he_14_pancancer_pretrain as P
    import he_16_geo_pdac_features as G
    M, N, dev = P.M, P.N, P.DEV
    G.ROOT = Path(rule["paths"]["geo_raw"])
    a12, a10, a9 = (root / rule["paths"][k] for k in ["a12", "a10", "a9"])
    old = pd.read_csv(a12 / "slide_rho.tsv", sep="\t")
    assert len(old) == 70 and old.patient.nunique() == 25
    geo = old[~old.source.isin(["own Visium HD", "HEST"])]
    assert len(geo) == 62
    protected = {str(f): digest(f) for f in list((a9 / "models").glob("pretrain_seed*.pt")) + [a9 / "models/feature_stats.pt"]
                 + list((a10 / "models").glob("bulk_seed?.pt")) + [a10 / "models/bulk_feature_stats.pt"]}

    def image_data(sid, source):
        """与 A12 相同的读图与中心；返回新旧两种 µm/px。"""
        gate()
        cache = a9 / "geo" / f"{sid}.npz"
        with np.load(cache, allow_pickle=True) as z:
            coords, barcode, px_old = z["coords"].copy(), z["barcode"].astype(str), float(z["pixel_size"])
        fs = G.files_of(G.ROOT / source / "samples" / sid)
        sf = json.loads(fs["scalefactors"]); hires = Image.open(io.BytesIO(fs["hires"])); img = G.read_image(fs["image"])
        ratio = img.shape[1] / (hires.size[0] / sf["tissue_hires_scalef"])
        mpp_old = OLD_SPOT_UM / sf["spot_diameter_fullres"] / ratio; mpp = SPOT_UM / sf["spot_diameter_fullres"] / ratio
        assert abs(mpp_old - px_old) < 1e-8
        pos = G.read_positions(fs["positions"]).loc[barcode]; assert (pos.in_tissue == 1).all()
        centers = np.c_[pos.pxl_col_in_fullres, pos.pxl_row_in_fullres] * ratio
        assert np.array_equal(np.round(centers).astype(np.int64), coords)
        return img, centers, mpp, mpp_old, cache

    def crop(img, center, mpp, field):
        h, w = img.shape[:2]; x, y = center; half = field / mpp / 2
        left, top, right, bottom = int(max(0, x - half)), int(max(0, y - half)), int(min(w, x + half)), int(min(h, y + half))
        return Image.fromarray(img[top:bottom, left:right]).resize((224, 224))

    @torch.inference_mode()
    def encode(model, prep, ims):
        gate(); r = model(torch.stack([prep(im) for im in ims]).to(dev)).float().cpu().numpy()
        assert np.isfinite(r).all(); time.sleep(PAUSE); return r.astype(np.float16)

    marker = out / "preflight/PASSED"
    if args.stage in ["all", "preflight", "extract"]:
        from he_05_extract_tcga_uni2h import load_model
        encoder, prep = load_model(str(dev)); torch.set_num_threads(2)
        if not marker.exists():
            # 1) 邻距：全部 62 张按新 µm/px，组织内斑点最近邻中位距须为 100 ± 2 µm
            spacing = []
            for _, r in geo.iterrows():
                with np.load(a9 / "geo" / f"{r['sample']}.npz", allow_pickle=True) as z:
                    xy = z["coords"].astype(float) * float(z["pixel_size"]) * SPOT_UM / OLD_SPOT_UM
                dmed = float(np.median(cKDTree(xy).query(xy, 2)[0][:, 1]))
                exc = r["sample"] in rule.get("spacing_exceptions", [])   # 用户确认的元数据邻距例外：照实记录，不改标尺
                spacing.append(dict(sample=r["sample"], source=r.source, nn_um_true=dmed, nn_um_old=dmed * OLD_SPOT_UM / SPOT_UM,
                                    within_100pm2=abs(dmed - 100) <= 2, documented_exception=exc, passed=abs(dmed - 100) <= 2 or exc))
            pd.DataFrame(spacing).to_csv(out / "preflight/spot_spacing.tsv", sep="\t", index=False)
            # 2) 旧尺度重建：每来源首张切片 8 个斑点，旧 µm/px 的 112 µm 图块与原缓存 cosine
            checks = []
            for source, sid in geo.groupby("source", sort=True).first()["sample"].items():
                img, centers, mpp, mpp_old, cache = image_data(sid, source)
                with np.load(cache, allow_pickle=True) as z:
                    idx = np.linspace(0, len(centers) - 1, 8, dtype=int); ref = z["features"][idx].astype(float)
                new = encode(encoder, prep, [crop(img, centers[i], mpp_old, 112.) for i in idx]).astype(float)
                cos = (new * ref).sum(1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(ref, axis=1))
                checks += [dict(sample=sid, source=source, spot=int(i), cosine=float(c)) for i, c in zip(idx, cos)]
                log(f"preflight {sid}: cosine median {np.median(cos):.4f} min {cos.min():.4f}")
                del img; gc.collect()
            pd.DataFrame(checks).to_csv(out / "preflight/legacy_cosines.tsv", sep="\t", index=False)
            sp = pd.DataFrame(spacing); cs = pd.DataFrame(checks).groupby("sample").cosine.agg(["median", "min"])
            log(f"spacing true µm: median {sp.nn_um_true.median():.2f}, range {sp.nn_um_true.min():.2f}–{sp.nn_um_true.max():.2f}")
            if not sp.passed.all() or not ((cs["median"] >= .98) & (cs["min"] >= .90)).all():
                raise RuntimeError("A16 preflight failed; see preflight/")
            marker.write_text("spacing 100±2 µm and legacy 112 µm reconstruction passed\n")
        if args.stage == "preflight":
            return
        qc = []
        for n, (_, r) in enumerate(geo.iterrows(), 1):
            sid = r["sample"]; meta = out / "features" / f"{sid}.json"
            if meta.exists():
                qc.append(json.loads(meta.read_text())); continue
            img, centers, mpp, mpp_old, cache = image_data(sid, r.source); ts = time.monotonic()
            arrs = {f: np.zeros((len(centers), 1536), np.float16) for f in FIELDS}
            for i in range(0, len(centers), 32):
                for f in FIELDS:
                    arrs[f][i:i + 32] = encode(encoder, prep, [crop(img, c, mpp, float(f)) for c in centers[i:i + 32]])
            for f in FIELDS:
                np.save(out / "features" / f"{sid}_true{f}.npy", arrs[f])
            info = dict(sample=sid, source=r.source, mpp_true=mpp, mpp_old=mpp_old, n_spots=len(centers), seconds=time.monotonic() - ts,
                        cache_sha256=digest(cache), **{f"sha256_true{f}": digest(out / "features" / f"{sid}_true{f}.npy") for f in FIELDS})
            atomic_json(meta, info); qc.append(info); pd.DataFrame(qc).to_csv(out / "extraction_qc.tsv", sep="\t", index=False)
            log(f"[{n}/62] {sid}: {len(centers)} spots, {info['seconds']:.0f}s")
            del img, arrs; gc.collect(); torch.mps.empty_cache() if dev.type == "mps" else None
            time.sleep(SECTION_PAUSE)
        (out / "EXTRACTION_COMPLETE").write_text("62 GEO slides at true 112 / 128 µm\n")
        del encoder; gc.collect()
        if args.stage == "extract":
            return
    assert marker.exists() and (out / "EXTRACTION_COMPLETE").exists()

    @torch.inference_mode()
    def local_ensemble(models, X, mu, sd, cols):
        vals = []
        for m in models:
            rows = []
            for sub in torch.from_numpy(np.asarray(X)).split(256):
                gate(); rows.append((torch.nn.functional.softplus(m.tile((sub.to(dev).float() - mu) / sd)) + 1e-4)[:, cols].cpu().numpy())
            vals.append(np.concatenate(rows))
        return np.mean(vals, axis=0)

    norms = [tuple(torch.load(p, map_location="cpu", weights_only=True)[k].to(dev) for k in ["mu", "sd"])
             for p in [a10 / "models/bulk_feature_stats.pt", a9 / "models/feature_stats.pt"]]
    models = [[], []]
    for k in range(3):
        m = N.I1().to(dev); m.load_state_dict(torch.load(a10 / "models" / f"bulk_seed{k}.pt", map_location="cpu", weights_only=True)); m.eval(); models[0].append(m)
        m = P.Head().to(dev); m.load_state_dict(torch.load(a9 / "models" / f"pretrain_seed{k}.pt", map_location="cpu", weights_only=True)); m.eval(); models[1].append(m)
    rows = []
    for _, r in old.iterrows():
        sid = r["sample"]
        base = dict(sample=sid, patient=r.patient, source=r.source, tissue=r.tissue, treated=r.treated, n_regions=r.n_regions,
                    S0_old128=r.S0_128, S1_old112=r.S1_112, S1_old128=r.S1_128, S2_historical=r.S2_historical)
        if sid not in set(geo["sample"]):   # HEST / 自有 HD：尺度本来正确，沿用原值
            base.update(S0_true112=r.S0_112, S0_true128=r.S0_128, S1_true112=r.S1_112, S1_true128=r.S1_128, rescaled=False)
            rows.append(base); continue
        P.FEAT = a9 / "geo"; s = P.load_sample(sid)
        reg = pd.read_csv(a12 / "regions" / f"{sid}.tsv.gz", sep="\t", float_precision="round_trip")
        assert np.array_equal(reg.region, np.arange(len(s["rel"]))) and np.array_equal(reg.caf_rich, s["caf"])
        assert int(s["caf"].sum()) == r.n_regions
        idx = s["idx"].numpy(); valid = idx >= 0; region = reg[["region", "truth", "caf_rich"]].copy()
        for f in FIELDS:
            X = np.load(out / "features" / f"{sid}_true{f}.npy", mmap_mode="r")
            for arm, (ms, (mu, sd), cols) in enumerate(zip(models, norms, [[M.HEAD_PROG, M.HEAD_PAN], [0, 1]])):
                L = local_ensemble(ms, X, mu, sd, cols)
                means = [np.where(valid, L[np.maximum(idx, 0), k], 0).sum(1) / valid.sum(1) for k in [0, 1]]
                pred = np.log(means[0]) - np.log(means[1]); key = f"S{arm}_true{f}"
                base[key] = float(stats.spearmanr(pred[s["caf"]], s["rel"][s["caf"]]).statistic); region[key] = pred
        base["rescaled"] = True; rows.append(base)
        region.to_csv(out / "regions" / f"{sid}.tsv.gz", sep="\t", index=False)
        log(f"evaluated {sid}: S0 old {r.S0_128:.3f} → true128 {base['S0_true128']:.3f}")
    slide = pd.DataFrame(rows); slide.to_csv(out / "slide_rho.tsv", sep="\t", index=False)
    pub = slide[slide.source != "own Visium HD"]
    tests = [pair_test(pub, "S0_true128", "S0_old128", "public66_21patients_primary"),
             pair_test(pub, "S1_true112", "S1_old112", "public66_21patients"),
             pair_test(slide, "S1_true112", "S0_true128", "all70_25patients"),
             pair_test(slide, "S1_true128", "S0_true128", "all70_25patients"),
             pair_test(pub, "S1_true128", "S0_true128", "public66_21patients"),
             pair_test(pub[~pub["sample"].isin(rule.get("spacing_exceptions", []))], "S0_true128", "S0_old128",
                       "public63_without_spacing_exceptions_sensitivity")]
    pd.DataFrame(tests).to_csv(out / "paired_tests.tsv", sep="\t", index=False)
    cols = ["S0_old128", "S0_true112", "S0_true128", "S1_old112", "S1_true112", "S1_true128"]
    slide.groupby("patient")[cols].mean().reset_index().to_csv(out / "patient_rho.tsv", sep="\t", index=False)
    slide.groupby("source")[cols].median().reset_index().to_csv(out / "source_medians.tsv", sep="\t", index=False)
    for f, h in protected.items():
        assert digest(f) == h, f"Protected weight changed: {f}"
    p = tests[0]; passed = p["median_paired_delta"] > 0 and p["p_wilcoxon"] < .05
    atomic_json(out / "verdict.json", dict(primary_passed=passed, primary=p, n_rescaled=62))
    (out / "COMPLETE").write_text("complete\n")
    log(f"A16 COMPLETE primary passed={passed}"); print(pd.DataFrame(tests).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
