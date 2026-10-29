"""修订 A3：自己从 TCGA-PAAD 全切片提取 UNI2-h 特征（hemorph 环境，MPS）。规则见 config/fig1.yaml 的 he_amendments.A3。

切块与预处理照搬胃癌项目（project_GC_CAF_pathomics/03_analysis/2026-10-06_Figure2_UNI2h_run01/extract_uni2h_embeddings.py）：
缩略图组织掩膜 → 128 µm 网格 → 组织占比 ≥ 0.5 → 每张切片按 file_id 哈希种子随机取 ≤ 1000 块 → UNI2-h（timm 预处理）。
只处理下载日志中 md5 已核对的切片；可反复运行（已完成的跳过），下载进行中也可以边下边提。
输出：results/he/cache_own/tcga/{patient}.npy（float16，图块 × 1536）与 {patient}_coords.npy（x, y, 切片序号），
      results/he/cache_own/tcga/slides/{file_id}.npz（单张切片特征与坐标），extraction_status.tsv
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

PROJ = Path(__file__).resolve().parents[1]
SVS = Path("/Volumes/INTENSO/TCGA_PAAD_svs")
DL = PROJ / "results" / "he" / "wsi_download" / "status.tsv"
OUT = PROJ / "results" / "he" / "cache_own" / "tcga"
SLIDES = OUT / "slides"
FIELD_UM, MIN_TISSUE, MAX_TILES, THUMB = 128.0, 0.5, 1000, 2048
BATCH, MAX_LOAD, PAUSE = 16, 24.0, 30


def wait_load():
    while os.getloadavg()[0] > MAX_LOAD:
        print(f"system busy (load {os.getloadavg()[0]:.1f}); pausing {PAUSE}s", flush=True)
        time.sleep(PAUSE)


def load_model(dev):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
              mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
              act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    m = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw)
    assert sum(p.numel() for p in m.parameters()) == 681394176, "unexpected UNI2-h architecture"
    return m.eval().to(dev), create_transform(**resolve_data_config(m.pretrained_cfg, model=m))


def mpp(slide):
    x, y = slide.properties.get("openslide.mpp-x"), slide.properties.get("openslide.mpp-y")
    if x and y:
        return float(x), float(y)
    obj = slide.properties.get("openslide.objective-power")
    v = 10.0 / float(obj) if obj else 0.25
    return v, v


def select_tiles(slide, file_id):
    W, H = slide.dimensions
    th = slide.get_thumbnail((THUMB, THUMB)).convert("RGB")
    rgb, hsv = np.asarray(th, np.uint8), np.asarray(th.convert("HSV"), np.uint8)
    br = rgb.mean(2)
    tissue = ((hsv[:, :, 1] > 18) & (br > 40) & (br < 235)).astype(np.uint32)
    I = np.pad(tissue.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    th_h, th_w = tissue.shape
    mx, my = mpp(slide)
    cw, ch = max(1, round(FIELD_UM / mx)), max(1, round(FIELD_UM / my))
    xs, ys = np.arange(0, max(1, W - cw + 1), cw), np.arange(0, max(1, H - ch + 1), ch)
    gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
    x0 = np.clip(np.floor(gx * th_w / W).astype(int), 0, th_w); x1 = np.clip(np.ceil((gx + cw) * th_w / W).astype(int), 1, th_w)
    y0 = np.clip(np.floor(gy * th_h / H).astype(int), 0, th_h); y1 = np.clip(np.ceil((gy + ch) * th_h / H).astype(int), 1, th_h)
    frac = (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / np.maximum(1, (x1 - x0) * (y1 - y0))
    good = np.flatnonzero(frac >= MIN_TISSUE)
    if len(good) > MAX_TILES:
        seed = int(hashlib.sha256(str(file_id).encode()).hexdigest()[:8], 16)
        good = np.sort(np.random.default_rng(seed).choice(good, MAX_TILES, replace=False))
    return gx[good], gy[good], cw, ch, len(np.flatnonzero(frac >= MIN_TISSUE)), mx


def main():
    os.nice(10)
    torch.set_num_threads(4)
    SLIDES.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, prep = load_model(dev)
    status_f = OUT / "extraction_status.tsv"
    done = pd.read_csv(status_f, sep="\t").to_dict("records") if status_f.exists() else []
    have = {r["file_id"] for r in done}
    dl = pd.read_csv(DL, sep="\t")
    dl = dl[dl.status.isin(["downloaded_verified", "verified_existing"]) & ~dl.file_id.isin(have)
            & (dl.file_name.str[13:15] == "01")]          # 只用原发肿瘤切片（清单里还有 11 = 正常组织切片）
    print(f"{len(dl)} verified slides to extract on {dev}", flush=True)
    for r in dl.itertuples():
        t0 = time.time()
        s = openslide.OpenSlide(str(SVS / r.file_name))
        try:
            xs, ys, cw, ch, n_tissue, mx = select_tiles(s, r.file_id)
            feats = []
            for i in range(0, len(xs), BATCH):
                wait_load()
                imgs = [s.read_region((int(x), int(y)), 0, (cw, ch)).convert("RGB") for x, y in zip(xs[i:i + BATCH], ys[i:i + BATCH])]
                with torch.inference_mode():
                    feats.append(model(torch.stack([prep(im) for im in imgs]).to(dev)).float().cpu().numpy())
            F = np.concatenate(feats).astype(np.float32)
            assert np.isfinite(F).all()
            np.savez(SLIDES / f"{r.file_id}.npz", features=F, x=xs, y=ys, tile_w=cw, tile_h=ch, mpp=mx,
                     width=s.dimensions[0], height=s.dimensions[1])
            done.append({"file_id": r.file_id, "file_name": r.file_name, "patient": r.patient, "status": "ok",
                         "tiles": len(xs), "tissue_tiles_total": n_tissue, "mpp": mx, "seconds": round(time.time() - t0)})
        except Exception as e:  # noqa: BLE001
            done.append({"file_id": r.file_id, "file_name": r.file_name, "patient": r.patient, "status": f"error: {e}"})
        finally:
            s.close()
        pd.DataFrame(done).to_csv(status_f, sep="\t", index=False)
        print(f"{done[-1]['status']:6s} {r.file_name[:23]} tiles {done[-1].get('tiles', 0)} / {done[-1].get('tissue_tiles_total', 0)} "
              f"{done[-1].get('seconds', 0)}s", flush=True)
    # patient-level cache (all slides of a patient; same layout as results/he/cache)
    st = pd.DataFrame(done)
    st = st[(st.status == "ok") & (st.file_name.str[13:15] == "01")]
    for p, g in st.groupby("patient"):
        F, C = [], []
        for k, fid in enumerate(sorted(g.file_id)):
            z = np.load(SLIDES / f"{fid}.npz")
            F.append(z["features"].astype(np.float16)); C.append(np.c_[z["x"], z["y"], np.full(len(z["x"]), k)])
        np.save(OUT / f"{p}.npy", np.concatenate(F)); np.save(OUT / f"{p}_coords.npy", np.concatenate(C))
    print(f"patients cached: {st.patient.nunique()}", flush=True)


if __name__ == "__main__":
    main()
