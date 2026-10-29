"""自有 Visium HD 的图块特征与图块层面实测标签（hemorph 环境，MPS）。

图像：spaceranger 嵌在 cloupe 里的全分辨率 H&E（只读）；mpp 来自 scalefactors_json.json。
切块：与 TCGA 自提特征相同（A3）：128 µm 网格（从图像原点起算，与 region_index 一致）、缩略图组织掩膜、组织占比 ≥ 0.5，
      但保留全部组织图块（空间验证需要致密覆盖）；UNI2-h 用 timm 自带预处理。
注意：A1/D1 约 0.36 µm/px，A2/D2 约 0.68 µm/px，比 TCGA（约 0.25）粗；128 µm 视野一律缩放到 224 px，分辨率差异作为局限报告。
标签：每个图块内的细胞（只用有细胞核多边形的细胞，同 hd_program_measurability）伪 bulk：程序 / 泛 CAF UMI、
      非腺泡非内分泌总 UMI、细胞数。区域层面（150 / 100 µm）的标签在评估脚本里由细胞重新汇总。
输出：results/he/hd/{sample}.npz（features, ix, iy, x_px, y_px, mpp, prog_umi, pan_umi, total_umi, n_cells）
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
import torch
import yaml

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from hemorph import config as C  # noqa: E402
from hemorph import expression as E  # noqa: E402
from hemorph import io_visiumhd as IO  # noqa: E402
from hemorph.he_image import read_cloupe_image  # noqa: E402

from he_05_extract_tcga_uni2h import load_model  # noqa: E402

OUT = PROJ / "results" / "he" / "hd"
OUT.mkdir(parents=True, exist_ok=True)
PROG = yaml.safe_load((PROJ / "results" / "fig1" / "signature" / "programs.yaml").read_text())
SETS = {"prog": PROG["CTHRC1_matrix_CAF"], "pan": PROG["pan_CAF"]}
FIELD_UM, MIN_TISSUE, BATCH = 128.0, 0.5, 16


def tissue_fraction(img, step_px):
    """Tissue fraction per grid cell using the same HSV/brightness thresholds as the TCGA extraction."""
    H, W = img.shape[:2]
    scale = 2048 / max(H, W)
    th = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(th, cv2.COLOR_RGB2HSV_FULL)
    br = th.mean(2)
    tissue = ((hsv[:, :, 1] > 18) & (br > 40) & (br < 235)).astype(np.uint32)
    I = np.pad(tissue.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    nx, ny = int(W // step_px), int(H // step_px)
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    gx, gy = gx.ravel(), gy.ravel()
    x0 = np.floor(gx * step_px * scale).astype(int); x1 = np.maximum(x0 + 1, np.ceil((gx + 1) * step_px * scale).astype(int))
    y0 = np.floor(gy * step_px * scale).astype(int); y1 = np.maximum(y0 + 1, np.ceil((gy + 1) * step_px * scale).astype(int))
    x1, y1 = np.minimum(x1, tissue.shape[1]), np.minimum(y1, tissue.shape[0])
    frac = (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / np.maximum(1, (x1 - x0) * (y1 - y0))
    return gx, gy, frac


def main():
    os.nice(10)
    torch.set_num_threads(4)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, prep = load_model(dev)
    from PIL import Image
    cfg = C.load_config()
    for s in C.samples(cfg):
        out = OUT / f"{s.sample_id}.npz"
        if out.exists():
            print(f"{s.sample_id}: exists, skip"); continue
        t0 = time.time()
        img = read_cloupe_image(s.he_image)
        mpp = s.mpp
        step = FIELD_UM / mpp
        gx, gy, frac = tissue_fraction(img, step)
        keep = frac >= MIN_TISSUE
        gx, gy = gx[keep], gy[keep]
        feats = []
        for i in range(0, len(gx), BATCH):
            ims = []
            for x, y in zip(gx[i:i + BATCH], gy[i:i + BATCH]):
                x0, y0 = int(round(x * step)), int(round(y * step))
                ims.append(Image.fromarray(img[y0:y0 + int(round(step)), x0:x0 + int(round(step))]))
            with torch.inference_mode():
                feats.append(model(torch.stack([prep(im) for im in ims]).to(dev)).float().cpu().numpy())
        F = np.concatenate(feats).astype(np.float16)
        # ---- tile-level labels from cells (nucleus-matched cells only)
        a = IO.read_cell_counts(s)
        nuc = IO.read_polygons(s, "nucleus")
        a = a[a.obs_names.isin(nuc.index)]
        X = sp.csr_matrix(a.X)
        var = list(a.var_names)
        progs = C.load_programs(cfg, a.var_names)
        excl = set(progs["acinar"]) | set(progs["endocrine"]) | set(E.ACINAR_EXTRA)
        nonac = np.array([g not in excl for g in var])
        xy = nuc.loc[a.obs_names, ["x", "y"]].to_numpy()
        cx, cy = E.region_index(xy[:, 0], xy[:, 1], mpp, FIELD_UM)
        key = cx.astype(np.int64) * 1_000_000 + cy
        keys, P = E.region_pseudobulk(X, key)
        u = E.program_umis(P, var, SETS)
        tot = np.asarray(P[:, nonac].sum(1)).ravel()
        ncell = np.bincount(np.unique(key, return_inverse=True)[1])
        lut = {k: i for i, k in enumerate(keys)}
        tk = gx.astype(np.int64) * 1_000_000 + gy
        j = np.array([lut.get(k, -1) for k in tk])
        g = lambda v: np.where(j >= 0, np.asarray(v)[np.maximum(j, 0)], 0)  # noqa: E731
        np.savez(out, features=F, ix=gx, iy=gy, x_px=(gx * step).astype(np.float32), y_px=(gy * step).astype(np.float32),
                 mpp=mpp, prog_umi=g(u["prog"].to_numpy()), pan_umi=g(u["pan"].to_numpy()), total_umi=g(tot),
                 n_cells=g(ncell), tissue_frac=frac[keep])
        print(f"{s.sample_id}: {len(gx)} tissue tiles, mpp {mpp:.3f}, tiles with cells {(j >= 0).sum()}, "
              f"{time.time() - t0:.0f}s", flush=True)
        del img


if __name__ == "__main__":
    main()
