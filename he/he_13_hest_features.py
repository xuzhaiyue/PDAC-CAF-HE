"""修订 A8 第 1 步：HEST 图块的 UNI2-h 特征与斑点层面计数（hemorph 环境，MPS，低负载）。规则见 config/fig1.yaml 的 A8（已冻结）。

输入（只读）：/Volumes/INTENSO_BIG/HEST1k/{patches,st}/{id}
输出：results/he/A8/hest/{id}.npz
      features  斑点图块 × 1536（float16）；coords（全分辨率像素坐标）；pixel_size（µm/px）
      prog_umi / pan_umi / total_umi（非腺泡非内分泌）  每个斑点的计数，用于之后的区域标签
      genes_prog_present / genes_pan_present  该样本表达矩阵里能找到的程序 / 泛 CAF 基因数
低负载：nice 19、torch 2 线程、每批 32 张、批间暂停 0.05 s、1 分钟负载 > 8 时暂停。可反复运行（已完成的样本跳过）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from hemorph import config as C  # noqa: E402
from hemorph import expression as E  # noqa: E402

from he_05_extract_tcga_uni2h import load_model  # noqa: E402

HEST = Path("/Volumes/INTENSO_BIG/HEST1k")
OUT = PROJ / "results" / "he" / "A8" / "hest"
OUT.mkdir(parents=True, exist_ok=True)
PROG = yaml.safe_load((PROJ / "results" / "fig1" / "signature" / "programs.yaml").read_text())
BATCH, PAUSE, MAX_LOAD = 32, 0.05, 8.0


def wait_load():
    while os.getloadavg()[0] > MAX_LOAD:
        time.sleep(30)


def spot_counts(sid, barcodes, cfg):
    a = ad.read_h5ad(HEST / "st" / f"{sid}.h5ad")
    a.var_names_make_unique()
    a = a[[b for b in barcodes if b in a.obs_names]]
    X = sp.csr_matrix(a.X)
    var = list(a.var_names)
    progs = C.load_programs(cfg, a.var_names)
    excl = set(progs["acinar"]) | set(progs["endocrine"]) | set(E.ACINAR_EXTRA)
    nonac = np.array([g not in excl for g in var])
    u = E.program_umis(X, var, {"prog": PROG["CTHRC1_matrix_CAF"], "pan": PROG["pan_CAF"]})
    tot = np.asarray(X[:, nonac].sum(1)).ravel()
    pres = (sum(g in var for g in PROG["CTHRC1_matrix_CAF"]), sum(g in var for g in PROG["pan_CAF"]))
    idx = {b: i for i, b in enumerate(a.obs_names)}
    j = np.array([idx.get(b, -1) for b in barcodes])
    g = lambda v: np.where(j >= 0, np.asarray(v)[np.maximum(j, 0)], np.nan)  # noqa: E731
    return g(u["prog"].to_numpy()), g(u["pan"].to_numpy()), g(tot), pres


def main():
    os.nice(19)
    torch.set_num_threads(2)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, prep = load_model(dev)
    cfg = C.load_config()
    sel = pd.read_csv(HEST / "_manifest" / "selection.tsv", sep="\t")
    t0 = time.time()
    for n, r in enumerate(sel.itertuples(), 1):
        out = OUT / f"{r.id}.npz"
        if out.exists():
            continue
        ts = time.time()
        with h5py.File(HEST / "patches" / f"{r.id}.h5", "r") as h:
            N = h["img"].shape[0]
            coords = h["coords"][:]
            bc = [b[0].decode() if isinstance(b[0], bytes) else str(b[0]) for b in h["barcode"][:]]
            px = float(h["img"].attrs.get("pixel_size", np.nan))
            feats = []
            for i in range(0, N, BATCH):
                wait_load()
                imgs = h["img"][i:i + BATCH]
                with torch.inference_mode():
                    feats.append(model(torch.stack([prep(Image.fromarray(im)) for im in imgs]).to(dev)).float().cpu().numpy())
                time.sleep(PAUSE)
        F = np.concatenate(feats).astype(np.float16)
        prog, pan, tot, pres = spot_counts(r.id, bc, cfg)
        meta = json.load(open(HEST / "metadata" / f"{r.id}.json"))
        np.savez(out, features=F, coords=coords, barcode=np.array(bc), pixel_size=px, prog_umi=prog, pan_umi=pan,
                 total_umi=tot, genes_prog_present=pres[0], genes_pan_present=pres[1],
                 organ=meta.get("organ"), dataset=meta.get("dataset_title"), st_technology=meta.get("st_technology"),
                 patient=str(meta.get("patient")), preservation=str(meta.get("preservation_method")))
        print(f"[{n}/{len(sel)}] {r.id} {meta.get('organ')}: {N} patches, genes {pres[0]}/9 + {pres[1]}/4, "
              f"{time.time() - ts:.0f}s (total {(time.time() - t0) / 3600:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
