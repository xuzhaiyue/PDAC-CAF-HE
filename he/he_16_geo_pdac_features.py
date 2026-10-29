"""修订 A9 第 1 步：公开胰腺癌 Visium 样本（GEO）的斑点图块特征与计数（hemorph 环境，MPS）。规则见 config/fig1.yaml 的 A9（已冻结）。

输入（只读）：/Volumes/INTENSO_BIG/PDAC_spatial_public/{GSE274557,GSE327056,GSE211895}/samples/{GSM}/
处理：读取全分辨率原图（jpg / tif，gzip）、斑点坐标（tissue_positions*.csv，表头有 / 无两种）、表达矩阵（h5 或 mtx）；
      原图与 spaceranger 全分辨率坐标的比例由 hires 图尺寸 / tissue_hires_scalef 核对，不一致时按比例换算坐标与 µm/px；
      以每个组织内斑点为中心切 112 µm 方块 → 224 px → UNI2-h（与 HEST / A8 一致）。
对齐质控：组织内斑点图块的平均 HSV 饱和度应明显高于组织外斑点（比值记录在 qc 中）。
输出：results/he/A9/geo/{GSM}.npz（字段与 A8 的 HEST 特征文件一致，另加 patient、tissue、treatment），results/he/A9/geo_qc.tsv
负载：先按用户要求 nice 10 / 8 线程运行，用户随后要求降低 → nice 19、2 线程、批量 32、批间暂停、负载 > 8 时暂停（已完成的样本不重做）。
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import tifffile
import torch
import yaml
from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from hemorph import config as C  # noqa: E402
from hemorph import expression as E  # noqa: E402

from he_05_extract_tcga_uni2h import load_model  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/Volumes/INTENSO_BIG/PDAC_spatial_public")
OUT = PROJ / "results" / "he" / "A9" / "geo"
OUT.mkdir(parents=True, exist_ok=True)
PROG = yaml.safe_load((PROJ / "results" / "fig1" / "signature" / "programs.yaml").read_text())
FIELD_UM, BATCH, PAUSE, MAX_LOAD = 112.0, 32, 0.05, 8.0
POS_COLS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]


def opener(p: Path):
    return gzip.open(p, "rb") if p.name.endswith(".gz") else open(p, "rb")


def files_of(sd: Path):
    """Collect per-sample inputs from loose files or from a spatial.tar.gz bundle."""
    f = {p.name: p for p in sd.iterdir() if not p.name.startswith("._")}
    out = {"image": next(p for n, p in f.items() if re.search(r"\.(jpg|jpeg|tif|tiff)\.gz$", n, re.I)
                         and not re.search(r"hires|lowres|fiducial|detected|aligned", n, re.I))}
    tar = next((p for n, p in f.items() if n.endswith("spatial.tar.gz")), None)
    if tar:
        t = tarfile.open(tar)
        get = lambda pat: next(t.extractfile(m).read() for m in t.getmembers() if re.search(pat, m.name))  # noqa: E731
        out["positions"], out["scalefactors"], out["hires"] = get(r"tissue_positions[^/]*\.csv$"), get(r"scalefactors_json\.json$"), get(r"tissue_hires_image\.png$")
    else:
        rd = lambda pat: opener(next(p for n, p in f.items() if re.search(pat, n))).read()  # noqa: E731
        out["positions"], out["scalefactors"], out["hires"] = rd(r"tissue_positions[^/]*\.csv"), rd(r"scalefactors_json"), rd(r"tissue_hires_image\.png")
    out["h5"] = next((p for n, p in f.items() if n.endswith("filtered_feature_bc_matrix.h5")), None)
    out["mtx"] = {k: next((p for n, p in f.items() if n.endswith(f"_{k}")), None) for k in ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]}
    return out


def read_positions(b: bytes) -> pd.DataFrame:
    head = b[:200].decode(errors="ignore")
    df = pd.read_csv(io.BytesIO(b), header=0 if head.startswith("barcode") else None)
    df.columns = POS_COLS
    return df.set_index("barcode")


def read_counts(fs):
    if fs["h5"] is not None:
        a = sc.read_10x_h5(fs["h5"])
    else:
        d = OUT.parent / "_tmp_mtx"; d.mkdir(exist_ok=True)
        for k, p in fs["mtx"].items():
            (d / k).write_bytes(p.read_bytes())
        a = sc.read_10x_mtx(d, var_names="gene_symbols")
    a.var_names_make_unique()
    return a


def read_image(p: Path) -> np.ndarray:
    raw = opener(p).read()
    if re.search(r"\.tiff?\.gz$", p.name, re.I):
        img = tifffile.imread(io.BytesIO(raw))
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        return np.ascontiguousarray(img[..., :3]).astype(np.uint8)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def main():
    os.nice(19)                      # 用户要求降低负载：nice 19、2 线程、批量 32、批间暂停、负载 > 8 时暂停
    torch.set_num_threads(2)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, prep = load_model(dev)
    cfg = C.load_config()
    inv = pd.read_csv(PROJ / "results" / "he" / "pdac_spatial_public_inventory.tsv", sep="\t")
    inv = inv[~inv.tissue.str.contains("PDX", na=False)]
    qc_f = OUT.parent / "geo_qc.tsv"
    qc_rows = pd.read_csv(qc_f, sep="\t").to_dict("records") if qc_f.exists() else []   # keep QC of earlier runs
    for r in inv.itertuples():
        out = OUT / f"{r.gsm}.npz"
        if out.exists():
            continue
        t0 = time.time()
        fs = files_of(ROOT / r.dataset / "samples" / r.gsm)
        sf = json.loads(fs["scalefactors"])
        hires = Image.open(io.BytesIO(fs["hires"]))
        img = read_image(fs["image"])
        H, W = img.shape[:2]
        full_w = hires.size[0] / sf["tissue_hires_scalef"]                 # width of the image spaceranger used
        ratio = W / full_w                                                  # >1 or <1 if GEO image differs in scale
        mpp = 55.0 / sf["spot_diameter_fullres"] / ratio
        pos = read_positions(fs["positions"])
        pos = pos[pos.in_tissue == 1]
        a = read_counts(fs)
        pos = pos.loc[pos.index.intersection(a.obs_names)]
        a = a[pos.index]
        xy = np.c_[pos.pxl_col_in_fullres, pos.pxl_row_in_fullres] * ratio
        half = FIELD_UM / mpp / 2
        feats, sat = [], []
        for i in range(0, len(xy), BATCH):
            while os.getloadavg()[0] > MAX_LOAD:
                time.sleep(30)
            time.sleep(PAUSE)
            ims = []
            for x, y in xy[i:i + BATCH]:
                x0, y0 = int(max(0, x - half)), int(max(0, y - half))
                crop = img[y0:int(min(H, y + half)), x0:int(min(W, x + half))]
                pil = Image.fromarray(crop).resize((224, 224))
                sat.append(np.asarray(pil.convert("HSV"))[..., 1].mean())
                ims.append(pil)
            with torch.inference_mode():
                feats.append(model(torch.stack([prep(im) for im in ims]).to(dev)).float().cpu().numpy())
        # alignment control: background patches at the image corners (no tissue expected) vs tissue spots
        corners = [img[:int(2 * half), :int(2 * half)], img[-int(2 * half):, -int(2 * half):]]
        bg_sat = np.mean([np.asarray(Image.fromarray(c).convert("HSV"))[..., 1].mean() for c in corners])
        X = sp.csr_matrix(a.X); var = list(a.var_names)
        progs = C.load_programs(cfg, a.var_names)
        excl = set(progs["acinar"]) | set(progs["endocrine"]) | set(E.ACINAR_EXTRA)
        nonac = np.array([g not in excl for g in var])
        u = E.program_umis(X, var, {"prog": PROG["CTHRC1_matrix_CAF"], "pan": PROG["pan_CAF"]})
        tot = np.asarray(X[:, nonac].sum(1)).ravel()
        gp, gc = sum(g in var for g in PROG["CTHRC1_matrix_CAF"]), sum(g in var for g in PROG["pan_CAF"])
        patient = re.sub(r"[A-Z]$", "", r.title) if r.dataset == "GSE274557" else ("GSE327056_patient" if r.dataset == "GSE327056" else r.gsm)
        np.savez(out, features=np.concatenate(feats).astype(np.float16), coords=np.round(xy).astype(np.int64),
                 barcode=np.array(pos.index), pixel_size=mpp, prog_umi=u["prog"].to_numpy(), pan_umi=u["pan"].to_numpy(),
                 total_umi=tot, genes_prog_present=gp, genes_pan_present=gc, organ="Pancreas", dataset=r.dataset,
                 st_technology="Visium", patient=patient, preservation="", tissue=r.tissue, treatment=str(r.treatment))
        q = {"gsm": r.gsm, "dataset": r.dataset, "title": r.title, "patient": patient, "tissue": r.tissue,
             "n_spots": len(xy), "um_per_px": mpp, "image_scale_ratio": ratio, "tissue_sat": float(np.mean(sat)),
             "background_sat": float(bg_sat), "genes_prog": gp, "genes_pan": gc, "seconds": round(time.time() - t0)}
        qc_rows.append(q)
        pd.DataFrame(qc_rows).to_csv(OUT.parent / "geo_qc.tsv", sep="\t", index=False, mode="w")
        print(f"{r.gsm} {r.title} ({r.tissue}): {len(xy)} spots, {mpp:.3f} um/px, scale {ratio:.2f}, "
              f"sat tissue {np.mean(sat):.0f} vs bg {bg_sat:.0f}, genes {gp}/9+{gc}/4, {time.time() - t0:.0f}s", flush=True)
        del img


if __name__ == "__main__":
    main()
