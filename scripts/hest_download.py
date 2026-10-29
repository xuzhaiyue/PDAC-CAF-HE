"""下载 HEST-1k 人类癌症子集到共享位置（用户 2026-10-21 同意，约 93 GB；供多个课题共用）。hemorph 环境。

选择：HEST v1.3.0 中 Homo sapiens、癌症、非 Xenium、技术为 Visium / Visium HD / Visium HD 3'、
      器官不为脑 / 眼 / 淋巴结 → 163 个样本（含 5 个胰腺癌）。
内容：每个样本的 patches/{id}.h5（224 px、0.5 µm/px 斑点图块）、st/{id}.h5ad（空间表达）、metadata/{id}.json，
      以及根目录的 HEST_v1_3_0.csv。不下载全切片（wsis/）。
布局：与 HEST 仓库一致（hf_hub_download 的 local_dir），可直接被 HEST 官方工具读取。
清单：_manifest/selection.tsv（样本列表与选择依据）、_manifest/download_log.tsv（每个文件的大小、sha256、时间）、README.md
可反复运行：已存在且大小正确的文件跳过。
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path("/Volumes/INTENSO_BIG/HEST1k")
MAN = ROOT / "_manifest"
REPO, VERSION_CSV = "MahmoodLab/hest", "HEST_v1_3_0.csv"
TECH = {"Visium", "Visium HD", "Visium HD 3'"}
EXCLUDE_ORGANS = {"Brain", "Eye", "Lymph node"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.nice(10)
    MAN.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    rev = api.dataset_info(REPO).sha
    meta_p = hf_hub_download(REPO, VERSION_CSV, repo_type="dataset", local_dir=ROOT, revision=rev)
    m = pd.read_csv(meta_p)
    sel = m[(m.species == "Homo sapiens") & m.disease_state.astype(str).str.contains("Cancer", case=False)
            & m.st_technology.isin(TECH) & ~m.organ.isin(EXCLUDE_ORGANS)].copy()
    sel["selection_rule"] = "human cancer; Visium/Visium HD/HD 3'; organ not brain/eye/lymph node"
    sel.to_csv(MAN / "selection.tsv", sep="\t", index=False)
    print(f"HEST revision {rev[:10]}; selected {len(sel)} samples", flush=True)
    files = [f"{d}/{i}{e}" for i in sel.id for d, e in [("metadata", ".json"), ("st", ".h5ad"), ("patches", ".h5")]]
    sizes = {}
    for k in range(0, len(files), 100):
        for x in api.get_paths_info(REPO, files[k:k + 100], repo_type="dataset", revision=rev):
            sizes[x.path] = x.size or (x.lfs.size if x.lfs else 0)
    log_p = MAN / "download_log.tsv"
    log = pd.read_csv(log_p, sep="\t").to_dict("records") if log_p.exists() else []
    done = {r["path"] for r in log if r.get("status") == "ok"}
    t0 = time.time()
    for n, f in enumerate(files, 1):
        dst = ROOT / f
        if f in done and dst.exists() and dst.stat().st_size == sizes.get(f):
            continue
        for attempt in range(5):
            try:
                hf_hub_download(REPO, f, repo_type="dataset", local_dir=ROOT, revision=rev)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt + 1} {f}: {type(e).__name__}", flush=True)
                time.sleep(30)
        ok = dst.exists() and dst.stat().st_size == sizes.get(f)
        log.append({"path": f, "bytes": dst.stat().st_size if dst.exists() else 0, "expected_bytes": sizes.get(f),
                    "sha256": sha256(dst) if ok else "", "status": "ok" if ok else "failed",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"), "hest_revision": rev})
        pd.DataFrame(log).to_csv(log_p, sep="\t", index=False)
        if n % 25 == 0 or not ok:
            gb = sum(r["bytes"] for r in log if r["status"] == "ok") / 1e9
            print(f"[{n}/{len(files)}] {'ok' if ok else 'FAILED'} {f}  total {gb:.1f} GB  {time.time() - t0:.0f}s", flush=True)
    s = pd.DataFrame(log).drop_duplicates("path", keep="last")
    print(s.status.value_counts().to_string(), f"\n{s[s.status == 'ok'].bytes.sum() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
