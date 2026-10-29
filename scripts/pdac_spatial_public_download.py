"""下载公开的胰腺癌空间转录组（带全分辨率 H&E）到共享位置（用户 2026-10-21 同意；多个课题共用）。hemorph 环境。

数据集：
  GSE274557  转移性胰腺癌空间图谱（快速尸检；原发 + 肝 / 肺 / 腹膜转移；55 个样本，13 名患者）  全部样本
  GSE327056  人 PDAC Visium（4 个样本）                                                         全部样本
  GSE211895  只取 3 例人胰腺癌（DonorA/B/C，FFPE，探针法）；其余为大鼠 / 小鼠样本，不下载
布局：/Volumes/INTENSO_BIG/PDAC_spatial_public/{GSE}/samples/{GSM}/<GEO 原始文件名>
      {GSE}/metadata/samples.tsv（GSM、标题、全部 characteristics、文件列表）、series.txt（GEO 系列描述）
      {GSE}/_manifest/files.tsv（URL、字节数、服务器字节数、sha256、时间）
      根目录 README.md
可反复运行：大小与服务器一致的文件跳过。
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/Volumes/INTENSO_BIG/PDAC_spatial_public")
DATASETS = {"GSE274557": None, "GSE327056": None, "GSE211895": re.compile(r"^Donor[ABC], FFPE, probes$")}
UA = ["-A", "Mozilla/5.0"]


def get(url):
    return subprocess.run(["curl", "-sL", "--max-time", "300", *UA, url], capture_output=True, text=True).stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def samples_of(gse):
    txt = get(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}&targ=gsm&form=text&view=brief")
    rows, cur = [], None
    for line in txt.splitlines():
        if line.startswith("^SAMPLE"):
            cur = {"gsm": line.split("=")[1].strip(), "characteristics": [], "files": []}
            rows.append(cur)
        elif cur is None:
            continue
        elif line.startswith("!Sample_title"):
            cur["title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_characteristics_ch1"):
            cur["characteristics"].append(line.split("=", 1)[1].strip())
        elif line.startswith("!Sample_supplementary_file") and "NONE" not in line:
            cur["files"].append(line.split("=", 1)[1].strip().replace("ftp://", "https://"))
    return rows


def remote_size(url):
    h = subprocess.run(["curl", "-sIL", "--max-time", "60", *UA, url], capture_output=True, text=True).stdout
    m = re.findall(r"(?i)content-length:\s*(\d+)", h)
    return int(m[-1]) if m else None


def main():
    os.nice(10)
    for gse, rule in DATASETS.items():
        d = ROOT / gse
        (d / "metadata").mkdir(parents=True, exist_ok=True); (d / "_manifest").mkdir(exist_ok=True)
        (d / "metadata" / "series.txt").write_text(get(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}&targ=self&form=text&view=brief"))
        S = [s for s in samples_of(gse) if rule is None or rule.match(s.get("title", ""))]
        pd.DataFrame([{"gsm": s["gsm"], "title": s.get("title"), "characteristics": " | ".join(s["characteristics"]),
                       "n_files": len(s["files"]), "files": " ".join(Path(u).name for u in s["files"])} for s in S]
                     ).to_csv(d / "metadata" / "samples.tsv", sep="\t", index=False)
        man_p = d / "_manifest" / "files.tsv"
        man = pd.read_csv(man_p, sep="\t").to_dict("records") if man_p.exists() else []
        done = {r["url"] for r in man if r["status"] == "ok"}
        print(f"{gse}: {len(S)} samples, {sum(len(s['files']) for s in S)} files", flush=True)
        for s in S:
            sd = d / "samples" / s["gsm"]; sd.mkdir(parents=True, exist_ok=True)
            for u in s["files"]:
                dst = sd / Path(u).name
                exp = remote_size(u)
                if u in done and dst.exists() and dst.stat().st_size == exp:
                    continue
                for _ in range(5):
                    subprocess.run(["nice", "-n", "10", "curl", "-sSL", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "20",
                                    *UA, "-o", str(dst), u])
                    if dst.exists() and exp and dst.stat().st_size == exp:
                        break
                    time.sleep(20)
                ok = dst.exists() and exp is not None and dst.stat().st_size == exp
                man = [r for r in man if r["url"] != u] + [{"gsm": s["gsm"], "file": dst.name, "url": u, "bytes": dst.stat().st_size if dst.exists() else 0,
                                                             "expected_bytes": exp, "sha256": sha256(dst) if ok else "",
                                                             "status": "ok" if ok else "failed", "time": time.strftime("%Y-%m-%d %H:%M:%S")}]
                pd.DataFrame(man).to_csv(man_p, sep="\t", index=False)
            print(f"  {s['gsm']} {s.get('title')} done", flush=True)
        m = pd.DataFrame(man)
        print(f"{gse}: {m.status.value_counts().to_dict()}, {m[m.status == 'ok'].bytes.sum() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
