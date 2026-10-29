"""Download TCGA-PAAD diagnostic whole-slide images (209 slides, ~245 GB). hemorph environment.

Manifest: gdc_paad_dx_slides.csv (read only); md5 fetched in batch from the GDC API.
Method: GDC data API + resumable curl (.part temporary files), 8 in parallel; size and md5 checked after each download before renaming to the final file name.
Cohort patients (the 147 ductal adenocarcinomas) are downloaded first. Existing files with correct md5 are skipped.
Log: results/he/wsi_download/status.tsv (one row per file; re-run to resume)
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
MANIFEST = Path("/Volumes/zhaiyue/TFF1/project_PanCancer_Neddylation/02_intermediate/fig8/gdc_paad_dx_slides.csv")
DEST = Path("/Volumes/INTENSO/TCGA_PAAD_svs")
TMP = DEST / "_partial"
LOG = PROJ / "results" / "he" / "wsi_download"
WORKERS = 8


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gdc_md5(ids: list[str]) -> dict[str, str]:
    body = json.dumps({"filters": {"op": "in", "content": {"field": "file_id", "value": ids}},
                       "fields": "file_id,md5sum,file_size", "size": len(ids) + 10}).encode()
    req = urllib.request.Request("https://api.gdc.cancer.gov/files", data=body, headers={"Content-Type": "application/json"})
    hits = json.load(urllib.request.urlopen(req, timeout=120))["data"]["hits"]
    return {h["file_id"]: h["md5sum"] for h in hits}


def download(row) -> dict:
    final, part = DEST / row.file_name, TMP / f"{row.file_id}.part"
    rec = {"file_id": row.file_id, "file_name": row.file_name, "patient": row.file_name[:12],
           "expected_bytes": int(row.file_size), "expected_md5": row.md5}
    t0 = time.time()
    if final.exists() and final.stat().st_size == row.file_size and md5(final) == row.md5:
        return {**rec, "status": "verified_existing", "seconds": 0}
    for attempt in range(5):
        r = subprocess.run(["nice", "-n", "10", "curl", "-sS", "-L", "--fail", "-C", "-", "--retry", "5",
                            "--retry-delay", "20", "--connect-timeout", "60", "-o", str(part),
                            f"https://api.gdc.cancer.gov/data/{row.file_id}"], capture_output=True, text=True)
        if part.exists() and part.stat().st_size == row.file_size:
            break
        if part.exists() and part.stat().st_size > row.file_size:
            part.unlink()
        time.sleep(30)
    if not part.exists() or part.stat().st_size != row.file_size:
        return {**rec, "status": "failed_size", "seconds": time.time() - t0, "error": r.stderr[-300:]}
    got = md5(part)
    if got != row.md5:
        part.unlink()
        return {**rec, "status": "failed_md5", "actual_md5": got, "seconds": time.time() - t0}
    part.rename(final)
    return {**rec, "status": "downloaded_verified", "actual_md5": got, "seconds": time.time() - t0}


def main():
    DEST.mkdir(exist_ok=True); TMP.mkdir(exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MANIFEST)
    m["md5"] = m.file_id.map(gdc_md5(list(m.file_id)))
    cohort = set(pd.read_csv(PROJ / "results" / "fig2" / "tcga_scores.tsv", sep="\t", index_col=0).index)
    m["in_cohort"] = m.file_name.str[:12].isin(cohort)
    m = m.sort_values(["in_cohort", "file_size"], ascending=[False, True])
    m.to_csv(LOG / "manifest_with_md5.tsv", sep="\t", index=False)
    print(f"{len(m)} slides, {m.file_size.sum() / 1e9:.1f} GB; cohort patients first ({m.in_cohort.sum()} slides)", flush=True)
    done = []
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for i, res in enumerate(ex.map(download, m.itertuples(index=False)), 1):
            done.append(res)
            pd.DataFrame(done).to_csv(LOG / "status.tsv", sep="\t", index=False)
            print(f"[{i}/{len(m)}] {res['status']:20s} {res['file_name'][:23]} {res['expected_bytes'] / 1e9:.2f} GB "
                  f"{res.get('seconds', 0):.0f}s", flush=True)
    s = pd.DataFrame(done)
    print(s.status.value_counts().to_string())


if __name__ == "__main__":
    main()
