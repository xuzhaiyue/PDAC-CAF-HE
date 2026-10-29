"""H&E image handling: alignment check against Space Ranger and resampling for the models.

Coordinate frames used throughout:
  fullres px  — pixels of the microscope image given to `spaceranger --image` (10x geometry)
  model px    — pixels of the prepared image at `image.model_mpp` (CellViT++, UNI2-h)
  model_px = fullres_px * src_mpp / model_mpp   (both images share the origin)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(8)


def read_cloupe_image(path: str | Path, threads: int = 16) -> np.ndarray:
    """Stitch the full-resolution microscope image embedded in a Loupe (.cloupe) file.

    Space Ranger embeds the `--image` H&E as a DeepZoom pyramid of lossless PNG tiles
    (512 px, 1 px overlap, names '<level>/<col>_<row>.png'); the top level is full resolution.
    """
    import gzip
    import json
    from concurrent.futures import ThreadPoolExecutor

    with open(path, "rb") as fh:
        header = json.loads(fh.read(4096).split(b"\x00")[0])
        ib = header["indexBlock"]
        fh.seek(ib["Start"])
        idx = json.loads(gzip.decompress(fh.read(ib["End"] - ib["Start"])))
    tiles = idx["SpatialImageTiles"][0]
    W, H = idx["SpatialImages"][0]["Dims"]
    size, ov = tiles["TileSize"], tiles["TileOverlap"]
    top = max(int(k.split("/")[0]) for k in tiles["Tiles"])
    entries = [(k, v) for k, v in tiles["Tiles"].items() if k.startswith(f"{top}/")]
    img = np.zeros((H, W, 3), np.uint8)

    def place(item):
        name, e = item
        c, r = map(int, name.split("/")[1][:-4].split("_"))
        with open(path, "rb") as fh:
            fh.seek(e["Start"])
            buf = np.frombuffer(fh.read(e["End"] - e["Start"]), np.uint8)
        t = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x0, y0 = c * size - (ov if c else 0), r * size - (ov if r else 0)
        h, w = min(t.shape[0], H - y0), min(t.shape[1], W - x0)
        img[y0:y0 + h, x0:x0 + w] = t[:h, :w]

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(place, entries))
    return img


def read_rgb(path: str | Path) -> np.ndarray:
    """Read the full-resolution H&E into memory (a Visium HD capture area fits easily in RAM)."""
    path = Path(path)
    if path.suffix.lower() == ".cloupe":
        return read_cloupe_image(path)
    if path.suffix.lower() in (".tif", ".tiff", ".btf", ".qptiff", ".ome.tif"):
        import tifffile
        img = np.asarray(tifffile.imread(path, series=0, level=0))
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:  # 16-bit scans
        img = (img / max(img.max(), 1) * 255).astype(np.uint8)
    return img


def check_alignment(img_shape: tuple[int, int], expected_shape: tuple[int, int], tol: float = 0.01) -> dict:
    """The H&E must be the exact file used by spaceranger: its size must match the implied full-res size."""
    h, w = img_shape[:2]
    eh, ew = expected_shape
    rel = max(abs(h - eh) / eh, abs(w - ew) / ew)
    return {"image_hw": (h, w), "expected_hw": (eh, ew), "rel_diff": rel, "ok": rel <= tol}


def resample(img: np.ndarray, src_mpp: float, model_mpp: float, effective_mpp: float | None) -> np.ndarray:
    """src -> (optional blur-down to effective_mpp) -> model_mpp.

    With effective_mpp set to the coarsest scan (0.68 µm/px), the 0.36 µm/px slides are degraded
    to the same optical detail before reaching 0.5 µm/px, so resolution is not confounded with patient.
    """
    def to(im, a, b):
        f = a / b
        size = (int(round(im.shape[1] * f)), int(round(im.shape[0] * f)))
        return cv2.resize(im, size, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)

    if effective_mpp is not None and effective_mpp > src_mpp:
        img = to(img, src_mpp, effective_mpp)
        src_mpp = effective_mpp
    return to(img, src_mpp, model_mpp)


def write_pyramidal_tiff(img: np.ndarray, dst: str | Path, mpp: float, tile: int = 512, min_size: int = 1024,
                         compression: str = "jpeg"):
    """Tiled multi-page pyramid readable by OpenSlide's generic-tiff driver (one level per page)."""
    import tifffile
    px_per_cm = 1e4 / mpp
    with tifffile.TiffWriter(dst, bigtiff=True) as tif:
        level, scale = img, 1
        while True:
            tif.write(level, tile=(tile, tile), photometric="rgb", compression=compression,
                      resolution=(px_per_cm / scale, px_per_cm / scale), resolutionunit="CENTIMETER",
                      subfiletype=0 if scale == 1 else 1)
            if min(level.shape[:2]) // 2 < min_size:
                break
            level = cv2.resize(level, (level.shape[1] // 2, level.shape[0] // 2), interpolation=cv2.INTER_AREA)
            scale *= 2


def write_sidecar(dst: str | Path, **kw):
    Path(str(dst) + ".json").write_text(json.dumps(kw, indent=2))


def read_sidecar(prepared: str | Path) -> dict:
    return json.loads(Path(str(prepared) + ".json").read_text())


# Ruifrok & Johnston H&E(+DAB) stain matrix; rows = stains, cols = RGB OD
_RUIFROK = np.array([[0.650, 0.704, 0.286], [0.072, 0.990, 0.105], [0.268, 0.570, 0.776]])
_RUIFROK_INV = np.linalg.inv(_RUIFROK / np.linalg.norm(_RUIFROK, axis=1, keepdims=True))


def hed(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> (H, E, D) optical-density channels."""
    od = -np.log10(np.clip(rgb.astype(np.float32), 1, 255) / 255.0)
    return np.clip(od.reshape(-1, 3) @ _RUIFROK_INV.astype(np.float32), 0, None).reshape(rgb.shape)
