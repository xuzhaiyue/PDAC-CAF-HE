"""Readers for Space Ranger (Visium HD) segmented outputs.

Coordinates: all 10x geometries are in full-res pixels of the microscope image passed to
`spaceranger --image` (x = column, y = row). Nucleus polygons are unions of 2 µm bins, so they
are pixelated at 2 µm — usable for locating nuclei, NOT for nuclear shape features (those come
from CellViT++ contours).
"""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import shapely

from .config import Sample


def cellid_str(i: int) -> str:
    return f"cellid_{int(i):09d}-1"


def read_cell_counts(s: Sample) -> ad.AnnData:
    a = sc.read_10x_h5(s.seg / "filtered_feature_cell_matrix.h5")
    a.var_names_make_unique()
    a.obs["sample_id"] = s.sample_id
    a.obs["patient_id"] = s.patient_id
    return a


def read_polygons(s: Sample, kind: str = "nucleus", keep_geometry: bool = False) -> pd.DataFrame:
    """kind: 'nucleus' or 'cell'. Returns one row per 10x segmented object."""
    path = s.seg / f"{kind}_segmentations.geojson"
    feats = json.loads(path.read_text())["features"]
    ids, polys = [], []
    for f in feats:
        g = f["geometry"]
        if g["type"] == "Polygon":
            polys.append(shapely.Polygon(g["coordinates"][0], g["coordinates"][1:]))
        else:  # MultiPolygon: keep the largest part
            parts = [shapely.Polygon(p[0], p[1:]) for p in g["coordinates"]]
            polys.append(max(parts, key=lambda p: p.area))
        ids.append(cellid_str(f["properties"]["cell_id"]))
    geoms = np.array(polys, dtype=object)
    cents = shapely.centroid(geoms)
    df = pd.DataFrame({
        "cell_id": ids,
        "x": shapely.get_x(cents),
        "y": shapely.get_y(cents),
        "area_px": shapely.area(geoms),
    }).set_index("cell_id")
    if keep_geometry:
        df["geometry"] = geoms
    return df


def fullres_shape_estimate(s: Sample) -> tuple[int, int]:
    """(height, width) of the full-res image implied by tissue_hires_image.png + scalefactor."""
    from PIL import Image, UnidentifiedImageError
    k = s.scalefactors()["tissue_hires_scalef"]  # identical across the run's spatial/ folders
    for d in (s.seg / "spatial", s.outs / "spatial", s.bins2um / "spatial"):
        try:  # some copies on disk are 0-byte placeholders
            w, h = Image.open(d / "tissue_hires_image.png").size
        except (FileNotFoundError, UnidentifiedImageError):
            continue
        return int(round(h / k)), int(round(w / k))
    raise FileNotFoundError(f"no readable tissue_hires_image.png under {s.outs}")
