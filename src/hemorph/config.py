"""Config loading and sample registry."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "aim1.yaml"


@dataclass
class Sample:
    sample_id: str
    patient_id: str
    outs: Path
    he_image: Path | None

    @property
    def seg(self) -> Path:
        return self.outs / "segmented_outputs"

    @property
    def bins2um(self) -> Path:
        return self.outs / "binned_outputs" / "square_002um"

    def scalefactors(self) -> dict:
        for p in (self.seg / "spatial/scalefactors_json.json",
                  self.bins2um / "spatial/scalefactors_json.json",
                  self.outs / "spatial/scalefactors_json.json"):
            if p.exists() and p.stat().st_size > 0:  # skip 0-byte placeholders
                return json.loads(p.read_text())
        raise FileNotFoundError(f"no scalefactors_json.json under {self.outs}")

    @property
    def mpp(self) -> float:
        """µm per pixel of the full-res microscope image given to spaceranger --image."""
        return float(self.scalefactors()["microns_per_pixel"])


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_path"] = str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    """Hash of the frozen analysis plan (analysis + gating + programs)."""
    programs = Path(cfg["paths"]["gene_programs"]).read_text()
    payload = json.dumps({"analysis": cfg["analysis"], "gating": cfg["gating"],
                          "image": cfg["image"], "programs": programs}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def samples(cfg: dict) -> list[Sample]:
    root = Path(cfg["paths"]["spaceranger_root"])
    he_dir = Path(cfg["paths"]["he_dir"]) if cfg["paths"].get("he_dir") else None
    out = []
    for s in cfg["samples"]:
        he = s.get("he_image")
        if he and not Path(he).is_absolute():  # relative names resolve against he_dir
            he = he_dir / he if he_dir else None
        out.append(Sample(s["sample_id"], s["patient_id"], root / s["outs"],
                          Path(he) if he else None))
    return out


def sample_dir(cfg: dict, sample_id: str) -> Path:
    d = Path(cfg["paths"]["work_dir"]) / sample_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_programs(cfg: dict, var_names=None) -> dict[str, list[str]]:
    progs = yaml.safe_load(Path(cfg["paths"]["gene_programs"]).read_text())
    if var_names is None:
        return progs
    present = set(var_names)
    return {k: [g for g in v if g in present] for k, v in progs.items()}
