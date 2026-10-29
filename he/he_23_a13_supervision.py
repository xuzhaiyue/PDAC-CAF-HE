"""Frozen A13: paired local versus counts-aggregated supervision, patient LOPO."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_name] = "2"

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import torch
from torch import nn
import yaml

ARMS = ("A", "B", "C1", "C2", "C3")
TARGETS = ("program", "pan", "relative")


def now():
    return dt.datetime.now().astimezone().isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_for(text, size=8):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:size], "little")


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def atomic_table(path, frame):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_checkpoint(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite committed checkpoint: {path}")
    tmp = path.with_suffix(".tmp.pt")
    torch.save(value, tmp)
    tmp.replace(path)


class Head(nn.Module):
    def __init__(self, dim=1536, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return nn.functional.softplus(self.net(x)) + 1e-6


def slide_loss(q, view, mu, sd):
    """The aggregated arm receives only depth weights and bag targets."""
    if view["kind"] == "A":
        assert set(view) == {"kind", "X", "weights", "target"}
        pred = torch.log1p((q * view["weights"][:, None]).sum(dim=0))
        return (((pred - mu) / sd - view["target"]) ** 2).mean()
    assert set(view) == {"kind", "X", "target"}
    return (((torch.log1p(q) - mu) / sd - view["target"]) ** 2).mean()


def safe_rho(truth, pred):
    if len(truth) < 3:
        return float("nan"), "fewer_than_3_units"
    if not np.isfinite(pred).all():
        raise RuntimeError("Nonfinite prediction on a technically valid unit")
    if np.ptp(truth) == 0:
        return float("nan"), "constant_truth"
    if np.ptp(pred) == 0:
        return float("nan"), "constant_prediction"
    value = float(spearmanr(truth.astype(np.float64), pred.astype(np.float64)).statistic)
    return (value, "") if np.isfinite(value) else (float("nan"), "nonfinite_rho")


def strict_mean(values):
    values = np.asarray(values, float)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else float("nan")


def paired_inference(differences, bootstrap, signs):
    d = np.asarray(differences, np.float64)
    if len(d) < 2:
        return {"delta_mean": float(d.mean()) if len(d) else None, "delta_median": None,
                "ci_low": None, "ci_high": None, "p_two_sided": None, "numeric_pass": False}
    delta = float(d.mean())
    ci = np.quantile(d[bootstrap].mean(axis=1), [.025, .975], method="linear")
    flipped = (signs * d[None, :]).mean(axis=1)
    p = float((1 + np.count_nonzero(np.abs(flipped) >= abs(delta))) / (len(signs) + 1))
    return {"delta_mean": delta, "delta_median": float(np.median(d)), "ci_low": float(ci[0]),
            "ci_high": float(ci[1]), "p_two_sided": p,
            "numeric_pass": bool(delta > 0 and ci[0] > 0 and p < .05)}


class Runner:
    def __init__(self, config_path):
        self.root = Path(os.environ["PROJECT_ROOT"]).resolve()
        self.config_path = Path(config_path).resolve()
        self.cfg = yaml.safe_load(self.config_path.read_text())
        assert self.cfg["status"].startswith("frozen_") and self.cfg["training_authorized"] is True
        assert digest(self.root / "config/fig1.yaml") == self.cfg["frozen_parent_config_sha256"]
        for path, expected in self.cfg["protected_sha256"].items():
            assert digest(self.root / path) == expected, f"Protected input changed before start: {path}"
        self.out = self.root / self.cfg["paths"]["output"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.config_sha = digest(self.config_path)
        self.code_sha = digest(__file__)
        self.meta = pd.read_csv(self.root / self.cfg["paths"]["metadata"], sep="\t",
            usecols=["sample", "patient", "source", "tissue", "held_out_group"])
        self.meta["main_scope"] = ~self.meta["sample"].isin(self.cfg["scope"]["reference_only"])
        assert len(self.meta) == 70 and self.meta["sample"].is_unique
        assert self.meta.main_scope.sum() == 68 and self.meta.patient.nunique() == 25
        assert (self.meta.patient == self.meta.held_out_group).all()
        self.groups = sorted(self.meta.patient.unique())
        self.lookup = self.meta.set_index("sample")
        self.features = {}
        self.image_masks = {}
        self.unit_masks = {}
        self.coverage = []
        self.events = self.out / "events.jsonl"
        self.start = time.time()
        self.device = torch.device("mps")
        if not torch.backends.mps.is_available():
            raise RuntimeError("Frozen MPS device unavailable; no automatic fallback")
        self.lock = open(self.out / "run.lock", "a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.event("PROCESS_STARTED", pid=os.getpid(), config_sha=self.config_sha, code_sha=self.code_sha)
        self.progress("PREPARING", completed_fits=self.completed_count())

    def event(self, event, **fields):
        value = {"time": now(), "event": event, **fields}
        with self.events.open("a") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def progress(self, status, **fields):
        atomic_json(self.out / "status.json", {"time": now(), "pid": os.getpid(), "status": status,
                    "total_fits": 125, "elapsed_seconds_process": time.time()-self.start, **fields})

    def completed_count(self):
        return len(list((self.out / "folds").glob("*/*/FIT_COMPLETE.json")))

    def pause(self, seconds=0):
        while os.getloadavg()[0] > self.cfg["load"]["max_load1"]:
            self.event("LOAD_PAUSE", load1=os.getloadavg()[0])
            time.sleep(5)
        if seconds:
            time.sleep(seconds)

    def feature_path(self, sid):
        paths = self.cfg["paths"]
        if sid in self.cfg["scope"]["own_hd"]:
            return self.root / paths["hd"] / f"{sid}.npz"
        return self.root / paths["features_128"] / f"{sid}.npy"

    def cache_path(self, sid):
        paths = self.cfg["paths"]
        folder = paths["hd"] if sid in self.cfg["scope"]["own_hd"] else paths["hest"] if sid.startswith("NCBI") else paths["geo"]
        return self.root / folder / f"{sid}.npz"

    def load_image_only(self, sid):
        """This prediction-side loader never accesses P/C/T or RNA masks."""
        if sid not in self.features:
            path = self.feature_path(sid)
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=True) as z:
                    x = z["features"].copy()
            else:
                x = np.array(np.load(path, mmap_mode="r"), copy=True)
            assert x.ndim == 2 and x.shape[1] == 1536
            self.features[sid] = x
        return self.features[sid]

    def prepare(self):
        """Data engineering checks precede all fitting, without viewing predictions."""
        prepared = self.out / "prepared_units"
        prepared.mkdir(exist_ok=True)
        (self.out / "permutations").mkdir(exist_ok=True)
        manifests = []
        prior = self.root / self.cfg["paths"]["preflight"]
        expected = pd.read_csv(prior / "input_manifest.tsv", sep="\t").drop_duplicates("path").set_index("path")
        for i, r in enumerate(self.meta.itertuples()):
            self.pause()
            sid = r.sample
            cache, feature = self.cache_path(sid), self.feature_path(sid)
            for p in dict.fromkeys([cache, feature]):
                h = digest(p)
                assert str(p) in expected.index and h == expected.loc[str(p), "sha256"], f"Changed input: {p}"
                st = p.stat()
                manifests.append({"sample": sid, "path": str(p), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h})
            X = self.load_image_only(sid)
            image_ok = np.isfinite(X).all(axis=1)
            with np.load(cache, allow_pickle=True) as z:
                counts = np.column_stack([z[k].astype(np.float64) for k in ["prog_umi", "pan_umi", "total_umi"]])
                if sid in self.cfg["scope"]["own_hd"]:
                    ids = np.array([f"{a}:{b}" for a, b in zip(z["ix"], z["iy"])])
                    coords = np.column_stack([z["ix"], z["iy"]]).astype(float) * 128
                    assert (z["tissue_frac"] >= .5).all()
                    tech, preservation = "Visium HD", "Unknown"
                else:
                    ids = z["barcode"].astype(str)
                    coords = z["coords"].astype(float) * float(z["pixel_size"])
                    assert int(z["genes_prog_present"]) == 9 and int(z["genes_pan_present"]) == 4
                    tech = str(z["st_technology"])
                    preservation = str(z["preservation"]) if "preservation" in z.files else "Unknown"
            if r.source == "GSE211895":
                preservation = "FFPE-probes"
            if preservation in ("", "nan", "None"):
                preservation = "Unknown"
            assert len(X) == len(counts) == len(ids) and len(np.unique(ids)) == len(ids)
            assert np.isfinite(counts).all() and (counts >= 0).all() and (counts == np.rint(counts)).all()
            assert (counts[:, 0]+counts[:, 1] <= counts[:, 2]).all()
            coord_ok = np.isfinite(coords).all(axis=1)
            image_ok &= coord_ok
            mask = image_ok & (counts[:, 2] > 0)
            assert mask.any(), f"No valid units: {sid}"
            audited = pd.read_csv(prior / "units" / f"{sid}.tsv.gz", sep="\t")
            assert np.array_equal(ids, audited.unit_id.astype(str))
            assert np.array_equal(counts, audited[["P", "C", "T"]].to_numpy())
            target = prepared / f"{sid}.npz"
            if not target.exists():
                np.savez_compressed(target, counts=counts, unit_id=ids, image_ok=image_ok, mask=mask,
                                    coords_um=coords, technology=tech, preservation=preservation)
            else:
                with np.load(target) as a:
                    assert np.array_equal(a["counts"], counts) and np.array_equal(a["mask"], mask)
            self.image_masks[sid], self.unit_masks[sid] = image_ok, mask
            c = counts[mask]
            local = 1e4*c[:, :2]/c[:, 2, None]
            truth = 1e4*c[:, :2].sum(0)/c[:, 2].sum()
            assert np.allclose(np.sum(local*(c[:, 2]/c[:, 2].sum())[:, None], 0), truth, rtol=1e-10, atol=1e-10)
            perms = {}
            for seed in self.cfg["permutation_seeds"]:
                index = np.random.default_rng(seed_for(f"{seed}:{sid}")).permutation(len(c))
                assert np.array_equal(c.sum(0), c[index].sum(0))
                perms[str(seed)] = index
            pf = self.out / "permutations" / f"{sid}.npz"
            if not pf.exists():
                np.savez_compressed(pf, **perms)
            else:
                with np.load(pf) as a:
                    for key, value in perms.items():
                        assert np.array_equal(a[key], value)
            reason = np.where(~image_ok, "nonfinite_image_or_coordinate", np.where(counts[:, 2] <= 0, "undefined_denominator", ""))
            mask_path = prepared / f"{sid}_mask.tsv.gz"
            if not mask_path.exists():
                pd.DataFrame({"unit_id": ids, "image_valid": image_ok, "main_unit_mask": mask,
                    "T_ge_100_mask": mask & (counts[:, 2] >= 100), "reason": reason}).to_csv(mask_path, sep="\t", index=False)
            self.coverage.append({"sample": sid, "patient": r.patient, "source": r.source, "tissue": r.tissue,
                "main_scope": bool(r.main_scope), "technology": tech, "preservation": preservation,
                "cached_units": len(X), "image_valid": int(image_ok.sum()), "T_positive": int(mask.sum()),
                "T_ge_100": int((mask & (counts[:, 2] >= 100)).sum()), "zero_T": int((counts[:, 2] == 0).sum())})
            self.event("SAMPLE_PREPARED", sample=sid, done=i+1, total=70)
        atomic_table(self.out / "scope_coverage.tsv", pd.DataFrame(self.coverage))
        atomic_table(self.out / "input_manifest.tsv", pd.DataFrame(manifests))
        atomic_table(self.out / "scope.tsv", self.meta)
        atomic_json(self.out / "PREPARED.json", {"time": now(), "samples": 70, "main_slides": 68,
                    "main_positive_units": sum(r["T_positive"] for r in self.coverage if r["main_scope"]),
                    "config_sha": self.config_sha})

    def fold_data(self, held, fd):
        train = self.meta[self.meta.main_scope & (self.meta.patient != held)]
        groups = sorted(train.patient.unique())
        assert len(groups) == 24 and held not in groups
        patient_sids = {p: sorted(train.loc[train.patient == p, "sample"]) for p in groups}
        scalers_path = fd / "scalers.npz"
        counts_by_sid, raw = {}, {}
        means, seconds, bags, weights = [], [], [], []
        for p in groups:
            for sid in patient_sids[p]:
                with np.load(self.out / "prepared_units" / f"{sid}.npz") as z:
                    m = z["mask"].copy()
                    c = z["counts"][m].copy()
                X = self.load_image_only(sid)[m]
                assert self.lookup.loc[sid, "patient"] != held
                counts_by_sid[sid] = c
                raw[sid] = X
                weight = 1/(24*len(patient_sids[p]))
                n, sm, sq = len(X), np.zeros(1536), np.zeros(1536)
                for i in range(0, n, 512):
                    a = X[i:i+512].astype(np.float64)
                    sm += a.sum(0)
                    sq += np.square(a).sum(0)
                means.append(sm/n)
                seconds.append(sq/n)
                bags.append(np.log1p(1e4*c[:, :2].sum(0)/c[:, 2].sum()))
                weights.append(weight)
        w = np.asarray(weights)
        fm = np.sum(np.asarray(means)*w[:, None], 0)
        fv = np.sum(np.asarray(seconds)*w[:, None], 0)-fm**2
        assert np.min(fv) >= -1e-8
        fs = np.maximum(np.sqrt(np.maximum(fv, 0)), 1e-6)
        ym = np.sum(np.asarray(bags)*w[:, None], 0)
        ys = np.sqrt(np.sum((np.asarray(bags)-ym)**2*w[:, None], 0))
        assert np.isfinite(ys).all() and (ys > 1e-8).all()
        if scalers_path.exists():
            with np.load(scalers_path) as a:
                for key, val in {"feature_mu": fm, "feature_sd": fs, "target_mu": ym, "target_sd": ys}.items():
                    assert np.array_equal(a[key], val)
        else:
            np.savez_compressed(scalers_path, feature_mu=fm, feature_sd=fs, target_mu=ym, target_sd=ys,
                                training_patients=np.array(groups), training_samples=np.array(list(raw)))
        Xnorm = {sid: ((x.astype(np.float32)-fm.astype(np.float32))/fs.astype(np.float32)).astype(np.float32)
                 for sid, x in raw.items()}
        views = {arm: {} for arm in ARMS}
        for sid, c in counts_by_sid.items():
            bag = np.log1p(1e4*c[:, :2].sum(0)/c[:, 2].sum())
            x = Xnorm[sid]
            views["A"][sid] = {"kind": "A", "X": x,
                "weights": torch.tensor(c[:, 2]/c[:, 2].sum(), dtype=torch.float32, device=self.device),
                "target": torch.tensor((bag-ym)/ys, dtype=torch.float32, device=self.device)}
            local = np.log1p(1e4*c[:, :2]/c[:, 2, None])
            views["B"][sid] = {"kind": "B", "X": x,
                "target": torch.tensor((local-ym)/ys, dtype=torch.float32, device=self.device)}
            with np.load(self.out / "permutations" / f"{sid}.npz") as a:
                for k, seed in enumerate(self.cfg["permutation_seeds"], 1):
                    perm = a[str(seed)]
                    views[f"C{k}"][sid] = {"kind": "C", "X": x,
                        "target": torch.tensor((local[perm]-ym)/ys, dtype=torch.float32, device=self.device)}
        rng = np.random.default_rng(seed_for("A13:order:"+held))
        order = [list(rng.permutation(groups)) for _ in range(self.cfg["training"]["rounds"])]
        order_path = fd / "patient_schedule.json"
        if order_path.exists():
            assert json.loads(order_path.read_text()) == order
        else:
            atomic_json(order_path, order)
        atomic_table(fd / "split.tsv", self.meta.assign(role=np.where(self.meta.patient == held, "test",
                     np.where(self.meta.main_scope, "train", "reference_excluded"))))
        return views, patient_sids, (fm, fs, ym, ys), order

    def train_fit(self, held, arm, initial, views, patient_sids, scalers, order, fd):
        dest = fd / arm
        dest.mkdir(exist_ok=True)
        complete = dest / "FIT_COMPLETE.json"
        initial_hash = state_hash(initial)
        if complete.exists():
            info = json.loads(complete.read_text())
            assert info["config_sha"] == self.config_sha and info["code_sha"] == self.code_sha
            assert info["initial_hash"] == initial_hash and info["updates"] == 480
            assert digest(dest / "round_020.pt") == info["final_checkpoint_sha"]
            return
        torch.manual_seed(seed_for("A13:init:"+held, 4))
        model = Head()
        model.load_state_dict(initial)
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), **self.cfg["training"]["optimizer"])
        mu = torch.tensor(scalers[2], dtype=torch.float32, device=self.device)
        sd = torch.tensor(scalers[3], dtype=torch.float32, device=self.device)
        checkpoints = sorted(dest.glob("round_*.pt"))
        done_round, updates, visits = 0, 0, 0
        if checkpoints:
            st = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
            assert st["config_sha"] == self.config_sha and st["code_sha"] == self.code_sha
            assert st["initial_hash"] == initial_hash
            model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            done_round, updates, visits = st["round"], st["updates"], st["unit_visits"]
            torch.set_rng_state(st["cpu_rng"])
            if st.get("mps_rng") is not None:
                torch.mps.set_rng_state(st["mps_rng"])
            self.event("FIT_RESUMED", held=held, arm=arm, round=done_round)
        else:
            save_checkpoint(dest / "round_000.pt", self.checkpoint(model, optimizer, 0, 0, 0, initial_hash))
        self.event("FIT_STARTED", held=held, arm=arm, initial_hash=initial_hash)
        expected_units = sum(len(v["X"]) for v in views.values())
        for round_idx in range(done_round, self.cfg["training"]["rounds"]):
            round_start, losses = time.time(), []
            model.train()
            for patient_idx, patient in enumerate(order[round_idx]):
                self.pause()
                optimizer.zero_grad(set_to_none=True)
                patient_loss = 0.
                sids = patient_sids[patient]
                for sid in sids:
                    view = views[sid]
                    X = view["X"]
                    chunks = []
                    for i in range(0, len(X), self.cfg["training"]["microbatch"]):
                        self.pause()
                        batch = torch.from_numpy(X[i:i+self.cfg["training"]["microbatch"]]).to(self.device)
                        chunks.append(model(batch))
                        torch.mps.synchronize()
                        time.sleep(self.cfg["load"]["microbatch_pause_s"])
                    q = torch.cat(chunks, dim=0)
                    loss = slide_loss(q, view, mu, sd)/len(sids)
                    if not bool(torch.isfinite(loss).item()):
                        raise RuntimeError(f"Nonfinite loss {held}/{arm}/{sid}")
                    patient_loss += float(loss.detach().cpu())
                    loss.backward()
                    visits += len(X)
                    del q, chunks, loss
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg["training"]["clip_grad_norm"], error_if_nonfinite=True)
                if not bool(torch.isfinite(norm).item()):
                    raise RuntimeError("Nonfinite gradient")
                optimizer.step()
                updates += 1
                losses.append(patient_loss)
                self.progress("TRAINING", held_patient=held, arm=arm, round=round_idx+1,
                    patient_in_round=patient_idx+1, optimizer_updates=updates, completed_fits=self.completed_count())
                time.sleep(self.cfg["load"]["patient_pause_s"])
            assert visits == (round_idx+1)*expected_units and updates == (round_idx+1)*24
            st = self.checkpoint(model, optimizer, round_idx+1, updates, visits, initial_hash)
            save_checkpoint(dest / f"round_{round_idx+1:03d}.pt", st)
            with (dest / "training_log.jsonl").open("a") as f:
                f.write(json.dumps({"time": now(), "round": round_idx+1, "updates": updates,
                    "unit_visits": visits, "mean_patient_train_loss": float(np.mean(losses)),
                    "seconds": time.time()-round_start})+"\n")
            self.event("ROUND_COMPLETE", held=held, arm=arm, round=round_idx+1,
                       updates=updates, seconds=time.time()-round_start)
        final_hash = state_hash(model.state_dict())
        assert final_hash != initial_hash and updates == 480
        atomic_json(complete, {"time": now(), "held": held, "arm": arm, "updates": updates, "unit_visits": visits,
            "initial_hash": initial_hash, "final_hash": final_hash, "final_checkpoint_sha": digest(dest / "round_020.pt"),
            "config_sha": self.config_sha, "code_sha": self.code_sha})
        self.event("FIT_COMPLETE", held=held, arm=arm, completed=self.completed_count(), total=125)
        del model, optimizer
        torch.mps.empty_cache()

    def checkpoint(self, model, optimizer, round_idx, updates, visits, initial_hash):
        return {"round": round_idx, "updates": updates, "unit_visits": visits, "initial_hash": initial_hash,
            "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
            "cpu_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state(),
            "config_sha": self.config_sha, "code_sha": self.code_sha}

    @torch.inference_mode()
    def predict(self, held, arm, scalers, fd):
        dest = fd / arm
        pred_dir = dest / "predictions"
        pred_dir.mkdir(exist_ok=True)
        marker = pred_dir / "PREDICTIONS_FROZEN.json"
        if marker.exists():
            info = json.loads(marker.read_text())
            for p, value in info["sha256"].items():
                assert digest(pred_dir / p) == value
            return
        model = Head().to(self.device)
        state = torch.load(dest / "round_020.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        fm, fs = scalers[:2]
        hashes = {}
        for sid in sorted(self.meta.loc[self.meta.patient == held, "sample"]):
            # Prediction has no counts, RNA weights, targets, or RNA-derived masks.
            X = self.load_image_only(sid)
            mask = np.isfinite(X).all(1)
            q = np.full((len(X), 2), np.nan, dtype=np.float32)
            rows = np.flatnonzero(mask)
            for i in range(0, len(rows), self.cfg["training"]["microbatch"]):
                self.pause()
                ix = rows[i:i+self.cfg["training"]["microbatch"]]
                xb = (X[ix].astype(np.float32)-fm.astype(np.float32))/fs.astype(np.float32)
                q[ix] = model(torch.from_numpy(xb).to(self.device)).cpu().numpy()
                time.sleep(self.cfg["load"]["microbatch_pause_s"])
            assert np.isfinite(q[mask]).all() and (q[mask] > 0).all()
            path = pred_dir / f"{sid}.npz"
            if path.exists():
                raise RuntimeError(f"Uncommitted prediction file exists: {path}; inspect before overwriting")
            np.savez_compressed(path, q=q, image_finite=mask, row_index=np.arange(len(X)))
            hashes[path.name] = digest(path)
        atomic_json(marker, {"time": now(), "held": held, "arm": arm, "input": "image_features_only",
            "sha256": hashes, "checkpoint_sha": digest(dest / "round_020.pt")})
        self.event("PREDICTIONS_FROZEN", held=held, arm=arm, n_slides=len(hashes))
        del model
        torch.mps.empty_cache()

    def evaluate_fold(self, held, fd):
        """RNA is opened only after all five predictions in this fold are committed."""
        markers = [fd / arm / "predictions" / "PREDICTIONS_FROZEN.json" for arm in ARMS]
        assert all(p.exists() for p in markers)
        metric_path = fd / "slide_metrics.tsv"
        if metric_path.exists():
            return
        self.event("RNA_EVALUATION_OPENED", held=held, frozen_prediction_manifests=[digest(p) for p in markers])
        rows = []
        for r in self.meta[self.meta.patient == held].itertuples():
            sid = r.sample
            with np.load(self.out / "prepared_units" / f"{sid}.npz") as z:
                counts, mask = z["counts"].copy(), z["mask"].copy()
            for arm in ARMS:
                with np.load(fd / arm / "predictions" / f"{sid}.npz") as z:
                    q = z["q"].astype(np.float64)
                    assert np.array_equal(z["row_index"], np.arange(len(counts)))
                for mask_name, mm in [("Tpositive", mask), ("Tge100", mask & (counts[:, 2] >= 100))]:
                    n = int(mm.sum())
                    truth_rate = 1e4*counts[mm, :2]/counts[mm, 2, None] if n else np.empty((0, 2))
                    truth_log, pred_log = np.log1p(truth_rate), np.log1p(q[mm])
                    if n and not np.isfinite(pred_log).all():
                        raise RuntimeError("Nonfinite model output on evaluation mask")
                    for h, target in enumerate(TARGETS):
                        true = truth_log[:, h] if h < 2 else truth_log[:, 0]-truth_log[:, 1]
                        pred = pred_log[:, h] if h < 2 else pred_log[:, 0]-pred_log[:, 1]
                        rho, reason = safe_rho(true, pred)
                        if n:
                            tg, pg = np.log1p(truth_rate.mean(0)), np.log1p(q[mm].mean(0))
                            agg_error = abs(pg[h]-tg[h]) if h < 2 else abs((pg[0]-pg[1])-(tg[0]-tg[1]))
                        else:
                            agg_error = np.nan
                        rows.append({"sample": sid, "patient": held, "source": r.source, "tissue": r.tissue,
                            "main_scope": bool(r.main_scope), "arm": arm, "mask": mask_name, "target": target,
                            "n_units": n, "rho": rho, "rho_reason": reason,
                            "mae_log": float(np.mean(abs(pred-true))) if n else np.nan,
                            "rmse_log": float(np.sqrt(np.mean((pred-true)**2))) if n else np.nan,
                            "equal_unit_rate_summary_abs_error": float(agg_error)})
        atomic_table(metric_path, pd.DataFrame(rows))
        self.event("FOLD_EVALUATED", held=held)

    def run(self):
        self.prepare()
        for fold_idx, held in enumerate(self.groups):
            fd = self.out / "folds" / held
            fd.mkdir(parents=True, exist_ok=True)
            self.progress("FOLD_SCALERS", held_patient=held, fold=fold_idx+1, completed_fits=self.completed_count())
            views, patient_sids, scalers, order = self.fold_data(held, fd)
            torch.manual_seed(seed_for("A13:init:"+held, 4))
            initial = cpu_tree(Head().state_dict())
            for arm in ARMS:
                self.train_fit(held, arm, initial, views[arm], patient_sids, scalers, order, fd)
                self.predict(held, arm, scalers, fd)
            self.evaluate_fold(held, fd)
            del views
            torch.mps.empty_cache()
        assert self.completed_count() == 125
        self.progress("SUMMARIZING", completed_fits=125)
        self.summarize()
        self.progress("COMPLETE", completed_fits=125, results="final_verdict.json")
        atomic_json(self.out / "COMPLETE.json", {"time": now(), "fits": 125, "config_sha": self.config_sha})
        self.event("RUN_COMPLETE", fits=125)

    def summarize(self):
        metrics = pd.concat([pd.read_csv(p, sep="\t") for p in sorted((self.out / "folds").glob("*/slide_metrics.tsv"))], ignore_index=True)
        assert len(metrics) == 70*5*2*3
        assert not metrics.duplicated(["sample", "arm", "mask", "target"]).any()
        atomic_table(self.out / "slide_metrics.tsv", metrics)
        patients = patient_metrics(metrics[metrics.main_scope])
        atomic_table(self.out / "patient_metrics.tsv", patients)
        primary = patients[(patients["target"] == "program") & (patients["mask"] == "Tpositive")]
        pivot = primary.pivot(index="patient", columns="arm", values="rho").reindex(self.groups)
        full = pivot[list(ARMS)].notna().all(axis=1)
        valid = pivot.loc[full]
        n = len(valid)
        rng = np.random.default_rng(self.cfg["statistics"]["bootstrap_seed"])
        bootstrap = rng.integers(0, n, size=(10000, n)) if n else np.empty((10000, 0), dtype=int)
        signs = np.random.default_rng(self.cfg["statistics"]["sign_seed"]).choice(np.array([-1, 1], np.int8), size=(99999, n))
        np.savez_compressed(self.out / "patient_resampling.npz", patient_ids=np.array(valid.index), bootstrap_indices=bootstrap, signs=signs)
        c = valid[["C1", "C2", "C3"]].mean(axis=1)
        differences = {"B-C": valid.B-c, "B-A": valid.B-valid.A}
        contrast_rows, first_pass = [], False
        for label, diff in differences.items():
            result = paired_inference(diff.to_numpy(), bootstrap, signs)
            confirmatory = label == "B-C" or first_pass
            passed = bool(confirmatory and n == 25 and result["numeric_pass"])
            contrast_rows.append({"contrast": label, "n_common_patients": n, "complete_25": n == 25,
                "confirmatory_status": confirmatory, "passed": passed, "patients_improved": int((diff > 0).sum()), **result})
            if label == "B-C":
                first_pass = passed
        atomic_table(self.out / "primary_contrasts.tsv", pd.DataFrame(contrast_rows))
        diff_table = pivot.copy()
        diff_table["Cmean"] = pivot[["C1", "C2", "C3"]].mean(axis=1, skipna=False)
        diff_table["B_minus_C"] = diff_table.B-diff_table.Cmean
        diff_table["B_minus_A"] = diff_table.B-diff_table.A
        diff_table["common_estimable"] = full
        atomic_table(self.out / "primary_patient_differences.tsv", diff_table.reset_index())
        summaries = secondary_summaries(metrics, pd.DataFrame(self.coverage), self.cfg)
        atomic_table(self.out / "secondary_strata.tsv", summaries)
        verdict = {"time": now(), "fits": 125, "common_estimable_patients": n,
            "complete_primary_cohort": n == 25, "hierarchy": contrast_rows,
            "both_primary_gates_pass": bool(all(x["passed"] for x in contrast_rows)),
            "interpretation_boundary": "Fixed models, budget and LOPO OOF predictions; no abundance-independent CAF state or clinical utility established.",
            "no_additional_experiments": True}
        atomic_json(self.out / "final_verdict.json", verdict)
        for path, expected in self.cfg["protected_sha256"].items():
            assert digest(self.root / path) == expected, f"Protected evidence changed: {path}"
        assert digest(self.root / "config/fig1.yaml") == self.cfg["frozen_parent_config_sha256"]
        atomic_json(self.out / "protected_verification.json", {"time": now(), "all_unchanged": True, "sha256": self.cfg["protected_sha256"]})
        write_figures(self.out, diff_table, pd.DataFrame(self.coverage))
        report = self.out / "RESULTS.md"
        lines = ["# A13完成结果", "", "125个预定拟合完成，规则与所有模型参数未按结果修改。", "",
                 f"主要共同可估患者：{n}/25。undefined保留在逐切片和患者表，不补0。", ""]
        for x in contrast_rows:
            lines.append(f"- {x['contrast']}：Δρ均值={x['delta_mean']}，95%CI=[{x['ci_low']}, {x['ci_high']}]，p={x['p_two_sided']}；确认性通过={x['passed']}。")
        lines.extend(["", "比较C采用三个单模型相关性均值，未集成预测。A仅训练损失使用局部RNA深度；测试局部预测不读RNA。", "",
            "主要结论以完整25患者及B−C→B−A层级为准；阴性/不可估不启动追加实验。来源差异、参考头及T≥100结果见secondary_strata.tsv。", "",
            "误差和患者CI基于固定LOPO预测，不包含全部初始化/重训练不确定性；不宣称纯CAF状态、因果机制、临床适用或完全未见来源泛化。"])
        report.write_text("\n".join(lines), encoding="utf-8")
        worklog = self.root / self.cfg["paths"]["completion_report"]
        if not worklog.exists():
            worklog.write_text("# A13执行完成\n\n"+"\n".join(lines[2:])+f"\n\n完整产物：{self.out}。\n", encoding="utf-8")


def patient_metrics(metrics):
    rows = []
    keys = ["patient", "mask", "target"]
    for key, group in metrics.groupby(keys, sort=True):
        expected_sids = set(group["sample"])
        arm_rows = []
        for arm in ARMS:
            g = group[group.arm == arm]
            assert set(g["sample"]) == expected_sids
            rr = {**dict(zip(keys, key)), "arm": arm, "n_slides": len(g),
                "rho": strict_mean(g.rho), "undefined_slides": int(g.rho.isna().sum()),
                "rho_reason": ";".join(f"{r.sample}:{r.rho_reason}" for r in g[g.rho.isna()].itertuples())}
            for col in ["mae_log", "rmse_log", "equal_unit_rate_summary_abs_error"]:
                rr[col] = strict_mean(g[col])
            rows.append(rr)
            arm_rows.append(rr)
        crows = [r for r in arm_rows if r["arm"].startswith("C")]
        rr = {**dict(zip(keys, key)), "arm": "Cmean", "n_slides": len(expected_sids),
              "rho": strict_mean([r["rho"] for r in crows]),
              "undefined_slides": max(r["undefined_slides"] for r in crows),
              "rho_reason": ";".join(r["rho_reason"] for r in crows if r["rho_reason"])}
        for col in ["mae_log", "rmse_log", "equal_unit_rate_summary_abs_error"]:
            rr[col] = strict_mean([r[col] for r in crows])
        rows.append(rr)
    return pd.DataFrame(rows)


def secondary_summaries(metrics, coverage, cfg):
    main = metrics[metrics.main_scope]
    strata = {"main_all": main,
        "primary_related": main[~main.tissue.str.contains("metastasis")],
        "metastasis_all": main[main.tissue.str.contains("metastasis")],
        "normal_reference": metrics[metrics["sample"] == "GSM9647222"],
        "adjacent_reference": metrics[metrics["sample"] == "GSM9647219"]}
    for tissue in ["Liver metastasis", "Lung metastasis", "Peritoneal metastasis"]:
        strata["tissue:"+tissue] = main[main.tissue == tissue]
    for source in sorted(coverage.source.unique()):
        strata["source:"+source] = main[main.source == source]
    for col in ["technology", "preservation"]:
        for value in sorted(coverage[col].unique()):
            ids = coverage.loc[coverage[col] == value, "sample"]
            strata[col+":"+value] = main[main["sample"].isin(ids)]
    rows = []
    for name, subset in strata.items():
        if subset.empty:
            continue
        patient = patient_metrics(subset)
        for key, g in patient.groupby(["arm", "mask", "target"]):
            for metric in ["rho", "mae_log", "rmse_log", "equal_unit_rate_summary_abs_error"]:
                valid = g[np.isfinite(g[metric])]
                values = valid[metric].to_numpy()
                ci = [np.nan, np.nan]
                if len(values) >= 3:
                    rng = np.random.default_rng(cfg["statistics"]["bootstrap_seed"])
                    indices = rng.integers(0, len(values), (10000, len(values)))
                    ci = np.quantile(values[indices].mean(1), [.025, .975], method="linear")
                rows.append({"stratum": name, "arm": key[0], "mask": key[1], "target": key[2], "metric": metric,
                    "n_patients_registered": len(g), "n_patients_estimable": len(values),
                    "mean": float(values.mean()) if len(values) else np.nan, "ci_low": ci[0], "ci_high": ci[1],
                    "minimum": float(values.min()) if len(values) else np.nan, "maximum": float(values.max()) if len(values) else np.nan})
        for key, g in patient.groupby(["mask", "target"]):
            for metric in ["rho", "mae_log", "rmse_log", "equal_unit_rate_summary_abs_error"]:
                pv = g.pivot(index="patient", columns="arm", values=metric)
                common = pv[list(ARMS)].notna().all(axis=1)
                pv = pv.loc[common]
                for contrast, control in [("B_minus_C", "Cmean"), ("B_minus_A", "A")]:
                    values = (pv.B-pv[control]).to_numpy()
                    ci = [np.nan, np.nan]
                    if len(values) >= 3:
                        rng = np.random.default_rng(cfg["statistics"]["bootstrap_seed"])
                        indices = rng.integers(0, len(values), (10000, len(values)))
                        ci = np.quantile(values[indices].mean(1), [.025, .975], method="linear")
                    rows.append({"stratum": name, "arm": contrast, "mask": key[0], "target": key[1],
                        "metric": metric+"_paired_difference", "n_patients_registered": len(common),
                        "n_patients_estimable": len(values), "mean": float(values.mean()) if len(values) else np.nan,
                        "ci_low": ci[0], "ci_high": ci[1], "minimum": float(values.min()) if len(values) else np.nan,
                        "maximum": float(values.max()) if len(values) else np.nan})
    return pd.DataFrame(rows)


def write_figures(out, differences, coverage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
                        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for _, r in differences.iterrows():
        axes[0].plot([0, 1, 2], [r.Cmean, r.A, r.B], color="#5D7B91", alpha=.45, marker="o", ms=3, lw=.7)
    axes[0].set(xticks=[0, 1, 2], xticklabels=["C: mean of 3 rho", "A: aggregated", "B: local"], ylabel="Patient mean local programme Spearman rho",
                title="A13: all registered patient outcomes")
    axes[0].text(.02, .02, f"Common estimable: {int(differences.common_estimable.sum())}/25\nUndefined values remain missing", transform=axes[0].transAxes, fontsize=8)
    g = coverage.groupby("source")[["cached_units", "T_positive", "T_ge_100"]].sum()
    x = np.arange(len(g))
    axes[1].bar(x-.16, g.T_positive/g.cached_units, width=.32, label="T>0 + QC", color="#5E8C83")
    axes[1].bar(x+.16, g.T_ge_100/g.cached_units, width=.32, label="T>=100 + QC", color="#A8BEB5")
    axes[1].set(xticks=x, xticklabels=g.index, ylim=(0, 1.05), ylabel="Retained units / paired cached footprint", title="Coverage, all 70 registered slides")
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].legend(frameon=False)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(out / f"A13_patient_effects_and_coverage.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        os.nice(19-os.getpriority(os.PRIO_PROCESS, 0))
    runner = None
    try:
        runner = Runner(args.config)
        runner.run()
    except Exception as exc:
        if runner is not None:
            runner.event("RUN_FAILED", error=repr(exc), traceback=traceback.format_exc())
            runner.progress("FAILED", error=repr(exc), completed_fits=runner.completed_count())
            atomic_json(runner.out / f"FAILURE_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        {"time": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
