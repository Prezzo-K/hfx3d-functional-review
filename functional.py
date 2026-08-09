"""
functional.py — per-instance functional-attribute data, colouring, and the
human-in-the-loop review store for the HFX3D functional QA workflow.

Three pieces:
  • FunctionalData   read-only load of a pipeline `functional_labels/*.h5`
                     (per-instance P / G / C / conf / vector matrices), plus
                     point-colour helpers keyed by per-point instance id.
  • ReviewStore      the editable human layer: per-instance human vector,
                     reviewed flag, instance flag, note. Saved as a JSON
                     sidecar; the pipeline output is never mutated.
  • export_reviewed  merge FunctionalData + ReviewStore into a reviewed .h5
                     that mirrors the pipeline schema plus human fields.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

INSTANCE_FLAGS = ["", "bad_segmentation", "wrong_class", "other"]

_ORPHAN = np.array([0.12, 0.12, 0.13], np.float32)   # point has no instance
_NA = np.array([0.20, 0.20, 0.22], np.float32)       # attribute not applicable


def _decode(v):
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


def attr_hue(j: int, n: int) -> np.ndarray:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((j / max(n, 1)) % 1.0, 0.65, 0.92)
    return np.array([r, g, b], np.float32)


# blue → cyan → green → yellow → red confidence ramp (vectorised)
_RAMP = np.array([
    [0.05, 0.10, 0.35], [0.13, 0.55, 0.85], [0.55, 0.85, 0.35],
    [0.95, 0.85, 0.15], [0.85, 0.15, 0.12],
], np.float32)


def ramp(t: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, np.float32), 0, 1)
    x = t * (len(_RAMP) - 1)
    i = np.floor(x).astype(int)
    i = np.clip(i, 0, len(_RAMP) - 2)
    f = (x - i)[:, None]
    return (_RAMP[i] * (1 - f) + _RAMP[i + 1] * f).astype(np.float32)


class FunctionalData:
    """Read-only per-instance functional attributes for one building."""

    def __init__(self, h5_path: Path):
        self.path = Path(h5_path)
        with h5py.File(self.path, "r") as f:
            self.attribute_names = [_decode(a) for a in f["metadata/attribute_names"][:]]
            g = f["instances"]
            self.ids = g["instance_id"][:].astype(np.int64)
            self.semantic_id = g["semantic_id"][:].astype(int)
            self.semantic_class = [_decode(c) for c in g["semantic_class"][:]]
            self.purity = g["semantic_purity"][:].astype(np.float32)
            self.point_count = g["point_count"][:].astype(np.int64)
            self.applicable = g["applicable_mask"][:].astype(np.uint8)
            self.P = g["semantic_prior_score"][:].astype(np.float32)
            self.G = g["geometry_score"][:].astype(np.float32)
            self.C = g["context_score"][:].astype(np.float32)
            self.conf = g["functional_attribute_confidence"][:].astype(np.float32)
            self.vec = g["functional_attribute_vector"][:].astype(np.uint8)

        self.n_attr = len(self.attribute_names)
        self._max_id = int(self.ids.max()) if self.ids.size else -1
        self._id_map = np.full(self._max_id + 2, -1, np.int64)
        self._id_map[self.ids] = np.arange(self.ids.shape[0])

    # ── lookups ───────────────────────────────────────────────────────────
    def row_of(self, inst_id: int) -> Optional[int]:
        if 0 <= inst_id <= self._max_id:
            r = int(self._id_map[inst_id])
            return r if r >= 0 else None
        return None

    def _rows(self, inst_ids: np.ndarray) -> np.ndarray:
        out = np.full(inst_ids.shape, -1, np.int64)
        v = (inst_ids >= 0) & (inst_ids <= self._max_id)
        out[v] = self._id_map[inst_ids[v]]
        return out

    def primary(self, row: int) -> int:
        """Dominant ENABLED attribute (argmax conf among vec==1), else -1."""
        on = self.vec[row].astype(bool)
        if not on.any():
            return -1
        c = np.where(on, self.conf[row], -1.0)
        return int(c.argmax())

    # ── point colouring (M = number of displayed points) ────────────────────
    def colors_primary(self, inst_ids: np.ndarray) -> np.ndarray:
        rows = self._rows(inst_ids)
        cols = np.tile(_ORPHAN, (inst_ids.shape[0], 1))
        v = rows >= 0
        rv = rows[v]
        cc = np.empty((rv.shape[0], 3), np.float32)
        cc[:] = _NA  # instances with no enabled attribute
        for k, r in enumerate(rv):
            p = self.primary(r)
            if p >= 0:
                cc[k] = attr_hue(p, self.n_attr)
        cols[v] = cc
        return cols

    def colors_attr(self, inst_ids: np.ndarray, j: int) -> np.ndarray:
        rows = self._rows(inst_ids)
        cols = np.tile(_ORPHAN, (inst_ids.shape[0], 1))
        v = rows >= 0
        rv = rows[v]
        appl = self.applicable[rv, j].astype(bool)
        cc = np.tile(_NA, (rv.shape[0], 1))
        if appl.any():
            cc[appl] = ramp(self.conf[rv[appl], j])
        cols[v] = cc
        return cols


class ReviewStore:
    """Editable human review layer, persisted as a JSON sidecar.

    Records are keyed by integer instance id. A record is created lazily from
    the pipeline suggestion the first time an instance is touched.
    """

    def __init__(self, path: Path, func: FunctionalData,
                 building: str, split: str, reviewer: str = ""):
        self.path = Path(path)
        self.func = func
        self.building = building
        self.split = split
        self.reviewer = reviewer
        self.dirty = False
        self.records: dict[int, dict] = {}
        if self.path.exists():
            self._load()

    # ── record access ───────────────────────────────────────────────────────
    def _pipeline_vec(self, iid: int) -> list[int]:
        r = self.func.row_of(iid)
        return self.func.vec[r].astype(int).tolist() if r is not None else [0] * self.func.n_attr

    def _default(self, iid: int) -> dict:
        return {"status": "unreviewed", "vector_human": self._pipeline_vec(iid),
                "instance_flag": "", "note": ""}

    def get(self, iid: int) -> dict:
        iid = int(iid)
        if iid not in self.records:
            self.records[iid] = self._default(iid)
        return self.records[iid]

    def changed_attrs(self, iid: int) -> list[int]:
        rec = self.get(iid)
        base = self._pipeline_vec(iid)
        return [j for j in range(self.func.n_attr) if rec["vector_human"][j] != base[j]]

    # ── edits (all set dirty) ─────────────────────────────────────────────
    def set_attr(self, iid: int, j: int, val: int):
        self.get(iid)["vector_human"][j] = int(bool(val)); self.dirty = True

    def set_all(self, iid: int, val: int):
        self.get(iid)["vector_human"] = [int(bool(val))] * self.func.n_attr; self.dirty = True

    def reset_to_pipeline(self, iid: int):
        self.get(iid)["vector_human"] = self._pipeline_vec(iid); self.dirty = True

    def set_status(self, iid: int, reviewed: bool):
        self.get(iid)["status"] = "reviewed" if reviewed else "unreviewed"; self.dirty = True

    def set_flag(self, iid: int, flag: str):
        self.get(iid)["instance_flag"] = flag if flag in INSTANCE_FLAGS else ""; self.dirty = True

    def set_note(self, iid: int, note: str):
        self.get(iid)["note"] = note; self.dirty = True

    # ── counts ────────────────────────────────────────────────────────────
    def n_reviewed(self) -> int:
        return sum(1 for r in self.records.values() if r["status"] == "reviewed")

    def n_total(self) -> int:
        return int(self.func.ids.shape[0])

    def is_reviewed(self, iid: int) -> bool:
        r = self.records.get(int(iid))
        return bool(r and r["status"] == "reviewed")

    # ── persistence ─────────────────────────────────────────────────────────
    def _load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.reviewer = data.get("reviewer", self.reviewer)
        for k, v in data.get("instances", {}).items():
            self.records[int(k)] = {
                "status": v.get("status", "unreviewed"),
                "vector_human": [int(x) for x in v["vector_human"]],
                "instance_flag": v.get("instance_flag", ""),
                "note": v.get("note", ""),
            }
        self.dirty = False

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "building": self.building,
            "split": self.split,
            "reviewer": self.reviewer,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "attribute_names": self.func.attribute_names,
            "source_functional_h5": str(self.func.path),
            "instances": {
                str(iid): {
                    "status": r["status"],
                    "vector_human": r["vector_human"],
                    "changed": self.changed_attrs(iid),
                    "instance_flag": r["instance_flag"],
                    "note": r["note"],
                }
                for iid, r in sorted(self.records.items())
            },
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.dirty = False
        return self.path


def export_reviewed(func: FunctionalData, store: ReviewStore, out_h5: Path) -> Path:
    """Write a reviewed .h5 mirroring the pipeline schema + human fields.

    Row order follows `func.ids`. Pipeline matrices are preserved for
    provenance; human decisions are added alongside.
    """
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    n = func.ids.shape[0]

    human = np.zeros((n, func.n_attr), np.uint8)
    status = np.empty(n, dtype="S12")
    flag = np.empty(n, dtype="S20")
    note = np.empty(n, dtype=h5py.string_dtype())
    for k, iid in enumerate(func.ids.tolist()):
        rec = store.records.get(int(iid))
        if rec is None:
            human[k] = func.vec[k]
            status[k] = b"unreviewed"; flag[k] = b""; note[k] = ""
        else:
            human[k] = np.array(rec["vector_human"], np.uint8)
            status[k] = rec["status"].encode(); flag[k] = rec["instance_flag"].encode()
            note[k] = rec["note"]

    with h5py.File(out_h5, "w") as f:
        m = f.create_group("metadata")
        m.create_dataset("attribute_names",
                         data=np.array(func.attribute_names, dtype=h5py.string_dtype()))
        m.create_dataset("attribute_count", data=np.int32(func.n_attr))
        m.create_dataset("reviewer", data=store.reviewer)
        m.create_dataset("source_functional_h5", data=str(func.path))
        m.create_dataset("exported_at", data=datetime.now().isoformat(timespec="seconds"))
        m.create_dataset("n_reviewed", data=np.int32(store.n_reviewed()))

        g = f.create_group("instances")
        g.create_dataset("instance_id", data=func.ids)
        g.create_dataset("semantic_id", data=func.semantic_id.astype(np.int32))
        g.create_dataset("semantic_class",
                         data=np.array(func.semantic_class, dtype=h5py.string_dtype()))
        g.create_dataset("point_count", data=func.point_count)
        g.create_dataset("applicable_mask", data=func.applicable)
        g.create_dataset("semantic_prior_score", data=func.P)
        g.create_dataset("geometry_score", data=func.G)
        g.create_dataset("context_score", data=func.C)
        g.create_dataset("functional_attribute_confidence", data=func.conf)
        g.create_dataset("functional_attribute_vector", data=func.vec)          # pipeline
        g.create_dataset("functional_attribute_vector_human", data=human)       # human
        g.create_dataset("review_status", data=status)
        g.create_dataset("instance_flag", data=flag)
        g.create_dataset("review_note", data=note)
    return out_h5


# ── locating the functional h5 for a loaded cloud ────────────────────────────

def building_stem(cloud_path: Path) -> str:
    """`HFX_BLD001_ZEB_CLEAN_instances_vis.ply` → `HFX_BLD001_ZEB_CLEAN`."""
    stem = Path(cloud_path).stem
    for suffix in ("_instances_vis", "_instances", "_vis"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def find_functional_h5(cloud_path: Path, func_root: Path) -> Optional[Path]:
    """Locate the pipeline functional_labels .h5 matching a loaded cloud."""
    func_root = Path(func_root)
    if not func_root.is_dir():
        return None
    stem = building_stem(cloud_path)
    split = Path(cloud_path).parent.name  # train/val/test if present
    cand = func_root / split / f"{stem}.h5"
    if cand.exists():
        return cand
    hits = list(func_root.rglob(f"{stem}.h5"))
    return hits[0] if hits else None
