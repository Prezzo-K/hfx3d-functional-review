#!/usr/bin/env python3
"""Editable per-point segmentation state layered on top of a read-only Bundle.

v1 of the review app was purely per-INSTANCE (attribute vectors). v2 lets the
reviewer correct the segmentation itself — reclass an instance, split one into
parts, or merge several — which is inherently per-POINT. This module holds that
mutable state and keeps it independent of Qt so the data logic can be tested
headless.

Design
------
The bundle's ``points``/``offsets`` (and ``src_index`` -> source-cloud rows) are
immutable. On top of them we keep, per *current* instance id:
  - ``rows[id]``  int32 array of points.npy rows that belong to it now
  - ``sem[id]``   its semantic class id
  - ``order``     the display order of ids (new split children slot in after
                  their parent; merged-away ids disappear)
Membership is the single source of truth; bbox / centroid / count / a per-point
label array are all derived from it on demand.

Every edit is logged (human-readable op) and pushed onto an undo stack that
restores exactly the ids it touched. purity cannot be honestly recomputed after
an edit (needs per-point fine labels we don't carry) so edited instances report
purity NaN rather than a fabricated value.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEM_NAMES = ["wall", "window", "door", "balcony", "vegetation", "stairs",
             "terrain", "roof", "blinds", "other", "column", "arch"]

# class -> attributes the ontology can ever assign to it (everything else is
# N/A for that class). Derived from pipeline/ontology_rules.yaml v2.0.
CLASS_APPLICABLE = {
    "wall": {"load_bearing", "thermal_envelope", "vegetation_support"},
    "window": {"thermal_envelope", "operable", "solar_shading", "ventilation", "natural_lighting"},
    "door": {"access", "operable", "thermal_envelope"},
    "balcony": {"load_bearing", "access", "vegetation_support", "fall_protection"},
    "vegetation": {"aesthetic"},
    "stairs": {"circulation", "fall_protection"},
    "terrain": {"circulation"},
    "roof": {"thermal_envelope", "load_bearing", "drainage", "fall_protection"},
    "blinds": {"solar_shading", "operable", "privacy_screening"},
    "other": set(),
    "column": {"load_bearing", "aesthetic"},
    "arch": {"load_bearing", "aesthetic", "illumination", "surveillance"},
}


def sem_name(sid: int) -> str:
    if sid < 0:
        return "unsegmented"                     # background / leftover points
    return SEM_NAMES[sid] if sid < len(SEM_NAMES) else "?"


class EditModel:
    def __init__(self, bundle):
        self.b = bundle
        self.pts = np.asarray(bundle.points)            # (N,3) float32
        off = bundle.offsets
        self.rows = {}
        self.sem = {}
        self.order = []
        for k, iid in enumerate(bundle.ids):
            iid = int(iid)
            self.rows[iid] = np.arange(off[k], off[k + 1], dtype=np.int32)
            self.sem[iid] = int(bundle.sem[k])
            self.order.append(iid)
        self.next_id = int(bundle.max_id) + 1
        self.log = []                                   # list of dicts (op records)
        self._undo = []                                 # list of restore snapshots
        self.edited = set()                             # ids whose geometry/class changed

    # ── queries ──────────────────────────────────────────────────────────
    def count(self, iid):
        return int(len(self.rows[iid]))

    def stats(self, iid):
        seg = self.pts[self.rows[iid]]
        bbox = np.concatenate([seg.min(0), seg.max(0)]).astype(np.float32)
        return int(len(seg)), bbox, seg.mean(0).astype(np.float32)

    def cls(self, iid):
        return sem_name(self.sem[iid])

    def applicable_mask(self, iid, attr_names):
        appl = CLASS_APPLICABLE.get(self.cls(iid), set())
        return np.array([nm in appl for nm in attr_names], bool)

    # ── undo ─────────────────────────────────────────────────────────────
    def _snapshot(self, ids):
        """Capture just enough to reverse an op touching `ids` (+ order/next_id)."""
        return {
            "order": list(self.order),
            "next_id": self.next_id,
            "rows": {i: (self.rows[i] if i in self.rows else None) for i in ids},
            "sem": {i: (self.sem[i] if i in self.sem else None) for i in ids},
            "log_len": len(self.log),
            "edited": set(self.edited),
        }

    def _restore(self, snap):
        self.order = snap["order"]
        self.next_id = snap["next_id"]
        for i, r in snap["rows"].items():
            if r is None:
                self.rows.pop(i, None)
            else:
                self.rows[i] = r
        for i, s in snap["sem"].items():
            if s is None:
                self.sem.pop(i, None)
            else:
                self.sem[i] = s
        self.edited = snap["edited"]
        del self.log[snap["log_len"]:]

    def can_undo(self):
        return bool(self._undo)

    def undo(self):
        if not self._undo:
            return None
        snap = self._undo.pop()
        undone = self.log[snap["log_len"]:]
        self._restore(snap)
        return undone[-1] if undone else None

    # ── edits ────────────────────────────────────────────────────────────
    def reclass(self, iid, new_sem):
        """Change an instance's semantic class. Membership unchanged."""
        new_sem = int(new_sem)
        if self.sem[iid] == new_sem:
            return
        self._undo.append(self._snapshot([iid]))
        old = self.sem[iid]
        self.sem[iid] = new_sem
        self.edited.add(iid)
        self.log.append({"op": "reclass", "id": int(iid),
                         "from": sem_name(old), "to": sem_name(new_sem)})

    def merge(self, ids):
        """Merge instances into one. Survivor id = the one with the most points
        (dominant); its class is kept. Returns the survivor id."""
        ids = [int(i) for i in ids if i in self.rows]
        if len(ids) < 2:
            return ids[0] if ids else None
        # survivor = dominant by point count, but never the background (-1):
        # merging unsegmented points into an object should keep the object.
        pool = [i for i in ids if i >= 0] or ids
        dst = max(pool, key=lambda i: len(self.rows[i]))
        gone = [i for i in ids if i != dst]
        self._undo.append(self._snapshot(ids))
        merged = np.concatenate([self.rows[i] for i in ids])
        merged.sort()
        self.rows[dst] = merged
        for i in gone:
            self.rows.pop(i); self.sem.pop(i)
            self.order.remove(i)
            self.edited.discard(i)
        self.edited.add(dst)
        self.log.append({"op": "merge", "into": dst, "from": gone,
                         "class": self.cls(dst)})
        return dst

    def split(self, iid, sel_rows, new_sem=None):
        """Move `sel_rows` (points.npy rows, a subset of iid) into a brand-new
        instance. Returns the new child id (or None if the selection is empty or
        everything / nothing would move)."""
        parent = self.rows[iid]
        sel = np.intersect1d(np.asarray(sel_rows, np.int32), parent)
        if len(sel) == 0 or len(sel) == len(parent):
            return None                                  # nothing to do / would empty parent
        self._undo.append(self._snapshot([iid, self.next_id]))
        child = self.next_id
        self.next_id += 1
        keep = np.setdiff1d(parent, sel, assume_unique=False)
        self.rows[iid] = keep.astype(np.int32)
        self.rows[child] = sel.astype(np.int32)
        self.sem[child] = int(new_sem) if new_sem is not None else self.sem[iid]
        self.order.insert(self.order.index(iid) + 1, child)
        self.edited.add(iid); self.edited.add(child)
        self.log.append({"op": "split", "id": int(iid), "child": int(child),
                         "moved": int(len(sel)), "class": sem_name(self.sem[child])})
        return child

    # ── export ───────────────────────────────────────────────────────────
    def per_point_labels(self):
        """Corrected (instance_id, semantic_id) for every source-cloud row that
        this bundle covers. Returns (src_rows, inst_ids, sem_ids) — apply to the
        source LAZ by index. Points not covered here (background) are left as-is."""
        src = np.asarray(self.b.src_index)
        n = len(self.pts)
        inst_out = np.empty(n, np.int64)
        sem_out = np.empty(n, np.int64)
        for iid in self.order:
            r = self.rows[iid]
            inst_out[r] = iid
            sem_out[r] = self.sem[iid]
        return src, inst_out, sem_out

    def write_edit_log(self, path, building, reviewer):
        payload = {"building": building, "reviewer": reviewer,
                   "n_edits": len(self.log), "edits": self.log}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def is_dirty(self):
        return bool(self.edited) or bool(self.log)

    # ── persistence (compact diff sidecar) ───────────────────────────────
    def _pt_inst(self):
        """Per points.npy row -> current instance id."""
        pi = np.empty(len(self.pts), np.int64)
        for iid in self.order:
            pi[self.rows[iid]] = iid
        return pi

    def save_edits(self, path):
        """Write only what changed vs the original segmentation (compact)."""
        off = self.b.offsets
        orig = np.empty(len(self.pts), np.int64)
        for k, iid in enumerate(self.b.ids):
            orig[off[k]:off[k + 1]] = int(iid)
        cur = self._pt_inst()
        diff = np.nonzero(cur != orig)[0].astype(np.int32)
        ids = np.array(self.order, np.int64)
        sem = np.array([self.sem[i] for i in self.order], np.int64)
        np.savez(path, order=ids, sem=sem,
                 diff_rows=diff, diff_inst=cur[diff].astype(np.int64),
                 edited=np.array(sorted(self.edited), np.int64),
                 next_id=np.int64(self.next_id))
        return path

    def load_edits(self, path):
        """Restore segmentation state written by save_edits (in place)."""
        d = np.load(path)
        off = self.b.offsets
        cur = np.empty(len(self.pts), np.int64)
        for k, iid in enumerate(self.b.ids):
            cur[off[k]:off[k + 1]] = int(iid)
        cur[d["diff_rows"]] = d["diff_inst"]
        # regroup rows by current instance id
        srt = np.argsort(cur, kind="stable")
        cs = cur[srt]
        uids, starts = np.unique(cs, return_index=True)
        starts = list(starts) + [len(cs)]
        self.rows = {int(u): srt[starts[i]:starts[i + 1]].astype(np.int32)
                     for i, u in enumerate(uids)}
        self.sem = {int(i): int(s) for i, s in zip(d["order"], d["sem"])}
        self.order = [int(i) for i in d["order"]]
        self.edited = set(int(i) for i in d["edited"])
        self.next_id = int(d["next_id"])
        self.log = []                    # provenance log lives in the .review.json
        self._undo = []

    def export_corrected_laz(self, src_laz, out_path):
        """Bake corrected per-point instance_id/semantic_id into a copy of the
        source cloud. Background / uncovered points are left untouched."""
        import laspy
        las = laspy.read(str(src_laz))
        src, inst, sem = self.per_point_labels()
        inst_arr = np.asarray(las["instance_id"]).copy()
        sem_arr = np.asarray(las["semantic_id"]).copy()
        inst_arr[src] = inst.astype(inst_arr.dtype)
        sem_arr[src] = sem.astype(sem_arr.dtype)
        las["instance_id"] = inst_arr
        las["semantic_id"] = sem_arr
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        las.write(str(out_path))
        return out_path
