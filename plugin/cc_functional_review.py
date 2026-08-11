#!/usr/bin/env python3
"""In-CloudCompare functional-attribute review panel (CloudCompare Python Runtime).

Runs inside CloudCompare (pycc). Reads a review LAZ built by build_review_cloud.py
--with-conf (carries `instance_id`, `semantic_id`, `purity`, `val_<attr>`,
`conf_<attr>` scalar fields) — needs only NumPy for that part, no h5py.

Team workflow: browse the instance list on the left → the picked instance is
highlighted in the 3D view and loaded into the editor on the right → tick the
attributes → "Confirm & Next". "Reviewed" is tracked automatically (any edit or
Confirm marks it), purely so the team can see coverage. "Save Review" writes
BOTH the review.json AND a reviewed .h5 in one action — nothing else to run
afterwards. Writing the .h5 needs h5py importable from CloudCompare's Python
(pip install h5py into that environment); if it isn't available, Save still
writes the json and tells you so instead of failing silently.

Run from the CloudCompare Python console:
    import sys; sys.path.append(r"C:\\path\\to\\hfx3d-functional-review\\plugin")
    import cc_functional_review as R; R.main()
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Set HFX3D_PROFILE=1 to print per-step timings to the CloudCompare console,
# to see where plugin-open / instance-selection time actually goes.
_PROFILE = bool(os.environ.get("HFX3D_PROFILE", "").strip())


def _ms(t0):
    return (time.perf_counter() - t0) * 1000.0

try:
    import h5py
except Exception:                     # save still works (json-only) without it
    h5py = None

try:
    import pycc
except Exception:                     # allows import outside CloudCompare
    pycc = None

# Qt binding must match CloudCompare's own Qt, loaded in the same process:
# CC 2.14 is built on Qt6 (use PyQt6), CC 2.13 on Qt5 (use PyQt5). Loading the
# wrong major version pulls a second, incompatible Qt into the process and
# crashes the whole app on load — so prefer PyQt6 and fall back to PyQt5.
try:
    from PyQt6 import QtCore, QtWidgets
    from PyQt6.QtGui import QColor
    _QT6 = True
except Exception:
    from PyQt5 import QtCore, QtWidgets
    from PyQt5.QtGui import QColor
    _QT6 = False

# PyQt6 scopes its enums (Qt.CheckState.Checked); PyQt5 does not (Qt.Checked).
# Normalise the handful this panel uses so the code below reads the same on both.
_Qt = QtCore.Qt
if _QT6:
    ALIGN_CENTER = _Qt.AlignmentFlag.AlignCenter
    CHECKED, UNCHECKED = _Qt.CheckState.Checked, _Qt.CheckState.Unchecked
    ITEM_CHECKABLE = _Qt.ItemFlag.ItemIsUserCheckable
    ITEM_ENABLED = _Qt.ItemFlag.ItemIsEnabled
    USER_ROLE = _Qt.ItemDataRole.UserRole
    STAY_ON_TOP = _Qt.WindowType.WindowStaysOnTopHint
    NO_EDIT = QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
    HDR_STRETCH = QtWidgets.QHeaderView.ResizeMode.Stretch
    MB = QtWidgets.QMessageBox.StandardButton
else:
    ALIGN_CENTER = _Qt.AlignCenter
    CHECKED, UNCHECKED = _Qt.Checked, _Qt.Unchecked
    ITEM_CHECKABLE = _Qt.ItemIsUserCheckable
    ITEM_ENABLED = _Qt.ItemIsEnabled
    USER_ROLE = _Qt.UserRole
    STAY_ON_TOP = _Qt.WindowStaysOnTopHint
    NO_EDIT = QtWidgets.QAbstractItemView.NoEditTriggers
    HDR_STRETCH = QtWidgets.QHeaderView.Stretch
    MB = QtWidgets.QMessageBox

FLAGS = ["", "bad_segmentation", "wrong_class", "other"]
SEM_NAMES = ["wall", "window", "door", "balcony", "vegetation", "stairs",
             "terrain", "roof", "blinds", "other", "column", "arch"]
# Where review files are written and who is reviewing — all come from
# environment variables so each machine/user is configured once, no code edits.
# HFX3D_REVIEW_ROOT : shared folder for review JSON files (e.g. a network drive)
# HFX3D_EXPORT_ROOT : shared folder for reviewed .h5 files (defaults to REVIEW_ROOT)
# HFX3D_REVIEWER    : this person's name, stamped into their review files
# If HFX3D_REVIEW_ROOT isn't set, fall back to a per-user folder that exists on
# any machine (not a hardcoded developer path) so Save never silently writes
# somewhere unexpected. Set the env vars (see README) to control this properly.
_FALLBACK_REVIEW_ROOT = Path.home() / "HFX3D_reviews"
_REVIEW_ROOT_ENV = os.environ.get("HFX3D_REVIEW_ROOT", "").strip()
if not _REVIEW_ROOT_ENV:
    print("HFX3D_REVIEW_ROOT is not set — reviews will be written to",
          _FALLBACK_REVIEW_ROOT, "(set the env var to change this).")
REVIEW_ROOT = Path(_REVIEW_ROOT_ENV) if _REVIEW_ROOT_ENV else _FALLBACK_REVIEW_ROOT
_EXPORT_ROOT_ENV = os.environ.get("HFX3D_EXPORT_ROOT", "").strip()
EXPORT_ROOT = Path(_EXPORT_ROOT_ENV) if _EXPORT_ROOT_ENV else REVIEW_ROOT
ENV_REVIEWER = os.environ.get("HFX3D_REVIEWER", "").strip()

_VAL_RE = re.compile(r"^val_(\d+)_(.+)$")
_CONF_RE = re.compile(r"^conf_(\d+)_(.+)$")

_PANEL = None                          # keep a global ref so the panel isn't GC'd


def _building_stem(name: str) -> str:
    name = Path(name).stem                       # drop any .laz/.las/.ply extension
    for suffix in ("_instances_vis", "_instances", "_vis", "_small"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()) or "unassigned"


def review_path_for(stem: str, reviewer: str) -> Path:
    """One review file per building per reviewer, so nobody overwrites anyone."""
    return REVIEW_ROOT / f"{stem}__{_safe(reviewer)}.review.json"


def export_path_for(stem: str, reviewer: str) -> Path:
    """The reviewed .h5 that pairs with review_path_for's json (same stem)."""
    return EXPORT_ROOT / f"{stem}__{_safe(reviewer)}.reviewed.h5"


# ── data pulled from the loaded cloud's scalar fields ────────────────────────

def _sf_np(cloud, idx):
    """Return a scalar field as a writable NumPy array.

    Across CloudCompare builds ``ScalarField.asArray()`` returns either a real
    numpy array or a ``cccorelib.ScalarFieldView``; ``np.asarray`` normalises
    both to an ndarray that shares the field's memory, so in-place writes still
    update the cloud (used for live re-colouring on edit / highlight).
    """
    return np.asarray(cloud.getScalarField(idx).asArray())


COMPANION_SUFFIX = ".attrs.npz"


def _find_companion(stem):
    """Locate a per-instance attribute companion for a slim review cloud.

    Slim clouds carry only instance_id/semantic_id/purity per point; the 15
    val + 15 conf attributes live in a small ``<stem>.attrs.npz`` beside them.
    CloudCompare's Python API doesn't expose the loaded cloud's file path, so
    we look the companion up by building name in the configured folders.
    """
    roots, seen = [], set()
    for env in ("HFX3D_ATTRS_ROOT", "HFX3D_REVIEW_ROOT", "HFX3D_EXPORT_ROOT"):
        v = os.environ.get(env, "").strip()
        if v:
            roots.append(Path(v))
    roots += [REVIEW_ROOT, EXPORT_ROOT, Path.cwd()]
    fname = f"{stem}{COMPANION_SUFFIX}"
    # check each root and its train/test/val subfolder directly. Targeted, not
    # a recursive walk — rglob over a root that happens to contain venvs, .git
    # or the raw review_clouds tree can take >10 s.
    for r in roots:
        for sd in ("", "train", "test", "val"):
            p = (r / sd / fname) if sd else (r / fname)
            if p not in seen:
                seen.add(p)
                if p.exists():
                    return p
    return None


class CloudModel:
    """Per-instance view of a review cloud.

    Attribute values/confidences are held per INSTANCE, not per point — they're
    constant across an instance's points, so this both saves memory and lets the
    panel work identically whether the cloud is a legacy "fat" LAZ (30 per-point
    val_/conf_ scalar fields) or a slim LAZ + ``.attrs.npz`` companion. Only
    instance_id (+ semantic_id/purity) is ever read per point.
    """

    def __init__(self, cloud, stem=""):
        self.cloud = cloud
        self.stem = stem
        self.companion_path = None
        names = [cloud.getScalarFieldName(i)
                 for i in range(cloud.getNumberOfScalarFields())]
        if "instance_id" not in names:
            raise ValueError("cloud has no 'instance_id' scalar field — "
                             "open a build_review_cloud.py LAZ")
        self.inst_idx = names.index("instance_id")
        self.sem_idx = names.index("semantic_id") if "semantic_id" in names else None
        self.purity_idx = names.index("purity") if "purity" in names else None

        # legacy per-point val_/conf_ fields, if present
        self._val_sf, self._conf_sf, sf_attr = {}, {}, {}
        for i, nm in enumerate(names):
            m = _VAL_RE.match(nm)
            if m:
                j = int(m.group(1)); self._val_sf[j] = i; sf_attr[j] = m.group(2); continue
            m = _CONF_RE.match(nm)
            if m:
                self._conf_sf[int(m.group(1))] = i

        # int32 is enough for any instance id and halves this copy's memory
        # vs int64 — real savings on 20-60M point buildings
        t0 = time.perf_counter() if _PROFILE else 0
        self._inst = _sf_np(cloud, self.inst_idx).astype(np.int32)
        uids, first, counts = np.unique(self._inst, return_index=True, return_counts=True)
        keep = uids >= 0
        self.ids = uids[keep].tolist()
        self._first = {int(u): int(f) for u, f in zip(uids[keep], first[keep])}
        self._count = {int(u): int(c) for u, c in zip(uids[keep], counts[keep])}
        self.idpos = {iid: k for k, iid in enumerate(self.ids)}
        self.n = len(self.ids)
        t_unique = time.perf_counter() if _PROFILE else 0

        sem = _sf_np(cloud, self.sem_idx) if self.sem_idx is not None else None
        self.sem_id = {iid: int(sem[self._first[iid]]) for iid in self.ids} if sem is not None else {}
        pur = _sf_np(cloud, self.purity_idx) if self.purity_idx is not None else None
        self.purity = {iid: float(pur[self._first[iid]]) for iid in self.ids} if pur is not None else {}

        # attribute table: legacy per-point SFs, else the slim companion file
        if self._val_sf:
            self.n_attr = max(self._val_sf) + 1
            self.attr_names = [sf_attr.get(j, f"attr{j}") for j in range(self.n_attr)]
            self._load_attrs_from_sfs()
        else:
            self._load_attrs_from_companion()

        # snapshot the pipeline vector so edits (which mutate val_h) never move
        # the "pipeline" reference used for the changed-vs-pipeline diff
        self.pipe0 = self.val_h.copy()
        self._has_conf = {j: bool(np.isfinite(self.conf_i[:, j]).any())
                          for j in range(self.n_attr)}
        t_attrs = time.perf_counter() if _PROFILE else 0

        self._hl_idx = self._ensure_highlight()
        if _PROFILE:
            print(f"[profile] CloudModel: unique/instances={(t_unique - t0) * 1000:.0f}ms "
                  f"attrs={(t_attrs - t_unique) * 1000:.0f}ms "
                  f"add_highlight_sf={_ms(t_attrs):.0f}ms "
                  f"({len(self._inst):,} pts, {self.n} instances, "
                  f"{'legacy' if self._val_sf else 'companion'})")
        self._colorby_idx = None
        self._hl_prev = None                  # point indices last highlighted
        self._ids_arr = np.asarray(self.ids, dtype=np.int64)
        self._maxid = int(self._inst.max()) if self._inst.size else 0

    # ── attribute-table loaders ─────────────────────────────────────────
    def _load_attrs_from_sfs(self):
        self.val_h = np.zeros((self.n, self.n_attr), np.int8)
        self.conf_i = np.full((self.n, self.n_attr), np.nan, np.float32)
        for j in range(self.n_attr):
            vp = _sf_np(self.cloud, self._val_sf[j])
            for k, iid in enumerate(self.ids):
                self.val_h[k, j] = 1 if float(vp[self._first[iid]]) > 0.5 else 0
        for j, si in self._conf_sf.items():
            cp = _sf_np(self.cloud, si)
            for k, iid in enumerate(self.ids):
                self.conf_i[k, j] = float(cp[self._first[iid]])

    def _load_attrs_from_companion(self):
        p = _find_companion(self.stem)
        if p is None:
            raise ValueError(
                "cloud has no 'val_*' scalar fields and no companion "
                f"'{self.stem}{COMPANION_SUFFIX}' was found (looked under "
                "HFX3D_ATTRS_ROOT / HFX3D_REVIEW_ROOT / HFX3D_EXPORT_ROOT / "
                "the working dir). Build one with tools/slim_review_cloud.py, "
                "or open the original build_review_cloud.py LAZ.")
        d = np.load(p, allow_pickle=True)
        self.companion_path = p
        val = np.asarray(d["val"])
        conf = np.asarray(d["conf"], dtype=np.float32)
        cid = np.asarray(d["instance_id"]).astype(np.int64)
        self.attr_names = [str(x) for x in d["attribute_names"]]
        self.n_attr = int(val.shape[1])
        rowof = {int(i): k for k, i in enumerate(cid)}
        self.val_h = np.zeros((self.n, self.n_attr), np.int8)
        self.conf_i = np.full((self.n, self.n_attr), np.nan, np.float32)
        for k, iid in enumerate(self.ids):
            r = rowof.get(int(iid))
            if r is None:
                continue                      # instance absent from companion
            self.val_h[k] = (val[r] > 0.5).astype(np.int8)
            self.conf_i[k] = conf[r]
        print("loaded attribute companion ->", p)

    def _ensure_highlight(self):
        return self._ensure_sf("review_highlight")

    def _ensure_sf(self, name):
        c = self.cloud
        for i in range(c.getNumberOfScalarFields()):
            if c.getScalarFieldName(i) == name:
                return i
        try:
            t0 = time.perf_counter() if _PROFILE else 0
            idx = c.addScalarField(name)
            if idx is None or idx < 0:
                return None
            t_add = time.perf_counter() if _PROFILE else 0
            _sf_np(c, idx)[:] = 0.0
            if _PROFILE:
                print(f"[profile] add SF '{name}': "
                      f"addScalarField={(t_add - t0) * 1000:.0f}ms zero={_ms(t_add):.0f}ms")
            return idx
        except Exception as exc:
            print(f"scalar field '{name}' unavailable:", exc)
            return None

    # ── per-instance reads ──────────────────────────────────────────────
    def cls(self, iid):
        return SEM_NAMES[self.sem_id[iid]] if iid in self.sem_id and \
            0 <= self.sem_id[iid] < len(SEM_NAMES) else "?"

    def purity_of(self, iid):
        return self.purity.get(iid, float("nan"))

    def count(self, iid):
        return self._count.get(iid, 0)

    def value(self, iid, j):
        return int(self.val_h[self.idpos[iid], j])

    def n_on(self, iid):
        return int(self.val_h[self.idpos[iid]].sum())

    def has_conf(self, j):
        return self._has_conf.get(j, False)

    def pipeline(self, iid, j):
        k = self.idpos[iid]
        if self._has_conf.get(j) and np.isfinite(self.conf_i[k, j]):
            return int(self.conf_i[k, j] >= 0.5)
        return int(self.pipe0[k, j])

    def confval(self, iid, j):
        return float(self.conf_i[self.idpos[iid], j])

    def applicable(self, iid, j):
        c = self.conf_i[self.idpos[iid], j]
        return not (np.isfinite(c) and c < 0.0)

    def set_value(self, iid, j, val):
        self.val_h[self.idpos[iid], j] = 1 if val else 0

    # ── 3D display helpers (build per-point fields on demand) ────────────
    def _broadcast(self, per_inst, fill):
        """Map a per-instance array onto every point via instance_id."""
        lut = np.full(self._maxid + 1, fill, np.float32)
        if self._ids_arr.size:
            lut[self._ids_arr] = np.asarray(per_inst, np.float32)
        pp = lut[np.clip(self._inst, 0, self._maxid)]
        pp[self._inst < 0] = fill
        return pp

    def color_by(self, kind, j):
        """Fill a single scratch scalar field with attr j's value/conf per point."""
        if self._colorby_idx is None:
            self._colorby_idx = self._ensure_sf("review_colorby")
        if self._colorby_idx is None:
            return None
        if kind == "conf":
            pp = self._broadcast(self.conf_i[:, j], float("nan"))
        else:
            pp = self._broadcast(self.val_h[:, j], 0.0)
        _sf_np(self.cloud, self._colorby_idx)[:] = pp
        return self._colorby_idx

    def set_highlight(self, iid):
        if self._hl_idx is None:
            return None
        arr = _sf_np(self.cloud, self._hl_idx)
        # clear only what we lit up last time, not all N points every click
        if self._hl_prev is None:
            arr[:] = 0.0
        else:
            arr[self._hl_prev] = 0.0
        cur = np.nonzero(self._inst == iid)[0]   # one O(N) scan (unavoidable)
        arr[cur] = 1.0
        self._hl_prev = cur
        return self._hl_idx


# ── review store (JSON, matches functional.ReviewStore) ──────────────────────

class Review:
    def __init__(self, path, model: CloudModel, building, reviewer=""):
        self.path = Path(path); self.model = model; self.building = building
        self.reviewer = reviewer; self.dirty = False; self.records = {}
        if self.path.exists():
            self._load()

    def _default(self, iid):
        return {"status": "unreviewed",
                "vector_human": [self.model.value(iid, j) for j in range(self.model.n_attr)],
                "instance_flag": "", "note": ""}

    def get(self, iid):
        iid = int(iid)
        if iid not in self.records:
            self.records[iid] = self._default(iid)
        return self.records[iid]

    def is_reviewed(self, iid):
        r = self.records.get(int(iid)); return bool(r and r["status"] == "reviewed")

    def mark_reviewed(self, iid, on=True):
        self.get(iid)["status"] = "reviewed" if on else "unreviewed"; self.dirty = True

    def n_reviewed(self):
        return sum(1 for r in self.records.values() if r["status"] == "reviewed")

    def _load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.reviewer = data.get("reviewer", self.reviewer)
        for k, v in data.get("instances", {}).items():
            self.records[int(k)] = {
                "status": v.get("status", "unreviewed"),
                "vector_human": [int(x) for x in v["vector_human"]],
                "instance_flag": v.get("instance_flag", ""), "note": v.get("note", "")}

    def save(self, split=""):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "building": self.building, "split": split, "reviewer": self.reviewer,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "attribute_names": self.model.attr_names,
            "instances": {str(iid): {
                "status": r["status"], "vector_human": r["vector_human"],
                "changed": [j for j in range(self.model.n_attr)
                            if r["vector_human"][j] != self.model.pipeline(iid, j)],
                "instance_flag": r["instance_flag"], "note": r["note"]}
                for iid, r in sorted(self.records.items())}}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.dirty = False
        return self.path


# ── reviewed .h5 export (minimal — mirrors functional.export_reviewed
# without the P/G/C scores; those live in the pipeline .h5 used to build
# this cloud, and can be joined back in later by instance_id if needed) ────

def write_reviewed_h5(path: Path, model: "CloudModel", review: "Review",
                      building: str) -> Path:
    if h5py is None:
        raise RuntimeError(
            "h5py is not importable from CloudCompare's Python — "
            "pip install h5py into that environment, then Save again.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = model.ids
    n, n_attr = len(ids), model.n_attr

    applicable = np.zeros((n, n_attr), np.uint8)
    conf = np.full((n, n_attr), np.nan, np.float32)
    pipeline_vec = np.zeros((n, n_attr), np.uint8)
    human_vec = np.zeros((n, n_attr), np.uint8)
    status = np.empty(n, dtype="S12")
    flag = np.empty(n, dtype="S20")
    note = np.empty(n, dtype=h5py.string_dtype())

    for k, iid in enumerate(ids):
        # read-only: don't touch review.records, so export never marks an
        # untouched instance as if a human had looked at it
        rec = review.records.get(iid)
        for j in range(n_attr):
            applicable[k, j] = int(model.applicable(iid, j))
            conf[k, j] = model.confval(iid, j)
            pipeline_vec[k, j] = model.pipeline(iid, j)
        human_vec[k] = (np.array(rec["vector_human"], np.uint8) if rec
                        else pipeline_vec[k])
        status[k] = (rec["status"] if rec else "unreviewed").encode()
        flag[k] = (rec["instance_flag"] if rec else "").encode()
        note[k] = rec["note"] if rec else ""

    with h5py.File(path, "w") as f:
        m = f.create_group("metadata")
        m.create_dataset("attribute_names",
                         data=np.array(model.attr_names, dtype=h5py.string_dtype()))
        m.create_dataset("attribute_count", data=np.int32(n_attr))
        m.create_dataset("reviewer", data=review.reviewer)
        m.create_dataset("building", data=building)
        m.create_dataset("exported_at", data=datetime.now().isoformat(timespec="seconds"))
        m.create_dataset("n_reviewed", data=np.int32(review.n_reviewed()))

        g = f.create_group("instances")
        g.create_dataset("instance_id", data=np.array(ids, np.int64))
        g.create_dataset("semantic_id",
                         data=np.array([model.sem_id.get(i, -1) for i in ids], np.int32))
        g.create_dataset("semantic_class",
                         data=np.array([model.cls(i) for i in ids], dtype=h5py.string_dtype()))
        g.create_dataset("semantic_purity",
                         data=np.array([model.purity_of(i) for i in ids], np.float32))
        g.create_dataset("point_count",
                         data=np.array([model.count(i) for i in ids], np.int64))
        g.create_dataset("applicable_mask", data=applicable)
        g.create_dataset("functional_attribute_confidence", data=conf)
        g.create_dataset("functional_attribute_vector", data=pipeline_vec)         # pipeline
        g.create_dataset("functional_attribute_vector_human", data=human_vec)      # human
        g.create_dataset("review_status", data=status)
        g.create_dataset("instance_flag", data=flag)
        g.create_dataset("review_note", data=note)
    return path


# ── panel ────────────────────────────────────────────────────────────────

class ReviewPanel(QtWidgets.QDialog):
    def __init__(self, cc, cloud):
        super().__init__()
        self.cc = cc
        self.stem = _building_stem(cloud.getName() or "cloud")
        self.model = CloudModel(cloud, self.stem)
        self.reviewer = ENV_REVIEWER
        self.review = Review(review_path_for(self.stem, self.reviewer),
                             self.model, self.stem, reviewer=self.reviewer)
        self._apply_loaded_review()  # crash/reopen recovery: push restored
                                      # decisions onto the scalar fields too,
                                      # so "Colour by val:" matches the table
        self.iid = None
        self._loading = False
        self._displayed_sf = None            # which SF is currently shown in 3D

        self.setWindowTitle(f"Functional Review — {self.stem}")
        self.setWindowFlags(self.windowFlags() | STAY_ON_TOP)
        self.resize(760, 640)
        self._build()
        self._refresh_list()
        self._try_register_picker()
        self._refresh_counter()

    def _apply_loaded_review(self):
        for iid, rec in self.review.records.items():
            if iid not in self.model._first:
                continue  # stale record from a since-changed cloud
            for j, val in enumerate(rec["vector_human"]):
                self.model.set_value(iid, j, val)

    def _build(self):
        outer = QtWidgets.QHBoxLayout(self)

        # ── left: instance list + filters ──────────────────────────────
        left = QtWidgets.QVBoxLayout()
        self.ed_search = QtWidgets.QLineEdit(); self.ed_search.setPlaceholderText("search id / class…")
        self.ed_search.textChanged.connect(self._refresh_list)
        left.addWidget(self.ed_search)
        frow = QtWidgets.QHBoxLayout()
        self.cb_class = QtWidgets.QComboBox(); self.cb_class.addItem("all classes", "")
        for c in sorted({self.model.cls(i) for i in self.model.ids}):
            self.cb_class.addItem(c, c)
        self.cb_class.currentIndexChanged.connect(self._refresh_list)
        frow.addWidget(self.cb_class)
        left.addLayout(frow)
        frow2 = QtWidgets.QHBoxLayout()
        self.cb_unrev = QtWidgets.QCheckBox("unreviewed"); self.cb_unrev.toggled.connect(self._refresh_list)
        self.cb_flag = QtWidgets.QCheckBox("flagged"); self.cb_flag.toggled.connect(self._refresh_list)
        frow2.addWidget(self.cb_unrev); frow2.addWidget(self.cb_flag); frow2.addStretch(1)
        left.addLayout(frow2)
        self.list = QtWidgets.QListWidget()
        self.list.currentItemChanged.connect(self._on_list_sel)
        left.addWidget(self.list, stretch=1)
        self.lbl_counter = QtWidgets.QLabel(""); left.addWidget(self.lbl_counter)
        outer.addLayout(left, stretch=1)

        # ── right: editor ──────────────────────────────────────────────
        right = QtWidgets.QVBoxLayout()
        rev = QtWidgets.QHBoxLayout()
        rev.addWidget(QtWidgets.QLabel("Reviewer"))
        self.ed_reviewer = QtWidgets.QLineEdit(self.review.reviewer)
        self.ed_reviewer.editingFinished.connect(self._on_reviewer)
        rev.addWidget(self.ed_reviewer)
        self.btn_save = QtWidgets.QPushButton("Save Review"); self.btn_save.clicked.connect(self._save)
        rev.addWidget(self.btn_save)
        right.addLayout(rev)

        cr = QtWidgets.QHBoxLayout()
        cr.addWidget(QtWidgets.QLabel("Colour by"))
        self.cb_color = QtWidgets.QComboBox()
        self.cb_color.addItem("Highlight selected", ("hl", -1))
        for j, nm in enumerate(self.model.attr_names):
            self.cb_color.addItem("val: " + nm, ("val", j))
        for j, nm in enumerate(self.model.attr_names):
            if self.model.has_conf(j):
                self.cb_color.addItem("conf: " + nm, ("conf", j))
        self.cb_color.currentIndexChanged.connect(self._on_color_by)
        cr.addWidget(self.cb_color, stretch=1)
        right.addLayout(cr)

        self.lbl_header = QtWidgets.QLabel("Select an instance")
        self.lbl_header.setStyleSheet("font-weight:600; font-size:14px;")
        right.addWidget(self.lbl_header)
        self.lbl_flagnote = QtWidgets.QHBoxLayout()
        self.lbl_flagnote.addWidget(QtWidgets.QLabel("Flag"))
        self.cmb_flag = QtWidgets.QComboBox()
        for fl in FLAGS:
            self.cmb_flag.addItem(fl or "(none)", fl)
        self.cmb_flag.currentIndexChanged.connect(self._on_flag)
        self.lbl_flagnote.addWidget(self.cmb_flag)
        right.addLayout(self.lbl_flagnote)

        self.table = QtWidgets.QTableWidget(self.model.n_attr, 3)
        self.table.setHorizontalHeaderLabels(["attribute", "conf", "on"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(NO_EDIT)
        self.table.horizontalHeader().setSectionResizeMode(0, HDR_STRETCH)
        self.table.itemChanged.connect(self._on_item)
        right.addWidget(self.table, stretch=1)

        bulk = QtWidgets.QHBoxLayout()
        for txt, fn in [("Accept all", lambda: self._bulk(1)),
                        ("Reject all", lambda: self._bulk(0)),
                        ("Reset to pipeline", self._reset)]:
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); bulk.addWidget(b)
        right.addLayout(bulk)

        right.addWidget(QtWidgets.QLabel("Note"))
        self.ed_note = QtWidgets.QLineEdit(); self.ed_note.editingFinished.connect(self._on_note)
        right.addWidget(self.ed_note)

        nav = QtWidgets.QHBoxLayout()
        self.btn_confirm = QtWidgets.QPushButton("Confirm ✓  &  Next ›")
        self.btn_confirm.setStyleSheet("font-weight:600; padding:6px;")
        self.btn_confirm.clicked.connect(self._confirm_next)
        self.btn_unmark = QtWidgets.QPushButton("Unmark reviewed")
        self.btn_unmark.clicked.connect(self._unmark)
        nav.addWidget(self.btn_confirm, stretch=2); nav.addWidget(self.btn_unmark, stretch=1)
        right.addLayout(nav)
        outer.addLayout(right, stretch=2)

    # ── list ────────────────────────────────────────────────────────────
    def _visible_ids(self):
        cls = self.cb_class.currentData()
        q = self.ed_search.text().lower().strip()
        out = []
        for iid in self.model.ids:
            if cls and self.model.cls(iid) != cls:
                continue
            rec = self.review.records.get(iid)
            if self.cb_unrev.isChecked() and rec and rec["status"] == "reviewed":
                continue
            if self.cb_flag.isChecked() and not (rec and rec["instance_flag"]):
                continue
            if q and q not in str(iid) and q not in self.model.cls(iid).lower():
                continue
            out.append(iid)
        return out

    def _refresh_list(self):
        self._loading = True
        self.list.clear()
        for iid in self._visible_ids():
            mark = "✓ " if self.review.is_reviewed(iid) else "   "
            flag = " ⚑" if (self.review.records.get(iid) or {}).get("instance_flag") else ""
            it = QtWidgets.QListWidgetItem(
                f"{mark}#{iid}  {self.model.cls(iid)}  ({self.model.n_on(iid)} on){flag}")
            it.setData(USER_ROLE, iid)
            self.list.addItem(it)
        self._loading = False

    def _on_list_sel(self, cur, _prev):
        if self._loading or cur is None:
            return
        self.select(int(cur.data(USER_ROLE)))

    # ── selection ───────────────────────────────────────────────────────
    def select(self, iid):
        iid = int(iid)
        if iid not in self.model._first:
            return
        if _PROFILE:
            print(f"[profile] select() called for #{iid}")
        t0 = time.perf_counter() if _PROFILE else 0
        self.iid = iid
        rec = self.review.get(iid)
        self._loading = True
        self.lbl_header.setText(
            f"Instance {iid} · {self.model.cls(iid)} · {self.model.count(iid):,} pts")
        self.cmb_flag.setCurrentIndex(max(0, self.cmb_flag.findData(rec["instance_flag"])))
        self.ed_note.setText(rec["note"])
        for j in range(self.model.n_attr):
            appl = self.model.applicable(iid, j)
            it0 = QtWidgets.QTableWidgetItem(self.model.attr_names[j])
            cf = self.model.confval(iid, j)
            it1 = QtWidgets.QTableWidgetItem("—" if (not appl or cf != cf) else f"{cf:.2f}")
            it1.setTextAlignment(ALIGN_CENTER)
            it2 = QtWidgets.QTableWidgetItem()
            it2.setFlags(ITEM_CHECKABLE | ITEM_ENABLED)
            it2.setCheckState(CHECKED if rec["vector_human"][j] else UNCHECKED)
            if not appl:
                for it in (it0, it1, it2):
                    it.setForeground(QColor("#888"))
            self.table.setItem(j, 0, it0); self.table.setItem(j, 1, it1); self.table.setItem(j, 2, it2)
        self._loading = False
        t_table = time.perf_counter() if _PROFILE else 0
        # highlight in 3D (0/1 field, range never changes → no recompute)
        hl = self.model.set_highlight(iid)
        t_hl = time.perf_counter() if _PROFILE else 0
        kind, _ = self.cb_color.currentData()
        if kind == "hl" and hl is not None:
            self._set_displayed_sf(hl, recompute=False)
        # keep list selection in sync (when driven by picking)
        self._sync_list_selection(iid)
        if _PROFILE:
            print(f"[profile] select #{iid}: "
                  f"table={(t_table - t0) * 1000:.0f}ms "
                  f"highlight={(t_hl - t_table) * 1000:.0f}ms "
                  f"draw={(time.perf_counter() - t_hl) * 1000:.0f}ms "
                  f"total={_ms(t0):.0f}ms")

    def _sync_list_selection(self, iid):
        for r in range(self.list.count()):
            if int(self.list.item(r).data(USER_ROLE)) == iid:
                self._loading = True
                self.list.setCurrentRow(r); self._loading = False
                return

    # ── edits (auto-mark reviewed) ───────────────────────────────────────
    def _touch_reviewed(self):
        if self.iid is not None and not self.review.is_reviewed(self.iid):
            self.review.mark_reviewed(self.iid, True)
            self._update_list_item(self.iid)

    def _on_item(self, item):
        if self._loading or self.iid is None or item.column() != 2:
            return
        j = item.row()
        val = 1 if item.checkState() == CHECKED else 0
        self.review.get(self.iid)["vector_human"][j] = val
        self.review.dirty = True
        self.model.set_value(self.iid, j, val)
        self._touch_reviewed()
        kind, cj = self.cb_color.currentData()
        if kind == "val" and cj == j:
            self._refresh_colorby()
        self._update_list_item(self.iid)
        self._refresh_counter()

    def _bulk(self, val):
        if self.iid is None:
            return
        rec = self.review.get(self.iid)
        for j in range(self.model.n_attr):
            rec["vector_human"][j] = val
            self.model.set_value(self.iid, j, val)
        self.review.dirty = True
        self._touch_reviewed()
        self.select(self.iid); self._refresh_colorby(); self._refresh_counter()

    def _reset(self):
        if self.iid is None:
            return
        rec = self.review.get(self.iid)
        for j in range(self.model.n_attr):
            pv = self.model.pipeline(self.iid, j)
            rec["vector_human"][j] = pv
            self.model.set_value(self.iid, j, pv)
        self.review.dirty = True
        self._touch_reviewed()
        self.select(self.iid); self._refresh_colorby(); self._refresh_counter()

    def _on_flag(self):
        if self._loading or self.iid is None:
            return
        self.review.get(self.iid)["instance_flag"] = self.cmb_flag.currentData()
        self.review.dirty = True
        self._touch_reviewed(); self._update_list_item(self.iid)

    def _on_note(self):
        if self._loading or self.iid is None:
            return
        self.review.get(self.iid)["note"] = self.ed_note.text(); self.review.dirty = True

    def _on_reviewer(self):
        name = self.ed_reviewer.text().strip()
        if name == self.reviewer:
            return
        if self.review.dirty:
            r = QtWidgets.QMessageBox.question(
                self, "Switch reviewer",
                f"Save {self.reviewer or 'current'}'s edits before switching to {name}?",
                MB.Save | MB.Discard | MB.Cancel)
            if r == MB.Cancel:
                self.ed_reviewer.setText(self.reviewer); return
            if r == MB.Save:
                self.review.save()
        self.reviewer = name
        self.review = Review(review_path_for(self.stem, name), self.model, self.stem, reviewer=name)
        self._refresh_list(); self._refresh_counter()
        if self.iid is not None:
            self.select(self.iid)

    def _confirm_next(self):
        if self.iid is not None:
            self.review.mark_reviewed(self.iid, True)
            self._update_list_item(self.iid); self._refresh_counter()
        self._nav_next_unreviewed()

    def _unmark(self):
        if self.iid is not None:
            self.review.mark_reviewed(self.iid, False)
            self._update_list_item(self.iid); self._refresh_counter()

    def _nav_next_unreviewed(self):
        ids = [int(self.list.item(r).data(USER_ROLE)) for r in range(self.list.count())]
        if not ids:
            return
        start = ids.index(self.iid) if self.iid in ids else -1
        for step in range(1, len(ids) + 1):
            k = (start + step) % len(ids)
            if not self.review.is_reviewed(ids[k]):
                self.select(ids[k]); return
        # none left unreviewed → just advance
        self.select(ids[(start + 1) % len(ids)])

    def _update_list_item(self, iid):
        for r in range(self.list.count()):
            if int(self.list.item(r).data(USER_ROLE)) == iid:
                mark = "✓ " if self.review.is_reviewed(iid) else "   "
                flag = " ⚑" if (self.review.records.get(iid) or {}).get("instance_flag") else ""
                self.list.item(r).setText(
                    f"{mark}#{iid}  {self.model.cls(iid)}  ({self.model.n_on(iid)} on){flag}")
                return

    # ── recolour ─────────────────────────────────────────────────────────
    def _on_color_by(self):
        kind, j = self.cb_color.currentData()
        if kind == "hl":
            if self.iid is not None:
                hl = self.model.set_highlight(self.iid)
                if hl is not None:
                    self._set_displayed_sf(hl)
        else:
            idx = self.model.color_by(kind, j)
            if idx is not None:
                self._set_displayed_sf(idx)

    def _refresh_colorby(self):
        """Re-apply the current val/conf colouring after edits change values."""
        kind, j = self.cb_color.currentData()
        if kind in ("val", "conf"):
            idx = self.model.color_by(kind, j)
            if idx is not None:
                self._set_displayed_sf(idx)

    def _set_displayed_sf(self, sf_idx, recompute=True):
        """Show scalar field ``sf_idx`` and redraw once.

        Only re-binds the displayed field / recomputes its colour range when it
        actually changes — on repeated highlight clicks the field index and its
        0/1 range are unchanged, so we skip straight to a single redraw. The old
        code re-bound the field, rescanned min/max, and rendered the whole cloud
        TWICE (redrawDisplay + redrawAll) on every click.
        """
        cloud = self.model.cloud
        try:
            if sf_idx != self._displayed_sf:
                cloud.setCurrentDisplayedScalarField(sf_idx)
                if hasattr(cloud, "showSF"):
                    cloud.showSF(True)
                self._displayed_sf = sf_idx
                recompute = True                  # new field must be scaled once
            if recompute:
                sf = cloud.getScalarField(sf_idx)
                if hasattr(sf, "computeMinAndMax"):
                    sf.computeMinAndMax()
            if hasattr(cloud, "redrawDisplay"):
                cloud.redrawDisplay()
            self.cc.updateUI()
        except Exception as exc:
            print("recolor failed:", exc)

    def _refresh_counter(self):
        self.lbl_counter.setText(
            f"{self.review.n_reviewed()} / {len(self.model.ids)} reviewed"
            + ("   • unsaved" if self.review.dirty else ""))

    def _save(self):
        p = self.review.save()
        self._refresh_counter()
        print("saved review ->", p)
        try:
            h5_path = export_path_for(self.stem, self.reviewer)
            write_reviewed_h5(h5_path, self.model, self.review, self.stem)
            print("exported reviewed h5 ->", h5_path)
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Saved:\n{p}\n{h5_path}\n\nUpload both files.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Saved (json only)",
                f"Saved the review json:\n{p}\n\n"
                f"Could not write the reviewed .h5:\n{exc}\n\n"
                "If h5py isn't installed in CloudCompare's Python, run "
                "'pip install h5py' there and Save again.")
        self.cc.updateUI()

    # ── picking (best-effort) ────────────────────────────────────────────
    def _try_register_picker(self):
        self._picker = None
        if pycc is None:
            return
        try:
            hub = self.cc.pickingHub()
            panel = self

            class _L(pycc.ccPickingListener):
                def __init__(self):
                    pycc.ccPickingListener.__init__(self)

                def onItemPicked(self, pi):
                    try:
                        idx = int(getattr(pi, "itemIndex", -1))
                        if idx >= 0:
                            iid = int(panel.model._inst[idx])
                            if iid >= 0:
                                panel.select(iid)
                    except Exception as e:
                        print("pick handler error:", e)

            self._picker = _L()
            hub.addListener(self._picker, False, True,
                            pycc.ccGLWindowInterface.PICKING_MODE.POINT_PICKING)
        except Exception as exc:
            print("picking listener not available (use the list):", exc)

    def closeEvent(self, ev):
        try:
            if self._picker is not None:
                self.cc.pickingHub().removeListener(self._picker)
        except Exception:
            pass
        if self.review.dirty:
            r = QtWidgets.QMessageBox.question(
                self, "Unsaved review", "Save review before closing?",
                MB.Save | MB.Discard | MB.Cancel)
            if r == MB.Save:
                self._save()
            elif r == MB.Cancel:
                ev.ignore(); return
        ev.accept()


# ── entry points ────────────────────────────────────────────────────────────

def main():
    global _PANEL
    cc = pycc.GetInstance()
    cloud = None
    for e in (cc.getSelectedEntities() or []):
        if isinstance(e, pycc.ccPointCloud):
            cloud = e; break
    if cloud is None:
        db = cc.dbRootObject()
        for i in range(db.getChildrenNumber() if db else 0):
            c = db.getChild(i)
            if isinstance(c, pycc.ccPointCloud):
                cloud = c; break
    if cloud is None:
        QtWidgets.QMessageBox.warning(None, "Functional Review",
                                      "Select the review point cloud first.")
        return
    try:
        _PANEL = ReviewPanel(cc, cloud)
    except Exception as exc:
        # Letting this propagate back into CC's action-dispatch can take the
        # whole app down with it — show it, log it, and stop here instead.
        import traceback
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(None, "Functional Review", str(exc))
        return
    _PANEL.show()


if pycc is not None:
    class FunctionalReviewPlugin(pycc.PythonPluginInterface):
        def __init__(self):
            pycc.PythonPluginInterface.__init__(self)

        def getIcon(self):
            return ""

        def getActions(self):
            return [pycc.Action(name="Functional Review", target=main)]
