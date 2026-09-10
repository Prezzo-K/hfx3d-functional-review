#!/usr/bin/env python3
"""Standalone functional-attribute review app (PySide6 + PyVista).

Reviews the per-instance functional attributes of a building point cloud WITHOUT
CloudCompare. It opens a "bundle" produced by build_bundle.py: the whole building
is drawn once as a decimated grey backdrop, and selecting an instance just swaps a
small red highlight actor (its points are an O(1) memmap slice) — so browsing ~300
instances is instant, even on an integrated GPU.

v2 adds segmentation editing (the review found upstream seg/instance errors):
    • reclass  — change an instance's semantic class (e.g. blinds -> window)
    • merge    — fuse several selected instances into one (keeps the dominant)
    • split    — green-lasso a sub-region into a brand-new instance
All edits are per-POINT under the hood (see edit_model.EditModel), logged, and
undoable (Ctrl+Z). Saving writes, alongside the v1 per-instance outputs:
    <building>__<reviewer>.edits.npz       compact segmentation diff (reload state)
    <building>__<reviewer>.edits.json      human-readable op log (provenance)
    <building>__<reviewer>.corrected.laz   source cloud with corrected labels baked

Output is otherwise identical to the CloudCompare plugin:
    <HFX3D_REVIEW_ROOT>/<building>__<reviewer>.review.json
    <HFX3D_EXPORT_ROOT>/<building>__<reviewer>.reviewed.h5

    python app/review_app.py bundles/HFX_BLD001        # or launch and File->Open

Env (same as the plugin): HFX3D_REVIEWER, HFX3D_REVIEW_ROOT, HFX3D_EXPORT_ROOT.
Requires: pip install pyside6 pyvista pyvistaqt numpy h5py laspy[lazrs]
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_model import EditModel, CLASS_APPLICABLE, sem_name

try:
    import h5py
except Exception:
    h5py = None

SEM_NAMES = ["wall", "window", "door", "balcony", "vegetation", "stairs",
             "terrain", "roof", "blinds", "other", "column", "arch"]
FLAGS = ["", "bad_segmentation", "wrong_class", "other"]
HL_DISPLAY_CAP = 250_000          # subsample only the 3D display of huge instances

# Settings persist in a small config file so you don't depend on env vars
# (which need a fresh shell after `setx`). Priority: env var > saved config >
# default. Whatever you set in the app is written back here.
CONFIG_PATH = Path.home() / ".hfx3d_review.json"


def _load_cfg():
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cfg(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as exc:
        print("could not save settings:", exc)


_CFG = _load_cfg()


def _cfg_reviewer():
    return os.environ.get("HFX3D_REVIEWER", "").strip() or _CFG.get("reviewer", "")


def _cfg_review_root():
    v = os.environ.get("HFX3D_REVIEW_ROOT", "").strip()
    return Path(v) if v else Path(_CFG.get("review_root") or (Path.home() / "HFX3D_reviews"))


def _cfg_export_root(review_root):
    v = os.environ.get("HFX3D_EXPORT_ROOT", "").strip()
    if v:
        return Path(v)
    return Path(_CFG["export_root"]) if _CFG.get("export_root") else Path(review_root)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()) or "unassigned"


class Bundle:
    """Loads a build_bundle.py folder; per-instance attribute access.

    Holds the IMMUTABLE original segmentation (points, offsets, per-instance
    attribute tables). Live segmentation edits live in an EditModel layered on
    top of this — see MainWindow.em. `src_index` (each points.npy row's original
    source-cloud row) is what lets edits be written back to the LAZ; older
    bundles without it open fine but can't be edited/exported as a corrected cloud.
    """

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.context = np.load(self.folder / "context.npy", mmap_mode="r")
        self.points = np.load(self.folder / "points.npy", mmap_mode="r")
        self.offsets = np.load(self.folder / "offsets.npy")
        cif = self.folder / "context_inst.npy"          # optional (older bundles)
        self.context_inst = np.load(cif) if cif.exists() else None
        sif = self.folder / "src_index.npy"             # optional (pre-v2 bundles)
        self.src_index = np.load(sif) if sif.exists() else None
        m = np.load(self.folder / "meta.npz", allow_pickle=True)
        self.ids = [int(x) for x in m["instance_id"]]
        self.sem = m["semantic_id"]
        self.purity = m["purity"]
        self.counts = m["point_count"]
        self.bbox = m["bbox"]
        self.centroid = m["centroid"]
        self.building = str(m["building"]) if "building" in m else self.folder.name
        if "val" in m:
            self.val = m["val"].astype(np.int8)
            self.conf = m["conf"].astype(np.float32)
            self.attr_names = [str(x) for x in m["attribute_names"]]
        else:                                    # slim bundle without attrs
            self.val = np.zeros((len(self.ids), 0), np.int8)
            self.conf = np.zeros((len(self.ids), 0), np.float32)
            self.attr_names = []
        self.n_attr = len(self.attr_names)
        self.idpos = {iid: k for k, iid in enumerate(self.ids)}
        self.max_id = max(self.ids) if self.ids else 0

    @property
    def editable(self):
        return self.src_index is not None

    # ---- original per-position (k) attribute access (unchanged from v1) ----
    def cls_k(self, k):
        s = int(self.sem[k])
        return SEM_NAMES[s] if 0 <= s < len(SEM_NAMES) else "?"

    def has_conf(self, j):
        return bool(np.isfinite(self.conf[:, j]).any()) if self.n_attr else False

    def pipeline_k(self, k, j):
        c = self.conf[k, j]
        return int(c >= 0.5) if np.isfinite(c) else int(self.val[k, j])

    def confval_k(self, k, j):
        return float(self.conf[k, j]) if self.n_attr else float("nan")

    def applicable_k(self, k, j):
        c = self.conf[k, j] if self.n_attr else float("nan")
        return not (np.isfinite(c) and c < 0.0)


class ReviewStore:
    """Per-instance review records (attributes/flag/note/status), keyed by
    instance id. File IO for the v1 JSON contract; the H5 export and the v2
    corrected-cloud export are driven by MainWindow (which owns the EditModel)."""

    def __init__(self, building, reviewer, default_vec, n_attr,
                 review_root=None, export_root=None):
        self.building = building
        self.reviewer = reviewer
        self.default_vec = default_vec          # callable(iid) -> list[int]
        self.n_attr = n_attr
        self.review_root = Path(review_root) if review_root else _cfg_review_root()
        self.export_root = Path(export_root) if export_root else _cfg_export_root(self.review_root)
        self.dirty = False
        self.records = {}
        self.path = self._json_path()
        self.loaded_from = None
        if self.path.exists():
            self._load()
            self.loaded_from = self.path

    def _json_path(self):
        return self.review_root / f"{self.building}__{_safe(self.reviewer)}.review.json"

    def _default(self, iid):
        return {"status": "unreviewed", "vector_human": list(self.default_vec(iid)),
                "instance_flag": "", "note": ""}

    def get(self, iid):
        if iid not in self.records:
            self.records[iid] = self._default(iid)
        return self.records[iid]

    def is_reviewed(self, iid):
        r = self.records.get(iid)
        return bool(r and r["status"] == "reviewed")

    def mark(self, iid, on=True):
        self.get(iid)["status"] = "reviewed" if on else "unreviewed"
        self.dirty = True

    def n_reviewed(self):
        return sum(1 for r in self.records.values() if r["status"] == "reviewed")

    def _load(self):
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.reviewer = d.get("reviewer", self.reviewer)
        for k, v in d.get("instances", {}).items():
            self.records[int(k)] = {
                "status": v.get("status", "unreviewed"),
                "vector_human": [int(x) for x in v["vector_human"]],
                "instance_flag": v.get("instance_flag", ""), "note": v.get("note", "")}

    def merge_from(self, path):
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return 0
        added = 0
        for k, v in d.get("instances", {}).items():
            iid = int(k)
            if iid not in self.records:
                self.records[iid] = {
                    "status": v.get("status", "unreviewed"),
                    "vector_human": [int(x) for x in v["vector_human"]],
                    "instance_flag": v.get("instance_flag", ""), "note": v.get("note", "")}
                added += 1
        return added

    def load_seed(self, path):
        """Pre-fill records from a part-1 seed (vectors + flags + notes) so the
        reviewer starts from the validated state. Does NOT set loaded_from, so
        the reviewer's first Save writes to their own file, not the seed."""
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return 0
        added = 0
        for k, v in d.get("instances", {}).items():
            iid = int(k)
            if iid not in self.records:
                self.records[iid] = {
                    "status": v.get("status", "unreviewed"),
                    "vector_human": [int(x) for x in v["vector_human"]],
                    "instance_flag": v.get("instance_flag", ""), "note": v.get("note", "")}
                added += 1
        return added


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, bundle: Bundle):
        super().__init__()
        self.b = bundle
        self.em = EditModel(bundle)
        self.cur = None                              # current instance id
        self._loading = False
        self._lasso_active = False
        self._prev_style = None
        self.review = ReviewStore(bundle.building, _cfg_reviewer(),
                                  self._default_vec, bundle.n_attr)
        self._maybe_load_edits()
        self._maybe_load_seed()
        self.setWindowTitle(f"Functional Review — {bundle.building}")
        self.resize(1320, 820)
        self._build()
        self._add_context()
        self._refresh_list()
        self._refresh_counter()
        self._start_update_check()

    # ── iid-keyed geometry/attribute helpers ─────────────────────────────
    def _k(self, iid):
        return self.b.idpos.get(iid)                 # original position, or None (new)

    def _pipe(self, iid, j):
        k = self._k(iid)
        return 0 if k is None else self.b.pipeline_k(k, j)

    def _conf(self, iid, j):
        k = self._k(iid)
        return float("nan") if k is None else self.b.confval_k(k, j)

    def _appl(self, iid, j):
        """Applicable? Use the pipeline's exact encoding while the class is
        unchanged (zero regression vs v1); switch to the ontology once an
        instance is reclassed or is brand-new (a split child)."""
        if self._cls(iid) == "unsegmented":
            return False                             # no attributes on leftover points
        k = self._k(iid)
        if k is not None and self.em.sem.get(iid) == int(self.b.sem[k]):
            return self.b.applicable_k(k, j)
        appl = CLASS_APPLICABLE.get(self.em.cls(iid), set())
        return self.b.attr_names[j] in appl

    def _default_vec(self, iid):
        return [self._pipe(iid, j) for j in range(self.b.n_attr)]

    def _cls(self, iid):
        return self.em.cls(iid)

    def _count(self, iid):
        return self.em.count(iid)

    def _inst_points(self, iid, cap=True):
        seg = np.asarray(self.b.points[self.em.rows[iid]])
        if cap and len(seg) > HL_DISPLAY_CAP:
            seg = seg[::len(seg) // HL_DISPLAY_CAP + 1]
        return seg

    def _attr_val(self, iid, j):
        r = self.review.records.get(iid)
        return r["vector_human"][j] if r else self._pipe(iid, j)

    def _changed(self, iid):
        return any(self._attr_val(iid, j) != self._pipe(iid, j) for j in range(self.b.n_attr))

    # ── edit-state persistence ───────────────────────────────────────────
    def _edits_npz_path(self):
        return self.review.review_root / f"{self.b.building}__{_safe(self.review.reviewer)}.edits.npz"

    def _maybe_load_edits(self):
        p = self._edits_npz_path()
        if p.exists() and self.b.editable:
            try:
                self.em.load_edits(p)
            except Exception as exc:
                print("could not load prior edits:", exc)

    def _maybe_load_seed(self):
        """If this reviewer has no file yet, seed from the part-1 review
        (seed.review.json staged into the bundle) so flags/notes/vectors show."""
        if self.review.loaded_from is not None:
            return
        seed = self.b.folder / "seed.review.json"
        if seed.exists():
            n = self.review.load_seed(seed)
            if n:
                print(f"loaded {n} instances from part-1 seed {seed}")

    # ── UI ───────────────────────────────────────────────────────────────
    def _build(self):
        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)

        # left: instance list + filters
        left = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(left)
        self.ed_search = QtWidgets.QLineEdit(); self.ed_search.setPlaceholderText("search id / class…")
        self.ed_search.textChanged.connect(self._refresh_list); lv.addWidget(self.ed_search)
        self.cb_class = QtWidgets.QComboBox(); self.cb_class.addItem("all classes", "")
        for c in SEM_NAMES:
            self.cb_class.addItem(c, c)
        self.cb_class.currentIndexChanged.connect(self._refresh_list); lv.addWidget(self.cb_class)
        arow = QtWidgets.QHBoxLayout()
        self.cb_attr = QtWidgets.QComboBox(); self.cb_attr.addItem("(any attribute)", -1)
        for j, nm in enumerate(self.b.attr_names):
            self.cb_attr.addItem(nm, j)
        self.cb_attr.currentIndexChanged.connect(self._refresh_list)
        self.cb_attr_state = QtWidgets.QComboBox()
        self.cb_attr_state.addItem("is on", 1); self.cb_attr_state.addItem("is off", 0)
        self.cb_attr_state.currentIndexChanged.connect(self._refresh_list)
        arow.addWidget(self.cb_attr, 1); arow.addWidget(self.cb_attr_state); lv.addLayout(arow)
        crow = QtWidgets.QHBoxLayout()
        self.cb_unrev = QtWidgets.QCheckBox("unreviewed"); self.cb_unrev.toggled.connect(self._refresh_list)
        self.cb_flag = QtWidgets.QCheckBox("flagged"); self.cb_flag.toggled.connect(self._refresh_list)
        self.cb_changed = QtWidgets.QCheckBox("changed"); self.cb_changed.toggled.connect(self._refresh_list)
        crow.addWidget(self.cb_unrev); crow.addWidget(self.cb_flag); crow.addWidget(self.cb_changed)
        crow.addStretch(1); lv.addLayout(crow)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.currentRowChanged.connect(self._on_current)
        self.list.itemSelectionChanged.connect(self._highlight_selected)
        lv.addWidget(self.list, 1)
        b_all = QtWidgets.QPushButton("Select all filtered ⭢ 3D")
        b_all.clicked.connect(self.list.selectAll); lv.addWidget(b_all)
        # segmentation-edit actions on the current selection
        seg = QtWidgets.QHBoxLayout()
        self.b_merge = QtWidgets.QPushButton("Merge selected")
        self.b_merge.setToolTip("Fuse all selected instances into one (keeps the "
                                "largest as the survivor and its attributes)")
        self.b_merge.clicked.connect(self._merge_selected)
        self.b_lasso = QtWidgets.QPushButton("Lasso split")
        self.b_lasso.setCheckable(True)
        self.b_lasso.setToolTip("Draw a green polygon around a sub-region of the "
                                "current instance to cut it into a new instance")
        self.b_lasso.toggled.connect(self._toggle_lasso)
        self.b_undo = QtWidgets.QPushButton("Undo")
        self.b_undo.setToolTip("Undo the last segmentation edit (reclass / merge / split) — Ctrl+Z")
        self.b_undo.clicked.connect(self._undo_edit)
        self.b_undo.setEnabled(False)
        seg.addWidget(self.b_merge); seg.addWidget(self.b_lasso); seg.addWidget(self.b_undo)
        lv.addLayout(seg)
        self.lbl_counter = QtWidgets.QLabel(""); lv.addWidget(self.lbl_counter)
        split.addWidget(left)

        # center: 3D view. left-drag rotate, right-drag pan, wheel zoom.
        self.plotter = QtInteractor()
        self._set_pan_on_right()
        self.plotter.interactor.installEventFilter(self)   # for lasso capture
        split.addWidget(self.plotter.interactor)

        # right: editor
        right = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(right)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Reviewer"))
        self.ed_rev = QtWidgets.QLineEdit(self.review.reviewer)
        self.ed_rev.setPlaceholderText("your name")
        top.addWidget(self.ed_rev)
        b_save = QtWidgets.QPushButton("Save"); b_save.clicked.connect(self._save); top.addWidget(b_save)
        rv.addLayout(top)
        srow = QtWidgets.QHBoxLayout(); srow.addWidget(QtWidgets.QLabel("Save to"))
        self.lbl_saveto = QtWidgets.QLabel(str(self.review.review_root)); self.lbl_saveto.setStyleSheet("color:#666")
        srow.addWidget(self.lbl_saveto, 1)
        b_ch = QtWidgets.QPushButton("Change…"); b_ch.clicked.connect(self._change_saveto)
        srow.addWidget(b_ch); rv.addLayout(srow)
        self.b_export_laz = QtWidgets.QPushButton("Export corrected .laz (point cloud)")
        self.b_export_laz.setToolTip("Bake your corrected per-point labels into a full copy of "
                                     "the source cloud (~150+ MB). Only needed for downstream "
                                     "training — the review itself doesn't require it.")
        self.b_export_laz.clicked.connect(self._export_corrected_cloud)
        self.b_export_laz.setEnabled(self.b.editable)
        rv.addWidget(self.b_export_laz)
        self.cb_follow = QtWidgets.QCheckBox("zoom to instance on select"); self.cb_follow.setChecked(True)
        rv.addWidget(self.cb_follow)
        cr = QtWidgets.QHBoxLayout(); cr.addWidget(QtWidgets.QLabel("Colour by"))
        self.cb_color = QtWidgets.QComboBox()
        self.cb_color.addItem("plain (grey)", ("hl", -1))
        for j, nm in enumerate(self.b.attr_names):
            self.cb_color.addItem("val: " + nm, ("val", j))
        for j, nm in enumerate(self.b.attr_names):
            if self.b.has_conf(j):
                self.cb_color.addItem("conf: " + nm, ("conf", j))
        self.cb_color.setEnabled(self.b.context_inst is not None)
        self.cb_color.currentIndexChanged.connect(self._apply_colorby)
        cr.addWidget(self.cb_color, 1); rv.addLayout(cr)

        self.lbl_header = QtWidgets.QLabel("Select an instance")
        self.lbl_header.setStyleSheet("font-weight:600;font-size:14px;"); rv.addWidget(self.lbl_header)

        # reclass + flag row
        clr = QtWidgets.QHBoxLayout()
        clr.addWidget(QtWidgets.QLabel("Class"))
        self.cmb_class = QtWidgets.QComboBox()
        for c in SEM_NAMES:
            self.cmb_class.addItem(c, c)
        self.cmb_class.setToolTip("Reclass this instance (e.g. blinds → window). "
                                  "Updates which attributes apply.")
        self.cmb_class.currentIndexChanged.connect(self._on_reclass)
        self.cmb_class.setEnabled(self.b.editable)
        clr.addWidget(self.cmb_class, 1)
        clr.addWidget(QtWidgets.QLabel("Flag"))
        self.cmb_flag = QtWidgets.QComboBox()
        for fl in FLAGS:
            self.cmb_flag.addItem(fl or "(none)", fl)
        self.cmb_flag.currentIndexChanged.connect(self._on_flag); clr.addWidget(self.cmb_flag)
        rv.addLayout(clr)

        self.checks = []
        grid = QtWidgets.QGridLayout()
        for j, nm in enumerate(self.b.attr_names):
            cb = QtWidgets.QCheckBox(nm); cb.stateChanged.connect(lambda _s, jj=j: self._on_check(jj))
            self.checks.append(cb); grid.addWidget(cb, j % 8, j // 8)
        rv.addLayout(grid)
        self.lbl_conf = QtWidgets.QLabel(""); self.lbl_conf.setStyleSheet("color:#888"); rv.addWidget(self.lbl_conf)

        br = QtWidgets.QHBoxLayout()
        for txt, fn in [("Accept all", lambda: self._bulk(1)), ("Reject all", lambda: self._bulk(0)),
                        ("Reset to pipeline", self._reset)]:
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); br.addWidget(b)
        rv.addLayout(br)
        bl = QtWidgets.QHBoxLayout(); bl.addWidget(QtWidgets.QLabel("Bulk set"))
        self.cb_bulk = QtWidgets.QComboBox()
        for j, nm in enumerate(self.b.attr_names):
            self.cb_bulk.addItem(nm, j)
        bl.addWidget(self.cb_bulk, 1)
        b_on = QtWidgets.QPushButton("ON ⭢ sel"); b_on.clicked.connect(lambda: self._bulk_selected(1))
        b_off = QtWidgets.QPushButton("OFF ⭢ sel"); b_off.clicked.connect(lambda: self._bulk_selected(0))
        bl.addWidget(b_on); bl.addWidget(b_off); rv.addLayout(bl)
        rv.addWidget(QtWidgets.QLabel("Note"))
        self.ed_note = QtWidgets.QLineEdit(); self.ed_note.editingFinished.connect(self._on_note); rv.addWidget(self.ed_note)
        nav = QtWidgets.QPushButton("Confirm ✓ & Next ›")
        nav.setToolTip("Mark all selected instances reviewed, then go to the next unreviewed")
        nav.clicked.connect(self._confirm_next); rv.addWidget(nav)
        rv.addStretch(1)
        split.addWidget(right)
        split.setSizes([280, 740, 300])

        # Ctrl+Z undoes the last segmentation edit
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self, activated=self._undo_edit)

    def _set_pan_on_right(self):
        try:                                          # PyVista >= 0.43
            self.plotter.enable_custom_trackball_style(
                left="rotate", middle="pan", right="pan")
            return
        except Exception:
            pass
        try:                                          # fallback: VTK subclass
            import vtk

            class _RightPan(vtk.vtkInteractorStyleTrackballCamera):
                def OnRightButtonDown(self):
                    self.StartPan()

                def OnRightButtonUp(self):
                    self.EndPan()

            self.plotter.iren.interactor.SetInteractorStyle(_RightPan())
        except Exception as exc:
            print("could not remap right-drag to pan:", exc)

    def _add_context(self):
        import pyvista as pv
        self.plotter.add_mesh(pv.PolyData(np.asarray(self.b.context)), color="lightgray",
                              point_size=1.0, render_points_as_spheres=False, name="context")
        self.plotter.reset_camera()

    # ── colour-by (whole-building diagnostic) ────────────────────────────
    def _context_scalars(self, kind, j):
        present = set(self.em.order)
        n_orig = len(self.b.ids)
        if kind == "val":
            per = np.full(n_orig, -1.0)
            for k, iid in enumerate(self.b.ids):
                if iid in present and self._appl(iid, j):
                    per[k] = self._attr_val(iid, j)
        else:
            per = np.array([self._conf(iid, j) if iid in present else np.nan
                            for iid in self.b.ids], float)
            per = np.where(per < 0, np.nan, per)
        lut = np.full(self.b.max_id + 1, np.nan)
        lut[np.array(self.b.ids)] = per
        scal = lut[np.clip(self.b.context_inst, 0, self.b.max_id)]
        scal[self.b.context_inst < 0] = np.nan
        return scal

    def _apply_colorby(self):
        import pyvista as pv
        ctx = pv.PolyData(np.asarray(self.b.context))
        kind, j = self.cb_color.currentData()
        if self.b.context_inst is None or kind == "hl":
            self.plotter.add_mesh(ctx, color="lightgray", point_size=1.0,
                                  render_points_as_spheres=False, name="context", reset_camera=False)
        elif kind == "val":
            self.plotter.add_mesh(ctx, scalars=self._context_scalars("val", j), name="context",
                                  cmap=["#3a3a3a", "#9aa0a6", "#00c853"], clim=[-1, 1], n_colors=3,
                                  nan_color="#202020", point_size=1.6, render_points_as_spheres=False,
                                  show_scalar_bar=False, reset_camera=False)
        else:
            self.plotter.add_mesh(ctx, scalars=self._context_scalars("conf", j), name="context",
                                  cmap="coolwarm", clim=[0.0, 1.0], nan_color="#202020",
                                  point_size=1.6, render_points_as_spheres=False,
                                  scalar_bar_args={"title": f"conf: {self.b.attr_names[j]}"},
                                  reset_camera=False)
        self.plotter.render()

    def _recolor_if_active(self, edited_j=None):
        kind, j = self.cb_color.currentData()
        if kind in ("val", "conf") and (edited_j is None or edited_j == j):
            self._apply_colorby()

    # ── list ─────────────────────────────────────────────────────────────
    def _visible(self):
        cls = self.cb_class.currentData(); q = self.ed_search.text().lower().strip()
        aj = self.cb_attr.currentData(); astate = self.cb_attr_state.currentData()
        out = []
        for iid in self.em.order:
            if cls and self._cls(iid) != cls:
                continue
            if aj is not None and aj >= 0 and self._attr_val(iid, aj) != astate:
                continue
            if self.cb_unrev.isChecked() and self.review.is_reviewed(iid):
                continue
            if self.cb_flag.isChecked() and not (self.review.records.get(iid) or {}).get("instance_flag"):
                continue
            if self.cb_changed.isChecked() and not self._changed(iid):
                continue
            if q and q not in str(iid) and q not in self._cls(iid).lower():
                continue
            out.append(iid)
        return out

    def _row_text(self, iid):
        mark = "✓ " if self.review.is_reviewed(iid) else "   "
        star = "✎" if iid in self.em.edited else " "
        flagged = bool((self.review.records.get(iid) or {}).get("instance_flag"))
        flag = "⚑" if flagged else " "
        return f"{mark}{star}{flag} #{iid}  {self._cls(iid)}  ({self._count(iid):,} pts)"

    def _refresh_list(self):
        self._loading = True
        self.list.clear()
        for iid in self._visible():
            it = QtWidgets.QListWidgetItem(self._row_text(iid))
            it.setData(QtCore.Qt.UserRole, iid); self.list.addItem(it)
        self._loading = False

    def _on_current(self, row):
        if self._loading or row < 0:
            return
        self.select(int(self.list.item(row).data(QtCore.Qt.UserRole)))

    def _selected_ids(self):
        # keep only ids that still exist (a merge/undo can leave stale selection)
        return [i for i in (int(it.data(QtCore.Qt.UserRole)) for it in self.list.selectedItems())
                if i in self.em.rows]

    # ── editor (one instance) ────────────────────────────────────────────
    def select(self, iid):
        self.cur = iid
        rec = self.review.get(iid)
        total = self._count(iid)
        self._loading = True
        nsel = len(self.list.selectedItems())
        selnote = f"    ·    {nsel} selected (bulk-set applies to all)" if nsel > 1 else ""
        edited = "  ✎ edited" if iid in self.em.edited else ""
        self.lbl_header.setText(f"Instance {iid} · {self._cls(iid)} · {total:,} pts{selnote}{edited}")
        self.cmb_class.setCurrentIndex(max(0, self.cmb_class.findData(self._cls(iid))))
        self.cmb_flag.setCurrentIndex(max(0, self.cmb_flag.findData(rec["instance_flag"])))
        self.ed_note.setText(rec["note"])
        confs = []
        for j, cb in enumerate(self.checks):
            cb.setChecked(bool(rec["vector_human"][j]))
            appl = self._appl(iid, j); cv = self._conf(iid, j)
            cb.setStyleSheet("" if appl else "color:#999; font-style:italic;")
            confs.append(f"{self.b.attr_names[j]}={cv:.2f}" if appl and np.isfinite(cv) else "")
        self.lbl_conf.setText("pipeline conf:  " + "  ".join(c for c in confs if c))
        self.b_merge.setEnabled(self.b.editable and nsel >= 2)
        self._loading = False

    # ── 3D highlight (all selected instances) ────────────────────────────
    def _highlight_selected(self):
        import pyvista as pv
        if self._loading:
            return
        ids = self._selected_ids()
        self.b_merge.setEnabled(self.b.editable and len(ids) >= 2)
        if not ids:
            try:
                self.plotter.remove_actor("hl")
            except Exception:
                pass
            self.plotter.render(); return
        pts = np.concatenate([self._inst_points(i) for i in ids])
        if len(pts) > HL_DISPLAY_CAP:
            pts = pts[::len(pts) // HL_DISPLAY_CAP + 1]
        self.plotter.add_mesh(pv.PolyData(pts), color="red", point_size=4.0,
                              render_points_as_spheres=False, name="hl", reset_camera=False)
        if self.cb_follow.isChecked():
            segs = [self.em.stats(i)[1] for i in ids]
            mn = np.min([s[:3] for s in segs], 0); mx = np.max([s[3:] for s in segs], 0)
            self.plotter.reset_camera(bounds=[mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]])
        self.plotter.render()

    def _bulk_selected(self, val):
        j = self.cb_bulk.currentData()
        ids = self._selected_ids()
        if j is None or not ids:
            return
        for iid in ids:
            self.review.get(iid)["vector_human"][j] = val
            self.review.mark(iid, True)
            self._update_row(iid)
        self.review.dirty = True
        if self.cur in ids:
            self.select(self.cur)
        self._recolor_if_active(j); self._refresh_counter()

    # ── edits: attributes ────────────────────────────────────────────────
    def _touch(self):
        if self.cur is not None and not self.review.is_reviewed(self.cur):
            self.review.mark(self.cur, True); self._update_row(self.cur)

    def _on_check(self, j):
        if self._loading or self.cur is None:
            return
        self.review.get(self.cur)["vector_human"][j] = 1 if self.checks[j].isChecked() else 0
        self.review.dirty = True; self._touch(); self._recolor_if_active(j); self._refresh_counter()

    def _bulk(self, v):
        if self.cur is None:
            return
        rec = self.review.get(self.cur)
        for j in range(self.b.n_attr):
            rec["vector_human"][j] = v
        self.review.dirty = True; self._touch(); self.select(self.cur)
        self._recolor_if_active(); self._refresh_counter()

    def _reset(self):
        if self.cur is None:
            return
        rec = self.review.get(self.cur)
        for j in range(self.b.n_attr):
            rec["vector_human"][j] = self._pipe(self.cur, j)
        self.review.dirty = True; self._touch(); self.select(self.cur)
        self._recolor_if_active(); self._refresh_counter()

    def _on_flag(self):
        if self._loading or self.cur is None:
            return
        self.review.get(self.cur)["instance_flag"] = self.cmb_flag.currentData()
        self.review.dirty = True; self._touch()

    def _on_note(self):
        if self._loading or self.cur is None:
            return
        self.review.get(self.cur)["note"] = self.ed_note.text(); self.review.dirty = True

    # ── edits: segmentation (reclass / merge / split) ────────────────────
    def _on_reclass(self):
        if self._loading or self.cur is None or not self.b.editable:
            return
        new = self.cmb_class.currentData()
        new_sem = SEM_NAMES.index(new) if new in SEM_NAMES else None
        if new_sem is None or self.em.sem.get(self.cur) == new_sem:
            return
        self.em.reclass(self.cur, new_sem)
        self.review.dirty = True; self._touch()
        self._update_row(self.cur); self.select(self.cur); self._refresh_counter()

    def _merge_selected(self):
        if not self.b.editable:
            return
        ids = self._selected_ids()
        if len(ids) < 2:
            return
        survivor = self.em.merge(ids)
        self.review.dirty = True
        # collapse review records onto the survivor (keep survivor's; drop others)
        for i in ids:
            if i != survivor:
                self.review.records.pop(i, None)
        self.review.mark(survivor, True)
        self._refresh_list(); self._refresh_counter()
        self._goto(survivor)

    def _toggle_lasso(self, on):
        if on and (not self.b.editable or self.cur is None):
            self.b_lasso.setChecked(False)
            QtWidgets.QMessageBox.information(self, "Lasso split",
                                              "Select a single instance first, then draw a "
                                              "polygon around the part to cut off.")
            return
        if on:
            self._start_lasso()
        else:
            self._end_lasso()

    def _start_lasso(self):
        # CloudCompare-style polygon: left-CLICK to drop each vertex, with a
        # rubber-band preview to the cursor; right-click / double-click / Enter
        # closes it and cuts; Esc cancels. Captured at the Qt level (see
        # eventFilter) since VTK observers don't fire reliably under pyvista.
        self._lasso_active = True
        self._lasso_pts = []                          # committed vertices
        self._lasso_cursor = None                     # live cursor (preview)
        self._make_lasso_overlay()
        self.plotter.interactor.setMouseTracking(True)
        self.plotter.interactor.setFocus()
        self.plotter.add_text("POLYGON: left-click to add points · right-click / double-click "
                              "to close & cut · Esc to cancel",
                              name="lasso_hint", color="#00c853", font_size=10)
        self._lasso_render()

    def _vtk_pos(self, event):
        """Qt widget position (top-left origin, logical px) -> VTK display
        position (bottom-left origin, device px), matching _project_to_display."""
        dpr = self.plotter.interactor.devicePixelRatioF()
        _w, h = self.plotter.render_window.GetSize()
        p = event.position()
        return (p.x() * dpr, h - p.y() * dpr)

    def eventFilter(self, obj, event):
        if not self._lasso_active or obj is not self.plotter.interactor:
            return super().eventFilter(obj, event)
        et = event.type()
        E = QtCore.QEvent.Type
        Btn = QtCore.Qt.MouseButton
        Key = QtCore.Qt.Key
        if et == E.MouseButtonDblClick and event.button() == Btn.LeftButton:
            self._finalize_lasso(); return True
        if et == E.MouseButtonPress:
            if event.button() == Btn.LeftButton:
                self._lasso_pts.append(self._vtk_pos(event)); self._update_lasso_overlay()
            elif event.button() == Btn.RightButton:
                self._finalize_lasso()
            return True
        if et == E.MouseMove:
            self._lasso_cursor = self._vtk_pos(event)
            if self._lasso_pts:                        # only preview after the 1st vertex
                self._update_lasso_overlay()
            return True
        if et == E.KeyPress:
            if event.key() == Key.Key_Escape:
                self._end_lasso()
            elif event.key() in (Key.Key_Return, Key.Key_Enter):
                self._finalize_lasso()
            return True
        if et in (E.MouseButtonRelease, E.Wheel):
            return True                                # swallow while lassoing
        return super().eventFilter(obj, event)

    def _finalize_lasso(self):
        poly = np.array(self._lasso_pts, float)
        try:
            if len(poly) >= 3 and self.cur is not None:
                self._apply_lasso_split(poly)
        finally:
            self._end_lasso()

    def _make_lasso_overlay(self):
        import vtk
        self._lasso_vpts = vtk.vtkPoints()
        self._lasso_lines = vtk.vtkCellArray()
        self._lasso_verts = vtk.vtkCellArray()
        self._lasso_pd = vtk.vtkPolyData()
        self._lasso_pd.SetPoints(self._lasso_vpts)
        self._lasso_pd.SetLines(self._lasso_lines)
        self._lasso_pd.SetVerts(self._lasso_verts)
        mapper = vtk.vtkPolyDataMapper2D(); mapper.SetInputData(self._lasso_pd)
        coord = vtk.vtkCoordinate(); coord.SetCoordinateSystemToDisplay()
        mapper.SetTransformCoordinate(coord)
        actor = vtk.vtkActor2D(); actor.SetMapper(mapper)
        pr = actor.GetProperty()
        pr.SetColor(0.0, 0.784, 0.325); pr.SetLineWidth(2); pr.SetPointSize(7)
        self._lasso_actor = actor
        self.plotter.renderer.AddViewProp(actor)

    def _lasso_render(self):
        """Immediate, non-threaded redraw — safe to call from a Qt/VTK callback
        (pyvistaqt's plotter.render() is threaded and re-enters/hangs here)."""
        rw = getattr(self.plotter, "render_window", None)
        (rw.Render() if rw is not None else self.plotter.render())

    def _update_lasso_overlay(self):
        verts = list(self._lasso_pts)
        loop = list(self._lasso_pts)
        if self._lasso_cursor is not None and loop:    # rubber-band to the cursor
            loop = loop + [self._lasso_cursor]
        self._lasso_vpts.Reset(); self._lasso_lines.Reset(); self._lasso_verts.Reset()
        for (x, y) in loop:
            self._lasso_vpts.InsertNextPoint(x, y, 0)
        n = len(loop)
        if n >= 2:                                     # closed preview loop
            self._lasso_lines.InsertNextCell(n + 1)
            for i in range(n):
                self._lasso_lines.InsertCellPoint(i)
            self._lasso_lines.InsertCellPoint(0)
        for i in range(len(verts)):                    # dots on committed vertices
            self._lasso_verts.InsertNextCell(1); self._lasso_verts.InsertCellPoint(i)
        self._lasso_vpts.Modified(); self._lasso_pd.Modified()
        self._lasso_render()

    def _end_lasso(self):
        if getattr(self, "_ending_lasso", False):
            return
        self._ending_lasso = True
        try:
            if getattr(self, "_lasso_actor", None) is not None:
                try:
                    self.plotter.renderer.RemoveViewProp(self._lasso_actor)
                except Exception:
                    pass
                self._lasso_actor = None
            try:
                self.plotter.remove_actor("lasso_hint")
            except Exception:
                pass
            self._lasso_active = False
            if self.b_lasso.isChecked():
                self.b_lasso.setChecked(False)          # re-enters _toggle_lasso(False)
            self._lasso_render()
        finally:
            self._ending_lasso = False

    def _project_to_display(self, pts3d):
        """World xyz -> display pixel coords (origin lower-left), matching VTK's
        GetEventPosition() so the drawn polygon and the points share a frame."""
        ren = self.plotter.renderer
        w, h = self.plotter.render_window.GetSize()
        cam = ren.GetActiveCamera()
        aspect = (w / h) if h else 1.0
        m = cam.GetCompositeProjectionTransformMatrix(aspect, -1, 1)
        M = np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)])
        P = np.column_stack([pts3d, np.ones(len(pts3d))])
        clip = P @ M.T
        wv = clip[:, 3:4]
        wv[wv == 0] = 1e-9
        ndc = clip[:, :3] / wv
        dx = (ndc[:, 0] * 0.5 + 0.5) * w
        dy = (ndc[:, 1] * 0.5 + 0.5) * h
        return dx, dy, ndc[:, 2]

    def _apply_lasso_split(self, poly):
        iid = self.cur
        rows = self.em.rows[iid]
        pts3d = np.asarray(self.b.points[rows])          # FULL RES — not the display cap
        dx, dy, ndz = self._project_to_display(pts3d)
        inside = _points_in_poly(dx, dy, poly[:, 0], poly[:, 1])
        # only points actually in front of the camera (ndc z in [-1,1])
        inside &= (ndz >= -1.0) & (ndz <= 1.0)
        nsel = int(inside.sum())
        if nsel == 0 or nsel == len(rows):
            QtWidgets.QMessageBox.information(self, "Lasso split",
                                              f"Selected {nsel:,} of {len(rows):,} points — "
                                              "need a strict subset. Nothing changed.")
            return
        # highlight exactly what will be cut, so the selection can be eyeballed
        # against the polygon before committing
        self._show_lasso_selection(pts3d[inside])
        cur_cls = self._cls(iid)
        default_idx = SEM_NAMES.index(cur_cls) if cur_cls in SEM_NAMES else 0
        try:
            child_cls, ok = QtWidgets.QInputDialog.getItem(
                self, "Split → new instance",
                f"{nsel:,} points will move to a new instance.\nClass for the new instance:",
                SEM_NAMES, default_idx, False)
        finally:
            self._clear_lasso_selection()
        if not ok:
            return
        child = self.em.split(iid, rows[inside], SEM_NAMES.index(child_cls))
        if child is None:
            return
        self.review.dirty = True
        self.review.mark(iid, True); self.review.mark(child, True)
        self._refresh_list(); self._refresh_counter(); self._goto(child)

    def _show_lasso_selection(self, sel_pts):
        import pyvista as pv
        if len(sel_pts) > HL_DISPLAY_CAP:
            sel_pts = sel_pts[::len(sel_pts) // HL_DISPLAY_CAP + 1]
        self.plotter.add_mesh(pv.PolyData(np.asarray(sel_pts)), color="#ffeb3b",
                              point_size=6.0, render_points_as_spheres=False,
                              name="lasso_sel", reset_camera=False)
        self._lasso_render()

    def _clear_lasso_selection(self):
        try:
            self.plotter.remove_actor("lasso_sel")
        except Exception:
            pass
        self._lasso_render()

    def _undo_edit(self):
        if not self.em.can_undo():
            return
        op = self.em.undo()
        self.review.dirty = True
        if self.cur is not None and self.cur not in self.em.rows:
            self.cur = None                          # current instance was undone away
        self._refresh_list(); self._refresh_counter()
        if op:
            self.statusBar().showMessage(f"Undid {op.get('op')}", 3000)

    def _confirm_next(self):
        ids = self._selected_ids() or ([self.cur] if self.cur is not None else [])
        for iid in ids:
            self.review.mark(iid, True); self._update_row(iid)
        self._refresh_counter()
        vis = self._visible()
        if not vis:
            return
        nxt = next((i for i in vis if not self.review.is_reviewed(i)), None)
        if nxt is None:
            anchor = self.cur if self.cur in vis else (ids[-1] if ids and ids[-1] in vis else vis[-1])
            nxt = vis[(vis.index(anchor) + 1) % len(vis)]
        self._goto(nxt)

    def _goto(self, iid):
        for r in range(self.list.count()):
            if int(self.list.item(r).data(QtCore.Qt.UserRole)) == iid:
                self.list.setCurrentRow(r)
                return

    def _update_row(self, iid):
        for r in range(self.list.count()):
            if int(self.list.item(r).data(QtCore.Qt.UserRole)) == iid:
                self.list.item(r).setText(self._row_text(iid))
                return

    def _refresh_counter(self):
        n_edit = len(self.em.edited)
        extra = f"   • {n_edit} edited" if n_edit else ""
        self.lbl_counter.setText(f"{self.review.n_reviewed()} / {len(self.em.order)} reviewed"
                                 + extra + ("   • unsaved" if self.review.dirty else ""))
        if hasattr(self, "b_undo"):
            self.b_undo.setEnabled(self.em.can_undo())

    def _change_saveto(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Save reviews to…",
                                                       str(self.review.review_root))
        if not d:
            return
        self.review.review_root = Path(d); self.review.export_root = Path(d)
        self.lbl_saveto.setText(d); self._persist_settings()

    def _persist_settings(self):
        _CFG["reviewer"] = self.ed_rev.text().strip()
        _CFG["review_root"] = str(self.review.review_root)
        _CFG["export_root"] = str(self.review.export_root)
        _save_cfg(_CFG)

    def _start_update_check(self):
        """Warn (small status-bar note) if this checkout is behind origin/main."""
        repo = APP_DIR.parent
        if not (repo / ".git").exists():
            return
        self.lbl_update = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.lbl_update)
        self._upd = _UpdateChecker(repo)
        self._upd.result.connect(self._on_update_result)
        self._upd.start()

    def _on_update_result(self, behind):
        if behind and hasattr(self, "lbl_update"):
            self.lbl_update.setText("  ⚠ App update available — close and run: git pull  ")
            self.lbl_update.setStyleSheet("color:#b26a00; font-weight:600;")
            self.lbl_update.setToolTip("A newer version is on the server. Save your work, "
                                       "close the app, run 'git pull', then reopen.")

    # ── save / export ────────────────────────────────────────────────────
    def _save_json(self):
        b = self.b
        self.review.path = self.review._json_path()
        self.review.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"building": b.building, "split": "", "reviewer": self.review.reviewer,
                   "updated_at": datetime.now().isoformat(timespec="seconds"),
                   "attribute_names": b.attr_names,
                   "edits": self.em.log,
                   "instances": {}}
        for iid in self.em.order:
            r = self.review.records.get(iid)
            if r is None:
                continue
            payload["instances"][str(iid)] = {
                "status": r["status"], "class": self._cls(iid),
                "vector_human": r["vector_human"],
                "changed": [j for j in range(b.n_attr)
                            if r["vector_human"][j] != self._pipe(iid, j)],
                "instance_flag": r["instance_flag"], "note": r["note"]}
        self.review.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.review.dirty = False
        return self.review.path

    def _export_h5(self):
        if h5py is None:
            raise RuntimeError("h5py not installed")
        b = self.b
        p = self.review.export_root / f"{b.building}__{_safe(self.review.reviewer)}.reviewed.h5"
        p.parent.mkdir(parents=True, exist_ok=True)
        order = self.em.order
        n, K = len(order), b.n_attr
        appl = np.zeros((n, K), np.uint8); conf = np.full((n, K), np.nan, np.float32)
        pipe = np.zeros((n, K), np.uint8); hum = np.zeros((n, K), np.uint8)
        status = np.empty(n, "S12"); flag = np.empty(n, "S20")
        note = np.empty(n, dtype=h5py.string_dtype())
        ids = np.zeros(n, np.int64); sem = np.zeros(n, np.int32)
        pur = np.zeros(n, np.float32); cnt = np.zeros(n, np.int64)
        clsname = []
        for i, iid in enumerate(order):
            r = self.review.records.get(iid)
            ids[i] = iid; sem[i] = self.em.sem[iid]; cnt[i] = self._count(iid)
            clsname.append(self._cls(iid))
            k = self._k(iid)
            # purity is only meaningful for an unedited original instance
            pur[i] = float(b.purity[k]) if (k is not None and iid not in self.em.edited) else np.nan
            for j in range(K):
                appl[i, j] = int(self._appl(iid, j)); conf[i, j] = self._conf(iid, j)
                pipe[i, j] = self._pipe(iid, j)
            hum[i] = (np.array(r["vector_human"], np.uint8) if r else pipe[i])
            status[i] = (r["status"] if r else "unreviewed").encode()
            flag[i] = (r["instance_flag"] if r else "").encode()
            note[i] = r["note"] if r else ""
        with h5py.File(p, "w") as f:
            m = f.create_group("metadata")
            m.create_dataset("attribute_names", data=np.array(b.attr_names, dtype=h5py.string_dtype()))
            m.create_dataset("attribute_count", data=np.int32(K))
            m.create_dataset("reviewer", data=self.review.reviewer)
            m.create_dataset("building", data=b.building)
            m.create_dataset("exported_at", data=datetime.now().isoformat(timespec="seconds"))
            m.create_dataset("n_reviewed", data=np.int32(self.review.n_reviewed()))
            m.create_dataset("n_edits", data=np.int32(len(self.em.log)))
            g = f.create_group("instances")
            g.create_dataset("instance_id", data=ids)
            g.create_dataset("semantic_id", data=sem)
            g.create_dataset("semantic_class", data=np.array(clsname, dtype=h5py.string_dtype()))
            g.create_dataset("semantic_purity", data=pur)
            g.create_dataset("point_count", data=cnt)
            g.create_dataset("applicable_mask", data=appl)
            g.create_dataset("functional_attribute_confidence", data=conf)
            g.create_dataset("functional_attribute_vector", data=pipe)
            g.create_dataset("functional_attribute_vector_human", data=hum)
            g.create_dataset("review_status", data=status)
            g.create_dataset("instance_flag", data=flag)
            g.create_dataset("review_note", data=note)
        return p

    def _save(self):
        name = self.ed_rev.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Reviewer name",
                                          "Enter your name in the Reviewer box first — it's "
                                          "stamped into your files so reviews don't collide.")
            return
        self.review.reviewer = name
        target = self.review._json_path()
        lf = self.review.loaded_from
        is_our_file = lf is not None and lf.exists() and target.resolve() == lf.resolve()
        if target.exists() and not is_our_file:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            box.setWindowTitle("A review file already exists here")
            box.setText(f"{target}\n\nalready exists and was NOT loaded this session.")
            box.setInformativeText(
                "Overwriting it will discard whatever review work is already in that "
                "file. You can instead merge (keep both — your current edits win on "
                "any instance you both changed).")
            b_merge = box.addButton("Merge (keep both)", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            b_over = box.addButton("Overwrite", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(b_merge)
            box.exec()
            clicked = box.clickedButton()
            if clicked is b_merge:
                added = self.review.merge_from(target)
                print(f"merged {added} instances from existing {target}")
            elif clicked is not b_over:
                return
        self._persist_settings()
        p = self._save_json(); self.review.loaded_from = p
        self._refresh_counter()
        msg = f"Saved:\n{p}"
        # v2 segmentation-edit sidecars (small; reload state + provenance).
        # The heavy corrected .laz is written on demand via _export_corrected_cloud.
        if self.b.editable and self.em.is_dirty():
            try:
                npz = self._edits_npz_path(); self.em.save_edits(npz); msg += f"\n{npz}"
                ej = self.review.review_root / f"{self.b.building}__{_safe(self.review.reviewer)}.edits.json"
                self.em.write_edit_log(ej, self.b.building, self.review.reviewer); msg += f"\n{ej}"
            except Exception as exc:
                msg += f"\n\n(edit sidecars skipped: {exc})"
        try:
            h5 = self._export_h5(); msg += f"\n{h5}"
        except Exception as exc:
            msg += f"\n\n(.h5 export skipped: {exc})"
        if self.b.editable and self.em.is_dirty():
            msg += "\n\n(Corrected .laz not written — use “Export corrected .laz” when you need the point cloud.)"
        QtWidgets.QMessageBox.information(self, "Saved", msg)

    def _export_corrected_cloud(self):
        name = self.ed_rev.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Reviewer name",
                                          "Enter your name first — it's stamped into the filename.")
            return
        self.review.reviewer = name
        if not self.b.editable:
            return
        if not self.em.is_dirty():
            QtWidgets.QMessageBox.information(self, "Export corrected cloud",
                                              "No segmentation edits yet — the corrected cloud "
                                              "would be identical to the source.")
            return
        src_laz = _find_laz(self.b.building)
        if src_laz is None:
            QtWidgets.QMessageBox.warning(self, "Export corrected cloud",
                                          "Source cloud not found — set HFX3D_CLOUDS_ROOT or put "
                                          "it under review_clouds/.")
            return
        out = self.review.export_root / f"{self.b.building}__{_safe(self.review.reviewer)}.corrected.laz"
        self._persist_settings()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.em.export_corrected_laz(src_laz, out)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.information(self, "Exported", f"Corrected point cloud:\n{out}")


def _points_in_poly(px, py, vx, vy):
    """Vectorised even-odd point-in-polygon. px,py: point coords; vx,vy: polygon
    vertices. Returns a boolean mask over the points."""
    px = np.asarray(px); py = np.asarray(py)
    n = len(vx)
    inside = np.zeros(len(px), bool)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = vx[i], vy[i], vx[j], vy[j]
        cond = ((yi > py) != (yj > py))
        denom = (yj - yi)
        denom = denom if denom != 0 else 1e-12
        xint = (xj - xi) * (py - yi) / denom + xi
        inside ^= cond & (px < xint)
        j = i
    return inside


APP_DIR = Path(__file__).resolve().parent
BUNDLES_DIR = Path(os.environ.get("HFX3D_BUNDLES_ROOT", "").strip() or (APP_DIR.parent / "bundles"))
BUNDLE_FILES = ("meta.npz", "points.npy", "offsets.npy", "context.npy",
                "context_inst.npy", "src_index.npy")   # src_index => v2-editable


def _bundle_ready(folder: Path) -> bool:
    return all((folder / f).exists() for f in BUNDLE_FILES)   # current-schema check


def _find_laz(name: str):
    p = Path(name)
    if p.suffix.lower() in (".laz", ".las") and p.exists():
        return p
    roots = []
    v = os.environ.get("HFX3D_CLOUDS_ROOT", "").strip()
    if v:
        roots.append(Path(v))
    roots.append(APP_DIR.parent / "review_clouds")
    for r in roots:
        if not r.exists():
            continue
        for pat in (f"{name}.laz", f"{name}.las", f"*{name}*.laz", f"*{name}*.las"):
            hit = next(iter(sorted(r.rglob(pat))), None)
            if hit:
                return hit
    return None


class _UpdateChecker(QtCore.QThread):
    """Quietly checks whether the local checkout is behind origin/main and, if
    so, tells the reviewer to pull. Runs off the UI thread; fails silent when
    git is missing, offline, or this isn't a git checkout."""
    result = QtCore.Signal(bool)

    def __init__(self, repo):
        super().__init__(); self.repo = str(repo)

    def run(self):
        try:
            import subprocess

            def g(*a):
                return subprocess.run(["git", "-C", self.repo, *a],
                                      capture_output=True, text=True, timeout=15)
            g("fetch", "--quiet")
            head = g("rev-parse", "HEAD").stdout.strip()
            rem = g("rev-parse", "origin/main").stdout.strip()
            if not head or not rem or head == rem:
                self.result.emit(False); return
            # behind == our HEAD is an ancestor of origin/main
            anc = g("merge-base", "--is-ancestor", "HEAD", "origin/main")
            self.result.emit(anc.returncode == 0)
        except Exception:
            self.result.emit(False)


class _BuildWorker(QtCore.QThread):
    progress = QtCore.Signal(str)
    ok = QtCore.Signal(str)
    fail = QtCore.Signal(str)

    def __init__(self, laz, out):
        super().__init__(); self.laz = laz; self.out = out

    def run(self):
        try:
            import build_bundle
            voxel = float(os.environ.get("HFX3D_CONTEXT_VOXEL", "").strip() or 0.05)
            build_bundle.build(Path(self.laz), Path(self.out), voxel, progress=self.progress.emit)
            self.ok.emit(str(self.out))
        except Exception as exc:
            import traceback; traceback.print_exc(); self.fail.emit(str(exc))


def _stage_seed(laz: Path, out: Path):
    """Copy a sibling part-1 seed (<name>.part1.json) into the bundle so the
    review app can pre-fill flags/notes/vectors from validation part one."""
    seed = laz.parent / (laz.stem + ".part1.json")
    if seed.exists():
        try:
            shutil.copyfile(seed, out / "seed.review.json")
        except Exception as exc:
            print("could not stage part-1 seed:", exc)


def _ensure_bundle(laz: Path):
    out = BUNDLES_DIR / laz.stem
    if _bundle_ready(out):
        _stage_seed(laz, out)
        return out
    dlg = QtWidgets.QProgressDialog(f"Preparing {laz.stem} …", None, 0, 0)
    dlg.setWindowTitle("Building bundle (one-time)")
    dlg.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    dlg.setMinimumWidth(440); dlg.setAutoClose(False); dlg.setAutoReset(False)
    dlg.show()
    worker = _BuildWorker(laz, out)
    state = {}
    worker.progress.connect(lambda s: dlg.setLabelText(f"Preparing {laz.stem} (one-time)…\n\n{s}"))
    worker.ok.connect(lambda o: state.__setitem__("out", o))
    worker.fail.connect(lambda e: state.__setitem__("err", e))
    loop = QtCore.QEventLoop()
    worker.ok.connect(loop.quit); worker.fail.connect(loop.quit)
    worker.start(); loop.exec(); worker.wait()
    dlg.close()
    if "err" in state:
        QtWidgets.QMessageBox.critical(None, "Build failed",
                                       f"Could not build bundle for {laz.stem}:\n{state['err']}")
        return None
    _stage_seed(laz, out)
    return Path(state["out"])


def _resolve(arg):
    if arg:
        p = Path(arg)
        if p.is_dir() and (p / "meta.npz").exists():
            return p
        laz = _find_laz(arg)
        if laz is None:
            QtWidgets.QMessageBox.critical(None, "Functional Review",
                                           f"No .laz found for '{arg}'. Pass a building name, a "
                                           ".laz path, or a bundle folder.")
            return None
        return _ensure_bundle(laz)
    f, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Open a review .laz (or a bundle's meta.npz)", str(APP_DIR.parent),
        "Review cloud / bundle (*.laz *.las meta.npz)")
    if not f:
        return None
    fp = Path(f)
    return fp.parent if fp.name == "meta.npz" else _ensure_bundle(fp)


def main():
    app = QtWidgets.QApplication(sys.argv)
    folder = _resolve(sys.argv[1] if len(sys.argv) > 1 else None)
    if not folder or not (folder / "meta.npz").exists():
        return
    win = MainWindow(Bundle(folder))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
