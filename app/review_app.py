#!/usr/bin/env python3
"""Standalone functional-attribute review app (PySide6 + PyVista).

Reviews the per-instance functional attributes of a building point cloud WITHOUT
CloudCompare. It opens a "bundle" produced by build_bundle.py: the whole building
is drawn once as a decimated grey backdrop, and selecting an instance just swaps a
small red highlight actor (its points are an O(1) memmap slice) — so browsing ~300
instances is instant, even on an integrated GPU.

Output is identical to the CloudCompare plugin, so uploads/workflow are unchanged:
    <HFX3D_REVIEW_ROOT>/<building>__<reviewer>.review.json
    <HFX3D_EXPORT_ROOT>/<building>__<reviewer>.reviewed.h5

    python app/review_app.py bundles/HFX_BLD001        # or launch and File->Open

Env (same as the plugin): HFX3D_REVIEWER, HFX3D_REVIEW_ROOT, HFX3D_EXPORT_ROOT.
Requires: pip install pyside6 pyvista pyvistaqt numpy h5py
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

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
    """Loads a build_bundle.py folder; per-instance attribute access."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.context = np.load(self.folder / "context.npy", mmap_mode="r")
        self.points = np.load(self.folder / "points.npy", mmap_mode="r")
        self.offsets = np.load(self.folder / "offsets.npy")
        cif = self.folder / "context_inst.npy"          # optional (older bundles)
        self.context_inst = np.load(cif) if cif.exists() else None
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

    def instance_points(self, k):
        seg = np.asarray(self.points[self.offsets[k]:self.offsets[k + 1]])
        if len(seg) > HL_DISPLAY_CAP:            # display only — data untouched
            step = len(seg) // HL_DISPLAY_CAP + 1
            seg = seg[::step]
        return seg

    def cls(self, k):
        s = int(self.sem[k])
        return SEM_NAMES[s] if 0 <= s < len(SEM_NAMES) else "?"

    def has_conf(self, j):
        return bool(np.isfinite(self.conf[:, j]).any()) if self.n_attr else False

    def pipeline(self, k, j):
        c = self.conf[k, j]
        return int(c >= 0.5) if np.isfinite(c) else int(self.val[k, j])

    def confval(self, k, j):
        return float(self.conf[k, j]) if self.n_attr else float("nan")

    def applicable(self, k, j):
        c = self.conf[k, j] if self.n_attr else float("nan")
        return not (np.isfinite(c) and c < 0.0)


class ReviewStore:
    """Same JSON/H5 contract as the CloudCompare plugin."""

    def __init__(self, bundle: Bundle, reviewer="", review_root=None, export_root=None):
        self.b = bundle
        self.reviewer = reviewer
        self.review_root = Path(review_root) if review_root else _cfg_review_root()
        self.export_root = Path(export_root) if export_root else _cfg_export_root(self.review_root)
        self.dirty = False
        self.records = {}
        self.path = self._json_path()
        if self.path.exists():
            self._load()

    def _json_path(self):
        return self.review_root / f"{self.b.building}__{_safe(self.reviewer)}.review.json"

    def _default(self, k):
        iid = self.b.ids[k]
        return {"status": "unreviewed",
                "vector_human": [self.b.pipeline(k, j) for j in range(self.b.n_attr)],
                "instance_flag": "", "note": ""}

    def get(self, k):
        iid = self.b.ids[k]
        if iid not in self.records:
            self.records[iid] = self._default(k)
        return self.records[iid]

    def is_reviewed(self, k):
        r = self.records.get(self.b.ids[k])
        return bool(r and r["status"] == "reviewed")

    def mark(self, k, on=True):
        self.get(k)["status"] = "reviewed" if on else "unreviewed"
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

    def save(self):
        b = self.b
        self.path = self._json_path()               # honour current reviewer/root
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"building": b.building, "split": "", "reviewer": self.reviewer,
                   "updated_at": datetime.now().isoformat(timespec="seconds"),
                   "attribute_names": b.attr_names,
                   "instances": {}}
        for k, iid in enumerate(b.ids):
            r = self.records.get(iid)
            if r is None:
                continue
            payload["instances"][str(iid)] = {
                "status": r["status"], "vector_human": r["vector_human"],
                "changed": [j for j in range(b.n_attr)
                            if r["vector_human"][j] != b.pipeline(k, j)],
                "instance_flag": r["instance_flag"], "note": r["note"]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.dirty = False
        return self.path

    def export_h5(self):
        if h5py is None:
            raise RuntimeError("h5py not installed")
        b = self.b
        p = self.export_root / f"{b.building}__{_safe(self.reviewer)}.reviewed.h5"
        p.parent.mkdir(parents=True, exist_ok=True)
        n, K = len(b.ids), b.n_attr
        appl = np.zeros((n, K), np.uint8); conf = np.full((n, K), np.nan, np.float32)
        pipe = np.zeros((n, K), np.uint8); hum = np.zeros((n, K), np.uint8)
        status = np.empty(n, "S12"); flag = np.empty(n, "S20")
        note = np.empty(n, dtype=h5py.string_dtype())
        for k, iid in enumerate(b.ids):
            r = self.records.get(iid)
            for j in range(K):
                appl[k, j] = int(b.applicable(k, j)); conf[k, j] = b.confval(k, j)
                pipe[k, j] = b.pipeline(k, j)
            hum[k] = (np.array(r["vector_human"], np.uint8) if r else pipe[k])
            status[k] = (r["status"] if r else "unreviewed").encode()
            flag[k] = (r["instance_flag"] if r else "").encode()
            note[k] = r["note"] if r else ""
        with h5py.File(p, "w") as f:
            m = f.create_group("metadata")
            m.create_dataset("attribute_names", data=np.array(b.attr_names, dtype=h5py.string_dtype()))
            m.create_dataset("attribute_count", data=np.int32(K))
            m.create_dataset("reviewer", data=self.reviewer)
            m.create_dataset("building", data=b.building)
            m.create_dataset("exported_at", data=datetime.now().isoformat(timespec="seconds"))
            m.create_dataset("n_reviewed", data=np.int32(self.n_reviewed()))
            g = f.create_group("instances")
            g.create_dataset("instance_id", data=np.array(b.ids, np.int64))
            g.create_dataset("semantic_id", data=b.sem.astype(np.int32))
            g.create_dataset("semantic_class", data=np.array([b.cls(k) for k in range(n)], dtype=h5py.string_dtype()))
            g.create_dataset("semantic_purity", data=b.purity.astype(np.float32))
            g.create_dataset("point_count", data=b.counts.astype(np.int64))
            g.create_dataset("applicable_mask", data=appl)
            g.create_dataset("functional_attribute_confidence", data=conf)
            g.create_dataset("functional_attribute_vector", data=pipe)
            g.create_dataset("functional_attribute_vector_human", data=hum)
            g.create_dataset("review_status", data=status)
            g.create_dataset("instance_flag", data=flag)
            g.create_dataset("review_note", data=note)
        return p


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, bundle: Bundle):
        super().__init__()
        self.b = bundle
        self.review = ReviewStore(bundle, _cfg_reviewer())
        self.k = None
        self._loading = False
        self.setWindowTitle(f"Functional Review — {bundle.building}")
        self.resize(1280, 800)
        self._build()
        self._add_context()
        self._refresh_list()
        self._refresh_counter()

    # ── UI ───────────────────────────────────────────────────────────────
    def _build(self):
        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)

        # left: instance list
        left = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(left)
        self.ed_search = QtWidgets.QLineEdit(); self.ed_search.setPlaceholderText("search id / class…")
        self.ed_search.textChanged.connect(self._refresh_list); lv.addWidget(self.ed_search)
        self.cb_class = QtWidgets.QComboBox(); self.cb_class.addItem("all classes", "")
        for c in sorted({self.b.cls(k) for k in range(len(self.b.ids))}):
            self.cb_class.addItem(c, c)
        self.cb_class.currentIndexChanged.connect(self._refresh_list); lv.addWidget(self.cb_class)
        # filter by a specific attribute being on/off
        arow = QtWidgets.QHBoxLayout()
        self.cb_attr = QtWidgets.QComboBox(); self.cb_attr.addItem("(any attribute)", -1)
        for j, nm in enumerate(self.b.attr_names):
            self.cb_attr.addItem(nm, j)
        self.cb_attr.currentIndexChanged.connect(self._refresh_list)
        self.cb_attr_state = QtWidgets.QComboBox()
        self.cb_attr_state.addItem("is on", 1); self.cb_attr_state.addItem("is off", 0)
        self.cb_attr_state.currentIndexChanged.connect(self._refresh_list)
        arow.addWidget(self.cb_attr, 1); arow.addWidget(self.cb_attr_state); lv.addLayout(arow)
        # status filters
        crow = QtWidgets.QHBoxLayout()
        self.cb_unrev = QtWidgets.QCheckBox("unreviewed"); self.cb_unrev.toggled.connect(self._refresh_list)
        self.cb_flag = QtWidgets.QCheckBox("flagged"); self.cb_flag.toggled.connect(self._refresh_list)
        self.cb_changed = QtWidgets.QCheckBox("changed"); self.cb_changed.toggled.connect(self._refresh_list)
        crow.addWidget(self.cb_unrev); crow.addWidget(self.cb_flag); crow.addWidget(self.cb_changed)
        crow.addStretch(1); lv.addLayout(crow)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.currentRowChanged.connect(self._on_current)       # drives the editor
        self.list.itemSelectionChanged.connect(self._highlight_selected)  # drives 3D
        lv.addWidget(self.list, 1)
        b_all = QtWidgets.QPushButton("Select all filtered ⭢ 3D")
        b_all.clicked.connect(self.list.selectAll); lv.addWidget(b_all)
        self.lbl_counter = QtWidgets.QLabel(""); lv.addWidget(self.lbl_counter)
        split.addWidget(left)

        # center: 3D view
        self.plotter = QtInteractor()
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
        self.cb_follow = QtWidgets.QCheckBox("zoom to instance on select"); self.cb_follow.setChecked(True)
        rv.addWidget(self.cb_follow)
        # colour the whole building by an attribute — to spot outliers
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
        fr = QtWidgets.QHBoxLayout(); fr.addWidget(QtWidgets.QLabel("Flag"))
        self.cmb_flag = QtWidgets.QComboBox()
        for fl in FLAGS:
            self.cmb_flag.addItem(fl or "(none)", fl)
        self.cmb_flag.currentIndexChanged.connect(self._on_flag); fr.addWidget(self.cmb_flag); rv.addLayout(fr)

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
        # bulk-assign one attribute across every currently-SELECTED instance
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
        nav = QtWidgets.QPushButton("Confirm ✓ & Next ›"); nav.clicked.connect(self._confirm_next); rv.addWidget(nav)
        rv.addStretch(1)
        split.addWidget(right)
        split.setSizes([260, 760, 300])

    def _add_context(self):
        import pyvista as pv
        self.plotter.add_mesh(pv.PolyData(np.asarray(self.b.context)), color="lightgray",
                              point_size=1.0, render_points_as_spheres=False, name="context")
        self.plotter.reset_camera()

    def _context_scalars(self, kind, j):
        """Per-context-point value for colour-by: instance's attr value / conf."""
        n = len(self.b.ids)
        if kind == "val":
            appl = np.array([self.b.applicable(k, j) for k in range(n)])
            per = np.where(appl, [self._attr_val(k, j) for k in range(n)], -1.0)
        else:                                            # conf
            per = np.array([self.b.confval(k, j) for k in range(n)], float)
            per = np.where(per < 0, np.nan, per)         # not-applicable -> nan
        lut = np.full(self.b.max_id + 1, np.nan)
        lut[np.array(self.b.ids)] = per
        scal = lut[np.clip(self.b.context_inst, 0, self.b.max_id)]
        scal[self.b.context_inst < 0] = np.nan           # unsegmented background
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
        else:                                            # conf: 0.5 = most uncertain
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
    def _attr_val(self, k, j):
        """Current value of attribute j for instance k (human edit, else pipeline)."""
        r = self.review.records.get(self.b.ids[k])
        return r["vector_human"][j] if r else self.b.pipeline(k, j)

    def _changed(self, k):
        return any(self._attr_val(k, j) != self.b.pipeline(k, j) for j in range(self.b.n_attr))

    def _visible(self):
        cls = self.cb_class.currentData(); q = self.ed_search.text().lower().strip()
        aj = self.cb_attr.currentData(); astate = self.cb_attr_state.currentData()
        out = []
        for k in range(len(self.b.ids)):
            if cls and self.b.cls(k) != cls:
                continue
            if aj is not None and aj >= 0 and self._attr_val(k, aj) != astate:
                continue
            if self.cb_unrev.isChecked() and self.review.is_reviewed(k):
                continue
            if self.cb_flag.isChecked() and not (self.review.records.get(self.b.ids[k]) or {}).get("instance_flag"):
                continue
            if self.cb_changed.isChecked() and not self._changed(k):
                continue
            if q and q not in str(self.b.ids[k]) and q not in self.b.cls(k).lower():
                continue
            out.append(k)
        return out

    def _refresh_list(self):
        self._loading = True
        self.list.clear()
        for k in self._visible():
            mark = "✓ " if self.review.is_reviewed(k) else "   "
            it = QtWidgets.QListWidgetItem(f"{mark}#{self.b.ids[k]}  {self.b.cls(k)}  ({self.b.counts[k]:,} pts)")
            it.setData(QtCore.Qt.UserRole, k); self.list.addItem(it)
        self._loading = False

    def _on_current(self, row):
        if self._loading or row < 0:
            return
        self.select(int(self.list.item(row).data(QtCore.Qt.UserRole)))

    def _selected_ks(self):
        return [int(it.data(QtCore.Qt.UserRole)) for it in self.list.selectedItems()]

    # ── editor (one instance) ────────────────────────────────────────────
    def select(self, k):
        self.k = k
        rec = self.review.get(k)
        total = int(self.b.counts[k])
        self._loading = True
        nsel = len(self.list.selectedItems())
        selnote = f"    ·    {nsel} selected (bulk-set applies to all)" if nsel > 1 else ""
        self.lbl_header.setText(f"Instance {self.b.ids[k]} · {self.b.cls(k)} · {total:,} pts{selnote}")
        self.cmb_flag.setCurrentIndex(max(0, self.cmb_flag.findData(rec["instance_flag"])))
        self.ed_note.setText(rec["note"])
        confs = []
        for j, cb in enumerate(self.checks):
            cb.setChecked(bool(rec["vector_human"][j]))
            appl = self.b.applicable(k, j); cv = self.b.confval(k, j)
            # keep EVERY attribute tickable — just dim ones the pipeline marks
            # non-applicable, as a hint (the reviewer can still override).
            cb.setStyleSheet("" if appl else "color:#999; font-style:italic;")
            confs.append(f"{self.b.attr_names[j]}={cv:.2f}" if appl and np.isfinite(cv) else "")
        self.lbl_conf.setText("pipeline conf:  " + "  ".join(c for c in confs if c))
        self._loading = False

    # ── 3D highlight (all selected instances) ────────────────────────────
    def _highlight_selected(self):
        import pyvista as pv
        if self._loading:                          # ignore churn during list rebuild
            return
        ks = self._selected_ks()
        if not ks:
            try:
                self.plotter.remove_actor("hl")
            except Exception:
                pass
            self.plotter.render(); return
        pts = np.concatenate([self.b.instance_points(k) for k in ks])
        if len(pts) > HL_DISPLAY_CAP:
            pts = pts[::len(pts) // HL_DISPLAY_CAP + 1]
        self.plotter.add_mesh(pv.PolyData(pts), color="red", point_size=4.0,
                              render_points_as_spheres=False, name="hl", reset_camera=False)
        if self.cb_follow.isChecked():
            bb = self.b.bbox[np.asarray(ks)]
            mn, mx = bb[:, :3].min(0), bb[:, 3:].max(0)
            self.plotter.reset_camera(bounds=[mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]])
        self.plotter.render()

    def _bulk_selected(self, val):
        """Set one attribute ON/OFF for EVERY selected instance at once."""
        j = self.cb_bulk.currentData()
        ks = self._selected_ks()
        if j is None or not ks:
            return
        for k in ks:
            self.review.get(k)["vector_human"][j] = val
            self.review.mark(k, True)
            self._update_row(k)
        self.review.dirty = True
        if self.k in ks:
            self.select(self.k)                    # refresh the editor checkboxes
        self._recolor_if_active(j); self._refresh_counter()

    # ── edits ────────────────────────────────────────────────────────────
    def _touch(self):
        if self.k is not None and not self.review.is_reviewed(self.k):
            self.review.mark(self.k, True); self._update_row(self.k)

    def _on_check(self, j):
        if self._loading or self.k is None:
            return
        self.review.get(self.k)["vector_human"][j] = 1 if self.checks[j].isChecked() else 0
        self.review.dirty = True; self._touch(); self._recolor_if_active(j); self._refresh_counter()

    def _bulk(self, v):
        if self.k is None:
            return
        rec = self.review.get(self.k)
        for j in range(self.b.n_attr):
            rec["vector_human"][j] = v
        self.review.dirty = True; self._touch(); self.select(self.k)
        self._recolor_if_active(); self._refresh_counter()

    def _reset(self):
        if self.k is None:
            return
        rec = self.review.get(self.k)
        for j in range(self.b.n_attr):
            rec["vector_human"][j] = self.b.pipeline(self.k, j)
        self.review.dirty = True; self._touch(); self.select(self.k)
        self._recolor_if_active(); self._refresh_counter()

    def _on_flag(self):
        if self._loading or self.k is None:
            return
        self.review.get(self.k)["instance_flag"] = self.cmb_flag.currentData()
        self.review.dirty = True; self._touch()

    def _on_note(self):
        if self._loading or self.k is None:
            return
        self.review.get(self.k)["note"] = self.ed_note.text(); self.review.dirty = True

    def _confirm_next(self):
        if self.k is not None:
            self.review.mark(self.k, True); self._update_row(self.k); self._refresh_counter()
        vis = self._visible()
        if not vis:
            return
        nxt = next((k for k in vis if not self.review.is_reviewed(k)), None)
        if nxt is None:
            cur = vis.index(self.k) if self.k in vis else -1
            nxt = vis[(cur + 1) % len(vis)]
        for r in range(self.list.count()):
            if int(self.list.item(r).data(QtCore.Qt.UserRole)) == nxt:
                self.list.setCurrentRow(r); return

    def _update_row(self, k):
        for r in range(self.list.count()):
            if int(self.list.item(r).data(QtCore.Qt.UserRole)) == k:
                mark = "✓ " if self.review.is_reviewed(k) else "   "
                self.list.item(r).setText(f"{mark}#{self.b.ids[k]}  {self.b.cls(k)}  ({self.b.counts[k]:,} pts)")
                return

    def _refresh_counter(self):
        self.lbl_counter.setText(f"{self.review.n_reviewed()} / {len(self.b.ids)} reviewed"
                                 + ("   • unsaved" if self.review.dirty else ""))

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

    def _save(self):
        name = self.ed_rev.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Reviewer name",
                                          "Enter your name in the Reviewer box first — it's "
                                          "stamped into your files so reviews don't collide.")
            return
        self.review.reviewer = name
        self._persist_settings()
        p = self.review.save(); self._refresh_counter()
        msg = f"Saved:\n{p}"
        try:
            h5 = self.review.export_h5(); msg += f"\n{h5}"
        except Exception as exc:
            msg += f"\n\n(.h5 export skipped: {exc})"
        QtWidgets.QMessageBox.information(self, "Saved", msg)


APP_DIR = Path(__file__).resolve().parent
BUNDLES_DIR = Path(os.environ.get("HFX3D_BUNDLES_ROOT", "").strip() or (APP_DIR.parent / "bundles"))
BUNDLE_FILES = ("meta.npz", "points.npy", "offsets.npy", "context.npy", "context_inst.npy")


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


def _ensure_bundle(laz: Path):
    """Return a ready bundle folder for this cloud, building it (once) if needed."""
    out = BUNDLES_DIR / laz.stem
    if _bundle_ready(out):
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
    return Path(state["out"])


def _resolve(arg):
    if arg:
        p = Path(arg)
        if p.is_dir() and (p / "meta.npz").exists():
            return p                                  # already a bundle folder
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
