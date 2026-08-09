#!/usr/bin/env python3
"""Functional-attribute review editor — a lag-free table companion to CloudCompare.

No 3D here: you view geometry in CloudCompare (a `build_review_cloud.py` LAZ),
point-pick an instance to read its id, then edit that instance's 15 functional
attributes in this table. Rows = instances; columns = the attributes as
checkboxes plus class / #pts / purity / reviewed. The selected instance's flag,
note, and bulk actions live in the strip at the bottom.

    python review_editor.py \
        --func ".../functional_labels/train/HFX_BLD001_ZEB_CLEAN.h5" \
        [--orig ".../instances_vis/train/HFX_BLD001_ZEB_CLEAN_instances_vis.ply"] \
        [--review <json>] [--reviewer NAME]

Save writes the review JSON; Export writes a reviewed .h5; Rebuild CC cloud
regenerates the LAZ (needs --orig) so a reload in CloudCompare shows your edits.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find sibling modules

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFileDialog, QMessageBox,
)

import functional as fx
import build_review_cloud

# fixed leading columns, then one checkbox column per attribute
BASE_COLS = ["id", "class", "pts", "purity", "rev", "#on"]


class ReviewEditor(QMainWindow):
    def __init__(self, func_h5: Path, review_path: Path,
                 orig: Optional[Path], reviewer: str):
        super().__init__()
        self.func = fx.FunctionalData(func_h5)
        stem = fx.building_stem(func_h5)
        split = func_h5.parent.name
        self.store = fx.ReviewStore(review_path, self.func, stem, split, reviewer=reviewer)
        self.orig = orig
        self._loading = False
        self._visible_ids: list[int] = []

        self.setWindowTitle(f"Functional Review — {stem}")
        self.resize(1180, 760)
        self._build()
        self._apply_filter()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build(self):
        root = QWidget(); self.setCentralWidget(root)
        v = QVBoxLayout(root)

        # toolbar
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Reviewer"))
        self.ed_reviewer = QLineEdit(self.store.reviewer)
        self.ed_reviewer.setFixedWidth(140)
        self.ed_reviewer.editingFinished.connect(self._on_reviewer)
        bar.addWidget(self.ed_reviewer)
        bar.addWidget(QLabel("Class"))
        self.cb_class = QComboBox(); self.cb_class.addItem("all", "")
        for c in sorted(set(self.func.semantic_class)):
            self.cb_class.addItem(c, c)
        self.cb_class.currentIndexChanged.connect(self._apply_filter)
        bar.addWidget(self.cb_class)
        self.cb_unrev = QCheckBox("unreviewed only"); self.cb_unrev.toggled.connect(self._apply_filter)
        bar.addWidget(self.cb_unrev)
        self.cb_flagged = QCheckBox("flagged only"); self.cb_flagged.toggled.connect(self._apply_filter)
        bar.addWidget(self.cb_flagged)
        bar.addWidget(QLabel("Jump to id"))
        self.ed_jump = QLineEdit(); self.ed_jump.setFixedWidth(70)
        self.ed_jump.returnPressed.connect(self._jump)
        bar.addWidget(self.ed_jump)
        bar.addStretch(1)
        self.lbl_counter = QLabel()
        bar.addWidget(self.lbl_counter)
        v.addLayout(bar)

        bar2 = QHBoxLayout()
        self.btn_save = QPushButton("Save Review"); self.btn_save.clicked.connect(self._save)
        self.btn_export = QPushButton("Export reviewed .h5"); self.btn_export.clicked.connect(self._export)
        self.btn_rebuild = QPushButton("Rebuild CC cloud"); self.btn_rebuild.clicked.connect(self._rebuild)
        self.btn_rebuild.setEnabled(self.orig is not None)
        for b in (self.btn_save, self.btn_export, self.btn_rebuild):
            bar2.addWidget(b)
        bar2.addStretch(1)
        v.addLayout(bar2)

        # table
        self.attrs = self.func.attribute_names
        self.cols = BASE_COLS + self.attrs
        self.table = QTableWidget(0, len(self.cols))
        self.table.setHorizontalHeaderLabels(self.cols)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        v.addWidget(self.table, stretch=1)

        # bottom strip: selected instance flag / note / bulk
        strip = QHBoxLayout()
        self.lbl_sel = QLabel("No instance selected"); self.lbl_sel.setMinimumWidth(220)
        strip.addWidget(self.lbl_sel)
        strip.addWidget(QLabel("Flag"))
        self.cb_flag = QComboBox()
        for fl in fx.INSTANCE_FLAGS:
            self.cb_flag.addItem(fl or "(none)", fl)
        self.cb_flag.currentIndexChanged.connect(self._on_flag)
        strip.addWidget(self.cb_flag)
        strip.addWidget(QLabel("Note"))
        self.ed_note = QLineEdit(); self.ed_note.editingFinished.connect(self._on_note)
        strip.addWidget(self.ed_note, stretch=1)
        self.btn_accept = QPushButton("Accept all"); self.btn_accept.clicked.connect(lambda: self._bulk(1))
        self.btn_reject = QPushButton("Reject all"); self.btn_reject.clicked.connect(lambda: self._bulk(0))
        self.btn_reset = QPushButton("Reset"); self.btn_reset.clicked.connect(self._reset)
        for b in (self.btn_accept, self.btn_reject, self.btn_reset):
            strip.addWidget(b)
        v.addLayout(strip)
        self._set_strip_enabled(False)

    # ── table population ──────────────────────────────────────────────────
    def _apply_filter(self):
        cls = self.cb_class.currentData()
        unrev = self.cb_unrev.isChecked()
        flagged = self.cb_flagged.isChecked()
        ids = []
        for r, iid in enumerate(self.func.ids.tolist()):
            if cls and self.func.semantic_class[r] != cls:
                continue
            rec = self.store.records.get(iid)
            if unrev and rec and rec["status"] == "reviewed":
                continue
            if flagged and not (rec and rec["instance_flag"]):
                continue
            ids.append(iid)
        self._visible_ids = ids
        self._populate()

    def _populate(self):
        self._loading = True
        self.table.setRowCount(len(self._visible_ids))
        for vr, iid in enumerate(self._visible_ids):
            r = self.func.row_of(iid)
            rec = self.store.get(iid)
            self._text(vr, 0, str(iid))
            self._text(vr, 1, self.func.semantic_class[r])
            self._text(vr, 2, f"{int(self.func.point_count[r]):,}")
            self._text(vr, 3, f"{self.func.purity[r]:.2f}")
            # reviewed checkbox (col 4)
            revit = QTableWidgetItem()
            revit.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            revit.setCheckState(Qt.Checked if rec["status"] == "reviewed" else Qt.Unchecked)
            revit.setData(Qt.UserRole, ("rev", iid))
            self.table.setItem(vr, 4, revit)
            self._text(vr, 5, str(int(sum(rec["vector_human"]))))
            # attribute checkboxes
            for j in range(self.func.n_attr):
                appl = bool(self.func.applicable[r, j])
                it = QTableWidgetItem()
                it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it.setCheckState(Qt.Checked if rec["vector_human"][j] else Qt.Unchecked)
                it.setData(Qt.UserRole, ("attr", iid, j))
                if not appl:
                    it.setBackground(QColor("#2b2b30"))
                    it.setToolTip("not applicable per ontology")
                self.table.setItem(vr, 6 + j, it)
        self.table.resizeColumnsToContents()
        self._loading = False
        self._refresh_counter()

    def _text(self, r, c, s):
        it = QTableWidgetItem(str(s))
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        if c >= 2:
            it.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(r, c, it)

    # ── edits ──────────────────────────────────────────────────────────────
    def _on_item_changed(self, item):
        if self._loading:
            return
        tag = item.data(Qt.UserRole)
        if not tag:
            return
        checked = item.checkState() == Qt.Checked
        if tag[0] == "attr":
            _, iid, j = tag
            self.store.set_attr(iid, j, 1 if checked else 0)
            self._loading = True
            self.table.item(item.row(), 5).setText(
                str(int(sum(self.store.get(iid)["vector_human"]))))
            self._loading = False
        elif tag[0] == "rev":
            _, iid = tag
            self.store.set_status(iid, checked)
        self._refresh_counter()

    def _on_row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self._set_strip_enabled(False); return
        iid = self._visible_ids[rows[0].row()]
        self._sel_iid = iid
        rec = self.store.get(iid)
        r = self.func.row_of(iid)
        self.lbl_sel.setText(f"Instance {iid} · {self.func.semantic_class[r]}")
        self._loading = True
        self.cb_flag.setCurrentIndex(max(0, self.cb_flag.findData(rec["instance_flag"])))
        self.ed_note.setText(rec["note"])
        self._loading = False
        self._set_strip_enabled(True)

    def _set_strip_enabled(self, on):
        for w in (self.cb_flag, self.ed_note, self.btn_accept, self.btn_reject, self.btn_reset):
            w.setEnabled(on)
        if not on:
            self._sel_iid = None

    def _on_flag(self):
        if self._loading or not getattr(self, "_sel_iid", None):
            return
        self.store.set_flag(self._sel_iid, self.cb_flag.currentData())
        self._refresh_counter()

    def _on_note(self):
        if self._loading or not getattr(self, "_sel_iid", None):
            return
        self.store.set_note(self._sel_iid, self.ed_note.text())

    def _bulk(self, val):
        if not getattr(self, "_sel_iid", None):
            return
        self.store.set_all(self._sel_iid, val)
        self._reload_row(self._sel_iid)

    def _reset(self):
        if not getattr(self, "_sel_iid", None):
            return
        self.store.reset_to_pipeline(self._sel_iid)
        self._reload_row(self._sel_iid)

    def _reload_row(self, iid):
        if iid not in self._visible_ids:
            return
        vr = self._visible_ids.index(iid)
        rec = self.store.get(iid)
        self._loading = True
        for j in range(self.func.n_attr):
            self.table.item(vr, 6 + j).setCheckState(
                Qt.Checked if rec["vector_human"][j] else Qt.Unchecked)
        self.table.item(vr, 5).setText(str(int(sum(rec["vector_human"]))))
        self._loading = False
        self._refresh_counter()

    def _on_reviewer(self):
        self.store.reviewer = self.ed_reviewer.text().strip()
        self.store.dirty = True
        self._refresh_counter()

    # ── actions ──────────────────────────────────────────────────────────
    def _jump(self):
        try:
            iid = int(self.ed_jump.text())
        except ValueError:
            return
        if iid not in self._visible_ids:
            # clear filters so the instance is visible
            self.cb_class.setCurrentIndex(0)
            self.cb_unrev.setChecked(False)
            self.cb_flagged.setChecked(False)
        if iid in self._visible_ids:
            self.table.selectRow(self._visible_ids.index(iid))
            self.table.scrollToItem(self.table.item(self._visible_ids.index(iid), 0))

    def _save(self):
        p = self.store.save()
        self._refresh_counter()
        self.statusBar().showMessage(f"Saved review → {p}", 6000)

    def _export(self):
        default = (self.store.path.parents[2] / "functional_labels_reviewed"
                   / self.store.split / f"{self.store.building}.h5")
        default.parent.mkdir(parents=True, exist_ok=True)
        out, _ = QFileDialog.getSaveFileName(self, "Export reviewed HDF5",
                                             str(default), "HDF5 (*.h5)")
        if not out:
            return
        fx.export_reviewed(self.func, self.store, Path(out))
        self.statusBar().showMessage(f"Exported → {out}", 6000)

    def _rebuild(self):
        if not self.orig:
            return
        if self.store.dirty:
            self._save()
        default = (self.store.path.parents[2] / "review_clouds"
                   / self.store.split / f"{self.store.building}.laz")
        default.parent.mkdir(parents=True, exist_ok=True)
        out = build_review_cloud.build(self.orig, self.func.path, default,
                                       review=self.store.path)
        self.statusBar().showMessage(f"Rebuilt CC cloud → {out}", 8000)

    def _refresh_counter(self):
        self.lbl_counter.setText(
            f"{self.store.n_reviewed()} / {self.store.n_total()} reviewed"
            + ("   • unsaved" if self.store.dirty else ""))

    def closeEvent(self, ev):
        if self.store.dirty:
            r = QMessageBox.question(self, "Unsaved review",
                                     "Save review edits before closing?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Save:
                self._save()
            elif r == QMessageBox.Cancel:
                ev.ignore(); return
        ev.accept()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--func", type=Path, required=True, help="functional_labels .h5")
    ap.add_argument("--orig", type=Path, default=None, help="instance cloud (for Rebuild CC)")
    ap.add_argument("--review", type=Path, default=None, help="review JSON (default derived)")
    ap.add_argument("--reviewer", default="", help="reviewer name")
    args = ap.parse_args()

    review = args.review
    if review is None:
        stem = fx.building_stem(args.func)
        split = args.func.parent.name
        review = args.func.parents[1] / "reviews" / split / f"{stem}.review.json"

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ReviewEditor(args.func, Path(review), args.orig, args.reviewer)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
