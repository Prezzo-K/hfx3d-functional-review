#!/usr/bin/env python3
"""Bake one "everything" LAZ per building for CloudCompare functional-QA review.

Joins the original instance cloud (per-point instance ids) with the pipeline
functional_labels .h5 and, if present, the human review JSON, then broadcasts
the CURRENT per-instance attribute values onto points as scalar fields. Open
the result in CloudCompare: colour by any `val_*` (current decision) or
`conf_*` (pipeline confidence), and point-pick to read an instance's id and
every attribute at once.

Scalar fields written per point (sentinel -1 = orphan / no instance):
    instance_id, semantic_id, num_active, primary_attr, review_status, flag
    val_<ii>_<name>    current value 0/1 (review override, else pipeline vector)
    conf_<ii>_<name>   pipeline fused confidence      (only with --with-conf)

Usage:
    python build_review_cloud.py \
        --orig ".../instances_vis/train/HFX_BLD001_ZEB_CLEAN_instances_vis.ply" \
        --func ".../functional_labels/train/HFX_BLD001_ZEB_CLEAN.h5" \
        --out  ".../review_clouds/train/HFX_BLD001_ZEB_CLEAN.laz"
    # --review <json>  overlay human decisions   --with-conf  add conf_* fields
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find sibling modules
import cloud_io
import functional as fx

try:
    import laspy
except ImportError as exc:  # pragma: no cover
    raise SystemExit("laspy is required: pip install laspy lazrs") from exc


def resolve_values(func: fx.FunctionalData, review_path: Optional[Path]):
    """Current per-instance values (N,15), review status (N,), flag code (N,)."""
    vals = func.vec.astype(np.uint8).copy()
    status = np.zeros(func.ids.shape[0], np.uint8)
    flag = np.zeros(func.ids.shape[0], np.uint8)
    if review_path and Path(review_path).exists():
        data = json.loads(Path(review_path).read_text(encoding="utf-8"))
        for k, rec in data.get("instances", {}).items():
            r = func.row_of(int(k))
            if r is None:
                continue
            vals[r] = np.array(rec["vector_human"], np.uint8)[: func.n_attr]
            status[r] = 1 if rec.get("status") == "reviewed" else 0
            fl = rec.get("instance_flag", "")
            flag[r] = fx.INSTANCE_FLAGS.index(fl) if fl in fx.INSTANCE_FLAGS else 0
    return vals, status, flag


def build(orig: Path, func_h5: Path, out: Path,
          review: Optional[Path] = None, with_conf: bool = False) -> Path:
    func = fx.FunctionalData(func_h5)
    vals, status, flag = resolve_values(func, review)

    d = cloud_io.read_points(orig)
    xyz = d["xyz"].astype(np.float64)
    inst = (d["instance"] if d["instance"] is not None
            else np.full(xyz.shape[0], -1, np.int32)).astype(np.int64)
    rgb = d["rgb"]
    rgb16 = (np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16) if rgb is not None
             else np.zeros((xyz.shape[0], 3), np.uint16))

    rows = func._rows(inst)              # per-point row index, -1 for orphan
    valid = rows >= 0
    rv = rows[valid]

    num_active = vals.sum(1).astype(np.uint8)
    primary = np.full(func.ids.shape[0], -1, np.int8)
    for r in range(func.ids.shape[0]):
        on = vals[r].astype(bool)
        if on.any():
            primary[r] = int(np.where(on, func.conf[r], -1.0).argmax())

    n = xyz.shape[0]

    def bcast(per_inst, fill, dtype):
        out_arr = np.full(n, fill, dtype)
        out_arr[valid] = per_inst[rv]
        return out_arr

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = xyz.mean(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    extra = [
        laspy.ExtraBytesParams(name="instance_id", type=np.int32),
        laspy.ExtraBytesParams(name="semantic_id", type=np.int32),
        laspy.ExtraBytesParams(name="num_active", type=np.uint8),
        laspy.ExtraBytesParams(name="primary_attr", type=np.int8),
        laspy.ExtraBytesParams(name="review_status", type=np.uint8),
        laspy.ExtraBytesParams(name="flag", type=np.uint8),
    ]
    val_names, conf_names = [], []
    for j, nm in enumerate(func.attribute_names):
        vn = f"val_{j:02d}_{nm}"[:32]; val_names.append(vn)
        extra.append(laspy.ExtraBytesParams(name=vn, type=np.float32))
    if with_conf:
        for j, nm in enumerate(func.attribute_names):
            cn = f"conf_{j:02d}_{nm}"[:32]; conf_names.append(cn)
            extra.append(laspy.ExtraBytesParams(name=cn, type=np.float32))
    header.add_extra_dims(extra)

    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.red, las.green, las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    las.instance_id = inst.astype(np.int32)
    las.semantic_id = bcast(func.semantic_id.astype(np.int32), -1, np.int32)
    las.num_active = bcast(num_active, 0, np.uint8)
    las.primary_attr = bcast(primary, -1, np.int8)
    las.review_status = bcast(status, 0, np.uint8)
    las.flag = bcast(flag, 0, np.uint8)
    for j, vn in enumerate(val_names):
        col = np.full(n, -1.0, np.float32)
        col[valid] = vals[rv, j].astype(np.float32)
        las[vn] = col
    for j, cn in enumerate(conf_names):
        col = np.full(n, -1.0, np.float32)
        col[valid] = np.where(func.applicable[rv, j].astype(bool),
                              func.conf[rv, j], -1.0).astype(np.float32)
        las[cn] = col

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have_laz = laspy.LazBackend.Lazrs in laspy.LazBackend.detect_available()
    if out.suffix.lower() == ".laz" and not have_laz:
        out = out.with_suffix(".las")
    if out.suffix.lower() == ".laz":
        las.write(str(out), do_compress=True, laz_backend=laspy.LazBackend.Lazrs)
    else:
        las.write(str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig", type=Path, required=True, help="instance cloud (.ply/.las/.laz)")
    ap.add_argument("--func", type=Path, required=True, help="functional_labels .h5")
    ap.add_argument("--out", type=Path, required=True, help="output .laz/.las")
    ap.add_argument("--review", type=Path, default=None, help="review JSON to overlay")
    ap.add_argument("--with-conf", action="store_true", help="also bake conf_* fields")
    args = ap.parse_args()

    out = build(args.orig, args.func, args.out, args.review, args.with_conf)
    func = fx.FunctionalData(args.func)
    print(f"wrote {out}  ({func.ids.shape[0]} instances, {func.n_attr} attributes"
          f"{', +conf' if args.with_conf else ''})")


if __name__ == "__main__":
    main()
