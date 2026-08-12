#!/usr/bin/env python3
"""Preprocess a review LAZ into a fast "bundle" for the standalone review app.

Run ONCE per building (offline). Reads a review cloud (the heavy build_review_cloud
LAZ, which carries xyz + instance_id + semantic_id + purity + 15 val_ + 15 conf_)
and writes a small folder the review app opens instantly and browses instantly:

    <out>/context.npy    decimated whole-building xyz backdrop (~1-2M pts, float32)
    <out>/points.npy     ALL points' xyz, SORTED BY instance_id (memmap; float32)
    <out>/offsets.npy    CSR start/end into points.npy per instance (int64)
    <out>/meta.npz       per-instance: instance_id, semantic_id, purity, point_count,
                         bbox, centroid, attribute_names, val (n,15), conf (n,15)

The one expensive scan of all N points happens here, offline, ONCE — so the
reviewer never waits. Instance i's full-res points are points[offsets[i]:offsets[i+1]].

    python app/build_bundle.py review_clouds/train/HFX_BLD001_ZEB_CLEAN.laz --out bundles/HFX_BLD001

Requires: laspy with a LAZ backend  ->  pip install "laspy[lazrs]"
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import laspy
except ImportError:
    sys.exit("laspy is not installed — run:  pip install \"laspy[lazrs]\"")

_VAL_RE = re.compile(r"^val_(\d+)_(.+)$")
_CONF_RE = re.compile(r"^conf_(\d+)_(.+)$")
SEM_NAMES = ["wall", "window", "door", "balcony", "vegetation", "stairs",
             "terrain", "roof", "blinds", "other", "column", "arch"]


def _voxel_indices(xyz, voxel):
    keys = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    dims = keys.max(0) + 1
    lin = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, first = np.unique(lin, return_index=True)
    return np.sort(first)


def _attr_table(las, dim_names, first_by_row_order):
    """Per-instance val/conf read at each instance's first point (row order)."""
    val_dim, conf_dim, names = {}, {}, {}
    for nm in dim_names:
        m = _VAL_RE.match(nm)
        if m:
            j = int(m.group(1)); val_dim[j] = nm; names[j] = m.group(2); continue
        m = _CONF_RE.match(nm)
        if m:
            conf_dim[int(m.group(1))] = nm
    if not val_dim:
        return None, None, None                     # slim cloud without attrs
    n_attr = max(val_dim) + 1
    attr_names = [names.get(j, f"attr{j}") for j in range(n_attr)]
    nrow = len(first_by_row_order)
    val = np.zeros((nrow, n_attr), np.uint8)
    conf = np.full((nrow, n_attr), np.nan, np.float32)
    for j in range(n_attr):
        val[:, j] = (np.asarray(las[val_dim[j]])[first_by_row_order] > 0.5).astype(np.uint8)
        if j in conf_dim:
            conf[:, j] = np.asarray(las[conf_dim[j]])[first_by_row_order].astype(np.float32)
    return attr_names, val, conf


def build(inp: Path, out: Path, context_voxel: float) -> dict:
    las = laspy.read(str(inp))
    xyz = np.column_stack((np.asarray(las.x), np.asarray(las.y),
                           np.asarray(las.z))).astype(np.float32)
    n = len(xyz)
    inst = np.asarray(las["instance_id"]).astype(np.int64)
    sem = np.asarray(las["semantic_id"]).astype(np.int32) if "semantic_id" in las.point_format.dimension_names else np.full(n, -1, np.int32)
    pur = np.asarray(las["purity"]).astype(np.float32) if "purity" in las.point_format.dimension_names else np.full(n, np.nan, np.float32)

    # sort every point by instance so each instance is a contiguous slice
    order = np.argsort(inst, kind="stable")
    inst_s = inst[order]
    xyz_s = xyz[order]
    uids, starts, counts = np.unique(inst_s, return_index=True, return_counts=True)
    keep = uids >= 0
    uids, starts, counts = uids[keep], starts[keep], counts[keep]
    ends = starts + counts
    # offsets are into the (background-stripped) contiguous region; realign so
    # points.npy holds only real instances back-to-back
    first_rows = starts                                   # first point of each instance in sorted order
    # build compact points array (drop background id<0 up front)
    lo = int(starts.min()) if len(starts) else 0
    hi = int(ends.max()) if len(ends) else 0
    points = xyz_s[lo:hi]
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)  # into `points`

    # per-instance metadata
    sem_s = sem[order]; pur_s = pur[order]
    inst_sem = sem_s[first_rows]
    inst_pur = pur_s[first_rows]
    bbox = np.zeros((len(uids), 6), np.float32)
    centroid = np.zeros((len(uids), 3), np.float32)
    for k in range(len(uids)):
        seg = points[offsets[k]:offsets[k + 1]]
        bbox[k, :3] = seg.min(0); bbox[k, 3:] = seg.max(0)
        centroid[k] = seg.mean(0)

    attr_names, val, conf = _attr_table(las, las.point_format.dimension_names, order[first_rows])

    # decimated backdrop (keep each kept point's instance_id so the app can
    # colour the whole building by an attribute)
    ctx_idx = _voxel_indices(xyz, context_voxel)
    context = xyz[ctx_idx]
    context_inst = inst[ctx_idx].astype(np.int64)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "context.npy", context)
    np.save(out / "context_inst.npy", context_inst)
    np.save(out / "points.npy", points)
    np.save(out / "offsets.npy", offsets)
    meta = dict(instance_id=uids.astype(np.int64), semantic_id=inst_sem,
                purity=inst_pur, point_count=counts.astype(np.int64),
                bbox=bbox, centroid=centroid,
                building=Path(inp).stem)
    if attr_names is not None:
        meta.update(attribute_names=np.array(attr_names), val=val, conf=conf)
    np.savez(out / "meta.npz", **meta)
    return dict(n=n, instances=int(len(uids)), context=int(len(context)),
                points=int(len(points)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("laz", type=Path, help="a review .laz/.las for one building")
    ap.add_argument("--out", type=Path, required=True, help="output bundle folder")
    ap.add_argument("--context-voxel", type=float, default=0.05,
                    help="voxel size (m) for the decimated backdrop (default 5cm)")
    args = ap.parse_args(argv)
    if not args.laz.exists():
        sys.exit(f"not found: {args.laz}")
    t0 = time.time()
    r = build(args.laz, args.out, args.context_voxel)
    print(f"bundle {args.out}  |  {r['n']:,} pts -> {r['instances']} instances, "
          f"context {r['context']:,} pts  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    raise SystemExit(main())
