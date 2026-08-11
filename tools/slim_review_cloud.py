#!/usr/bin/env python3
"""Split a heavy review LAZ into a slim LAZ + a per-instance companion file.

The heavy review clouds store 15 ``val_*`` and 15 ``conf_*`` scalar fields on
EVERY point — but those values are identical for all points of an instance, so
~120 of the 136 bytes/point are pure duplication. On a 20-48M point building
that is several GB of redundant scalar data, which is why CloudCompare takes
minutes to load them and the plugin has 30 giant arrays to wade through.

This tool rewrites each cloud losslessly and at FULL point density:

  * ``<name>.laz``        — keeps only per-point instance_id / semantic_id /
                            purity (~12 bytes/point instead of ~136), so
                            CloudCompare loads it many times faster.
  * ``<name>.attrs.npz``  — the 15 val + 15 conf values per INSTANCE (a few
                            hundred rows), which the Functional Review plugin
                            loads instead of the per-point fields.

No points are dropped and no attribute value is quantised — it is a lossless
reorganisation. The plugin finds the companion by building name under
HFX3D_ATTRS_ROOT / HFX3D_REVIEW_ROOT / HFX3D_EXPORT_ROOT / the working dir, so
put the ``.attrs.npz`` in one of those (by default it is written next to the
slim ``.laz``; point HFX3D_ATTRS_ROOT there, or copy it into your review root).

    python tools/slim_review_cloud.py review_clouds --out review_clouds_slim
    python tools/slim_review_cloud.py review_clouds/train/HFX_BLD001_ZEB_CLEAN.laz

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
KEEP_PER_POINT = ("instance_id", "semantic_id", "purity")


def _attr_layout(dim_names):
    """Return (n_attr, names, {j: val_dim}, {j: conf_dim}) from extra-dim names."""
    val_dim, conf_dim, names = {}, {}, {}
    for nm in dim_names:
        m = _VAL_RE.match(nm)
        if m:
            j = int(m.group(1)); val_dim[j] = nm; names[j] = m.group(2); continue
        m = _CONF_RE.match(nm)
        if m:
            conf_dim[int(m.group(1))] = nm
    if not val_dim:
        raise ValueError("no val_* fields — not a heavy review cloud")
    n_attr = max(val_dim) + 1
    return n_attr, [names.get(j, f"attr{j}") for j in range(n_attr)], val_dim, conf_dim


def slim_one(inp: Path, laz_out: Path, npz_out: Path) -> tuple[int, int]:
    las = laspy.read(str(inp))
    dim_names = list(las.point_format.dimension_names)
    n_attr, attr_names, val_dim, conf_dim = _attr_layout(dim_names)

    inst = np.asarray(las["instance_id"]).astype(np.int64)
    uids, first = np.unique(inst, return_index=True)      # one row per instance
    n = len(uids)

    val = np.zeros((n, n_attr), np.uint8)
    conf = np.full((n, n_attr), np.nan, np.float32)
    for j in range(n_attr):
        val[:, j] = (np.asarray(las[val_dim[j]])[first] > 0.5).astype(np.uint8)
        if j in conf_dim:
            conf[:, j] = np.asarray(las[conf_dim[j]])[first].astype(np.float32)

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_out, instance_id=uids, val=val, conf=conf,
             attribute_names=np.array(attr_names))

    # slim LAZ: same points, drop the val_/conf_ (and housekeeping) extra dims
    keep = [k for k in KEEP_PER_POINT if k in dim_names]
    out_hdr = laspy.LasHeader(version=str(las.header.version),
                              point_format=laspy.PointFormat(las.header.point_format.id))
    out_hdr.scales = las.header.scales
    out_hdr.offsets = las.header.offsets
    for vlr in las.header.vlrs:
        if vlr.__class__.__name__ != "ExtraBytesVlr":      # laspy rebuilds this
            out_hdr.vlrs.append(vlr)
    for name in keep:
        out_hdr.add_extra_dim(laspy.ExtraBytesParams(name=name, type=las[name].dtype))
    out = laspy.LasData(out_hdr)
    for name in las.point_format.standard_dimension_names:
        out[name] = las[name]
    for name in keep:
        out[name] = las[name]
    laz_out.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(laz_out))
    return len(inst), n


def iter_laz(root: Path):
    if root.is_file():
        yield root
    else:
        yield from sorted(root.rglob("*.laz"))
        yield from sorted(root.rglob("*.las"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="a heavy review .laz/.las OR a folder")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file or folder (default: sibling '<name>_slim')")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild outputs that already exist (default: skip)")
    args = ap.parse_args(argv)

    src = args.path
    if not src.exists():
        sys.exit(f"not found: {src}")

    if src.is_file():
        laz_out = args.out or src.with_name(src.stem + ".laz")
        if laz_out.resolve() == src.resolve():
            sys.exit("refusing to overwrite the input; pass --out")
        jobs = [(src, laz_out)]
    else:
        out_root = args.out or src.with_name(src.name + "_slim")
        jobs = [(f, out_root / f.relative_to(src).with_suffix(".laz")) for f in iter_laz(src)]

    if not jobs:
        sys.exit(f"no .laz/.las files under {src}")

    print(f"{len(jobs)} file(s)\n")
    for inp, laz_out in jobs:
        npz_out = laz_out.with_name(laz_out.stem + ".attrs.npz")
        if laz_out.exists() and npz_out.exists() and not args.overwrite:
            print(f"skip (exists): {laz_out.name}")
            continue
        t0 = time.time()
        try:
            n_pts, n_inst = slim_one(inp, laz_out, npz_out)
        except Exception as exc:
            print(f"FAILED {inp.name}: {exc}")
            continue
        print(f"{inp.name:32s} {n_pts:>11,d} pts, {n_inst:>4d} instances  "
              f"{time.time() - t0:5.1f}s")
        print(f"    -> {laz_out}")
        print(f"    -> {npz_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
