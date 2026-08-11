#!/usr/bin/env python3
"""Voxel-downsample HFX3D review clouds so CloudCompare loads them fast.

The review LAZ carries ~136 bytes of scalar data PER POINT (instance_id,
semantic_id, purity + 15 val_* + 15 conf_* fields). On a 20-48M point building
that is 3-6 GB of scalar data, so CloudCompare stalls for minutes on load and
can run out of memory on the biggest buildings.

Reviewing is done per *instance*, not per point, so full scan density is wasted.
This tool keeps ONE point per small 3D voxel: dense areas are thinned, but every
instance and every thin feature (railings, mullions...) keeps at least one point,
and ALL scalar fields are preserved unchanged. The output opens in CloudCompare
in seconds and works with the Functional Review plugin exactly like the original.

Run it (from a machine with the tools venv — see tools/README below):

    # one file
    python tools/downsample_review_cloud.py review_clouds/train/HFX_BLD001_ZEB_CLEAN.laz

    # a whole tree (train/test/val), mirroring the folder layout into out/
    python tools/downsample_review_cloud.py review_clouds --out review_clouds_small

    # coarser/finer thinning (bigger voxel = fewer points = faster load)
    python tools/downsample_review_cloud.py review_clouds --voxel 0.05

Requires: laspy with a LAZ backend, e.g.  pip install "laspy[lazrs]"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import laspy
except ImportError:
    sys.exit("laspy is not installed — run:  pip install \"laspy[lazrs]\"")


def voxel_keep_indices(x, y, z, voxel: float) -> np.ndarray:
    """Return indices of one representative point per occupied voxel.

    Keeps the FIRST point encountered in each voxel; result is sorted so the
    output preserves the original point order (nice for diffing / stability).
    """
    xyz = np.column_stack((x, y, z)).astype(np.float64)
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    dims = keys.max(axis=0) + 1                     # per-axis voxel-grid extent
    # linear voxel id; int64 is safe for building-scale extents at cm voxels
    lin = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, first = np.unique(lin, return_index=True)
    return np.sort(first)


def downsample_one(inp: Path, outp: Path, voxel: float) -> tuple[int, int]:
    las = laspy.read(str(inp))
    n_in = len(las.points)
    idx = voxel_keep_indices(las.x, las.y, las.z, voxel)
    las.points = las.points[idx]                    # keeps xyz+rgb+all extra dims
    outp.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(outp))
    return n_in, len(idx)


def iter_laz(root: Path):
    if root.is_file():
        yield root
    else:
        yield from sorted(root.rglob("*.laz"))
        yield from sorted(root.rglob("*.las"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path,
                    help="a review .laz/.las file OR a folder to walk recursively")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file or folder (default: sibling '<name>_small')")
    ap.add_argument("--voxel", type=float, default=0.03,
                    help="voxel edge length in metres (default 0.03 = 3 cm; "
                         "larger = fewer points = faster load)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild outputs that already exist (default: skip them)")
    args = ap.parse_args(argv)

    src = args.path
    if not src.exists():
        sys.exit(f"not found: {src}")

    # Resolve where each output goes, mirroring the input tree when given a dir.
    if src.is_file():
        default_out = src.with_name(src.stem + "_small" + src.suffix)
        jobs = [(src, args.out or default_out)]
    else:
        out_root = args.out or src.with_name(src.name + "_small")
        jobs = [(f, out_root / f.relative_to(src)) for f in iter_laz(src)]

    if not jobs:
        sys.exit(f"no .laz/.las files under {src}")

    print(f"voxel = {args.voxel} m   ·   {len(jobs)} file(s)\n")
    total_in = total_out = 0
    for inp, outp in jobs:
        if outp.exists() and not args.overwrite:
            print(f"skip (exists): {outp.name}")
            continue
        t0 = time.time()
        try:
            n_in, n_out = downsample_one(inp, outp, args.voxel)
        except Exception as exc:                    # keep going on the rest
            print(f"FAILED {inp.name}: {exc}")
            continue
        total_in += n_in
        total_out += n_out
        pct = 100.0 * n_out / n_in if n_in else 0.0
        print(f"{inp.name:32s} {n_in:>11,d} -> {n_out:>10,d} pts "
              f"({pct:4.1f}%)  {time.time() - t0:5.1f}s  -> {outp}")

    if total_in:
        print(f"\nTOTAL {total_in:,d} -> {total_out:,d} pts "
              f"({100.0 * total_out / total_in:.1f}% kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
