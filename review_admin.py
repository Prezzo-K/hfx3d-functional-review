#!/usr/bin/env python3
"""HFX3D functional review — single admin entry point (build clouds / export reviewed).

One command with subcommands, so there is one thing to remember:

  build       one review LAZ for CloudCompare (from a cloud + its pipeline .h5)
  build-all   every building under a folder tree
  export      one reviewed .h5 (from pipeline .h5 + a review.json)
  export-all  every review.json under a folder for one reviewer

Examples (Windows, from the repo folder with the venv active):

  python review_admin.py build ^
      --orig  "...\\instances_vis\\train\\HFX_BLD001_ZEB_CLEAN_instances_vis.ply" ^
      --func  "...\\functional_labels\\train\\HFX_BLD001_ZEB_CLEAN.h5" ^
      --out   "S:\\HFX3D\\review_clouds\\train\\HFX_BLD001_ZEB_CLEAN.laz"

  python review_admin.py build-all ^
      --inst-root "...\\instances_vis" --func-root "...\\functional_labels" ^
      --out-root  "S:\\HFX3D\\review_clouds"

  python review_admin.py export ^
      --func   "...\\functional_labels\\train\\HFX_BLD001_ZEB_CLEAN.h5" ^
      --review "S:\\HFX3D\\reviews\\HFX_BLD001_ZEB_CLEAN__abdi.review.json" ^
      --out    "S:\\HFX3D\\functional_labels_reviewed\\train"

  python review_admin.py export-all --reviewer abdi ^
      --func-root    "...\\functional_labels" ^
      --reviews-root "S:\\HFX3D\\reviews" ^
      --out-root     "S:\\HFX3D\\functional_labels_reviewed"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find sibling modules
import build_review_cloud
import functional as fx


def _find_func_h5(func_root: Path, stem: str) -> Path | None:
    hits = list(Path(func_root).rglob(f"{stem}.h5"))
    return hits[0] if hits else None


def cmd_build(a):
    out = build_review_cloud.build(a.orig, a.func, a.out,
                                   review=a.review, with_conf=not a.no_conf)
    print(f"built {out}")


def cmd_build_all(a):
    n = 0
    for ply in sorted(Path(a.inst_root).rglob("*_instances_vis.ply")):
        split = ply.parent.name
        stem = fx.building_stem(ply)
        func = _find_func_h5(a.func_root, stem)
        if not func:
            print(f"  ! no functional .h5 for {stem} — skipped"); continue
        out = Path(a.out_root) / split / f"{stem}.laz"
        build_review_cloud.build(ply, func, out, with_conf=not a.no_conf)
        print(f"  built {split}/{stem}"); n += 1
    print(f"done — {n} review clouds under {a.out_root}")


def _export_one(func_h5: Path, review: Path, out_arg: Path) -> Path:
    func = fx.FunctionalData(func_h5)
    stem = fx.building_stem(func_h5)
    store = fx.ReviewStore(review, func, stem, func_h5.parent.name)
    out = out_arg
    if out.is_dir() or out.suffix.lower() != ".h5":
        out = out / f"{stem}.h5"
    return fx.export_reviewed(func, store, out)


def cmd_export(a):
    if a.func.suffix.lower() != ".h5":
        raise SystemExit(f"--func must be the pipeline .h5, not '{a.func.name}'")
    out = _export_one(a.func, a.review, a.out)
    print(f"wrote {out}")


def cmd_export_all(a):
    suffix = f"__{a.reviewer}.review.json"
    n = 0
    for rev in sorted(Path(a.reviews_root).glob(f"*{suffix}")):
        stem = rev.name[: -len(suffix)]
        func = _find_func_h5(a.func_root, stem)
        if not func:
            print(f"  ! no functional .h5 for {stem} — skipped"); continue
        out = _export_one(func, rev, Path(a.out_root) / func.parent.name)
        print(f"  exported {stem} -> {out}"); n += 1
    print(f"done — {n} reviewed files under {a.out_root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="one review LAZ")
    b.add_argument("--orig", type=Path, required=True)
    b.add_argument("--func", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--review", type=Path, default=None, help="overlay a review.json")
    b.add_argument("--no-conf", action="store_true", help="omit conf_* (panel needs conf!)")
    b.set_defaults(fn=cmd_build)

    ba = sub.add_parser("build-all", help="every building in a tree")
    ba.add_argument("--inst-root", type=Path, required=True, help="instances_vis root")
    ba.add_argument("--func-root", type=Path, required=True, help="functional_labels root")
    ba.add_argument("--out-root", type=Path, required=True)
    ba.add_argument("--no-conf", action="store_true")
    ba.set_defaults(fn=cmd_build_all)

    e = sub.add_parser("export", help="one reviewed .h5")
    e.add_argument("--func", type=Path, required=True)
    e.add_argument("--review", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True, help="file or folder")
    e.set_defaults(fn=cmd_export)

    ea = sub.add_parser("export-all", help="all reviews for one reviewer")
    ea.add_argument("--reviewer", required=True)
    ea.add_argument("--func-root", type=Path, required=True)
    ea.add_argument("--reviews-root", type=Path, required=True)
    ea.add_argument("--out-root", type=Path, required=True)
    ea.set_defaults(fn=cmd_export_all)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
