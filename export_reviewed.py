#!/usr/bin/env python3
"""Merge a review.json (from the CC panel or the editor) with the pipeline
functional .h5 into a reviewed .h5 (pipeline + human vectors + status/flag/note).

    python export_reviewed.py \
        --func   ".../functional_labels/train/HFX_BLD001_ZEB_CLEAN.h5" \
        --review ".../reviews/HFX_BLD001_ZEB_CLEAN.review.json" \
        --out    ".../functional_labels_reviewed/train/HFX_BLD001_ZEB_CLEAN.h5"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find sibling modules
import functional as fx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--func", type=Path, required=True)
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.func.suffix.lower() != ".h5":
        raise SystemExit(
            f"--func must be the pipeline functional_labels .h5 "
            f"(e.g. functional_labels/train/<building>.h5), not '{args.func.name}'.")
    if not args.func.exists():
        raise SystemExit(f"--func not found: {args.func}")
    if not args.review.exists():
        raise SystemExit(f"--review not found: {args.review}")

    func = fx.FunctionalData(args.func)
    stem = fx.building_stem(args.func)
    store = fx.ReviewStore(args.review, func, stem, args.func.parent.name)

    out = args.out
    # allow --out to be a directory (or have no .h5 suffix) → derive <stem>.h5
    if out.is_dir() or out.suffix.lower() != ".h5":
        out = out / f"{stem}.h5"
    out = fx.export_reviewed(func, store, out)
    print(f"wrote {out}  ({store.n_reviewed()}/{store.n_total()} reviewed)")


if __name__ == "__main__":
    main()
