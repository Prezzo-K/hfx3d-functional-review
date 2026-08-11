# tools — one-time helpers (not needed for normal reviewing)

You only need this if your review cloud is **slow to load or crashes
CloudCompare**. It shrinks the `.laz` so CloudCompare opens it in seconds.

## Why loading is slow

A review `.laz` stores ~136 bytes of scalar data **per point** (`instance_id`,
`semantic_id`, `purity`, plus 15 `val_*` and 15 `conf_*` fields). On a 20–48M
point building that is several GB of scalar data, so CloudCompare stalls for
minutes and can run out of memory on the largest buildings.

Reviewing happens per **instance**, not per point, so full scan density is
wasted. `downsample_review_cloud.py` keeps **one point per small 3D voxel**:
dense areas are thinned, but every instance and every thin feature (railings,
mullions…) keeps points, and **all scalar fields are preserved**. The output
works with the Functional Review plugin exactly like the original — and because
the plugin strips a trailing `_small` from the building name, your
`.review.json` / `.reviewed.h5` are named the same either way.

Measured on `HFX_BLD001` (19.4M pts) at the default 3 cm voxel:
**19.4M → 5.8M points (30%)**, all 265 instances kept.

## Run it

This is a plain Python script — run it with **any** Python 3.9+ that has
`laspy` (it does **not** use CloudCompare's Python). Easiest is a throwaway
virtual environment:

```powershell
# from the repo root
python -m venv .tools-venv
.\.tools-venv\Scripts\python -m pip install "laspy[lazrs]" numpy

# one building
.\.tools-venv\Scripts\python tools\downsample_review_cloud.py review_clouds\train\HFX_BLD001_ZEB_CLEAN.laz

# or a whole tree at once (mirrors train/test/val into review_clouds_small\)
.\.tools-venv\Scripts\python tools\downsample_review_cloud.py review_clouds --out review_clouds_small
```

Then open the **downsampled** file in CloudCompare and click Functional Review
as usual.

## Options

| flag | meaning |
|------|---------|
| `--voxel 0.03` | voxel edge length in metres (default 3 cm). **Bigger = fewer points = faster load**, but coarser geometry. Try `0.04`–`0.05` for the giant buildings (BLD005/010/012/013). |
| `--out PATH` | output file or folder. For a single file the default is `<name>_small.laz` next to the input; for a folder the default is `<folder>_small\` mirroring the layout. |
| `--overwrite` | rebuild outputs that already exist (default: skip them, so re-running is cheap). |

## Notes

- The output is a valid LAZ with the identical extra-byte fields, so nothing in
  the plugin changes.
- If you get `laspy is not installed` or a LAZ backend error, install the
  backend: `pip install "laspy[lazrs]"`.
- Keep the originals until you've confirmed the downsampled clouds review fine —
  downsampling is lossy by design (that's the point).
