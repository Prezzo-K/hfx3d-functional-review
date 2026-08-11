# tools — make review clouds load fast

Use these if a review cloud is **slow to load or crashes CloudCompare**. Both
run with any Python 3.9+ that has `laspy` (NOT CloudCompare's Python). Set up a
throwaway env once:

```powershell
python -m venv .tools-venv
.\.tools-venv\Scripts\python -m pip install "laspy[lazrs]" numpy
```

## Why loading is slow

A review `.laz` stores ~136 bytes of scalar data **per point**: `instance_id`,
`semantic_id`, `purity`, plus **15 `val_*` and 15 `conf_*`** fields. Those 30
attribute fields are **identical for every point of an instance** — pure
duplication — yet CloudCompare loads all 34 scalar fields into memory as
float32. On a 20–48M point building that is several GB, so loads take minutes
and the biggest buildings can run out of memory.

---

## Recommended: `slim_review_cloud.py` — lossless, full density

Moves the 30 attribute fields **out of the point cloud** into a tiny
per-instance companion file. **No points dropped, no values changed.**

```powershell
# one building
.\.tools-venv\Scripts\python tools\slim_review_cloud.py review_clouds\train\HFX_BLD001_ZEB_CLEAN.laz

# a whole tree
.\.tools-venv\Scripts\python tools\slim_review_cloud.py review_clouds --out review_clouds_slim
```

Each building becomes two files:

- `<name>.laz` — full points, only `instance_id`/`semantic_id`/`purity` per
  point (**3 scalar fields instead of 34** → CloudCompare loads far less and
  far faster).
- `<name>.attrs.npz` — the 15 val + 15 conf values per instance (~24 KB). The
  plugin loads this instead of the per-point fields.

### Point the plugin at the companion

CloudCompare's Python can't see the `.laz`'s path, so the plugin finds the
companion **by building name**. Set one env var to the folder holding them
(searched recursively, so `train/test/val` subfolders are fine):

```powershell
setx HFX3D_ATTRS_ROOT "C:\path\to\review_clouds_slim"
```

(Or drop the `.attrs.npz` in your `HFX3D_REVIEW_ROOT` / `HFX3D_EXPORT_ROOT`, or
the working dir — those are searched too.) Restart CloudCompare, open the slim
`.laz`, click **Functional Review** — it works exactly as before. Measured on
`HFX_BLD001` (19.4M pts): all 264 instances and all 11,880 attribute values
identical to the original, verified.

---

## Alternative: `downsample_review_cloud.py` — fewer points

If you'd rather thin the cloud (e.g. it's still heavy for other reasons), this
keeps one point per small 3D voxel — all scalar fields preserved, every
instance kept, but **fewer points** (lossy by design). Prefer `slim` unless you
specifically want a lighter/smaller cloud.

```powershell
.\.tools-venv\Scripts\python tools\downsample_review_cloud.py review_clouds --out review_clouds_small
.\.tools-venv\Scripts\python tools\downsample_review_cloud.py review_clouds --voxel 0.05   # coarser
```

The two tools compose: you can `slim` then `downsample` for the absolute
lightest cloud.

## Notes

- Keep the originals until you've confirmed the slim/downsampled clouds review
  fine.
- LAZ backend error? `pip install "laspy[lazrs]"`.
