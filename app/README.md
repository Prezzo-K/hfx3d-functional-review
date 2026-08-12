# Standalone review app (prototype)

A CloudCompare-free reviewer: draws the whole building once as a grey backdrop and
swaps a small red highlight when you pick an instance, so browsing ~300 instances is
instant even on an integrated GPU. Writes the **same** `.review.json` + `.reviewed.h5`
as the CloudCompare plugin, so the upload workflow is unchanged.

## One-time setup

```powershell
git clone <repo-url>
cd hfx3d-functional-review

# a Python 3.10–3.13 (NOT 3.14 yet — PySide6 wheels)
py -3.13 -m venv .app-venv
.\.app-venv\Scripts\python -m pip install -r app\requirements.txt
```

## Use

**1. Build a bundle** from a review `.laz` (once per building; ~10–15 s):

```powershell
.\.app-venv\Scripts\python app\build_bundle.py review_clouds\train\HFX_BLD001_ZEB_CLEAN.laz --out bundles\HFX_BLD001
```

**2. Launch the reviewer:**

```powershell
setx HFX3D_REVIEWER "your-name"        # once
setx HFX3D_REVIEW_ROOT "C:\path\to\reviews"
.\.app-venv\Scripts\python app\review_app.py bundles\HFX_BLD001
```

- Left: instance list with filters — search id/class, filter by **class**, by a specific
  **attribute is on/off**, and by **unreviewed / flagged / changed**. Click a row → it
  highlights in 3D and the attributes load on the right.
- Center: **left-drag rotate, right-drag / wheel zoom, middle-drag pan** (VTK). Toggle
  "zoom to instance on select" if you'd rather keep a fixed view.
- **Colour by** (right panel): colour the whole building by any attribute to spot
  outliers — `val: <attr>` (green = on, grey = off, dark = not-applicable) or
  `conf: <attr>` (blue→red; ~0.5 = the pipeline was most unsure, so your call matters
  most). Needs a bundle built with this version (rebuild older bundles).
- Right: tick the 15 attributes, set a flag/note, **Confirm ✓ & Next**, then **Save**
  (writes both files to `HFX3D_REVIEW_ROOT` / `HFX3D_EXPORT_ROOT`).

### Fix a whole group at once (e.g. "4 windows missing `operable`")

1. Filter the list: class = `window`, attribute = `operable`, **is off**.
2. Click **Select all filtered ⭢ 3D** (or Ctrl/Shift-click a few) — every matching
   instance highlights together in red so you can eyeball them.
3. In **Bulk set**, pick `operable` and click **ON ⭢ sel** — it sets that attribute
   for *all* selected instances at once and marks them reviewed. **Save**.

The whole team can run this in parallel — each person their own bundles and reviews.

## Notes
- `points.npy` in a bundle is the full-resolution cloud (100–500 MB); it is memory-mapped,
  never fully loaded. Huge instances are subsampled **for display only** (data is intact).
- This is a prototype to evaluate the approach — packaging into a single `.exe` for
  non-technical teammates is a later step (PyInstaller).
