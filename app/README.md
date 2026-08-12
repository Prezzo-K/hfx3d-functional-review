# HFX3D functional-attribute review app

A fast, CloudCompare-free reviewer for the 15 functional attributes. It draws the
whole building once as a grey backdrop and just swaps a small red highlight when
you pick an instance, so browsing ~300 instances stays instant even on an
integrated-GPU laptop. It writes the **same** `<building>__<you>.review.json` and
`<building>__<you>.reviewed.h5` as before, so uploads and the team workflow don't
change.

Everyone on the team runs the same two scripts in parallel, each on their own
buildings.

---

## 1. Setup (once per machine)

You need **Python 3.10–3.13** (not 3.14 yet — PySide6 has no wheels for it).

```powershell
git clone <repo-url>
cd hfx3d-functional-review

py -3.13 -m venv .app-venv
.\.app-venv\Scripts\python -m pip install -r app\requirements.txt
```

That's the whole install. You do **not** need CloudCompare.

---

## 2. Start reviewing

Put the review clouds you downloaded (the `HFX_BLDxxx_ZEB_CLEAN.laz` files) under
`review_clouds\`, then launch by **building name**:

```powershell
.\.app-venv\Scripts\python app\review_app.py HFX_BLD001
```

- The **first** time you open a building, it spends ~10–15 s preparing it (a
  progress dialog shows the steps). **Every open after that is instant.**
- You can pass a **building name** (`HFX_BLD001` or just `BLD001`), a full path to a
  `.laz`, or an already-prepared bundle folder — or run with no argument to pick a
  `.laz` from a file dialog.

**First run — set your identity (once):** in the top-right of the window, type your
name in **Reviewer**, click **Change…** next to **Save to** and choose where your
review files should go, then click **Save**. Both are remembered for next time — no
environment variables, no shell restarts needed.

---

## 3. The window, part by part

The window has three columns: **instance list** (left), **3D view** (center),
**editor** (right).

### Left — find the instance(s) you want

- **Search box** — type an id or class name.
- **Class filter** — show only `window`, `wall`, `stairs`, …
- **Attribute filter** — pick an attribute + **is on / is off** to show only
  instances that (don't) have it. e.g. `operable` + *is off* → every instance
  missing `operable`.
- **unreviewed / flagged / changed** — narrow to what still needs work, what you
  flagged, or what you've edited away from the pipeline.
- Click a row to review it. The list is **multi-select** (Ctrl/Shift-click), and
  **Select all filtered ⭢ 3D** grabs the whole filtered set at once.

### Center — look at it in 3D

- **Left-drag** rotate · **wheel / right-drag** zoom · **middle-drag** pan.
- The grey cloud is the whole building (a lightweight overview). Selected
  instances light up **red**; when several are selected they all light up together.
- **"zoom to instance on select"** (right panel) frames each pick automatically —
  untick it to keep a steady camera.

### Right — record your decision

- **Reviewer / Save / Save to** — your name, the save button, and the output
  folder (with **Change…**).
- **Colour by** — recolour the *whole* building to spot outliers across it:
  - `val: <attr>` → **green = has it**, grey = doesn't, dark = not-applicable.
  - `conf: <attr>` → blue→red heatmap of pipeline confidence (**~0.5 = the pipeline
    was most unsure**, so your judgment matters most there).
  - `plain (grey)` → back to neutral.
- **The 15 attribute checkboxes** — tick the ones this instance has. Attributes the
  pipeline marked *not-applicable* are dimmed as a hint, but you can still tick any
  of them. The line under them shows the pipeline's confidence per attribute.
- **Accept all / Reject all / Reset to pipeline** — quick ways to set the current
  instance's whole vector.
- **Bulk set `<attr>` ON/OFF ⭢ sel** — set one attribute across **every selected
  instance** at once (see the group workflow below).
- **Flag** (`bad_segmentation` / `wrong_class` / `other`) and **Note** — for
  instances that themselves look wrong.
- **Confirm ✓ & Next** — mark this instance reviewed and jump to the next one that
  isn't. **Save** writes your two files.

---

## 4. Common workflows

**Review one instance:** click it → check the 3D shape → tick/untick attributes →
**Confirm ✓ & Next**. Repeat. **Save** every so often.

**Fix a whole group at once** — e.g. "4 windows are missing `operable`":

1. Left filters: class = `window`, attribute = `operable`, **is off**.
2. Click **Select all filtered ⭢ 3D** — all four light up in red so you can confirm
   they're the right ones.
3. **Bulk set** → pick `operable` → **ON ⭢ sel**. All four get it and are marked
   reviewed. **Save**.

**Spot outliers across the building:** set **Colour by → `val: <attr>`** and orbit —
anything coloured differently from its neighbours is worth a look. Then use the
attribute filter + bulk-set to fix them.

---

## 5. Where your files go / upload

**Save** writes both files to your chosen **Save to** folder:

- `<building>__<you>.review.json`
- `<building>__<you>.reviewed.h5`

Upload **both**, exactly as in the main README. Your name in the filename keeps your
review from colliding with a teammate's review of the same building.

---

## 6. Prep buildings ahead of time (optional)

The app auto-prepares a building on first open, but you can build the bundles up
front (e.g. overnight, before handing them out):

```powershell
.\.app-venv\Scripts\python app\build_bundle.py review_clouds\train\HFX_BLD001_ZEB_CLEAN.laz --out bundles\HFX_BLD001
```

---

## Settings & paths (reference)

The app reads settings in this order: **environment variable → saved settings
(`~/.hfx3d_review.json`) → default.** Setting them in the app writes the saved file,
so you normally never touch env vars. If you prefer them, these still work (in a
**new** shell after `setx`):

| Variable | Meaning | Default |
|---|---|---|
| `HFX3D_REVIEWER` | your name, stamped into filenames | (ask in-app) |
| `HFX3D_REVIEW_ROOT` | where `.review.json` / `.reviewed.h5` are saved | `~/HFX3D_reviews` |
| `HFX3D_EXPORT_ROOT` | separate folder for the `.h5` (rarely needed) | same as review root |
| `HFX3D_CLOUDS_ROOT` | where your `.laz` files live (for name lookup) | `review_clouds\` |
| `HFX3D_BUNDLES_ROOT` | where prepared bundles are cached | `bundles\` |

## Notes

- A bundle's `points.npy` is the full-resolution cloud (100–500 MB); it's
  memory-mapped, never fully loaded into RAM. Very large instances are subsampled
  **for display only** — your saved review data is always full/unchanged.
- Bundles are cached and reused; delete a building's bundle folder to force a rebuild
  (e.g. after updating the app).
