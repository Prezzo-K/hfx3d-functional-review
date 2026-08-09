# HFX3D functional-attribute review

This is how we check and fix the automatic functional attributes on the HFX3D
buildings. You review inside **CloudCompare** — open a building, click an
instance, and confirm or correct its 15 attributes. Your edits go into your own
review file; the original data is never touched. When a building is done we
export a reviewed `.h5` for training.

Flow:

```
functional_labels/*.h5  (pipeline)
        │  build the review cloud
        ▼
review_clouds/<split>/*.laz   ── open in CloudCompare, review with the panel ──►  reviews/<building>__<you>.review.json
                                                                                          │  export
                                                                                          ▼
                                                                          functional_labels_reviewed/<split>/*.h5
```

Two roles: **Admin** (sets things up, builds the clouds, runs the export) and
**Reviewer** (does the labelling). Jump to your part.

The 15 attributes, in order:
`load_bearing, thermal_envelope, vegetation_support, operable, solar_shading,
ventilation, natural_lighting, access, drainage, fall_protection, aesthetic,
privacy_screening, circulation, illumination, surveillance`.

You can also flag an instance `bad_segmentation` / `wrong_class` / `other` when
the *instance itself* looks wrong (not the attributes). Flags are just notes for
us — they don't drop or change anything on export.

---

## Paths — set these to yours

The examples below use my machine. **Change the parts marked `← change`.**

```
Tools (this repo):   C:\Users\s4824030\PycharmProjects\hfx3d-functional-review        ← change to where you cloned it
Pipeline output:     C:\Users\s4824030\PycharmProjects\HFX3D_functional_attribute_refinement\HFX3D_functional_attribute_refinement\results\functional_labels   ← change
Instance clouds:     C:\Users\s4824030\PycharmProjects\hfx3d-benchmark\HFX3D_Instance+Semantic\instances_vis                                                     ← change (Admin only)
Shared data folder:  <put this on a drive everyone can reach>\HFX3D        ← change (e.g. a network share)
```

The **shared data folder** holds what the team passes around. There's a ready-made
copy in this repo under `data\` you can drop onto the share as a starting point:

```
<shared>\HFX3D\
  review_clouds\<split>\*.laz              reviewers open these
  reviews\<building>__<reviewer>.json      one file per building per reviewer
  functional_labels_reviewed\<split>\*.h5  exported result (for training)
  functional_labels\<split>\*.h5           pipeline output (Admin needs it to export)
```

---

## Admin — one-time

1. Make the environment (in the repo folder):

   ```powershell
   cd C:\Users\s4824030\PycharmProjects\hfx3d-functional-review     ← change
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Build a review cloud for every building. This bakes the current attributes
   into one `.laz` per building that reviewers open:

   ```powershell
   python review_admin.py build-all `
     --inst-root "C:\Users\s4824030\PycharmProjects\hfx3d-benchmark\HFX3D_Instance+Semantic\instances_vis" `
     --func-root "C:\Users\s4824030\PycharmProjects\HFX3D_functional_attribute_refinement\HFX3D_functional_attribute_refinement\results\functional_labels" `
     --out-root  "<shared>\HFX3D\review_clouds"
   ```

3. Put `review_clouds\` on the shared drive and tell the team the path.

That's it until reviews come back — then see **Export** at the bottom.

---

## Reviewer — one-time (do this once, it stays set up)

1. **CloudCompare 2.13 with the Python plugin.** Install CloudCompare 2.13 and,
   in the installer, tick **Python plugin**. Open it once — you should see a
   Python console/menu. (No console = re-run the installer and enable it.)

2. **Add the review panel as a button.** Copy this repo's folder to your PC
   (e.g. `C:\HFX3D\tools\hfx3d-functional-review`), then in CloudCompare open the
   Python plugin settings and add that folder as a custom-plugins path. Restart
   CloudCompare — a **Functional Review** button appears. (If your build has no
   such setting, use the console way at the bottom.)

3. **Set your name and the shared folder** (once, in a terminal):

   ```powershell
   setx HFX3D_REVIEWER "abdi"                      ← change to your name
   setx HFX3D_REVIEW_ROOT "<shared>\HFX3D\reviews"  ← change to the shared reviews folder
   ```

   Close and reopen CloudCompare afterwards (it reads these at startup). Your
   name keeps your review files separate from everyone else's.

---

## Reviewer — doing a building

1. In CloudCompare: **File → Open** →
   `<shared>\HFX3D\review_clouds\<split>\HFX_BLDxxx_ZEB_CLEAN.laz`.
2. Click the cloud in the tree (left) to select it, then click **Functional
   Review**. The panel opens.
3. Work down the **instance list** on the left (filter by class, or tick
   "unreviewed"). Click a row — that instance lights up in the 3D view and its
   attributes load on the right.
4. Judge it, then tick/untick attributes (or **Accept all / Reject all / Reset
   to pipeline**). Add a **Flag** or **Note** if useful. Hit **Confirm ✓ & Next**
   to move on. (Any edit also counts it as reviewed — the counter is just so we
   can see how far along a building is.)
5. Click **Save Review** now and then, and before you close. It writes
   `<shared>\HFX3D\reviews\<building>__<you>.review.json`.

Splitting work: either one building per person, or several people on the same
building — everyone writes their own `__<name>` file, so nobody overwrites
anyone.

**Comparing while you review:** use *Colour by* at the top —
`val: <attr>` shows the current decision for that attribute across the whole
building (good for spotting outliers), `conf: <attr>` shows the pipeline's
confidence (near 0.5 = it was unsure, so your call matters most).

---

## Admin — export the reviewed dataset

When reviews are in, merge each reviewer's JSON with the pipeline `.h5` into a
reviewed `.h5`. Originals aren't touched — the reviewed file just adds your
decisions next to the pipeline ones.

All of one reviewer's buildings at once:

```powershell
python review_admin.py export-all --reviewer abdi `
  --func-root    "C:\Users\s4824030\PycharmProjects\HFX3D_functional_attribute_refinement\HFX3D_functional_attribute_refinement\results\functional_labels" `
  --reviews-root "<shared>\HFX3D\reviews" `
  --out-root     "<shared>\HFX3D\functional_labels_reviewed"
```

One building:

```powershell
python review_admin.py export `
  --func   "...\functional_labels\train\HFX_BLD001_ZEB_CLEAN.h5" `
  --review "<shared>\HFX3D\reviews\HFX_BLD001_ZEB_CLEAN__abdi.review.json" `
  --out    "<shared>\HFX3D\functional_labels_reviewed\train"
```

Watch the two easy mistakes: `--func` is the pipeline **`.h5`** (not the review
`.laz`), and `--out` can be a folder (the filename is filled in) or a full path.

The reviewed `.h5` has the full per-instance vectors for **every** instance:
`functional_attribute_vector` (pipeline, kept) and
`functional_attribute_vector_human` (what training should read), plus
`review_status`, `instance_flag`, `review_note`. Unreviewed instances just carry
the pipeline vector.

---

## What changes, what doesn't

- Never touched: the original `HFX3D_Instance+Semantic\*.h5` and the pipeline
  `functional_labels\*.h5`.
- Your edits live in your `reviews\...json`.
- The training file is `functional_labels_reviewed\<split>\<building>.h5`.
- In CloudCompare, editing only changes the on-screen colours and your review
  file — the `.laz` on disk isn't rewritten.

---

## If something's off

- *"cloud has no instance_id / val_* field"* → wrong file open; open a
  `review_clouds\...laz`.
- Clicking a **point** doesn't select → use the **list** on the left, that's the
  intended way (point-picking is finicky across CC builds).
- Colours don't update after an edit → set *Colour by* to `val: <that attribute>`.
- **Functional Review** button missing → finish the plugin-path step, restart CC,
  or use the console way below.
- Reviews save as `unassigned` or to the wrong place → `HFX3D_REVIEWER` /
  `HFX3D_REVIEW_ROOT` aren't set; redo the `setx` step and restart CloudCompare.
- `h5py` errors *inside* CloudCompare → you don't need it there; it's only for the
  Admin build/export scripts.

**Console way to launch the panel** (if the button isn't set up):

```python
import sys; sys.path.append(r"C:\HFX3D\tools\hfx3d-functional-review")   # ← change
import cc_functional_review as R
R.main()
```

To reload after the script changes: `import importlib; importlib.reload(R); R.main()`.
