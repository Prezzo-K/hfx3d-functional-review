# HFX3D functional-attribute review

This repo contains the **admin tools and CloudCompare plugin** for reviewing HFX3D
functional attributes. The review pipeline works like this:

- **Admin** (you): Set up environments, build review clouds, merge reviewer feedback
- **Reviewers** (your team): Clone this repo, open buildings in CloudCompare, tick/untick attributes

Your edits go into personal review files; originals are never touched. When done, the admin
exports a reviewed `.h5` file for training.

---

## Get started (clone this repo)

```powershell
git clone <repo-url>
cd hfx3d-functional-review
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Where data lives

This repo contains **tools only**. You need to set up paths to:

- **`<FUNC_ROOT>`** — functional attribute labels (HDF5 files)
  - Admin: Generate or receive from pipeline
  - Reviewers: Downloaded from AI server → stored locally or on shared drive
- **`<INST_ROOT>`** — instance point clouds (HDF5 or LAZ)
  - Located in your `hfx3d-benchmark` repo checkout
- **`<SHARED>`** — shared team folder (network drive or common location)
  - `review_clouds/` — downloaded from AI server, opened by reviewers
  - `reviews/` — your JSON edits
  - `functional_labels_reviewed/` — exported HDF5 after review

---

## Setup by role

Pick your part below.

---

## Admin

**One-time setup:**

1. ✅ You've already cloned this repo and activated the venv

2. Point to your local data paths. Edit the commands below to use:
   - `<INST_ROOT>` — path to `hfx3d-benchmark\HFX3D_Instance+Semantic\instances_vis`
   - `<FUNC_ROOT>` — path to functional labels (`.h5` files from pipeline)
   - `<SHARED>` — where you want review clouds and team files to live

3. **Build review clouds** (combines instances + attributes into `.laz` for CloudCompare):

   ```powershell
   python review_admin.py build-all `
     --inst-root "<INST_ROOT>" `
     --func-root "<FUNC_ROOT>" `
     --out-root  "<SHARED>\review_clouds"
   ```

4. **Upload review clouds to AI server** so reviewers can download them

That's it! When reviews come back, see **Export** below.

---

## Reviewer — one-time setup

You only do this once. Afterwards, you can review any building.

### 1. Install CloudCompare 2.13 with Python plugin

- Download CloudCompare 2.13
- In the installer, tick **Python plugin**
- Open CloudCompare once — you should see a Python console/menu

### 2. Clone this repo (the plugin)

```powershell
git clone <repo-url>
```

### 3. Add this repo as a CloudCompare custom plugin

- Open CloudCompare
- **Tools → Python → Edit plugin search paths** (or similar, varies by build)
- Add the path to this cloned repo folder
- Restart CloudCompare
- A **Functional Review** button should appear

### 4. Set your name and shared folder (once)

```powershell
setx HFX3D_REVIEWER "your-name"
setx HFX3D_REVIEW_ROOT "<SHARED>\reviews"
```

Close and reopen CloudCompare. Your name keeps reviews separate from everyone else's.

---

## Reviewer — start reviewing

The 15 attributes you'll review:

`load_bearing, thermal_envelope, vegetation_support, operable, solar_shading,
ventilation, natural_lighting, access, drainage, fall_protection, aesthetic,
privacy_screening, circulation, illumination, surveillance`

You can also flag an instance `bad_segmentation` / `wrong_class` / `other` if the instance itself looks wrong.

**Workflow:**

1. **Download review cloud** from the AI server to `<SHARED>\review_clouds\<split>\`
2. **Open in CloudCompare**: File → Open → `<SHARED>\review_clouds\<split>\HFX_BLDxxx_ZEB_CLEAN.laz`
3. Click the cloud in the tree (left), then click **Functional Review** button
4. Work through instances (left panel):
   - Click an instance row → highlights in 3D, attributes load on right
   - Tick/untick attributes or use **Accept all / Reject all / Reset to pipeline**
   - Add a **Flag** or **Note** if needed
   - Click **Confirm ✓ & Next** to move on
5. Click **Save Review** often — it writes to `<SHARED>\reviews\<building>__<you>.review.json`

**Tips:**
- Filter by class or tick "unreviewed" to jump to what needs work
- Use *Colour by* → `val: <attr>` to spot outliers across the building
- Use *Colour by* → `conf: <attr>` to see pipeline confidence (0.5 = it was unsure)
- **Splitting work**: one building per person, or several people on the same building — everyone writes their own `__<name>` file

---

## Admin — export the reviewed dataset

When reviews are in, merge each reviewer's JSON with the pipeline `.h5` into a
reviewed `.h5`. Originals aren't touched — the reviewed file just adds your
decisions next to the pipeline ones.

All of one reviewer's buildings:

```powershell
python review_admin.py export-all --reviewer your-name `
  --func-root    "<FUNC_ROOT>" `
  --reviews-root "<SHARED>\reviews" `
  --out-root     "<SHARED>\functional_labels_reviewed"
```

One building:

```powershell
python review_admin.py export `
  --func   "<FUNC_ROOT>\train\HFX_BLD001_ZEB_CLEAN.h5" `
  --review "<SHARED>\reviews\HFX_BLD001_ZEB_CLEAN__your-name.review.json" `
  --out    "<SHARED>\functional_labels_reviewed\train"
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
import sys; sys.path.append(r"<path-to-this-repo>")
import cc_functional_review as R
R.main()
```

To reload after the script changes: `import importlib; importlib.reload(R); R.main()`.
