# HFX3D functional-attribute review

Human validation of HFX3D's automatic functional-attribute assignments, done
inside CloudCompare. Everyone on the team uses the same setup and the same
repo — clone it, install one requirements file, add it as a CloudCompare
plugin, and review your assigned building(s).

---

## One-time setup

Do this once, then you can review any building.

### 1. Install CloudCompare 2.13 with the Python plugin

- Download CloudCompare 2.13
- In the installer, tick **Python plugin**
- Open CloudCompare once — you should see a Python console/menu

### 2. Clone this repo and install dependencies

```powershell
git clone <repo-url>
cd hfx3d-functional-review
```

(Optional) create and activate a virtual environment first, if you'd rather
keep this isolated from other Python projects on your machine:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Then install the requirements:

```powershell
pip install -r requirements.txt
```

**Important:** this must land in whichever Python CloudCompare's Python
plugin is actually configured to use — check via CC's Python console
(Tools → Python → Python console, or similar) with `import sys;
print(sys.executable)`. If that's a different interpreter than the one
above (a system Python, or one CC bundles itself), run the `pip install`
against that interpreter instead — a venv only helps if you point
CloudCompare at it. This is what makes **Save Review** able to write the
reviewed `.h5` (it needs `h5py`) alongside the `.review.json`.

### 3. Add this repo as a CloudCompare custom plugin

- Open CloudCompare
- **Tools → Python → Edit plugin search paths** (wording varies by build)
- Add the full path to this cloned repo folder
- Restart CloudCompare
- A **Functional Review** button should appear in the toolbar

### 4. Set your name and where your files get saved (once)

The plugin reads three environment variables — set them once and every
future review picks them up automatically:

- **`HFX3D_REVIEWER`** — your name. It gets stamped into your files
  (`<building>__your-name.review.json`), so your review of a building never
  overwrites a teammate's review of the *same* building.
- **`HFX3D_REVIEW_ROOT`** — the folder **Save Review** writes your
  `.review.json` to. This is the small, editable file — your actual ticks,
  flags and notes — and it's also what gets *read back* the next time you
  open that building (so if CloudCompare crashes, reopening the same `.laz`
  and clicking Functional Review restores everything you'd already saved).
- **`HFX3D_EXPORT_ROOT`** — the folder **Save Review** writes your
  `.reviewed.h5` to. This is the larger binary file built *from* the JSON,
  meant for upload, not for editing.If you don't set it, it just defaults to `HFX3D_REVIEW_ROOT`.

```powershell
setx HFX3D_REVIEWER "your-name"
setx HFX3D_REVIEW_ROOT "C:\path\to\where\you\save\reviews"
setx HFX3D_EXPORT_ROOT "C:\path\to\where\you\save\reviews"
```

Close and reopen CloudCompare after running these — `setx` only takes effect
in new processes.

---

## Start reviewing

**Before your first review**, skim `ontology_rules.yaml` from the bundle Tao
shared: <https://drive.google.com/file/d/1Wk1MZfkADI39etUeDtXXbmE1BfMc4GJ7/view?usp=drive_link>.
It defines what each of the 15 attributes actually means, which IFC/AAT
standard backs it, and the geometric condition the pipeline used to suggest
a value — that's the context your judgment call needs. The main thing to
watch for: the same semantic class can get different attributes depending
on geometry (e.g. stairs with railings get `fall_protection`, stairs
without don't) — that's the actual point of this review, not an edge case
to explain away.

The 15 attributes you'll review:

`load_bearing, thermal_envelope, vegetation_support, operable, solar_shading,
ventilation, natural_lighting, access, drainage, fall_protection, aesthetic,
privacy_screening, circulation, illumination, surveillance`

### Workflow

1. **Download** the review cloud for your assigned building:
   - **AI server path:** `/data/images/lidar_data/Functional_Attribute/review_clouds`
   - Download the `.laz` file, e.g. `HFX_BLDxxx_ZEB_CLEAN.laz`

2. **Open in CloudCompare**: File → Open → `HFX_BLDxxx_ZEB_CLEAN.laz`

3. Click the cloud in the tree (left), then click **Functional Review**.

4. Work through instances (left panel):
   - Click an instance row → it highlights in 3D, attributes load on the right
   - Review the 15 attributes — tick ✓ if the instance has it, leave blank if not
   - Or use **Accept all** / **Reject all** / **Reset to pipeline** shortcuts
   - Add a **Flag** (`bad_segmentation` / `wrong_class` / `other`) if the instance itself looks wrong
   - Add a **Note** if useful
   - Click **Confirm ✓ & Next** to move on

5. Click **Save Review** when done (and periodically while working). This
   writes **both** files in one click:
   - `<building>__<you>.review.json`
   - `<building>__<you>.reviewed.h5`

   You'll get a popup confirming both paths, or a warning if the `.h5` part
   failed (see Troubleshooting).

### Tips

- Filter by class or instance ID, or tick "unreviewed" to jump to what needs work
- Use *Colour by* → `val: <attr>` to spot outliers across the building
- Use *Colour by* → `conf: <attr>` to see pipeline confidence (0.5 = very unsure, your call matters most)

---

## After reviewing — upload

Upload **both** files back to the AI server:

**Upload to:** `/data/images/lidar_data/Functional_Attribute/reviews`

- `<building>__<you>.review.json`
- `<building>__<you>.reviewed.h5`

---

## Troubleshooting

- *"cloud has no instance_id / val_* field"* → wrong file; make sure you have a `.laz` review cloud built by `build_review_cloud.py`
- Clicking a **point** doesn't work → use the **instance list** on the left instead (more reliable)
- Colours don't update after an edit → set *Colour by* to `val: <that attribute>`
- **Functional Review** button missing → finish step 3 above, restart CloudCompare
- Reviews save in the wrong place → check `HFX3D_REVIEWER` / `HFX3D_REVIEW_ROOT` / `HFX3D_EXPORT_ROOT` are set; redo the `setx` steps
- Save writes the JSON but warns it couldn't write the `.h5` → `h5py` isn't importable from CloudCompare's Python; `pip install h5py` there and Save again (the JSON is still valid on its own if you need to upload before fixing this)

**Alternative: launch via Python console** (if the button doesn't appear):

```python
import sys
sys.path.append(r"C:\path\to\hfx3d-functional-review")
import cc_functional_review as R
R.main()
```

To reload after changes: `import importlib; importlib.reload(R); R.main()`
