# HFX3D functional-attribute review

Human validation of HFX3D's automatic functional-attribute assignments, done
inside CloudCompare. Everyone on the team uses the same setup and the same
repo — clone it, install one requirements file, add it as a CloudCompare
plugin, and review your assigned building(s).

---

## One-time setup

Do this once, then you can review any building.

### 1. Install CloudCompare with the Python plugin

- Download CloudCompare 2.13 or newer (2.14 beta works too)
- In the installer, tick **Python plugin** (a.k.a. *Python Runtime*)
- Open CloudCompare once and confirm you have a **Plugins → Python Plugin**
  menu (older builds: **Tools → Python**). The exact menu wording below is
  for 2.14's *Python Plugin* menu; on 2.13 the same items live under Tools.

### 2. Clone this repo

```powershell
git clone <repo-url>
cd hfx3d-functional-review
```

Nothing to install yet — the next step handles it.

### 3. Point CloudCompare at the plugin folder

- Open **Plugins → Python Plugin → Show Settings**
  *(on 2.13: Tools → Python → settings/plugin search paths)*
- Add the path to the **`plugin/` subfolder inside this repo** — i.e.
  `...\hfx3d-functional-review\plugin`, **not** the repo root.

  > Point CloudCompare at `plugin/`, not the repo root. The `plugin/` folder
  > contains only the plugin and its `requirements.txt`, so CloudCompare has
  > nothing else to try to import. If you add the repo root instead, CC tries
  > to import `.gitignore`, `.idea`, `.venv`, `README.md`, etc. as Python
  > modules and floods the log with `ModuleNotFoundError` at every startup.

- Restart CloudCompare
- A **Functional Review** button should appear in the toolbar (also runnable
  from **Plugins → Python Plugin → Show Action Launcher**)

### 4. Install the Python packages the plugin needs

The plugin needs `numpy`, `h5py` (for the `.h5` export on Save), and a **Qt
binding** — all in **CloudCompare's own Python**, not your system Python.

> ⚠️ **The Qt binding must match CloudCompare's Qt version.** CloudCompare
> **2.14 is built on Qt6** → install **PyQt6**. CloudCompare **2.13 is Qt5** →
> install **PyQt5**. Installing the *wrong* one loads a second, incompatible
> Qt into the process and **crashes the whole app** the moment you load a
> cloud — no error message, CloudCompare just closes. This is the #1 gotcha.
>
> To check your Qt version: the title bar shows the CC version, or look at
> `C:\Program Files\CloudCompare\Qt6Core.dll` (Qt6 = 2.14) vs `Qt5Core.dll`.

Easiest route — **Plugins → Python Plugin → Package Manager** — install:

- **`numpy`** and **`h5py`** (always)
- on **2.14**: **`PyQt6`** — pin it to your CC's Qt6 minor version, e.g.
  `PyQt6==6.8.1` + `PyQt6-Qt6==6.8.2` if `Qt6Core.dll` reports 6.8.x
- on **2.13**: **`PyQt5`**

Then **restart CloudCompare**.

The per-plugin **virtual-environment popup** (from `requirements.txt`) installs
`numpy`/`h5py` automatically but deliberately **not** the Qt binding (so it
can't install the wrong one and crash you) — add PyQt6/PyQt5 yourself as above.
If the popup is flaky and the plugin fails with `No module named 'numpy'`, use
the Package Manager or the manual fallback below.

**Manual fallback** — install against the exact interpreter CC uses. In
**Plugins → Python Plugin → Show REPL** (or Show Editor), run:

```python
import sys; print(sys.executable)
```

then in a normal PowerShell window:

```powershell
& "<path-printed-by-CC>" -m pip install -r "C:\path\to\hfx3d-functional-review\plugin\requirements.txt"
```

> Note: the `.venv` folder that may appear in the repo root is a normal
> developer/PyCharm venv — CloudCompare does **not** use it. Only the
> interpreter CC's REPL prints is the one that matters.

### 5. Set your name and where your files get saved (once)

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
without don't) — that's the actual point of this review and see if something is off and needs changing i.e If an attribute shouldn't be assigned to a specific instance.

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
- **Functional Review** button missing → finish steps 3–4 above (plugin path
  + packages), restart CloudCompare; check the Console for a load error
- Plugin fails to load with `No module named 'numpy'` (or `h5py`/`PyQt5`) →
  they aren't installed in **CloudCompare's own Python venv**. Install them via
  **Plugins → Python Plugin → Package Manager**, or against the interpreter the
  Console prints (`import sys; print(sys.executable)`), then restart.
- Review cloud takes minutes to load / stalls / crashes CloudCompare → the
  `.laz` is huge (tens of millions of points × 30+ scalar fields). Shrink it
  once with `tools/downsample_review_cloud.py` and open the downsampled file
  instead — see [`tools/README.md`](tools/README.md).
- Reviews save in the wrong place → check `HFX3D_REVIEWER` / `HFX3D_REVIEW_ROOT` / `HFX3D_EXPORT_ROOT` are set; redo the `setx` steps
- Save writes the JSON but warns it couldn't write the `.h5` → `h5py` isn't importable from CloudCompare's Python; `pip install h5py` there and Save again (the JSON is still valid on its own if you need to upload before fixing this)

**Alternative: launch via Python console** (if the button doesn't appear):

```python
import sys
sys.path.append(r"C:\path\to\hfx3d-functional-review\plugin")
import cc_functional_review as R
R.main()
```

To reload after changes: `import importlib; importlib.reload(R); R.main()`
