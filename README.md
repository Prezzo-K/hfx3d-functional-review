# HFX3D Functional-Attribute Review — Team Pipeline

Human-in-the-loop validation of the automatic functional-attribute labels, done
**inside CloudCompare**. Reviewers open one point cloud per building, click an
instance, and confirm/correct its 15 functional attributes. Nothing in the
original dataset is ever modified — every decision is written to a separate
review file and later exported to a reviewed dataset.

```
 pipeline output (functional_labels/*.h5)
            │
   [ADMIN] build_review_cloud.py  ─────────►  review_clouds/<split>/*.laz   (open in CloudCompare)
            │
   [REVIEWER] CloudCompare + "Functional Review" panel
            │  edits, per person
            ▼
        reviews/<building>__<reviewer>.review.json      (your decisions — safe, resumable)
            │
   [ADMIN] export_reviewed.py  ─────────────►  functional_labels_reviewed/<split>/*.h5   (for training/eval)
```

There are **two roles**: an **Admin** who sets the shared data up once, and
**Reviewers** who do the labelling. Follow the section for your role.

---

## 0. Fixed vocabulary (what the columns mean)

15 functional attributes, in this fixed order:

`load_bearing, thermal_envelope, vegetation_support, operable, solar_shading,
ventilation, natural_lighting, access, drainage, fall_protection, aesthetic,
privacy_screening, circulation, illumination, surveillance`

Instance flags (one per instance, optional): `bad_segmentation`, `wrong_class`,
`other`. Use these when the *instance itself* is wrong (bad geometry / wrong
semantic class), not the attributes.

---

## 1. Where everything lives (canonical locations)

Pick a **shared drive** everyone can reach (network share or synced folder).
In the examples below it is `S:\HFX3D`. Replace `S:\HFX3D` with your real
shared path everywhere. The layout is:

```
S:\HFX3D\
├── review_clouds\           # the LAZ clouds reviewers open (built by Admin)
│   ├── train\  HFX_BLD001_ZEB_CLEAN.laz ...
│   ├── val\    ...
│   └── test\   ...
├── reviews\                 # review JSON files, one per building PER reviewer
│   ├── HFX_BLD001_ZEB_CLEAN__abdi.review.json
│   └── HFX_BLD001_ZEB_CLEAN__sara.review.json
└── functional_labels_reviewed\   # exported reviewed .h5 (built by Admin)
    ├── train\  HFX_BLD001_ZEB_CLEAN.h5 ...
    └── ...
```

Source data used only by the Admin build step (can stay on the Admin machine):

```
functional_labels\<split>\<building>.h5        # pipeline output (per-instance attributes)
instances_vis\<split>\<building>_instances_vis.ply   # cloud carrying per-point instance ids
```

The **tools** (Python scripts) all live in **one self-contained repo**:

```
C:\Users\<you>\PycharmProjects\hfx3d-functional-review\
├── review_admin.py           # Admin: ONE command → build clouds / export (recommended)
├── cc_functional_review.py   # Reviewer: the CloudCompare panel (add this folder to CC)
├── build_review_cloud.py     # (used by review_admin; also runnable directly)
├── export_reviewed.py        # (used by review_admin; also runnable directly)
├── functional.py             # shared library
├── cloud_io.py               # shared library (LAS/PLY reader)
├── review_editor.py          # optional standalone table editor (no CloudCompare)
├── requirements.txt
└── README.md                 # this file
```

Every script adds its own folder to `sys.path`, so you can run them from any
working directory and `import functional` will never fail — even if you `cd`
elsewhere first.

> **Two argument gotchas that cause errors:**
> `--func` is always the pipeline **`.h5`** (`functional_labels/<split>/<b>.h5`),
> **never** the review `.laz`. `--out` may be a **folder** (the filename is
> derived) or a full `.h5` path.

---

## 2. ADMIN — one-time setup

### 2.1 Python environment (Admin machine only)

The build/export scripts run in this repo's virtual environment (Python 3.10+):

```powershell
cd C:\Users\<you>\PycharmProjects\hfx3d-functional-review
python -m venv .venv                     # if it doesn't exist yet
.venv\Scripts\activate
pip install -r requirements.txt
```

(`lazrs` gives compressed `.laz`; without it you get larger `.las` — still works.)

### 2.2 Build the review clouds for every building

This bakes each building's current attributes into one LAZ. **Always keep the
confidence fields** — the panel needs them (`review_admin` includes them by
default; `build_review_cloud.py` needs `--with-conf`).

**Recommended — one command for all buildings:**

```powershell
.venv\Scripts\python.exe review_admin.py build-all `
  --inst-root "C:\...\HFX3D_Instance+Semantic\instances_vis" `
  --func-root "C:\...\results\functional_labels" `
  --out-root  "S:\HFX3D\review_clouds"
```

Single building (equivalent, explicit):

```powershell
.venv\Scripts\python.exe build_review_cloud.py `
  --orig "C:\...\instances_vis\train\HFX_BLD001_ZEB_CLEAN_instances_vis.ply" `
  --func "C:\...\functional_labels\train\HFX_BLD001_ZEB_CLEAN.h5" `
  --out  "S:\HFX3D\review_clouds\train\HFX_BLD001_ZEB_CLEAN.laz" `
  --with-conf
```

All buildings at once (edit the three roots, then paste into PowerShell):

```powershell
$INST = "C:\...\HFX3D_Instance+Semantic\instances_vis"
$FUNC = "C:\...\results\functional_labels"
$OUT  = "S:\HFX3D\review_clouds"
Get-ChildItem $INST -Recurse -Filter *_instances_vis.ply | ForEach-Object {
  $split = $_.Directory.Name
  $stem  = $_.BaseName -replace "_instances_vis$",""
  $func  = Join-Path $FUNC "$split\$stem.h5"
  if (Test-Path $func) {
    .venv\Scripts\python.exe build_review_cloud.py `
      --orig $_.FullName --func $func `
      --out (Join-Path $OUT "$split\$stem.laz") --with-conf
  }
}
```

Copy the resulting `review_clouds\` to the shared drive (`S:\HFX3D\review_clouds`).

### 2.3 Tell reviewers what to install

Send every reviewer **Section 3** below, plus the shared path `S:\HFX3D`.

---

## 3. REVIEWER — one-time setup (permanent, per person)

Do this **once** on each reviewer's PC. After it, reviewing is just: open a
cloud → click the toolbar button → work.

### 3.1 Install CloudCompare 2.13 **with the Python plugin**

1. Download CloudCompare 2.13 (Windows) from https://cloudcompare.org.
2. Run the installer and **tick "Python plugin" (CloudCompare Python Runtime)**
   in the components list. Finish the install.
3. Launch CloudCompare. You should see a **Python** entry (console/editor). If
   you do not, re-run the installer and enable the plugin.

### 3.2 Make the "Functional Review" button permanent

So it appears as a real toolbar/menu action (no typing in the console each time):

1. Put the `hfx3d-functional-review` folder on the reviewer PC (e.g.
   `C:\HFX3D\tools\hfx3d-functional-review\`). Only `cc_functional_review.py`
   is needed by CloudCompare, but keeping the whole folder is fine.
2. In CloudCompare open the **Python plugin settings** (the Python plugin
   toolbar button → its settings/gear, labelled something like *"Manage custom
   Python plugins / scripts path"*).
3. Add the folder `C:\HFX3D\tools` as a custom-plugins path.
4. **Restart CloudCompare.** A **"Functional Review"** action now appears in the
   Python plugin menu/toolbar.

> If your build has no such setting, use the console fallback in Section 6.

### 3.3 Set your identity and the shared folder (permanent env vars)

These tell the panel **who you are** (stamped into your files, keeps them
separate from teammates') and **where to write reviews**. Set once via `setx`
in a terminal (or System Properties → Environment Variables):

```powershell
setx HFX3D_REVIEWER "abdi"
setx HFX3D_REVIEW_ROOT "S:\HFX3D\reviews"
```

- `HFX3D_REVIEWER` — your short name (lowercase, no spaces is easiest).
- `HFX3D_REVIEW_ROOT` — the shared reviews folder (must be the same for everyone).

**Close and reopen CloudCompare** after `setx` (env vars load at startup).

---

## 4. REVIEWER — daily workflow

1. In CloudCompare: **File → Open** the building from
   `S:\HFX3D\review_clouds\<split>\HFX_BLDxxx_ZEB_CLEAN.laz`.
2. **Click the cloud** in the DB Tree (left) to select it.
3. Click the **Functional Review** toolbar button. The panel opens.
4. Review, one instance at a time:
   - The **left list** shows every instance (`#id · class · N on · ✓ reviewed · ⚑ flagged`).
     Filter with the **class** dropdown, **unreviewed** / **flagged** checkboxes,
     or the **search** box.
   - **Click a list row** → that instance is **highlighted in the 3D view** and
     its attributes load on the right.
   - Judge it in 3D, then in the attribute table **tick/untick** each attribute,
     or use **Accept all / Reject all / Reset to pipeline**. Non-applicable
     attributes are greyed but you can still tick them.
   - Optionally set a **Flag** (bad segmentation / wrong class) and a **Note**.
   - Click **Confirm ✓ & Next** to mark it reviewed and jump to the next
     unreviewed instance. (Any edit also auto-marks it reviewed.)
5. Click **Save Review** regularly (and before closing). Your file is written to
   `S:\HFX3D\reviews\<building>__<yourname>.review.json`.

The counter (bottom-left) shows `reviewed / total` so you can see progress.
"Reviewed" is only a coverage tracker — it does **not** change any attribute.

**Splitting work across the team:** either assign one building per person, or
have several people review the same building — each person's decisions go to
their own `__<name>.review.json`, so there is no clobbering.

---

## 5. Comparing / QC while reviewing

Use the **Colour by** dropdown at the top of the panel:

- **Highlight selected** — the picked instance is bright, everything else dim
  (default; best for locating the instance you're judging).
- **val: `<attribute>`** — colours the whole building by the **current decision**
  (1 = on) for that attribute. Great for spotting outliers ("why is this wall
  marked operable?").
- **conf: `<attribute>`** — colours by the **pipeline confidence** (0→1). Compare
  `val` vs `conf`: where confidence is near 0.5 the pipeline was unsure and your
  judgement matters most.

The list's `N on` count and, after export, the `changed` field let you see where
humans disagreed with the pipeline.

To compare **two reviewers** on the same building: export each of their review
files (Section 7) and diff the `functional_attribute_vector_human` arrays.

---

## 6. Console fallback (if the toolbar button isn't set up)

Open the CloudCompare **Python console**, select the cloud, and run:

```python
import sys; sys.path.append(r"C:\HFX3D\tools\hfx3d-functional-review")
import cc_functional_review as R
R.main()
```

To reload after the script is updated:

```python
import importlib, cc_functional_review as R
importlib.reload(R); R.main()
```

---

## 7. ADMIN — export the reviewed dataset

When a building's review is done, merge the reviewer's JSON with the pipeline
`.h5` into a reviewed `.h5` (originals are never touched).

**Recommended — all of one reviewer's files at once:**

```powershell
.venv\Scripts\python.exe review_admin.py export-all --reviewer abdi `
  --func-root    "C:\...\results\functional_labels" `
  --reviews-root "S:\HFX3D\reviews" `
  --out-root     "S:\HFX3D\functional_labels_reviewed"
```

Single building (explicit; `--out` may be a file or a folder):

```powershell
.venv\Scripts\python.exe export_reviewed.py `
  --func   "C:\...\functional_labels\train\HFX_BLD001_ZEB_CLEAN.h5" `
  --review "S:\HFX3D\reviews\HFX_BLD001_ZEB_CLEAN__abdi.review.json" `
  --out    "S:\HFX3D\functional_labels_reviewed\train"
```

Batch all of one reviewer's files:

```powershell
$FUNC = "C:\...\results\functional_labels"
$REV  = "S:\HFX3D\reviews"
$OUT  = "S:\HFX3D\functional_labels_reviewed"
Get-ChildItem $REV -Filter *__abdi.review.json | ForEach-Object {
  $stem = $_.BaseName -replace "__abdi\.review$",""
  $func = Get-ChildItem $FUNC -Recurse -Filter "$stem.h5" | Select-Object -First 1
  if ($func) {
    $split = $func.Directory.Name
    .venv\Scripts\python.exe export_reviewed.py `
      --func $func.FullName --review $_.FullName `
      --out (Join-Path $OUT "$split\$stem.h5")
  }
}
```

### What the reviewed `.h5` contains

Same schema as the pipeline output (same instance order), **plus** the human layer:

| dataset | meaning |
|---|---|
| `functional_attribute_vector` | original **pipeline** suggestion (kept for provenance) |
| `functional_attribute_vector_human` | **reviewed decision** — use this for training/eval |
| `review_status` | `reviewed` / `unreviewed` per instance |
| `instance_flag` | `bad_segmentation` / `wrong_class` / `other` / empty |
| `review_note` | free-text note |
| `functional_attribute_confidence`, `applicable_mask`, `instance_id`, `semantic_class`, … | copied through unchanged |

Downstream code reads **`functional_attribute_vector_human`**.

### Re-baking a review cloud (optional)

To colour a review cloud by the *edited* values (e.g. for a QA screenshot), run
`build_review_cloud.py` again with `--review <that json>` — it overlays the human
decisions onto the `val_*` fields.

---

## 8. Golden rules (what changes, what never does)

- **Never modified:** `HFX3D_Instance+Semantic\*.h5` (points/semantic/instance)
  and `functional_labels\*.h5` (pipeline output).
- **Your edits live in:** `reviews\<building>__<you>.review.json` (written on Save).
- **The training file is:** `functional_labels_reviewed\<split>\<building>.h5`
  (created by `export_reviewed.py`).
- In CloudCompare, editing changes only the on-screen colours + your review file;
  the `.laz` on disk is not rewritten unless the Admin re-bakes it.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Panel says *"cloud has no instance_id / val_* field"* | You opened the wrong cloud. Open a `review_clouds\...laz` built with `--with-conf`. |
| Clicking a **point** doesn't select an instance | Known: point-picking varies by build. Use the **left list** instead (that's the intended way). |
| Colours don't refresh after an edit | Set **Colour by** to `val: <that attribute>`; it recolours on edit. Otherwise your edit is still saved. |
| "Functional Review" button missing | Finish Section 3.2, restart CloudCompare, or use the console fallback (Section 6). |
| Reviews saving to the wrong place / name is "unassigned" | `HFX3D_REVIEWER` / `HFX3D_REVIEW_ROOT` not set — redo Section 3.3 and **restart CloudCompare**. |
| `h5py` errors inside CloudCompare | Not needed there — the panel uses only NumPy. `h5py` is only for the Admin build/export scripts. |

---

## 10. Quick reference

**Reviewer, every session:** open `review_clouds\<split>\<building>.laz` → select it →
**Functional Review** → click list rows, edit, **Confirm ✓ & Next** → **Save Review**.

**Admin, per batch:** `build_review_cloud.py --with-conf` (make LAZs) →
reviewers work → `export_reviewed.py` (make reviewed `.h5`).
