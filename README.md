# HFX3D functional-attribute review

Review HFX3D building facade attributes in CloudCompare. Simple workflow:

1. **Download** review clouds from the server
2. **Review** 15 attributes per instance in CloudCompare
3. **Upload** your JSON edits back to the server

---

## One-time setup

Do this once, then you can review any building.

### 1. Install CloudCompare 2.13 with Python plugin

- Download CloudCompare 2.13
- In the installer, tick **Python plugin**
- Open CloudCompare once — you should see a Python console/menu

### 2. Clone this repo (the plugin)

```powershell
git clone <repo-url>
cd hfx3d-functional-review
```

### 3. Add this repo as a CloudCompare custom plugin

- Open CloudCompare
- **Tools → Python → Edit plugin search paths** (or similar, varies by build)
- Add the full path to this cloned repo folder
- Restart CloudCompare
- A **Functional Review** button should appear in the toolbar

### 4. Set your reviewer name and review folder (once)

```powershell
setx HFX3D_REVIEWER "your-name"
setx HFX3D_REVIEW_ROOT "C:\path\to\where\you\save\reviews"
```

Close and reopen CloudCompare. Your name keeps your reviews separate from everyone else's, and the folder path tells the plugin where to save your JSON files.

---

## Start reviewing

The 15 attributes you'll review:

`load_bearing, thermal_envelope, vegetation_support, operable, solar_shading,
ventilation, natural_lighting, access, drainage, fall_protection, aesthetic,
privacy_screening, circulation, illumination, surveillance`

### Workflow

1. **Download** a review cloud from the server:
   - **AI server path:** `/data/images/lidar_data/Functional_Attribute/review_clouds`
   - Download the `.laz` file for your assigned building

2. **Open in CloudCompare**: File → Open → `HFX_BLDxxx_ZEB_CLEAN.laz`

3. Click the cloud in the tree (left), then click **Functional Review** button

4. Work through instances (left panel):
   - Click an instance row → it highlights in 3D, attributes load on right
   - Review the 15 attributes — tick ✓ if the instance has it, leave blank if not
   - Or use **Accept all** / **Reject all** / **Reset to pipeline** shortcuts
   - Add a **Flag** (`bad_segmentation` / `wrong_class` / `other`) if the instance itself looks wrong
   - Add a **Note** if useful
   - Click **Confirm ✓ & Next** to move on

5. Click **Save Review** when done — it saves your edits as a JSON file locally

### Tips

- Filter by class or instance ID, or tick "unreviewed" to jump to what needs work
- Use *Colour by* → `val: <attr>` to spot outliers across the building
- Use *Colour by* → `conf: <attr>` to see pipeline confidence (0.5 = very unsure, your call matters most)

---

## After reviewing — upload your JSON

When you finish a building, your edits are saved as a JSON file locally. Upload it back to the AI server.

**Your review file is:** `<building-name>__<your-name>.json`

**Upload to:** `/data/images/lidar_data/Functional_Attribute/reviews`


---

## Troubleshooting

- *"cloud has no instance_id / val_* field"* → wrong file; make sure you have a `.laz` review cloud
- Clicking a **point** doesn't work → use the **instance list** on the left instead (more reliable)
- Colours don't update after an edit → set *Colour by* to `val: <that attribute>`
- **Functional Review** button missing → finish step 3 above, restart CloudCompare
- Reviews save in the wrong place → check your `HFX3D_REVIEWER` env var is set; redo the `setx` step

**Alternative: launch via Python console** (if the button doesn't appear):

```python
import sys
sys.path.append(r"C:\path\to\hfx3d-functional-review")
import cc_functional_review as R
R.main()
```

To reload after changes: `import importlib; importlib.reload(R); R.main()`
