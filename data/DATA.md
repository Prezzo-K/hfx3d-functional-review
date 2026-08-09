# `data/` bundle — reviewer layout

This folder mirrors the team working folder and the AI server layout.

## What reviewers need

- `review_clouds/` — download these from the AI server and open them in CloudCompare
- `reviews/` — your JSON review files get saved here
- `functional_labels_reviewed/` — exported output is written here after review is merged

## AI server locations

- **Review clouds:** `/data/images/lidar_data/Functional_Attribute/review_clouds`
- **Review JSON upload folder:** `/data/images/lidar_data/Functional_Attribute/reviews`

## Local setup

Set `HFX3D_REVIEW_ROOT` to your local `reviews/` folder, for example:

```powershell
setx HFX3D_REVIEW_ROOT "C:\path\to\hfx3d-functional-review\data\reviews"
```

## Folder layout

```text
data/
├── review_clouds/             review .laz files downloaded from the AI server
│   └── train/ val/ test/  *.laz
├── reviews/                   review JSON files saved locally
│   └── <building>__<reviewer>.review.json
├── functional_labels_reviewed/ exported HDF5 after export/merge
│   └── train/ val/ test/  *.h5
├── functional_labels/         pipeline HDF5 inputs used by export
│   └── train/ val/ test/  *.h5
└── reference/                 read-only refs like `ontology_rules.yaml` and `label_map.json`
```

## Notes

- Reviewers only need `review_clouds/` and write access to `reviews/`.
- The admin/export step uses `functional_labels/` as input and writes to `functional_labels_reviewed/`.
