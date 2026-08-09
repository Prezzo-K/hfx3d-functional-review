"""
cloud_io.py — format-agnostic point-cloud reader + instance-colour helpers.

Supports:
  • .las / .laz   via laspy   (lazy-imported)
  • .ply          binary little/big-endian and ascii, via numpy (no heavy deps)

read_points() returns a plain dict so callers (the viewer and the video
renderer) share one loader and one instance-colour scheme.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

# Instance id used for "not assigned to any instance" (background / noise).
INSTANCE_UNLABELED = -1

# ── PLY parsing ────────────────────────────────────────────────────────────────

_PLY_NP = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _read_ply(path: Path):
    """Return (structured ndarray of the vertex element, list of property names)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")

        fmt = None
        n_vertex = None
        props: list[tuple[str, str]] = []
        in_vertex = False

        while True:
            raw = f.readline()
            if not raw:
                raise ValueError("unexpected EOF in PLY header")
            s = raw.decode("latin-1").strip()
            if s.startswith("format"):
                fmt = s.split()[1]
            elif s.startswith("element"):
                parts = s.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n_vertex = int(parts[2])
            elif s.startswith("property") and in_vertex:
                parts = s.split()
                if parts[1] == "list":
                    raise ValueError("list properties on vertex are not supported")
                props.append((parts[2], parts[1]))   # (name, type)
            elif s == "end_header":
                break

        if n_vertex is None:
            raise ValueError("no vertex element found in PLY header")

        if fmt in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if fmt.endswith("little_endian") else ">"
            dt = np.dtype([(nm, endian + _PLY_NP[ty]) for nm, ty in props])
            arr = np.fromfile(f, dtype=dt, count=n_vertex)
        elif fmt == "ascii":
            dt = np.dtype([(nm, _PLY_NP[ty]) for nm, ty in props])
            arr = np.loadtxt(f, dtype=dt, max_rows=n_vertex)
        else:
            raise ValueError(f"unsupported PLY format: {fmt}")

    return arr, [nm for nm, _ in props]


def _find(names, *candidates):
    """Case-insensitive lookup of the first matching field name."""
    low = {nm.lower(): nm for nm in names}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


# ── unified reader ──────────────────────────────────────────────────────────────

def read_points(path) -> dict:
    """
    Load a point cloud and return:
        xyz        (N, 3) float32
        labels     (N,)   int32     semantic class (0 if absent)
        instance   (N,)   int32 or None
        rgb        (N, 3) float32 in 0..1, or None
        intensity  (N,)   float32 (0..1 normalised), or None
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".las", ".laz"):
        import laspy
        las = laspy.read(str(path))
        xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
        labels = np.array(las.classification, dtype=np.int32)
        rgb = None
        if hasattr(las, "red"):
            rgb = (np.column_stack([las.red, las.green, las.blue])
                   .astype(np.float32) / 65535.0)
        inten = None
        if hasattr(las, "intensity"):
            inten = _norm_intensity(np.array(las.intensity, dtype=np.float32))
        return dict(xyz=xyz, labels=labels, instance=None, rgb=rgb, intensity=inten)

    if ext == ".ply":
        arr, names = _read_ply(path)
        xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)

        cls_f = _find(names, "scalar_Classification", "classification", "class", "label")
        labels = (arr[cls_f].astype(np.int32) if cls_f
                  else np.zeros(len(xyz), dtype=np.int32))

        inst_f = _find(names, "scalar_Instance", "instance")
        instance = arr[inst_f].astype(np.int32) if inst_f else None

        rf, gf, bf = _find(names, "red", "r"), _find(names, "green", "g"), _find(names, "blue", "b")
        rgb = None
        if rf and gf and bf:
            rgb = np.column_stack([arr[rf], arr[gf], arr[bf]]).astype(np.float32)
            if rgb.max() > 1.0:            # uchar 0..255 → 0..1
                rgb /= 255.0

        int_f = _find(names, "scalar_Intensity", "intensity")
        inten = _norm_intensity(arr[int_f].astype(np.float32)) if int_f else None

        return dict(xyz=xyz, labels=labels, instance=instance, rgb=rgb, intensity=inten)

    raise ValueError(f"unsupported file type: {ext}")


def has_instance(path) -> bool:
    """Cheaply report whether a cloud carries per-point instance ids.

    Reads only the PLY header (not the point data); .las/.laz never carry
    instances in this pipeline.
    """
    path = Path(path)
    if path.suffix.lower() != ".ply":
        return False
    try:
        with open(path, "rb") as f:
            if f.readline().strip() != b"ply":
                return False
            for _ in range(200):
                raw = f.readline()
                if not raw:
                    break
                s = raw.decode("latin-1").strip()
                if s == "end_header":
                    break
                if s.startswith("property") and "instance" in s.lower():
                    return True
    except OSError:
        return False
    return False


def _norm_intensity(inten: np.ndarray) -> np.ndarray:
    """Percentile-stretch intensity into 0..1."""
    p1, p99 = float(np.percentile(inten, 1)), float(np.percentile(inten, 99))
    return np.clip((inten - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)


# ── deterministic instance colours ──────────────────────────────────────────────

_INSTANCE_LUT: np.ndarray | None = None
_GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def instance_lut(n: int = 256) -> np.ndarray:
    """
    A fixed (n, 3) qualitative palette. Consecutive entries are ~0.618 turns
    apart in hue (golden-ratio walk), so neighbouring instance ids are always
    visually distinct. Built once and cached — identical every run.
    """
    global _INSTANCE_LUT
    if _INSTANCE_LUT is None or len(_INSTANCE_LUT) != n:
        import colorsys
        golden = 0.618033988749895
        lut = np.empty((n, 3), dtype=np.float32)
        h = 0.0
        for i in range(n):
            lut[i] = colorsys.hsv_to_rgb(h, 0.65, 0.95)
            h = (h + golden) % 1.0
        _INSTANCE_LUT = lut
    return _INSTANCE_LUT


def instance_colors(ids: np.ndarray) -> np.ndarray:
    """
    (N,) instance ids → (N, 3) float32 RGB.

    Deterministic: the same id always maps to the same colour (across runs,
    files, and GT-vs-prediction). INSTANCE_UNLABELED (-1) → neutral grey.
    """
    ids = np.asarray(ids)
    lut = instance_lut()
    cols = lut[np.mod(ids.astype(np.int64), len(lut))].astype(np.float32).copy()
    cols[ids < 0] = _GRAY
    return cols
