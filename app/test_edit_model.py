#!/usr/bin/env python3
"""Headless checks for edit_model.EditModel against a real bundle (no Qt)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_model import EditModel, CLASS_APPLICABLE, sem_name


class MiniBundle:
    """Just the fields EditModel needs — avoids importing the Qt app."""
    def __init__(self, folder):
        f = Path(folder)
        self.points = np.load(f / "points.npy", mmap_mode="r")
        self.offsets = np.load(f / "offsets.npy")
        self.src_index = np.load(f / "src_index.npy")
        m = np.load(f / "meta.npz", allow_pickle=True)
        self.ids = [int(x) for x in m["instance_id"]]
        self.sem = m["semantic_id"]
        self.conf = m["conf"].astype(np.float32)
        self.attr_names = [str(x) for x in m["attribute_names"]]
        self.max_id = max(self.ids)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "bundles/HFX_BLD001_ZEB_CLEAN"
    b = MiniBundle(folder)
    em = EditModel(b)
    fails = []

    # 1. membership partitions all points exactly once
    total = sum(em.count(i) for i in em.order)
    assert total == len(b.points), f"membership total {total} != {len(b.points)}"
    allrows = np.concatenate([em.rows[i] for i in em.order]); allrows.sort()
    assert np.array_equal(allrows, np.arange(len(b.points))), "rows not a clean partition"
    print(f"[ok] partition: {len(em.order)} instances cover all {len(b.points):,} points once")

    # 2. ontology applicability vs pipeline conf<0 encoding (unedited)
    mism = 0
    for k, iid in enumerate(b.ids):
        conf_appl = np.isfinite(b.conf[k]) & (b.conf[k] >= 0.0)
        # attributes with conf exactly nan are "no evidence" not "N/A"; the app
        # treats nan as applicable. Compare only the definitely-N/A (conf<0) cells.
        na_conf = np.isfinite(b.conf[k]) & (b.conf[k] < 0.0)
        onto_appl = em.applicable_mask(iid, b.attr_names)
        # every conf<0 cell should be non-applicable under the ontology
        bad = np.where(na_conf & onto_appl)[0]
        if len(bad):
            mism += 1
    print(f"[{'ok' if mism == 0 else 'WARN'}] ontology vs conf<0: {mism} instances "
          f"where a conf<0 (N/A) cell is ontology-applicable")

    # 3. reclass + undo (pick a target class different from current)
    iid = b.ids[10]
    old = em.cls(iid)
    target = "door" if old != "door" else "wall"
    em.reclass(iid, SEM_NAMES_index(target))
    assert em.cls(iid) == target, "reclass failed"
    assert iid in em.edited
    em.undo()
    assert em.cls(iid) == old and iid not in em.edited, "reclass undo failed"
    print(f"[ok] reclass #{iid} {old}->{target} then undo -> {em.cls(iid)}")

    # 4. split + undo (move half of a mid-size instance)
    iid = max(b.ids, key=lambda i: em.count(i) if em.count(i) < 200000 else 0)
    parent_rows = em.rows[iid].copy()
    half = parent_rows[: len(parent_rows) // 2]
    child = em.split(iid, half, new_sem=SEM_NAMES_index("window"))
    assert child is not None and child == b.max_id + 1
    assert em.count(iid) + em.count(child) == len(parent_rows), "split lost/dup points"
    assert set(em.rows[child].tolist()) == set(half.tolist()), "child rows wrong"
    assert em.cls(child) == "window"
    print(f"[ok] split #{iid} -> child #{child} ({em.count(child):,} pts), parent {em.count(iid):,}")
    em.undo()
    assert np.array_equal(em.rows[iid], parent_rows), "split undo did not restore parent"
    assert child not in em.rows and child not in em.order, "split undo left child"
    print(f"[ok] split undo restored parent #{iid} ({em.count(iid):,} pts), child gone")

    # 5. merge + undo (keep dominant)
    a, c = b.ids[0], b.ids[1]
    ca, cc = em.count(a), em.count(c)
    dom = a if ca >= cc else c
    order_before = list(em.order)
    survivor = em.merge([a, c])
    assert survivor == dom, f"merge survivor {survivor} != dominant {dom}"
    assert em.count(survivor) == ca + cc, "merge point count wrong"
    gone = c if dom == a else a
    assert gone not in em.rows and gone not in em.order, "merged-away id lingered"
    print(f"[ok] merge #{a}+#{c} -> #{survivor} ({em.count(survivor):,} pts, dominant kept)")
    em.undo()
    assert em.order == order_before and em.count(a) == ca and em.count(c) == cc, "merge undo failed"
    print(f"[ok] merge undo restored both instances")

    # 6. per_point_labels integrity after a real edit sequence
    em.reclass(b.ids[3], SEM_NAMES_index("window"))
    child = em.split(b.ids[4], em.rows[b.ids[4]][: em.count(b.ids[4]) // 3])
    src, inst, sem = em.per_point_labels()
    assert len(src) == len(inst) == len(sem) == len(b.points)
    assert len(np.unique(src)) == len(src), "src rows duplicated"
    # every point's label matches its instance's current class
    for iid in em.order:
        r = em.rows[iid]
        assert np.all(inst[r] == iid) and np.all(sem[r] == em.sem[iid]), "label mismatch"
    print(f"[ok] per_point_labels: {len(src):,} rows, labels consistent after reclass+split")

    # 7. persistence round-trip (compact diff sidecar)
    import tempfile, os
    em2 = EditModel(b)
    em2.reclass(b.ids[3], SEM_NAMES_index("window"))
    em2.merge([b.ids[6], b.ids[7]])
    ch = em2.split(b.ids[8], em2.rows[b.ids[8]][: em2.count(b.ids[8]) // 4])
    tmp = Path(tempfile.gettempdir()) / "_em_edits.npz"
    em2.save_edits(tmp)
    em3 = EditModel(b)
    em3.load_edits(tmp)
    assert em3.order == em2.order, "order not restored"
    assert em3.sem == em2.sem, "sem not restored"
    for iid in em2.order:
        assert set(em3.rows[iid].tolist()) == set(em2.rows[iid].tolist()), f"rows differ for {iid}"
    assert em3.next_id == em2.next_id and em3.edited == em2.edited
    s2 = em2.per_point_labels(); s3 = em3.per_point_labels()
    assert np.array_equal(s2[1], s3[1]) and np.array_equal(s2[2], s3[2]), "labels differ after reload"
    os.remove(tmp)
    print(f"[ok] persistence round-trip: {len(em2.order)} instances, "
          f"{em2.edited.__len__()} edited, labels identical after reload")

    # 8. corrected-LAZ write-back (only if a source cloud is given)
    if len(sys.argv) > 2:
        src_laz = sys.argv[2]
        out = Path(tempfile.gettempdir()) / "_corrected.laz"
        em2.export_corrected_laz(src_laz, out)
        import laspy
        chk = laspy.read(str(out))
        ci = np.asarray(chk["instance_id"]); cs = np.asarray(chk["semantic_id"])
        src, inst, sem = em2.per_point_labels()
        assert np.all(ci[src] == inst) and np.all(cs[src] == sem), "write-back mismatch"
        # background (uncovered) rows unchanged
        orig = laspy.read(str(src_laz))
        mask = np.ones(len(ci), bool); mask[src] = False
        assert np.all(ci[mask] == np.asarray(orig["instance_id"])[mask]), "background changed"
        os.remove(out)
        print(f"[ok] corrected-LAZ write-back: {len(src):,} labels baked, background intact")
    else:
        print("[skip] corrected-LAZ write-back (pass source .laz as 2nd arg to test)")

    print("\nALL CHECKS PASSED" if not fails else f"\nFAILURES: {fails}")


def SEM_NAMES_index(name):
    from edit_model import SEM_NAMES
    return SEM_NAMES.index(name)


if __name__ == "__main__":
    main()
