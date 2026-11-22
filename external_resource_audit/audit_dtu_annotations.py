"""Read-only audit of downloaded DTU COCO labels; no images or model evaluation.

Run from this directory. One-to-one matches are diagnostic correspondences,
not evidence of independent annotators or true physical-instance identity.
"""
from pathlib import Path
import collections
import csv
import json
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "DTU_annotations"


def load(p):
    obj = json.loads(p.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in obj["categories"]}
    ims = {i["id"]: i for i in obj["images"]}
    out = {i["file_name"]: {"meta": i, "boxes": []} for i in ims.values()}
    for a in obj["annotations"]:
        out[ims[a["image_id"]]["file_name"]]["boxes"].append(
            {"id": a["id"], "class": cats[a["category_id"]], "bbox": a["bbox"]}
        )
    return out


def source_id(name):
    return re.sub(r"_\d+_\d+$", "", Path(name).stem)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return inter / max(aw * ah + bw * bh - inter, 1e-12)


def compare(a, b, variant, split):
    rows = []
    counts = collections.Counter()
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            counts["images_only_base" if name in a else "images_only_variant"] += 1
            continue
        counts["common_images"] += 1
        aa, bb = a[name]["boxes"], b[name]["boxes"]
        counts["base_boxes_common_images"] += len(aa)
        counts["variant_boxes_common_images"] += len(bb)
        if len(aa) != len(bb):
            counts["common_images_with_box_count_change"] += 1
        mat = np.zeros((len(aa), len(bb)))
        for i, av in enumerate(aa):
            for j, bv in enumerate(bb):
                if av["class"] == bv["class"]:
                    mat[i, j] = iou(av["bbox"], bv["bbox"])
        pairs = []
        if mat.size:
            ai, bi = linear_sum_assignment(-mat)
            pairs = [(int(i), int(j), float(mat[i, j])) for i, j in zip(ai, bi) if mat[i, j] >= .5]
        counts["one_to_one_same_class_iou_ge_0_5_pairs"] += len(pairs)
        same_image = len(aa) == len(bb) == len(pairs)
        for i, j, overlap in pairs:
            delta = max(abs(x - y) for x, y in zip(aa[i]["bbox"], bb[j]["bbox"]))
            exact = delta <= 1e-4
            counts["matched_boxes_equal_within_1e_4_px" if exact else "matched_boxes_changed"] += 1
            same_image &= exact
            rows.append({"variant": variant, "split": split, "image": name,
                         "source_id_inferred": source_id(name), "class": aa[i]["class"],
                         "base_annotation_id": aa[i]["id"], "variant_annotation_id": bb[j]["id"],
                         "same_class_iou": overlap, "max_xywh_delta_px": delta,
                         "equal_within_1e_4_px": exact,
                         "base_bbox_xywh": json.dumps(aa[i]["bbox"]),
                         "variant_bbox_xywh": json.dumps(bb[j]["bbox"])})
        if same_image:
            counts["common_images_with_identical_box_sets"] += 1
    pairs = counts["one_to_one_same_class_iou_ge_0_5_pairs"]
    counts["base_unmatched_common_images"] = counts["base_boxes_common_images"] - pairs
    counts["variant_unmatched_common_images"] = counts["variant_boxes_common_images"] - pairs
    result = dict(counts)
    result["paired_fraction_of_base_boxes_on_common_images"] = pairs / max(1, counts["base_boxes_common_images"])
    result["paired_fraction_of_variant_boxes_on_common_images"] = pairs / max(1, counts["variant_boxes_common_images"])
    result["equal_fraction_of_matched_pairs"] = counts["matched_boxes_equal_within_1e_4_px"] / max(1, pairs)
    return result, rows


def main():
    base = {s: load(DATA / "annotations" / (s + "1024-s.json")) for s in ["train", "val", "test"]}
    inventories = []
    variants = {"base": base}
    for v in ["D2", "D3"]:
        variants[v] = {s: load(DATA / "re-annotation" / v / (s + ".json")) for s in base}
    split_checks = {}
    for v, splits in variants.items():
        ids = {}
        for s, ims in splits.items():
            ids[s] = {source_id(x) for x in ims}
            inventories.append({"variant": v, "split": s, "patch_images": len(ims),
                                "boxes": sum(len(i["boxes"]) for i in ims.values()),
                                "inferred_source_images": len(ids[s]),
                                "empty_patch_images": sum(not i["boxes"] for i in ims.values())})
        split_checks[v] = {a + "_" + b: sorted(ids[a] & ids[b])
                           for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    comparisons, all_rows = {}, []
    for v in ["D2", "D3"]:
        comparisons[v] = {}
        for s in base:
            result, rows = compare(base[s], variants[v][s], v, s)
            comparisons[v][s] = result
            all_rows += rows
        # Compare entire pools as image assignments differ across the releases.
        aa = {n: i for ims in base.values() for n, i in ims.items()}
        bb = {n: i for ims in variants[v].values() for n, i in ims.items()}
        result, rows = compare(aa, bb, v, "all_splits_pooled")
        comparisons[v]["all_splits_pooled"] = result
        all_rows += rows
    cross_release = {}
    for v in ["D2", "D3"]:
        origin = {n: s for s, ims in base.items() for n in ims}
        other = {n: s for s, ims in variants[v].items() for n in ims}
        moved = [{"image": n, "base_split": origin[n], "variant_split": other[n]}
                 for n in sorted(origin.keys() & other.keys()) if origin[n] != other[n]]
        cross_release[v] = moved
    summary = {"method": "same filename, same class, maximum-total-IoU Hungarian assignment, retain IoU>=0.5; exact xywh tolerance 1e-4 px",
               "warning": "Diagnostic pairing does not prove independent annotators. Inferred parent ID strips two trailing numeric patch indices; raw-image identity/overlap with WTBD was not checked because images were not downloaded.",
               "inventories": inventories, "comparisons": comparisons,
               "inferred_parent_source_overlap_between_splits": split_checks,
               "patches_moved_between_base_and_variant_splits": cross_release}
    (ROOT / "DTU_pairing_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "DTU_pairing_pairs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    with (ROOT / "DTU_inventory.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventories[0]))
        writer.writeheader(); writer.writerows(inventories)
    print(json.dumps({"comparisons_pooled": {v: comparisons[v]["all_splits_pooled"] for v in comparisons},
                      "source_split_overlap_counts": {v: {k: len(x) for k, x in d.items()} for v, d in split_checks.items()},
                      "cross_release_split_moves": {v: len(x) for v, x in cross_release.items()}}, indent=2))


if __name__ == "__main__":
    main()
