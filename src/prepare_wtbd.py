"""Prepare grouped WTBD splits and YOLO labels from both Pascal VOC annotation sets.

Training labels depend on the supervision configuration. Fused validation
labels support native checkpoint selection. Separate annotation roots retain A
for the primary evaluation and B for annotation sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


CLASS_NAMES = [
    "craze",
    "crack",
    "corrosion",
    "hide_craze",
    "surface_injure",
    "thunderstrike",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class Box:
    cls: str
    xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_voc(path: Path) -> tuple[int, int, list[Box]]:
    root = ET.parse(path).getroot()
    width = int(root.findtext("size/width", "1024"))
    height = int(root.findtext("size/height", "1024"))
    boxes: list[Box] = []
    for obj in root.findall("object"):
        cls = (obj.findtext("name") or "").strip()
        if cls not in CLASS_TO_ID:
            raise ValueError(f"Unknown class {cls!r} in {path}")
        node = obj.find("bndbox")
        if node is None:
            continue
        x1 = max(0.0, min(float(node.findtext("xmin", "0")), width - 1.0))
        y1 = max(0.0, min(float(node.findtext("ymin", "0")), height - 1.0))
        x2 = max(x1 + 1.0, min(float(node.findtext("xmax", "0")), float(width)))
        y2 = max(y1 + 1.0, min(float(node.findtext("ymax", "0")), float(height)))
        boxes.append(Box(cls, (x1, y1, x2, y2)))
    return width, height, boxes


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / max(a.area + b.area - inter, 1e-9)


def geometry(a: Box, b: Box, width: int, height: int) -> tuple[float, float, float]:
    iou = box_iou(a, b)
    acx, acy = a.center
    bcx, bcy = b.center
    center_diag = math.hypot(acx - bcx, acy - bcy) / math.hypot(width, height)
    area_ratio = max(a.area, b.area) / max(min(a.area, b.area), 1e-9)
    return iou, center_diag, area_ratio


def fuse_annotations(
    boxes_a: list[Box], boxes_b: list[Box], width: int, height: int
) -> tuple[list[Box], list[dict[str, object]]]:
    """Match spatial targets independent of class and fuse each target once.

    A pair is eligible when it overlaps at least weakly, or when its centers and
    scales are sufficiently close. Class-disagreement pairs retain annotator A's
    class because `Annotations` is the article's primary public label set. Box
    coordinates are averaged. Unmatched boxes from either annotator are retained.
    """
    audit: list[dict[str, object]] = []
    fused: list[Box] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    if boxes_a and boxes_b:
        cost = np.full((len(boxes_a), len(boxes_b)), 1e6, dtype=np.float64)
        metrics: dict[tuple[int, int], tuple[float, float, float]] = {}
        for i, a in enumerate(boxes_a):
            for j, b in enumerate(boxes_b):
                iou, center_diag, area_ratio = geometry(a, b, width, height)
                eligible = iou >= 0.10 or (center_diag <= 0.035 and area_ratio <= 5.0)
                if eligible:
                    cost[i, j] = (
                        (1.0 - iou)
                        + 1.5 * center_diag
                        + 0.15 * abs(math.log(max(area_ratio, 1e-9)))
                    )
                    metrics[(i, j)] = (iou, center_diag, area_ratio)
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if cost[i, j] >= 1e5:
                continue
            a, b = boxes_a[i], boxes_b[j]
            iou, center_diag, area_ratio = metrics[(i, j)]
            xyxy = tuple((x + y) / 2.0 for x, y in zip(a.xyxy, b.xyxy))
            status = "matched_same_class" if a.cls == b.cls else "matched_class_conflict"
            fused.append(Box(a.cls, xyxy))
            matched_a.add(i)
            matched_b.add(j)
            audit.append(
                {
                    "status": status,
                    "class_a": a.cls,
                    "class_b": b.cls,
                    "resolved_class": a.cls,
                    "iou": round(iou, 6),
                    "center_distance_diag": round(center_diag, 6),
                    "area_ratio": round(area_ratio, 6),
                    "box_a": " ".join(f"{v:.2f}" for v in a.xyxy),
                    "box_b": " ".join(f"{v:.2f}" for v in b.xyxy),
                    "box_fused": " ".join(f"{v:.2f}" for v in xyxy),
                }
            )
    for i, a in enumerate(boxes_a):
        if i not in matched_a:
            fused.append(a)
            audit.append(
                {
                    "status": "only_annotator_a",
                    "class_a": a.cls,
                    "class_b": "",
                    "resolved_class": a.cls,
                    "iou": "",
                    "center_distance_diag": "",
                    "area_ratio": "",
                    "box_a": " ".join(f"{v:.2f}" for v in a.xyxy),
                    "box_b": "",
                    "box_fused": " ".join(f"{v:.2f}" for v in a.xyxy),
                }
            )
    for j, b in enumerate(boxes_b):
        if j not in matched_b:
            fused.append(b)
            audit.append(
                {
                    "status": "only_annotator_b",
                    "class_a": "",
                    "class_b": b.cls,
                    "resolved_class": b.cls,
                    "iou": "",
                    "center_distance_diag": "",
                    "area_ratio": "",
                    "box_a": "",
                    "box_b": " ".join(f"{v:.2f}" for v in b.xyxy),
                    "box_fused": " ".join(f"{v:.2f}" for v in b.xyxy),
                }
            )
    return fused, audit


def perceptual_hash(path: Path) -> np.ndarray:
    # cv2.imread may fail on non-ASCII Windows paths; imdecode is Unicode-safe.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].reshape(-1)[1:]
    return low > np.median(low)


def build_groups(images: list[Path], threshold: int) -> tuple[list[int], list[str]]:
    hashes = np.stack([perceptual_hash(path) for path in images])
    uf = UnionFind(len(images))
    for i in range(len(images) - 1):
        distances = np.count_nonzero(hashes[i + 1 :] != hashes[i], axis=1)
        for offset in np.flatnonzero(distances <= threshold):
            uf.union(i, i + 1 + int(offset))
    roots = [uf.find(i) for i in range(len(images))]
    root_to_id = {root: idx for idx, root in enumerate(sorted(set(roots)))}
    groups = [root_to_id[root] for root in roots]
    hash_text = ["".join("1" if bit else "0" for bit in row) for row in hashes]
    return groups, hash_text


def assign_group_splits(
    images: list[Path], groups: list[int], fused: dict[str, list[Box]], seed: int
) -> dict[str, str]:
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    names = list(ratios)
    by_group: dict[int, list[str]] = defaultdict(list)
    for path, group in zip(images, groups):
        by_group[group].append(path.stem)

    total_images = len(images)
    total_class = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for boxes in fused.values():
        for cls in {box.cls for box in boxes}:
            total_class[CLASS_TO_ID[cls]] += 1.0
    target_images = {name: ratios[name] * total_images for name in names}
    target_class = {name: ratios[name] * total_class for name in names}
    current_images = {name: 0 for name in names}
    current_class = {name: np.zeros(len(CLASS_NAMES), dtype=np.float64) for name in names}

    rng = random.Random(seed)
    group_ids = list(by_group)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda gid: len(by_group[gid]), reverse=True)
    group_split: dict[int, str] = {}
    for gid in group_ids:
        stems = by_group[gid]
        present = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        for stem in stems:
            for cls in {box.cls for box in fused[stem]}:
                present[CLASS_TO_ID[cls]] += 1.0
        candidate_scores: list[tuple[float, float, str]] = []
        for name in names:
            image_deficit = (target_images[name] - current_images[name]) / max(target_images[name], 1.0)
            mask = present > 0
            if np.any(mask):
                class_deficits = (target_class[name][mask] - current_class[name][mask]) / np.maximum(
                    target_class[name][mask], 1.0
                )
                class_score = float(np.mean(class_deficits))
            else:
                class_score = 0.0
            overflow = max(0.0, current_images[name] + len(stems) - target_images[name])
            score = image_deficit + 0.55 * class_score - 4.0 * overflow / max(target_images[name], 1.0)
            candidate_scores.append((score, target_images[name] - current_images[name], name))
        _, _, chosen = max(candidate_scores)
        group_split[gid] = chosen
        current_images[chosen] += len(stems)
        current_class[chosen] += present

    return {path.stem: group_split[group] for path, group in zip(images, groups)}


def write_yolo(path: Path, boxes: list[Box], width: int, height: int) -> None:
    lines: list[str] = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        xc = (x1 + x2) / (2.0 * width)
        yc = (y1 + y2) / (2.0 * height)
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{CLASS_TO_ID[box.cls]} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def count_classes(labels: dict[str, list[Box]], stems: list[str]) -> dict[str, int]:
    counts = Counter(box.cls for stem in stems for box in labels[stem])
    return {name: counts[name] for name in CLASS_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-threshold", type=int, default=8)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    images_dir = source / "JPEGImages"
    ann_a_dir = source / "Annotations"
    ann_b_dir = source / "annotation_second_person"
    images = sorted(images_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not images:
        raise FileNotFoundError(f"No JPEG images under {images_dir}")

    annotations_a: dict[str, list[Box]] = {}
    annotations_b: dict[str, list[Box]] = {}
    consensus: dict[str, list[Box]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    audit_rows: list[dict[str, object]] = []
    for image in images:
        stem = image.stem
        width_a, height_a, boxes_a = parse_voc(ann_a_dir / f"{stem}.xml")
        width_b, height_b, boxes_b = parse_voc(ann_b_dir / f"{stem}.xml")
        if (width_a, height_a) != (width_b, height_b):
            raise ValueError(f"Image size mismatch for {stem}")
        fused, rows = fuse_annotations(boxes_a, boxes_b, width_a, height_a)
        annotations_a[stem] = boxes_a
        annotations_b[stem] = boxes_b
        consensus[stem] = fused
        sizes[stem] = (width_a, height_a)
        for row in rows:
            row["image"] = image.name
            audit_rows.append(row)

    groups, hashes = build_groups(images, args.phash_threshold)
    split_by_stem = assign_group_splits(images, groups, consensus, args.seed)
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "split", "phash_group", "phash_bits"])
        writer.writeheader()
        for image, group, hash_bits in zip(images, groups, hashes):
            writer.writerow(
                {
                    "image": image.name,
                    "split": split_by_stem[image.stem],
                    "phash_group": group,
                    "phash_bits": hash_bits,
                }
            )

    audit_path = output / "dual_annotation_audit.csv"
    audit_fields = [
        "image",
        "status",
        "class_a",
        "class_b",
        "resolved_class",
        "iou",
        "center_distance_diag",
        "area_ratio",
        "box_a",
        "box_b",
        "box_fused",
    ]
    with audit_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    label_sources = {
        "primary": annotations_a,
        "second": annotations_b,
        "fused": consensus,
    }
    for variant, train_labels in label_sources.items():
        root = output / variant
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image in images:
            stem = image.stem
            split = split_by_stem[stem]
            link_or_copy(image, root / "images" / split / image.name)
            # Use fused labels in the combined validation and test directories.
            boxes = train_labels[stem] if split == "train" else consensus[stem]
            width, height = sizes[stem]
            write_yolo(root / "labels" / split / f"{stem}.txt", boxes, width, height)
        yaml_path = output / f"wtbd_{variant}.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    f"path: {root.as_posix()}",
                    "train: images/train",
                    "val: images/val",
                    "test: images/test",
                    "names:",
                    *[f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES)],
                    "",
                ]
            ),
            encoding="utf-8",
        )

    # Preserve the original A and B annotations in separate evaluation roots.
    for eval_name, eval_labels in {
        "gt_primary": annotations_a,
        "gt_second": annotations_b,
        "gt_fused": consensus,
    }.items():
        root = output / eval_name
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image in images:
            stem = image.stem
            split = split_by_stem[stem]
            link_or_copy(image, root / "images" / split / image.name)
            width, height = sizes[stem]
            write_yolo(root / "labels" / split / f"{stem}.txt", eval_labels[stem], width, height)
        (output / f"wtbd_{eval_name}.yaml").write_text(
            "\n".join(
                [
                    f"path: {root.as_posix()}",
                    "train: images/train",
                    "val: images/val",
                    "test: images/test",
                    "names:",
                    *[f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES)],
                    "",
                ]
            ),
            encoding="utf-8",
        )

    status_counts = Counter(str(row["status"]) for row in audit_rows)
    split_summary: dict[str, object] = {}
    for split in ("train", "val", "test"):
        stems = [stem for stem, assigned in split_by_stem.items() if assigned == split]
        split_summary[split] = {
            "images": len(stems),
            "consensus_boxes": sum(len(consensus[stem]) for stem in stems),
            "class_boxes": count_classes(consensus, stems),
        }
    group_sizes = Counter(groups)
    summary = {
        "source": str(source),
        "seed": args.seed,
        "phash_hamming_threshold": args.phash_threshold,
        "dataset_sha256": hashlib.sha256(
            "\n".join(f"{path.name}:{path.stat().st_size}" for path in images).encode("utf-8")
        ).hexdigest(),
        "images": len(images),
        "phash_groups": len(group_sizes),
        "largest_phash_group": max(group_sizes.values()),
        "boxes_primary": sum(map(len, annotations_a.values())),
        "boxes_second": sum(map(len, annotations_b.values())),
        "boxes_consensus": sum(map(len, consensus.values())),
        "fusion_status": dict(sorted(status_counts.items())),
        "splits": split_summary,
        "class_names": CLASS_NAMES,
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
