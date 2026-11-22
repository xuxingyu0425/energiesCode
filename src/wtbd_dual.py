"""Dual-annotation data and losses for WTBD object detection.

The public fused YOLO labels are kept as the assignment targets. During
training, the paired annotator-A/annotator-B boxes are carried through the
same geometric augmentations as auxiliary keypoints. They are consumed only
by the custom regression losses and are not additional model outputs.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import bbox2dist, make_anchors
from ultralytics.utils.torch_utils import unwrap_model


# Four A corners, four B corners, and four interior metadata points.
AUX_KEYPOINTS = 12
SUPPORTED_STRATEGIES = {
    "dual_target",
    "distributional",
    "distributional_soft_class",
    "full",
}


def _parse_box(text: str) -> np.ndarray | None:
    """Parse an absolute xyxy audit box, returning None for a missing box."""
    if not text.strip():
        return None
    values = np.asarray([float(value) for value in text.split()], dtype=np.float32)
    if values.shape != (4,):
        raise ValueError(f"Expected four xyxy values, received: {text!r}")
    return values


def _corners(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.tolist()
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _interior_points(box: np.ndarray) -> np.ndarray:
    """Return four interior points so metadata survives ordinary crops."""
    x1, y1, x2, y2 = box.tolist()
    return np.asarray(
        [
            [0.75 * x1 + 0.25 * x2, 0.75 * y1 + 0.25 * y2],
            [0.25 * x1 + 0.75 * x2, 0.75 * y1 + 0.25 * y2],
            [0.25 * x1 + 0.75 * x2, 0.25 * y1 + 0.75 * y2],
            [0.75 * x1 + 0.25 * x2, 0.25 * y1 + 0.75 * y2],
        ],
        dtype=np.float32,
    )


class DualAnnotationYOLODataset(YOLODataset):
    """YOLO detection dataset carrying paired boxes through augmentation."""

    def __init__(self, *args, audit_path: str | Path, **kwargs) -> None:
        self.audit_path = Path(audit_path).resolve()
        if not self.audit_path.exists():
            raise FileNotFoundError(f"Dual-annotation audit file not found: {self.audit_path}")
        self._audit_by_image = self._read_audit(self.audit_path)

        # YOLO uses these fields when keypoints are transformed by flips.
        data = dict(kwargs.get("data") or {})
        data["kpt_shape"] = [AUX_KEYPOINTS, 3]
        data["flip_idx"] = list(range(AUX_KEYPOINTS))
        kwargs["data"] = data
        super().__init__(*args, **kwargs)

    @staticmethod
    def _read_audit(path: Path) -> dict[str, list[dict[str, str]]]:
        rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[row["image"]].append(row)
        if not rows:
            raise ValueError(f"No records found in dual-annotation audit: {path}")
        return dict(rows)

    def get_labels(self) -> list[dict[str, Any]]:
        """Load ordinary detection labels, verify audit alignment, then attach A/B boxes."""
        labels = super().get_labels()
        names = self.data["names"]
        if isinstance(names, dict):
            name_to_id = {str(name): int(idx) for idx, name in names.items()}
        else:
            name_to_id = {str(name): idx for idx, name in enumerate(names)}

        status_counts: Counter[str] = Counter()
        for label in labels:
            image_name = Path(label["im_file"]).name
            audit_rows = self._audit_by_image.get(image_name)
            if audit_rows is None:
                raise ValueError(f"Audit rows missing for training image: {image_name}")
            if len(audit_rows) != len(label["bboxes"]):
                raise ValueError(
                    f"Audit/YOLO object count mismatch for {image_name}: "
                    f"audit={len(audit_rows)}, yolo={len(label['bboxes'])}"
                )

            height, width = (float(value) for value in label["shape"])
            keypoints = np.zeros((len(audit_rows), AUX_KEYPOINTS, 3), dtype=np.float32)
            for index, row in enumerate(audit_rows):
                status_counts[row["status"]] += 1
                box_a = _parse_box(row["box_a"])
                box_b = _parse_box(row["box_b"])
                box_fused = _parse_box(row["box_fused"])
                if box_fused is None or (box_a is None and box_b is None):
                    raise ValueError(f"Invalid audit boxes for {image_name}, object {index}")

                resolved_name = row["resolved_class"]
                resolved_id = name_to_id[resolved_name]
                label_id = int(label["cls"][index, 0])
                if resolved_id != label_id:
                    raise ValueError(
                        f"Audit/YOLO class mismatch for {image_name}, object {index}: "
                        f"audit={resolved_id}, yolo={label_id}"
                    )

                # The audit was emitted in exactly the same order as fused labels.
                yolo_xyxy = xywh2xyxy(torch.from_numpy(label["bboxes"][[index]])).numpy()[0]
                yolo_xyxy *= np.asarray([width, height, width, height], dtype=np.float32)
                if not np.allclose(yolo_xyxy, box_fused, atol=0.06, rtol=0.0):
                    raise ValueError(
                        f"Audit/YOLO box mismatch for {image_name}, object {index}: "
                        f"audit={box_fused.tolist()}, yolo={yolo_xyxy.tolist()}"
                    )

                has_pair = box_a is not None and box_b is not None
                box_a = box_fused if box_a is None else box_a
                box_b = box_fused if box_b is None else box_b
                class_b_name = row["class_b"] or resolved_name
                class_b = name_to_id[class_b_name]

                points = np.concatenate((_corners(box_a), _corners(box_b), _interior_points(box_fused)))
                points[:, 0] /= width
                points[:, 1] /= height
                keypoints[index, :, :2] = points

                # Packed in the visibility field, which geometric YOLO transforms preserve.
                # 10 * (class_b + 1) + pair_flag is always positive.
                keypoints[index, :, 2] = 10 * (class_b + 1) + int(has_pair)

            label["keypoints"] = keypoints

        # This is intentionally enabled only after ordinary detection labels have
        # been read, otherwise YOLO would expect pose columns in the .txt files.
        self.use_keypoints = True
        self.dual_status_counts = dict(status_counts)
        return labels


class DualAnnotationDetectionLoss(v8DetectionLoss):
    """YOLO detection loss with paired boundary and optional label uncertainty."""

    def __init__(
        self,
        model: DetectionModel,
        strategy: str,
        sigma0: float = 0.5,
        beta: float = 0.5,
        sigma_min: float = 0.5,
        sigma_max: float = 3.0,
        single_annotator_weight: float = 0.6,
    ) -> None:
        super().__init__(model)
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported dual-annotation strategy: {strategy}")
        if not 0.0 < single_annotator_weight <= 1.0:
            raise ValueError("single_annotator_weight must be in (0, 1]")
        self.strategy = strategy
        self.sigma0 = float(sigma0)
        self.beta = float(beta)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.single_annotator_weight = float(single_annotator_weight)

    def _pad_by_image(self, batch_idx: torch.Tensor, values: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Pad flattened per-object values using the same ordering as YOLO preprocess()."""
        if len(values) == 0:
            return values.new_zeros((batch_size, 0, *values.shape[1:]))
        batch_idx = batch_idx.long()
        counts = torch.bincount(batch_idx, minlength=batch_size)
        output = values.new_zeros((batch_size, int(counts.max().item()), *values.shape[1:]))
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=values.device)
        offsets[1:] = counts.cumsum(0)
        within_idx = torch.arange(len(values), device=values.device) - offsets[batch_idx]
        output[batch_idx, within_idx] = values
        return output

    @staticmethod
    def _gather_assigned(padded: torch.Tensor, target_gt_idx: torch.Tensor) -> torch.Tensor:
        index = target_gt_idx
        for _ in padded.shape[2:]:
            index = index.unsqueeze(-1)
        return padded.gather(1, index.expand(-1, -1, *padded.shape[2:]))

    @staticmethod
    def _points_to_xyxy(points: torch.Tensor) -> torch.Tensor:
        mins = points.amin(dim=1)
        maxs = points.amax(dim=1)
        return torch.cat((mins, maxs), dim=-1)

    def _paired_bbox_loss(
        self,
        pred_distri: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor,
        target_scores: torch.Tensor,
        normalizer: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg = pred_bboxes[fg_mask]
        target_a_fg = target_a[fg_mask]
        target_b_fg = target_b[fg_mask]
        iou_a = bbox_iou(pred_fg, target_a_fg, xywh=False, CIoU=True)
        iou_b = bbox_iou(pred_fg, target_b_fg, xywh=False, CIoU=True)
        loss_iou = ((1.0 - 0.5 * (iou_a + iou_b)) * weight).sum() / normalizer

        if not self.use_dfl:
            raise RuntimeError("WTBD dual-annotation losses require a DFL-enabled detection head")
        dist_a = bbox2dist(anchor_points, target_a, self.reg_max - 1)
        dist_b = bbox2dist(anchor_points, target_b, self.reg_max - 1)
        pred_fg_dist = pred_distri[fg_mask]

        if self.strategy == "dual_target":
            logits = pred_fg_dist.reshape(-1, self.reg_max)
            loss_a = self.bbox_loss.dfl_loss(logits, dist_a[fg_mask])
            loss_b = self.bbox_loss.dfl_loss(logits, dist_b[fg_mask])
            loss_dfl = (0.5 * (loss_a + loss_b) * weight).sum() / normalizer
            return loss_iou, loss_dfl

        mu = 0.5 * (dist_a[fg_mask] + dist_b[fg_mask])
        sigma = (self.sigma0 + self.beta * (dist_a[fg_mask] - dist_b[fg_mask]).abs()).clamp(
            self.sigma_min, self.sigma_max
        )
        bins = torch.arange(self.reg_max, device=mu.device, dtype=mu.dtype).view(1, 1, -1)
        target_distribution = torch.softmax(-0.5 * ((bins - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)).square(), dim=-1)
        log_prob = F.log_softmax(pred_fg_dist.reshape(-1, 4, self.reg_max), dim=-1)
        distribution_ce = -(target_distribution * log_prob).sum(-1).mean(-1, keepdim=True)
        loss_dfl = (distribution_ce * weight).sum() / normalizer
        return loss_iou, loss_dfl

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        if "keypoints" not in batch:
            # Validation deliberately uses the ordinary fused dataset so that
            # metrics and validation loss remain directly comparable with S1.
            # Only augmented training batches carry the paired A/B boxes.
            return super().get_assigned_targets_and_loss(preds, batch)

        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        scale_xyxy = imgsz[[1, 0, 1, 0]]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=scale_xyxy)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # Decode paired annotations after they underwent the exact same augmentation.
        keypoints = batch["keypoints"].to(self.device)
        box_a_flat = self._points_to_xyxy(keypoints[:, 0:4, :2]) * scale_xyxy
        box_b_flat = self._points_to_xyxy(keypoints[:, 4:8, :2]) * scale_xyxy
        packed_meta = keypoints[:, :, 2].amax(dim=1).round().long()
        has_pair_flat = packed_meta.remainder(10).eq(1)
        class_b_flat = torch.div(packed_meta, 10, rounding_mode="floor") - 1
        class_a_flat = batch["cls"].view(-1).to(self.device).long()
        invalid_meta = (class_b_flat < 0) | (class_b_flat >= self.nc)
        class_b_flat = torch.where(invalid_meta, class_a_flat, class_b_flat)
        has_pair_flat = has_pair_flat & ~invalid_meta

        batch_idx = batch["batch_idx"].to(self.device).long()
        gt_a = self._pad_by_image(batch_idx, box_a_flat, batch_size)
        gt_b = self._pad_by_image(batch_idx, box_b_flat, batch_size)
        gt_has_pair = self._pad_by_image(batch_idx, has_pair_flat.unsqueeze(-1), batch_size)
        gt_class_b = self._pad_by_image(batch_idx, class_b_flat.unsqueeze(-1), batch_size)
        gt_class_a = self._pad_by_image(batch_idx, class_a_flat.unsqueeze(-1), batch_size)

        assigned_a = self._gather_assigned(gt_a, target_gt_idx)
        assigned_b = self._gather_assigned(gt_b, target_gt_idx)
        assigned_has_pair = self._gather_assigned(gt_has_pair, target_gt_idx).squeeze(-1).bool()
        assigned_class_b = self._gather_assigned(gt_class_b, target_gt_idx).squeeze(-1).long()
        assigned_class_a = self._gather_assigned(gt_class_a, target_gt_idx).squeeze(-1).long()

        # Extremely cropped auxiliary quadrilaterals may collapse at an image edge.
        # Falling back to the valid assignment box is safer than creating an invalid target.
        valid_a = ((assigned_a[..., 2] - assigned_a[..., 0]) > 0.5) & (
            (assigned_a[..., 3] - assigned_a[..., 1]) > 0.5
        )
        valid_b = ((assigned_b[..., 2] - assigned_b[..., 0]) > 0.5) & (
            (assigned_b[..., 3] - assigned_b[..., 1]) > 0.5
        )
        assigned_a = torch.where(valid_a.unsqueeze(-1), assigned_a, target_bboxes)
        assigned_b = torch.where(valid_b.unsqueeze(-1), assigned_b, target_bboxes)

        # Keep YOLO's original quality-mass denominator fixed so lowering a
        # one-sided target does not inadvertently amplify all negative anchors.
        normalizer = target_scores.sum().clamp_min(1.0)
        if self.strategy in {"distributional_soft_class", "full"}:
            quality = target_scores.sum(-1)
            conflict = fg_mask & assigned_has_pair & assigned_class_a.ne(assigned_class_b)
            if conflict.any():
                image_idx, anchor_idx = conflict.nonzero(as_tuple=True)
                class_a = assigned_class_a[conflict]
                class_b = assigned_class_b[conflict]
                half_quality = 0.5 * quality[conflict]
                target_scores[image_idx, anchor_idx] = 0.0
                target_scores[image_idx, anchor_idx, class_a] = half_quality
                target_scores[image_idx, anchor_idx, class_b] += half_quality

        if self.strategy == "full":
            one_sided = fg_mask & ~assigned_has_pair
            target_scores[one_sided] *= self.single_annotator_weight

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / normalizer

        if fg_mask.any():
            target_a_stride = assigned_a / stride_tensor
            target_b_stride = assigned_b / stride_tensor
            loss[0], loss[2] = self._paired_bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_a_stride,
                target_b_stride,
                target_scores,
                normalizer,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


class DualAnnotationDetectionModel(DetectionModel):
    """DetectionModel selecting the configured WTBD dual-annotation loss."""

    def __init__(
        self,
        cfg: str | dict = "yolo11n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        dual_strategy: str = "dual_target",
        dual_params: dict[str, float] | None = None,
    ) -> None:
        self.dual_strategy = dual_strategy
        self.dual_params = dict(dual_params or {})
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self) -> DualAnnotationDetectionLoss:
        return DualAnnotationDetectionLoss(
            self,
            strategy=self.dual_strategy,
            **self.dual_params,
        )


class DualAnnotationTrainer(DetectionTrainer):
    """Detection trainer injecting the paired dataset and custom model."""

    dual_strategy = "dual_target"
    dual_params: dict[str, float] = {}
    dual_audit_path: str = ""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode != "train":
            return super().build_dataset(img_path, mode=mode, batch=batch)
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return DualAnnotationYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0,
            prefix=colorstr("train: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
            audit_path=self.dual_audit_path,
        )

    def get_model(self, cfg: str | dict | None = None, weights: str | None = None, verbose: bool = True):
        model = DualAnnotationDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            dual_strategy=self.dual_strategy,
            dual_params=self.dual_params,
        )
        if weights:
            model.load(weights)
        return model


def make_dual_trainer(
    strategy: str,
    audit_path: str | Path,
    **dual_params: float,
) -> type[DualAnnotationTrainer]:
    """Create a configured, importable-in-process trainer class for YOLO.train()."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")

    class ConfiguredDualTrainer(DualAnnotationTrainer):
        pass

    ConfiguredDualTrainer.dual_strategy = strategy
    ConfiguredDualTrainer.dual_params = dict(dual_params)
    ConfiguredDualTrainer.dual_audit_path = str(Path(audit_path).resolve())
    ConfiguredDualTrainer.__name__ = f"WTBD_{strategy}_Trainer"
    ConfiguredDualTrainer.__qualname__ = ConfiguredDualTrainer.__name__
    return ConfiguredDualTrainer


def build_standard_dataset(*args, **kwargs):
    """Small public wrapper retained for interface parity in tests."""
    return build_yolo_dataset(*args, **kwargs)
