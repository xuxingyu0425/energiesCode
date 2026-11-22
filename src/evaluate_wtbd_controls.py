"""Capture native Ultralytics validator coordinates and confidence ranks."""

from pathlib import Path

from ultralytics.models.yolo.detect.val import DetectionValidator


class CaptureValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.captured = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            # Keep exact validator coordinates and FP confidence ranking. No JSON rounding.
            record = dict(image=Path(pbatch["im_file"]).name, imgsz=list(pbatch["imgsz"]),
                          gt_boxes=pbatch["bboxes"].detach().cpu().tolist(),
                          gt_cls=pbatch["cls"].detach().cpu().tolist(),
                          pred_boxes=predn["bboxes"].detach().cpu().tolist(),
                          pred_cls=predn["cls"].detach().cpu().tolist(),
                          conf=predn["conf"].detach().cpu().tolist())
            self.captured.append(record)
