"""S1 box regression with the paired A/B augmentation pipeline used by S2.

DetectionModel supplies the native single-target detection loss."""
import math
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from wtbd_dual import DualAnnotationTrainer

class S1PairedPipelineTrainer(DualAnnotationTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=DetectionModel(cfg,nc=self.data["nc"],ch=self.data["channels"],
                             verbose=verbose and RANK==-1)
        if weights:model.load(weights)
        return model

    def _build_train_pipeline(self):
        if self.args.batch!=16 or self.batch_size!=16:
            raise RuntimeError("Fixed historical batch16; no automatic batch reduction")
        return super()._build_train_pipeline()

    def _handle_nan_recovery(self,epoch):
        if self.loss is not None and not torch.isfinite(self.loss).all():
            raise RuntimeError(f"Non-finite loss at epoch {epoch+1}; preserve failed run")
        if self.fitness is not None and not math.isfinite(float(self.fitness)):
            raise RuntimeError(f"Non-finite validation fitness at epoch {epoch+1}")
        return False

def make_s1_paired_trainer(audit_path):
    class BoundS1PairedTrainer(S1PairedPipelineTrainer):
        dual_audit_path=str(audit_path)
    return BoundS1PairedTrainer
