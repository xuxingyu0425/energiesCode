"""CPU import, architecture-shape and parameter-count smoke checks (no training)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import sys
import json
import importlib.metadata as metadata
ROOT=Path(__file__).resolve().parents[1]
config_dir=ROOT/"outputs/ultralytics_config"
config_dir.mkdir(parents=True,exist_ok=True)
os.environ["YOLO_CONFIG_DIR"]=str(config_dir)
sys.path.insert(0,str(ROOT/"src"))
import torch
from ultralytics import YOLO
from wtbd_modules import register_wtbd_modules, register_standard_modules
from wtbd_attention import register_attention_modules
from wtbd_dual import make_dual_trainer
from revision_s1 import make_s1_paired_trainer

def tensors(value):
    if torch.is_tensor(value):return [value]
    if isinstance(value,dict):return sum((tensors(v) for v in value.values()),[])
    if isinstance(value,(list,tuple)):return sum((tensors(v) for v in value),[])
    return []

def main():
    torch.set_num_threads(2)
    checks=[]
    for kind,config,expected in [("baseline","yolov8n_wtbd.yaml",3012018),
                                 ("esb","yolov8n_wtbd_esb.yaml",3043768),
                                 ("cbam","yolov8n_wtbd_cbam.yaml",3014982)]:
        if kind=="baseline":register_standard_modules()
        elif kind=="esb":register_wtbd_modules()
        else:register_attention_modules()
        model=YOLO(str(ROOT/"configs"/config)).model.to("cpu").eval()
        count=sum(p.numel() for p in model.parameters())
        assert count==expected,(kind,count,expected)
        with torch.inference_mode():result=model(torch.zeros(1,3,64,64))
        ts=tensors(result)
        assert ts and all(torch.isfinite(t).all() for t in ts)
        assert all(t.device.type=="cpu" for t in ts)
        checks.append(dict(architecture=kind,parameters=count,cpu_forward_finite=True,
                           input_shape=[1,3,64,64],output_tensor_shapes=[list(t.shape) for t in ts]))
    # Construction only: S2 registers a custom loss. S1 uses ordinary regression
    # on coordinate-averaged labels and does not use a custom 'hard_average' loss.
    assert make_dual_trainer("dual_target",ROOT/"data/dual_annotation_audit.csv")
    trainer_type=make_s1_paired_trainer(ROOT/"data/dual_annotation_audit.csv")
    from types import SimpleNamespace
    test_model=trainer_type.get_model(SimpleNamespace(data={"nc":6,"channels":3}),cfg=str(ROOT/"configs/yolov8n_wtbd.yaml"),verbose=False)
    # The real trainer assigns its hyperparameters after get_model().
    test_model.args=SimpleNamespace(box=7.5,cls=.5,dfl=1.5)
    assert type(test_model.init_criterion()).__name__=="v8DetectionLoss"
    packages={}
    for package in ["torch","ultralytics","numpy","scipy","pandas","opencv-python","matplotlib"]:
        try:packages[package]=metadata.version(package)
        except metadata.PackageNotFoundError:packages[package]=None
    report=dict(device="cpu",cuda_visible_devices="",training_performed=False,
                pretrained_weights_loaded=False,architectures=checks,S1_paired_native_criterion_verified=True,current_test_environment=packages)
    (ROOT/"outputs/model_cpu_check.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
