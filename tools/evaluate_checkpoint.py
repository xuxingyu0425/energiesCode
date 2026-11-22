"""Capture exact validator coordinates from a checkpoint using explicit data paths."""
import argparse
from pathlib import Path
import hashlib
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--split",choices=["val","test"],required=True)
    parser.add_argument("--architecture",choices=["baseline","esb","cbam"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--cell",default="unassigned")
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit("Refusing to overwrite an existing prediction cache.")
    config_dir=ROOT/"outputs/ultralytics_config";config_dir.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR",str(config_dir))
    sys.path.insert(0,str(ROOT/"src"))
    import ultralytics
    if ultralytics.__version__!="8.4.37":raise SystemExit("Use recorded Ultralytics 8.4.37; validator coordinates/AP are version-dependent.")
    from ultralytics import YOLO
    from wtbd_modules import register_standard_modules,register_wtbd_modules
    from wtbd_attention import register_attention_modules
    from evaluate_wtbd_controls import CaptureValidator
    if args.architecture=="baseline":register_standard_modules()
    elif args.architecture=="esb":register_wtbd_modules()
    else:register_attention_modules()
    holder=[]
    def factory(*a,**kw):
        obj=CaptureValidator(*a,**kw);holder.append(obj);return obj
    result=YOLO(str(args.weights)).val(validator=factory,data=str(args.data),split=args.split,
        imgsz=768,batch=16,workers=0,device=args.device,conf=.001,iou=.7,max_det=300,
        half=False,plots=False,verbose=False,save_json=False,
        project=str(args.output.parent/"validator_logs"),name=args.output.stem,exist_ok=False)
    records=holder[-1].captured
    native=dict(P=float(result.box.mp),R=float(result.box.mr),AP50=float(result.box.map50),AP=float(result.box.map))
    report=dict(cell=args.cell,seed=args.seed,split=args.split,weight_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                native_metrics=native,records=records,configured_imgsz=768,
                actual_shapes=sorted({tuple(r["imgsz"]) for r in records}),ultralytics=ultralytics.__version__)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2))

if __name__=="__main__":main()
