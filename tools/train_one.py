"""Plan or run one configured WTBD training experiment."""
import argparse
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CELL_DATA={"A":"primary","B":"second","S1":"fused","S2":"fused",
           "A_ESB":"primary","S2_ESB":"fused","A_CBAM":"primary","S2_CBAM":"fused","S1_CBAM":"fused"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell",choices=CELL_DATA,required=True)
    parser.add_argument("--seed",type=int,choices=range(42,47),required=True)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--pretrained",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="0")
    parser.add_argument("--execute",action="store_true",help="Explicitly start 200-epoch training; omitted means plan only")
    args=parser.parse_args()
    config="yolov8n_wtbd_cbam.yaml" if "CBAM" in args.cell else "yolov8n_wtbd_esb.yaml" if "ESB" in args.cell else "yolov8n_wtbd.yaml"
    plan=dict(cell=args.cell,seed=args.seed,epochs=200,imgsz=768,batch=16,
              architecture_config=str(ROOT/"configs"/config),
              training_data=str(args.data_dir/("wtbd_"+CELL_DATA[args.cell]+".yaml")),
              supervision="dual_target" if args.cell.startswith("S2") else "ordinary regression",
              pretrained=str(args.pretrained),output=str(args.output),
              paired_reference_check=args.cell=="S1_CBAM",
              execution_requested=args.execute)
    if args.cell=="S1_CBAM":plan["supervision"]="paired S2 augmentation; native single fused-box v8DetectionLoss"
    print(json.dumps(plan,indent=2))
    if not args.execute:return
    if args.output.exists() and any(args.output.iterdir()):raise SystemExit("Output directory must be new/empty; historical runs are never overwritten.")
    if not args.pretrained.is_file():raise FileNotFoundError(args.pretrained)
    if not Path(plan["training_data"]).is_file():raise FileNotFoundError(plan["training_data"])
    config_dir=ROOT/"outputs/ultralytics_config";config_dir.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR",str(config_dir))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    sys.path.insert(0,str(ROOT/"src"))
    import torch
    import ultralytics
    if ultralytics.__version__!="8.4.37" or not torch.__version__.startswith("2.3.1"):
        raise SystemExit("Reconstruct the recorded historical environment (torch 2.3.1, Ultralytics 8.4.37) before execution. The current environment is not silently substituted.")
    from types import SimpleNamespace
    from ultralytics.utils.torch_utils import init_seeds
    import run_wtbd_yolov8_ab_s2_f1 as core
    core.WEIGHTS=args.pretrained.resolve();core.F0_CONFIG=ROOT/"configs/yolov8n_wtbd.yaml";core.ESB_CONFIG=ROOT/"configs/yolov8n_wtbd_esb.yaml"
    core.DATA=args.data_dir.resolve();core.AUDIT=core.DATA/"dual_annotation_audit.csv"
    init_seeds(args.seed,deterministic=True)
    if "CBAM" in args.cell:
        import run_wtbd_yolov8_attention as attention
        attention.ROOT=ROOT;attention.core.WEIGHTS=args.pretrained.resolve()
        model,_,initialization=attention.build("cbam",args.seed,callbacks=False)
    else:
        spec=dict(architecture="f1_esb" if "ESB" in args.cell else "f0")
        model=core.build_model(spec);initialization={"builder":"preserved historical builder","seeded_before_construction":True}
    defaults=SimpleNamespace(epochs=200,imgsz=768,batch=16,workers=4)
    spec=dict(data="wtbd_"+CELL_DATA[args.cell]+".yaml")
    kwargs=core.train_args(args.seed,defaults,spec)
    kwargs.update(device=args.device,project=str(args.output.resolve().parent),name=args.output.name,exist_ok=False)
    if args.cell.startswith("S2"):
        from wtbd_dual import make_dual_trainer
        kwargs["trainer"]=make_dual_trainer("dual_target",core.AUDIT)
    elif args.cell=="S1_CBAM":
        from revision_s1 import make_s1_paired_trainer
        kwargs["trainer"]=make_s1_paired_trainer(core.AUDIT)
        reference=ROOT/"provenance/original_attention"/f"seed_{args.seed}"
        expected=json.loads((reference/"initialization.json").read_text(encoding="utf-8"))
        assert initialization["full_state_sha256"]==expected["full_state_sha256"]
        def on_start(trainer):
            assert attention.state_hash(trainer.model.state_dict())==expected["full_state_sha256"]
            assert type(trainer.model.init_criterion()).__name__=="v8DetectionLoss"
            init_seeds(args.seed,deterministic=True)
            original=trainer.preprocess_batch;seen=False
            def preprocess(batch):
                nonlocal seen
                if not seen:
                    fields={k:v for k,v in batch.items() if isinstance(v,torch.Tensor)}
                    previous=json.loads((reference/"first_batch.json").read_text(encoding="utf-8"))
                    actual=attention.state_hash(fields)
                    assert actual==previous["tensor_sha256"],"Augmented first-batch tensors differ from historical S2"
                    names=lambda seq:[str(x).replace("\\","/").rsplit("/",1)[-1] for x in seq]
                    assert names(batch.get("im_file",[]))==names(previous["image_files"])
                    (args.output/"first_batch_equivalence.json").write_text(json.dumps(dict(tensor_sha256=actual,image_names=names(batch.get("im_file",[])),matches_historical_s2_cbam=True),indent=2),encoding="utf-8")
                    seen=True
                return original(batch)
            trainer.preprocess_batch=preprocess
        model.add_callback("on_train_start",on_start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    (args.output.parent/(args.output.name+"_plan.json")).write_text(json.dumps(dict(plan,initialization=initialization),indent=2),encoding="utf-8")
    model.train(**kwargs)

if __name__=="__main__":main()
