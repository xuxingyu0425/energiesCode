"""CPU-only reconstruction of all five historical CBAM initializations."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
import gc
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained",type=Path,required=True)
    args=parser.parse_args()
    config_dir=ROOT/"outputs/ultralytics_config";config_dir.mkdir(parents=True,exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"]=str(config_dir)
    sys.path.insert(0,str(ROOT/"src"))
    import torch
    import ultralytics
    if torch.__version__!="2.3.1+cu121" or ultralytics.__version__!="8.4.37":
        raise SystemExit("Exact initialization check requires the historical core environment.")
    torch.set_num_threads(2)
    import run_wtbd_yolov8_attention as attention
    attention.ROOT=ROOT;attention.core.WEIGHTS=args.pretrained.resolve()
    checks=[]
    for seed in range(42,47):
        model,common,meta=attention.build("cbam",seed,callbacks=False)
        expected=json.loads((ROOT/"provenance/original_attention"/f"seed_{seed}/initialization.json").read_text(encoding="utf-8"))
        for key in ["full_state_sha256","common_state_sha256","pretrained_weights_sha256","parameters"]:
            assert meta[key]==expected[key],(seed,key,meta[key],expected[key])
        assert all(p.device.type=="cpu" for p in model.model.parameters())
        checks.append(dict(seed=seed,full_state_sha256=meta["full_state_sha256"],common_state_sha256=meta["common_state_sha256"],matches_historical=True))
        del model,common;gc.collect()
    report=dict(device="cpu",pretrained_weights_sha256=meta["pretrained_weights_sha256"],checks=checks,training_performed=False,gpu_used=False)
    (ROOT/"outputs/initialization_check.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
