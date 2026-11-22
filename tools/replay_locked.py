"""Recompute 35 locked operating points from cached predictions, without torch/GPU."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT, DATA, counts, metrics, read_csv, load_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/replay")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    original=read_csv("locked_per_seed.csv")
    split=read_csv("split_manifest.csv")
    assert len(original)==35
    expected_images={s:set(split.loc[split["split"]==s,"image"]) for s in ("val","test")}
    errors=[]; checks=[]
    for row in original.to_dict("records"):
        prefix=f'{row["cell"]}_{int(row["seed"])}'
        selection=load_json(DATA/"locked"/(prefix+"_selection.json"))
        curve=selection["validation_curve"]
        assert len(curve)==99
        assert np.allclose(sorted(x["threshold"] for x in curve),np.arange(1,100)/100)
        chosen=max(curve,key=lambda x:(x["F1"],-x["threshold"]))
        assert chosen["threshold"]==row["threshold"]==selection["threshold"]
        for subset in ("val","test"):
            cache=load_json(DATA/"locked"/(prefix+"_"+subset+".json"))
            assert cache["weight_sha256"]==selection["weight_sha256"]
            assert {r["image"] for r in cache["records"]}==expected_images[subset]
            assert all(r["imgsz"]==[800,800] for r in cache["records"])
            gt,pred,tp=counts(cache["records"],row["threshold"])
            actual=metrics(gt.sum(),pred.sum(),tp.sum())
            expected=chosen if subset=="val" else row
            for key in ["GT","TP","FP","FN","P","R","F1"]:
                error=abs(actual[key]-expected[key]); errors.append(error)
                if error>1e-8:raise AssertionError((prefix,subset,key,actual[key],expected[key]))
            checks.append(dict(cell=row["cell"],seed=int(row["seed"]),split=subset,
                               threshold=row["threshold"],**actual))
    import pandas as pd
    pd.DataFrame(checks).to_csv(args.output/"replayed_operating_points.csv",index=False)
    report=dict(runs=35,subsets_replayed=len(checks),max_absolute_error=max(errors),
                metric_tolerance=1e-8,validation_grid_points=99,tie_break="lower threshold",
                exact_partition_verified=True,coordinate_shape=[800,800],
                training_performed=False,inference_performed=False,gpu_required=False)
    (args.output/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
