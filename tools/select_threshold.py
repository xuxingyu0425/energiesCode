"""Select an operating point from validation predictions only, then freeze it."""
import argparse
import json
from pathlib import Path
from common import load_json, counts, metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit("Refusing to overwrite a frozen selection.")
    cache=load_json(args.validation_cache)
    if cache.get("split")!="val":raise SystemExit("Threshold selection accepts validation caches only.")
    curve=[]
    for index in range(1,100):
        threshold=index/100;gt,pred,tp=counts(cache["records"],threshold)
        curve.append(dict(threshold=threshold,**metrics(gt.sum(),pred.sum(),tp.sum())))
    chosen=max(curve,key=lambda row:(row["F1"],-row["threshold"]))
    result=dict(threshold=chosen["threshold"],weight_sha256=cache["weight_sha256"],validation_curve=curve,
                rule="pooled instance F1; grid .01 to .99; smallest threshold breaks exact ties",
                test_data_used=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"threshold":result["threshold"],"validation_F1":chosen["F1"]},indent=2))

if __name__=="__main__":main()
