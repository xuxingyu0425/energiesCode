"""Verify bundled source/data fingerprints and coverage without model imports."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def csv_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))

def main():
    errors=[]
    source_records=json.loads((ROOT/"provenance/source_files.json").read_text(encoding="utf-8"))
    for row in source_records:
        path=ROOT/row["bundle_path"]
        if not path.is_file() or digest(path)!=row["sha256"]:errors.append("fingerprint: "+row["bundle_path"])
    for row in json.loads((ROOT/"provenance/figure_path_adaptations.json").read_text(encoding="utf-8")):
        if digest(ROOT/row["bundle_path"])!=row["adapted_sha256"]:errors.append("figure fingerprint: "+row["bundle_path"])
    split=csv_rows(ROOT/"data/split_manifest.csv")
    counts=Counter(r["split"] for r in split)
    groups=defaultdict(set)
    for row in split:groups[row["phash_group"]].add(row["split"])
    assert counts==dict(train=744,val=160,test=161)
    assert len({r["image"] for r in split})==1065
    assert len(groups)==750 and all(len(v)==1 for v in groups.values())
    arguments=list((ROOT/"historical/run_arguments").glob("*.yaml"))
    logs=list((ROOT/"historical/epoch_logs").glob("*.csv"))
    assert len(arguments)==len(logs)==40
    for path in logs:
        rows=csv_rows(path)
        assert len(rows)==200,(path.name,len(rows))
    weights=csv_rows(ROOT/"provenance/weight_manifest.csv")
    assert len(weights)==80
    assert len({(r["cell"],r["seed"],r["checkpoint"]) for r in weights})==80
    assert all(r["weight_present"]=="True" and len(r["weight_sha256"])==64 for r in weights)
    missing_imports=[]
    for path in (ROOT/"src").glob("*.py"):
        tree=ast.parse(path.read_text(encoding="utf-8"));mods=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):mods.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:mods.append(node.module)
        for module in mods:
            top=module.split(".")[0]
            if top.startswith(("wtbd","run_wtbd","evaluate_wtbd")) and not any((ROOT/"src"/name).exists() for name in [top,top+".py"]):
                missing_imports.append((path.name,module))
    assert not missing_imports,missing_imports
    assert not errors,errors
    report=dict(copied_fingerprints_verified=len(source_records),image_partition=dict(counts),
                phash_groups=750,group_cross_partition_overlap=False,
                original_training_argument_files=40,original_epoch_logs=40,epochs_per_run=200,
                original_checkpoint_manifest_rows=80,missing_local_source_imports=[],
                source_code_modified_during_audit=False,gpu_used=False)
    (ROOT/"outputs/package_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
