"""Isolated nearest-control and 2x2 ESB study; existing runs stay read-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "controls_esb_v2"
RUNS = ROOT / "runs_controls_esb_v2"
SEEDS = (42, 43, 44, 45, 46)
NEW = {
    "v8_s1": dict(version="v8", cell="S1", data="wtbd_fused.yaml", architecture="f0", supervision="hard_average"),
    "v8_a_esb": dict(version="v8", cell="A_ESB", data="wtbd_primary.yaml", architecture="f1_esb", supervision="single_a"),
    "v11_a_esb": dict(version="v11", cell="A_ESB", data="wtbd_primary.yaml", architecture="f1_esb", supervision="single_a"),
}
GROUPS = {
    "all": {},
    "ar_lt5": {"aspect_min": 1.0, "aspect_max": 5.0},
    "ar_ge5": {"aspect_min": 5.0},
    "ar_lt3": {"aspect_min": 1.0, "aspect_max": 3.0},
    "ar_3to5": {"aspect_min": 3.0, "aspect_max": 5.0},
    "area_lt1pct": {"area_max": .01},
    "area_ge1pct": {"area_min": .01},
    "area_1to10pct": {"area_min": .01, "area_max": .1},
    "area_ge10pct": {"area_min": .1},
}


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def epochs(path):
    csv_path = Path(path) / "results.csv"
    if not csv_path.exists():
        return 0
    rows = csv_read(csv_path)
    if not rows:
        return 0
    last = int(float(rows[-1]["epoch"]))
    if last != len(rows):
        raise ValueError(f"Noncontiguous epoch log at {path}")
    return last


def completed(path):
    return epochs(path) == 200 and all((path / "weights" / f"{c}.pt").exists() for c in ("best", "last"))


def run_path(experiment, seed):
    return RUNS / experiment / f"seed_{seed}"


def cells():
    """Canonical paths, using each original baseline only once."""
    result = []
    for seed in SEEDS:
        for cell, name, legacy in (
            ("A", "a_primary", "a_primary_yolo11n"),
            ("B", "b_second", "b_second_yolo11n"),
            ("S1", "s1_average", "c_fused_yolo11n"),
        ):
            path = ROOT / "runs" / legacy if seed == 42 else ROOT / "runs_repeated" / name / f"seed_{seed}"
            result.append(dict(version="v11", cell=cell, seed=seed, run=str(path)))
        for cell, name in (("S2", "f0_s2_yolo11n"), ("S2_ESB", "f1_s2_esb")):
            result.append(dict(version="v11", cell=cell, seed=seed, run=str(ROOT / "runs_feature_s2_v1" / name / f"seed_{seed}")))
        for cell, name in (("A", "v8_a_f0"), ("B", "v8_b_f0"), ("S2", "v8_s2_f0"), ("S2_ESB", "v8_s2_f1")):
            result.append(dict(version="v8", cell=cell, seed=seed, run=str(ROOT / "runs_yolov8_ab_s2_f1_v1" / name / f"seed_{seed}")))
        for experiment, spec in NEW.items():
            result.append(dict(version=spec["version"], cell=spec["cell"], seed=seed, run=str(run_path(experiment, seed))))
    return result


def freeze():
    import torch
    import ultralytics
    files = [ROOT / "data" / f for f in ("split_manifest.csv", "dual_annotation_audit.csv", "preparation_summary.json")]
    files += [ROOT / "configs" / f for f in ("yolov8n_wtbd.yaml", "yolov8n_wtbd_esb.yaml", "yolo11n_wtbd_esb.yaml")]
    files += [ROOT / "weights" / f for f in ("yolov8n.pt", "yolo11n.pt")]
    files += [ROOT / "scripts" / f for f in ("wtbd_modules.py", "wtbd_dual.py", "run_wtbd_yolov8_ab_s2_f1.py", "run_wtbd_feature_experiments.py")]
    files += [ROOT / "scripts" / f for f in ("wtbd_control_study.py", "evaluate_wtbd_controls.py", "analyze_wtbd_controls.py")]
    payload = dict(study_id="WTBD-CONTROLS-ESB-V2", experiments=NEW, seeds=list(SEEDS), epochs=200,
                   imgsz=768, batch=16, workers=4, groups=GROUPS,
                   torch=torch.__version__, ultralytics=ultralytics.__version__,
                   source_hashes={str(p): digest(p) for p in files},
                   primary="validation-selected best.pt, mean of A/B test AP50:95",
                   sensitivity="last.pt at epoch200; never replace primary based on test results",
                   checkpoint_caveat="Historical A and B select on their own val labels; S1/S2 select on fused val. New A_ESB follows A val; new S1 follows fused val. Also report fixed-last interaction.",
                   primary_contrasts=["S2-A", "S2-B", "S2-S1", "S2_ESB-S2", "A_ESB-A", "(S2_ESB-S2)-(A_ESB-A)"],
                   geometry="aspect=max(w/h,h/w), area=w*h/(image_w*image_h); GT-specific membership",
                   inference=dict(imgsz=768, batch=16, conf=.001, iou=.7, max_det=300, half=False),
                   subgroup_metric="conditional subset AP with full-image FP exposure and one-to-one ignored-GT matching; not COCO small/medium/large AP",
                   subgroup_primary="ar_ge5 versus ar_lt5; area_lt1pct versus area_ge1pct; detailed bins secondary",
                   statistics="paired seed t intervals and exact two-sided sign flips; Holm over primary AP contrasts; exploratory subgroup/class intervals; no optional stopping",
                   scope="same fixed dataset split; no new external data; not asset-level validation")
    if payload["torch"] != "2.3.1+cu121" or payload["ultralytics"] != "8.4.37":
        raise RuntimeError("Runtime changed relative to historical study; refusing a confounded extension")
    path = REPORT / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "frozen_at"} != payload:
            raise RuntimeError("Frozen protocol/source changed; do not silently resume")
    else:
        json_write(path, dict(frozen_at=now(), **payload))
    return payload


def queue():
    rows = [dict(experiment=e, seed=s, epochs=epochs(run_path(e, s)), complete=completed(run_path(e, s)), run=str(run_path(e, s))) for s in SEEDS for e in NEW]
    json_write(REPORT / "queue.json", dict(updated_at=now(), completed=sum(r["complete"] for r in rows), total=15, rows=rows))
    return rows


def train_one(experiment, seed):
    import torch
    from ultralytics import YOLO
    from run_wtbd_yolov8_ab_s2_f1 import build_model as build_v8, train_args as args_v8
    from run_wtbd_feature_experiments import train_args as args_v11
    from wtbd_modules import register_wtbd_modules
    freeze()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    path = run_path(experiment, seed)
    if completed(path):
        return
    spec = NEW[experiment]
    defaults = SimpleNamespace(epochs=200, imgsz=768, batch=16, workers=4)
    if spec["version"] == "v8":
        model = build_v8(spec)
        kwargs = args_v8(seed, defaults, spec)
    else:
        register_wtbd_modules()
        model = YOLO(str(ROOT / "configs" / "yolo11n_wtbd_esb.yaml")).load(str(ROOT / "weights" / "yolo11n.pt"))
        kwargs = args_v11(seed, defaults)
        kwargs["data"] = str(ROOT / "data" / spec["data"])
    kwargs.update(project=str(RUNS / experiment), name=f"seed_{seed}", exist_ok=True)
    if path.exists() and any(path.iterdir()):
        if not (path / "weights" / "last.pt").exists():
            raise RuntimeError(f"Partial run without resumable checkpoint: {path}")
        model = YOLO(str(path / "weights" / "last.pt"))
        model.train(resume=True, device=0, workers=4)
    else:
        # These three controls intentionally use the ordinary detection trainer.
        model.train(**kwargs)
    if not completed(path):
        raise RuntimeError(f"Training did not produce complete 200-epoch artifacts: {path}")
    queue()


def stage(label, command):
    REPORT.mkdir(parents=True, exist_ok=True)
    status = dict(stage=label, updated_at=now(), controller_pid=os.getpid(), command=command)
    json_write(REPORT / "pipeline_status.json", status)
    with (REPORT / f"{label}.stdout.log").open("a", encoding="utf-8", buffering=1) as out, (REPORT / f"{label}.stderr.log").open("a", encoding="utf-8", buffering=1) as err:
        process = subprocess.Popen(command, cwd=str(ROOT.parent), stdout=out, stderr=err)
        status["child_pid"] = process.pid
        json_write(REPORT / "pipeline_status.json", status)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Stage {label} failed with exit code {returncode}")


def pipeline():
    import ctypes
    freeze()
    queue()
    # Keep the already-authorized overnight computation awake; restored on exit.
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        # Finish all five S1 seeds first: this is the user's nearest-control priority.
        for experiment in NEW:
            for seed in SEEDS:
                if not completed(run_path(experiment, seed)):
                    stage(f"train_{experiment}_{seed}", [sys.executable, __file__, "train-one", "--experiment", experiment, "--seed", str(seed)])
        stage("evaluate", [sys.executable, str(ROOT / "scripts" / "evaluate_wtbd_controls.py"), "--new"])
        stage("geometry", [sys.executable, str(ROOT / "scripts" / "evaluate_wtbd_controls.py"), "--geometry"])
        stage("analysis", [sys.executable, str(ROOT / "scripts" / "analyze_wtbd_controls.py"), "--require-complete"])
        queue()
        json_write(REPORT / "pipeline_status.json", dict(stage="complete", updated_at=now(), completed_runs=15,
                   paper=str(ROOT.parent / "papers" / "WTBD_experiments_controls_v2.md")))
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("COMET_MODE", "DISABLED")
    os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "train-one", "run", "status"))
    parser.add_argument("--experiment", choices=NEW)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.mode == "status":
        print(json.dumps(queue(), ensure_ascii=False))
    elif args.mode == "plan":
        freeze()
        print(json.dumps(queue(), ensure_ascii=False))
    elif args.mode == "train-one":
        if args.experiment is None or args.seed is None:
            parser.error("train-one requires --experiment and --seed")
        train_one(args.experiment, args.seed)
    else:
        try:
            pipeline()
        except Exception as exc:
            status_path = REPORT / "pipeline_status.json"
            previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            json_write(status_path, dict(stage="failed", updated_at=now(), error=repr(exc), previous_stage=previous))
            raise


if __name__ == "__main__":
    main()
