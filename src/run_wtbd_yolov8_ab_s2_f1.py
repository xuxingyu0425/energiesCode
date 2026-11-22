"""Run the WTBD YOLOv8n A/B/S2-F0/S2-F1 cross-version study.

Formal defaults are five paired seeds (42--46), 200 epochs, imgsz 768, and
batch 16.  The four cells test whether S2 outperforms either single annotator
and whether the F1 elongated-strip block transfers from YOLO11n to YOLOv8n.
Every run has its own ``results.csv`` with one row per epoch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO

from wtbd_dual import make_dual_trainer
from wtbd_modules import register_standard_modules, register_wtbd_modules


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs_yolov8_ab_s2_f1_v1"
REPORTS = ROOT / "reports" / "yolov8_ab_s2_f1_v1"
WEIGHTS = ROOT / "weights" / "yolov8n.pt"
AUDIT = DATA / "dual_annotation_audit.csv"
F0_CONFIG = ROOT / "configs" / "yolov8n_wtbd.yaml"
ESB_CONFIG = ROOT / "configs" / "yolov8n_wtbd_esb.yaml"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
STUDY_ID = "WTBD-YOLOV8-AB-S2-F1-V1"


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "v8_a_f0": {
        "paper_id": "V8-A",
        "label": "YOLOv8n annotator A",
        "data": "wtbd_primary.yaml",
        "supervision": "single_a",
        "architecture": "f0",
    },
    "v8_b_f0": {
        "paper_id": "V8-B",
        "label": "YOLOv8n annotator B",
        "data": "wtbd_second.yaml",
        "supervision": "single_b",
        "architecture": "f0",
    },
    "v8_s2_f0": {
        "paper_id": "V8-S2-F0",
        "label": "YOLOv8n S2-F0",
        "data": "wtbd_fused.yaml",
        "supervision": "s2_dual_target",
        "architecture": "f0",
    },
    "v8_s2_f1": {
        "paper_id": "V8-S2-F1",
        "label": "YOLOv8n S2-F1 (ESB)",
        "data": "wtbd_fused.yaml",
        "supervision": "s2_dual_target",
        "architecture": "f1_esb",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(experiment: str, seed: int) -> Path:
    return RUNS / experiment / f"seed_{seed}"


def observed_epochs(path: Path) -> int:
    results = path / "results.csv"
    if not results.exists():
        return 0
    try:
        with results.open("r", encoding="utf-8") as handle:
            return len(list(csv.DictReader(handle)))
    except (OSError, csv.Error):
        return 0


def completed_run(path: Path, epochs: int) -> bool:
    return observed_epochs(path) >= epochs and (path / "weights" / "last.pt").exists()


def register_for(spec: dict[str, Any]) -> None:
    if spec["architecture"] == "f0":
        register_standard_modules()
    else:
        register_wtbd_modules()


def build_model(spec: dict[str, Any]) -> YOLO:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing pretrained YOLOv8n weights: {WEIGHTS}")
    register_for(spec)
    if spec["architecture"] == "f0":
        return YOLO(str(F0_CONFIG)).load(str(WEIGHTS))
    return YOLO(str(ESB_CONFIG)).load(str(WEIGHTS))


def trainer_for(spec: dict[str, Any]):
    if spec["supervision"] != "s2_dual_target":
        return None
    return make_dual_trainer("dual_target", AUDIT)


def train_args(seed: int, args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": str(DATA / str(spec["data"])),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": 0,
        "workers": args.workers,
        "cache": "disk",
        "amp": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "cos_lr": True,
        "patience": args.epochs + 1,
        "close_mosaic": 20,
        "degrees": 15.0,
        "translate": 0.10,
        "scale": 0.35,
        "fliplr": 0.5,
        "flipud": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "seed": seed,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "multi_scale": False,
        "save": True,
        "save_period": 50,
        "plots": False,
        "val": True,
        "verbose": True,
    }


def selected_pairs(selected: list[str], seeds: list[int], order: str) -> list[tuple[str, int]]:
    if order == "seed-major":
        return [(experiment, seed) for seed in seeds for experiment in selected]
    return [(experiment, seed) for experiment in selected for seed in seeds]


def write_manifest(selected: list[str], seeds: list[int], args: argparse.Namespace) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "study_id": STUDY_ID,
        "question_1": "Does S2 outperform annotator-A-only and annotator-B-only training on YOLOv8n?",
        "question_2": "Does F1-ESB transfer from YOLO11n to YOLOv8n under fixed S2 supervision?",
        "runs_root": str(RUNS),
        "experiments": {name: EXPERIMENTS[name] for name in selected},
        "seeds": seeds,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "order": args.order,
        "primary_metric": "mean annotator-A/B held-out test mAP50-95 from validation-selected best.pt",
        "epoch_metrics": "one row per epoch in each run_dir/results.csv",
        "checkpoint_policy": "best.pt, last.pt, and save_period=50",
        "dataset_audit_sha256": sha256(AUDIT),
        "pretrained_weights_sha256": sha256(WEIGHTS) if WEIGHTS.exists() else None,
        "f0_config_sha256": sha256(F0_CONFIG),
        "esb_config_sha256": sha256(ESB_CONFIG),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    (REPORTS / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_queue(selected: list[str], seeds: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, (experiment, seed) in enumerate(selected_pairs(selected, seeds, args.order), 1):
        path = run_dir(experiment, seed)
        done = observed_epochs(path)
        status = "complete" if completed_run(path, args.epochs) else "partial_resumable" if done else "pending"
        spec = EXPERIMENTS[experiment]
        rows.append(
            {
                "queue_index": index,
                "paper_id": spec["paper_id"],
                "experiment": experiment,
                "method": spec["label"],
                "seed": seed,
                "target_epochs": args.epochs,
                "observed_epochs": done,
                "status": status,
                "run_dir": str(path),
            }
        )
    with (REPORTS / "run_queue.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def train_selected(selected: list[str], seeds: list[int], args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Formal YOLOv8 runs require CUDA")
    RUNS.mkdir(parents=True, exist_ok=True)
    write_manifest(selected, seeds, args)

    for experiment, seed in selected_pairs(selected, seeds, args.order):
        spec = EXPERIMENTS[experiment]
        register_for(spec)
        trainer = trainer_for(spec)
        path = run_dir(experiment, seed)
        if completed_run(path, args.epochs):
            print(f"[skip] {experiment}, seed={seed}: complete at {path}")
            write_queue(selected, seeds, args)
            continue

        partial = path / "weights" / "last.pt"
        if partial.exists() and (path / "results.csv").exists():
            print(f"[resume] {experiment}, seed={seed}: {partial}")
            model = YOLO(str(partial))
            resume_args: dict[str, Any] = {"resume": True, "device": 0, "workers": args.workers}
            if trainer is not None:
                resume_args["trainer"] = trainer
            model.train(**resume_args)
            write_queue(selected, seeds, args)
            continue

        model = build_model(spec)
        kwargs = train_args(seed, args, spec)
        kwargs.update(
            {
                "project": str(RUNS / experiment),
                "name": f"seed_{seed}",
                "exist_ok": True,
            }
        )
        if trainer is not None:
            kwargs["trainer"] = trainer
        print(f"[train] {experiment}, seed={seed}\n" + json.dumps(kwargs, ensure_ascii=False, indent=2, default=str))
        model.train(**kwargs)
        write_queue(selected, seeds, args)


def parse_selected(text: str) -> list[str]:
    if text == "all":
        return list(EXPERIMENTS)
    selected = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in selected if item not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}; valid={list(EXPERIMENTS)}")
    return selected


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be a non-empty unique list")
    return seeds


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("COMET_MODE", "DISABLED")
    os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "train"))
    parser.add_argument("--experiments", default="all")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--order", choices=("seed-major", "experiment-major"), default="seed-major")
    parser.add_argument("--log-prefix", default="")
    args = parser.parse_args()
    if args.log_prefix:
        prefix = Path(args.log_prefix).resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = prefix.with_suffix(".stdout.log").open("a", encoding="utf-8", buffering=1)
        sys.stderr = prefix.with_suffix(".stderr.log").open("a", encoding="utf-8", buffering=1)
        print(f"[log] study={STUDY_ID}, pid={os.getpid()}, cwd={Path.cwd()}")

    selected = parse_selected(args.experiments)
    seeds = parse_seeds(args.seeds)
    write_manifest(selected, seeds, args)
    rows = write_queue(selected, seeds, args)
    if args.mode == "plan":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        train_selected(selected, seeds, args)


if __name__ == "__main__":
    main()
