"""Run the isolated WTBD S2 feature-module experiment matrix F0--F5.

Every method uses the fixed S2 dual-target supervision. The only experimental
factor is the image-feature architecture. Formal defaults are five deterministic
seeds (42--46), 200 epochs, imgsz 768, and batch 16 for an RTX 4060 8 GB.

This study is deliberately isolated from ``runs_repeated``. Per-epoch numbers
are stored by Ultralytics in each run's ``results.csv``; best/last and 50-epoch
checkpoints follow the same policy as the previous dual-annotation study.
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
RUNS = ROOT / "runs_feature_s2_v1"
REPORTS = ROOT / "reports" / "feature_s2_v1"
WEIGHTS = ROOT / "weights" / "yolo11n.pt"
AUDIT = DATA / "dual_annotation_audit.csv"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
STUDY_ID = "WTBD-S2-FEATURE-V1"


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "f0_s2_yolo11n": {
        "paper_id": "F0",
        "label": "S2-YOLO11n",
        "architecture": "baseline",
        "config": None,
    },
    "f1_s2_esb": {
        "paper_id": "F1",
        "label": "S2-ESB",
        "architecture": "feature",
        "config": "yolo11n_wtbd_esb.yaml",
    },
    "f2_s2_lsk": {
        "paper_id": "F2",
        "label": "S2-LSK",
        "architecture": "feature",
        "config": "yolo11n_wtbd_lsk.yaml",
    },
    "f3_s2_dysample": {
        "paper_id": "F3",
        "label": "S2-DySample",
        "architecture": "feature",
        "config": "yolo11n_wtbd_dysample.yaml",
    },
    "f4_s2_esb_dysample": {
        "paper_id": "F4",
        "label": "S2-ESB-DySample",
        "architecture": "feature",
        "config": "yolo11n_wtbd_esb_dysample.yaml",
    },
    "f5_s2_lsk_dysample": {
        "paper_id": "F5",
        "label": "S2-LSK-DySample",
        "architecture": "feature",
        "config": "yolo11n_wtbd_lsk_dysample.yaml",
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


def completed_run(path: Path, epochs: int) -> bool:
    results = path / "results.csv"
    last = path / "weights" / "last.pt"
    if not results.exists() or not last.exists():
        return False
    try:
        with results.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return bool(rows) and int(float(rows[-1]["epoch"])) >= epochs
    except (KeyError, ValueError):
        return False


def observed_epochs(path: Path) -> int:
    results = path / "results.csv"
    if not results.exists():
        return 0
    try:
        with results.open("r", encoding="utf-8") as handle:
            return len(list(csv.DictReader(handle)))
    except (OSError, csv.Error):
        return 0


def register_for(spec: dict[str, Any]) -> None:
    if spec["architecture"] == "baseline":
        register_standard_modules()
    else:
        register_wtbd_modules()


def build_model(spec: dict[str, Any]) -> YOLO:
    register_for(spec)
    config = spec.get("config")
    if config is None:
        return YOLO(str(WEIGHTS))
    return YOLO(str(ROOT / "configs" / str(config))).load(str(WEIGHTS))


def train_args(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    """Frozen protocol shared by all F0--F5 runs."""
    return {
        "data": str(DATA / "wtbd_fused.yaml"),
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
    configs = {
        name: (
            None
            if EXPERIMENTS[name]["config"] is None
            else {
                "path": str(ROOT / "configs" / str(EXPERIMENTS[name]["config"])),
                "sha256": sha256(ROOT / "configs" / str(EXPERIMENTS[name]["config"])),
            }
        )
        for name in selected
    }
    manifest = {
        "study_id": STUDY_ID,
        "separate_from": ["runs", "runs_repeated"],
        "runs_root": str(RUNS),
        "experiments": {name: EXPERIMENTS[name] for name in selected},
        "configs": configs,
        "seeds": seeds,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "order": args.order,
        "supervision": "S2 dual-target IoU + dual-target DFL for every method",
        "epoch_metrics": "one row per epoch in each run_dir/results.csv",
        "checkpoint_policy": "best.pt, last.pt, and save_period=50",
        "dataset_audit_sha256": sha256(AUDIT),
        "pretrained_weights_sha256": sha256(WEIGHTS),
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
    for queue_index, (experiment, seed) in enumerate(
        selected_pairs(selected, seeds, args.order), start=1
    ):
        path = run_dir(experiment, seed)
        epochs_done = observed_epochs(path)
        if completed_run(path, args.epochs):
            status = "complete"
        elif (path / "weights" / "last.pt").exists() and epochs_done:
            status = "partial_resumable"
        else:
            status = "pending"
        spec = EXPERIMENTS[experiment]
        rows.append(
            {
                "queue_index": queue_index,
                "paper_id": spec["paper_id"],
                "experiment": experiment,
                "method": spec["label"],
                "seed": seed,
                "target_epochs": args.epochs,
                "observed_epochs": epochs_done,
                "status": status,
                "run_dir": str(path),
            }
        )
    queue_path = REPORTS / "run_queue.csv"
    with queue_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def train_selected(selected: list[str], seeds: list[int], args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("F0--F5 formal runs require CUDA")
    RUNS.mkdir(parents=True, exist_ok=True)
    write_manifest(selected, seeds, args)
    trainer = make_dual_trainer("dual_target", AUDIT)

    for experiment, seed in selected_pairs(selected, seeds, args.order):
        spec = EXPERIMENTS[experiment]
        register_for(spec)
        path = run_dir(experiment, seed)
        if completed_run(path, args.epochs):
            print(f"[skip] {experiment}, seed={seed}: complete at {path}")
            write_queue(selected, seeds, args)
            continue

        partial = path / "weights" / "last.pt"
        if partial.exists() and (path / "results.csv").exists():
            print(f"[resume] {experiment}, seed={seed}: {partial}")
            model = YOLO(str(partial))
            model.train(resume=True, device=0, workers=args.workers, trainer=trainer)
            write_queue(selected, seeds, args)
            continue

        model = build_model(spec)
        kwargs = train_args(seed, args)
        kwargs.update(
            {
                "project": str(RUNS / experiment),
                "name": f"seed_{seed}",
                "exist_ok": True,
                "trainer": trainer,
            }
        )
        print(
            f"[train] {experiment}, seed={seed}\n"
            + json.dumps(kwargs, ensure_ascii=False, indent=2, default=str)
        )
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
    parser.add_argument(
        "--log-prefix",
        default="",
        help="Optional absolute/relative prefix for line-buffered .stdout.log and .stderr.log files",
    )
    args = parser.parse_args()
    if args.log_prefix:
        log_prefix = Path(args.log_prefix).resolve()
        log_prefix.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = log_prefix.with_suffix(".stdout.log").open(
            "a", encoding="utf-8", buffering=1
        )
        sys.stderr = log_prefix.with_suffix(".stderr.log").open(
            "a", encoding="utf-8", buffering=1
        )
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
