"""Reviewer-requested S1+CBAM extension. Historical artifacts are read-only.

Use the historical S2+CBAM initialization and paired augmentation pipeline.
Only the training regression criterion changes to native single fused-box loss.
Run with the existing CV_uav environment. Each seed runs in a fresh process.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
BASE = Path(__file__).resolve().parent
EXP = BASE.parents[2] / "experiments"
HIST = EXP / "reports/yolov8_attention_comparison_v1"
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))

# The current wtbd_dual.py later acquired unrelated soft-class strategies.
# Import the exact frozen file used for the historical S2+CBAM experiment.
FROZEN_DUAL = HIST / "frozen_sources/scripts/wtbd_dual.py"
spec = importlib.util.spec_from_file_location("wtbd_dual", FROZEN_DUAL)
module = importlib.util.module_from_spec(spec)
sys.modules["wtbd_dual"] = module
spec.loader.exec_module(module)

import torch
import ultralytics
import yaml
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import init_seeds
from wtbd_dual import DualAnnotationTrainer
import run_wtbd_yolov8_attention as historical
from run_wtbd_yolov8_ab_s2_f1 import train_args

SEEDS = (42, 43, 44, 45, 46)
RUNS = BASE / "runs"
RECORDS = BASE / "records"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def verify_inputs():
    old = json.loads((HIST / "protocol.json").read_text(encoding="utf-8"))
    checked = []
    for name, expected in old["sha256"].items():
        p = FROZEN_DUAL if Path(name).name == "wtbd_dual.py" else Path(name)
        actual = sha(p)
        assert actual == expected, (str(p), "Historical input changed")
        checked.append(dict(path=str(p), sha256=actual))
    assert platform.python_version() == old["python"]
    assert torch.__version__ == old["torch"]
    assert ultralytics.__version__ == old["ultralytics"]
    assert torch.cuda.is_available(), "CUDA is required; refusing CPU substitution"
    assert torch.cuda.get_device_name(0) == old["gpu"]
    return old, checked


class S1PairedPipelineTrainer(DualAnnotationTrainer):
    """Carry paired boxes through the same augmentations, train on fused boxes.

    Native DetectionModel uses v8DetectionLoss and ignores auxiliary keypoints.
    Assignment, quality targets, classification, and loss gains stay native.
    Validation remains the original ordinary fused-validation dataset.
    """

    dual_audit_path = str(EXP / "data/dual_annotation_audit.csv")

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"],
                               verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def _build_train_pipeline(self):
        if self.args.batch != 16 or self.batch_size != 16:
            raise RuntimeError("Fixed historical batch16; no automatic batch reduction")
        return super()._build_train_pipeline()

    def _handle_nan_recovery(self, epoch):
        if self.loss is not None and not torch.isfinite(self.loss).all():
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}; preserve failed run")
        if self.fitness is not None and not math.isfinite(float(self.fitness)):
            raise RuntimeError(f"Non-finite validation fitness at epoch {epoch + 1}")
        return False


def arguments(seed):
    args = train_args(seed, argparse.Namespace(epochs=200, imgsz=768, batch=16, workers=4),
                      {"data": "wtbd_fused.yaml"})
    args.update(project=str(RUNS), name=f"seed_{seed}", exist_ok=True)
    return args


def preflight():
    old, checked = verify_inputs()
    shared = {}
    for seed in SEEDS:
        model, common, meta = historical.build("cbam", seed, callbacks=False)
        expected = json.loads((HIST / f"records/cbam/seed_{seed}/initialization.json").read_text(encoding="utf-8"))
        assert meta["full_state_sha256"] == expected["full_state_sha256"], (seed, "Initial model differs")
        assert meta["common_state_sha256"] == expected["common_state_sha256"]
        assert meta["parameters"] == 3014982
        shared[str(seed)] = meta
        old_args = yaml.safe_load((EXP / f"runs_yolov8_attention_comparison_v1/cbam/seed_{seed}/args.yaml").read_text(encoding="utf-8"))
        for key, value in arguments(seed).items():
            if key not in {"project", "name", "exist_ok"}:
                actual = str(value) if key == "device" else value
                assert old_args[key] == actual, (seed, key, old_args[key], actual)
        del model, common
        gc.collect()
    protocol = dict(
        study="Reviewer-requested S1+CBAM vs historical S2+CBAM",
        created_at=now(), seeds=list(SEEDS), source_environment=dict(
            python=platform.python_version(), torch=torch.__version__,
            ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0)),
        primary="Epoch200, annotator-A AP50-95 and common-object IoU, paired by seed",
        secondary="Original fused-validation-selected best checkpoints, B reference sensitivity, validation-locked P/R/F1/FP/FN",
        threshold_rule="A validation grid0.01..0.99 step0.01, maximize pooled F1; smallest threshold breaks ties",
        test_rule="No configuration, checkpoint, or threshold tuning using test results",
        intentional_difference="Native single fused-box regression instead of averaged dual original-box regression",
        unchanged="CBAM architecture, full seed-specific initialization, paired data/augmentation, assignment targets, class convention, gains, training hyperparameters, split, environment",
        first_batch_check="Compare every tensor and image filename with historical S2+CBAM first augmented batch",
        no_fallback="No automatic batch reduction, shortened training, silent restart, or NaN rollback",
        checked_inputs=checked, initialization=shared, train_args={str(s): arguments(s) for s in SEEDS},
        runner_sha256=sha(__file__), historical_protocol_sha256=sha(HIST / "protocol.json"),
    )
    path = BASE / "protocol.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        assert {k: v for k, v in prior.items() if k != "created_at"} == {
            k: v for k, v in protocol.items() if k != "created_at"}, "Frozen extension protocol changed"
    else:
        save(path, protocol)
    dest = BASE / "frozen_sources"
    for item in checked:
        src = Path(item["path"])
        rel = src.relative_to(HIST / "frozen_sources") if src == FROZEN_DUAL else src.relative_to(EXP)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(src, target)
        else:
            assert sha(target) == item["sha256"]
    shutil.copy2(__file__, dest / "runner.py")
    save(BASE / "preflight.json", dict(passed=True, time=now(), seeds=list(SEEDS),
                                      full_initialization_identical=True,
                                      training_arguments_identical_except_output_paths=True))
    print("PREFLIGHT_PASSED", flush=True)


def train_one(seed):
    verify_inputs()
    protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["runner_sha256"] == sha(__file__)
    rec = RECORDS / f"seed_{seed}"
    rec.mkdir(parents=True, exist_ok=True)
    run = RUNS / f"seed_{seed}"
    if (rec / "complete.json").exists():
        done = json.loads((rec / "complete.json").read_text(encoding="utf-8"))
        assert done["last_sha256"] == sha(run / "weights/last.pt")
        print("ALREADY_COMPLETE", seed, flush=True)
        return
    if (run / "results.csv").exists() or (run / "weights/last.pt").exists():
        raise RuntimeError(f"Preserved partial run requires explicit recovery: {run}")
    model, common, meta = historical.build("cbam", seed, callbacks=False)
    assert meta["full_state_sha256"] == protocol["initialization"][str(seed)]["full_state_sha256"]
    save(rec / "initialization.json", meta)
    started = time.perf_counter()

    def on_start(t):
        actual = historical.state_hash(t.model.state_dict())
        assert actual == meta["full_state_sha256"], "Model changed before training"
        assert type(t.model.init_criterion()).__name__ == "v8DetectionLoss"
        expected_args = yaml.safe_load((EXP / f"runs_yolov8_attention_comparison_v1/cbam/seed_{seed}/args.yaml").read_text(encoding="utf-8"))
        current_args = vars(t.args)
        excluded = {"project", "name", "save_dir"}
        differences = {k: [v, current_args.get(k)] for k, v in expected_args.items()
                       if k not in excluded and current_args.get(k) != v}
        assert not differences, differences
        save(rec / "arguments_verified.json", dict(passed=True, excluded_output_fields=sorted(excluded)))
        save(rec / "initialization_verified.json", dict(passed=True, full_state_sha256=actual))
        init_seeds(seed, deterministic=True)
        torch.cuda.reset_peak_memory_stats()
        original = t.preprocess_batch
        seen = False

        def preprocess(batch):
            nonlocal seen
            if not seen:
                fields = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
                record = dict(seed=seed, tensor_sha256=historical.state_hash(fields),
                              tensor_shapes={k: list(v.shape) for k, v in fields.items()},
                              image_files=list(batch.get("im_file", [])), time=now())
                expected = json.loads((HIST / f"records/cbam/seed_{seed}/first_batch.json").read_text(encoding="utf-8"))
                record["matches_historical_s2_cbam"] = (
                    record["tensor_sha256"] == expected["tensor_sha256"] and
                    record["image_files"] == expected["image_files"])
                save(rec / "first_batch.json", record)
                assert record["matches_historical_s2_cbam"], "First augmented batch differs from historical S2"
                seen = True
            return original(batch)
        t.preprocess_batch = preprocess

    def on_epoch(t):
        if t.loss is not None and not torch.isfinite(t.loss).all():
            raise RuntimeError(f"Non-finite loss at epoch {t.epoch + 1}")
        status = dict(seed=seed, epoch=t.epoch + 1, target_epochs=200, time=now(),
                      elapsed_seconds=time.perf_counter() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      fitness=float(t.fitness) if t.fitness is not None else None)
        save(rec / "progress.json", status)
        if (t.epoch + 1) % 10 == 0:
            print("EPOCH_PROGRESS", json.dumps(status), flush=True)

    model.add_callback("on_train_start", on_start)
    model.add_callback("on_fit_epoch_end", on_epoch)
    args = arguments(seed)
    save(rec / "train_arguments.json", args)
    save(rec / "gpu_before_training.json", dict(time=now(), output=subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,utilization.gpu", "--format=csv"], text=True)))
    model.train(**args, trainer=S1PairedPipelineTrainer)
    rows = historical.read_csv(run / "results.csv")
    assert len(rows) == 200
    assert all(math.isfinite(float(v)) for row in rows for k, v in row.items() if "loss" in k)
    assert (run / "weights/best.pt").is_file()
    save(rec / "complete.json", dict(seed=seed, complete=True, epochs=200, time=now(),
         wall_seconds=time.perf_counter() - started, csv_cumulative_seconds=float(rows[-1]["time"]),
         last_sha256=sha(run / "weights/last.pt"), best_sha256=sha(run / "weights/best.pt"),
         first_batch_identical=True, full_initialization_identical=True))
    print("SEED_COMPLETE", seed, flush=True)


def queue():
    assert (BASE / "preflight.json").exists()
    for index, seed in enumerate(SEEDS, 1):
        rec = RECORDS / f"seed_{seed}"
        if (rec / "complete.json").exists():
            continue
        save(BASE / "queue_status.json", dict(status="training", seed=seed,
             index=index, total=5, time=now(), pid=os.getpid()))
        log = BASE / f"seed_{seed}.console.log"
        with log.open("a", encoding="utf-8") as f:
            result = subprocess.run([sys.executable, "-u", str(Path(__file__)), "train-one", "--seed", str(seed)],
                                    stdout=f, stderr=subprocess.STDOUT, cwd=str(EXP))
        if result.returncode:
            save(BASE / "queue_status.json", dict(status="failed", seed=seed,
                 returncode=result.returncode, time=now(), log=str(log)))
            raise RuntimeError(f"Seed {seed} failed; inspect {log}")
    save(BASE / "queue_status.json", dict(status="training_complete", seeds=list(SEEDS), time=now()))
    print("FIVE_SEEDS_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "train-one", "queue"])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    try:
        {"preflight": preflight, "queue": queue, "train-one": lambda: train_one(args.seed)}[args.stage]()
    except Exception:
        traceback.print_exc()
        raise
