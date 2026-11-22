"""Run and evaluate the WTBD YOLOv8n dual-annotation fusion study.

S4 (cell D1): S2 dual-target boxes + 0.5/0.5 soft class targets on matched class conflicts.
S5: S4 + ESB.
D2 (S4 + one-sided objects down-weighted to 0.6) and D3 (D2 + ESB) were stopped on
2026-09-29: D2 was below S2 at best and last on seed 42, failing the pre-registered rule.

All training hyper-parameters are imported from the A/B/S2 study so the new
cells are directly comparable with the existing A, B, S2, A+ESB and S2+ESB runs.
Each run is evaluated on the annotator-A and annotator-B test labels (best and
last) immediately after training, then the comparison table is refreshed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from run_wtbd_yolov8_ab_s2_f1 import AUDIT, DATA, ROOT, build_model, observed_epochs, register_for, sha256, train_args
from wtbd_dual import make_dual_trainer


RUNS = ROOT / "runs_yolov8_dual_fusion_v1"
REPORTS = ROOT / "reports" / "yolov8_dual_fusion_v1"
BASELINE_RAW = ROOT / "reports" / "controls_esb_v2" / "combined_raw.csv"
# S2 + CBAM (5 seeds) from the YOLOv8 attention study; same evaluation protocol as BASELINE_RAW.
S2_CBAM_RAW = ROOT / "reports" / "yolov8_attention_comparison_v1" / "evaluation_raw.csv"
STUDY_ID = "WTBD-YOLOV8-DUAL-FUSION-V1"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
ACTIVE_EXPERIMENTS = ("v8_d1_soft_class", "v8_s5_d1_esb")
SEED_EXTENSION_NOTE = (
    "Added 2026-09-30 after seeds 42-44 were seen. S2 and S4 each get seeds 47-49. The pre-registered "
    "5-seed verdict stands; the 8-seed S4-S2 comparison is post-hoc exploratory analysis with no pass threshold."
)
CLASS_WEIGHT_A = 0.5  # fixed a priori: no preference between the two label files
SINGLE_ANNOTATOR_WEIGHT = 0.6

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "v8_d1_soft_class": {
        "cell": "D1",
        "label": "S4: S2 + soft class",
        "strategy": "dual_target_soft_class",
        "architecture": "f0",
    },
    "v8_s5_d1_esb": {
        "cell": "S5",
        "label": "S5: S4 + ESB",
        "strategy": "dual_target_soft_class",
        "architecture": "f1_esb",
    },
    # Post-hoc seed extension (47-49) for the S4-vs-S2 comparison; same training path as the S2 baseline.
    "v8_s2_extra": {
        "cell": "S2",
        "label": "S2 extra seeds (post hoc)",
        "strategy": "dual_target",
        "architecture": "f0",
    },
    # A-only labels + CBAM at the same neck positions as ESB (layers 15/18); built exactly like the
    # S2+CBAM attention study, trained like the A baseline (wtbd_primary.yaml, standard trainer).
    "v8_a_cbam": {
        "cell": "A_CBAM",
        "label": "A + CBAM",
        "strategy": None,
        "architecture": "cbam",
        "data": "wtbd_primary.yaml",
    },
    # Stopped after seed 42 (see module docstring); kept so recorded results stay traceable.
    "v8_d2_full": {
        "cell": "D2",
        "label": "S2 + soft class + one-sided 0.6 (stopped)",
        "strategy": "dual_target_full",
        "architecture": "f0",
    },
    "v8_d3_full_esb": {
        "cell": "D3",
        "label": "D2 + ESB (stopped at epoch 77, never evaluated)",
        "strategy": "dual_target_full",
        "architecture": "f1_esb",
    },
}
EVAL_SETS = {
    "annotator_a": DATA / "wtbd_gt_primary.yaml",
    "annotator_b": DATA / "wtbd_gt_second.yaml",
}
CLASS_NAMES = ("craze", "crack", "corrosion", "hide_craze", "surface_injure", "thunderstrike")
BASELINE_CELLS = ("A", "B", "S2", "A_ESB", "S2_ESB")
BASELINE_LABELS = {
    "A": "单 A",
    "B": "单 B",
    "S2": "S2 仅位置融合",
    "A_ESB": "A + ESB",
    "S2_ESB": "S2 + ESB",
    "D1": "S4 = S2 + 软类别",
    "S5": "S5 = S4 + ESB",
    "A_CBAM": "A + CBAM",
    "S2_CBAM": "S2 + CBAM",
    "D2": "D2 = S4 + 单侧 0.6（已终止）",
}
RAW_FIELDS = [
    "experiment", "cell", "seed", "checkpoint", "ground_truth",
    "precision", "recall", "map50", "map50_95", "inference_ms",
    *[f"map50_95_{name}" for name in CLASS_NAMES],
]


def run_dir(experiment: str, seed: int) -> Path:
    return RUNS / experiment / f"seed_{seed}"


def write_status(**fields: Any) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    fields["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (REPORTS / "queue_status.json").write_text(json.dumps(fields, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manifest(seeds: list[int], args: argparse.Namespace) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "study_id": STUDY_ID,
        "question": "Does fusing class and existence information from both label files improve on S2 (box-only fusion)?",
        "experiments": EXPERIMENTS,
        "class_weight_a": CLASS_WEIGHT_A,
        "class_weight_rule": "fixed a priori at 0.5 (no preference between label files); not tuned on val or test",
        "single_annotator_weight": SINGLE_ANNOTATOR_WEIGHT,
        "seeds": seeds,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "primary_metric": "annotator-A test mAP50-95, validation-selected best.pt; last.pt and mean A/B reported as sensitivity",
        "decision_rule": "Fixed 2026-09-29 before seeds 43-46 were run; seed 42 is included. On annotator-A test "
        "mAP50-95, S4 is effective only if S4-S2 has a positive mean and >=4/5 positive seeds at both best and last; "
        "S5 is effective only if S5-(S2+ESB) and S5-(A+ESB) each satisfy the same condition.",
        "stopped_cells": "D2 failed its original rule on seed 42 (below S2 at best and last); D3 stopped at epoch 77.",
        "seed_extension": SEED_EXTENSION_NOTE,
        "baselines": str(BASELINE_RAW),
        "dataset_audit_sha256": sha256(AUDIT),
    }
    (REPORTS / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def register_modules(spec: dict[str, Any]) -> None:
    if spec["architecture"] == "cbam":
        from wtbd_attention import register_attention_modules
        from wtbd_modules import register_standard_modules

        register_standard_modules()
        register_attention_modules()
    else:
        register_for(spec)


def train_one(experiment: str, seed: int, args: argparse.Namespace) -> None:
    spec = EXPERIMENTS[experiment]
    register_modules(spec)
    # Single-annotator cells (strategy None) use the standard trainer, like the A/B baselines.
    trainer = (
        make_dual_trainer(spec["strategy"], AUDIT, single_annotator_weight=SINGLE_ANNOTATOR_WEIGHT)
        if spec["strategy"]
        else None
    )
    path = run_dir(experiment, seed)
    if observed_epochs(path) >= args.epochs and (path / "weights" / "last.pt").exists():
        print(f"[skip] {experiment} seed={seed}: complete")
        return
    partial = path / "weights" / "last.pt"
    if partial.exists() and (path / "results.csv").exists():
        print(f"[resume] {experiment} seed={seed}")
        resume_kwargs: dict[str, Any] = {"resume": True, "device": 0, "workers": args.workers}
        if trainer is not None:
            resume_kwargs["trainer"] = trainer
        YOLO(str(partial)).train(**resume_kwargs)
        return
    if spec["architecture"] == "cbam":
        from run_wtbd_yolov8_attention import build as build_attention_model

        model = build_attention_model("cbam", seed, callbacks=False)[0]
    else:
        model = build_model(spec)
    kwargs = train_args(seed, args, {**spec, "data": spec.get("data", "wtbd_fused.yaml")})
    kwargs.update({"project": str(RUNS / experiment), "name": f"seed_{seed}", "exist_ok": True})
    if trainer is not None:
        kwargs["trainer"] = trainer
    print(f"[train] {experiment} seed={seed}\n" + json.dumps(kwargs, ensure_ascii=False, indent=2, default=str))
    model.train(**kwargs)


def read_raw() -> list[dict[str, Any]]:
    path = REPORTS / "evaluation_raw.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_raw(rows: list[dict[str, Any]]) -> None:
    with (REPORTS / "evaluation_raw.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one(experiment: str, seed: int, args: argparse.Namespace) -> None:
    spec = EXPERIMENTS[experiment]
    rows = read_raw()
    done = {(r["experiment"], int(r["seed"]), r["checkpoint"], r["ground_truth"]) for r in rows}
    register_modules(spec)
    for checkpoint in ("best", "last"):
        weight = run_dir(experiment, seed) / "weights" / f"{checkpoint}.pt"
        missing = [gt for gt in EVAL_SETS if (experiment, seed, checkpoint, gt) not in done]
        if not missing:
            continue
        model = YOLO(str(weight))
        for gt in missing:
            metrics = model.val(
                data=str(EVAL_SETS[gt]), split="test", imgsz=args.imgsz, batch=args.batch, device=0,
                workers=args.workers, plots=False, save_json=False, verbose=False,
                project=str(run_dir(experiment, seed) / "evaluation"), name=f"{checkpoint}_{gt}", exist_ok=True,
            )
            row = {
                "experiment": experiment, "cell": spec["cell"], "seed": seed, "checkpoint": checkpoint,
                "ground_truth": gt, "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
                "inference_ms": float(metrics.speed.get("inference", 0.0)),
            }
            for index, name in enumerate(CLASS_NAMES):
                row[f"map50_95_{name}"] = float(metrics.box.maps[index])
            rows.append(row)
            write_raw(rows)
            print("[eval] " + json.dumps(row, ensure_ascii=False))
        del model
        gc.collect()
        torch.cuda.empty_cache()


def collect_scores(key: str = "map50_95") -> dict[tuple[str, int, str, str], float]:
    """(cell, seed, checkpoint, gt) -> metric `key` in percent, for baselines and new cells."""
    scores: dict[tuple[str, int, str, str], float] = {}
    with BASELINE_RAW.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["version"] == "v8" and row["cell"] in BASELINE_CELLS:
                scores[(row["cell"], int(row["seed"]), row["checkpoint"], row["ground_truth"])] = 100 * float(row[key])
    with S2_CBAM_RAW.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "cbam" and row["split"] == "test":
                scores[("S2_CBAM", int(row["seed"]), row["checkpoint"], row["ground_truth"])] = 100 * float(row[key])
    for row in read_raw():
        scores[(row["cell"], int(row["seed"]), row["checkpoint"], row["ground_truth"])] = 100 * float(row[key])
    return scores


def metric(scores: dict, cell: str, seed: int, checkpoint: str, view: str) -> float | None:
    a = scores.get((cell, seed, checkpoint, "annotator_a"))
    b = scores.get((cell, seed, checkpoint, "annotator_b"))
    if view == "A":
        return a
    if view == "B":
        return b
    return None if a is None or b is None else 0.5 * (a + b)


def fmt(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.fmean(values):.2f} ± {statistics.stdev(values):.2f}"


def write_comparison() -> None:
    scores = collect_scores()
    # Seeds in which at least one active cell has finished; baselines are restricted to the same seeds.
    active_cells = [EXPERIMENTS[name]["cell"] for name in ACTIVE_EXPERIMENTS] + ["S2", "A_CBAM"]
    seeds = sorted({seed for (cell, seed, ck, gt) in scores if cell in active_cells and ck == "last"})
    if not seeds:
        return
    lines = [
        f"# YOLOv8n 双标注融合对比（种子 {','.join(map(str, seeds))}）",
        "",
        f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S')}。只统计新实验已完成的种子，基线取同一批种子，保证配对。",
        "主指标：A 测试真值、best 权重的 AP50:95；AP50 作为补充指标。均值 ± 标准差；只有 1 个种子时不给标准差。",
    ]
    for key, title in (("map50_95", "AP50:95（主指标）"), ("map50", "AP50（补充指标）")):
        lines += ["", f"## {title}", ""]
        lines += comparison_section(collect_scores(key), seeds)
    lines += [
        "",
        "判定规则（2026-09-29 在种子 43–46 开跑前固定，种子 42 计入，只针对 AP50:95）：S4 − S2 在 best 和 last 下均值都为正，"
        "且 5 个种子中至少 4 个为正，S4 才算有效；S5 需要对 S2+ESB 和 A+ESB 分别满足同样条件。",
        "D2（S4 + 单侧 0.6）在种子 42 上 best、last 均低于 S2，按原规则终止，仅保留种子 42 的记录。",
        "事后追加（2026-09-30，看到种子 42–44 后决定）：S2 与 S4 各补种子 47–49。原 5 种子判定结论不变；"
        "8 种子的 S4 − S2 只作探索性报告，不设通过门槛。各行的种子数见“完成种子”列，配对差值只在两方都有的种子上计算。",
    ]
    (REPORTS / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_section(scores: dict, seeds: list[int]) -> list[str]:
    cells = ["A", "B", "S2", "D1", "D2", "A_ESB", "A_CBAM", "S2_ESB", "S2_CBAM", "S5"]
    lines = [
        "| 方法 | A 真值 best | A 真值 last | B 真值 best | A/B 均值 best | 完成种子 |",
        "|---|---|---|---|---|---|",
    ]
    for cell in cells:
        done = [s for s in seeds if metric(scores, cell, s, "best", "A") is not None]
        cols = [
            fmt([metric(scores, cell, s, ck, view) for s in done if metric(scores, cell, s, ck, view) is not None])
            for ck, view in (("best", "A"), ("last", "A"), ("best", "B"), ("best", "mean"))
        ]
        lines.append(f"| {BASELINE_LABELS[cell]} | " + " | ".join(cols) + f" | {len(done)} |")

    lines += ["", "配对差值（A 真值，百分点；逐种子差值与胜负）：", "", "| 比较 | best | last |", "|---|---|---|"]
    for left, right in (("D1", "S2"), ("D1", "A"), ("S5", "S2_ESB"), ("S5", "A_ESB"), ("S5", "D1"), ("A_CBAM", "A"), ("A_ESB", "A_CBAM"), ("S2_CBAM", "S2"), ("A_CBAM", "S2_CBAM"), ("S2_ESB", "S2_CBAM")):
        cols = []
        for ck in ("best", "last"):
            diffs = []
            for s in seeds:
                x, y = metric(scores, left, s, ck, "A"), metric(scores, right, s, ck, "A")
                if x is not None and y is not None:
                    diffs.append(x - y)
            if diffs:
                wins = sum(d > 0 for d in diffs)
                per = ", ".join(f"{d:+.2f}" for d in diffs)
                cols.append(f"{statistics.fmean(diffs):+.2f}（{wins}/{len(diffs)} 胜；{per}）")
            else:
                cols.append("—")
        short = {"A": "单 A", "B": "单 B", "S2": "S2", "A_ESB": "A+ESB", "S2_ESB": "S2+ESB", "D1": "S4", "S5": "S5", "A_CBAM": "A+CBAM", "S2_CBAM": "S2+CBAM"}
        lines.append(f"| {short[left]} − {short[right]} | " + " | ".join(cols) + " |")
    return lines


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("queue", "compare"))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--experiments", default=",".join(ACTIVE_EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "compare":
        write_comparison()
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    selected = [e.strip() for e in args.experiments.split(",")]
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_manifest(seeds, args)
    queue = [(e, s) for s in seeds for e in selected]  # seed-major: paired results arrive earliest
    for index, (experiment, seed) in enumerate(queue, 1):
        write_status(status="training", current=f"{experiment} seed={seed}", position=f"{index}/{len(queue)}", pid=os.getpid())
        train_one(experiment, seed, args)
        write_status(status="evaluating", current=f"{experiment} seed={seed}", position=f"{index}/{len(queue)}", pid=os.getpid())
        evaluate_one(experiment, seed, args)
        write_comparison()
    write_status(status="complete", position=f"{len(queue)}/{len(queue)}", pid=os.getpid())


if __name__ == "__main__":
    main()
