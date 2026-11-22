# Wind turbine blade defect detection and localization

**Author** 许星宇 (Xingyu Xu)

This repository contains the dual-target training supervision (S2), the coordinate-averaged control (S1), and the YOLOv8n integrations for CBAM and ESB. It includes the exact WTBD split, model configurations, training and evaluation entry points, and the numerical inputs for reproducing aggregate tables and plots.

Clone this repository or use GitHub’s Download ZIP option. Run the commands below from the repository root.

## What is available

| Path | Contents |
| --- | --- |
| `src/`, `configs/`, `tools/` | Training, evaluation, threshold selection, and reproduction scripts |
| `data/split_manifest.csv` | Fixed WTBD split and perceptual-hash groups |
| `historical/`, `revision_s1_cbam/` | Training arguments and epoch logs for all 45 primary runs, with archived per-seed results |
| `benchmarks/` | Matched class-precedence and detector comparisons, per-seed scores, and training records for YOLOv8s and YOLO11n |
| `external_dtu_false_alarms/` | IDs and checksums of the 11 evaluated frames, and all 15 per-seed and 165 per-frame records |
| `outputs/main_text_tables/`, `figures_source/` | CSV snapshots for Tables 1–21 and the editable experiment workflow |
| `benchmark_800/`, `weights/included_weights.csv` | Timing records and the available checkpoint inventory |

The archived training arguments record the paths used during the experiments. Use the commands below with local data paths when replaying the work. The main-text CSV snapshots follow the current manuscript numbering. Some historical scripts and reports use the numbering of the earlier manuscript.

## Data and environment

- Obtain the WTBD images and two XML annotation sets from [Figshare](https://doi.org/10.6084/m9.figshare.30210175). See the [dataset paper](https://doi.org/10.1038/s41597-026-06762-x). The supplied partition assigns 744 images to training, 160 to validation and 161 to testing, with visually near-duplicate groups kept intact.
- The model environment used Python 3.12.13, PyTorch 2.3.1 with CUDA 12.1, and Ultralytics 8.4.37. The YOLOv8n initialization checkpoint used for the study is included under `weights/`. Dependencies and data retain their upstream terms.

~~~sh
python -m pip install -r environment/requirements-model.txt
python tools/prepare_data.py --source /path/to/extracted/WTBD --output work/wtbd
~~~

The source directory contains `JPEGImages/`, `Annotations/`, and `annotation_second_person/`. Use a new or empty output directory. The preparation script checks its output against `data/split_manifest.csv`.

## Train and evaluate

~~~sh
python tools/train_one.py --cell S2_CBAM --seed 42 --data-dir work/wtbd --pretrained weights/yolov8n.pt --output work/S2_CBAM_42
~~~

This command prints the training plan. Add `--execute` to train for 200 epochs. The nine supported configurations are `A`, `B`, `S1`, `S2`, `A_ESB`, `S2_ESB`, `A_CBAM`, `S1_CBAM`, and `S2_CBAM`. Use seeds 42–46. The S1+CBAM wrapper checks its initial model state and first augmented batch against fingerprints from paired reference runs.

Use the same epoch-200 checkpoint and original A annotations for validation and test:

~~~sh
python tools/evaluate_checkpoint.py --weights work/S2_CBAM_42/weights/last.pt --data work/wtbd/wtbd_gt_primary.yaml --split val --architecture cbam --cell S2_CBAM --seed 42 --output work/val.json --device cpu
python tools/select_threshold.py --validation-cache work/val.json --output work/selection.json
python tools/evaluate_checkpoint.py --weights work/S2_CBAM_42/weights/last.pt --data work/wtbd/wtbd_gt_primary.yaml --split test --architecture cbam --cell S2_CBAM --seed 42 --output work/test.json --device cpu
python tools/score_test.py --test-cache work/test.json --selection work/selection.json --output work/test_scores.json
~~~

Training validation uses fused annotations for native checkpoint selection. For thresholded screening, maximize pooled F1 on original A-validation annotations over the 0.01–0.99 confidence grid, with ties favoring the lower threshold. Apply that threshold to the test predictions from the same checkpoint.

## Offline and external checks

~~~sh
python -m pip install -r environment/requirements-offline.txt
python tools/replay_locked.py
~~~

The offline replay checks the stored WTBD validation and test predictions against the locked screening results. `revision_s1_cbam/` contains the separate matched-control replay materials. The class-precedence and detector results in `benchmarks/` use the common grouped WTBD split and epoch-200 checkpoints for seeds 42–46. AP values are in percent and paired AP differences are in percentage points.

The [DTU inspection frames](https://doi.org/10.17632/hd96prn3nc.2) are available from the original deposit. The identifiers and checksums in `external_dtu_false_alarms/frozen_sample.csv` identify the 11 source frames. The archived A, S1+CBAM, and S2+CBAM evaluations contain 5, 6, and 5 frame–seed cases with false alarms and 7, 9, and 6 false-positive boxes, respectively. Each detector was evaluated on the same 11 frames for five seeds with its WTBD A-validation threshold.

Figure 7 credits the photos in DTU version 1 under CC BY 4.0. The SHA-256 hashes match the 11 version 2 evaluation frames. [Figure 7 photo sources](external_dtu_false_alarms/figure7_v1_sources.csv).

## Tables, figures and timing

~~~sh
python tools/reproduce_tables.py --annotation-audit work/wtbd/dual_annotation_audit.csv
python tools/reproduce_revised_figures.py --annotation-audit work/wtbd/dual_annotation_audit.csv
~~~

The commands export numerical tables and the dataset, localization and screening plots from the supplied per-seed results and locally regenerated annotation audit. The editable workflow schematic is in `figures_source/`. For optional photo examples, pass `--image-root` and supply locally retained per-instance matches and cached predictions at the paths used in `figures_original.py` and `localization.py`.

The `tools/benchmark_800.py` timing script uses the 32 images listed in `benchmark_800/protocol.json` and locally supplied epoch-200 checkpoints at `weights/revision/{A,A_CBAM,S2_CBAM}/seed_42/last.pt`. Use `--image-root /path/to/WTBD/JPEGImages --execute` to run the GPU benchmark.

## Source notices

The repository includes the authors’ source code, experiment records, prediction caches, and selected checkpoints. Obtain the original images and XML annotations from the dataset deposit. The archived checkpoint inventory lists the available weights and their hashes. The licenses and commit references for vendored CBAM, coordinate attention and efficient channel attention are in `src/wtbd_attention_vendor/`.
