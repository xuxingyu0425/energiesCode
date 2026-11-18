# Wind turbine blade defect detection and localization

**Author** 许星宇 (Xingyu Xu)

This source package contains the dual-target training supervision (S2), the coordinate-averaged control (S1), and the YOLOv8n integrations for CBAM and ESB. It includes the exact WTBD split, model configurations, training and evaluation entry points, and the numerical inputs for reproducing aggregate tables and plots.

Download and extract [Energies_Reproducibility_Code.zip](Energies_Reproducibility_Code.zip). Run the commands below from the extracted package root.

## Data and environment

- Obtain the WTBD images and two XML annotation sets from [Figshare](https://doi.org/10.6084/m9.figshare.30210175). See the [dataset paper](https://doi.org/10.1038/s41597-026-06762-x). The supplied partition assigns 744 images to training, 160 to validation and 161 to testing, with visually near-duplicate groups kept intact.
- The model environment used Python 3.12.13, PyTorch 2.3.1 with CUDA 12.1, and Ultralytics 8.4.37. Obtain the YOLOv8n initialization checkpoint from Ultralytics. Dependencies and data retain their upstream terms.

~~~sh
python -m pip install -r environment/requirements-model.txt
python tools/prepare_data.py --source /path/to/extracted/WTBD --output work/wtbd
~~~

The source directory contains `JPEGImages/`, `Annotations/`, and `annotation_second_person/`. Use a new or empty output directory. The preparation script checks its output against `data/split_manifest.csv`.

## Train and evaluate

~~~sh
python tools/train_one.py --cell S2_CBAM --seed 42 --data-dir work/wtbd --pretrained /path/to/yolov8n.pt --output work/S2_CBAM_42
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

## Tables, figures and timing

~~~sh
python tools/reproduce_tables.py --annotation-audit work/wtbd/dual_annotation_audit.csv
python tools/reproduce_revised_figures.py --annotation-audit work/wtbd/dual_annotation_audit.csv
~~~

The commands export numerical tables and the dataset, localization and screening plots from the supplied per-seed results and locally regenerated annotation audit. The editable workflow schematic is in `figures_source/`. For optional photo examples, pass `--image-root` and supply locally retained per-instance matches and cached predictions at the paths used in `figures_original.py` and `localization.py`.

The `tools/benchmark_800.py` timing script uses the 32 images listed in `benchmark_800/protocol.json` and locally supplied epoch-200 checkpoints at `weights/revision/{A,A_CBAM,S2_CBAM}/seed_42/last.pt`. Use `--image-root /path/to/WTBD/JPEGImages --execute` to run the GPU benchmark.

## Source notices

The package includes the authors’ source code, derived aggregate results, and the split manifest. Obtain the original images and XML annotations from the dataset deposit. Generate per-image prediction caches and model checkpoints locally. The licenses and commit references for vendored CBAM, coordinate attention and efficient channel attention are in `src/wtbd_attention_vendor/`.
