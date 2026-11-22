# Reproducibility materials for WTBD and the reviewer-requested control

This author-review package was assembled on 17 November 2026 for the Energies
revision. It contains preserved implementation files, the exact original image
partition, configurations for all 40 original training runs, weight fingerprints,
and cached predictions supporting the original manuscript. The five new S1+CBAM runs and their completed evaluation are stored separately
in `revision_s1_cbam/`; they are not silently mixed into the original results.

## Quick start: no GPU or model weights required

From this directory, with Python and the packages in
`environment/requirements-offline.txt`:

```sh
python tools/replay_locked.py
python tools/reproduce_tables.py
```

The first command independently rematches all predictions at the selected
threshold for 35 runs on validation and test images (70 evaluations). It verifies
the exact partition, the 99-point validation grid, the lower-threshold tie rule,
weight hashes, and the archived P/R/F1 and TP/FP/FN counts. The second exports
numerical versions of original Tables 1–14, including dataset metadata and five-seed sample standard
deviations and paired Student t intervals (n=5, df=4, unadjusted).

```sh
python tools/reproduce_revised_figures.py
python tools/reproduce_revised_figures.py --image-root /path/to/WTBD/JPEGImages
```

The first figure command reproduces original result Figures 2, 3 and 5. Supplying
the source-photo directory also recreates the original Figure 4 example selection,
matching and crops. Photos are obtained from the original dataset rather than
redistributed in this package. Numerical rules are preserved; renderer/font
differences can change the exported pixels. These commands produce files only in
`outputs/`, and do not perform training or model inference. Figure 1 is an authored
architecture schematic; its editable PowerPoint is supplied in the LaTeX
source package. Run `figures_source/make_experimental_workflow.py` to export
the experimental workflow (Figure 2).

With the model environment, an additional CPU-only check is available:

```sh
python tools/model_cpu_check.py
```

This sets `CUDA_VISIBLE_DEVICES` to an empty string before importing torch. It
checks baseline, ESB and CBAM construction, parameter counts and finite 64×64 CPU
forward outputs. It is not a training, speed, accuracy, or deployment benchmark.

The outputs can be checked against the submitted baseline TeX:

```sh
python tools/check_manuscript_numbers.py --manuscript /path/to/original/manuscript.tex
python tools/audit_package.py
```

The numerical comparison covers all 841 displayed numbers in original Tables
2–14 (101 rows). Table 1 is a supervision-definition table, rather than a
numerical result. The checked original manuscript is not edited by this command.

## Data source and exact partition

WTBD is described by Ji et al., *Scientific Data* (2026),
https://doi.org/10.1038/s41597-026-06762-x.
The original images and two XML annotation sets are available at
https://doi.org/10.6084/m9.figshare.30210175.

`data/split_manifest.csv` lists all 1,065 image names, their training/validation/test
assignment, pHash group and pHash bits. The fixed partition is 744/160/161 images.
`data/dual_annotation_audit.csv` records the annotation pairing and original A/B
coordinates. A is the primary reporting reference and B is the sensitivity
reference. The dataset's images/annotations are not claimed as newly collected
by this study, and their upstream terms continue to apply.

To regenerate the historical label directories in a **new, empty directory**:

```sh
python tools/prepare_data.py --source /path/to/extracted/WTBD --output /path/to/work/data
```

The preserved preparation algorithm is called with seed 42 and pHash Hamming
threshold 8. The wrapper refuses to accept a resulting image partition different
from the archived manifest. This command was supplied but has not yet been
run as part of this package's acceptance tests, because it creates the full image
layout. Source annotation parsing, label generation and data hashing should also
be reviewed when preparing the final public release.

Historical `primary`, `second` and `fused` training roots use the historical
consensus labels for their native validation sets. Separate `gt_primary` and
`gt_second` roots retain original A and B annotations for reporting. The epoch-200
screening thresholds are subsequently selected on the common original-A
validation set. These stages must not be conflated or silently changed while
reconstructing the original best-checkpoint results.

## Training code and original run records

`src/wtbd_dual.py` is the historical frozen S2 implementation (SHA256
`ec8476eaef4a0f3b558fc461467aab43798514f4619745ad562ff07289b71155`),
not the later working-copy implementation containing additional soft-class
experiments. Original A/B target coordinates are carried through the same image
augmentations and used by the dual-target regression loss. S1 instead regresses
the coordinate-averaged boxes using ordinary regression.

`src/wtbd_modules.py` implements and registers ESB; `src/wtbd_attention.py`
integrates the canonical attention implementation in `src/wtbd_attention_vendor/`.
The upstream CBAM/ECA/coordinate-attention source notices and commit provenance
are retained. The three paper architecture configurations are in `configs/`.

The original 40 resolved argument files are preserved **without modification** in
`historical/run_arguments/`: A, B, S1, S2, A+ESB, S2+ESB, A+CBAM, and S2+CBAM,
each for seeds 42–46. Corresponding epoch logs are in `historical/epoch_logs/`.
Their original absolute paths are provenance, not current-machine requirements.

The convenience training entry below rebases paths and uses the preserved
builders/hyperparameters. It prints a plan by default:

```sh
python tools/train_one.py --cell S2_CBAM --seed 42 --data-dir /path/to/work/data --pretrained /path/to/yolov8n.pt --output /path/to/new/run
```

Adding `--execute` explicitly starts 200-epoch training. Existing run directories
are never overwritten. Execution requires the recorded torch 2.3.1 and
Ultralytics 8.4.37 environment; no silent substitution of a different API is made.
Retraining this wrapper has not been validated end to end as part of the CPU-only
package checks and is not promised to be bitwise identical to historical runs.
The exact original controller sources are also retained in `src/` for audit;
their old `main()` paths are not the portable entry points.

The new S1+CBAM revision control is maintained separately by the revision runner.
It uses the historical paired-augmentation data path while retaining the native
single-target detection loss; this is needed to isolate the regression rule from
augmentation differences. Its five 200-epoch runs and post-training evaluation completed on 17 November
2026. Full initial-model and first augmented-batch hashes match the historical
S2+CBAM references for all five seeds. Completed records and analysis are linked
in `revision_s1_cbam/`; see the revision replay and weight sections below. Its unmodified runner, frozen
protocol and preflight snapshot are in `provenance/revision_s1_cbam/`.
`src/revision_s1.py` exposes the same paired-pipeline/native-criterion distinction,
and `tools/train_one.py --cell S1_CBAM` provides an optional path-rebased entry
that checks full initial-state and first-batch tensor hashes. This portable
training entry was dry-run/CPU-construction checked, not used to launch the
active revision training. Do not substitute a freshly
constructed ordinary-dataset run and call it the audited new control.

## Weights, evaluation and threshold selection

`provenance/weight_manifest.csv` covers all 80 original best/last checkpoints,
including B-only, with SHA256 hashes, historical relative run locations, argument
hashes and evaluation shape. All 80 original files were found and hashed locally.
Thirteen checkpoint/initialization files are included for the completed core
control and timing comparisons; see `weights/included_weights.csv` and the
included-weight section below. The remaining original checkpoints have verified
hashes and training records but are not bundled.

For a supplied original checkpoint, after preparing the data:

```sh
python tools/evaluate_checkpoint.py --weights /path/to/last.pt --data /path/to/work/data/wtbd_gt_primary.yaml --split val --architecture cbam --cell S2_CBAM --seed 42 --output /path/to/new/val.json --device cpu
python tools/select_threshold.py --validation-cache /path/to/new/val.json --output /path/to/new/selection.json
```

Generate the test cache separately with `--split test`, keeping the frozen
threshold. Threshold selection refuses a cache labelled as test. Native
validation uses configured `imgsz=768`, batch 16, confidence floor 0.001, NMS IoU
0.7, max_det 300 and FP32. The historical evaluator pads these square WTBD inputs
to actual 800×800 tensors; raw prediction and ground-truth coordinates in the
included caches are on that padded canvas. Native AP matching and custom
confidence-descending screening matching serve different purposes and are not
interchanged. The separate cost benchmark used 768×768 FP32 forward tensors,
batch 1, 200 warmups and three rounds of 1,000 measured passes.

Checkpoint inference and full retraining require the original images and
the appropriate weights (the core-control weights are included). The inference entry has not been run in this CPU-only packaging task;
the cached predictions, chosen operating points and all reproduced table inputs
are independently checked offline. This package does not imply independent-data,
normal-blade, embedded-device, or end-to-end UAV validation.

## Result mapping and units

| Original item | Archived input | Portable numerical output |
|---|---|---|
| Table 5 | `yolov8_reeval_table.csv` | `table05_detection.csv` |
| Table 6 | same, paired by seed | `table06_detection_differences.csv` |
| Table 7 | `localization_v12.csv`, `common_tp_v12.csv` | `table07_localization.csv` |
| Table 8 | native AP differences S2−A and S2−S1 | `table08_supervision.csv` |
| Table 9 | `cbam_increment_per_seed_v16.csv` | `table09_cbam_increment.csv` |
| Tables 10–11 | `locked_per_seed.csv` | `table10_locked_screening.csv`, `table11_screening_differences.csv` |
| Table 12 | `screening_per_seed.csv` | `table12_fixed_thresholds.csv` |
| Table 13 | `annotation_sensitivity.csv` | `table13_annotation_sensitivity.csv` |
| Table 14 | `model_complexity.csv`, `latency_unified.json` | `table14_model_cost.csv` |
| Figure 2 | `fig2a_aspect_ratio.csv` | `fig2_dataset.*` |
| Figure 3 | AP/IoU per seed and paired intervals | `fig3_localization.*` |
| Figure 4 | `localization_instances_v12.csv`, cached predictions, WTBD photos | `fig4_localization_examples.*` |
| Figure 5 | `locked_per_seed.csv`, `locked_per_class.csv` | `fig5_application.*` |

All files in the second column are under `data/`; outputs are under `outputs/`.
AP/P/R/F1 are percentages, their differences are percentage points, IoU is
unitless, TP/FP/FN are counts, latency is milliseconds, and parameter counts are
integers. The statistical replicate is the training seed, not the individual
detected object. Small precision differences do not establish significance when
their paired intervals include zero.

## Provenance and release status

`provenance/source_files.json` traces every copied source/data/config/log to its
workspace origin and SHA256. `figure_path_adaptations.json` identifies the figure
scripts whose paths were rebased, without changing their calculations.
`missing_materials.json` records remaining gaps. `outputs/` contains acceptance
results, not newly trained models. A publication repository URL/DOI and an overall
author-code release licence have not been invented; they require the actual
release decision. Third-party notices are preserved separately.

See `environment/README.md` for the distinction between recorded historical
versions and the extant-environment inventory.

## Hardware metadata

The authors confirmed that the original experiments used the same computer now
queried for hardware metadata: Intel Core i9-13980HX (24 cores/32 threads), about
32 GB installed RAM, and NVIDIA GeForce RTX4060 Laptop GPU (about 8 GB VRAM).
The current system reports Windows 11 Home Chinese 64-bit. The currently observed
Windows build is not asserted to be the historical build. See
`provenance/hardware_environment.json` for the read-only query record and scope.


## Completed S1+CBAM revision control and portable offline replay

The five new S1+CBAM runs (seeds 42–46, 200 epochs) use the same paired dataset
and augmentation path as S2+CBAM. Only the regression criterion changes to native
regression to the fused box. Initial states, training arguments (except output
paths), and each first augmented batch were checked against the historical
references. The original extension scripts/protocol are preserved unchanged in
`revision_s1_cbam/original_extension/`; these provenance scripts retain original
workspace paths and are not the portable entry points.

```sh
python tools/replay_revision.py
python tools/reproduce_revised_tables.py
```

`replay_revision.py` requires no images, torch, model weights, GPU, or inference.
It rematches localization and the 20 A/B screening rows from archived predictions,
reselects all validation thresholds from the full 99-point grid, and recomputes
all 34 paired differences/intervals. All results match the frozen extension
analysis within the numerical tolerance of 1e-10. Native AP values are archived
native-evaluator outputs, not recomputed AP from rounded summary tables.
`reproduce_revised_tables.py` replays the original numerical analyses and new
paired comparisons, then exports the 19 tables embedded in the revised
`data/manuscript_revised.tex`. CSVs retain the exact rounded presentation cells
and the output mapping records final main-text numbering. No prose supplement
is required to read the revised article.

`revision_mechanism/` contains the historical gradient audit and XML-verified
annotation/object mapping. Gradient differences are not interpreted as fractions
of AP gain. The new S2-minus-S1 object-level strata do not establish larger gains
on more discrepant annotations; all favorable and unfavorable strata are retained.

## Included weights and new GPU timing benchmark

`weights/included_weights.csv` identifies 13 files by SHA256: all five S1+CBAM and
five S2+CBAM epoch-200 EMA checkpoints, two seed-42 A/A+CBAM checkpoints used in
timing, and the identical COCO-pretrained initialization file. The other original
model checkpoints remain identified by the original 80-entry manifest; they are
not all bundled. Model weights were copied without modifying the source files.
For model inference, add `src/` to the Python path or use the supplied evaluator
so the preserved CBAM and dual-model classes can be imported.

`benchmark_800/` preserves the actual fixed-800 experiment: raw synchronized
forward timings, image-pipeline samples, the prior-frozen protocol, and summary.
This is a single-seed desktop GPU benchmark, not a UAV/onboard experiment.
The original 768-pixel results remain separate.

```sh
python tools/benchmark_800.py
python tools/benchmark_800.py --image-root /path/to/WTBD/JPEGImages --execute
```

The first command prints a plan only; the second performs the GPU benchmark in
the pinned environment using the included weights and 32 source images. Images
are preloaded, resized to 768 and padded to a true 800 tensor; no raw photographs
are bundled. All images are checked against their archived hashes. The portable
benchmark entry was dry-run checked; the original benchmark implementation was
actually executed in the recorded environment. Rerun latency will vary with
hardware state and system load.

The review-code ZIP omits derived result-figure image files under
`outputs/figures/` to limit upload size. Their numerical inputs, selection records,
plotting scripts, and QA reports are retained; use the figure reproduction entry
above to regenerate them. Upstream source/data notices and vendor licenses are
retained. No new blanket license or public repository DOI is asserted here.

## External resource availability audit

`external_resource_audit/` contains the dated 18-query log, seven-candidate
inspection table, original author URLs and fixed DTU commit, download checksums,
our pairing script, and diagnostic outputs. The source DTU annotation files and
images are not redistributed. Retrieve the original inputs under their terms
before rerunning that optional audit. This is a resource-availability audit,
not an independent-dataset or defect-free-image detector-performance test.

```sh
python tools/audit_revision_package.py
```

The revision checksum index verifies preserved files outside `outputs/`; the
outputs directory contains regenerated acceptance artifacts and can be recreated.
See `REVISION_INTEGRATION.md` for exactly which checks were executed.

## Revised main-text numbering

The article has 19 main-text tables and six figures. There is no prose
supplement. The source-code archive is provided for review and reproducibility.

`python tools/reproduce_revised_tables.py` replays the original numerical
analyses and the five-seed extension, and exports the final main-text table
cells with a label-to-number mapping. Native AP remains the archived native
evaluator result; localization, locked thresholds, screening and paired
intervals are recalculated from cached predictions. No inference is performed
by these replay commands.

For the old plotting entry points, original Figures 2, 3, 4 and 5 correspond
to revised Figures 3, 4, 5 and 6. The author-edited Figure 1 PowerPoint is
supplied as `figure_sources/fig1_author_enlarged.pptx` in the LaTeX source
package. Figure 2 has an editable SVG and deterministic PNG/PDF export code
in `figures_source/`. The source-photo examples require the upstream WTBD images.

## Validation references

Training annotations differ across A, B, S1 and S2, while native checkpoint
selection uses the same fused validation annotations. In src/prepare_wtbd.py,
training subsets receive the selected training labels and validation subsets
receive consensus labels. Original A/B reference annotations are retained
separately for evaluation. Operating thresholds use epoch-200 weights and
original A-validation labels for eight configurations (40 model-seed operating
points; B-only is excluded). The older checkpoint_caveat description in
src/wtbd_control_study.py is superseded by this account of the actual data
preparation and validation-cache records; the frozen source is preserved.

## DTU inspection-image false-alarm evaluation

The external_dtu_false_alarms directory contains the locked frame list and the
per-image and per-seed detection counts for Figure 7 and the DTU results.
Its README gives the data DOI, the source image hashes and the metric definitions.
