# Environment provenance

The original experiment manifest independently records Python 3.12.13,
torch 2.3.1+cu121, Ultralytics 8.4.37 and an RTX4060 Laptop GPU.
It is preserved at `../provenance/original_environment_manifest.json`.

The original `CV_uav` environment still existed on 17 November 2026. Its package
metadata was read to create `historical-environment.json` and
`historical-environment-packages.txt`. Versions other than the independently
recorded core versions describe the **extant environment at that date**, not a
claim that every dependency was frozen before the original experiments.

`requirements-offline.txt` pins the numerical packages used for cached replay.
`requirements-model.txt` adds the measured model dependencies and the PyTorch
CUDA12.1 wheel index. The full environment inventory includes unrelated packages;
it is an audit record rather than a minimal installation requirement. If a clean
installation is built, use a separate environment and keep its successful lock
file; clean installation has not yet been tested by this packaging task.

Acceptance was run with the extant historical core environment, using CPU only:
35 operating points on both subsets, Tables 5–14 export, and finite forward checks
for the three architectures. `../outputs/model_cpu_check.json` records the actual
versions used. No GPU was used and no training was started by these checks.
