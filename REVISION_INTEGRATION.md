# Revision integration and verification

The five new S1+CBAM runs and evaluation were completed on 17 November 2026.
This review attachment preserves the frozen original protocol/scripts alongside
portable wrappers; it does not modify the already completed experiment.

Executed checks:

- Original package audit: 306 original copied fingerprints, 40 complete logs,
  80 original checkpoint fingerprints, fixed 744/160/161 partition and 750 groups.
- New offline replay: 34 paired contrasts, 20 locked-screening rows and 1608
  common-object observations; maximum absolute numerical error 3.55e-15.
- Portable supplement generation; five-page TeX/PDF compiled twice, rendered
  and visually inspected with no overflow or compiler warnings.
- Actual fixed-800 CUDA benchmark using archived 32-photo selection and
  seed-42 A/A+CBAM/S2+CBAM weights. The portable benchmark was plan-checked;
  the preserved original implementation was the one actually executed.
- All 13 bundled weights copied byte-for-byte and SHA256 checked.

Previously completed original-package checks remain documented in STAGE_REPORT.
They include cached-result replay, original tables/figures, and CPU model and
initialization checks. Optional full data preparation and full fresh training
from a clean installation have not been rerun end to end. Native AP in the new
replay is preserved evaluator output, not recomputed from rounded table values.

The new timing data are a separate desktop-GPU processing benchmark and do not
replace the original 768-forward measurements. There is no new external/healthy
image predictive test or embedded/UAV hardware validation. Resource-audit files
are evidence of source inspection only.

The ZIP excludes derived outputs/figures images and cache/config noise; all
figure scripts, numerical inputs, selection records and QA are retained. No WTBD
photos or raw source DTU annotation files are redistributed. Original local
absolute paths in preserved provenance are descriptive records. Portable entry
points use package-relative paths or explicit input arguments.
