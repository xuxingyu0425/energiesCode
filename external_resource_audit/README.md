# External dual-annotation resource availability audit

This is a resource availability audit, not an external model-performance test.
The dated search log records 18 queries and seven inspected candidates, including
WTBD as the existing-data reference. It is not an exhaustive systematic review.

`fixed_source.json` identifies the audited DTU repository commit. The download
manifest records source URLs and SHA256 values for the 11 small annotation/README
inputs. Those downloaded source annotation files are **not redistributed** here
because the annotation repository has no explicit license. To repeat the audit,
obtain the inputs from the original author repository under its terms, place
them at the relative locations in `DTU_download_manifest.json` (under
`DTU_annotations/`), and run `python audit_dtu_annotations.py` in this directory.
The underlying DTU image dataset has separate CC BY-NC 3.0 terms; no original
DTU images were downloaded or compared with WTBD in this audit.

The diagnostic comparison pairs boxes on the same file and class using maximum
sum-IoU Hungarian matching, retains IoU>=0.5, and checks coordinate equality to
1e-4 pixel tolerance. This does not establish independent annotators. The paired
results/inventory CSVs and audit JSON are the outputs of this script. Almost all
retained pairs have identical coordinates; targeted splitting/addition/deletion
is not equivalent to two retained independent boundary judgements.

The search did not identify an additional verified, immediately usable
per-annotator dual-box blade benchmark. This is not a claim that none exists.
External evaluation of a frozen detector with one reliable reference set remains
possible and was not performed here.
