# DTU inspection-image false alarms

The evaluation used 11 uncropped images of one Nordtank turbine from DTU version 2.
`frozen_sample.csv` records the original file IDs, source URLs, and SHA-256 hashes.
Mendeley Data lists version 2 under CC BY-NC 3.0. [DTU version 2](https://doi.org/10.17632/hd96prn3nc.2)

Figure 7 credits the photographs published in DTU version 1 under CC BY 4.0.
The version 1 SHA-256 hashes match the evaluation originals for all 11 frames.
`figure7_v1_sources.csv` records the version 1 file ID, download URL, and
matching hash for each panel. The figure presents the resized photographs in panels.
[DTU version 1](https://doi.org/10.17632/hd96prn3nc.1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Related DTU study](https://doi.org/10.3390/en12040676)

Candidates were identified by year and frame order after removing originals matched
to a published defect annotation. Of the 120 reviewed candidates, 70 had ambiguous
surface features, 23 had insufficient blade visibility, and 16 repeated an accepted
view. The remaining 11 frames were fixed before any detector ran on DTU.
One author reviewed all 11 retained frames and confirmed that no defects were visible on the assessable blade surfaces.
[Public DTU defect annotations](https://doi.org/10.3390/machines11100953)

per_image.csv gives one row per frame and model run. per_seed.csv reports the
weight hash, original WTBD A-validation cutoff, and the corresponding run totals.
All 15 runs used epoch-200 checkpoints. Cell A is YOLOv8n trained on annotation set A. The full frames (5280 by 2970 pixels) used
the established rectangular preprocessing with imgsz 768 and NMS IoU 0.70.
The actual tensor dimensions were 448 by 800. A frame with any detection remaining
at the cutoff counts once toward the image-level false-alarm rate. The count of all
retained boxes divided by 11 gives false-positive boxes per image. Means average
the five run-level rates or counts per image for each configuration.

Across 55 frame–seed evaluations per detector, A, S1_CBAM, and S2_CBAM had 5, 6, and 5 frames with false alarms and 7, 9, and 6 retained false-positive boxes, respectively. The rerun reproduced both CBAM configurations on every frame and seed.
