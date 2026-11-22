# DTU inspection-image false alarms

The manuscript's Figure 7 shows the 11 full frames listed in frozen_sample.csv.
The identifiers, source URLs, and SHA-256 hashes specify the exact originals.
The files were acquired on one Nordtank turbine in 2017 and 2018 and are
available from the DTU dataset, version 2, under CC BY-NC 3.0.
[DTU inspection images](https://doi.org/10.17632/hd96prn3nc.2)

Candidates were identified by year and frame order after removing originals matched
to a published defect annotation. Of the 120 reviewed candidates, 70 had ambiguous
surface features, 23 had insufficient blade visibility, and 16 repeated an accepted
view. The remaining 11 frames were fixed before any detector ran on DTU.
The visual screen was assisted by OpenAI Codex. One author manually reviewed all 11 retained frames and confirmed that no defects were visible on the assessable blade surfaces.
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
