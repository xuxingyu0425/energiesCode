"""CPU-only revision analysis of existing annotation/gradient/localization records.

Does not load torch, run inference, train, tune thresholds, or change old data.
The current localization contrast is S2+CBAM minus A+CBAM, not S2 minus S1.
"""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
LEGACY = WORKSPACE / 'experiments/reports/mechanism_diagnostics_v1'
PAPER = WORKSPACE / 'papers/mdpi_wind_20261019_v25_en'
DATASET = WORKSPACE / 'WT blade defect dataset/WT blade defect dataset'
SEEDS = list(range(42, 47))
CLASS_NAMES = ['craze', 'crack', 'corrosion', 'hide_craze', 'surface_injure', 'thunderstrike']
SOURCES = {}
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261117


def source(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_csv(path):
    return pd.read_csv(source(path))


def read_json(path):
    return json.loads(source(path).read_text(encoding='utf-8'))


def paired_ci(values):
    values = np.asarray(values, dtype=float)
    assert len(values) == 5 and np.isfinite(values).all()
    mean, sd = values.mean(), values.std(ddof=1)
    half = t.ppf(0.975, 4) * sd / np.sqrt(5)
    return dict(mean=mean, seed_sd=sd, seed_ci_low=mean-half,
                seed_ci_high=mean+half, positive_seeds=int((values > 0).sum()),
                seed_n=5, seed_df=4)


def match(record):
    """Historical confidence-ordered, same-class one-to-one IoU>=.5 matching."""
    gt = np.asarray(record['gt_boxes'], dtype=float).reshape(-1, 4)
    pred = np.asarray(record['pred_boxes'], dtype=float).reshape(-1, 4)
    gc, pc, cf = (np.asarray(record[k]) for k in ['gt_cls', 'pred_cls', 'conf'])
    lower = np.maximum(gt[:, None, :2], pred[None, :, :2])
    upper = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    intersection = np.maximum(upper-lower, 0).prod(axis=2)
    ga = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(axis=1)
    pa = np.maximum(pred[:, 2:]-pred[:, :2], 0).prod(axis=1)
    overlap = intersection / np.maximum(ga[:, None]+pa[None, :]-intersection, 1e-12)
    indices = np.flatnonzero(cf >= .25)
    indices = indices[np.argsort(-cf[indices], kind='stable')]
    matched = np.full(len(gt), np.nan)
    for j in indices:
        candidates = np.flatnonzero((gc == pc[j]) & np.isnan(matched) & (overlap[:, j] >= .5))
        if len(candidates):
            k = candidates[np.argmax(overlap[candidates, j])]
            matched[k] = overlap[k, j]
    return matched


def main():
    source(Path(__file__))
    out = ROOT / 'results'
    archive = ROOT / 'historical_records'
    out.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    annotations = read_csv(LEGACY/'annotation_objects.csv')
    assert len(annotations) == 1584
    assert not annotations.duplicated(['image', 'object_index']).any()
    annotations['xml_gt_index'] = -1

    # Verify the original A XML object sequence and box/class values before joining
    # cached-prediction gt_index to annotation object_index. XML filename metadata
    # can be stale: use the actual XML basename matched to the image basename.
    xml = {}
    for image, a in annotations.groupby('image', sort=False):
        p = source(DATASET/'Annotations'/Path(image).with_suffix('.xml'))
        root = ET.parse(p).getroot()
        objs = root.findall('object')
        a = a.sort_values('object_index')
        assert a.object_index.tolist() == list(range(len(objs))), image
        width = int(root.findtext('size/width'))
        height = int(root.findtext('size/height'))
        boxes, classes = [], []
        for index, obj in enumerate(objs):
            cls = obj.findtext('name').strip()
            box = np.array([float(obj.findtext('bndbox/'+key)) for key in ['xmin', 'ymin', 'xmax', 'ymax']])
            # The audit uses paired-first/fallback-last order. It is NOT the
            # original A XML order when A-only objects occur between paired boxes.
            # Map by unique A coordinates AND class; never join the two indices.
            candidates = [j for j, row in a.iterrows()
                          if cls == row.class_a and np.array_equal(box, np.fromstring(row.box_a, sep=' '))]
            assert len(candidates) == 1, (image, index, candidates)
            annotations.loc[candidates[0], 'xml_gt_index'] = index
            boxes.append(box)
            classes.append(CLASS_NAMES.index(cls))
        xml[image] = (np.array(boxes), classes, width, height)
    assert (annotations.xml_gt_index >= 0).all()
    assert not annotations.duplicated(['image', 'xml_gt_index']).any()
    test = annotations[annotations.split.eq('test')].copy()
    assert len(test) == 233 and test.image.nunique() == 161

    common = read_csv(PAPER/'source_data/cbam_common_objects_v16.csv')
    assert len(common) == 1612
    assert not common.duplicated(['checkpoint', 'seed', 'image', 'gt_index']).any()
    rebuilt = []
    max_gt_error = 0.0
    for checkpoint in ['last', 'best']:
        for seed in SEEDS:
            records = {}
            for cell in ['A_CBAM', 'S2_CBAM']:
                p = (PAPER/f'source_data/locked/{cell}_{seed}_test.json' if checkpoint == 'last'
                     else PAPER/f'reproducibility/predictions/v8_{cell}_{seed}_best_annotator_a.json')
                cache = read_json(p)
                rs = {r['image']: r for r in cache['records']}
                assert len(rs) == 161 and set(rs) == set(test.image)
                records[cell] = rs
            for image, ar in records['A_CBAM'].items():
                sr = records['S2_CBAM'][image]
                for field in ['gt_boxes', 'gt_cls', 'imgsz']:
                    assert ar[field] == sr[field], (image, field)
                boxes, classes, width, height = xml[image]
                assert ar['gt_cls'] == classes, (image, checkpoint, seed)
                # Existing evaluator used nominal imgsz 768 and rectangular padding;
                # square 1024 images become 768 image pixels within an 800 tensor.
                assert ar['imgsz'] == [800, 800] and width == height == 1024
                expected = boxes * .75 + 16
                error = np.max(np.abs(np.asarray(ar['gt_boxes'])-expected))
                max_gt_error = max(max_gt_error, float(error))
                assert error < .002, (image, error)
                ai, si = match(ar), match(sr)
                for k in np.flatnonzero(np.isfinite(ai) & np.isfinite(si)):
                    rebuilt.append(dict(checkpoint=checkpoint, seed=seed, image=image,
                                        gt_index=int(k), category=classes[k],
                                        A_CBAM_iou=ai[k], S2_CBAM_iou=si[k], difference=si[k]-ai[k]))
    rebuilt = pd.DataFrame(rebuilt)
    keys = ['checkpoint', 'seed', 'image', 'gt_index']
    a = common.set_index(keys).sort_index()
    b = rebuilt.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    for c in ['category', 'A_CBAM_iou', 'S2_CBAM_iou', 'difference']:
        assert np.allclose(a[c], b[c], atol=1e-12, rtol=0), c

    joined = common.merge(test.rename(columns={'object_index': 'audit_object_index', 'xml_gt_index': 'gt_index'}),
                          on=['image', 'gt_index'], how='left', validate='many_to_one')
    assert joined.disagreement_bin.notna().all()
    assert np.all(joined.category.astype(int).map(dict(enumerate(CLASS_NAMES))) == joined.class_a)
    joined['primary_stratum'] = np.where(joined.disagreement_bin.eq('unpaired'), 'unpaired',
                                         np.where(joined.identical, 'identical', 'nonidentical'))
    test['primary_stratum'] = np.where(test.disagreement_bin.eq('unpaired'), 'unpaired',
                                       np.where(test.identical, 'identical', 'nonidentical'))
    joined.to_csv(out/'cbam_common_objects_with_discrepancy.csv', index=False)
    test.to_csv(out/'test_annotation_index_verified.csv', index=False)

    # Each seed is one statistical replicate; objects/images are counts, not n=1612.
    per_seed = []
    summaries = []
    group_ids = np.sort(test.phash_group.unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(group_ids), np.full(len(group_ids), 1/len(group_ids)), size=BOOTSTRAP_REPLICATES)
    group_idx = {g: i for i, g in enumerate(group_ids)}
    bootstrap_distributions = {}
    for dimension in ['primary_stratum', 'disagreement_bin']:
        strata = ['all'] + sorted(test[dimension].unique().tolist())
        for checkpoint in ['last', 'best']:
            for stratum in strata:
                population = test if stratum == 'all' else test[test[dimension].eq(stratum)]
                d = joined[joined.checkpoint.eq(checkpoint)]
                if stratum != 'all': d = d[d[dimension].eq(stratum)]
                values = []
                boot_means = []
                for seed in SEEDS:
                    rows = d[d.seed.eq(seed)]
                    assert len(rows) > 0, (dimension, stratum, checkpoint, seed)
                    values.append(rows.difference.mean())
                    per_seed.append(dict(dimension=dimension, stratum=stratum, checkpoint=checkpoint,
                                         seed=seed, common_objects=len(rows), common_images=rows.image.nunique(),
                                         common_phash_groups=rows.phash_group.nunique(),
                                         A_CBAM_mean_iou=rows.A_CBAM_iou.mean(), S2_CBAM_mean_iou=rows.S2_CBAM_iou.mean(),
                                         difference=rows.difference.mean()))
                    # Fixed-model test-sampling sensitivity: resample whole near-duplicate
                    # groups, applying the same draw to all seeds and retaining within-
                    # group objects. This is separate from the seed t interval.
                    sums = np.zeros(len(group_ids)); counts = np.zeros(len(group_ids))
                    grouped = rows.groupby('phash_group').difference.agg(['sum', 'count'])
                    for gid, row in grouped.iterrows():
                        sums[group_idx[gid]] = row['sum']; counts[group_idx[gid]] = row['count']
                    den = weights @ counts
                    num = weights @ sums
                    boot_means.append(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))
                summary = dict(dimension=dimension, stratum=stratum, checkpoint=checkpoint,
                               test_objects=len(population), test_images=population.image.nunique(),
                               test_phash_groups=population.phash_group.nunique(),
                               common_unique_objects=d[['image', 'gt_index']].drop_duplicates().shape[0],
                               common_objects_per_seed_min=int(d.groupby('seed').size().min()),
                               common_objects_per_seed_max=int(d.groupby('seed').size().max()),
                               common_images_per_seed_min=int(d.groupby('seed').image.nunique().min()),
                               common_images_per_seed_max=int(d.groupby('seed').image.nunique().max()),
                               **paired_ci(values))
                boot_means = np.asarray(boot_means)
                valid = np.isfinite(boot_means).all(axis=0)
                distribution = boot_means[:, valid].mean(axis=0)
                lo, hi = np.quantile(distribution, [.025, .975])
                bootstrap_reported = population.phash_group.nunique() >= 10
                if not bootstrap_reported:
                    lo, hi = np.nan, np.nan
                summary.update(group_bootstrap_low=lo, group_bootstrap_high=hi,
                               group_bootstrap_valid_replicates=int(valid.sum()),
                               group_bootstrap_reported=bootstrap_reported,
                               bootstrap_type='fixed-model percentile; common objects; phash groups')
                summaries.append(summary)
                bootstrap_distributions[(dimension, checkpoint, stratum)] = (valid, boot_means)
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(out/'discrepancy_localization_per_seed.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary[summary.dimension.eq('primary_stratum')].to_csv(out/'discrepancy_localization_primary.csv', index=False)
    summary[summary.dimension.eq('disagreement_bin')].to_csv(out/'discrepancy_localization_exploratory_bins.csv', index=False)
    contrasts = []
    for checkpoint in ['last', 'best']:
        d = per_seed[(per_seed.dimension == 'primary_stratum') & per_seed.checkpoint.eq(checkpoint)]
        delta = (d[d.stratum.eq('nonidentical')].set_index('seed').loc[SEEDS, 'difference']
                 - d[d.stratum.eq('identical')].set_index('seed').loc[SEEDS, 'difference'])
        v1, x1 = bootstrap_distributions[('primary_stratum', checkpoint, 'nonidentical')]
        v0, x0 = bootstrap_distributions[('primary_stratum', checkpoint, 'identical')]
        valid = v1 & v0
        dist = (x1[:, valid]-x0[:, valid]).mean(axis=0)
        lo, hi = np.quantile(dist, [.025, .975])
        contrasts.append(dict(checkpoint=checkpoint, contrast='nonidentical minus identical',
                              **paired_ci(delta), group_bootstrap_low=lo, group_bootstrap_high=hi,
                              group_bootstrap_valid_replicates=int(valid.sum())))
    pd.DataFrame(contrasts).to_csv(out/'discrepancy_effect_modification.csv', index=False)

    # Reconcile the original per-object gradient records with group and reference sums.
    grad = read_csv(LEGACY/'gradient_objects_joined.csv')
    grad = grad[grad.version.eq('v8')].copy()
    refs = read_csv(LEGACY/'gradient_reference_summary.csv'); refs = refs[refs.version.eq('v8')]
    groups = read_csv(LEGACY/'gradient_group_summary.csv'); groups = groups[groups.version.eq('v8')]
    comps = read_csv(LEGACY/'gradient_component_summary.csv'); comps = comps[comps.version.eq('v8')]
    gb = read_csv(LEGACY/'gradient_bin_summary.csv'); gb = gb[gb.version.eq('v8')]
    assert len(grad) == 11320 and len(refs) == 10
    for (cell, seed), g in grad.groupby(['cell', 'seed']):
        assert len(g) == 1132 and g.image.nunique() == 744
        r = refs[(refs.cell == cell) & refs.seed.eq(seed)].iloc[0]
        for metric in ['g1_sq', 'g2_sq', 'diff_sq', 'dot']:
            assert np.isclose(g[metric].sum(), r[metric], atol=1e-10, rtol=1e-7), (cell, seed, metric)
        for segment, part in g.groupby('disagreement_bin'):
            old = groups[(groups.cell == cell) & groups.seed.eq(seed) & groups.dimension.eq('disagreement_bin') & groups.segment.eq(segment)].iloc[0]
            assert len(part) == old.n_objects
            assert np.isclose(part.diff_sq.sum()/g.diff_sq.sum(), old.difference_energy_share, atol=1e-9, rtol=1e-7)
    for name, d in [('annotation_objects.csv', annotations), ('gradient_objects_v8.csv', grad),
                    ('gradient_references_v8.csv', refs), ('gradient_groups_v8.csv', groups),
                    ('gradient_components_seed42_v8.csv', comps), ('gradient_bins_v8.csv', gb)]:
        d.to_csv(archive/name, index=False)
    for name in ['protocol.json', 'analysis_qa.json', 'gradient_qa.csv', 'ANALYSIS_NOTES.md']:
        shutil.copy2(source(LEGACY/name), archive/name)

    timings = []
    for seed in SEEDS:
        p = WORKSPACE/f'experiments/runs_yolov8_attention_comparison_v1/cbam/seed_{seed}/results.csv'
        d = read_csv(p); d.columns = d.columns.str.strip()
        assert len(d) == 200
        timings.append(dict(cell='historical S2+CBAM', seed=seed, completed_epochs=len(d),
                            cumulative_training_seconds=float(d.time.iloc[-1]),
                            source=str(p), interpretation='historical runtime only; not new S1 runtime'))
    pd.DataFrame(timings).to_csv(out/'historical_cbam_training_times.csv', index=False)

    qa = dict(analysis='exploratory; historical trained models; no training or inference',
              comparison='S2+CBAM minus A+CBAM; assignment and regression both change',
              missing_causal_control='S1+CBAM (under preparation by the main revision task)',
              all_A_XML_object_indices_and_boxes_verified=True, XML_images_verified=1065,
              audit_to_XML_order_mapping='unique A box coordinates plus class, all images; audit indices cannot be joined directly',
              objects_with_audit_index_different_from_XML=int((annotations.object_index != annotations.xml_gt_index).sum()),
              cached_prediction_files_verified=20, cached_test_images_per_file=161,
              cached_A_groundtruth_class_and_coordinate_order_verified=True,
              cached_GT_max_absolute_error_pixels=max_gt_error,
              cached_GT_transform='original coordinates * 0.75 + 16 in nominal768/padded800 evaluation',
              common_object_rows_reproduced=1612, common_object_IoUs_verified_atol=1e-12,
              test_objects=233, test_images=161, test_phash_groups=len(group_ids),
              seed_interval='unadjusted two-sided Student t; n=5, df=4; fixed split',
              test_sampling_interval='separate percentile cluster bootstrap of phash groups; fixed models and common detections',
              bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
              group_bootstrap_reporting_rule='not reported for strata with fewer than 10 original test phash groups',
              gradient_v8_references_verified=10, gradient_v8_object_rows_verified=11320,
              gradient_sums_reconciled=True,
              gradient_interpretation='change in objective gradient; not attribution of AP benefit',
              original_AP_AB_averages_not_used=True,
              limitations=['post-hoc exploratory stratification', 'conditional common-detection selection',
                           'group identity is perceptual near-duplicate, not blade identity',
                           'small nonidentical and unpaired test subsets',
                           'seed and group-bootstrap uncertainty are not a joint population interval',
                           'no adjudicated ground truth for annotation correctness'])
    (out/'qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT/'source_manifest.json').write_text(json.dumps(SOURCES, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[summary.dimension.eq('primary_stratum')][['checkpoint','stratum','test_objects','mean','seed_ci_low','seed_ci_high','group_bootstrap_low','group_bootstrap_high']].to_string(index=False))
    print('QA passed; CPU-only outputs:', out)


if __name__ == '__main__':
    main()
