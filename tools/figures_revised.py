"""Plot detection, localization and screening results with the S1+CBAM control.

Estimates are paired by training seed against the A baseline. Localization
panels use mean IoU over each model's true positives."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

import figures_original as original
from analyse_prf import contrast
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(42, 47))
CELLS = ['S2', 'A_ESB', 'A_CBAM', 'S2_ESB', 'S1_CBAM', 'S2_CBAM']
WIDTH = 184.6 / 25.4
COLORS = {**original.COLORS, 'S1_CBAM': '#729BB8'}
LABELS = {**original.LABEL, 'S1_CBAM': 'S1 + CBAM'}
ORIGINAL_D = deepcopy(original.D)
D = deepcopy(original.D)
INPUTS = ['data/baseline_contrasts_v12.json', 'data/yolov8_reeval_table.csv',
          'data/localization_v12.csv', 'data/locked_per_seed.csv',
          'data/locked_per_class.csv', 'revision_s1_cbam/analysis/native_AP_raw.csv',
          'revision_s1_cbam/analysis/localization_per_seed.csv',
          'revision_s1_cbam/analysis/locked_screening_per_seed.csv']


def prepare():
    native = pd.read_csv(ROOT / INPUTS[5])
    native = native[native.reference == 'A'].pivot(index=['cell', 'seed', 'checkpoint'], columns='metric', values='value').reset_index().rename(columns={'checkpoint': 'ck'})
    loc = pd.read_csv(ROOT / INPUTS[6]).rename(columns={'checkpoint': 'ck'})
    locked = pd.read_csv(ROOT / INPUTS[7])
    locked = locked[locked.reference == 'A']
    # Verify that both inputs contain the same S2+CBAM results.
    for ck in ['best', 'last']:
        old_ap = original.AP[(original.AP.cell == 'S2_CBAM') & (original.AP.ck == ck)].set_index('seed').loc[SEEDS]
        new_ap = native[(native.cell == 'S2_CBAM') & (native.ck == ck)].set_index('seed').loc[SEEDS]
        for metric in ['AP', 'AP75']:
            assert np.allclose(old_ap[metric], new_ap[metric], atol=1e-12, rtol=0)
        old_loc = original.LQ[(original.LQ.cell == 'S2_CBAM') & (original.LQ.ck == ck)].set_index('seed').loc[SEEDS]
        new_loc = loc[(loc.cell == 'S2_CBAM') & (loc.ck == ck)].set_index('seed').loc[SEEDS]
        assert np.allclose(old_loc.tp_mean_iou, new_loc.tp_mean_iou, atol=1e-12, rtol=0)
    ap = pd.concat([original.AP, native[native.cell == 'S1_CBAM']], ignore_index=True)
    lq = pd.concat([original.LQ, loc[loc.cell == 'S1_CBAM']], ignore_index=True)
    screening = pd.concat([pd.read_csv(ROOT / INPUTS[3]), locked[locked.cell == 'S1_CBAM']], ignore_index=True)
    for ck in ['best', 'last']:
        for source, metrics in [(ap, ['AP', 'AP75']), (lq, ['tp_mean_iou'])]:
            data = source[source.ck == ck].set_index(['cell', 'seed'])
            for metric in metrics:
                delta = data.loc['S1_CBAM'].loc[SEEDS, metric] - data.loc['A'].loc[SEEDS, metric]
                D[f'{ck}:S1_CBAM:{metric}'] = contrast(delta)
    data = screening.set_index(['cell', 'seed'])
    D['locked:S1_CBAM:F1'] = contrast(data.loc['S1_CBAM'].loc[SEEDS, 'F1'] - data.loc['A'].loc[SEEDS, 'F1'])
    assert all(D[key] == value for key, value in ORIGINAL_D.items())
    return ap, lq, screening


def save(fig, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,
        json_out=str(ROOT / 'outputs/figure_qa' / f'{name}.alignment.json'),
        overlay_svg=str(ROOT / 'outputs/figure_qa' / f'{name}.alignment.svg'),
        require_panel_labels=True, strict=True)
    fig.savefig(ROOT / 'outputs/figures' / f'{name}.pdf')
    fig.savefig(ROOT / 'outputs/figures' / f'{name}.svg')
    fig.savefig(ROOT / 'outputs/figures' / f'{name}.png', dpi=600.1)
    with Image.open(ROOT / 'outputs/figures' / f'{name}.png') as raster:
        raster.convert('RGB').save(ROOT / 'outputs/figures' / f'{name}.png', dpi=(600.1, 600.1))
    fig.savefig(ROOT / 'outputs/figures' / f'{name}_preview.png', dpi=300)
    plt.close(fig)


def forest(ax, values, labels, xlabel, xlim):
    for y, (cell, effect, raw) in enumerate(values):
        color = COLORS[cell]
        assert len(raw) == 5 and np.isfinite(raw).all()
        ax.scatter(raw, np.full(5, y) - .15 + np.arange(5) * .035, s=12, color=color, alpha=.45, zorder=3, edgecolors='none')
        ax.errorbar(effect['mean'], y + .13,
            xerr=[[effect['mean'] - effect['lo']], [effect['hi'] - effect['mean']]],
            fmt='D', ms=4, color=color, lw=1.3, capsize=2, zorder=4)
    ax.axvline(0, color='#555555', lw=.8, ls='--', zorder=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .55, -.55)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color='#eeeeee', lw=.6)
    ax.tick_params(axis='y', length=0, pad=5)


def fig3(ap, lq):
    fig, axs = plt.subplots(3, 2, figsize=(WIDTH, WIDTH * 7.4 / 7.3))
    fig.subplots_adjust(left=.16, right=.98, bottom=.07, top=.88, wspace=.44, hspace=.62)
    source_rows = []
    for ax, (ck, metric), letter in zip(axs.flat,
            [('best', 'AP'), ('last', 'AP'), ('best', 'AP75'), ('last', 'AP75'), ('best', 'tp_mean_iou'), ('last', 'tp_mean_iou')], 'abcdef'):
        source = ap if metric.startswith('AP') else lq
        data = source[source.ck == ck].set_index(['cell', 'seed'])
        values = []
        for cell in CELLS:
            raw = (data.loc[cell].loc[SEEDS, metric] - data.loc['A'].loc[SEEDS, metric]).values
            effect = D[f'{ck}:{cell}:{metric}']
            assert np.allclose([effect[k] for k in ['mean', 'lo', 'hi']], [contrast(raw)[k] for k in ['mean', 'lo', 'hi']], atol=1e-12, rtol=0)
            values.append((cell, effect, raw))
            for seed, delta in zip(SEEDS, raw):
                source_rows.append(dict(panel=letter, checkpoint=ck, metric=metric, cell=cell, seed=seed, paired_difference=float(delta), mean=effect['mean'], lo=effect['lo'], hi=effect['hi']))
        xlabel = {'AP': 'Change in AP50–95 (percentage points)', 'AP75': 'Change in AP75 (percentage points)', 'tp_mean_iou': 'Change in mean TP IoU'}[metric]
        xlim = {'AP': (-3, 5), 'AP75': (-8, 17), 'tp_mean_iou': (-.017, .043)}[metric]
        forest(ax, values, [LABELS[c] for c in CELLS], xlabel, xlim)
        ax.set_title('Validation-selected best checkpoint' if ck == 'best' else 'Fixed epoch 200', loc='left', pad=10)
        original.letter(ax, letter)
        if metric == 'tp_mean_iou':
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(.02))
    fig.text(.16, .945, 'Small dots: five paired seeds    Diamonds and bars: mean difference and 95% CI', fontsize=7.5, color='#555555')
    save(fig, 'fig3_localization')
    pd.DataFrame(source_rows).to_csv(ROOT / 'outputs/figure_qa/fig3_revised_source_data.csv', index=False)


def fig5(screening):
    data = screening.set_index(['cell', 'seed'])
    classes = pd.read_csv(ROOT / 'data/locked_per_class.csv')
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 3.4 / 7.3))
    fig.subplots_adjust(left=.16, right=.98, bottom=.18, top=.72, wspace=.60)
    cells = ['S1'] + CELLS
    values = []
    source_rows = []
    for cell in cells:
        raw = (data.loc[cell].loc[SEEDS, 'F1'] - data.loc['A'].loc[SEEDS, 'F1']).values
        effect = D[f'locked:{cell}:F1']
        assert np.allclose([effect[k] for k in ['mean', 'lo', 'hi']], [contrast(raw)[k] for k in ['mean', 'lo', 'hi']], atol=1e-12, rtol=0)
        values.append((cell, effect, raw))
        for seed, delta in zip(SEEDS, raw):
            source_rows.append(dict(panel='a', metric='F1', cell=cell, seed=seed, paired_difference=float(delta), mean=effect['mean'], lo=effect['lo'], hi=effect['hi']))
    forest(axs[0], values, [LABELS[c] for c in cells], 'Change in F1 (percentage points)', (-6, 7))
    original.letter(axs[0], 'a')
    axs[0].set_title('Overall screening', loc='left', pad=10)
    values = []
    labels = ['Crazing', 'Cracks', 'Corrosion', 'Hidden crazing', 'Surface damage', 'Lightning damage']
    for cid, label in enumerate(labels):
        sub = classes[classes.class_id == cid].set_index(['cell', 'seed'])
        raw = (sub.loc['S2_CBAM'].loc[SEEDS, 'R'] - sub.loc['A'].loc[SEEDS, 'R']).values
        effect = contrast(raw)
        values.append(('S2_CBAM', effect, raw))
        for seed, delta in zip(SEEDS, raw):
            source_rows.append(dict(panel='b', metric='R', cell=label, seed=seed, paired_difference=float(delta), mean=effect['mean'], lo=effect['lo'], hi=effect['hi']))
    forest(axs[1], values, labels, 'Change in recall (percentage points)', (-30, 30))
    original.letter(axs[1], 'b')
    axs[1].set_title('S2 + CBAM by defect class', loc='left', pad=10)
    fig.text(.16, .94, 'Application check with validation-locked thresholds', fontsize=11, fontweight='bold')
    fig.text(.16, .86, 'Differences relative to YOLOv8n (A); five seeds, mean and 95% CI; fixed epoch 200', fontsize=7.5, color='#555555')
    save(fig, 'fig5_application')
    pd.DataFrame(source_rows).to_csv(ROOT / 'outputs/figure_qa/fig5_revised_source_data.csv', index=False)


def main():
    (ROOT / 'outputs/figures').mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs/figure_qa').mkdir(parents=True, exist_ok=True)
    ap, lq, screening = prepare()
    fig3(ap, lq)
    fig5(screening)
    report = dict(
        manuscript_figures=[4, 6], added_configuration='S1_CBAM',
        seeds=SEEDS, replicate_unit='training seed', n=5, df=4,
        center='mean paired difference against A baseline',
        interval='two-sided paired Student t 95% CI, unadjusted',
        original_effect_estimates_unchanged=all(D[k] == v for k, v in ORIGINAL_D.items()),
        TP_IoU_definition='difference in each model mean TP IoU; not common-object intersection IoU',
        source_files=[dict(path=p, sha256=hashlib.sha256((ROOT / p).read_bytes()).hexdigest()) for p in INPUTS],
        new_effects={k: v for k, v in D.items() if ':S1_CBAM:' in k},
        inference_performed=False, training_performed=False)
    (ROOT / 'outputs/figure_qa/revised_figure_data_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['new_effects'], indent=2))


if __name__ == '__main__':
    main()
