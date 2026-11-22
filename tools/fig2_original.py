"""Plot the aspect-ratio distribution of the 1,584 boxes in annotation set A."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[1]
BLUE,ORANGE,INK,INK2='#2a78d6','#eb6834','#0b0b0b','#52514e'
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--annotation-audit", type=Path, required=True)
args = parser.parse_args()
annotation = pd.read_csv(args.annotation_audit)
coordinates = np.asarray([list(map(float, box.split())) for box in annotation["box_a"]])
width = coordinates[:, 2] - coordinates[:, 0]
height = coordinates[:, 3] - coordinates[:, 1]
shape = pd.DataFrame({"cls": annotation["class_a"], "rho": np.maximum(width / height, height / width)})
assert len(shape)==1584 and np.isfinite(shape.rho).all() and (shape.rho>0).all()
classes=['craze','crack','corrosion','hide_craze','surface_injure','thunderstrike']
fig,ax=plt.subplots(figsize=(5.6,2.7))
rng = np.random.default_rng(0)
for i, c in enumerate(classes):
    v = np.log10(shape.loc[shape["cls"] == c, "rho"].to_numpy())
    y = i + rng.uniform(-0.28, 0.28, len(v))
    ax.scatter(v, y, s=3, color=BLUE, alpha=0.35, linewidths=0)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([q1, q3], [i, i], color=INK, lw=2.2, solid_capstyle="round")
    ax.plot(med, i, "o", ms=4.5, color="white", mec=INK, mew=1.0)
    share = (10 ** v >= 3).mean() * 100
    ax.text(1.62, i, f"{share:.1f}%", va="center", ha="left", fontsize=7.5, color=INK2)
ax.axvline(np.log10(3), color=ORANGE, lw=1.0, ls="--")
ax.text(np.log10(3) + 0.02, 5.62, r"$\rho = 3$", color=INK2, fontsize=7.5)
ax.text(1.62, 5.62, r"$\rho \geq 3$", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(classes)), classes)
ax.invert_yaxis()
ticks = [1, 2, 3, 5, 10, 20, 40]
ax.set_xticks(np.log10(ticks), [str(t) for t in ticks])
ax.set_xlim(-0.02, 1.85)
ax.set_ylim(5.9, -0.9)
ax.set_xlabel(r"Box aspect ratio $\rho = \max(w,h)/\min(w,h)$ (log scale)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
require_matplotlib_panel_alignment(fig,json_out=str(ROOT/'outputs/figure_qa/fig2_alignment.json'),strict=True)
fig.savefig(ROOT/'outputs/figures/fig2_dataset.pdf',bbox_inches='tight')
fig.savefig(ROOT/'outputs/figures/fig2_dataset.svg',bbox_inches='tight')
fig.savefig(ROOT/'outputs/figures/fig2_dataset.png',dpi=600,bbox_inches='tight')
plt.close(fig)
