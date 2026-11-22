"""Deterministic editable-vector diagrams for the manuscript revision.

All geometry is expressed in final-size points. The SVG retains live text;
the PDF embeds TrueType text, and the PNG is exported at 600 dpi.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
PT_PER_MM = 72 / 25.4
WIDTH = 184.6 * PT_PER_MM
FONT = "Arial"
INK = "#1c3444"
EDGE = "#708898"
FLOW = "#3a586e"
BLUE = "#eaf2f7"
PALE = "#f5f7f8"
MODULE = "#dcebf4"
ACCENT = "#fce8cd"
ORANGE = "#b67826"

plt.rcParams.update({"font.family": FONT, "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none", "savefig.facecolor": "white"})


class Diagram:
    def __init__(self, name, height):
        self.name, self.height = name, height
        self.fig = plt.figure(figsize=(WIDTH / 72, height / 72), dpi=72)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH), ylim=(height, 0))
        self.ax.axis("off")
        self.root = Element("svg", {"xmlns": "http://www.w3.org/2000/svg",
                            "width": "184.6mm", "height": f"{height / PT_PER_MM:.5f}mm",
                            "viewBox": f"0 0 {WIDTH:.6f} {height:.6f}"})
        SubElement(self.root, "rect", {"x": "0", "y": "0", "width": str(WIDTH),
                   "height": str(height), "fill": "white"})
        defs = SubElement(self.root, "defs")
        marker = SubElement(defs, "marker", {"id": "arrow", "markerWidth": "6",
            "markerHeight": "6", "refX": "6", "refY": "3", "orient": "auto",
            "markerUnits": "userSpaceOnUse", "viewBox": "0 0 6 6"})
        SubElement(marker, "path", {"d": "M 0 0 L 6 3 L 0 6 Z", "fill": FLOW})
        self.text_records = []
        self.components = []

    def text(self, x, y, text, size=9, weight="normal", anchor="middle", group=None):
        parent = group if group is not None else self.root
        node = SubElement(parent, "text", {"x": str(x), "y": str(y),
             "font-family": "Arial, Helvetica, sans-serif", "font-size": f"{size}pt",
             "font-weight": weight, "fill": INK, "text-anchor": anchor,
             "dominant-baseline": "central"})
        node.text = text
        artist = self.ax.text(x, y, text, fontsize=size, fontweight=weight,
                              ha={"middle": "center", "start": "left", "end": "right"}[anchor],
                              va="center", color=INK)
        self.text_records.append((text, size, artist))
        return artist

    def box(self, ident, x, y, w, h, lines, fill=BLUE, border=EDGE,
            size=9, lineheight=10.6, sizes=None):
        group = SubElement(self.root, "g", {"id": ident})
        SubElement(group, "rect", {"x": str(x), "y": str(y), "width": str(w),
            "height": str(h), "fill": fill, "stroke": border,
            "stroke-width": "0.9"})
        patch = Rectangle((x, y), w, h, facecolor=fill, edgecolor=border, linewidth=.9)
        self.ax.add_patch(patch)
        center = y + h / 2
        arts = []
        for i, line in enumerate(lines):
            arts.append(self.text(x + w / 2, center + (i - (len(lines) - 1) / 2) * lineheight,
                                 line, size=(sizes[i] if sizes else size), group=group))
        self.components.append(dict(id=ident, x=x, y=y, w=w, h=h, text_artists=arts))

    def arrow(self, ident, points, width=1.05):
        path = "M " + " L ".join(f"{x},{y}" for x, y in points)
        SubElement(self.root, "path", {"id": ident, "d": path, "fill": "none",
            "stroke": FLOW, "stroke-width": str(width), "stroke-linejoin": "round",
            "marker-end": "url(#arrow)"})
        p = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        self.ax.add_patch(FancyArrowPatch(path=p, arrowstyle="-|>,head_length=4,head_width=2",
                                         mutation_scale=1, linewidth=width, color=FLOW,
                                         shrinkA=0, shrinkB=0, capstyle="round", joinstyle="round"))

    def connector(self, ident, points, width=1.05):
        """Undirected trunk of a branching or converging directed flow."""
        path = "M " + " L ".join(f"{x},{y}" for x, y in points)
        SubElement(self.root, "path", {"id": ident, "d": path, "fill": "none",
            "stroke": FLOW, "stroke-width": str(width), "stroke-linejoin": "round"})
        self.ax.plot([p[0] for p in points], [p[1] for p in points],
                     color=FLOW, linewidth=width, solid_capstyle="round")

    def dot(self, x, y):
        SubElement(self.root, "circle", {"cx": str(x), "cy": str(y), "r": "1.5", "fill": FLOW})
        self.ax.add_patch(plt.Circle((x, y), 1.5, color=FLOW, linewidth=0))

    def export(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        problems = []
        for c in self.components:
            for a in c["text_artists"]:
                bb = a.get_window_extent(renderer=renderer)
                xy = self.ax.transData.inverted().transform(bb.get_points())
                xmin, xmax = sorted(xy[:, 0]); ymin, ymax = sorted(xy[:, 1])
                if xmin < c["x"] + 3 or xmax > c["x"] + c["w"] - 3 or ymin < c["y"] + 2 or ymax > c["y"] + c["h"] - 2:
                    problems.append(dict(component=c["id"], text=a.get_text(), bbox=[xmin, ymin, xmax, ymax]))
        assert not problems, problems
        ElementTree(self.root).write(OUT / f"{self.name}.svg", encoding="utf-8", xml_declaration=True)
        self.fig.savefig(OUT / f"{self.name}.pdf", dpi=600, pad_inches=0)
        # A slight rendering margin prevents pixel rounding below 600 effective dpi.
        self.fig.savefig(OUT / f"{self.name}.png", dpi=600.1, pad_inches=0)
        from PIL import Image
        png_path = OUT / f"{self.name}.png"
        with Image.open(png_path) as image:
            rgb = image.convert("RGB")
            rgb.save(png_path, dpi=(600.1, 600.1))
        self.fig.savefig(OUT / f"{self.name}_preview.png", dpi=150, pad_inches=0)
        qa = dict(width_mm=184.6, height_mm=self.height / PT_PER_MM,
                  minimum_font_pt=min(s for _, s, _ in self.text_records),
                  maximum_font_pt=max(s for _, s, _ in self.text_records),
                  live_SVG_text=True, PDF_font_type=42, PNG_dpi=600.1,
                  font=FONT, font_file=font_manager.findfont(FONT),
                  component_text_containment_passed=True,
                  text_labels=[t for t, _, _ in self.text_records])
        (OUT / f"{self.name}_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
        plt.close(self.fig)

SEEDS = set(range(42, 47))
MATRIX = (
    ('YOLOv8n', ('A', 'B', 'S1', 'S2', None)),
    ('YOLOv8n + CBAM', ('A_CBAM', None, 'S1_CBAM', 'S2_CBAM', 's2_cbam_b_priority')),
    ('YOLOv8n + ESB', ('A_ESB', None, None, 'S2_ESB', None)),
    ('YOLOv8s', ('yolov8s_a', None, None, None, None)),
    ('YOLO11n', ('yolo11n_a', None, None, None, None)),
)


def draw_configuration_matrix(diagram, n_configs):
    diagram.box('configuration_matrix', 14, 183, 495, 156, [], fill='#f8fbfd')
    diagram.text(24, 196, 'WTBD training configurations', 10.0, 'bold', 'start')
    diagram.text(499, 196, f'{n_configs} configurations × {len(SEEDS)} seeds = {n_configs * len(SEEDS)} runs',
                 9.2, 'normal', 'end')
    for y in (207, 242, 258, 274, 290, 306, 322):
        diagram.connector(f'matrix_row_rule_{y}', [(23, y), (500, y)], width=0.42)
    columns = (173, 244, 315, 386, 457)
    diagram.text(315, 215, 'Training supervision', 8.9, 'bold')
    diagram.text(24, 229, 'Detector / feature module', 9.1, 'bold', 'start')
    for x, label in zip(columns[:3], ('A only', 'B only', 'S1'), strict=True):
        diagram.text(x, 230, label, 9.1, 'bold')
    for x, priority in zip(columns[3:], ('A priority', 'B priority'), strict=True):
        diagram.text(x, 226, 'S2', 9.1, 'bold')
        diagram.text(x, 236, priority, 8.8)
    for (label, cells), y in zip(MATRIX, (250, 266, 282, 298, 314), strict=True):
        diagram.text(24, y, label, 9.25, 'normal', 'start')
        for cell, x in zip(cells, columns, strict=True):
            if cell:
                diagram.text(x, y, '●', 9.0)
    diagram.text(24, 330.5, '●  Trained configuration', 8.8, 'normal', 'start')
    diagram.text(499, 330.5, 'S2 priority applies to conflicting class labels', 8.8, 'normal', 'end')


def main():
    counts = {'train': 744, 'val': 160, 'test': 161}
    n_configs = sum(cell is not None for _, variants in MATRIX for cell in variants)
    d = Diagram('fig2_experimental_workflow', 603)

    d.box('dataset', 113, 7, 297, 36, [
        'WTBD (1,065 images) with annotations A and B', '750 perceptual-hash groups',
    ], size=9.7, lineheight=12)
    d.text(14, 47, 'Grouped image split', 9.3, 'bold', 'start')
    d.connector('dataset_to_splits', [(261.5, 43), (261.5, 55)])
    d.connector('split_connector', [(88.5, 55), (434.5, 55)])
    d.dot(261.5, 55)
    for subset, left, title in (('train', 14, 'Training'), ('val', 187, 'Validation'), ('test', 360, 'Test')):
        text = [title, f'{counts[subset]} images']
        if subset == 'val':
            text.append('Fused and A labels')
        x = left + 74.5
        d.box('split_' + subset, left, 69, 149, 37, text, size=9.2, lineheight=10.5 if subset == 'val' else 12)
        d.arrow('split_to_' + subset, [(x, 55), (x, 69)])
    d.box('training_protocol', 14, 132, 495, 35, [
        'Shared WTBD training protocol',
        'COCO initialization, 200 epochs, 768 × 768 input, AdamW, batch 16, seeds 42–46',
    ], size=9.5, lineheight=12)
    d.arrow('train_to_protocol', [(88.5, 106), (88.5, 119), (261.5, 119), (261.5, 132)])
    d.arrow('protocol_to_configurations', [(261.5, 167), (261.5, 183)])
    draw_configuration_matrix(d, n_configs)

    d.connector('configurations_to_checkpoints', [(261.5, 339), (261.5, 353)])
    d.connector('checkpoint_branch', [(88.5, 353), (434.5, 353)])
    d.dot(261.5, 353)
    d.arrow('branch_to_epoch_200', [(88.5, 353), (88.5, 369)])
    d.arrow('branch_to_validation_best', [(434.5, 353), (434.5, 369)])
    d.box('epoch_200', 14, 369, 149, 45, [
        'Fixed epoch-200 checkpoints', 'All 12 configurations',
    ], size=9.15, lineheight=12)
    d.box('locked_thresholds', 187, 369, 149, 45, [
        'Locked screening thresholds', 'Maximize F1 on A validation', '8 configurations × 5 seeds',
    ], size=9.1, lineheight=10.7)
    d.box('validation_best', 360, 369, 149, 45, [
        'Validation-selected best', 'Fused validation AP50–95', '9 primary configurations',
    ], size=9.1, lineheight=10.7)
    d.arrow('epoch_200_to_thresholds', [(163, 391.5), (187, 391.5)])

    d.box('test_evaluation', 14, 431, 495, 28, ['Held-out WTBD test evaluation on 161 images'], size=10)
    for x, name in ((88.5, 'epoch_200'), (261.5, 'thresholds'), (434.5, 'validation_best')):
        d.arrow(name + '_to_test', [(x, 414), (x, 431)])
    d.arrow('test_images_to_evaluation', [(509, 87.5), (519, 87.5), (519, 445), (509, 445)])

    for left, name, text in (
        (14, 'primary_comparisons', ['Primary WTBD comparisons', 'Detection AP and box IoU',
                                    'Epoch 200 and validation best', 'Locked-threshold screening', 'Test reference A']),
        (183, 'class_precedence', ['Class-label precedence', 'YOLOv8n with A or B only',
                                   'S2+CBAM with A or B priority', 'Fixed epoch 200', 'Test references A and B']),
        (352, 'detector_benchmark', ['Detector benchmark', 'YOLOv8n, YOLOv8s, YOLO11n', 'A only for these baselines',
                                     'S2+CBAM (A priority)', 'Epoch 200, test reference A']),
    ):
        d.arrow('test_to_' + name, [(left + 78.5, 459), (left + 78.5, 476)])
        d.box(name, left, 476, 157, 65, text, fill=BLUE, size=9.15, lineheight=10.8)
    d.text(261.5, 552, 'WTBD differences are paired by seed with 95% intervals', 9.15)
    d.box('dtu_false_alarm', 14, 564, 495, 32, [
        'Exploratory DTU false-alarm evaluation on 11 uncropped frames',
        'YOLOv8n (A), S1+CBAM, S2+CBAM (A priority), epoch 200, WTBD A-validation thresholds',
    ], fill=PALE, size=9.2, lineheight=11.5)
    d.export()


if __name__ == '__main__':
    main()
