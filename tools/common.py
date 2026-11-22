"""Portable CPU-only numerical utilities for replaying the published study."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CELLS = ["A", "B", "S1", "S2", "A_ESB", "A_CBAM", "S2_ESB", "S2_CBAM"]

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def read_csv(name):
    return pd.read_csv(DATA / name)

def interval(values):
    x = np.asarray(values, float)
    if len(x) != 5 or not np.isfinite(x).all():
        raise ValueError("The original inferential unit is five seedwise differences.")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    half = float(t.ppf(.975, 4) * sd / np.sqrt(5))
    return dict(mean=mean, sample_sd=sd, ci_low=mean-half,
                ci_high=mean+half, positive_seeds=int((x > 0).sum()), n=5, df=4)

def pairwise_iou(gt, pred):
    left = np.maximum(gt[:, None, :2], pred[None, :, :2])
    right = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    inter = np.clip(right-left, 0, None).prod(2)
    ga = (gt[:, 2:]-gt[:, :2]).prod(1)
    pa = (pred[:, 2:]-pred[:, :2]).prod(1)
    return inter / np.maximum(ga[:, None]+pa[None, :]-inter, 1e-12)

def counts(records, threshold):
    ng = np.zeros(6, int); npred = np.zeros(6, int); tp = np.zeros(6, int)
    for record in records:
        gt = np.asarray(record["gt_boxes"], float).reshape(-1, 4)
        gc = np.asarray(record["gt_cls"], int)
        confidence = np.asarray(record["conf"], float)
        keep = np.flatnonzero(confidence >= threshold)
        keep = keep[np.argsort(-confidence[keep], kind="stable")]
        pred = np.asarray(record["pred_boxes"], float).reshape(-1, 4)[keep]
        pc = np.asarray(record["pred_cls"], int)[keep]
        overlaps = pairwise_iou(gt, pred); used = np.zeros(len(gt), bool)
        ng += np.bincount(gc, minlength=6)
        npred += np.bincount(pc, minlength=6)
        for j, cls in enumerate(pc):
            candidates = np.flatnonzero((gc == cls) & (~used) & (overlaps[:, j] >= .5))
            if len(candidates):
                index = candidates[np.argmax(overlaps[candidates, j])]
                used[index] = True; tp[cls] += 1
    return ng, npred, tp

def metrics(gt, predictions, tp):
    fp = predictions-tp; fn = gt-tp
    return dict(GT=int(gt), TP=int(tp), FP=int(fp), FN=int(fn),
                P=100*tp/predictions if predictions else 0.,
                R=100*tp/gt if gt else 0.,
                F1=200*tp/(gt+predictions) if gt+predictions else 0.)

def summary(frame, keys, metrics):
    rows=[]
    for key, part in frame.groupby(keys, dropna=False, sort=False):
        if not isinstance(key, tuple): key=(key,)
        row=dict(zip(keys,key)); row["n"]=len(part)
        for metric in metrics:
            row[metric+"_mean"]=float(part[metric].mean())
            row[metric+"_sample_sd"]=float(part[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)

def save_csv(frame, folder, name):
    folder.mkdir(parents=True,exist_ok=True)
    frame.to_csv(folder/name,index=False,float_format="%.12g")
