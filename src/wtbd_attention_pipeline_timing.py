"""Fixed-image batch-1 preprocessing/forward/NMS timing; disk IO excluded."""
from pathlib import Path
import random
import time

def load_benchmark_images(root, smoke=False):
    import cv2
    directory = Path(root) / 'data/gt_primary/images/test'
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
    assert len(paths) == 161, f'Unexpected test image count: {len(paths)}'
    selected = sorted(random.Random(20260918).sample(paths, 2 if smoke else 32))
    images = [cv2.imread(str(p)) for p in selected]
    assert all(im is not None for im in images)
    return selected, images

def time_image_pipeline(model, images, size):
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import nms
    letterbox = LetterBox(new_shape=(size,size), auto=False, stride=32)
    rows = []
    def one(im, record=True):
        torch.cuda.synchronize()
        start = time.perf_counter()
        rgb_chw = np.ascontiguousarray(letterbox(image=im)[...,::-1].transpose(2,0,1))
        tensor = torch.from_numpy(rgb_chw).unsqueeze(0).cuda().float().div_(255)
        torch.cuda.synchronize()
        pre = time.perf_counter()
        prediction = model(tensor)
        torch.cuda.synchronize()
        infer = time.perf_counter()
        det = nms.non_max_suppression(prediction, conf_thres=.001, iou_thres=.7, max_det=300, nc=6)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if record:
            rows.append({'preprocess_ms':1000*(pre-start),'inference_ms':1000*(infer-pre),
                         'postprocess_ms':1000*(end-infer),'pipeline_ms':1000*(end-start),
                         'detections':len(det[0])})
    with torch.inference_mode():
        one(images[0], record=False)
        for im in images:
            one(im)
    result = {f'mean_{k}':float(np.mean([row[k] for row in rows])) for k in
              ('preprocess_ms','inference_ms','postprocess_ms','pipeline_ms')}
    result['p95_pipeline_ms'] = float(np.percentile([row['pipeline_ms'] for row in rows],95))
    result['scope'] = 'fixed preloaded images; resize/letterbox, BGR-RGB, H2D, FP32 normalize, unfused forward, NMS; excludes disk IO, coordinate rescaling, visualization and storage'
    result['image_samples'] = rows
    return result
