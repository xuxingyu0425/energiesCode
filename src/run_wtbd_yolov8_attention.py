"""Train and benchmark S2 attention configurations across five seeds.

Per-seed workers verify model initialization before training and run batch-one latency benchmarks."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback

# Set before the first CUDA context is created, including preflight forwards.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import run_wtbd_yolov8_feature_completion as core

ROOT = core.ROOT
FORMAL_REPORT = ROOT / 'reports/yolov8_attention_comparison_v1'
REPORT = FORMAL_REPORT
RUNS = ROOT / 'runs_yolov8_attention_comparison_v1'
SEEDS = (42, 43, 44, 45, 46)
METHODS = {n: (title, f'yolov8n_wtbd_{n}.yaml') for n, title in
           [('eca', 'ECA'), ('cbam', 'CBAM'), ('ca', 'CA')]}
HISTORICAL = {'f0_baseline': ('No module', 'v8_s2_f0'),
              'f1_esb': ('ESB', 'v8_s2_f1'), 'f2_lsk': ('LSK-style', 'f2_lsk')}
ALL = list(HISTORICAL) + list(METHODS)
METRICS = ('map50_95', 'map50', 'map75', 'precision', 'recall')
SMOKE = False
now, sha, write_json, read_csv, write_csv, state_hash = (
    core.now, core.sha, core.write_json, core.read_csv, core.write_csv, core.state_hash)

def label(method):
    return (METHODS if method in METHODS else HISTORICAL)[method][0]

def source_run(method, seed):
    if method in METHODS:
        return RUNS / method / f'seed_{seed}'
    if method == 'f2_lsk':
        return ROOT / 'runs_yolov8_feature_completion_v1/f2_lsk' / f'seed_{seed}'
    return ROOT / 'runs_yolov8_ab_s2_f1_v1' / HISTORICAL[method][1] / f'seed_{seed}'

def record_dir(method, seed):
    return REPORT / 'records' / method / f'seed_{seed}'

def cohort(method):
    return 'new_attention_v1' if method in METHODS else 'historical_lsk' if method == 'f2_lsk' else 'historical_baseline_esb'

def build(method, seed, callbacks=True):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    from wtbd_modules import register_standard_modules
    from wtbd_attention import register_attention_modules, ATTENTION_TYPES, C2fAttention
    init_seeds(seed, deterministic=True)
    register_standard_modules()
    base = YOLO(str(ROOT / 'configs/yolov8n_wtbd.yaml')).load(str(core.WEIGHTS))
    common = {k: v.detach().clone() for k, v in base.model.state_dict().items()}
    register_attention_modules()
    # Attention construction must not depend on another candidate's parameter count.
    init_seeds(seed + 100000, deterministic=True)
    model = YOLO(str(ROOT / 'configs' / METHODS[method][1])).load(str(core.WEIGHTS))
    target = model.model.state_dict()
    assert all(k in target and target[k].shape == v.shape for k, v in common.items())
    loaded = model.model.load_state_dict(common, strict=False)
    assert not loaded.unexpected_keys
    assert loaded.missing_keys and all('.feature_block.' in k for k in loaded.missing_keys)
    assert state_hash({k: model.model.state_dict()[k] for k in common}) == state_hash(common)
    enabled = [i for i, m in enumerate(model.model.model)
               if isinstance(m, C2fAttention) and m.attention_name != 'none']
    assert enabled == [15, 18], enabled
    assert sum(isinstance(m, ATTENTION_TYPES[method]) for m in model.model.modules()) == 2
    metadata = {'method': method, 'seed': seed, 'common_state_sha256': state_hash(common),
                'common_state_tensors': len(common), 'full_state_sha256': state_hash(model.model.state_dict()),
                'new_state_keys': loaded.missing_keys, 'attention_layers': enabled,
                'parameters': sum(p.numel() for p in model.model.parameters()),
                'pretrained_weights_sha256': sha(core.WEIGHTS),
                'shared_native_path_including_six_class_head': True,
                'attention_rng_seed': seed + 100000,
                'attention_batchnorm': {n: {'eps': m.eps, 'momentum': m.momentum}
                    for n,m in model.model.named_modules()
                    if '.feature_block.' in n and isinstance(m,torch.nn.BatchNorm2d)},
                'batchnorm_policy': 'Ultralytics global model initialization applies eps=.001/momentum=.03 to all BatchNorm2d, including canonical attention blocks',
                'initial_forward_identical_to_baseline': False}
    if callbacks:
        def audit_start(trainer):
            actual = trainer.model.state_dict()
            # Verify initialization of the backbone and attention parameters.
            assert state_hash(actual) == metadata['full_state_sha256'], 'Model changed before training'
            init_seeds(seed, deterministic=True)
            original_preprocess = trainer.preprocess_batch
            seen = False
            def audited_preprocess(batch):
                nonlocal seen
                if not seen:
                    fields = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    item = {'method': method, 'seed': seed, 'tensor_sha256': state_hash(fields),
                            'tensor_shapes': {k: list(v.shape) for k, v in fields.items()},
                            'image_files': list(batch.get('im_file', [])), 'checked_at': now(),
                            'scope': 'first augmented batch only, not all training batches'}
                    for other in METHODS:
                        p = record_dir(other, seed) / 'first_batch.json'
                        if p.exists():
                            previous = json.loads(p.read_text(encoding='utf8'))
                            assert previous['tensor_sha256'] == item['tensor_sha256'], f'First batch mismatch: {other}'
                            assert previous['image_files'] == item['image_files']
                    write_json(record_dir(method, seed) / 'first_batch.json', item)
                    seen = True
                return original_preprocess(batch)
            trainer.preprocess_batch = audited_preprocess
        model.add_callback('on_train_start', audit_start)
    del base
    gc.collect()
    return model, common, metadata

def evaluate(method, seed, run, record, imgsz=768, batch=16, workers=4):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from wtbd_attention import register_attention_modules
    register_attention_modules()
    for ck in ('best', 'last'):
        weight = run / 'weights' / f'{ck}.pt'
        digest = sha(weight)
        for gt, data in [('annotator_a', 'wtbd_gt_primary.yaml'), ('annotator_b', 'wtbd_gt_second.yaml')]:
            dest = record / f'{ck}_{gt}.json'
            if dest.exists():
                assert json.loads(dest.read_text(encoding='utf8'))['weight_sha256'] == digest
                continue
            model = YOLO(str(weight))
            start = time.perf_counter()
            metrics = model.val(data=str(ROOT / 'data' / data), split='test', imgsz=imgsz,
                batch=batch, workers=workers, device=0, half=False, conf=.001, iou=.7,
                max_det=300, plots=True, save_txt=True, save_conf=True, save_json=False,
                verbose=False, project=str(record / 'validation'), name=f'{ck}_{gt}', exist_ok=True)
            row = {'method': method, 'label': label(method), 'cohort': cohort(method),
                   'seed': seed, 'checkpoint': ck, 'ground_truth': gt, 'split': 'test',
                   'map50_95': float(metrics.box.map), 'map50': float(metrics.box.map50),
                   'map75': float(metrics.box.map75), 'precision': float(metrics.box.mp),
                   'recall': float(metrics.box.mr), 'evaluation_wall_seconds': time.perf_counter()-start,
                   'weight_path': str(weight), 'weight_sha256': digest, 'imgsz': imgsz,
                   'batch': batch, 'half': False, 'updated_at': now(),
                   'pr_definition': 'Ultralytics 8.4.37 metric at smoothed mean-F1-optimal evaluation threshold; not a fixed operating threshold',
                   'predictions_format': 'YOLO class xywh-normalized confidence; empty files may be absent',
                   'prediction_directory': str(record / 'validation' / f'{ck}_{gt}' / 'labels')}
            for stage in ('inference', 'preprocess', 'postprocess'):
                row[f'{stage}_ms'] = float(metrics.speed[stage])
            ap = np.asarray(metrics.box.all_ap)
            indices = list(map(int, metrics.box.ap_class_index))
            for i, name in enumerate(core.CLASSES):
                ix = indices.index(i) if i in indices else None
                row[f'ap50_{name}'] = float(ap[ix, 0]) if ix is not None else None
                row[f'ap75_{name}'] = float(ap[ix, 5]) if ix is not None else None
                row[f'ap50_95_{name}'] = float(ap[ix].mean()) if ix is not None else None
                counts = getattr(metrics, 'nt_per_class', None)
                row[f'targets_{name}'] = int(counts[i]) if counts is not None else None
            for j in range(10):
                row[f'ap_iou_{50+5*j}'] = float(ap[:, j].mean())
            write_json(dest, row)
            del model
            gc.collect()
            torch.cuda.empty_cache()

def train_one(method, seed, smoke=False):
    record_dir(method,seed).mkdir(parents=True,exist_ok=True)
    try:
        telemetry=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,power.draw,power.limit,memory.used',
            '--format=csv'],capture_output=True,text=True,timeout=15)
        write_json(record_dir(method,seed)/'gpu_before_training.json',
            {'time':now(),'returncode':telemetry.returncode,'output':telemetry.stdout.strip(),'stderr':telemetry.stderr.strip()})
    except (OSError,subprocess.TimeoutExpired) as exc:
        write_json(record_dir(method,seed)/'gpu_before_training.json',{'time':now(),'unavailable':str(exc)})
    core.train_one(method, seed, smoke)
    p = record_dir(method, seed) / 'training_timing.json'
    item = json.loads(p.read_text(encoding='utf8'))
    item['cohort'] = cohort(method)
    item['first_batch_checked'] = (p.parent / 'first_batch.json').exists()
    write_json(p, item)

def summarize():
    raw, times, per, summary = [], [], [], []
    for method in ALL:
        for seed in SEEDS:
            rec = record_dir(method, seed)
            timing = rec / 'training_timing.json'
            if timing.exists():
                item = json.loads(timing.read_text(encoding='utf8'))
                item['cohort'] = cohort(method)
                times.append(item)
            elif method in HISTORICAL:
                rows = read_csv(source_run(method, seed) / 'results.csv')
                if len(rows) == 200:
                    item = {'method': method, 'seed': seed, 'epochs': 200, 'cohort': cohort(method),
                            'csv_cumulative_seconds': float(rows[-1]['time'])}
                    if method == 'f2_lsk':
                        old = ROOT / 'reports/yolov8_feature_completion_v1/records/f2_lsk' / f'seed_{seed}/training_timing.json'
                        if old.exists():
                            item.update(json.loads(old.read_text(encoding='utf8')))
                            item['cohort'] = cohort(method)
                    times.append(item)
            for ck in ('best', 'last'):
                pair = [json.loads(p.read_text(encoding='utf8')) for gt in ('annotator_a', 'annotator_b')
                        if (p := rec / f'{ck}_{gt}.json').exists()]
                raw.extend(pair)
                if len(pair) == 2:
                    row = {'method': method, 'label': label(method), 'cohort': cohort(method),
                           'seed': seed, 'checkpoint': ck}
                    keys = list(METRICS) + ['inference_ms', 'preprocess_ms', 'postprocess_ms']
                    keys += [k for k in pair[0] if k.startswith(('ap50_', 'ap75_', 'ap_iou_'))]
                    for k in keys:
                        if all(r.get(k) is not None for r in pair):
                            row[k] = statistics.fmean(r[k] for r in pair)
                    per.append(row)
    for method in ALL:
        for ck in ('best', 'last'):
            group = [r for r in per if r['method'] == method and r['checkpoint'] == ck]
            if not group:
                continue
            row = {'method': method, 'label': label(method), 'cohort': cohort(method),
                   'checkpoint': ck, 'n_seeds': len(group)}
            for k in METRICS + ('inference_ms',):
                vals = [r[k] for r in group]
                row[f'mean_{k}'] = statistics.fmean(vals)
                row[f'std_{k}'] = statistics.stdev(vals) if len(vals) > 1 else None
            for k in ('csv_cumulative_seconds', 'training_call_wall_seconds', 'peak_cuda_allocated_bytes'):
                vals = [r[k] for r in times if r['method'] == method and r.get(k) is not None]
                row[f'n_{k}'] = len(vals)
                row[f'mean_{k}'] = statistics.fmean(vals) if vals else None
            summary.append(row)
    for name, rows in [('evaluation_raw.csv', raw), ('evaluation_mean_ab_per_seed.csv', per),
                       ('five_seed_summary.csv', summary), ('training_timing.csv', times)]:
        write_csv(REPORT / name, rows)
    lines = ['# YOLOv8注意力对照：实时结果', '', f'更新：{now()}', '',
             '以下仅为已完成的读数，n不足5时不是最终结果。A/B先在种子内平均。', '',
             '| 模块 | 批次 | 检查点 | n | AP50–95 (%) | AP50 (%) | AP75 (%) | P (%) | R (%) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        sd = row['std_map50_95']
        ap = f"{100*row['mean_map50_95']:.2f}" + (f' ± {100*sd:.2f}' if sd is not None else '')
        vals = ' | '.join(f"{100*row['mean_'+k]:.2f}" for k in ('map50', 'map75', 'precision', 'recall'))
        lines.append(f"| {row['label']} | {row['cohort']} | {row['checkpoint']} | {row['n_seeds']} | {ap} | {vals} |")
    lines += ['', '历史组与新增组存在批次差异；排名与平均差不是统计显著性。主要终点预定为last AP50–95，best同时完整报告。',
              '训练时间见training_timing.csv；严格batch1前向基准在全部训练和复评之后执行，见latency_summary.csv。',
              '预测框、PR曲线、逐类别AP、AP75、各IoU AP和权重哈希均保留在records与CSV。']
    (REPORT / '实时结果.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    return len([r for r in times if r['method'] in METHODS]), len(raw), len(per)

def code_files():
    files = [Path(__file__), Path(core.__file__), ROOT/'scripts/wtbd_attention.py',
             ROOT/'scripts/wtbd_attention_pipeline_timing.py',
             ROOT/'scripts/wtbd_modules.py', ROOT/'scripts/wtbd_dual.py',
             ROOT/'scripts/run_wtbd_yolov8_ab_s2_f1.py', core.WEIGHTS, core.AUDIT,
             ROOT/'configs/yolov8n_wtbd.yaml']
    files += sorted(p for p in (ROOT/'scripts/wtbd_attention_vendor').iterdir() if p.is_file())
    files += [ROOT/'configs'/v[1] for v in METHODS.values()]
    files += [ROOT/'data'/n for n in ('wtbd_fused.yaml','wtbd_gt_primary.yaml','wtbd_gt_second.yaml')]
    return files

def fingerprints():
    return {str(p): sha(p) for p in code_files()}

def preflight():
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    from wtbd_attention import make_attention
    result = {'passed': False, 'created_at': now(), 'checks': [], 'sha256': fingerprints()}
    for name in METHODS:
        # Shape, all-parameter gradients and spatially non-square CA handling.
        for channels in (64, 128):
            module = make_attention(name, channels).cuda().train()
            x = torch.randn(2, channels, 19, 23, device='cuda', requires_grad=True)
            y = module(x)
            assert y.shape == x.shape and torch.isfinite(y).all()
            y.square().mean().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
            assert any(p.grad.abs().sum().item() > 0 for p in module.parameters())
            del module, x, y
        result['checks'].append(f'{name}: shape and finite gradients at 64/128 channels')
    common_hashes = {}
    complexities = []
    for seed in SEEDS:
        for method in METHODS:
            model, common, meta = build(method, seed, callbacks=False)
            if seed in common_hashes:
                assert meta['common_state_sha256'] == common_hashes[seed]
            common_hashes[seed] = meta['common_state_sha256']
            if seed == 42:
                feature_shapes = {}
                handles = [model.model.model[i].register_forward_hook(
                    lambda m, args, out, i=i: feature_shapes.update({str(i): list(out.shape)})) for i in (15,18)]
                model.model.eval().cuda()
                with torch.inference_mode():
                    pred = model.model(torch.zeros(1,3,768,768,device='cuda'))[0]
                assert list(pred.shape) == [1,10,12096], pred.shape
                assert feature_shapes == {'15':[1,64,96,96], '18':[1,128,48,48]}
                for handle in handles:
                    handle.remove()
                model.model.cpu()
                out = REPORT/'preflight_checkpoints'/f'{method}.pt'
                out.parent.mkdir(parents=True,exist_ok=True)
                model.save(str(out))
                loaded = YOLO(str(out))
                # Ultralytics.save stores FP16; compare to the same explicit conversion.
                expected = {k:v.half().float() if v.is_floating_point() else v for k,v in model.model.state_dict().items()}
                actual = loaded.model.float().state_dict()
                assert state_hash(actual) == state_hash(expected)
                complexities.append({'method': method, 'parameters': meta['parameters'],
                    'gflops_estimate': get_flops(model.model,imgsz=768), 'input':768,
                    'flops_note':'Ultralytics/THOP estimate (2 FLOPs/MAC); elementwise attention operations may be uncounted',
                    'feature_shapes':feature_shapes})
                del loaded, pred
            del model, common
            gc.collect()
            torch.cuda.empty_cache()
    result['checks'] += ['native initialization identical across three methods for all five seeds',
                         'exactly two attention blocks at layers15/18', '768 forward shapes and checkpoint roundtrip']
    result['common_state_hashes'] = common_hashes
    result['passed'] = True
    write_json(REPORT/'preflight.json',result)
    write_csv(REPORT/'model_complexity.csv',complexities)
    print(json.dumps(result,ensure_ascii=False,indent=2))

def freeze():
    import torch, ultralytics
    from run_wtbd_yolov8_ab_s2_f1 import train_args
    current = fingerprints()
    path = REPORT/'protocol.json'
    if path.exists():
        obj = json.loads(path.read_text(encoding='utf8'))
        assert obj['sha256'] == current, 'Frozen inputs changed'
        return obj
    qa = json.loads((REPORT/'preflight.json').read_text(encoding='utf8'))
    assert qa['passed'] and qa['sha256'] == current
    old = json.loads((ROOT/'reports/yolov8_feature_completion_v1/protocol.json').read_text(encoding='utf8'))
    assert sha(core.WEIGHTS) == old['sha256'][str(core.WEIGHTS)]
    assert sha(core.AUDIT) == old['sha256'][str(core.AUDIT)]
    assert torch.__version__ == old['torch'] and ultralytics.__version__ == old['ultralytics']
    jobs = []
    for seed in SEEDS:
        candidates = list(METHODS)
        random.Random(20260918+seed).shuffle(candidates)
        jobs.extend([m, seed, 'train-one'] for m in candidates)
    jobs += [[m,s,'eval-one'] for s in SEEDS for m in HISTORICAL]
    obj = {'study_id':'WTBD-YOLOV8-ATTENTION-COMPARISON-V1', 'created_at':now(),
           'methods':METHODS, 'historical_controls':HISTORICAL, 'seeds':SEEDS,
           'new_training_runs':15, 'jobs':jobs, 'sha256':current,
           'train_args':train_args(42,argparse.Namespace(epochs=200,imgsz=768,batch=16,workers=4),{'data':'wtbd_fused.yaml'}),
           'primary_endpoint':'last (epoch200) test mAP50:95, mean of A/B within seed',
           'secondary_endpoint':'best selected by unchanged fused-validation fitness; AP50, AP75, P, R, class AP',
           'initialization':'new three share complete native path; new-module RNG seed+100000; training RNG reset at on_train_start; first augmented batch checked',
           'comparison_scope':'historical controls retained; cross-cohort contrasts descriptive',
           'evaluation':{'batch':16,'imgsz':768,'half':False,'conf':.001,'iou':.7,'max_det':300},
           'benchmark':{'batch':1,'imgsz':768,'fp32':True,'warmup':200,'iterations':1000,'rounds':3,'fuse':False},
           'image_pipeline_benchmark':{'images':32,'sampling_seed':20260918,'conf':.001,'iou':.7,'max_det':300,
               'scope':'preloaded images; preprocessing + unfused forward + NMS; IO and coordinate rescaling excluded'},
           'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
           'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
           'gpu':torch.cuda.get_device_name(0),'upstream':json.loads((ROOT/'scripts/wtbd_attention_vendor/provenance.json').read_text(encoding='utf8'))}
    write_json(path,obj)
    for p in code_files():
        if p != core.WEIGHTS:
            dest = REPORT/'frozen_sources'/p.relative_to(ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest)
    return obj

def benchmark(smoke=False):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    from wtbd_attention import register_attention_modules
    from wtbd_attention_pipeline_timing import load_benchmark_images,time_image_pipeline
    register_attention_modules()
    torch.manual_seed(20260918)
    size, warm, iterations, rounds = (320,2,3,1) if smoke else (768,200,1000,3)
    x = torch.rand(1,3,size,size,device='cuda',dtype=torch.float32)
    image_paths,images=load_benchmark_images(ROOT,smoke)
    write_json(REPORT/'benchmark_images.json',{'images':[{'path':str(p),'sha256':sha(p)} for p in image_paths],
        'selection_seed':20260918,'input':size,'disk_io_excluded':True})
    tasks = [(m,s,c) for m in (list(METHODS) if smoke else ALL)
             for s in ((42,) if smoke else SEEDS) for c in ('best','last')]
    complexities = []
    if (REPORT/'model_complexity.csv').exists():
        complexities = read_csv(REPORT/'model_complexity.csv')
    for r in range(rounds):
        order = tasks.copy()
        random.Random(4100+r).shuffle(order)
        for i,(method,seed,ck) in enumerate(order):
            dest=REPORT/'latency'/f'{method}_seed{seed}_{ck}_round{r+1}.json'
            if dest.exists() and json.loads(dest.read_text(encoding='utf8')).get('schema_version')==2:
                continue
            weight = source_run(method,seed)/'weights'/f'{ck}.pt'
            model = YOLO(str(weight)).model.float().cuda().eval()
            if not any(row['method']==method for row in complexities):
                complexities.append({'method':method,'parameters':sum(p.numel() for p in model.parameters()),
                    'gflops_estimate':get_flops(model,imgsz=size),'input':size,
                    'flops_note':'Ultralytics/THOP estimate, elementwise ops may be uncounted'})
                write_csv(REPORT/'model_complexity.csv',complexities)
            with torch.inference_mode():
                for _ in range(warm):
                    model(x)
                torch.cuda.synchronize()
                samples=[]
                for _ in range(iterations):
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    model(x)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter()-start)*1000)
            pipeline=time_image_pipeline(model,images,size)
            obj={'schema_version':2,'method':method,'seed':seed,'checkpoint':ck,'round':r+1,'batch':1,'imgsz':size,
                 'fp32':True,'fuse':False,'warmup':warm,'iterations':iterations,
                 'mean_ms':float(np.mean(samples)),'median_ms':float(np.median(samples)),
                 'p95_ms':float(np.percentile(samples,95)),'samples_ms':samples,
                 'weight_sha256':sha(weight),'created_at':now(),
                 'image_pipeline':pipeline,
                 'scope':'network forward only; excludes preprocessing, NMS and IO'}
            write_json(dest,obj)
            write_json(REPORT/'benchmark_status.json',{'round':r+1,'rounds':rounds,'index':i+1,
                'per_round':len(order),'method':method,'seed':seed,'checkpoint':ck,'updated_at':now()})
            del model
            gc.collect()
            torch.cuda.empty_cache()
    rows=[]
    for p in sorted((REPORT/'latency').glob('*.json')):
        row=json.loads(p.read_text(encoding='utf8'));row.pop('samples_ms')
        pipeline=row.pop('image_pipeline')
        row.update({f'image_pipeline_{k}':v for k,v in pipeline.items() if k not in ('image_samples','scope')})
        rows.append(row)
    write_csv(REPORT/'latency_raw.csv',rows)
    summary=[]
    for method in (METHODS if smoke else ALL):
        for ck in ('best','last'):
            seed_means=[statistics.fmean(r['mean_ms'] for r in rows
                if r['method']==method and r['seed']==s and r['checkpoint']==ck)
                for s in ((42,) if smoke else SEEDS)]
            avg=statistics.fmean(seed_means)
            summary.append({'method':method,'checkpoint':ck,'n_seeds':len(seed_means),
                'mean_ms':avg,'std_across_seed_mean_ms':statistics.stdev(seed_means) if len(seed_means)>1 else None,
                'fps_from_mean_latency':1000/avg,'batch':1,'imgsz':size,
                'image_pipeline_mean_ms':statistics.fmean(row['image_pipeline_mean_pipeline_ms'] for row in rows
                    if row['method']==method and row['checkpoint']==ck)})
    write_csv(REPORT/'latency_summary.csv',summary)
    write_json(REPORT/'benchmark_status.json',{'status':'complete','updated_at':now(),'records':len(rows)})

def queue(smoke=False):
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=REPORT/'queue.lock'
    with lock.open('x',encoding='utf8') as f:
        f.write(str(os.getpid()))
    execution_state=None
    try:
        if os.name=='nt':
            import ctypes
            # Keep this local training process awake; do not force the display on.
            execution_state=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        frozen = None if smoke else freeze()
        if not smoke:
            smoke_report=ROOT/'smoke/yolov8_attention_comparison_v1/reports'
            smoke_qa=json.loads((smoke_report/'queue_status.json').read_text(encoding='utf8'))
            assert smoke_qa['status']=='complete'
            assert smoke_qa['source_sha256']==frozen['sha256'], 'Smoke tested different source'
        jobs=[[m,42,'train-one'] for m in METHODS] if smoke else frozen['jobs']
        env=os.environ.copy()
        env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',WANDB_DISABLED='true',COMET_MODE='DISABLED',CLEARML_OFFLINE_MODE='1')
        for index,(method,seed,mode) in enumerate(jobs,1):
            if frozen:
                assert fingerprints()==frozen['sha256']
            if shutil.disk_usage(ROOT).free<4*1024**3:
                raise RuntimeError('Less than4GiB free; preserving all results')
            rec=record_dir(method,seed);rec.mkdir(parents=True,exist_ok=True)
            env['PYTHONHASHSEED']=str(seed)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),mode,'--method',method,'--seed',str(seed)]
            if smoke:
                cmd.append('--smoke')
            with (rec/'worker.log').open('a',encoding='utf8') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT.parent))
                write_json(REPORT/'queue_status.json',{'status':'running','updated_at':now(),'queue_pid':os.getpid(),
                    'worker_pid':child.pid,'job_index':index,'total_jobs':len(jobs),'stage':mode,
                    'method':method,'seed':seed,'log':str(rec/'worker.log')})
                code=child.wait()
            if code:
                raise RuntimeError(f'{method} seed{seed} {mode} failed ({code}); see {rec / "worker.log"}')
            summarize()
        if not smoke:
            counts=summarize()
            assert counts==(15,120,60),counts
        write_json(REPORT/'queue_status.json',{'status':'running','stage':'batch1_latency_benchmark','updated_at':now(),'queue_pid':os.getpid()})
        benchmark(smoke)
        write_json(REPORT/'completion_audit.json',{'complete':True,'smoke':smoke,'updated_at':now(),
            'new_training_runs':3 if smoke else 15,'epochs_per_new_run':1 if smoke else 200,
            'counts_training_raw_eval_paired_eval':summarize(),
            'all_new_runs_have_init_and_first_batch_audits':all(
                (record_dir(m,s)/name).exists() for m in METHODS for s in ((42,) if smoke else SEEDS)
                for name in ('initialization_verified.json','first_batch.json'))})
        write_json(REPORT/'queue_status.json',{'status':'complete','updated_at':now(),'jobs':len(jobs),
                   'source_sha256':fingerprints()})
    except BaseException as exc:
        write_json(REPORT/'queue_status.json',{'status':'failed','updated_at':now(),'error':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if execution_state:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)

def configure(smoke=False):
    global REPORT,RUNS,SMOKE
    SMOKE=smoke
    if smoke:
        REPORT=ROOT/'smoke/yolov8_attention_comparison_v1/reports'
        RUNS=ROOT/'smoke/yolov8_attention_comparison_v1/runs'
        # Exercise the real resolution, fixed batch and Windows worker path.
        # Only epoch count / fraction differ in smoke; smoke metrics are excluded.
        core.Namespace=lambda **kw: argparse.Namespace(**dict(kw, imgsz=768, batch=16, workers=4))
    core.REPORT,core.RUNS=REPORT,RUNS
    core.METHODS,core.HISTORICAL=METHODS,HISTORICAL
    core.build,core.evaluate,core.source_run=build,evaluate,source_run
    core.record_dir,core.label=record_dir,label

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','queue','train-one','eval-one','summarize','benchmark','freeze'])
    parser.add_argument('--method',choices=ALL)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=42)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    configure(args.smoke)
    REPORT.mkdir(parents=True,exist_ok=True)
    if args.mode=='preflight':preflight()
    elif args.mode=='freeze':freeze()
    elif args.mode=='queue':queue(args.smoke)
    elif args.mode=='train-one':train_one(args.method,args.seed,args.smoke)
    elif args.mode=='eval-one':evaluate(args.method,args.seed,source_run(args.method,args.seed),record_dir(args.method,args.seed))
    elif args.mode=='benchmark':benchmark(args.smoke)
    else:summarize()

if __name__=='__main__':
    main()
