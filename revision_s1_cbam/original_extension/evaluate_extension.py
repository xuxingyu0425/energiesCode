"""Post-training evaluator and optional wait-and-run watcher for S1+CBAM.

GPU imports and inference occur only after every requested training run completes.
No training, test-based selection, batch fallback, or historical writes occur here.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import ctypes
import gc
import json
import os
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

from analyze_extension import (BASE, WORKSPACE, PAPER, EXP, SEEDS, CLASSES, CACHE,
                               sha, load, save, select_threshold, verify_historical,
                               run_analysis, cache_hash)

EXPECTED = {'A': {'val': (160,219), 'test': (161,233)},
            'B': {'val': (160,214), 'test': (161,229)}}
OPTIONS = dict(imgsz=768, batch=16, workers=0, device=0, conf=.001, iou=.7,
               max_det=300, half=False, plots=False, verbose=False, save_json=False)


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')


def status(phase, **kw):
    save(BASE/'evaluation_status.json', dict(status=phase, time=now(), pid=os.getpid(), **kw))


def directory_digest(directory):
    import hashlib
    h=hashlib.sha256();files=sorted(Path(directory).glob('*'))
    files=[p for p in files if p.is_file()]
    for p in files:
        h.update(p.name.encode('utf-8'));h.update(p.read_bytes())
    return dict(sha256=h.hexdigest(), files=len(files))


def prepare_protocol():
    """Freeze endpoint/selection rules and code before any new test inference."""
    training=load(BASE/'protocol.json')
    required={}
    for ref,gt in [('A','gt_primary'),('B','gt_second')]:
        for split in ['val','test']:
            required[f'{ref}_{split}_labels']=directory_digest(EXP/f'data/{gt}/labels/{split}')
            assert required[f'{ref}_{split}_labels']['files']==EXPECTED[ref][split][0]
        required[ref+'_yaml']=sha(EXP/f'data/wtbd_gt_{"primary" if ref=="A" else "second"}.yaml')
    for split in ['val','test']:
        required[split+'_images_A']=directory_digest(EXP/f'data/gt_primary/images/{split}')
        required[split+'_images_B']=directory_digest(EXP/f'data/gt_second/images/{split}')
        assert required[split+'_images_A']==required[split+'_images_B']
    deps=[EXP/'scripts/evaluate_wtbd_controls.py',PAPER/'figure_sources/analyse_prf.py',
          PAPER/'figure_sources/localization.py',EXP/'scripts/wtbd_modules.py',EXP/'scripts/wtbd_attention.py',
          EXP/'data/split_manifest.csv',PAPER/'source_data/yolov8_reeval_table.csv',
          PAPER/'source_data/locked_per_seed.csv',PAPER/'source_data/localization_v12.csv',
          PAPER/'source_data/cbam_increment_summary_v16.json',PAPER/'source_data/annotation_sensitivity.csv',
          PAPER/'reproducibility/weight_manifest.csv',EXP/'reports/yolov8_attention_comparison_v1/protocol.json']
    inputs={str(p):sha(p) for p in deps}
    frozen=BASE/'evaluation/frozen_dependencies';frozen.mkdir(parents=True,exist_ok=True)
    for p in deps[:5]:
        dest=frozen/p.name
        if dest.exists():assert sha(dest)==sha(p),'Frozen dependency changed: '+str(p)
        else:shutil.copy2(p,dest)
    core=dict(study='Reviewer-requested S1+CBAM extension evaluation',
        cells=['S1_CBAM','S2_CBAM'],seeds=SEEDS,training_protocol_sha256=sha(BASE/'protocol.json'),
        code_sha256={p.name:sha(p) for p in [Path(__file__),BASE/'analyze_extension.py']},
        source_environment=training['source_environment'],
        primary={'checkpoint':'last epoch200','reference':'A','metrics':['AP50-95','common-object IoU'],
                 'comparison':'S2_CBAM minus S1_CBAM; paired seeds'},
        secondary={'checkpoints':['best (original fused-validation selection)'],
                   'references':['B'],'metrics':['AP50','AP75','TP IoU','IoU coverage','validation-locked P/R/F1/FP/FN']},
        native_evaluation=OPTIONS,expected_actual_shape=[800,800],
        threshold_selection={'checkpoint':'last','reference':'A','split':'val','grid':[k/100 for k in range(1,100)],
                             'criterion':'max pooled instance F1','ties':'first/smallest threshold',
                             'freeze_before':'all test inference for this seed'},
        matching='confidence-descending, same-class, one-to-one IoU>=.5',
        localization_confidence=.25,coverage='max same-class IoU>=.5/.75/.85 per reference; not one-to-one',
        CI='two-sided unadjusted paired Student t; seeds42-46, n=5, df=4; fixed WTBD split',
        no_test_selection=True,no_AB_mean_AP=True,no_GPU_while_training=True,
        calibration={'model':'historical S2_CBAM seed42 last','reference':'A test','native_max_abs_tolerance':2e-5,
                     'metrics':['P','R','AP50','AP75','AP'],'expected_images':161,'expected_GT':233,
                     'expected_actual_shape':[800,800],'stop_on_failure':True},
        discrepancy_analysis='exploratory identical/nonidentical/unpaired; XML-verified object mapping; seed intervals',
        expected_counts=EXPECTED,dataset_fingerprints=required,inputs=inputs)
    # Compare JSON-normalized values so tuple counts round-trip as lists.
    core=json.loads(json.dumps(core,ensure_ascii=False))
    plan_path=BASE/'evaluation_protocol.json'
    if plan_path.exists():
        prior=load(plan_path);assert {k:v for k,v in prior.items() if k!='created_at'}==core,'Frozen evaluation protocol changed'
    else:save(plan_path,dict(created_at=now(),**core))
    save(BASE/'evaluation/schema.json',dict(cache='unrounded native validator predictions and A/B references in actual padded800 coordinates',
        native_metrics='native P/R/AP50/AP75/AP fractions; class AP matrix is 6x10 if all classes represented',
        selection='A validation threshold/curve plus weight/cache/protocol SHA-256; persisted before test capture',
        result_scope='one native A/B best/last test evaluation plus A last validation per seed; S2 retained historical rows/caches'))
    return load(plan_path)


def process_alive(pid):
    """Read-only Windows process-state check; never signal or terminate it."""
    if os.name!='nt' or not isinstance(pid,int) or pid<=0:return None
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint,ctypes.c_int,ctypes.c_uint]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint]
    kernel.WaitForSingleObject.restype=ctypes.c_uint
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000,False,pid)  # SYNCHRONIZE, query wait state only
    if not handle:
        return False if ctypes.get_last_error()==87 else None  # nonexistent PID vs access denied
    try:
        state=kernel.WaitForSingleObject(handle,0)
        return True if state==258 else (False if state==0 else None)
    finally:kernel.CloseHandle(handle)


def wait_for_training(poll_seconds=15,max_wait_hours=24):
    """Use no CUDA APIs or GPU imports in the waiting phase."""
    previous=None;started=time.monotonic()
    while True:
        q=load(BASE/'queue_status.json') if (BASE/'queue_status.json').exists() else {'status':'not_started'}
        state=q.get('status')
        if state=='failed':
            status('failed',stage='waiting_for_training',training_status=q)
            raise RuntimeError('Training queue failed; evaluation will not start: '+json.dumps(q))
        if state=='training_complete':return
        alive=process_alive(q.get('pid')) if state=='training' else None
        if alive is False:
            raise RuntimeError('Training queue PID has exited while status remains training; no evaluation started: '+json.dumps(q))
        elapsed=time.monotonic()-started
        if elapsed>max_wait_hours*3600:
            raise TimeoutError(f'Training wait exceeded {max_wait_hours} hours; no evaluation started')
        status('waiting_for_training',training_status=q,no_GPU_imports=True,queue_pid_alive=alive,
               waited_seconds=round(elapsed,1),max_wait_hours=max_wait_hours)
        marker=(state,q.get('seed'))
        if marker!=previous:print('WAITING_FOR_TRAINING',json.dumps(q,ensure_ascii=False),flush=True);previous=marker
        time.sleep(poll_seconds)


def verify_training_complete():
    q=load(BASE/'queue_status.json')
    assert q.get('status')=='training_complete','Refusing GPU evaluation while training queue is active'
    for seed in SEEDS:
        marker=load(BASE/f'records/seed_{seed}/complete.json')
        assert marker['complete'] and marker['epochs']==200 and marker['first_batch_identical'] and marker['full_initialization_identical']
        for checkpoint in ['best','last']:
            p=BASE/f'runs/seed_{seed}/weights/{checkpoint}.pt'
            assert sha(p)==marker[checkpoint+'_sha256'],(seed,checkpoint,'trained weight changed')
        d=pd.read_csv(BASE/f'runs/seed_{seed}/results.csv');assert len(d)==200


def evaluate():
    plan=prepare_protocol();verify_training_complete();verify_historical()
    # No torch/Ultralytics/CUDA calls are made before this guarded point.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    sys.path.insert(0,str(EXP/'scripts'))
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from wtbd_modules import register_wtbd_modules
    from wtbd_attention import register_attention_modules
    env=plan['source_environment']
    assert platform.python_version()==env['python'] and torch.__version__==env['torch'] and ultralytics.__version__==env['ultralytics']
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)==env['gpu']
    register_wtbd_modules();register_attention_modules()
    class CaptureValidator(DetectionValidator):
        def init_metrics(self,model):
            super().init_metrics(model);self.captured=[]
        def update_metrics(self,preds,batch):
            super().update_metrics(preds,batch)
            for si,pred in enumerate(preds):
                b=self._prepare_batch(si,batch);p=self._prepare_pred(pred)
                self.captured.append(dict(image=Path(b['im_file']).name,imgsz=list(b['imgsz']),
                    gt_boxes=b['bboxes'].detach().cpu().tolist(),gt_cls=b['cls'].detach().cpu().tolist(),
                    pred_boxes=p['bboxes'].detach().cpu().tolist(),pred_cls=p['cls'].detach().cpu().tolist(),
                    conf=p['conf'].detach().cpu().tolist()))
    CACHE.mkdir(parents=True,exist_ok=True)
    plan_sha=sha(BASE/'evaluation_protocol.json');raw=[]

    def calibrate():
        weight=EXP/'runs_yolov8_attention_comparison_v1/cbam/seed_42/weights/last.pt'
        oldpath=PAPER/'source_data/locked/S2_CBAM_42_test.json';oldcache=load(oldpath)
        assert sha(weight)==cache_hash(oldcache)
        destination=BASE/'evaluation/calibration_S2_CBAM_42_last_A_test.json'
        key=dict(weight_sha256=sha(weight),evaluation_protocol_sha256=plan_sha,legacy_cache_sha256=sha(oldpath))
        if destination.exists():
            prior=load(destination);assert prior['key']==key and prior['passed'],'Existing calibration did not pass'
            return
        holder=[]
        def factory(*args,**kwargs):
            validator=CaptureValidator(*args,**kwargs);holder.append(validator);return validator
        status('calibrating',cell='S2_CBAM',seed=42,checkpoint='last',reference='A')
        model=YOLO(str(weight))
        result=model.val(validator=factory,data=str(EXP/'data/wtbd_gt_primary.yaml'),split='test',
                         project=str(BASE/'evaluation/native_runs'),name='calibration_S2_CBAM_42_last_A_test',
                         exist_ok=True,**OPTIONS)
        records=holder[-1].captured;box=result.box
        actual=dict(P=float(box.mp),R=float(box.mr),AP50=float(box.map50),AP75=float(box.map75),AP=float(box.map))
        old=pd.read_csv(PAPER/'source_data/yolov8_reeval_table.csv')
        row=old[(old.cell=='S2_CBAM')&(old.seed==42)&old.ck.eq('last')&old['gt'].eq('annotator_a')].iloc[0]
        errors={m:abs(actual[m]-float(row[m])/100) for m in actual}
        count_ok=(len(records)==161 and len({r['image'] for r in records})==161 and
                  sum(len(r['gt_cls']) for r in records)==233 and all(r['imgsz']==[800,800] for r in records))
        passed=count_ok and max(errors.values())<2e-5
        save(destination,dict(key=key,passed=passed,count_and_shape_ok=count_ok,native_metrics=actual,
                              errors=errors,max_absolute_error=max(errors.values()),tolerance=2e-5,records=records))
        del model,result;holder.clear();gc.collect();torch.cuda.empty_cache()
        assert passed,'GPU calibration failed; preserve calibration cache and stop: '+str(errors)
        print('HISTORICAL_S2_GPU_CALIBRATION_PASSED',json.dumps(errors),flush=True)

    calibrate()
    def capture(seed,checkpoint,reference,split):
        weight=BASE/f'runs/seed_{seed}/weights/{checkpoint}.pt'
        dest=CACHE/f'S1_CBAM_{seed}_{checkpoint}_{reference}_{split}.json'
        key=dict(cell='S1_CBAM',seed=seed,checkpoint=checkpoint,reference=reference,split=split,
                 weight_sha256=sha(weight),evaluation_protocol_sha256=plan_sha,
                 labels_sha256=plan['dataset_fingerprints'][f'{reference}_{split}_labels']['sha256'])
        if dest.exists():
            cached=load(dest);assert cached['key']==key,'Cache mismatch; preserve and investigate '+str(dest)
            return cached
        holder=[]
        def factory(*args,**kwargs):
            validator=CaptureValidator(*args,**kwargs);holder.append(validator);return validator
        status('evaluating',seed=seed,checkpoint=checkpoint,reference=reference,split=split)
        model=YOLO(str(weight))
        result=model.val(validator=factory,data=str(EXP/f'data/wtbd_gt_{"primary" if reference=="A" else "second"}.yaml'),
                         split=split,project=str(BASE/'evaluation/native_runs'),
                         name=f'S1_CBAM_{seed}_{checkpoint}_{reference}_{split}',exist_ok=True,**OPTIONS)
        records=holder[-1].captured;expected_n,expected_gt=EXPECTED[reference][split]
        assert len(records)==expected_n and len({r['image'] for r in records})==expected_n
        assert sum(len(r['gt_cls']) for r in records)==expected_gt
        assert all(r['imgsz']==[800,800] for r in records),'Actual evaluation shape changed'
        box=result.box
        native=dict(P=float(box.mp),R=float(box.mr),AP50=float(box.map50),AP75=float(box.map75),AP=float(box.map),
                    all_ap=np.asarray(box.all_ap).tolist(),ap_class_index=[int(i) for i in box.ap_class_index])
        cached=dict(key=key,weight=str(weight),weight_sha256=key['weight_sha256'],cell='S1_CBAM',seed=seed,
                    checkpoint=checkpoint,reference=reference,split=split,native_metrics=native,records=records)
        save(dest,cached);print('CACHE_COMPLETE',dest.name,json.dumps({k:native[k] for k in ['AP50','AP75','AP']}),flush=True)
        del model,result;holder.clear();gc.collect();torch.cuda.empty_cache()
        return cached
    for seed in SEEDS:
        val=capture(seed,'last','A','val')
        selected,curve=select_threshold(val['records'])
        selection=dict(cell='S1_CBAM',seed=seed,checkpoint='last',reference='A',threshold=selected['threshold'],
                       validation=selected,validation_curve=curve,weight_sha256=cache_hash(val),
                       validation_cache_sha256=sha(CACHE/f'S1_CBAM_{seed}_last_A_val.json'),
                       evaluation_protocol_sha256=plan_sha,selection_uses_test=False)
        selection_path=BASE/f'evaluation/selections/S1_CBAM_{seed}_last_A.json'
        if selection_path.exists():assert load(selection_path)==selection,'Frozen validation threshold changed'
        else:save(selection_path,selection)
        print('THRESHOLD_FROZEN',seed,selected['threshold'],flush=True)
        # The persisted validation decision above precedes the first test call.
        for checkpoint in ['last','best']:
            for reference in ['A','B']:
                cache=capture(seed,checkpoint,reference,'test')
                native=cache['native_metrics']
                raw.append(dict(cell='S1_CBAM',seed=seed,checkpoint=checkpoint,reference=reference,
                                **{k:100*native[k] for k in ['P','R','AP50','AP75','AP']},
                                actual_shape='800x800',weight_sha256=cache['weight_sha256']))
                pd.DataFrame(raw).to_csv(BASE/'evaluation/native_test_raw.csv',index=False)
    status('analyzing',new_native_evaluations=25,historical_calibration_evaluations=1)
    run_analysis()
    status('complete',completion_file=str(BASE/'extension_complete.json'),new_native_evaluations=25,historical_calibration_evaluations=1)
    print('S1_CBAM_EXTENSION_EVALUATION_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare-only','evaluate','wait-and-run'])
    parser.add_argument('--poll-seconds',type=int,default=15)
    parser.add_argument('--max-wait-hours',type=float,default=24);args=parser.parse_args()
    if not 1<=args.poll_seconds<=60:raise ValueError('poll seconds must be 1..60')
    if not 0<args.max_wait_hours<=168:raise ValueError('max wait hours must be >0 and <=168')
    keep_awake=False
    try:
        if args.stage=='prepare-only':
            prepare_protocol();verify_historical();print('EVALUATION_PROTOCOL_PREPARED_CPU_ONLY',flush=True);return
        if os.name=='nt':
            # Keep the system awake, without keeping the display on or changing a power plan.
            fn=ctypes.windll.kernel32.SetThreadExecutionState
            fn.argtypes=[ctypes.c_uint];fn.restype=ctypes.c_uint
            previous=fn(0x80000000|0x00000001)
            if not previous:raise OSError('Could not set temporary system-required execution state')
            keep_awake=True
        prepare_protocol()
        if args.stage=='wait-and-run':wait_for_training(args.poll_seconds,args.max_wait_hours)
        evaluate()
    except Exception:
        error=traceback.format_exc();status('failed',stage=args.stage,error=error)
        print(error,flush=True);raise
    finally:
        if keep_awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
