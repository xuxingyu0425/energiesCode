"""Isolated YOLOv8 S2 F2--F5 completion: serial training, timing and evaluation."""
from __future__ import annotations
import argparse,csv,gc,hashlib,json,os,platform,shutil,statistics,subprocess,sys,time,traceback
from datetime import datetime
from pathlib import Path
from argparse import Namespace

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/yolov8_feature_completion_v1'
RUNS=ROOT/'runs_yolov8_feature_completion_v1'
OLD=ROOT/'runs_yolov8_ab_s2_f1_v1'
SEEDS=(42,43,44,45,46)
METHODS={
    'f2_lsk':('LSK','yolov8n_wtbd_lsk.yaml',(0,2,0)),
    'f3_dysample':('DySample','yolov8n_wtbd_dysample.yaml',(0,0,2)),
    'f4_esb_dysample':('ESB+DySample','yolov8n_wtbd_esb_dysample.yaml',(2,0,2)),
    'f5_lsk_dysample':('LSK+DySample','yolov8n_wtbd_lsk_dysample.yaml',(0,2,2)),
}
HISTORICAL={'f0_baseline':('No module','v8_s2_f0'),'f1_esb':('ESB','v8_s2_f1')}
WEIGHTS=ROOT/'weights/yolov8n.pt'
AUDIT=ROOT/'data/dual_annotation_audit.csv'
CLASSES=('craze','crack','corrosion','hide_craze','surface_injure','thunderstrike')

def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)
def read_csv(p):
    if not p.exists():return []
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    if not rows:return
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(p)
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def label(method):return (METHODS if method in METHODS else HISTORICAL)[method][0]
def source_run(method,seed):
    return RUNS/method/f'seed_{seed}' if method in METHODS else OLD/HISTORICAL[method][1]/f'seed_{seed}'
def record_dir(method,seed):return REPORT/'records'/method/f'seed_{seed}'

def build(method,seed):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    from wtbd_modules import register_standard_modules,register_wtbd_modules
    init_seeds(seed,deterministic=True)
    register_standard_modules()
    base=YOLO(str(ROOT/'configs/yolov8n_wtbd.yaml')).load(str(WEIGHTS))
    common={k:v.detach().clone() for k,v in base.model.state_dict().items()}
    register_wtbd_modules()
    # Mark the wrapper as checkpoint-loaded so YOLO.train passes this initialized
    # model to the custom trainer instead of constructing a fresh random model.
    model=YOLO(str(ROOT/'configs'/METHODS[method][1])).load(str(WEIGHTS))
    target=model.model.state_dict()
    assert all(k in target and target[k].shape==v.shape for k,v in common.items())
    loaded=model.model.load_state_dict(common,strict=False)
    assert not loaded.unexpected_keys
    actual={k:model.model.state_dict()[k] for k in common}
    assert state_hash(actual)==state_hash(common)
    metadata={'seed':seed,'method':method,'common_state_sha256':state_hash(common),
              'common_state_tensors':len(common),'new_state_keys':loaded.missing_keys,
              'parameters':sum(p.numel() for p in model.model.parameters()),
              'pretrained_weights_sha256':sha(WEIGHTS),'seeded_before_construction':True,
              'shared_native_path_including_six_class_head':True}
    del base;gc.collect()
    return model,common,metadata

def evaluate(method,seed,run,record,imgsz=768,batch=16,workers=4):
    import torch
    from ultralytics import YOLO
    from wtbd_modules import register_wtbd_modules
    register_wtbd_modules()
    for checkpoint in ('best','last'):
        weight=run/'weights'/f'{checkpoint}.pt'
        if not weight.exists():raise FileNotFoundError(weight)
        digest=sha(weight);model=None
        for gt,file in [('annotator_a','wtbd_gt_primary.yaml'),('annotator_b','wtbd_gt_second.yaml')]:
            dest=record/f'{checkpoint}_{gt}.json'
            if dest.exists():
                old=json.loads(dest.read_text(encoding='utf8'))
                assert old['weight_sha256']==digest
                continue
            if model is None:model=YOLO(str(weight))
            start=time.perf_counter()
            metrics=model.val(data=str(ROOT/'data'/file),split='test',imgsz=imgsz,batch=batch,
                 workers=workers,device=0,half=False,conf=.001,iou=.7,max_det=300,
                 plots=False,save_json=False,verbose=False,project=str(record/'validation'),
                 name=f'{checkpoint}_{gt}',exist_ok=True)
            row={'method':method,'label':label(method),'seed':seed,'checkpoint':checkpoint,
                 'ground_truth':gt,'split':'test','map50_95':float(metrics.box.map),
                 'map50':float(metrics.box.map50),'precision':float(metrics.box.mp),
                 'recall':float(metrics.box.mr),'inference_ms':float(metrics.speed['inference']),
                 'preprocess_ms':float(metrics.speed['preprocess']),
                 'postprocess_ms':float(metrics.speed['postprocess']),
                 'evaluation_wall_seconds':time.perf_counter()-start,'weight_path':str(weight),
                 'weight_sha256':digest,'imgsz':imgsz,'batch':batch,'half':False,'updated_at':now()}
            row.update({f'ap_{c}':float(metrics.box.maps[i]) for i,c in enumerate(CLASSES)})
            write_json(dest,row)
        del model;gc.collect();torch.cuda.empty_cache()

def train_one(method,seed,smoke=False):
    import torch
    from wtbd_dual import make_dual_trainer
    from run_wtbd_yolov8_ab_s2_f1 import train_args
    record=record_dir(method,seed);record.mkdir(parents=True,exist_ok=True)
    run=source_run(method,seed);epochs=1 if smoke else 200
    if (record/'training_timing.json').exists():
        old=json.loads((record/'training_timing.json').read_text(encoding='utf8'))
        if old.get('complete'):
            assert len(read_csv(run/'results.csv'))==epochs
            evaluate(method,seed,run,record,320 if smoke else 768,4 if smoke else 16,0 if smoke else 4);return
    if (run/'results.csv').exists() or (run/'weights/last.pt').exists():
        raise RuntimeError(f'Partial run preserved; explicit recovery needed: {run}')
    start=time.perf_counter();started=now()
    model,common,meta=build(method,seed)
    write_json(record/'initialization.json',meta)
    epoch_rows=[];clock={}
    def on_start(t):
        actual=t.model.state_dict()
        assert state_hash({k:actual[k] for k in common})==meta['common_state_sha256']
        write_json(record/'initialization_verified.json',{'verified':True,'time':now(),'sha256':meta['common_state_sha256']})
        torch.cuda.reset_peak_memory_stats()
    def epoch_start(t):
        torch.cuda.synchronize();clock['start']=time.perf_counter()
    def epoch_end(t):
        if epoch_rows and epoch_rows[-1]['epoch']==t.epoch+1:return
        torch.cuda.synchronize()
        epoch_rows.append({'epoch':t.epoch+1,'train_plus_validation_seconds':time.perf_counter()-clock['start']})
        write_csv(record/'epoch_timing.csv',epoch_rows)
    model.add_callback('on_train_start',on_start)
    model.add_callback('on_train_epoch_start',epoch_start)
    model.add_callback('on_fit_epoch_end',epoch_end)
    args=Namespace(epochs=epochs,imgsz=320 if smoke else 768,batch=4 if smoke else 16,workers=0 if smoke else 4)
    kwargs=train_args(seed,args,{'data':'wtbd_fused.yaml'})
    class StrictDualTrainer(make_dual_trainer('dual_target',AUDIT)):
        def _build_train_pipeline(self):
            if self.args.batch!=args.batch or self.batch_size!=args.batch:
                raise RuntimeError('Fixed batch protocol: refusing automatic OOM batch reduction')
            return super()._build_train_pipeline()
        def _handle_nan_recovery(self,epoch):
            if self.loss is not None and not torch.isfinite(self.loss).all():
                raise RuntimeError(f'Non-finite loss at epoch {epoch+1}; automatic rollback disabled')
            if self.fitness is not None and not __import__('math').isfinite(float(self.fitness)):
                raise RuntimeError(f'Non-finite fitness at epoch {epoch+1}; automatic rollback disabled')
            return False
    kwargs.update(project=str(run.parent),name=run.name,exist_ok=True,trainer=StrictDualTrainer)
    if smoke:kwargs.update(fraction=.05,cache=False,save_period=-1,close_mosaic=0)
    write_json(record/'train_arguments.json',{k:v for k,v in kwargs.items() if k!='trainer'})
    call=time.perf_counter();model.train(**kwargs);end=time.perf_counter()
    rows=read_csv(run/'results.csv')
    assert len(rows)==epochs and all((run/'weights'/f'{c}.pt').exists() for c in ('best','last'))
    for row in rows:
        for k,v in row.items():
            if 'loss' in k:assert __import__('math').isfinite(float(v)),(k,v)
    timing={'method':method,'seed':seed,'complete':True,'started_at':started,'ended_at':now(),
            'epochs':epochs,'training_call_wall_seconds':end-call,'worker_setup_plus_training_seconds':end-start,
            'csv_cumulative_seconds':float(rows[-1]['time']),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
            'parameters':meta['parameters'],'run_dir':str(run),'cohort':'new_20260918'}
    write_json(record/'training_timing.json',timing)
    del model,common;gc.collect();torch.cuda.empty_cache()
    evaluate(method,seed,run,record,args.imgsz,args.batch,args.workers)

def summarize():
    raw=[];times=[];per=[];summ=[]
    for method in list(HISTORICAL)+list(METHODS):
        for seed in SEEDS:
            rec=record_dir(method,seed);t=rec/'training_timing.json'
            if t.exists():times.append(json.loads(t.read_text(encoding='utf8')))
            elif method in HISTORICAL:
                hist=read_csv(source_run(method,seed)/'results.csv')
                if len(hist)==200:times.append({'method':method,'seed':seed,'complete':True,'epochs':200,
                    'csv_cumulative_seconds':float(hist[-1]['time']),'cohort':'historical_training',
                    'training_call_wall_seconds':None,'worker_setup_plus_training_seconds':None})
            for ck in ('best','last'):
                pair=[]
                for gt in ('annotator_a','annotator_b'):
                    p=rec/f'{ck}_{gt}.json'
                    if p.exists():pair.append(json.loads(p.read_text(encoding='utf8')))
                raw.extend(pair)
                if len(pair)==2:
                    row={'method':method,'label':label(method),'seed':seed,'checkpoint':ck}
                    for key in ('map50_95','map50','precision','recall','inference_ms','preprocess_ms','postprocess_ms'):
                        row[key]=statistics.fmean(r[key] for r in pair)
                    per.append(row)
    for method in list(HISTORICAL)+list(METHODS):
        for ck in ('best','last'):
            group=[r for r in per if r['method']==method and r['checkpoint']==ck]
            if not group:continue
            row={'method':method,'label':label(method),'checkpoint':ck,'n_seeds':len(group)}
            for key in ('map50_95','map50','precision','recall','inference_ms'):
                vals=[r[key] for r in group];row[f'mean_{key}']=statistics.fmean(vals)
                row[f'std_{key}']=statistics.stdev(vals) if len(vals)>1 else None
            tt=[r for r in times if r['method']==method]
            for key in ('csv_cumulative_seconds','training_call_wall_seconds'):
                vals=[r[key] for r in tt if r.get(key) is not None]
                row[f'n_{key}']=len(vals)
                row[f'mean_{key}']=statistics.fmean(vals) if vals else None
                row[f'std_{key}']=statistics.stdev(vals) if len(vals)>1 else None
            summ.append(row)
    for filename,rows in [('evaluation_raw.csv',raw),('evaluation_mean_ab_per_seed.csv',per),
                          ('training_timing.csv',times),('five_seed_summary.csv',summ)]:write_csv(REPORT/filename,rows)
    md=['# YOLOv8模块补充实验：实时结果','',f'更新时间：{now()}','',
        'AP为A/B测试参照均值；下表只列已完成的评价，n不足5时不是最终五种子结论。', '',
        '| 模块 | 检查点 | n | AP50:95(%) | 推理(ms/图) | 训练CSV累计(min) |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in summ:
        tm=r['mean_csv_cumulative_seconds'];tm=f'{tm/60:.2f}' if tm is not None else '—'
        md.append(f"| {r['label']} | {r['checkpoint']} | {r['n_seeds']} | {r['mean_map50_95']*100:.2f} | {r['mean_inference_ms']:.2f} | {tm} |")
    md+=['','新增四组共享同种子的原生路径初始化。F0/F1保留历史训练，推理在本次统一设置复评；历史与新增训练耗时存在收集批次差异。',
         'CSV时间含训练循环与验证；训练调用总耗时还含准备及结束评价。推理为评价器记录的模型阶段时间，不含预处理和后处理。',
         '完整逐种子结果、样本标准差、计时与权重哈希分别见同目录CSV及records。均值高低不自动等同于统计显著。']
    (REPORT/'实时结果.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    return len([t for t in times if t['method'] in METHODS]),len(per)

def freeze():
    import torch,ultralytics
    from run_wtbd_yolov8_ab_s2_f1 import train_args
    files=[Path(__file__),ROOT/'scripts/wtbd_modules.py',ROOT/'scripts/wtbd_dual.py',
           ROOT/'scripts/run_wtbd_yolov8_ab_s2_f1.py',WEIGHTS,AUDIT,
           ROOT/'configs/yolov8n_wtbd.yaml']+[ROOT/'configs'/m[1] for m in METHODS.values()]
    files += [ROOT/'data'/n for n in ('wtbd_fused.yaml','wtbd_gt_primary.yaml','wtbd_gt_second.yaml')]
    fingerprints={str(f):sha(f) for f in files}
    p=REPORT/'protocol.json'
    if p.exists():
        frozen=json.loads(p.read_text(encoding='utf8'))
        assert frozen['sha256']==fingerprints,'Frozen inputs changed; do not mix protocols'
        return frozen
    old=json.loads((ROOT/'reports/yolov8_ab_s2_f1_v1/experiment_manifest.json').read_text(encoding='utf8'))
    assert sha(AUDIT)==old['dataset_audit_sha256'] and sha(WEIGHTS)==old['pretrained_weights_sha256']
    frozen={'study_id':'WTBD-YOLOV8-FEATURE-COMPLETION-V1','created_at':now(),'methods':METHODS,
       'seeds':SEEDS,'new_training_runs':20,'epochs':200,'imgsz':768,'batch':16,'workers':4,
       'train_args':train_args(42,Namespace(epochs=200,imgsz=768,batch=16,workers=4),{'data':'wtbd_fused.yaml'}),
       'initialization':'New four methods share complete native path (including nc=6 head) within seed; historical F0/F1 not claimed identical.',
       'historical_baselines':HISTORICAL,'evaluation':'best and epoch200; test A/B independently; mean AP50:95; FP32 conf=.001 IoU=.7 max_det=300',
       'timing':'CSV cumulative; training call wall; setup+training wall; epoch wall; evaluator inference/pre/post; peak allocated/reserved CUDA bytes',
       'order':'seed-major serial training in separate child processes; evaluate each completed run; re-evaluate historical F0/F1 at end',
       'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
       'gpu':torch.cuda.get_device_name(0),'sha256':fingerprints}
    assert frozen['torch']==old['torch'] and frozen['ultralytics']==old['ultralytics']
    write_json(p,frozen)
    snap=REPORT/'frozen_sources';snap.mkdir(exist_ok=True)
    for f in files:
        if f.suffix in ('.py','.yaml','.csv'):shutil.copy2(f,snap/f.name)
    return frozen

def queue(smoke=False):
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=REPORT/'queue.lock'
    with lock.open('x',encoding='utf8') as f:f.write(str(os.getpid()))
    try:
        frozen=None if smoke else freeze()
        if not smoke:
            qa=json.loads((REPORT/'preflight.json').read_text(encoding='utf8'));assert qa['passed']
            smoke_status=json.loads((ROOT/'smoke/yolov8_feature_completion_v1/reports/queue_status.json').read_text(encoding='utf8'))
            assert smoke_status['status']=='complete','Four smoke runs must finish first'
        jobs=[(m,s,'train-one') for s in ((42,) if smoke else SEEDS) for m in METHODS]
        if not smoke:jobs += [(m,s,'eval-one') for s in SEEDS for m in HISTORICAL]
        env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',WANDB_DISABLED='true',COMET_MODE='DISABLED',CLEARML_OFFLINE_MODE='1')
        for index,(method,seed,mode) in enumerate(jobs,1):
            if frozen:
                for file,digest in frozen['sha256'].items():assert sha(file)==digest,f'Changed frozen input: {file}'
            if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Less than 3 GiB free; preserving partial results')
            rec=record_dir(method,seed);rec.mkdir(parents=True,exist_ok=True)
            env['PYTHONHASHSEED']=str(seed)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),mode,'--method',method,'--seed',str(seed)]
            if smoke:cmd.append('--smoke')
            with (rec/'worker.log').open('a',encoding='utf8') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT.parent))
                write_json(REPORT/'queue_status.json',{'status':'running','updated_at':now(),'queue_pid':os.getpid(),
                    'worker_pid':child.pid,'job_index':index,'total_jobs':len(jobs),'stage':mode,'method':method,'seed':seed,'log':str(rec/'worker.log')})
                code=child.wait()
            if code:raise RuntimeError(f'{method} seed {seed} {mode} failed ({code}); see {rec / "worker.log"}')
            summarize()
        if not smoke:
            ntrain,nper=summarize();assert ntrain==20 and nper==60,(ntrain,nper)
        write_json(REPORT/'queue_status.json',{'status':'complete','updated_at':now(),'jobs':len(jobs)})
    except BaseException as e:
        write_json(REPORT/'queue_status.json',{'status':'failed','updated_at':now(),'error':str(e),'traceback':traceback.format_exc()});raise
    finally:lock.unlink(missing_ok=True)

def main():
    global REPORT,RUNS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['queue','train-one','eval-one','summarize','freeze'])
    p.add_argument('--method',choices=list(METHODS)+list(HISTORICAL))
    p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.smoke:REPORT=ROOT/'smoke/yolov8_feature_completion_v1/reports';RUNS=ROOT/'smoke/yolov8_feature_completion_v1/runs'
    if args.mode=='queue':queue(args.smoke)
    elif args.mode=='train-one':train_one(args.method,args.seed,args.smoke)
    elif args.mode=='eval-one':evaluate(args.method,args.seed,source_run(args.method,args.seed),record_dir(args.method,args.seed))
    elif args.mode=='freeze':freeze()
    else:summarize()

if __name__=='__main__':main()
