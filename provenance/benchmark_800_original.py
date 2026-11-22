"""Fixed-shape desktop-GPU forward and image-pipeline benchmark; no training."""
from pathlib import Path
import os,json,hashlib,time,random,sys,platform,gc
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
BASE=Path(__file__).resolve().parent
REV=BASE.parent
EXP=REV.parents[1]/'experiments'
sys.path.insert(0,str(REV/'reproducibility/src'))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def save(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 import torch,ultralytics,cv2,numpy as np
 from ultralytics import YOLO
 from ultralytics.utils import nms
 from wtbd_attention import register_attention_modules
 register_attention_modules()
 assert platform.python_version()=='3.12.13'
 assert torch.__version__=='2.3.1+cu121' and ultralytics.__version__=='8.4.37'
 assert torch.cuda.is_available()
 source=EXP/'reports/yolov8_attention_comparison_v1/benchmark_images.json'
 selection=json.loads(source.read_text(encoding='utf-8'))['images']
 assert len(selection)==32
 images=[]
 for r in selection:
  p=Path(r['path']);assert sha(p)==r['sha256'];im=cv2.imread(str(p));assert im is not None and im.shape[:2]==(1024,1024);images.append(im)
 runs={'A':'runs_yolov8_ab_s2_f1_v1/v8_a_f0','A_CBAM':'runs_yolov8_dual_fusion_v1/v8_a_cbam','S2_CBAM':'runs_yolov8_attention_comparison_v1/cbam'}
 weights={k:EXP/v/'seed_42/weights/last.pt' for k,v in runs.items()}
 protocol=dict(created_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),benchmark='Fixed800 desktop GPU; seed42 epoch200 EMA checkpoints',
  gpu=torch.cuda.get_device_name(0),python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__,
  precision='FP32',batch=1,fused=False,actual_tensor=[1,3,800,800],image_resize=[768,768],padding_each_side=16,
  forward=dict(warmup=200,iterations=1000,rounds=3,input='one fixed random tensor, seed20261117; raw network output'),
  image_pipeline=dict(warmup=200,images_per_round=32,rounds=3,order_seed=20261117,confidence=.001,nms_iou=.7,max_det=300,
   stages='cv2 resize768, border16, BGR to RGB/CHW, contiguous, host-to-device, FP32 divide255, unfused forward, NMS',
   excluded='disk IO, coordinate rescaling, visualisation and result storage'),
  synchronization='torch.cuda.synchronize before/after each measured stage',
  statistics='means/p50/p95 of measured technical timings, not model-training replication',
  source_image_manifest_sha256=sha(source),images=selection,
  weights={k:dict(path=str(p),sha256=sha(p)) for k,p in weights.items()},script_sha256=sha(__file__))
 out=BASE/'benchmark_800';out.mkdir(exist_ok=True)
 pp=out/'protocol.json'
 if pp.exists():
  old=json.loads(pp.read_text(encoding='utf-8'));assert {k:v for k,v in old.items() if k!='created_at'}=={k:v for k,v in protocol.items() if k!='created_at'}
 else:save(pp,protocol)
 save(out/'status.json',dict(status='running',time=protocol['created_at'],pid=os.getpid()))
 torch.manual_seed(20261117);x=torch.rand(1,3,800,800,device='cuda',dtype=torch.float32)
 raw=[]
 def pipeline(model,im,record=False):
  torch.cuda.synchronize();t0=time.perf_counter()
  resized=cv2.resize(im,(768,768),interpolation=cv2.INTER_LINEAR)
  padded=cv2.copyMakeBorder(resized,16,16,16,16,cv2.BORDER_CONSTANT,value=(114,114,114))
  chw=np.ascontiguousarray(padded[...,::-1].transpose(2,0,1))
  ten=torch.from_numpy(chw).unsqueeze(0).cuda().float().div_(255)
  assert tuple(ten.shape)==(1,3,800,800)
  torch.cuda.synchronize();t1=time.perf_counter();pred=model(ten)
  torch.cuda.synchronize();t2=time.perf_counter()
  det=nms.non_max_suppression(pred,conf_thres=.001,iou_thres=.7,max_det=300,nc=6)
  torch.cuda.synchronize();t3=time.perf_counter()
  if record:return dict(preprocess_ms=1000*(t1-t0),forward_ms=1000*(t2-t1),nms_ms=1000*(t3-t2),pipeline_ms=1000*(t3-t0),detections=len(det[0]))
 tasks=[(r,c) for r in range(1,4) for c in runs]
 # Balance round order deterministically without selecting the fastest result.
 ordered=[]
 for r in range(1,4):
  cells=list(runs);random.Random(20261117+r).shuffle(cells);ordered.extend((r,c) for c in cells)
 for rnd,cell in ordered:
  dest=out/f'{cell}_round{rnd}.json'
  if dest.exists():raw.append(json.loads(dest.read_text(encoding='utf-8')));continue
  model=YOLO(str(weights[cell])).model.float().cuda().eval()
  assert any(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()),'Model unexpectedly fused'
  with torch.inference_mode():
   pred=model(x);assert tuple(pred[0].shape)==(1,10,13125)
   for _ in range(200):model(x)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();forward=[]
   for _ in range(1000):
    torch.cuda.synchronize();t0=time.perf_counter();model(x);torch.cuda.synchronize();forward.append(1000*(time.perf_counter()-t0))
   fpeak=torch.cuda.max_memory_allocated()
   for j in range(200):pipeline(model,images[j%32])
   torch.cuda.reset_peak_memory_stats();image_rows=[]
   order=list(range(32));random.Random(20261117+rnd).shuffle(order)
   for ix in order:image_rows.append(dict(image=Path(selection[ix]['path']).name,**pipeline(model,images[ix],True)))
   obj=dict(cell=cell,round=rnd,weight_sha256=sha(weights[cell]),forward_samples_ms=forward,
    forward_mean_ms=float(np.mean(forward)),forward_p50_ms=float(np.median(forward)),forward_p95_ms=float(np.percentile(forward,95)),
    forward_peak_allocated_bytes=fpeak,image_rows=image_rows,pipeline_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    parameters=sum(p.numel() for p in model.parameters()),time=time.strftime('%Y-%m-%d %H:%M:%S %z'))
   save(dest,obj);raw.append(obj)
   print('BENCHMARK_ROUND',cell,rnd,obj['forward_mean_ms'],float(np.mean([v['pipeline_ms'] for v in image_rows])),flush=True)
  del model,pred;gc.collect();torch.cuda.empty_cache()
 import pandas as pd
 summary=[];pipeline_raw=[]
 for cell in runs:
  rows=[r for r in raw if r['cell']==cell];assert len(rows)==3
  fw=[v for r in rows for v in r['forward_samples_ms']];ir=[dict(cell=cell,round=r['round'],**x) for r in rows for x in r['image_rows']]
  pipeline_raw.extend(ir);vals=[x['pipeline_ms'] for x in ir]
  row=dict(cell=cell,parameters=rows[0]['parameters'],forward_n=len(fw),forward_mean_ms=float(np.mean(fw)),forward_p50_ms=float(np.median(fw)),forward_p95_ms=float(np.percentile(fw,95)),
   pipeline_n=len(vals),pipeline_mean_ms=float(np.mean(vals)),pipeline_p50_ms=float(np.median(vals)),pipeline_p95_ms=float(np.percentile(vals,95)),
   peak_allocated_MiB=max(r['pipeline_peak_allocated_bytes'] for r in rows)/1024**2)
  for k in ['preprocess_ms','forward_ms','nms_ms']:row['image_'+k]=float(np.mean([x[k] for x in ir]))
  summary.append(row)
 pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
 pd.DataFrame(pipeline_raw).to_csv(out/'pipeline_samples.csv',index=False)
 save(out/'complete.json',dict(status='complete',cells=list(runs),actual_shape=[800,800],rounds=3,forward_samples_per_model=3000,pipeline_samples_per_model=96,
  no_training=True,no_embedded_device_claim=True,protocol_sha256=sha(pp),summary=summary))
 save(out/'status.json',dict(status='complete',time=time.strftime('%Y-%m-%d %H:%M:%S %z')))
 print('BENCHMARK_800_COMPLETE',flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  save(BASE/'benchmark_800/status.json',dict(status='failed',error=traceback.format_exc()));raise
