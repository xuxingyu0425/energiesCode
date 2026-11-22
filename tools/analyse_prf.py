"""Post-hoc screening analysis. No training, inference, or test-based threshold optimization.

Fixed thresholds are sensitivity scenarios, not a validated deployment policy.
Match predictions in descending confidence, same class, IoU >= .5, one-to-one.
All uncertainty intervals describe five paired training seeds on one fixed split.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
REP=ROOT.parents[1]/'experiments/reports'
CELLS=['A','S2','A_ESB','S2_ESB','A_CBAM','S2_CBAM']
CLASSES=['craze','crack','corrosion','hide_craze','surface_injure','thunderstrike']
SEEDS=list(range(42,47))
THRESHOLDS=[.10,.25,.50,.75]

def iou(a,b):
    lt=np.maximum(a[:,None,:2],b[None,:,:2]); rb=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.clip(rb-lt,0,None).prod(2)
    aa=(a[:,2:]-a[:,:2]).prod(1); bb=(b[:,2:]-b[:,:2]).prod(1)
    return inter/(aa[:,None]+bb[None,:]-inter)

def counts(records,threshold):
    ng=np.zeros(6,int); npred=np.zeros(6,int); tp=np.zeros(6,int)
    perimage=[]
    for r in records:
        g=np.asarray(r['gt_boxes'],float).reshape(-1,4); gc=np.asarray(r['gt_cls'],int)
        conf=np.asarray(r['conf']); keep=np.flatnonzero(conf>=threshold)
        keep=keep[np.argsort(-conf[keep],kind='stable')]
        p=np.asarray(r['pred_boxes'],float).reshape(-1,4)[keep]; pc=np.asarray(r['pred_cls'],int)[keep]
        m=iou(g,p); used=np.zeros(len(g),bool); itp=0
        for j,c in enumerate(pc):
            candidates=np.flatnonzero((gc==c)&(~used)&(m[:,j]>=.5))
            if len(candidates):
                k=candidates[np.argmax(m[candidates,j])]; used[k]=True; tp[c]+=1; itp+=1
        ng+=np.bincount(gc,minlength=6); npred+=np.bincount(pc,minlength=6)
        perimage.append(dict(image=r['image'],GT=len(g),TP=itp,FP=len(p)-itp,FN=len(g)-itp))
    return ng,npred,tp,perimage

def metrics(ng,npr,tp):
    p=100*tp/npr if npr else 0.; r=100*tp/ng if ng else 0.
    return dict(GT=int(ng),TP=int(tp),FP=int(npr-tp),FN=int(ng-tp),P=p,R=r,F1=2*p*r/(p+r) if p+r else 0.)

def contrast(x):
    x=np.asarray(x,float); half=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return dict(mean=float(x.mean()),lo=float(x.mean()-half),hi=float(x.mean()+half),positive=int((x>0).sum()),n=len(x))

def main():
    runs=[]; cls=[]; images=[]; provenance=[]; canonical=None
    for cell in CELLS:
        folder='yolov8_revision_v1' if 'CBAM' in cell else 'controls_esb_v2'
        for seed in SEEDS:
            path=REP/folder/'prediction_cache'/f'v8_{cell}_{seed}_best_annotator_a.json'
            packaged=ROOT/'data/predictions'/path.name
            if packaged.exists(): path=packaged
            raw=path.read_bytes(); d=json.loads(raw); records=d['records']
            assert len(records)==161 and sum(len(r['gt_cls']) for r in records)==233
            key=d['key']; assert key['cell']==cell and key['seed']==seed and key['checkpoint']=='best'
            truth=[(r['image'],r['gt_boxes'],r['gt_cls']) for r in records]
            if canonical is None: canonical=truth
            assert truth==canonical,'Ground truth/order mismatch'
            provenance.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),key=key))
            for threshold in THRESHOLDS:
                ng,npr,tp,im=counts(records,threshold)
                run=dict(cell=cell,seed=seed,threshold=threshold,**metrics(ng.sum(),npr.sum(),tp.sum()))
                run['FP_per_100_images']=100*run['FP']/len(records); runs.append(run)
                for c,name in enumerate(CLASSES): cls.append(dict(cell=cell,seed=seed,threshold=threshold,category=name,**metrics(ng[c],npr[c],tp[c])))
                if threshold==.25: images.extend(dict(cell=cell,seed=seed,**r) for r in im)
    df=pd.DataFrame(runs); cd=pd.DataFrame(cls)
    df.to_csv(ROOT/'data/screening_per_seed.csv',index=False)
    cd.to_csv(ROOT/'data/screening_per_class.csv',index=False)
    pd.DataFrame(images).to_csv(ROOT/'data/screening_per_image.csv',index=False)
    cols=['P','R','F1','TP','FP','FN','FP_per_100_images']
    df.groupby(['threshold','cell'])[cols].agg(['mean','std']).to_csv(ROOT/'data/screening_summary.csv')
    pairs=[('S2_CBAM','A'),('S2_CBAM','A_ESB'),('S2_CBAM','A_CBAM'),('S2_CBAM','S2'),('S2_CBAM','S2_ESB'),('S2','A')]
    out={}
    for threshold in THRESHOLDS:
        for a,b in pairs:
            x=df[(df.cell==a)&(df.threshold==threshold)].set_index('seed').loc[SEEDS]
            y=df[(df.cell==b)&(df.threshold==threshold)].set_index('seed').loc[SEEDS]
            for metric in cols: out[f'{threshold}:{a}-{b}:{metric}']=contrast(x[metric]-y[metric])
    (ROOT/'data/screening_contrasts.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    d=pd.read_csv(ROOT/'data/yolov8_reeval_table.csv'); original={}
    for ck in ['best','last']:
        for gt in ['annotator_a','annotator_b']:
            for a,b in pairs:
                x=d[(d.cell==a)&(d.ck==ck)&(d['gt']==gt)].set_index('seed').loc[SEEDS]
                y=d[(d.cell==b)&(d.ck==ck)&(d['gt']==gt)].set_index('seed').loc[SEEDS]
                for metric in ['P','R','F1','AP50','AP75','AP']: original[f'{ck}:{gt}:{a}-{b}:{metric}']=contrast(x[metric]-y[metric])
    (ROOT/'data/prf_original_contrasts.json').write_text(json.dumps(original,indent=2),encoding='utf-8')
    (ROOT/'data/screening_provenance.json').write_text(json.dumps(dict(thresholds=THRESHOLDS,matching='confidence-descending same-class one-to-one IoU >= 0.5',scope='post-hoc fixed-threshold test sensitivity, best checkpoints only',n_images=161,n_gt=233,caches=provenance),ensure_ascii=False,indent=2),encoding='utf-8')
    # Cross-check TP counts against the independently saved localization audit.
    old=pd.read_csv(ROOT/'data/locquality_per_seed.csv').set_index(['cell','seed'])
    current=df[df.threshold==.25].set_index(['cell','seed'])
    assert np.array_equal(current.TP.sort_index(),old.pred_tp.sort_index())
    print(df[df.threshold==.25].groupby('cell')[cols].mean().round(3).to_string())
    print('Paired original metrics, S2_CBAM - A_ESB:')
    for k,v in original.items():
        if ':annotator_a:S2_CBAM-A_ESB:' in k: print(k,v)
    print('Fixed .25, S2_CBAM - A_ESB:')
    for k,v in out.items():
        if k.startswith('0.25:S2_CBAM-A_ESB'): print(k,v)

if __name__=='__main__': main()

