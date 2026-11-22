"""All contrasts against the unmodified A-supervised YOLOv8n; saved predictions only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyse_prf import iou, contrast

ROOT=Path(__file__).resolve().parents[1]
CELLS=['A','S2','A_ESB','A_CBAM','S2_ESB','S2_CBAM']
SEEDS=list(range(42,47))

def records(cell,seed,ck):
    p=ROOT/'data/predictions'/f'v8_{cell}_{seed}_best_annotator_a.json' if ck=='best' else ROOT/'data/locked'/f'{cell}_{seed}_test.json'
    return json.loads(p.read_text(encoding='utf-8'))['records']

def match(r,threshold=.25):
    g=np.array(r['gt_boxes']).reshape(-1,4); gc=np.array(r['gt_cls'],int)
    p=np.array(r['pred_boxes']).reshape(-1,4); pc=np.array(r['pred_cls'],int); cf=np.array(r['conf'])
    ids=np.flatnonzero(cf>=threshold); ids=ids[np.argsort(-cf[ids],kind='stable')]
    m=iou(g,p); best=np.zeros(len(g)); matched=np.full(len(g),np.nan); pred_id=np.full(len(g),-1)
    for k in range(len(g)):
        candidates=ids[pc[ids]==gc[k]]
        if len(candidates): best[k]=m[k,candidates].max()
    for j in ids:
        candidates=np.flatnonzero((gc==pc[j]) & np.isnan(matched) & (m[:,j]>=.5))
        if len(candidates):
            k=candidates[np.argmax(m[candidates,j])]; matched[k]=m[k,j]; pred_id[k]=j
    return best,matched,pred_id

def main():
    run=[]; instances=[]; common=[]
    for ck in ['best','last']:
        for seed in SEEDS:
            baseline={r['image']:r for r in records('A',seed,ck)}
            baseline_matches={k:match(r) for k,r in baseline.items()}
            for cell in CELLS:
                bs=[]; ts=[]; dc=[]; n_common=0
                for r in records(cell,seed,ck):
                    rb=baseline[r['image']]
                    assert r['gt_boxes']==rb['gt_boxes'] and r['gt_cls']==rb['gt_cls'] and r['imgsz']==rb['imgsz']
                    b,t,ids=match(r); _,ta,_=baseline_matches[r['image']]
                    valid=np.isfinite(t)&np.isfinite(ta); dc.extend((t-ta)[valid]);n_common+=sum(valid)
                    bs.extend(b);ts.extend(t[np.isfinite(t)])
                    for k in range(len(b)):
                        instances.append(dict(cell=cell,seed=seed,ck=ck,image=r['image'],gt_index=k,category=int(r['gt_cls'][k]),best_iou=b[k],matched_iou=t[k],prediction_index=int(ids[k]),baseline_iou=ta[k]))
                run.append(dict(cell=cell,seed=seed,ck=ck,n_gt=len(bs),TP=len(ts),tp_mean_iou=np.mean(ts),gt_iou50=100*np.mean(np.array(bs)>=.5),gt_iou75=100*np.mean(np.array(bs)>=.75),gt_iou85=100*np.mean(np.array(bs)>=.85)))
                common.append(dict(cell=cell,seed=seed,ck=ck,n_common=int(n_common),mean_delta_iou=np.mean(dc)))
    df=pd.DataFrame(run);cd=pd.DataFrame(common)
    old=pd.read_csv(ROOT/'data/locquality_per_seed.csv').set_index(['cell','seed']).sort_index()
    new=df[df.ck=='best'].set_index(['cell','seed']).sort_index()
    assert np.allclose(new.tp_mean_iou,old.tp_mean_iou,atol=1e-12)
    df.to_csv(ROOT/'data/localization_v12.csv',index=False)
    cd.to_csv(ROOT/'data/common_tp_v12.csv',index=False)
    pd.DataFrame(instances).to_csv(ROOT/'data/localization_instances_v12.csv',index=False)
    ap=pd.read_csv(ROOT/'data/yolov8_reeval_table.csv'); ap=ap[ap['gt']=='annotator_a']
    locked=pd.read_csv(ROOT/'data/locked_per_seed.csv')
    out={}
    for ck in ['best','last']:
        for source,metrics in [(df,['tp_mean_iou','gt_iou50','gt_iou75','gt_iou85']),(ap,['AP50','AP75','AP'])]:
            d=source[source.ck==ck].set_index(['cell','seed'])
            for cell in CELLS[1:]:
                for m in metrics: out[f'{ck}:{cell}:{m}']=contrast(d.loc[cell].loc[SEEDS,m]-d.loc['A'].loc[SEEDS,m])
        for cell in CELLS[1:]:out[f'{ck}:{cell}:common_tp']=contrast(cd[(cd.ck==ck)&(cd.cell==cell)].sort_values('seed').mean_delta_iou)
    d=locked.set_index(['cell','seed'])
    for cell in ['S1']+CELLS[1:]:
        for m in ['P','R','F1','FP','FN']:out[f'locked:{cell}:{m}']=contrast(d.loc[cell].loc[SEEDS,m]-d.loc['A'].loc[SEEDS,m])
    (ROOT/'data/baseline_contrasts_v12.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    for ck in ['best','last']:
        for cell in CELLS[1:]:
            print(ck,cell,{m:{k:round(v,5) if isinstance(v,float) else v for k,v in out[f'{ck}:{cell}:{m}'].items()} for m in ['AP75','tp_mean_iou','common_tp']})

if __name__=='__main__':main()
