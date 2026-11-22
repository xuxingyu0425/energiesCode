"""CPU-only, frozen-endpoint S2+CBAM minus S1+CBAM extension analysis.

No torch import, inference, training, threshold tuning on test, or historical writes.
Matching/count definitions reproduce the archived v25 analysis independently.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import t

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parents[2]
PAPER = WORKSPACE/'papers/mdpi_wind_20261019_v25_en'
EXP = WORKSPACE/'experiments'
SEEDS = list(range(42, 47))
CLASSES = ['craze','crack','corrosion','hide_craze','surface_injure','thunderstrike']
CACHE = BASE/'evaluation/cache'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def iou(gt, pred):
    gt = np.asarray(gt, float).reshape(-1,4); pred = np.asarray(pred, float).reshape(-1,4)
    lo = np.maximum(gt[:,None,:2],pred[None,:,:2]); hi = np.minimum(gt[:,None,2:],pred[None,:,2:])
    intersection = np.maximum(hi-lo,0).prod(2)
    ga = (gt[:,2:]-gt[:,:2]).prod(1); pa = (pred[:,2:]-pred[:,:2]).prod(1)
    return intersection/np.maximum(ga[:,None]+pa[None,:]-intersection, 1e-12)


def match(record, threshold=.25):
    gt=np.asarray(record['gt_boxes'],float).reshape(-1,4); gc=np.asarray(record['gt_cls'],int)
    pred=np.asarray(record['pred_boxes'],float).reshape(-1,4); pc=np.asarray(record['pred_cls'],int); cf=np.asarray(record['conf'])
    keep=np.flatnonzero(cf>=threshold); keep=keep[np.argsort(-cf[keep],kind='stable')]
    overlaps=iou(gt,pred); matched=np.full(len(gt),np.nan); best=np.zeros(len(gt)); ids=np.full(len(gt),-1,dtype=int)
    for k in range(len(gt)):
        candidates=keep[pc[keep]==gc[k]]
        if len(candidates): best[k]=overlaps[k,candidates].max()
    for j in keep:
        candidates=np.flatnonzero((gc==pc[j])&np.isnan(matched)&(overlaps[:,j]>=.5))
        if len(candidates):
            k=candidates[np.argmax(overlaps[candidates,j])]; matched[k]=overlaps[k,j]; ids[k]=j
    return best, matched, ids


def metrics(gt, predicted, tp):
    p=100*tp/predicted if predicted else 0.;r=100*tp/gt if gt else 0.
    return dict(GT=int(gt),TP=int(tp),FP=int(predicted-tp),FN=int(gt-tp),P=float(p),R=float(r),F1=float(2*p*r/(p+r) if p+r else 0.))


def counts(records, threshold):
    ng=np.zeros(6,int);npr=np.zeros(6,int);tp=np.zeros(6,int); images=[]
    for record in records:
        gc=np.asarray(record['gt_cls'],int);pc=np.asarray(record['pred_cls'],int);cf=np.asarray(record['conf'])
        _,matched,ids=match(record,threshold)
        ng+=np.bincount(gc,minlength=6);npr+=np.bincount(pc[cf>=threshold],minlength=6)
        tp+=np.bincount(gc[np.isfinite(matched)],minlength=6)
        images.append(dict(image=record['image'],**metrics(len(gc),int((cf>=threshold).sum()),int(np.isfinite(matched).sum()))))
    return ng,npr,tp,images


def threshold_curve(records):
    confidence=[];success=[];gt=0
    for record in records:
        gt+=len(record['gt_cls'])
        # Prefix matches at the cache floor remain valid at every higher cutoff.
        _,matched,ids=match(record,0.)
        good=np.zeros(len(record['conf']),bool);good[ids[ids>=0]]=True
        confidence.extend(record['conf']);success.extend(good)
    confidence=np.asarray(confidence);success=np.asarray(success)
    return [dict(threshold=k/100,**metrics(gt,int((confidence>=k/100).sum()),int(success[confidence>=k/100].sum()))) for k in range(1,100)]


def select_threshold(records):
    curve=threshold_curve(records)
    selected=max(curve,key=lambda r:r['F1'])  # ascending grid => first/smallest exact tie
    ng,npr,tp,_=counts(records,selected['threshold'])
    assert metrics(ng.sum(),npr.sum(),tp.sum())=={k:v for k,v in selected.items() if k!='threshold'}
    return selected,curve


def contrast(values):
    values=np.asarray(values,float);assert len(values)==5 and np.isfinite(values).all()
    mean=values.mean();sd=values.std(ddof=1);half=t.ppf(.975,4)*sd/np.sqrt(5)
    return dict(mean=float(mean),sd=float(sd),lo=float(mean-half),hi=float(mean+half),positive=int((values>0).sum()),n=5,df=4)


def s2_cache(seed,checkpoint,split='test'):
    if split=='val' or checkpoint=='last':return PAPER/f'source_data/locked/S2_CBAM_{seed}_{split}.json'
    return PAPER/f'reproducibility/predictions/v8_S2_CBAM_{seed}_best_annotator_a.json'


def new_cache(seed,checkpoint,reference='A',split='test'):
    return CACHE/f'S1_CBAM_{seed}_{checkpoint}_{reference}_{split}.json'


def cache_hash(cache):
    return cache.get('weight_sha256',cache.get('key',{}).get('weight_sha256'))


def change_reference(records, reference):
    """Hold all A predictions fixed, replace only annotation boxes/classes."""
    assert reference=='B'
    output=copy.deepcopy(records)
    root=WORKSPACE/'WT blade defect dataset/WT blade defect dataset/annotation_second_person'
    for record in output:
        assert record['imgsz']==[800,800]
        xml=ET.parse(root/Path(record['image']).with_suffix('.xml')).getroot()
        assert xml.findtext('size/width')==xml.findtext('size/height')=='1024'
        objs=xml.findall('object')
        record['gt_boxes']=[[(float(obj.findtext('bndbox/'+k))*.75+16) for k in ['xmin','ymin','xmax','ymax']] for obj in objs]
        record['gt_cls']=[CLASSES.index(obj.findtext('name').strip()) for obj in objs]
    return output


def localization(records):
    matched=[];best=[]
    for record in records:
        b,m,_=match(record);best.extend(b);matched.extend(m[np.isfinite(m)])
    best=np.asarray(best)
    return dict(n_gt=len(best),TP=len(matched),tp_mean_iou=float(np.mean(matched)),
                gt_iou50=float(100*(best>=.5).mean()),gt_iou75=float(100*(best>=.75).mean()),
                gt_iou85=float(100*(best>=.85).mean()))


def common_objects(left,right,seed,checkpoint):
    lm={r['image']:r for r in left};rm={r['image']:r for r in right};assert set(lm)==set(rm)
    rows=[]
    for image,a in lm.items():
        b=rm[image]
        for field in ['gt_boxes','gt_cls','imgsz']:assert a[field]==b[field],(seed,checkpoint,image,field)
        _,li,_=match(a);_,ri,_=match(b)
        for k in np.flatnonzero(np.isfinite(li)&np.isfinite(ri)):
            rows.append(dict(checkpoint=checkpoint,seed=seed,image=image,gt_index=int(k),category=int(a['gt_cls'][k]),
                             left_iou=float(li[k]),right_iou=float(ri[k]),difference=float(ri[k]-li[k])))
    return rows


def verify_historical():
    """Independent CPU reconciliation against archived AP/IoU/screening values."""
    proof=[];source_files={}
    ap=pd.read_csv(PAPER/'source_data/yolov8_reeval_table.csv')
    loc=pd.read_csv(PAPER/'source_data/localization_v12.csv')
    locked=pd.read_csv(PAPER/'source_data/locked_per_seed.csv')
    sensitivity=pd.read_csv(PAPER/'source_data/annotation_sensitivity.csv')
    increment=load(PAPER/'source_data/cbam_increment_summary_v16.json')
    for p in [PAPER/'source_data/yolov8_reeval_table.csv',PAPER/'source_data/localization_v12.csv',
              PAPER/'source_data/locked_per_seed.csv',PAPER/'source_data/annotation_sensitivity.csv',
              PAPER/'source_data/cbam_increment_summary_v16.json']:
        source_files[str(p)]=sha(p)
    for checkpoint in ['last','best']:
        mean_deltas=[]
        for seed in SEEDS:
            p=s2_cache(seed,checkpoint);cache=load(p);source_files[str(p)]=sha(p)
            weight=EXP/f'runs_yolov8_attention_comparison_v1/cbam/seed_{seed}/weights/{checkpoint}.pt'
            assert sha(weight)==cache_hash(cache)
            source_files[str(weight)]=sha(weight)
            old=loc[(loc.cell=='S2_CBAM')&(loc.seed==seed)&(loc.ck==checkpoint)].iloc[0]
            actual=localization(cache['records'])
            for metric in ['TP','tp_mean_iou','gt_iou50','gt_iou75','gt_iou85']:
                assert np.isclose(actual[metric],old[metric],atol=1e-11,rtol=0),(checkpoint,seed,metric)
            native=cache['native_metrics']; AP=native.get('AP',native.get('map50_95')); AP50=native.get('AP50',native.get('map50'))
            row=ap[(ap.cell=='S2_CBAM')&(ap.seed==seed)&(ap.ck==checkpoint)&ap['gt'].eq('annotator_a')].iloc[0]
            assert abs(AP*100-row.AP)<.002 and abs(AP50*100-row.AP50)<.002
            other=(PAPER/f'source_data/locked/A_CBAM_{seed}_test.json' if checkpoint=='last' else PAPER/f'reproducibility/predictions/v8_A_CBAM_{seed}_best_annotator_a.json')
            source_files[str(other)]=sha(other)
            rows=common_objects(load(other)['records'],cache['records'],seed,checkpoint)
            mean_deltas.append(np.mean([r['difference'] for r in rows]))
            proof.append(dict(checkpoint=checkpoint,seed=seed,localization_reconciled=True,native_AP_reconciled=True,weight_hash_reconciled=True))
        result=contrast(mean_deltas)
        for field in ['mean','lo','hi']:assert np.isclose(result[field],increment[checkpoint+':common_IoU'][field],atol=1e-12,rtol=0)
    for seed in SEEDS:
        val=load(s2_cache(seed,'last','val'));selected,curve=select_threshold(val['records'])
        source_files[str(s2_cache(seed,'last','val'))]=sha(s2_cache(seed,'last','val'))
        selpath=PAPER/f'source_data/locked/S2_CBAM_{seed}_selection.json'
        saved=load(selpath);source_files[str(selpath)]=sha(selpath)
        assert selected['threshold']==saved['threshold']
        test=load(s2_cache(seed,'last'))['records']
        for reference in ['A','B']:
            records=test if reference=='A' else change_reference(test,'B')
            ng,npr,tp,_=counts(records,selected['threshold']);actual=metrics(ng.sum(),npr.sum(),tp.sum())
            expected=sensitivity[(sensitivity.cell=='S2_CBAM')&(sensitivity.seed==seed)&(sensitivity.truth==reference)&sensitivity.protocol.eq('last_val_locked')].iloc[0]
            for metric in ['GT','TP','FP','FN','P','R','F1']:
                assert np.isclose(actual[metric],expected[metric],atol=1e-10,rtol=0),(seed,reference,metric,actual[metric],expected[metric])
    result=dict(passed=True,CPU_only=True,no_new_inference=True,checked_reference_models=10,
                all_legacy_AP_localization_screening_and_thresholds_reconciled=True,
                separate_reference_B_screening_reconciled=True,records=proof,sources=source_files,
                analysis_script_sha256=sha(__file__))
    save(BASE/'analysis/historical_reconciliation.json',result)
    print('HISTORICAL_CPU_RECONCILIATION_PASSED',flush=True)
    return result


def run_analysis():
    proof=verify_historical();out=BASE/'analysis';out.mkdir(exist_ok=True)
    rows=[];locrows=[];lockedrows=[];classrows=[];imagerows=[];common=[];sources={}
    old_ap=pd.read_csv(PAPER/'source_data/yolov8_reeval_table.csv')
    for checkpoint in ['last','best']:
        for seed in SEEDS:
            s2=load(s2_cache(seed,checkpoint))
            s1=load(new_cache(seed,checkpoint))
            sources[str(s2_cache(seed,checkpoint))]=sha(s2_cache(seed,checkpoint))
            for reference in ['A','B']:
                p=new_cache(seed,checkpoint,reference);cache=load(p);sources[str(p)]=sha(p)
                native=cache['native_metrics']
                old=old_ap[(old_ap.cell=='S2_CBAM')&(old_ap.seed==seed)&(old_ap.ck==checkpoint)&old_ap['gt'].eq('annotator_a' if reference=='A' else 'annotator_b')].iloc[0]
                for metric in ['AP50','AP75','AP']:
                    rows.append(dict(cell='S1_CBAM',seed=seed,checkpoint=checkpoint,reference=reference,metric=metric,value=100*float(native[metric]),units='percent'))
                    rows.append(dict(cell='S2_CBAM',seed=seed,checkpoint=checkpoint,reference=reference,metric=metric,value=float(old[metric]),units='percent'))
            for cell,cache in [('S1_CBAM',s1),('S2_CBAM',s2)]:
                locrows.append(dict(cell=cell,seed=seed,checkpoint=checkpoint,**localization(cache['records'])))
            common.extend(common_objects(s1['records'],s2['records'],seed,checkpoint))
    for seed in SEEDS:
        s1=load(new_cache(seed,'last'));s2=load(s2_cache(seed,'last'))
        for cell,cache,val in [('S1_CBAM',s1,load(new_cache(seed,'last','A','val'))),
                               ('S2_CBAM',s2,load(s2_cache(seed,'last','val')))]:
            valpath=new_cache(seed,'last','A','val') if cell=='S1_CBAM' else s2_cache(seed,'last','val')
            sources[str(valpath)]=sha(valpath)
            selected,curve=select_threshold(val['records'])
            if cell=='S1_CBAM':
                saved=load(BASE/f'evaluation/selections/S1_CBAM_{seed}_last_A.json')
                assert saved['threshold']==selected['threshold'] and saved['weight_sha256']==cache_hash(cache)
            for reference in ['A','B']:
                records=cache['records'] if reference=='A' else change_reference(cache['records'],'B')
                ng,npr,tp,images=counts(records,selected['threshold'])
                row=dict(cell=cell,seed=seed,reference=reference,threshold=selected['threshold'],val_F1=selected['F1'],**metrics(ng.sum(),npr.sum(),tp.sum()))
                row['FP_per_100_images']=100*row['FP']/len(records);lockedrows.append(row)
                classrows.extend(dict(cell=cell,seed=seed,reference=reference,category=CLASSES[c],threshold=selected['threshold'],**metrics(ng[c],npr[c],tp[c])) for c in range(6))
                imagerows.extend(dict(cell=cell,seed=seed,reference=reference,threshold=selected['threshold'],**r) for r in images)
    ap=pd.DataFrame(rows);location=pd.DataFrame(locrows);screen=pd.DataFrame(lockedrows);objects=pd.DataFrame(common)
    ap.to_csv(out/'native_AP_raw.csv',index=False);location.to_csv(out/'localization_per_seed.csv',index=False)
    screen.to_csv(out/'locked_screening_per_seed.csv',index=False);pd.DataFrame(classrows).to_csv(out/'locked_screening_per_class.csv',index=False)
    pd.DataFrame(imagerows).to_csv(out/'locked_screening_per_image.csv',index=False);objects.to_csv(out/'S2_minus_S1_common_objects.csv',index=False)
    contrast_rows=[]
    for (checkpoint,reference,metric),d in ap.groupby(['checkpoint','reference','metric']):
        a=d[d.cell.eq('S1_CBAM')].set_index('seed').loc[SEEDS,'value'];b=d[d.cell.eq('S2_CBAM')].set_index('seed').loc[SEEDS,'value']
        contrast_rows.append(dict(checkpoint=checkpoint,reference=reference,metric=metric,units='percentage points',**contrast(b-a),S1_mean=float(a.mean()),S2_mean=float(b.mean())))
    for checkpoint in ['last','best']:
        for metric in ['tp_mean_iou','gt_iou50','gt_iou75','gt_iou85','TP']:
            d=location[location.checkpoint.eq(checkpoint)];a=d[d.cell.eq('S1_CBAM')].set_index('seed').loc[SEEDS,metric];b=d[d.cell.eq('S2_CBAM')].set_index('seed').loc[SEEDS,metric]
            contrast_rows.append(dict(checkpoint=checkpoint,reference='A',metric=metric,units='IoU' if metric=='tp_mean_iou' else ('count' if metric=='TP' else 'percentage points'),**contrast(b-a),S1_mean=float(a.mean()),S2_mean=float(b.mean())))
        d=objects[objects.checkpoint.eq(checkpoint)]
        seed_values=d.groupby('seed').difference.mean().loc[SEEDS]
        contrast_rows.append(dict(checkpoint=checkpoint,reference='A',metric='common_object_IoU',units='IoU',**contrast(seed_values),n_common_min=int(d.groupby('seed').size().min()),n_common_max=int(d.groupby('seed').size().max())))
    for reference in ['A','B']:
        for metric in ['P','R','F1','FP','FN']:
            d=screen[screen.reference.eq(reference)];a=d[d.cell.eq('S1_CBAM')].set_index('seed').loc[SEEDS,metric];b=d[d.cell.eq('S2_CBAM')].set_index('seed').loc[SEEDS,metric]
            contrast_rows.append(dict(checkpoint='last_locked',reference=reference,metric=metric,units='count' if metric in ['FP','FN'] else 'percentage points',**contrast(b-a),S1_mean=float(a.mean()),S2_mean=float(b.mean())))
    contrasts=pd.DataFrame(contrast_rows);contrasts.to_csv(out/'paired_S2_minus_S1_contrasts.csv',index=False)
    ap.groupby(['cell','checkpoint','reference','metric']).value.agg(['mean','std']).to_csv(out/'native_AP_summary.csv')
    screen.groupby(['cell','reference'])[['P','R','F1','FP','FN']].agg(['mean','std']).to_csv(out/'locked_screening_summary.csv')
    # Optional mechanism follow-through with the already XML-verified index map.
    indexfile=BASE.parent/'mechanism/results/test_annotation_index_verified.csv'
    if indexfile.exists():
        index=pd.read_csv(indexfile);assert not index.duplicated(['image','xml_gt_index']).any()
        merged=objects.merge(index.rename(columns={'xml_gt_index':'gt_index','object_index':'audit_object_index'}),on=['image','gt_index'],how='left',validate='many_to_one')
        assert merged.disagreement_bin.notna().all()
        merged['stratum']=np.where(merged.disagreement_bin.eq('unpaired'),'unpaired',np.where(merged.identical,'identical','nonidentical'))
        merged.to_csv(out/'S2_minus_S1_common_objects_discrepancy.csv',index=False)
        ds=[];ss=[]
        for (checkpoint,stratum),d in merged.groupby(['checkpoint','stratum']):
            by=d.groupby('seed').difference.mean().reindex(SEEDS)
            ds.extend(dict(checkpoint=checkpoint,stratum=stratum,seed=int(seed),mean_delta_IoU=float(value),n_common=int((d.seed==seed).sum())) for seed,value in by.items())
            estimate=contrast(by) if by.notna().all() else dict(mean=float(by.mean()),lo=np.nan,hi=np.nan,n=int(by.notna().sum()),df=np.nan,positive=int((by>0).sum()))
            ss.append(dict(checkpoint=checkpoint,stratum=stratum,**estimate,n_common_min=int(d.groupby('seed').size().reindex(SEEDS,fill_value=0).min()),n_common_max=int(d.groupby('seed').size().max()),scope='exploratory fixed-split seed CI, not population generalization'))
        pd.DataFrame(ds).to_csv(out/'S2_minus_S1_discrepancy_per_seed.csv',index=False)
        pd.DataFrame(ss).to_csv(out/'S2_minus_S1_discrepancy_summary.csv',index=False)
    save(out/'analysis_provenance.json',dict(comparison='S2_CBAM minus S1_CBAM',primary='epoch200 A AP50-95 and common-object IoU',
        secondary='best checkpoint, B reference, coverage and locked operating-point metrics',
        unit='training seed on the fixed WTBD split',CI='two-sided unadjusted Student t; n=5, df=4',
        test_selection='none',common_objects='confidence .25, same class, one-to-one IoU>=.5, recomputed pair intersection',
        threshold='each run A validation .01..99 grid, max pooled F1, first/smallest exact tie; freeze before test',
        B_screening='hold A predictions/threshold fixed and replace only reference annotation',
        no_AB_mean_AP=True,historical_reconciliation_passed=proof['passed'],sources=sources,
        analysis_script_sha256=sha(__file__),limitations=['seed intervals do not estimate independent blade/wind-farm generalization','exploratory subgroup/secondary intervals are not multiplicity adjusted','original best selected using fused-validation reference, unlike common epoch200 endpoint']))
    save(out/'schema.json',dict(native_AP_raw='one cell, seed, checkpoint, reference, metric per row; AP values percent',
        locked_screening_per_seed='pooled instance-level P/R/F1 percent; FP/FN counts; per-run A-validation threshold',
        paired_contrasts='positive means S2>S1; lower FP/FN is preferable; differences before rounding',
        common_objects='conditional on both models detecting a matched A object; not all-object IoU'))
    save(BASE/'extension_complete.json',dict(status='complete',seeds=SEEDS,primary_checkpoint='last',comparison='S2_CBAM minus S1_CBAM',
        paired_contrasts=str(out/'paired_S2_minus_S1_contrasts.csv'),analysis_provenance=str(out/'analysis_provenance.json')))
    print(contrasts[(contrasts.checkpoint=='last')&contrasts.reference.eq('A')].to_string(index=False),flush=True)
    print('S1_CBAM_EXTENSION_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['verify-historical','analyze']);args=parser.parse_args()
    {'verify-historical':verify_historical,'analyze':run_analysis}[args.stage]()
