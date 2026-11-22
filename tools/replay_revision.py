"""Portable CPU replay of reviewer-requested S1+CBAM vs S2+CBAM results.

No raw images, model weights, torch or inference are required. Native AP is read
from archived native-evaluator output; localisation and screening are rematched.
"""
from pathlib import Path
import importlib.util,json,hashlib,copy
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'revision_s1_cbam'
spec=importlib.util.spec_from_file_location('revision_analysis',EXT/'original_extension/analyze_extension.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
SEEDS=[42,43,44,45,46]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def s1(seed,ck='last',ref='A',split='test'):return load(EXT/f'cache/S1_CBAM_{seed}_{ck}_{ref}_{split}.json')
def s2(seed,ck='last',split='test'):
 p=ROOT/f'data/locked/S2_CBAM_{seed}_{split}.json' if ck=='last' else ROOT/f'data/predictions/v8_S2_CBAM_{seed}_best_annotator_a.json'
 return load(p)
def reference_b(records,seed):
 original={r['image']:r for r in s1(seed,'last','B')['records']}
 result=copy.deepcopy(records)
 for r in result:
  other=original[r['image']];assert r['imgsz']==other['imgsz']==[800,800]
  r['gt_boxes']=copy.deepcopy(other['gt_boxes']);r['gt_cls']=copy.deepcopy(other['gt_cls'])
 return result
def main():
 out=ROOT/'outputs/revision_replay';out.mkdir(parents=True,exist_ok=True)
 oldap=pd.read_csv(ROOT/'data/yolov8_reeval_table.csv');rows=[];loc=[];objects=[];screen=[]
 for ck in ['last','best']:
  for seed in SEEDS:
   left=s1(seed,ck);right=s2(seed,ck)
   assert len(left['records'])==len(right['records'])==161
   assert sum(len(x['gt_cls']) for x in left['records'])==233
   for ref in ['A','B']:
    native=s1(seed,ck,ref)['native_metrics']
    old=oldap[(oldap.cell=='S2_CBAM')&(oldap.seed==seed)&(oldap.ck==ck)&oldap['gt'].eq('annotator_a' if ref=='A' else 'annotator_b')].iloc[0]
    for metric in ['AP','AP50','AP75']:
     rows.extend([dict(cell='S1_CBAM',seed=seed,checkpoint=ck,reference=ref,metric=metric,value=100*native[metric]),dict(cell='S2_CBAM',seed=seed,checkpoint=ck,reference=ref,metric=metric,value=old[metric])])
   for cell,cache in [('S1_CBAM',left),('S2_CBAM',right)]:loc.append(dict(cell=cell,seed=seed,checkpoint=ck,**analysis.localization(cache['records'])))
   objects.extend(analysis.common_objects(left['records'],right['records'],seed,ck))
 for seed in SEEDS:
  for cell in ['S1_CBAM','S2_CBAM']:
   validation=s1(seed,split='val') if cell=='S1_CBAM' else s2(seed,split='val')
   selected,curve=analysis.select_threshold(validation['records'])
   saved=load(EXT/f'selections/S1_CBAM_{seed}_last_A.json') if cell=='S1_CBAM' else load(ROOT/f'data/locked/S2_CBAM_{seed}_selection.json')
   assert selected['threshold']==saved['threshold']
   records=s1(seed)['records'] if cell=='S1_CBAM' else s2(seed)['records']
   for ref in ['A','B']:
    current=records if ref=='A' else reference_b(records,seed)
    ng,npr,tp,_=analysis.counts(current,selected['threshold'])
    screen.append(dict(cell=cell,seed=seed,reference=ref,threshold=selected['threshold'],**analysis.metrics(ng.sum(),npr.sum(),tp.sum())))
 ap=pd.DataFrame(rows);location=pd.DataFrame(loc);obj=pd.DataFrame(objects);sc=pd.DataFrame(screen);contrasts=[]
 for (ck,ref,metric),d in ap.groupby(['checkpoint','reference','metric']):
  a=d[d.cell.eq('S1_CBAM')].set_index('seed').loc[SEEDS,'value'];b=d[d.cell.eq('S2_CBAM')].set_index('seed').loc[SEEDS,'value']
  contrasts.append(dict(checkpoint=ck,reference=ref,metric=metric,**analysis.contrast(b-a)))
 for ck in ['last','best']:
  d=location[location.checkpoint.eq(ck)]
  for metric in ['tp_mean_iou','gt_iou50','gt_iou75','gt_iou85','TP']:
   a=d[d.cell.eq('S1_CBAM')].set_index('seed').loc[SEEDS,metric];b=d[d.cell.eq('S2_CBAM')].set_index('seed').loc[SEEDS,metric]
   contrasts.append(dict(checkpoint=ck,reference='A',metric=metric,**analysis.contrast(b-a)))
  contrasts.append(dict(checkpoint=ck,reference='A',metric='common_object_IoU',**analysis.contrast(obj[obj.checkpoint.eq(ck)].groupby('seed').difference.mean().loc[SEEDS])))
 for ref in ['A','B']:
  d=sc[sc.reference.eq(ref)]
  for metric in ['P','R','F1','FP','FN']:
   a=d[d.cell.eq('S1_CBAM')].set_index('seed').loc[SEEDS,metric];b=d[d.cell.eq('S2_CBAM')].set_index('seed').loc[SEEDS,metric]
   contrasts.append(dict(checkpoint='last_locked',reference=ref,metric=metric,**analysis.contrast(b-a)))
 result=pd.DataFrame(contrasts);expected=pd.read_csv(EXT/'analysis/paired_S2_minus_S1_contrasts.csv')
 keys=['checkpoint','reference','metric'];merged=result.merge(expected,on=keys,suffixes=('_actual','_expected'),validate='one_to_one')
 assert len(merged)==len(expected)==34
 errors=[]
 for metric in ['mean','sd','lo','hi','positive','n','df']:
  error=float(np.max(np.abs(merged[metric+'_actual']-merged[metric+'_expected'])));errors.append(error);assert error<1e-10,(metric,error)
 expected_screen=pd.read_csv(EXT/'analysis/locked_screening_per_seed.csv')
 merged=sc.merge(expected_screen,on=['cell','seed','reference'],suffixes=('_actual','_expected'),validate='one_to_one')
 for metric in ['threshold','GT','TP','FP','FN','P','R','F1']:
  assert np.allclose(merged[metric+'_actual'],merged[metric+'_expected'],atol=1e-10,rtol=0),metric
 result.to_csv(out/'paired_contrasts.csv',index=False);sc.to_csv(out/'screening_per_seed.csv',index=False)
 obj.to_csv(out/'common_objects.csv',index=False)
 report=dict(passed=True,CPU_only=True,inference_or_training_performed=False,paired_contrasts=34,screening_rows=len(sc),common_objects=len(obj),max_absolute_error=max(errors),AP_note='Native-evaluator archived AP values, not AP recomputed from raw predictions',rematched='common-object IoU and validation-selected screening counts')
 (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
