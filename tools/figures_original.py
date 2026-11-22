import os
"""Localization evidence, paired image examples, and screening boundaries.
All effect estimates compare with unmodified A-supervised YOLOv8n.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from localization import ROOT, CELLS, SEEDS, records, match
from analyse_prf import contrast
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'legend.frameon':False})
COLORS={'S2':'#8B8F97','A_ESB':'#B87536','A_CBAM':'#4D82A8','S2_ESB':'#B87536','S2_CBAM':'#16796F','S1':'#A9ADB5'}
LABEL={'A':'YOLOv8n (A)','S2':'S2','A_ESB':'A + ESB','A_CBAM':'A + CBAM','S2_ESB':'S2 + ESB','S2_CBAM':'S2 + CBAM','S1':'S1'}
D=json.loads((ROOT/'data/baseline_contrasts_v12.json').read_text())
AP=pd.read_csv(ROOT/'data/yolov8_reeval_table.csv'); AP=AP[AP['gt']=='annotator_a']
LQ=pd.read_csv(ROOT/'data/localization_v12.csv')

def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(ROOT/'outputs/figure_qa'/f'{name}.alignment.json'),overlay_svg=str(ROOT/'outputs/figure_qa'/f'{name}.alignment.svg'),require_panel_labels=True,strict=True)
    fig.savefig(ROOT/'outputs/figures'/f'{name}.pdf')
    fig.savefig(ROOT/'outputs/figures'/f'{name}.svg')
    fig.savefig(ROOT/'outputs/figures'/f'{name}.png',dpi=600)
    plt.close(fig)

def letter(ax,s):
    ax.annotate(s,xy=(0,1),xycoords='axes fraction',xytext=(-8,14),textcoords='offset points',fontweight='bold',fontsize=10,ha='right',va='bottom')

def forest(ax,values,labels,xlabel,xlim):
    for y,(cell,c,raw) in enumerate(values):
        color=COLORS[cell]
        ax.scatter(raw,np.full(5,y)-.15+np.arange(5)*.035,s=12,color=color,alpha=.45,zorder=3,edgecolors='none')
        ax.errorbar(c['mean'],y+.13,xerr=[[c['mean']-c['lo']],[c['hi']-c['mean']]],fmt='D',ms=4,color=color,lw=1.3,capsize=2,zorder=4)
    ax.axvline(0,color='#555555',lw=.8,ls='--',zorder=0)
    ax.set_yticks(range(len(labels)),labels);ax.set_ylim(len(labels)-.55,-.55)
    ax.set_xlim(*xlim);ax.set_xlabel(xlabel);ax.grid(axis='x',color='#eeeeee',lw=.6)
    ax.tick_params(axis='y',length=0,pad=5)

def fig3():
    fig,axs=plt.subplots(3,2,figsize=(7.3,7.4));fig.subplots_adjust(left=.16,right=.98,bottom=.07,top=.88,wspace=.44,hspace=.62)
    for ax,(ck,m),s in zip(axs.flat,[('best','AP'),('last','AP'),('best','AP75'),('last','AP75'),('best','tp_mean_iou'),('last','tp_mean_iou')],'abcdef'):
        d=(AP if m.startswith('AP') else LQ); d=d[d.ck==ck].set_index(['cell','seed'])
        vals=[(cell,D[f'{ck}:{cell}:{m}'],(d.loc[cell].loc[SEEDS,m]-d.loc['A'].loc[SEEDS,m]).values) for cell in CELLS[1:]]
        xlabel={'AP':'Change in AP50–95 (percentage points)','AP75':'Change in AP75 (percentage points)','tp_mean_iou':'Change in mean TP IoU'}[m]
        xlim={'AP':(-3,5),'AP75':(-8,17),'tp_mean_iou':(-.017,.043)}[m]
        forest(ax,vals,[LABEL[c] for c in CELLS[1:]],xlabel,xlim)
        ax.set_title(('Original best checkpoint' if ck=='best' else 'Fixed epoch 200'),loc='left',pad=10);letter(ax,s)
        if m=='tp_mean_iou':ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(.02))
    fig.text(.16,.945,'Small dots: five paired seeds    Diamonds and bars: mean difference and 95% CI',fontsize=7.5,color='#555555')
    save(fig,'fig3_localization')

def fig4():
    d=pd.read_csv(ROOT/'data/localization_instances_v12.csv')
    d=d[(d.cell=='S2_CBAM')&(d.seed==42)&(d.ck=='best')].copy(); d['delta']=d.matched_iou-d.baseline_iou
    common=d.dropna(subset=['delta']); positive=common[common.delta>0].sort_values(['delta','image','gt_index']);negative=common[common.delta<0].sort_values(['delta','image','gt_index'])
    crossing=positive[(positive.baseline_iou<.75)&(positive.matched_iou>=.75)]
    bycell={c:{r['image']:r for r in records(c,42,'best')} for c in ['A','S2_CBAM']}
    def fits(sel):
        boxes=[]
        for cell in ['A','S2_CBAM']:
            r=bycell[cell][sel['image']];k=int(sel.gt_index);_,_,ids=match(r)
            boxes += [(np.array(r['gt_boxes'][k])-16)/.75,(np.array(r['pred_boxes'][ids[k]])-16)/.75]
        b=np.array(boxes);lo=b[:,:2].min(0);hi=b[:,2:].max(0)
        return (lo>=0).all() and (hi<=1024).all() and (hi[1]-lo[1]+70)<=512 and (hi[0]-lo[0]+70)<=1024
    positive_fit=positive[positive.apply(fits,axis=1)];negative_fit=negative[negative.apply(fits,axis=1)]
    strongest=positive_fit.sort_values(['delta','image','gt_index'],ascending=[False,True,True]).drop_duplicates('category').head(3)
    picks=[r for _,r in strongest.iterrows()]+[negative_fit.iloc[len(negative_fit)//2]]
    fig,axs=plt.subplots(4,2,figsize=(7.3,8.8));fig.subplots_adjust(left=.055,right=.985,top=.875,bottom=.055,wspace=.055,hspace=.46)
    out=[]
    classes=['Crazing','Cracks','Corrosion','Hidden crazing','Surface damage','Lightning damage']
    names=[classes[int(r.category)]+': localization improvement' for _,r in strongest.iterrows()]+[classes[int(picks[-1].category)]+': localization regression']
    for row,(sel,tag) in enumerate(zip(picks,names)):
        filename=sel['image'];k=int(sel.gt_index)
        source=Path(os.environ['WTBD_IMAGE_DIR'])/filename
        im=np.array(Image.open(source).convert('RGB')); H,W=im.shape[:2]
        boxes=[];details=[]
        for cell in ['A','S2_CBAM']:
            r=bycell[cell][filename];_,mi,ids=match(r);g=np.array(r['gt_boxes'][k]);p=np.array(r['pred_boxes'][ids[k]])
            # Original 1024 image letterboxed to 800 with 16-pixel symmetric padding.
            ih,iw=r['imgsz'];gain=min(768/W,768/H); pad=np.array([(iw-W*gain)/2,(ih-H*gain)/2]*2)
            g=(g-pad)/gain;p=(p-pad)/gain;boxes.extend([g,p]);details.append((cell,g,p,float(mi[k])))
        b=np.array(boxes);lo=np.maximum(b[:,:2].min(0)-35,0);hi=np.minimum(b[:,2:].max(0)+35,[W,H]);center=(lo+hi)/2
        # Same physical crop and aspect in both panels of a row; preserve source pixels.
        width=min(max(hi[0]-lo[0],(hi[1]-lo[1])*2.0,150),W,H*2.0);height=width/2.0
        center=np.clip(center,[width/2,height/2],[W-width/2,H-height/2])
        x0,y0=center-[width/2,height/2];x1,y1=center+[width/2,height/2]
        for col,(cell,g,p,mi) in enumerate(details):
            ax=axs[row,col];ax.imshow(im);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
            for box,color,ls in [(g,'#FFFFFF','--'),(p,'#F2B544' if cell=='A' else '#42E1C3','-')]:ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,edgecolor=color,ls=ls,lw=1.6))
            ax.set_title(f'{LABEL[cell]}   |   IoU {mi:.3f}',loc='left',fontsize=9,pad=5);letter(ax,chr(97+row*2+col))
        axs[row,0].text(0,1.29,f'{tag}  |  image {Path(filename).stem}, target {k+1}',transform=axs[row,0].transAxes,fontsize=8,fontweight='bold',va='bottom',ha='left')
        out.append(dict(row=row+1,selection=tag,image=filename,gt_index=k,baseline_iou=float(sel.baseline_iou),enhanced_iou=float(sel.matched_iou),delta=float(sel.delta)))
        dest=Path(os.environ['WTBD_IMAGE_DIR'])
        if source.resolve()!=(dest/filename).resolve():__import__('shutil').copy2(source,dest/filename)
    fig.text(.055,.982,'Dashed white: annotation A    Solid: matched prediction    Same crop within each pair',fontsize=7.5)
    fig.text(.055,.018,'Seed 42, original best, confidence >= 0.25. Selected examples show both improvement and regression.',fontsize=7.5,color='#555555')
    (ROOT/'outputs/figures/fig4_selection_reproduced.json').write_text(json.dumps(dict(n_total=len(d),n_common=len(common),n_positive=len(positive),n_negative=len(negative),n_positive_fit=len(positive_fit),n_negative_fit=len(negative_fit),n_crossing=len(crossing),selection='Require all GT/matched boxes within original image and their union plus 35px margin to fit a 2:1 crop (height <=512, width <=1024). Three largest positive differences with distinct classes among eligible targets, then upper median eligible negative difference; tie-break by image name and target index. Purposeful illustrations, not a prevalence sample; no targets excluded from quantitative analysis.',cases=out),indent=2),encoding='utf-8')
    save(fig,'fig4_localization_examples')

def fig5():
    d=pd.read_csv(ROOT/'data/locked_per_seed.csv').set_index(['cell','seed'])
    cd=pd.read_csv(ROOT/'data/locked_per_class.csv'); print('class cols',list(cd.columns))
    fig,axs=plt.subplots(1,2,figsize=(7.3,3.4));fig.subplots_adjust(left=.16,right=.98,bottom=.18,top=.72,wspace=.60)
    cells=['S1']+CELLS[1:]; vals=[(c,D[f'locked:{c}:F1'],(d.loc[c].loc[SEEDS,'F1']-d.loc['A'].loc[SEEDS,'F1']).values) for c in cells]
    forest(axs[0],vals,[LABEL[c] for c in cells],'Change in F1 (percentage points)',(-6,7));letter(axs[0],'a');axs[0].set_title('Overall screening',loc='left',pad=10)
    cats=['craze','crack','corrosion','hide_craze','surface_injure','thunderstrike']; labels=['Crazing','Cracks','Corrosion','Hidden crazing','Surface damage','Lightning damage'];vals=[]
    for cid,cat in enumerate(cats):
        sub=cd[cd.class_id==cid].set_index(['cell','seed']);delta=(sub.loc['S2_CBAM'].loc[SEEDS,'R']-sub.loc['A'].loc[SEEDS,'R']).values
        vals.append(('S2_CBAM',contrast(delta),delta))
    forest(axs[1],vals,labels,'Change in recall (percentage points)',(-30,30));letter(axs[1],'b');axs[1].set_title('S2 + CBAM by defect class',loc='left',pad=10)
    fig.text(.16,.94,'Application check with validation-locked thresholds',fontsize=11,fontweight='bold')
    fig.text(.16,.86,'Differences relative to YOLOv8n (A); five seeds, mean and 95% CI; fixed epoch 200',fontsize=7.5,color='#555555')
    save(fig,'fig5_application')

if __name__=='__main__':
    for f in [fig3,fig4,fig5]:f()
