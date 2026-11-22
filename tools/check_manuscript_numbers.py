"""Check regenerated numerical tables against the original manuscript TeX.

The original labels/row order are used only for auditing the submitted baseline;
this tool is not intended to silently accept renumbered/new revision tables.
"""
import argparse
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from common import ROOT

LABEL={"A":"YOLOv8n (A)","B":"B only","S1":"S1","S2":"S2",
       "A_ESB":"A + ESB","A_CBAM":"A + CBAM","S2_ESB":"S2 + ESB","S2_CBAM":"S2 + CBAM"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript",type=Path,required=True)
    args=parser.parse_args()
    text=args.manuscript.read_text(encoding="utf-8-sig")
    blocks=re.findall(r"\\begin\{table\}.*?\\end\{table\}",text,re.S)
    assert len(blocks)==14
    def table(number):
        return [line for line in blocks[number-1].splitlines() if "&" in line and "\\\\" in line]
    def read(name):return pd.read_csv(ROOT/"outputs/tables"/name)
    checked=[]
    def verify(table_number,line,values,skip=1):
        payload="&".join(line.split("&")[skip:])
        tokens=re.findall(r"[+-]?\d+(?:\.\d+)?",payload)
        assert len(tokens)==len(values),(table_number,line,tokens,values)
        for token,value in zip(tokens,values):
            decimals=len(token.split(".")[1]) if "." in token else 0
            actual=float(token);expected=round(float(value),decimals)
            assert np.isclose(actual,expected,atol=1e-10,rtol=0),(table_number,token,value,line)
        checked.append(dict(table=table_number,row=line.split("&")[0].strip(),numbers=len(values)))
    d=read("table02_class_distribution.csv")
    for row in d.to_dict("records"):
        label=row["dataset_label"].replace("_",r"\_")
        line=next(x for x in table(2) if x.split("&")[0].strip()==label)
        verify(2,line,[row[k] for k in ["A_boxes","A_share_percent","B_boxes","elongated_A_percent"]],2)
    d=read("table03_partition.csv")
    for row in d.to_dict("records"):
        label={"train":"Training","val":"Validation","test":"Test"}[row["subset"]]
        line=next(x for x in table(3) if x.startswith(label+" &"))
        verify(3,line,[row[k] for k in ["images","A_boxes","B_boxes"]])
    line=next(x for x in table(3) if x.startswith("Total &"))
    verify(3,line,[d[k].sum() for k in ["images","A_boxes","B_boxes"]])
    inventory=read("table04_dataset_files.csv")
    for index,line in enumerate(x for x in table(4) if x.startswith(r"\texttt{")):
        verify(4,line,[inventory.iloc[index]["count"]])
    ap=read("table05_detection.csv")
    for cell,label in LABEL.items():
        values=[]
        for ck in ["best","last"]:
            row=ap.loc[(ap["cell"]==cell)&(ap["ck"]==ck)].iloc[0]
            for metric in ["AP50","AP75","AP"]:values += [row[metric+"_mean"],row[metric+"_sample_sd"]]
        line=next(x for x in table(5) if x.startswith(label+" &"))
        verify(5,line,values)
    effects=read("table06_detection_differences.csv")
    for ck in ["best","last"]:
        for cell,label in LABEL.items():
            if cell=="A":continue
            prefix=("Best" if ck=="best" else "Epoch 200")+" / "+label
            line=next(x for x in table(6) if x.startswith(prefix+" &"));values=[]
            for metric in ["AP50","AP75","AP"]:
                row=effects.loc[(effects["cell"]==cell)&(effects["ck"]==ck)&(effects["metric"]==metric)].iloc[0]
                values += [row[k] for k in ["mean","ci_low","ci_high","positive_seeds","n"]]
            verify(6,line,values)
    loc=read("table07_localization.csv")
    for row in loc.to_dict("records"):
        prefix=("Best" if row["ck"]=="best" else "Epoch 200")+" / "+LABEL[row["cell"]]
        line=next(x for x in table(7) if x.startswith(prefix+" &"))
        values=[row[k] for k in ["tp_mean_iou_mean","tp_mean_iou_sample_sd","coverage75_mean","coverage75_sample_sd"]]
        if row["cell"]!="A":values += [row[k] for k in ["n_common_mean","n_common_min","n_common_max","mean","ci_low","ci_high"]]
        verify(7,line,values)
    supervision=read("table08_supervision.csv")
    for row in supervision.to_dict("records"):
        prefix="Best" if row["checkpoint"]=="best" else "Epoch 200"
        line=next(x for x in table(8) if x.startswith(prefix+" &") and "$"+row["comparison"]+"$" in x)
        verify(8,line,[row[k] for k in ["mean","ci_low","ci_high","positive_seeds","n"]],2)
    increments=read("table09_cbam_increment.csv")
    for row in increments.to_dict("records"):
        prefix={"best":"Best","last":"Epoch 200","last_locked":"Epoch 200, locked"}[row["checkpoint"]]
        label={"AP50":"AP50 (pp)","AP75":"AP75 (pp)","AP":"AP50--95 (pp)",
               "TP_mean_IoU":"Mean TP IoU","common_IoU":"Common-object IoU",
               "P":"Precision (pp)","R":"Recall (pp)","F1":"F1 (pp)","FP":"FP count","FN":"FN count"}[row["metric"]]
        line=next(x for x in table(9) if x.startswith(prefix+" & "+label+" &"))
        verify(9,line,[row[k] for k in ["mean","ci_low","ci_high","positive_seeds","n"]],2)
    screening=read("table10_locked_screening.csv")
    for row in screening.to_dict("records"):
        label=LABEL[row["cell"]];line=next(x for x in table(10) if x.startswith(label+" &"));values=[]
        for metric in ["P","R","F1"]:values += [row[metric+"_mean"],row[metric+"_sample_sd"]]
        values += [row["FP_mean"],row["FN_mean"]]
        verify(10,line,values)
    screening_effects=read("table11_screening_differences.csv")
    for cell in screening_effects["cell"].unique():
        line=next(x for x in table(11) if x.startswith(LABEL[cell]+" &"));values=[]
        for metric in ["P","R","F1","FP","FN"]:
            row=screening_effects.loc[(screening_effects["cell"]==cell)&(screening_effects["metric"]==metric)].iloc[0]
            values += [row[k] for k in ["mean","ci_low","ci_high"]]
        verify(11,line,values)
    fixed=read("table12_fixed_thresholds.csv")
    for row in fixed.to_dict("records"):
        prefix=f'{row["threshold"]:.2f} & '+LABEL[row["cell"]]+" &"
        line=next(x for x in table(12) if x.startswith(prefix))
        verify(12,line,[row[m+"_mean"] for m in ["P","R","F1","FP","FN"]],2)
    sensitivity=read("table13_annotation_sensitivity.csv")
    for cell in ["A","S2","A_ESB","S2_ESB","A_CBAM","S2_CBAM"]:
        label="A baseline" if cell=="A" else LABEL[cell]
        line=next(x for x in table(13) if x.startswith(label+" &"))
        a=sensitivity.loc[(sensitivity["cell"]==cell)&(sensitivity["truth"]=="A")].iloc[0]
        b=sensitivity.loc[(sensitivity["cell"]==cell)&(sensitivity["truth"]=="B")].iloc[0]
        values=[a["F1_mean"],a["F1_sample_sd"]]
        for metric in ["P","R","F1","FP","FN"]:values += [b[metric+"_mean"],b[metric+"_sample_sd"]]
        verify(13,line,values)
    costs=read("table14_model_cost.csv")
    for row in costs.to_dict("records"):
        label="None" if row["module"]=="none" else row["module"].upper()
        line=next(x for x in table(14) if x.startswith(label+" &"))
        values=[row["parameters"]/1e6]
        if row["module"]!="none":values += [row["parameter_increase_percent"]]
        values += [row["gflops_estimate"],row["forward_latency_ms"]]
        verify(14,line,values)
    report=dict(tables_checked=list(range(2,15)),rows_checked=len(checked),
                numerical_values_checked=sum(r["numbers"] for r in checked),all_values_match_displayed_rounding=True,
                manuscript_name=args.manuscript.name,rows=checked)
    (ROOT/"outputs/manuscript_number_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))

if __name__=="__main__":main()
