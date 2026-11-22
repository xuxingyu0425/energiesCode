"""Recalculate the archived analyses and export the 19 main-manuscript tables.

Original numerical tables and the new paired contrast calculations are replayed
from their underlying records, not estimated from PDF values. The last step
exports the rounded presentation cells from the exact revised TeX, including
non-computational strategy, file-list and environment tables.
"""
from pathlib import Path
import subprocess,sys,re,csv,json

ROOT=Path(__file__).resolve().parents[1]

def unwrap(text):
    pattern=re.compile(r'\\rev\s*\{')
    while (m:=pattern.search(text)) is not None:
        start=m.end();depth=1;i=start
        while depth:
            if text[i]=='{' and (i==0 or text[i-1]!='\\'):depth+=1
            if text[i]=='}' and (i==0 or text[i-1]!='\\'):depth-=1
            i+=1
        text=text[:m.start()]+text[start:i-1]+text[i:]
    return text

def main():
    for script in ['reproduce_tables.py','replay_revision.py']:
        subprocess.run([sys.executable,str(ROOT/'tools'/script)],cwd=ROOT,check=True)
    source=(ROOT/'data/manuscript_revised.tex').read_text(encoding='utf-8')
    out=ROOT/'outputs/main_text_tables';out.mkdir(exist_ok=True)
    mapping=[]
    tables=re.findall(r'\\begin\{table\}.*?\\end\{table\}',source,re.S)
    assert len(tables)==19
    for number,table in enumerate(tables,1):
        label=re.search(r'\\label\{([^}]+)\}',table).group(1)
        region=unwrap(table.split(r'\toprule',1)[1].split(r'\bottomrule',1)[0])
        # shortstack line breaks occur inside cells; preserve them for the CSV.
        region=re.sub(r'\\shortstack\{([^{}]+)\}',lambda m:m[1].replace('\\\\',' / '),region)
        rows=[]
        for raw in region.split('\\\\'):
            raw=re.sub(r'\\(?:midrule|cmidrule)(?:\([^)]*\))?(?:\{[^}]*\})?','',raw).strip()
            if not raw:continue
            cells=[cell.strip() for cell in raw.split('&')]
            if len(cells)>1:rows.append(cells)
        path=out/f'Table_{number:02}_{label.replace(":","_")}.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(rows)
        mapping.append({'main_table':number,'label':label,'presentation_csv':path.name,'rows':len(rows)})
    (out/'table_numbering.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    print(json.dumps({'main_tables_exported':len(mapping),'original_numeric_analysis':'outputs/tables/','new_control_replay':'outputs/revision_replay/','rounded_presentation_cells':str(out)},indent=2))

if __name__=='__main__':main()
