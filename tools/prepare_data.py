"""Run the preserved data preparation and require the exact archived partition."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def pairs(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return {(r["image"],r["split"]) for r in csv.DictReader(f)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True,help="Extracted WTBD root containing JPEGImages and both XML annotation folders")
    parser.add_argument("--output",type=Path,required=True,help="New writable directory; do not overwrite historical experiment data")
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Use a new empty output directory.")
    subprocess.run([sys.executable,str(ROOT/"src/prepare_wtbd.py"),"--source",str(args.source.resolve()),
                    "--output",str(args.output.resolve()),"--seed","42","--phash-threshold","8"],check=True)
    if pairs(args.output/"split_manifest.csv")!=pairs(ROOT/"data/split_manifest.csv"):
        raise SystemExit("Generated partition differs from the archived study partition. Do not train or silently substitute this split.")
    print("Exact archived 744/160/161 image partition verified.")

if __name__=="__main__":main()
