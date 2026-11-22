"""Regenerate the WTBD dataset, localization and screening plots."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    # Generate dataset plots before adding the S1+CBAM result series.
    subprocess.run([sys.executable, str(ROOT / 'tools/reproduce_figures.py'), *sys.argv[1:]], check=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/figures_revised.py')], check=True)
