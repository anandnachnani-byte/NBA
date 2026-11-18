#!/bin/bash
# Rebuilds every figure/table from cached data, then renders eda.pdf.
set -e
cd "$(dirname "$0")"
python eda.py
cd report
pandoc eda.md -o ../eda.pdf --pdf-engine=xelatex --resource-path=..:.
echo "wrote eda.pdf"
