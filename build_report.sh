#!/bin/bash
# Rebuilds every figure and table from the cached data, fills the
# {{table_*}} placeholders in report/eda.md with the Markdown tables
# that eda.py wrote to results/, and renders eda.pdf with pandoc.
# Pass --no-eda to skip re-running eda.py.
set -e
cd "$(dirname "$0")"
if [ "$1" != "--no-eda" ]; then
    python eda.py > results/eda_stdout.txt
fi
python - <<'EOF'
import re
text = open('report/eda.md').read()
def fill(match):
    with open('results/' + match.group(1) + '.md') as f:
        return f.read()
open('report/_build.md', 'w').write(re.sub(r'\{\{(table_\w+)\}\}', fill,
                                           text))
EOF
pandoc report/_build.md -o eda.pdf --pdf-engine=xelatex --resource-path=.
rm report/_build.md
echo "wrote eda.pdf"
