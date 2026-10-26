#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Explainable Regime-Aware Investing replication
# =============================================================================
# Usage:
#   chmod +x reproduce.sh && ./reproduce.sh
#
# Requirements:
#   Python >=3.10, pip
#   Internet access for Yahoo Finance data download
# =============================================================================
set -u

echo "======================================================================"
echo "  Explainable Regime-Aware Investing — Full Reproduction"
echo "======================================================================"

# 1. Install dependencies
echo ""
echo ">>> Step 1: Installing dependencies ..."
pip install -q -r requirements.txt

# 2. Download data
echo ""
echo ">>> Step 2: Downloading market data (Yahoo Finance) ..."
python data/download_data.py

# 3. Run unit tests
echo ""
echo ">>> Step 3: Running unit & integration tests ..."
python -m pytest tests/ -v --tb=short 2>&1

# 4. Run experiments
echo ""
echo ">>> Step 4: Running Experiment 1 — Wasserstein HMM + MVO ..."
python exp/exp1_parametric_hmm.py

echo ""
echo ">>> Step 5: Running Experiment 2 — KNN + MVO ..."
python exp/exp2_knn.py

echo ""
echo ">>> Step 6: Running Experiment 3 — Benchmark comparison ..."
python exp/exp3_benchmarks.py

# 5. Write RESULTS.md
echo ""
echo ">>> Step 7: Writing results summary ..."
python -c "
import pandas as pd, os, sys
sys.path.insert(0, '.')
from src.metrics import sharpe_ratio, sortino_ratio, max_drawdown
import numpy as np

lines = ['# Experiment Results\n',
         '## Explainable Regime-Aware Investing Replication\n',
         f'Generated automatically by reproduce.sh\n\n']

for name, fname in [('HMM+MVO (Exp1)', 'results/exp1_returns.csv'),
                    ('KNN+MVO (Exp2)', 'results/exp2_returns.csv')]:
    if os.path.exists(fname):
        r = pd.read_csv(fname, index_col=0, parse_dates=True).squeeze().dropna().values
        lines.append(f'### {name}\n')
        lines.append(f'- Sharpe:       {sharpe_ratio(r):.4f}\n')
        lines.append(f'- Sortino:      {sortino_ratio(r):.4f}\n')
        lines.append(f'- Max Drawdown: {max_drawdown(r):.4f}\n\n')

cmp = 'results/exp3_comparison.csv'
if os.path.exists(cmp):
    lines.append('### Benchmark Comparison (Exp3)\n')
    df = pd.read_csv(cmp, index_col=0)
    lines.append(df[['sharpe','sortino','max_drawdown','annual_return']].to_markdown())
    lines.append('\n')

with open('results/RESULTS.md', 'w') as f:
    f.writelines(lines)
print('Results written to results/RESULTS.md')
"

echo ""
echo "======================================================================"
echo "  Reproduction complete! See results/ for outputs."
echo "======================================================================"
