# Explainable Regime-Aware Investing

Replication of **"Explainable Regime-Aware Investing"** (Amine Boukardagha, Columbia University).

## Overview

This repository implements a **Wasserstein HMM-based regime-aware portfolio construction framework (R2-RD)**:

- **Exp 1 (parametric):** Expanding-window Gaussian HMM with predictive model-order selection, Wasserstein template-based regime tracking, and transaction-cost-aware MVO.
- **Exp 2 (non-parametric):** K-Nearest Neighbours conditional-moment estimation + same MVO optimizer.
- **Exp 3 (benchmarks):** Comparison against SPX Buy & Hold and Equal-Weight 20% passive portfolios.

## Repository Structure

```
├── config.py              # All parameters (assumed values documented)
├── data/
│   ├── download_data.py   # Yahoo Finance data loader
│   └── prices.csv         # Downloaded prices (gitignored)
├── src/
│   ├── features.py        # Log returns + causal rolling features
│   ├── wasserstein.py     # Closed-form 2-Wasserstein between Gaussians
│   ├── hmm_model.py       # GaussianHMM fitting + predictive log-scoring
│   ├── templates.py       # Wasserstein template tracking
│   ├── mvo.py             # Transaction-cost-aware MVO (CVXPY)
│   ├── knn_model.py       # KNN conditional moment estimation
│   └── metrics.py         # Sharpe, Sortino, MDD, turnover, N_eff
├── exp/
│   ├── exp1_parametric_hmm.py   # Exp 1: HMM + MVO
│   ├── exp2_knn.py              # Exp 2: KNN + MVO
│   └── exp3_benchmarks.py       # Exp 3: Benchmark comparison
├── tests/
│   ├── test_src.py        # Unit tests for all src/ modules
│   └── test_exp.py        # Integration tests for exp/ pipelines
├── results/               # Output metrics, CSVs, plots
├── requirements.txt
└── reproduce.sh           # Full reproduction script
```

## Usage

```bash
# Full reproduction (installs deps, downloads data, runs all experiments)
chmod +x reproduce.sh && ./reproduce.sh

# Or run individually
python data/download_data.py
python -m pytest tests/ -v
python exp/exp1_parametric_hmm.py
python exp/exp2_knn.py
python exp/exp3_benchmarks.py
```

## Five-Asset Universe

| Asset | Yahoo Finance Ticker | Description |
|-------|---------------------|-------------|
| SPX   | ^GSPC               | S&P 500 Index |
| BOND  | TLT                 | iShares 20+ Year Treasury |
| GOLD  | GLD                 | SPDR Gold Shares |
| OIL   | USO                 | United States Oil Fund |
| USD   | UUP                 | Invesco DB US Dollar Bullish |

Data period: 2005-01-01 to 2026-01-31. Train/test split: 2023-05-01.

## Paper-Reported Benchmarks

| Strategy | Sharpe | Sortino | Max DD |
|----------|--------|---------|--------|
| HMM+MVO  | 2.18   | 2.82    | -5.43% |
| KNN+MVO  | 1.81   | —       | -12.52%|
| Equal-Weight | 1.59 | 2.27 | -9.87%|
| SPX B&H  | 1.18   | 1.50    | -14.62%|

## Key Assumptions

Parameters not specified in the paper are documented in `config.py`.
Major assumptions: tickers, train/test split date (2023-05-01), G=5 templates,
K ∈ {2,...,6}, |V|=60, η=0.05, γ=1.0, τ=0.001, w_max=0.5.

## Results

See `results/RESULTS.md` for generated metrics and `results/` for plots.
