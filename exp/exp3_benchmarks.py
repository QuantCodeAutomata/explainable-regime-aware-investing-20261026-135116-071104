"""
Experiment 3 — Passive benchmark comparison.

Paper: "Explainable Regime-Aware Investing", Section 10 (p.17-19)
Benchmarks: SPX Buy & Hold, Equal-Weight 20% each.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ASSET_NAMES, TRAIN_TEST_SPLIT, ANNUALIZATION_FACTOR, RESULTS_DIR
from data.download_data import load_prices
from src.features import compute_log_returns, compute_features
from src.metrics import compute_diagnostics, sharpe_ratio, sortino_ratio, max_drawdown


def build_benchmarks(prices: pd.DataFrame, test_dates: pd.DatetimeIndex):
    """Build SPX B&H and Equal-Weight return series over test_dates."""
    returns = compute_log_returns(prices)
    # Align to test_dates
    returns = returns.reindex(test_dates).dropna(how="all").ffill()

    n_assets = len(ASSET_NAMES)
    # SPX Buy & Hold: full weight on SPX (first asset column)
    spx_col = ASSET_NAMES[0]  # "SPX"
    spx_ret = returns[spx_col].fillna(0).values

    # Equal-Weight: 20% each asset
    ew_weights = np.ones(n_assets) / n_assets
    ew_ret = returns[ASSET_NAMES].fillna(0).values @ ew_weights

    return spx_ret, ew_ret, returns


def run_exp3(
    exp1_returns: pd.Series = None,
    exp2_returns: pd.Series = None,
    prices: pd.DataFrame = None,
    verbose: bool = True,
):
    """
    Compare regime strategies against passive benchmarks.

    If exp1_returns / exp2_returns are None, tries to load from results/*.csv.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if prices is None:
        prices = load_prices()

    # Load strategy returns if not provided
    if exp1_returns is None:
        path = f"{RESULTS_DIR}/exp1_returns.csv"
        if os.path.exists(path):
            exp1_returns = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        else:
            print("Warning: exp1 returns not found — skipping exp1 in comparison.")

    if exp2_returns is None:
        path = f"{RESULTS_DIR}/exp2_returns.csv"
        if os.path.exists(path):
            exp2_returns = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        else:
            print("Warning: exp2 returns not found — skipping exp2 in comparison.")

    # Determine test dates from available strategy results
    if exp1_returns is not None:
        test_dates = exp1_returns.index
    elif exp2_returns is not None:
        test_dates = exp2_returns.index
    else:
        features = compute_features(compute_log_returns(prices))
        test_dates = features[features.index > TRAIN_TEST_SPLIT].index

    spx_ret, ew_ret, returns = build_benchmarks(prices, test_dates)

    # Aligned lengths
    n = min(len(test_dates), len(spx_ret), len(ew_ret))
    test_dates = test_dates[:n]
    spx_ret = spx_ret[:n]
    ew_ret  = ew_ret[:n]

    results_table = {}

    # SPX B&H
    n_assets = len(ASSET_NAMES)
    spx_weights = np.zeros((n, n_assets))
    spx_weights[:, 0] = 1.0
    results_table["SPX_BuyHold"] = compute_diagnostics(spx_ret, spx_weights, ANNUALIZATION_FACTOR)

    # Equal-Weight
    ew_weights_arr = np.tile(np.ones(n_assets) / n_assets, (n, 1))
    results_table["EqualWeight"] = compute_diagnostics(ew_ret, ew_weights_arr, ANNUALIZATION_FACTOR)

    # Strategy returns
    if exp1_returns is not None:
        e1 = exp1_returns.reindex(test_dates).fillna(0).values
        w1_path = f"{RESULTS_DIR}/exp1_weights.csv"
        if os.path.exists(w1_path):
            w1_df = pd.read_csv(w1_path, index_col=0, parse_dates=True)
            w1 = w1_df.reindex(test_dates).fillna(method="ffill").values
        else:
            w1 = np.tile(np.ones(n_assets) / n_assets, (n, 1))
        results_table["HMM_MVO"] = compute_diagnostics(e1, w1[:len(e1)], ANNUALIZATION_FACTOR)

    if exp2_returns is not None:
        e2 = exp2_returns.reindex(test_dates).fillna(0).values
        w2_path = f"{RESULTS_DIR}/exp2_weights.csv"
        if os.path.exists(w2_path):
            w2_df = pd.read_csv(w2_path, index_col=0, parse_dates=True)
            w2 = w2_df.reindex(test_dates).fillna(method="ffill").values
        else:
            w2 = np.tile(np.ones(n_assets) / n_assets, (n, 1))
        results_table["KNN_MVO"] = compute_diagnostics(e2, w2[:len(e2)], ANNUALIZATION_FACTOR)

    # Save comparison table
    comp_df = pd.DataFrame(results_table).T
    comp_df.to_csv(f"{RESULTS_DIR}/exp3_comparison.csv")

    if verbose:
        print("\nExp3 comparison table:")
        print(comp_df[["sharpe", "sortino", "max_drawdown",
                        "cumulative_log_return", "annual_return",
                        "annual_volatility"]].to_string())

    # ── Plot cumulative returns ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    cum_spx = np.cumsum(spx_ret)
    cum_ew  = np.cumsum(ew_ret)
    ax.plot(test_dates[:len(cum_spx)], cum_spx, label="SPX Buy & Hold", lw=1.5, color="firebrick")
    ax.plot(test_dates[:len(cum_ew)],  cum_ew,  label="Equal-Weight",   lw=1.5, color="goldenrod")
    if exp1_returns is not None:
        cum_hmm = np.cumsum(e1)
        ax.plot(test_dates[:len(cum_hmm)], cum_hmm, label="HMM+MVO",
                lw=2.0, color="steelblue")
    if exp2_returns is not None:
        cum_knn = np.cumsum(e2)
        ax.plot(test_dates[:len(cum_knn)], cum_knn, label="KNN+MVO",
                lw=1.5, color="tomato", linestyle="--")
    ax.set_title("Cumulative OOS Log Returns — Strategy vs Benchmarks")
    ax.set_ylabel("Cumulative log return")
    ax.legend()

    # Drawdown paths
    ax2 = axes[1]
    def dd_path(r):
        c = np.cumsum(r); rm = np.maximum.accumulate(c); return c - rm

    ax2.plot(test_dates[:len(spx_ret)], dd_path(spx_ret),
             label="SPX Buy & Hold", color="firebrick", lw=1.5)
    ax2.plot(test_dates[:len(ew_ret)],  dd_path(ew_ret),
             label="Equal-Weight", color="goldenrod", lw=1.5)
    if exp1_returns is not None:
        ax2.plot(test_dates[:len(e1)], dd_path(e1), label="HMM+MVO",
                 color="steelblue", lw=2.0)
    if exp2_returns is not None:
        ax2.plot(test_dates[:len(e2)], dd_path(e2), label="KNN+MVO",
                 color="tomato", lw=1.5, linestyle="--")
    ax2.set_title("Drawdown Paths")
    ax2.set_ylabel("Drawdown (log return)")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp3_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/exp3_comparison.png")

    return results_table


if __name__ == "__main__":
    run_exp3(verbose=True)
