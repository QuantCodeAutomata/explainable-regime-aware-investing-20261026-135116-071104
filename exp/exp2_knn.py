"""
Experiment 2 — Non-parametric KNN conditional-moment investing + MVO.

Paper: "Explainable Regime-Aware Investing", Section 6, 7.2, 8 (p.6-14)
[paper p.6]: "K-nearest neighbours on the same strictly causal feature
              representation, fed into the same MVO optimizer."

Assumptions:
  * Same tickers/split/params as exp1.
  * K=20 nearest neighbours                [assumed]
  * Min lookback L=252 before KNN begins   [assumed]
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ASSET_NAMES, TRAIN_TEST_SPLIT, KNN_K, KNN_LOOKBACK,
    GAMMA, TAU, W_MAX, ANNUALIZATION_FACTOR, RESULTS_DIR
)
from data.download_data import load_prices
from src.features import compute_log_returns, compute_features
from src.knn_model import knn_conditional_moments
from src.mvo import solve_mvo
from src.metrics import compute_diagnostics


def run_exp2(prices: pd.DataFrame = None, verbose: bool = True):
    """
    Run the KNN conditional-moment investing backtest.

    Returns dict with keys: weights_df, returns_series, diagnostics.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────
    if prices is None:
        prices = load_prices()

    n_assets = len(ASSET_NAMES)
    returns  = compute_log_returns(prices)
    features = compute_features(returns)
    returns  = returns.loc[features.index]

    # ── 2. Train/test split ─────────────────────────────────────────────────
    t0 = TRAIN_TEST_SPLIT
    feat_test = features[features.index > t0]
    test_dates = feat_test.index

    if verbose:
        print(f"KNN Test: {test_dates[0].date()} to {test_dates[-1].date()} "
              f"({len(test_dates)} days)")

    train_len = len(features[features.index <= t0])
    all_features = features.values
    all_returns  = returns.values

    w_prev = np.ones(n_assets) / n_assets
    weights_list = []
    port_returns = []

    # ── 3. Daily OOS loop ────────────────────────────────────────────────────
    for i, date in enumerate(test_dates):
        t = train_len + i

        # Expanding history strictly prior to t
        X_hist = all_features[:t]
        R_hist = all_returns[:t]

        if len(X_hist) < KNN_LOOKBACK:
            w_t = w_prev.copy()
        else:
            x_query = all_features[t]  # feature at t (uses data up to t-1 for sigma/m)
            try:
                mu_t, Sigma_t = knn_conditional_moments(
                    x_query, X_hist, R_hist[:, :n_assets], KNN_K
                )
                w_t = solve_mvo(mu_t, Sigma_t, w_prev, GAMMA, TAU, W_MAX)
            except Exception as e:
                warnings.warn(f"KNN/MVO failed at {date}: {e}")
                w_t = w_prev.copy()

        port_r = float(np.dot(w_t, all_returns[t, :n_assets]))
        weights_list.append(w_t.copy())
        port_returns.append(port_r)
        w_prev = w_t.copy()

        if verbose and i % 100 == 0:
            print(f"  {date.date()} | w={np.round(w_t, 3)}")

    # ── 4. Package results ───────────────────────────────────────────────────
    weights_arr = np.array(weights_list)
    ret_arr     = np.array(port_returns)

    weights_df = pd.DataFrame(weights_arr, index=test_dates, columns=ASSET_NAMES)
    returns_s  = pd.Series(ret_arr, index=test_dates, name="exp2_knn_mvo")

    diag = compute_diagnostics(ret_arr, weights_arr, ANNUALIZATION_FACTOR)

    # ── 5. Save outputs ──────────────────────────────────────────────────────
    weights_df.to_csv(f"{RESULTS_DIR}/exp2_weights.csv")
    returns_s.to_csv(f"{RESULTS_DIR}/exp2_returns.csv")

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    _plot_exp2(weights_df, returns_s)

    if verbose:
        print("\nExp2 diagnostics:")
        for k, v in diag.items():
            print(f"  {k:35s}: {v:.4f}")

    return {
        "weights_df":     weights_df,
        "returns_series": returns_s,
        "diagnostics":    diag,
    }


def _plot_exp2(weights_df, returns_s):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    cum = returns_s.cumsum()
    axes[0].plot(cum.index, cum.values, lw=1.5, color="tomato")
    axes[0].set_title("Exp2: Cumulative OOS Log Return (KNN + MVO)")
    axes[0].set_ylabel("Cumulative log return")
    weights_df.plot.area(ax=axes[1], linewidth=0, alpha=0.85)
    axes[1].set_title("KNN Portfolio Weights over Time")
    axes[1].set_ylabel("Weight")
    axes[1].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig("results/exp2_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/exp2_diagnostics.png")


if __name__ == "__main__":
    result = run_exp2(verbose=True)
