"""
Experiment 1 — Parametric Wasserstein HMM + MVO strategy.

Paper: "Explainable Regime-Aware Investing", Sections 5, 7.1, 8 (p.4-14)
[paper p.4-6]: Five-asset expanding-window Gaussian HMM with predictive
               model-order selection, Wasserstein template tracking, and
               transaction-cost-aware MVO.

Assumptions (not specified in paper):
  * Tickers: ^GSPC, TLT, GLD, USO, UUP   [assumed, see config.py]
  * t0 = 2023-05-01                        [assumed]
  * G=5 templates                          [assumed]
  * K_min=2, K_max=6, |V|=60              [assumed]
  * eta=0.05, gamma=1.0, tau=0.001, w_max=0.5  [assumed]
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ASSET_NAMES, TRAIN_TEST_SPLIT, N_TEMPLATES, K_MIN, K_MAX,
    ORDER_SELECTION_FREQ, VALIDATION_SLICE_LEN, LAMBDA_K,
    TEMPLATE_ETA, INIT_CALIBRATION_WINDOW, HMM_N_ITER, HMM_RANDOM_STATE,
    HMM_N_INIT, GAMMA, TAU, W_MAX, ANNUALIZATION_FACTOR, RESULTS_DIR,
    MAX_TRAIN_WINDOW, HMM_COVAR_TYPE,
)
from data.download_data import load_prices
from src.features import compute_log_returns, compute_features
from src.hmm_model import fit_gaussian_hmm, compute_filtered_probabilities, select_hmm_order
from src.templates import (
    initialize_templates, aggregate_template_probabilities,
    compute_template_component_averages, update_templates,
    compute_mixture_moments_from_templates,
)
from src.wasserstein import assign_components_to_templates
from src.mvo import solve_mvo
from src.metrics import compute_diagnostics


def run_exp1(prices: pd.DataFrame = None, verbose: bool = True):
    """
    Run the parametric Wasserstein HMM + MVO backtest.

    Returns dict with keys: weights_df, returns_series, diagnostics,
    regime_path, K_path, template_probs_df.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────
    if prices is None:
        prices = load_prices()

    n_assets = len(ASSET_NAMES)
    returns   = compute_log_returns(prices)
    features  = compute_features(returns)
    returns   = returns.loc[features.index]

    # ── 2. Train/test split ─────────────────────────────────────────────────
    t0 = TRAIN_TEST_SPLIT
    feat_train = features[features.index <= t0]
    feat_test  = features[features.index >  t0]
    ret_test   = returns[returns.index > t0]

    test_dates = feat_test.index
    if verbose:
        print(f"Train: {feat_train.index[0].date()} to {feat_train.index[-1].date()} "
              f"({len(feat_train)} days)")
        print(f"Test:  {feat_test.index[0].date()} to {feat_test.index[-1].date()} "
              f"({len(feat_test)} days)")

    # ── 3. Initialize templates from training calibration window ────────────
    X_init = feat_train.values[-INIT_CALIBRATION_WINDOW:]
    tmpl_means, tmpl_covars = initialize_templates(X_init, N_TEMPLATES, HMM_RANDOM_STATE)
    if verbose:
        print(f"Initialized {N_TEMPLATES} templates from {len(X_init)}-day calibration window")

    # ── 4. Initial portfolio: equal weight ──────────────────────────────────
    w_prev = np.ones(n_assets) / n_assets
    current_K = K_MIN
    last_order_select = -ORDER_SELECTION_FREQ  # force order selection on day 1
    hmm_model = None  # cached model — only refit on order-selection days

    # Storage
    weights_list   = []
    port_returns   = []
    regime_path    = []
    K_path         = []
    tmpl_probs_all = []

    all_features = features.values
    all_returns  = returns.values
    all_dates    = features.index

    train_len = len(feat_train)

    # ── 5. Daily OOS loop ────────────────────────────────────────────────────
    for i, date in enumerate(test_dates):
        t = train_len + i  # index into full feature matrix

        # Rolling cap: use most recent MAX_TRAIN_WINDOW observations (paper: expanding;
        # capped here for runtime feasibility — assumption documented in config.py)
        win_start   = max(0, t - MAX_TRAIN_WINDOW)
        X_hist      = all_features[win_start:t]
        R_hist      = all_returns[win_start:t]

        if len(X_hist) < max(K_MIN * 10, VALIDATION_SLICE_LEN + K_MIN * 5):
            # Not enough history yet — use equal weight
            w_t = w_prev.copy()
            port_r = float(np.dot(w_t, all_returns[t]))
            weights_list.append(w_t.copy())
            port_returns.append(port_r)
            regime_path.append(0)
            K_path.append(K_MIN)
            tmpl_probs_all.append(np.ones(N_TEMPLATES) / N_TEMPLATES)
            w_prev = w_t.copy()
            continue

        # ── 5a. Order selection (every F_K days) ────────────────────────────
        if i - last_order_select >= ORDER_SELECTION_FREQ or hmm_model is None:
            if verbose and i % 50 == 0:
                print(f"  [{i}/{len(test_dates)}] {date.date()} | order-select …")
            try:
                current_K, hmm_model = select_hmm_order(
                    X_hist, K_MIN, K_MAX, VALIDATION_SLICE_LEN,
                    LAMBDA_K, HMM_N_ITER, HMM_RANDOM_STATE, HMM_N_INIT,
                    covariance_type=HMM_COVAR_TYPE,
                )
            except Exception as e:
                warnings.warn(f"Order selection failed at {date}: {e}")
                if hmm_model is None:
                    try:
                        hmm_model = fit_gaussian_hmm(
                            X_hist, current_K, HMM_N_ITER, 1e-4,
                            HMM_RANDOM_STATE, 1, covariance_type=HMM_COVAR_TYPE,
                        )
                    except Exception:
                        w_t = w_prev.copy()
                        weights_list.append(w_t.copy())
                        port_returns.append(float(np.dot(w_t, all_returns[t])))
                        regime_path.append(0)
                        K_path.append(current_K)
                        tmpl_probs_all.append(np.ones(N_TEMPLATES) / N_TEMPLATES)
                        w_prev = w_t.copy()
                        continue
            last_order_select = i
        # else: reuse cached hmm_model — no daily refitting

        # ── 5b. Filtered probabilities ───────────────────────────────────────
        try:
            filt_probs = compute_filtered_probabilities(hmm_model, X_hist)
        except Exception:
            filt_probs = np.ones(current_K) / current_K

        comp_means = hmm_model.means_   # (K, 3N)

        # Expand diagonal covariance to full (K, 3N, 3N) for Wasserstein distance.
        # hmmlearn "diag" stores covars_ as (K, 3N); "full" stores as (K, 3N, 3N).
        raw_covars = hmm_model.covars_
        if raw_covars.ndim == 2:  # diag → expand
            comp_covars = np.array([np.diag(raw_covars[k]) for k in range(raw_covars.shape[0])])
        else:
            comp_covars = raw_covars  # already (K, 3N, 3N)

        # ── 5c. Wasserstein template assignment ──────────────────────────────
        try:
            assignments = assign_components_to_templates(
                comp_means, comp_covars, tmpl_means, tmpl_covars
            )
        except Exception:
            assignments = np.arange(current_K) % N_TEMPLATES

        # ── 5d. Aggregate template probabilities ─────────────────────────────
        tmpl_probs = aggregate_template_probabilities(
            filt_probs, assignments, N_TEMPLATES
        )

        # ── 5e. Compute template component averages ──────────────────────────
        bar_means, bar_covars, has_update = compute_template_component_averages(
            comp_means, comp_covars, filt_probs, assignments, N_TEMPLATES
        )

        # ── 5f. Update templates ─────────────────────────────────────────────
        tmpl_means, tmpl_covars = update_templates(
            tmpl_means, tmpl_covars, bar_means, bar_covars, has_update, TEMPLATE_ETA
        )

        # ── 5g. Portfolio moments from templates ─────────────────────────────
        alloc_mu, alloc_Sigma = compute_mixture_moments_from_templates(
            tmpl_probs, tmpl_means, tmpl_covars, n_assets
        )

        # ── 5h. MVO ──────────────────────────────────────────────────────────
        w_t = solve_mvo(alloc_mu, alloc_Sigma, w_prev, GAMMA, TAU, W_MAX)

        # ── 5i. Realize return r_t^p = w_t^T r_t ─────────────────────────────
        port_r = float(np.dot(w_t, all_returns[t]))

        # ── 5j. Record ───────────────────────────────────────────────────────
        dom_tmpl = int(np.argmax(tmpl_probs))
        weights_list.append(w_t.copy())
        port_returns.append(port_r)
        regime_path.append(dom_tmpl)
        K_path.append(current_K)
        tmpl_probs_all.append(tmpl_probs.copy())
        w_prev = w_t.copy()

        if verbose and i % 50 == 0:
            print(f"  {date.date()} | K={current_K} | tmpl={dom_tmpl} | "
                  f"w={np.round(w_t, 3)}")

    # ── 6. Package results ───────────────────────────────────────────────────
    weights_arr = np.array(weights_list)
    ret_arr     = np.array(port_returns)

    weights_df = pd.DataFrame(weights_arr, index=test_dates, columns=ASSET_NAMES)
    returns_s  = pd.Series(ret_arr, index=test_dates, name="exp1_hmm_mvo")

    diag = compute_diagnostics(ret_arr, weights_arr, ANNUALIZATION_FACTOR)

    # ── 7. Save outputs ──────────────────────────────────────────────────────
    weights_df.to_csv(f"{RESULTS_DIR}/exp1_weights.csv")
    returns_s.to_csv(f"{RESULTS_DIR}/exp1_returns.csv")

    # Regime path
    regime_df = pd.DataFrame({
        "dominant_template": regime_path,
        "K_t":               K_path,
    }, index=test_dates)
    regime_df.to_csv(f"{RESULTS_DIR}/exp1_regime_path.csv")

    # Template probs
    tmpl_df = pd.DataFrame(
        np.array(tmpl_probs_all),
        index=test_dates,
        columns=[f"tmpl_{g}" for g in range(N_TEMPLATES)]
    )
    tmpl_df.to_csv(f"{RESULTS_DIR}/exp1_template_probs.csv")

    # ── 8. Plots ─────────────────────────────────────────────────────────────
    _plot_exp1(weights_df, returns_s, regime_df, K_path)

    if verbose:
        print("\nExp1 diagnostics:")
        for k, v in diag.items():
            print(f"  {k:35s}: {v:.4f}")

    return {
        "weights_df":       weights_df,
        "returns_series":   returns_s,
        "diagnostics":      diag,
        "regime_path":      regime_df,
        "K_path":           K_path,
        "template_probs_df": tmpl_df,
    }


def _plot_exp1(weights_df, returns_s, regime_df, K_path):
    """Save diagnostic plots for exp1."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Cumulative returns
    cum = returns_s.cumsum()
    axes[0].plot(cum.index, cum.values, lw=1.5, color="steelblue")
    axes[0].set_title("Exp1: Cumulative OOS Log Return (Wasserstein HMM + MVO)")
    axes[0].set_ylabel("Cumulative log return")

    # Portfolio weights
    weights_df.plot.area(ax=axes[1], linewidth=0, alpha=0.85)
    axes[1].set_title("Portfolio Weights over Time")
    axes[1].set_ylabel("Weight")
    axes[1].legend(loc="upper left", fontsize=8)

    # Dominant template
    axes[2].plot(regime_df.index, regime_df["dominant_template"],
                 color="darkorange", lw=0.8)
    axes[2].set_title("Dominant Template (Regime Label)")
    axes[2].set_ylabel("Template index")

    # Selected K_t
    axes[3].plot(regime_df.index, K_path, color="purple", lw=0.8)
    axes[3].set_title("Selected HMM Order K_t")
    axes[3].set_ylabel("K")

    plt.tight_layout()
    path = f"{os.environ.get('RESULTS_DIR', 'results')}/exp1_diagnostics.png"
    fig.savefig("results/exp1_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/exp1_diagnostics.png")


if __name__ == "__main__":
    result = run_exp1(verbose=True)
