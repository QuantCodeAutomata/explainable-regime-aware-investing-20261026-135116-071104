"""
Portfolio performance metrics and diagnostics.

Paper: "Explainable Regime-Aware Investing", Sections 7-8 (p.7-14)
"""

import numpy as np
from typing import Dict, Any

ANNUALIZATION = 252


def sharpe_ratio(returns: np.ndarray, annualization: int = ANNUALIZATION) -> float:
    """Annualized Sharpe ratio (rf=0)."""
    if len(returns) == 0 or returns.std() < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annualization))


def sortino_ratio(returns: np.ndarray, annualization: int = ANNUALIZATION) -> float:
    """Annualized Sortino ratio (MAR=0)."""
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf")
    dstd = downside.std()
    if dstd < 1e-12:
        return float("inf")
    return float(returns.mean() / dstd * np.sqrt(annualization))


def max_drawdown(returns: np.ndarray) -> float:
    """
    Maximum drawdown from sequence of daily returns.
    Returns a negative value (or zero).
    """
    if len(returns) == 0:
        return 0.0
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    return float(drawdowns.min())


def daily_turnover(w_t: np.ndarray, w_prev: np.ndarray) -> float:
    """Daily portfolio turnover: TO_t = 0.5 * ||w_t - w_{t-1}||_1.

    [paper p.10 eq.]: TO_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|
    """
    return 0.5 * float(np.sum(np.abs(w_t - w_prev)))


def effective_n_positions(w: np.ndarray) -> float:
    """
    Effective number of positions (inverse Herfindahl).

    [paper p.13]: N_eff,t = 1 / sum_i w_{t,i}^2
    """
    denom = float(np.sum(w ** 2))
    if denom < 1e-12:
        return float(len(w))
    return 1.0 / denom


def compute_diagnostics(
    returns_oos: np.ndarray,
    weights: np.ndarray,
    annualization: int = ANNUALIZATION,
) -> Dict[str, Any]:
    """
    Compute full set of OOS diagnostics.

    Parameters
    ----------
    returns_oos : np.ndarray, shape (T,)
        Daily realized portfolio returns.
    weights : np.ndarray, shape (T, N)
        Daily portfolio weights; weights[t] generates returns_oos[t].
    annualization : int

    Returns
    -------
    dict with metrics: sharpe, sortino, max_drawdown, cumulative_log_return,
    annual_return, annual_volatility, turnover_mean/std/p95/pct_above_1pct/5pct,
    neff_mean/median.
    """
    turnovers = np.array([
        daily_turnover(weights[t], weights[t - 1])
        for t in range(1, len(weights))
    ])

    n_eff_arr = np.array([effective_n_positions(weights[t]) for t in range(len(weights))])

    return {
        "sharpe":                     sharpe_ratio(returns_oos, annualization),
        "sortino":                    sortino_ratio(returns_oos, annualization),
        "max_drawdown":               max_drawdown(returns_oos),
        "cumulative_log_return":      float(np.sum(returns_oos)),
        "annual_return":              float(returns_oos.mean() * annualization),
        "annual_volatility":          float(returns_oos.std() * np.sqrt(annualization)),
        "turnover_mean":              float(turnovers.mean()) if len(turnovers) else 0.0,
        "turnover_std":               float(turnovers.std()) if len(turnovers) else 0.0,
        "turnover_p95":               float(np.percentile(turnovers, 95)) if len(turnovers) else 0.0,
        "turnover_pct_above_1pct":    float(np.mean(turnovers > 0.01)) if len(turnovers) else 0.0,
        "turnover_pct_above_5pct":    float(np.mean(turnovers > 0.05)) if len(turnovers) else 0.0,
        "neff_mean":                  float(n_eff_arr.mean()),
        "neff_median":                float(np.median(n_eff_arr)),
    }
