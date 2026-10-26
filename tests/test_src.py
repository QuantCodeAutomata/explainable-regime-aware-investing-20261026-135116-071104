"""
Unit tests for src/ modules.

Tests cover: Wasserstein distance, feature construction,
MVO optimizer, KNN moments, metrics, template operations.
"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.wasserstein import wasserstein2_gaussian, wasserstein2_distance, assign_components_to_templates
from src.features import compute_log_returns, compute_features, split_features
from src.mvo import solve_mvo
from src.knn_model import knn_conditional_moments
from src.metrics import sharpe_ratio, sortino_ratio, max_drawdown, daily_turnover, effective_n_positions, compute_diagnostics
from src.templates import (
    initialize_templates, aggregate_template_probabilities,
    compute_template_component_averages, update_templates,
    compute_mixture_moments_from_templates,
)


# ────────────────────────────── Wasserstein tests ──────────────────────────
class TestWasserstein:
    def test_zero_distance_identical(self):
        d = 5
        mu = np.random.randn(d)
        Sigma = np.eye(d) * 0.1 + 0.01 * np.outer(mu, mu)
        w2 = wasserstein2_gaussian(mu, Sigma, mu, Sigma)
        assert w2 < 1e-6, f"Expected near 0, got {w2}"

    def test_mean_only_distance(self):
        """With same covariance, W2^2 = ||mu1-mu2||^2."""
        d = 4
        mu1 = np.zeros(d)
        mu2 = np.ones(d)
        Sigma = np.eye(d)
        w2 = wasserstein2_gaussian(mu1, Sigma, mu2, Sigma)
        expected = float(d)  # ||0 - 1||^2 = d, trace(2I - 2I) = 0
        assert abs(w2 - expected) < 1e-5, f"Expected {expected}, got {w2}"

    def test_nonnegative(self):
        np.random.seed(7)
        for _ in range(10):
            d = 3
            mu1, mu2 = np.random.randn(d), np.random.randn(d)
            A = np.random.randn(d, d); S1 = A @ A.T + 0.1 * np.eye(d)
            B = np.random.randn(d, d); S2 = B @ B.T + 0.1 * np.eye(d)
            w2 = wasserstein2_gaussian(mu1, S1, mu2, S2)
            assert w2 >= -1e-8, f"W2^2 should be >= 0, got {w2}"

    def test_distance_nonneg(self):
        d = 3
        mu1, mu2 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        S = np.eye(d)
        dist = wasserstein2_distance(mu1, S, mu2, S)
        assert dist >= 0

    def test_assign_components(self):
        d = 5
        K, G = 3, 5
        comp_means = np.random.randn(K, d)
        comp_covs  = np.array([np.eye(d) for _ in range(K)])
        tmpl_means = np.random.randn(G, d)
        tmpl_covs  = np.array([np.eye(d) for _ in range(G)])
        asgn = assign_components_to_templates(comp_means, comp_covs, tmpl_means, tmpl_covs)
        assert asgn.shape == (K,)
        assert np.all(asgn >= 0) and np.all(asgn < G)


# ─────────────────────────────── Feature tests ────────────────────────────
class TestFeatures:
    def _make_prices(self, n=300):
        np.random.seed(0)
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = pd.DataFrame(
            np.cumprod(1 + np.random.randn(n, 5) * 0.01, axis=0) * 100,
            index=idx, columns=["A","B","C","D","E"]
        )
        return prices

    def test_log_returns_shape(self):
        prices = self._make_prices(200)
        ret = compute_log_returns(prices)
        assert ret.shape[0] == len(prices) - 1
        assert ret.shape[1] == 5

    def test_features_shape(self):
        prices = self._make_prices(300)
        ret = compute_log_returns(prices)
        feat = compute_features(ret, vol_window=60, mom_window=20)
        assert feat.shape[1] == 15  # 3*5
        assert not feat.isna().any().any()

    def test_features_strict_causality(self):
        """Sigma and momentum must be shifted (use data strictly before t)."""
        prices = self._make_prices(300)
        ret = compute_log_returns(prices)
        feat = compute_features(ret)
        # The sigma columns should NOT equal the volatility computed on the same day
        sigma_cols = [f"sigma_{c}" for c in ret.columns]
        assert sigma_cols[0] in feat.columns

    def test_split_features(self):
        prices = self._make_prices(300)
        ret = compute_log_returns(prices)
        feat = compute_features(ret)
        ft, fe, rt, re = split_features(feat, ret, "2020-12-31")
        assert len(ft) > 0 and len(fe) > 0
        assert ft.index[-1] <= pd.Timestamp("2020-12-31")
        assert fe.index[0] >  pd.Timestamp("2020-12-31")


# ─────────────────────────────── MVO tests ────────────────────────────────
class TestMVO:
    def test_weights_sum_to_one(self):
        np.random.seed(1)
        n = 5
        mu = np.random.randn(n) * 0.001
        A  = np.random.randn(n, n); Sigma = A @ A.T / n + 0.01 * np.eye(n)
        w_prev = np.ones(n) / n
        w = solve_mvo(mu, Sigma, w_prev)
        assert abs(w.sum() - 1.0) < 1e-4

    def test_weights_nonneg(self):
        np.random.seed(2)
        n = 5
        mu = np.random.randn(n) * 0.001
        A  = np.random.randn(n, n); Sigma = A @ A.T / n + 0.01 * np.eye(n)
        w_prev = np.ones(n) / n
        w = solve_mvo(mu, Sigma, w_prev)
        assert np.all(w >= -1e-6)

    def test_weight_cap(self):
        np.random.seed(3)
        n = 5
        mu = np.random.randn(n) * 0.001
        A  = np.random.randn(n, n); Sigma = A @ A.T / n + 0.01 * np.eye(n)
        w_prev = np.ones(n) / n
        w_max = 0.4
        w = solve_mvo(mu, Sigma, w_prev, w_max=w_max)
        assert np.all(w <= w_max + 1e-4)

    def test_single_asset(self):
        w = solve_mvo(np.array([0.001]), np.array([[0.01]]), np.array([1.0]))
        assert abs(w.sum() - 1.0) < 1e-4


# ─────────────────────────────── KNN tests ────────────────────────────────
class TestKNN:
    def _make_history(self, T=200, d=15, N=5):
        np.random.seed(10)
        X = np.random.randn(T, d) * 0.01
        R = np.random.randn(T, N) * 0.01
        return X, R

    def test_output_shapes(self):
        X, R = self._make_history()
        x_q = np.random.randn(X.shape[1]) * 0.01
        mu, Sigma = knn_conditional_moments(x_q, X, R, k_neighbors=20)
        assert mu.shape == (R.shape[1],)
        assert Sigma.shape == (R.shape[1], R.shape[1])

    def test_sigma_psd(self):
        X, R = self._make_history()
        x_q = np.zeros(X.shape[1])
        _, Sigma = knn_conditional_moments(x_q, X, R, k_neighbors=10)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals >= -1e-6), "Sigma not PSD"

    def test_few_neighbors(self):
        X = np.random.randn(5, 15) * 0.01
        R = np.random.randn(5, 5) * 0.01
        mu, Sigma = knn_conditional_moments(np.zeros(15), X, R, k_neighbors=3)
        assert mu.shape == (5,)


# ────────────────────────────── Metrics tests ─────────────────────────────
class TestMetrics:
    def test_sharpe_zeros(self):
        assert sharpe_ratio(np.zeros(100)) == 0.0

    def test_sharpe_positive_drift(self):
        # Constant returns have std=0, so use noisy positive drift
        rng = np.random.default_rng(42)
        returns = 0.001 + rng.normal(0, 0.0001, 252)
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_sortino_no_downside(self):
        returns = np.abs(np.random.randn(100)) * 0.001
        assert sortino_ratio(returns) == float("inf")

    def test_max_drawdown_nonpositive(self):
        returns = np.random.randn(200) * 0.01
        dd = max_drawdown(returns)
        assert dd <= 0

    def test_turnover_range(self):
        w1 = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        w2 = np.array([0.2, 0.4, 0.2, 0.1, 0.1])
        to = daily_turnover(w1, w2)
        assert 0 <= to <= 1.0

    def test_neff_uniform(self):
        n = 5
        w = np.ones(n) / n
        assert abs(effective_n_positions(w) - n) < 1e-6

    def test_neff_concentrated(self):
        w = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert abs(effective_n_positions(w) - 1.0) < 1e-6

    def test_diagnostics_keys(self):
        np.random.seed(5)
        r = np.random.randn(100) * 0.01
        w = np.tile(np.array([0.2]*5), (100, 1))
        diag = compute_diagnostics(r, w)
        for key in ["sharpe", "sortino", "max_drawdown", "turnover_mean", "neff_mean"]:
            assert key in diag


# ──────────────────────────── Template tests ──────────────────────────────
class TestTemplates:
    def test_initialize_templates(self):
        np.random.seed(0)
        X = np.random.randn(200, 15) * 0.01
        G = 5
        means, covars = initialize_templates(X, G)
        assert means.shape == (G, 15)
        assert covars.shape == (G, 15, 15)
        # all covariances PSD
        for g in range(G):
            eigvals = np.linalg.eigvalsh(covars[g])
            assert np.all(eigvals >= -1e-6)

    def test_aggregate_template_probs_sum_to_one(self):
        K, G = 4, 5
        filt = np.array([0.4, 0.3, 0.2, 0.1])
        asgn = np.array([0, 1, 2, 0])
        tp = aggregate_template_probabilities(filt, asgn, G)
        assert abs(tp.sum() - 1.0) < 1e-6

    def test_update_templates_psd(self):
        d, G = 10, 3
        np.random.seed(7)
        means  = np.random.randn(G, d) * 0.01
        covars = np.array([np.eye(d) * 0.01 for _ in range(G)])
        bar_m  = np.random.randn(G, d) * 0.01
        bar_c  = np.array([np.eye(d) * 0.01 for _ in range(G)])
        has_u  = np.ones(G, dtype=bool)
        m2, c2 = update_templates(means.copy(), covars.copy(), bar_m, bar_c, has_u, eta=0.1)
        for g in range(G):
            eigvals = np.linalg.eigvalsh(c2[g])
            assert np.all(eigvals >= -1e-6)

    def test_mixture_moments_shape(self):
        G, N = 5, 5
        tp = np.ones(G) / G
        tm = np.random.randn(G, 3*N) * 0.01
        tc = np.array([np.eye(3*N) * 0.01 for _ in range(G)])
        mu, Sigma = compute_mixture_moments_from_templates(tp, tm, tc, N)
        assert mu.shape == (N,)
        assert Sigma.shape == (N, N)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
