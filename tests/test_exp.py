"""
Integration tests for experiment scripts.

These tests run a mini-backtest to confirm the experiment pipelines
execute end-to-end without errors and produce correct-shaped outputs.
"""
import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import compute_log_returns, compute_features
from src.metrics import compute_diagnostics


def _make_mock_prices(n=600, n_assets=5, seed=0):
    """Create synthetic price data for testing."""
    np.random.seed(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    returns = np.random.randn(n, n_assets) * 0.01
    prices  = pd.DataFrame(
        np.cumprod(1 + returns, axis=0) * 100,
        index=idx,
        columns=["SPX", "BOND", "GOLD", "OIL", "USD"]
    )
    return prices


class TestExp1Pipeline:
    """Tests that exp1 pipeline components chain correctly on synthetic data."""

    def test_feature_pipeline(self):
        prices = _make_mock_prices(300)
        ret    = compute_log_returns(prices)
        feat   = compute_features(ret)
        assert feat.shape[1] == 15
        assert len(feat) > 0

    def test_diagnostics_on_random_returns(self):
        np.random.seed(1)
        ret = np.random.randn(100) * 0.01
        w   = np.tile(np.array([0.2]*5), (100, 1))
        d = compute_diagnostics(ret, w)
        assert "sharpe" in d
        assert "max_drawdown" in d
        assert d["max_drawdown"] <= 0

    def test_portfolio_returns_bounded(self):
        """Realized portfolio returns should stay in realistic range."""
        prices = _make_mock_prices(300)
        ret = compute_log_returns(prices)
        feat = compute_features(ret, 60, 20)
        # Simulate equal-weight portfolio
        n = 5
        port_r = ret.loc[feat.index].values @ (np.ones(n)/n)
        assert np.all(np.abs(port_r) < 0.5), "Unrealistic returns"


class TestExp2Pipeline:
    """Tests for the KNN pipeline components."""

    def test_knn_moment_estimation(self):
        from src.knn_model import knn_conditional_moments
        np.random.seed(5)
        X = np.random.randn(300, 15) * 0.01
        R = np.random.randn(300, 5) * 0.01
        x_q = np.zeros(15)
        mu, S = knn_conditional_moments(x_q, X, R, k_neighbors=20)
        assert mu.shape == (5,)
        assert S.shape == (5, 5)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -1e-6)

    def test_mvo_after_knn(self):
        from src.knn_model import knn_conditional_moments
        from src.mvo import solve_mvo
        np.random.seed(6)
        X = np.random.randn(300, 15) * 0.01
        R = np.random.randn(300, 5) * 0.005
        x_q = np.zeros(15)
        mu, S = knn_conditional_moments(x_q, X, R)
        w_prev = np.ones(5)/5
        w = solve_mvo(mu, S, w_prev)
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-6)


class TestExp3Pipeline:
    """Tests for the benchmark comparison."""

    def test_benchmark_returns_shape(self):
        from exp.exp3_benchmarks import build_benchmarks
        prices = _make_mock_prices(300)
        feat = compute_features(compute_log_returns(prices))
        test_dates = feat.index[-100:]
        spx_r, ew_r, _ = build_benchmarks(prices, test_dates)
        assert len(spx_r) == len(test_dates)
        assert len(ew_r) == len(test_dates)

    def test_spx_returns_finite(self):
        from exp.exp3_benchmarks import build_benchmarks
        prices = _make_mock_prices(300)
        feat = compute_features(compute_log_returns(prices))
        test_dates = feat.index[-100:]
        spx_r, ew_r, _ = build_benchmarks(prices, test_dates)
        assert np.all(np.isfinite(spx_r))
        assert np.all(np.isfinite(ew_r))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
