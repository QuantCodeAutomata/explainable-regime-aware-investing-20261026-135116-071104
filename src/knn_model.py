"""
K-nearest neighbours conditional moment estimation.

Paper: "Explainable Regime-Aware Investing", Section 6 (p.6)

Using scikit-learn NearestNeighbors + LedoitWolf — Context7 confirmed
"""

import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf


def knn_conditional_moments(
    x_query: np.ndarray,
    X_hist: np.ndarray,
    R_hist: np.ndarray,
    k_neighbors: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate conditional mean and covariance of returns given x_query
    using K-nearest neighbours.

    [paper p.6 sec.6]:
        mu_t    = (1/K) sum_{s in N_t} r_s
        Sigma_t = Cov({r_s}_{s in N_t}) with Ledoit-Wolf shrinkage

    Strict causality: x_query is built from data up to t-1 only; X_hist
    and R_hist contain only history prior to current day t.

    Parameters
    ----------
    x_query : np.ndarray, shape (d,)
        Current feature vector x_t.
    X_hist : np.ndarray, shape (T_hist, d)
        Historical feature matrix (strictly prior to t).
    R_hist : np.ndarray, shape (T_hist, N)
        Historical returns aligned with X_hist.
    k_neighbors : int
        Number of nearest neighbours K.

    Returns
    -------
    Tuple (mu_t, Sigma_t), shapes (N,) and (N, N).
    """
    k_eff = min(k_neighbors, len(X_hist))
    n_assets = R_hist.shape[1]

    if k_eff < 2:
        return np.zeros(n_assets), np.eye(n_assets) * 1e-4

    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", algorithm="auto")
    nn.fit(X_hist)
    _, indices = nn.kneighbors(x_query.reshape(1, -1))
    neighbor_idx = indices[0]

    neighbor_returns = R_hist[neighbor_idx]  # shape (k_eff, N)
    mu_t = neighbor_returns.mean(axis=0)

    if len(neighbor_returns) >= 2:
        lw = LedoitWolf()
        lw.fit(neighbor_returns)
        Sigma_t = lw.covariance_
    else:
        Sigma_t = np.eye(n_assets) * 1e-4

    # Ensure PSD
    eigvals, eigvecs = np.linalg.eigh(Sigma_t)
    eigvals = np.maximum(eigvals, 1e-8)
    Sigma_t = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return mu_t, Sigma_t
