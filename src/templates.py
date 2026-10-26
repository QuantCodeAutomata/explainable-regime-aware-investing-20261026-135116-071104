"""
Wasserstein template tracking for persistent regime identity.

Paper: "Explainable Regime-Aware Investing", Section 5.4 (p.5-6)
"""

import numpy as np
from typing import Tuple, List
from sklearn.cluster import KMeans
from src.wasserstein import assign_components_to_templates


def initialize_templates(
    X_init: np.ndarray,
    n_templates: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize G templates from an initial calibration window using K-means.

    [paper p.7 alg.1 line 3]: "Initialize templates {Theta_g} using initial calibration window"
    [assumed] Using K-means on the feature matrix to initialize template means/covariances.

    Parameters
    ----------
    X_init : np.ndarray, shape (T_init, d)
        Initial calibration window features.
    n_templates : int
        Number of templates G.
    random_state : int

    Returns
    -------
    Tuple of (template_means, template_covars), shapes (G, d) and (G, d, d).
    """
    d = X_init.shape[1]
    G = n_templates

    kmeans = KMeans(n_clusters=G, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_init)

    template_means = np.zeros((G, d))
    template_covars = np.zeros((G, d, d))
    reg = 1e-6 * np.eye(d)

    for g in range(G):
        mask = labels == g
        if mask.sum() < 2:
            template_means[g] = kmeans.cluster_centers_[g]
            template_covars[g] = np.cov(X_init.T) + reg
        else:
            template_means[g] = X_init[mask].mean(axis=0)
            template_covars[g] = np.cov(X_init[mask].T) + reg

    return template_means, template_covars


def aggregate_template_probabilities(
    filtered_probs: np.ndarray,
    assignments: np.ndarray,
    n_templates: int,
) -> np.ndarray:
    """
    Aggregate HMM component probabilities into template probabilities.

    [paper p.6]: p_{t,g} = sum_{k: g(k)=g} p_{t,k}

    Parameters
    ----------
    filtered_probs : np.ndarray, shape (K,)
        Filtered HMM component probabilities.
    assignments : np.ndarray, shape (K,), dtype int
        Template assignment for each component: assignments[k] = g.
    n_templates : int
        Number of templates G.

    Returns
    -------
    np.ndarray, shape (G,)
        Template probabilities summing to 1.
    """
    template_probs = np.zeros(n_templates)
    for k, g in enumerate(assignments):
        template_probs[g] += filtered_probs[k]
    # Normalize for numerical safety
    total = template_probs.sum()
    if total > 1e-10:
        template_probs /= total
    else:
        template_probs = np.ones(n_templates) / n_templates
    return template_probs


def compute_template_component_averages(
    comp_means: np.ndarray,
    comp_covars: np.ndarray,
    filtered_probs: np.ndarray,
    assignments: np.ndarray,
    n_templates: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute template-level component averages (mu_bar, Sigma_bar) using posterior weights.

    [paper p.7 alg.1 line 20]:
        "Compute template-level component averages (bar_mu_{t,g}, bar_Sigma_{t,g})
         from assigned components using posterior weights"

    Parameters
    ----------
    comp_means : np.ndarray, shape (K, d)
    comp_covars : np.ndarray, shape (K, d, d)
    filtered_probs : np.ndarray, shape (K,)
        Filtered probabilities for each component.
    assignments : np.ndarray, shape (K,), dtype int
    n_templates : int

    Returns
    -------
    Tuple of (bar_means, bar_covars, has_update), shapes (G, d), (G, d, d), (G,).
    has_update[g] = True if template g had at least one assigned component.
    """
    d = comp_means.shape[1]
    G = n_templates
    bar_means  = np.zeros((G, d))
    bar_covars = np.zeros((G, d, d))
    has_update = np.zeros(G, dtype=bool)

    for g in range(G):
        mask = assignments == g
        if not mask.any():
            continue
        weights = filtered_probs[mask]
        w_sum = weights.sum()
        if w_sum < 1e-12:
            continue
        weights_norm = weights / w_sum
        bar_means[g]  = np.sum(weights_norm[:, None] * comp_means[mask], axis=0)
        cov_sum = np.zeros((d, d))
        for i, k in enumerate(np.where(mask)[0]):
            cov_sum += weights_norm[i] * comp_covars[k]
        bar_covars[g] = cov_sum
        has_update[g] = True

    return bar_means, bar_covars, has_update


def update_templates(
    template_means: np.ndarray,
    template_covars: np.ndarray,
    bar_means: np.ndarray,
    bar_covars: np.ndarray,
    has_update: np.ndarray,
    eta: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update templates via exponential smoothing.

    [paper p.6]:
        mu_g  <- (1-eta)*mu_g  + eta*bar_mu_{t,g}
        Sigma_g <- (1-eta)*Sigma_g + eta*bar_Sigma_{t,g}

    Ensures PSD by eigenvalue clipping after update.

    Parameters
    ----------
    template_means : np.ndarray, shape (G, d)  [in-place updated]
    template_covars : np.ndarray, shape (G, d, d)  [in-place updated]
    bar_means : np.ndarray, shape (G, d)
    bar_covars : np.ndarray, shape (G, d, d)
    has_update : np.ndarray, shape (G,), dtype bool
    eta : float
        Smoothing rate.

    Returns
    -------
    Updated (template_means, template_covars).
    """
    G, d = template_means.shape
    reg = 1e-8 * np.eye(d)

    for g in range(G):
        if not has_update[g]:
            continue
        template_means[g]  = (1 - eta) * template_means[g]  + eta * bar_means[g]
        new_cov = (1 - eta) * template_covars[g] + eta * bar_covars[g] + reg
        # Project to nearest PSD by eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(new_cov)
        eigvals = np.maximum(eigvals, 1e-8)
        template_covars[g] = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return template_means, template_covars


def compute_mixture_moments_from_templates(
    template_probs: np.ndarray,
    template_means: np.ndarray,
    template_covars: np.ndarray,
    n_assets: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute portfolio allocation moments from template probabilities and parameters.

    [paper p.6 sec 5.5]:
        mu_t = sum_g p_{t,g} * mu_g
        Sigma_t = sum_g p_{t,g} * Sigma_g

    The templates have 3N-dimensional Gaussian parameters (features = [r, sigma, m]).
    For portfolio allocation, we use the return component (first N dimensions).

    [implementation note]: Use first N components of template means/covars as
    asset return moments — the natural interpretation given x_t = [r_t; sigma_t; m_t].

    Parameters
    ----------
    template_probs : np.ndarray, shape (G,)
    template_means : np.ndarray, shape (G, 3N)
    template_covars : np.ndarray, shape (G, 3N, 3N)
    n_assets : int  (N)

    Returns
    -------
    Tuple (alloc_mu, alloc_Sigma), shapes (N,) and (N, N).
    """
    G = len(template_probs)
    alloc_mu    = np.zeros(n_assets)
    alloc_Sigma = np.zeros((n_assets, n_assets))

    for g in range(G):
        p_g = template_probs[g]
        alloc_mu    += p_g * template_means[g, :n_assets]
        alloc_Sigma += p_g * template_covars[g, :n_assets, :n_assets]

    # Ensure PSD
    eigvals, eigvecs = np.linalg.eigh(alloc_Sigma)
    eigvals = np.maximum(eigvals, 1e-8)
    alloc_Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return alloc_mu, alloc_Sigma
