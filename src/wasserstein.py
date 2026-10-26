"""
Closed-form 2-Wasserstein distance between multivariate Gaussian distributions.

Paper: "Explainable Regime-Aware Investing", Section 5.4 (p.5-6)

W_2^2(N(mu_1,Sigma_1), N(mu_2,Sigma_2)) =
    ||mu_1 - mu_2||_2^2 + Tr(Sigma_1 + Sigma_2 - 2*(Sigma_2^{1/2} Sigma_1 Sigma_2^{1/2})^{1/2})

# Custom — Context7 found no library equivalent for this specific Gaussian Wasserstein formula
"""

import numpy as np
from scipy.linalg import eigvalsh, eigh


def _psd_sqrt(A: np.ndarray) -> np.ndarray:
    """
    Compute symmetric PSD matrix square root via eigendecomposition.

    More numerically stable than scipy.linalg.sqrtm for PSD matrices.
    Clips negative eigenvalues to zero to enforce PSD.

    Parameters
    ----------
    A : np.ndarray, shape (d, d)
        Symmetric positive semi-definite matrix.

    Returns
    -------
    np.ndarray, shape (d, d)
        Square root matrix S such that S @ S ≈ A.
    """
    eigvals, eigvecs = eigh(A)
    eigvals = np.maximum(eigvals, 0.0)  # clip negatives from numerical noise
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def wasserstein2_gaussian(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    """
    Compute the squared 2-Wasserstein distance between two multivariate Gaussians.

    [paper p.5]: W_2^2(N(mu_1,Sigma_1), N(mu_2,Sigma_2)) =
        ||mu_1 - mu_2||_2^2 + Tr(Sigma_1 + Sigma_2 - 2*(Sigma_2^{1/2} Sigma_1 Sigma_2^{1/2})^{1/2})

    Parameters
    ----------
    mu1 : np.ndarray, shape (d,)
        Mean of first Gaussian.
    Sigma1 : np.ndarray, shape (d, d)
        Covariance of first Gaussian.
    mu2 : np.ndarray, shape (d,)
        Mean of second Gaussian.
    Sigma2 : np.ndarray, shape (d, d)
        Covariance of second Gaussian.

    Returns
    -------
    float
        Squared 2-Wasserstein distance W_2^2 >= 0.
    """
    diff_mu = mu1 - mu2
    mean_term = float(np.dot(diff_mu, diff_mu))

    # Compute Sigma_2^{1/2}
    Sigma2_sqrt = _psd_sqrt(Sigma2)

    # Compute M = Sigma_2^{1/2} @ Sigma_1 @ Sigma_2^{1/2}
    M = Sigma2_sqrt @ Sigma1 @ Sigma2_sqrt

    # Compute M^{1/2}
    M_sqrt = _psd_sqrt(M)

    # Trace term: Tr(Sigma_1 + Sigma_2 - 2 * M^{1/2})
    trace_term = np.trace(Sigma1) + np.trace(Sigma2) - 2.0 * np.trace(M_sqrt)
    trace_term = max(trace_term, 0.0)  # numerical safety

    return mean_term + trace_term


def wasserstein2_distance(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    """
    Compute the 2-Wasserstein distance (not squared) between two Gaussians.

    Parameters
    ----------
    mu1, Sigma1 : mean and covariance of first Gaussian.
    mu2, Sigma2 : mean and covariance of second Gaussian.

    Returns
    -------
    float
        W_2 distance >= 0.
    """
    return float(np.sqrt(max(wasserstein2_gaussian(mu1, Sigma1, mu2, Sigma2), 0.0)))


def assign_components_to_templates(
    comp_means: np.ndarray,
    comp_covars: np.ndarray,
    template_means: np.ndarray,
    template_covars: np.ndarray,
) -> np.ndarray:
    """
    Map each HMM component to its nearest template via 2-Wasserstein distance.

    [paper p.5-6]: g(k) = argmin_{g<=G} W_2(N(mu_g,Sigma_g), N(mu_{t,k},Sigma_{t,k}))

    Parameters
    ----------
    comp_means : np.ndarray, shape (K, d)
        Means of K HMM components at time t.
    comp_covars : np.ndarray, shape (K, d, d)
        Covariances of K HMM components.
    template_means : np.ndarray, shape (G, d)
        Current template means.
    template_covars : np.ndarray, shape (G, d, d)
        Current template covariances.

    Returns
    -------
    np.ndarray, shape (K,), dtype int
        Assignment array: assignments[k] = g means component k maps to template g.
    """
    K = comp_means.shape[0]
    G = template_means.shape[0]
    assignments = np.zeros(K, dtype=int)

    for k in range(K):
        min_dist = np.inf
        best_g = 0
        for g in range(G):
            d = wasserstein2_distance(
                template_means[g], template_covars[g],
                comp_means[k],     comp_covars[k],
            )
            if d < min_dist:
                min_dist = d
                best_g = g
        assignments[k] = best_g

    return assignments
