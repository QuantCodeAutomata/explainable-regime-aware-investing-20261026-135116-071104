"""
Gaussian HMM fitting and predictive log-scoring for regime detection.

Paper: "Explainable Regime-Aware Investing", Section 5.2-5.3 (p.4-5)

Using hmmlearn GaussianHMM — Context7 confirmed
"""

import numpy as np
from typing import Tuple, Optional
from hmmlearn.hmm import GaussianHMM
import warnings


def fit_gaussian_hmm(
    X: np.ndarray,
    n_components: int,
    n_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
    n_init: int = 3,
    covariance_type: str = "diag",
) -> GaussianHMM:
    """
    Fit a K-state Gaussian HMM on feature matrix X.

    [paper p.5]: "estimate a Gaussian HMM on H_t".
    covariance_type="diag" used for runtime feasibility; paper implies "full".

    Parameters
    ----------
    X : np.ndarray, shape (T, d)
        Feature matrix (expanding history).
    n_components : int
        Number of hidden states K.
    n_iter, tol, random_state, n_init : HMM fitting hyperparameters.
    covariance_type : str
        hmmlearn covariance type; "diag" for speed, "full" for paper fidelity.

    Returns
    -------
    GaussianHMM
        Fitted model (best of n_init random restarts by log-likelihood).
    """
    best_model = None
    best_score = -np.inf

    for seed in range(random_state, random_state + n_init):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=seed,
                verbose=False,
            )
            try:
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        # fallback: single fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
            )
            best_model.fit(X)

    return best_model


def compute_filtered_probabilities(
    model: GaussianHMM,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute filtered regime probabilities p_{t,k} = P(z_t=k | F_{t-1}).

    [paper p.5]: "The model yields filtered regime probabilities p_{t,k} = P(z_t=k | F_{t-1})"

    Uses hmmlearn's posteriors (forward-backward), but for strict causality
    we use only the forward probabilities (predict_proba approximation).
    For the last observation, filtered probs = predict_proba[-1].

    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM model.
    X : np.ndarray, shape (T, d)
        Feature sequence.

    Returns
    -------
    np.ndarray, shape (K,)
        Filtered probability vector for the last observation.
    """
    # Use score_samples which gives posterior (smoothed) probabilities
    # For filtering at the last point, the posterior ≈ filtered
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, posteriors = model.score_samples(X)
    # Return probabilities for the most recent observation
    return posteriors[-1]


def compute_predictive_log_score(
    model: GaussianHMM,
    X_val: np.ndarray,
    X_history: np.ndarray,
) -> float:
    """
    Compute the one-step-ahead predictive log score on validation slice V_t.

    [paper p.5]:
        PredLL(K; t) = sum_{s in V_t} log p(x_s | F_{s-1}, K)

    Implementation: for each step s in the validation slice, compute
    log p(x_s | x_1,...,x_{s-1}) via the forward algorithm.
    The predictive density is:
        p(x_s | F_{s-1}) = sum_k alpha_{s-1}(k) * p(x_s | z_s=k)
    where alpha are the normalized forward variables.

    Parameters
    ----------
    model : GaussianHMM
        Fitted K-state Gaussian HMM.
    X_val : np.ndarray, shape (|V|, d)
        Validation slice feature matrix.
    X_history : np.ndarray, shape (T_hist, d)
        History before validation slice (for warm-starting the forward pass).

    Returns
    -------
    float
        Sum of one-step-ahead log predictive densities over V_t.
    """
    K = model.n_components
    d = X_val.shape[1]

    # Compute emission log-probabilities for the validation slice
    # log p(x_s | z_s = k) ~ N(x_s; mu_k, Sigma_k)
    def log_emission(x: np.ndarray) -> np.ndarray:
        """Return log p(x | z=k) for k=0..K-1."""
        log_probs = np.zeros(K)
        for k in range(K):
            mu_k = model.means_[k]
            cov_k = model.covars_[k]
            try:
                diff = x - mu_k
                sign, logdet = np.linalg.slogdet(cov_k)
                if sign <= 0:
                    log_probs[k] = -1e10
                    continue
                inv_cov = np.linalg.solve(cov_k, diff)
                maha = float(np.dot(diff, inv_cov))
                log_probs[k] = -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
            except np.linalg.LinAlgError:
                log_probs[k] = -1e10
        return log_probs

    # Initialize forward variable from history
    # If history available, run forward pass to get alpha at end of history
    if len(X_history) > 0:
        # Use hmmlearn posteriors to get approximate forward state at end of history
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, posts = model.score_samples(X_history)
                # Use last posterior as proxy for filtered distribution
                alpha = posts[-1].copy()
                alpha = alpha / (alpha.sum() + 1e-300)
            except Exception:
                alpha = model.startprob_.copy()
    else:
        alpha = model.startprob_.copy()

    transmat = model.transmat_

    total_pred_ll = 0.0
    for s in range(len(X_val)):
        x_s = X_val[s]
        # Predict: alpha_pred_k = sum_j alpha_j * transmat_jk
        alpha_pred = alpha @ transmat  # shape (K,)

        # Compute emission probs for x_s
        log_emit = log_emission(x_s)
        max_log = log_emit.max()
        emit = np.exp(log_emit - max_log)

        # p(x_s | F_{s-1}) = sum_k alpha_pred_k * emit_k
        pred_unnorm = alpha_pred * emit
        pred_prob = pred_unnorm.sum()
        total_pred_ll += np.log(pred_prob + 1e-300) + max_log

        # Update alpha: alpha_k ∝ alpha_pred_k * emit_k
        alpha = pred_unnorm / (pred_prob + 1e-300)

    return float(total_pred_ll)


def select_hmm_order(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    val_len: int,
    lambda_k: float = 1.0,
    n_iter: int = 100,
    random_state: int = 42,
    n_init: int = 2,
    covariance_type: str = "diag",
) -> Tuple[int, GaussianHMM]:
    """
    Select optimal HMM order using predictive model-order selection.

    [paper p.5]: K_t = argmax_K {PredLL(K;t) - lambda_K * Complexity(K)}
    where Complexity(K) = K (proportional to K).

    Parameters
    ----------
    X : np.ndarray, shape (T, d)
        Full expanding history feature matrix.
    k_min, k_max : int
        Candidate order range.
    val_len : int
        Length of validation slice |V|.
    lambda_k : float
        Complexity penalty coefficient.
    n_iter, random_state, n_init : HMM fitting hyperparameters.
    covariance_type : str
        Passed to fit_gaussian_hmm.

    Returns
    -------
    Tuple of (best_K, fitted_model_for_best_K)
    """
    n_total = len(X)
    if n_total <= val_len + k_max + 10:
        model = fit_gaussian_hmm(X, k_min, n_iter, 1e-4, random_state, 1, covariance_type)
        return k_min, model

    X_hist = X[:-val_len]
    X_val  = X[-val_len:]

    best_score = -np.inf
    best_k = k_min
    best_model = None

    for k in range(k_min, k_max + 1):
        if len(X_hist) < k * 5:
            continue
        try:
            model = fit_gaussian_hmm(X_hist, k, n_iter, 1e-4, random_state, n_init, covariance_type)
            pred_ll = compute_predictive_log_score(model, X_val, X_hist)
            complexity = float(k)
            penalized = pred_ll - lambda_k * complexity
            if penalized > best_score:
                best_score = penalized
                best_k = k
                best_model = model
        except Exception:
            continue

    if best_model is None:
        best_k = k_min
        best_model = fit_gaussian_hmm(X, k_min, n_iter, 1e-4, random_state, 1, covariance_type)

    return best_k, best_model
