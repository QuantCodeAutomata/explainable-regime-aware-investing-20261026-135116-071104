"""
Transaction-cost-aware mean-variance optimization.

Paper: "Explainable Regime-Aware Investing", Section 5.6 (p.6)

maximize  mu^T w - gamma * w^T Sigma w - tau * ||w - w_prev||_1
s.t.      1^T w = 1,  w >= 0,  w_i <= w_max

Using CVXPY — Context7 confirmed
"""

import numpy as np
import cvxpy as cp


def solve_mvo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    gamma: float = 1.0,
    tau: float = 0.001,
    w_max: float = 0.5,
    solver: str = "CLARABEL",
    warm_start: bool = True,
) -> np.ndarray:
    """
    Solve daily transaction-cost-aware long-only MVO.

    [paper p.6]:
        maximize  mu_t^T w_t - gamma * w_t^T Sigma_t w_t - tau * ||w_t - w_{t-1}||_1
        s.t.      1^T w_t = 1,  w_t >= 0,  ||w_t||_inf <= w_max

    Parameters
    ----------
    mu : np.ndarray, shape (N,)
        Expected return vector.
    Sigma : np.ndarray, shape (N, N)
        Covariance matrix (PSD).
    w_prev : np.ndarray, shape (N,)
        Previous portfolio weights.
    gamma : float
        Risk aversion coefficient.
    tau : float
        Transaction cost penalty (L1 on weight changes).
    w_max : float
        Maximum weight per asset.
    solver : str
        CVXPY solver name.
    warm_start : bool
        Whether to use warm starting.

    Returns
    -------
    np.ndarray, shape (N,)
        Optimal portfolio weights (sum=1, long-only, each <= w_max).
    """
    n = len(mu)

    # Ensure Sigma is PSD by eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-8)
    Sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    w = cp.Variable(n)

    objective = (
        mu @ w
        - gamma * cp.quad_form(w, Sigma_psd)
        - tau * cp.norm1(w - w_prev)
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= w_max,
    ]

    prob = cp.Problem(cp.Maximize(objective), constraints)

    solvers_to_try = [solver, "SCS", "ECOS"]
    for s in solvers_to_try:
        try:
            prob.solve(solver=s, warm_start=warm_start)
            if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception:
            continue

    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return w_prev.copy()

    result = np.array(w.value).flatten()
    result = np.maximum(result, 0.0)
    total = result.sum()
    if total < 1e-8:
        return np.ones(n) / n
    return result / total
