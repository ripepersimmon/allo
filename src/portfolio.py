import numpy as np
import cvxpy as cp


def gmv_unconstrained(cov: np.ndarray) -> np.ndarray:
    """Analytical closed-form GMV (allows short positions)."""
    ones = np.ones(cov.shape[0])
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    w = cov_inv @ ones
    return w / w.sum()


def gmv_long_only(cov: np.ndarray) -> np.ndarray:
    """Long-only GMV via cvxpy: min w'Σw  s.t. 1'w=1, w≥0."""
    n = cov.shape[0]
    w = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if w.value is None:
        # fallback: equal weight
        return np.full(n, 1.0 / n)
    weights = np.array(w.value)
    weights = np.maximum(weights, 0)
    return weights / weights.sum()


def effective_n(weights: np.ndarray) -> float:
    """Herfindahl-based effective number of assets: 1 / sum(w_i^2)."""
    w = np.asarray(weights)
    return 1.0 / np.sum(w ** 2)


def turnover(w_prev: np.ndarray, w_curr: np.ndarray) -> float:
    return 0.5 * np.sum(np.abs(np.asarray(w_curr) - np.asarray(w_prev)))
