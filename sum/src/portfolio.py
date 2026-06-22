"""Global minimum-variance portfolio construction."""
import numpy as np
import cvxpy as cp


def gmv_long_only(cov: np.ndarray) -> np.ndarray:
    """Long-only GMV: min w'Σw  s.t. 1'w = 1, w ≥ 0. Falls back to equal weight."""
    n = cov.shape[0]
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)),
                      [cp.sum(w) == 1, w >= 0])
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if w.value is None:
        return np.full(n, 1.0 / n)
    weights = np.maximum(np.asarray(w.value), 0)
    return weights / weights.sum()


def effective_n(weights: np.ndarray) -> float:
    """Herfindahl-based effective number of holdings: 1 / Σ wᵢ²."""
    w = np.asarray(weights)
    return 1.0 / np.sum(w ** 2)
