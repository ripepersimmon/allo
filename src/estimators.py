import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().values


def lw_cov(returns: pd.DataFrame) -> np.ndarray:
    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.covariance_


def gerber_cov(returns: pd.DataFrame, threshold: float = 0.3) -> np.ndarray:
    """
    Gerber (2022) statistic.
    For each pair, counts co-movements where both assets move beyond
    threshold * std dev, then builds a robust correlation matrix.
    """
    X = returns.values
    n, p = X.shape
    stds = X.std(axis=0)

    # Indicator matrices: +1 if above threshold, -1 if below, 0 neutral
    U = (X >  threshold * stds).astype(float)   # up
    D = (X < -threshold * stds).astype(float)   # down

    # Concordant = both up OR both down; discordant = one up other down
    conc = U.T @ U + D.T @ D
    disc = U.T @ D + D.T @ U

    denom = conc + disc
    # avoid division by zero for degenerate pairs
    denom = np.where(denom == 0, 1, denom)

    G = (conc - disc) / denom
    np.fill_diagonal(G, 1.0)

    # Ensure positive semi-definiteness via nearest correlation matrix
    G = _nearest_psd_corr(G)

    # Scale to covariance
    D_mat = np.diag(stds)
    cov = D_mat @ G @ D_mat
    return cov


def bbc_permutation(A: np.ndarray) -> np.ndarray:
    """
    Bidirectional Block Construction (BBC) — Algorithm 1, Kim et al. (2025).

    Permutes precision matrix rows/cols to expose two-block structure:
    high-row-sum (high-weight) stocks at the front, low-row-sum at the back.

    Parameters
    ----------
    A : (n, n) precision matrix (Sigma^{-1})

    Returns
    -------
    pi : (n,) int array — permutation index such that A[np.ix_(pi, pi)]
         shows the block structure.
    """
    n = A.shape[0]
    placed = np.zeros(n, dtype=bool)
    pi = np.empty(n, dtype=int)

    # 1. Largest |off-diagonal| entry → seed the high block
    A_abs = np.abs(A).copy()
    np.fill_diagonal(A_abs, -np.inf)
    i_star, j_star = np.unravel_index(np.argmax(A_abs), (n, n))
    pi[0], pi[1] = i_star, j_star
    placed[i_star] = placed[j_star] = True

    # 2. Antipodal anchor: minimum precision value to pi[0] → seeds the low block
    cands = np.where(~placed)[0]
    k_star = cands[np.argmin(A[pi[0], cands])]
    pi[n - 1] = k_star
    placed[k_star] = True

    # 3. Bidirectional fill
    l, r = 2, n - 2
    while l <= r:
        # Front: highest mean precision to current front group pi[0..l-1]
        cands = np.where(~placed)[0]
        front_scores = A[np.ix_(cands, pi[:l])].sum(axis=1) / l
        m = cands[np.argmax(front_scores)]
        pi[l] = m
        placed[m] = True
        l += 1

        if l <= r:
            # Back: highest mean precision to current back group pi[r+1..n-1]
            cands = np.where(~placed)[0]
            back_scores = A[np.ix_(cands, pi[r + 1:])].sum(axis=1) / (n - r)
            m = cands[np.argmax(back_scores)]
            pi[r] = m
            placed[m] = True
            r -= 1

    return pi


def _nearest_psd_corr(C: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the nearest PSD correlation matrix."""
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-8)
    C_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize to correlation matrix
    d = np.sqrt(np.diag(C_psd))
    d = np.where(d == 0, 1, d)
    C_corr = C_psd / np.outer(d, d)
    np.fill_diagonal(C_corr, 1.0)
    return C_corr
