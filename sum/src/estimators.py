"""Covariance estimators. Signature: (returns) -> ndarray."""
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_cov(returns: pd.DataFrame):
    return returns.cov().values


def lw_cov(returns: pd.DataFrame):
    """Ledoit-Wolf (2004) linear shrinkage covariance."""
    return LedoitWolf().fit(returns.values).covariance_
