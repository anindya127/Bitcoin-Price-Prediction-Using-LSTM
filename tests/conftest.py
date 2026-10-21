import numpy as np
import pandas as pd
import pytest


def make_df(returns, start=100.0):
    close = start * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D", name="Date")
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}, index=idx)


@pytest.fixture
def random_walk_df():
    r = np.random.default_rng(0).normal(0, 0.02, 700)
    return make_df(r)


@pytest.fixture
def ar1_df():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 0.01, 1200)
    r = np.zeros(1200)
    for t in range(1, 1200):
        r[t] = 0.7 * r[t - 1] + e[t]
    return make_df(r)
