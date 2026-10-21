"""Data loading, log returns, supervised windows and the walk-forward splitter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "btc_usd_daily.csv"


def load_snapshot(path: str | Path | None = None) -> pd.DataFrame:
    """Load the committed single-header daily OHLCV snapshot (index = Date)."""
    df = pd.read_csv(path or DEFAULT_SNAPSHOT, index_col=0, parse_dates=True)
    return df.sort_index()


def download(start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """Optional live download via yfinance (not needed for the offline benchmark)."""
    import yfinance as yf

    raw = yf.download("BTC-USD", start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    return df


def log_returns(close: np.ndarray) -> np.ndarray:
    """r[k] = log(close[k+1] / close[k]); length n-1."""
    close = np.asarray(close, dtype=float)
    return np.diff(np.log(close))


def make_windows(returns: np.ndarray, lookback: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """X[i] = returns[i:i+lookback], y[i] = returns[i+lookback] (the next-day return)."""
    r = np.asarray(returns, dtype=float)
    m = len(r) - lookback
    if m <= 0:
        raise ValueError("series shorter than lookback")
    X = np.stack([r[i : i + lookback] for i in range(m)])
    return X, r[lookback:].copy()


def returns_to_prices(prev_close: np.ndarray, pred_returns: np.ndarray) -> np.ndarray:
    """Convert predicted next-day log returns to price level: prev_close * exp(r)."""
    return np.asarray(prev_close, dtype=float) * np.exp(np.asarray(pred_returns, dtype=float))


class Standardizer:
    """Single scalar mean/std (shared by all lags), fitted on training data only."""

    def fit(self, *arrays: np.ndarray) -> "Standardizer":
        flat = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
        self.mean_ = float(flat.mean())
        self.std_ = float(flat.std()) or 1.0
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, dtype=float) - self.mean_) / self.std_

    def inverse(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=float) * self.std_ + self.mean_


@dataclass
class Fold:
    """Sample indices of one walk-forward fold (indices into the window arrays)."""

    train_idx: np.ndarray  # all training samples (expanding window)
    fit_idx: np.ndarray  # train minus validation tail (used by the LSTM)
    val_idx: np.ndarray  # last val_frac of the train part (early stopping only)
    test_idx: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_idx)


def walk_forward_folds(
    n_samples: int, initial_train_frac: float = 0.6, n_folds: int = 5, val_frac: float = 0.1
) -> list[Fold]:
    """Expanding-window walk-forward folds.

    Fold k trains on samples [0, t_k) and tests on [t_k, t_k + size); the next fold's training set
    additionally contains the previous test block. Test blocks are consecutive and cover everything
    after the initial training part. The validation tail is the last val_frac of the train part.
    """
    first = int(n_samples * initial_train_frac)
    size = (n_samples - first) // n_folds
    if size < 1:
        raise ValueError("not enough samples for the requested folds")
    folds = []
    for k in range(n_folds):
        start = first + k * size
        stop = n_samples if k == n_folds - 1 else start + size
        n_val = max(1, int(round(start * val_frac)))
        folds.append(
            Fold(
                train_idx=np.arange(0, start),
                fit_idx=np.arange(0, start - n_val),
                val_idx=np.arange(start - n_val, start),
                test_idx=np.arange(start, stop),
            )
        )
    return folds


@dataclass
class Dataset:
    """Windows plus the price context needed to turn return forecasts into price forecasts."""

    X: np.ndarray
    y: np.ndarray
    prev_close: np.ndarray  # close the day before the target day
    actual_close: np.ndarray  # close on the target day
    dates: pd.DatetimeIndex  # target day


def build_dataset(df: pd.DataFrame, lookback: int = 60) -> Dataset:
    close = df["Close"].to_numpy(dtype=float)
    r = log_returns(close)
    X, y = make_windows(r, lookback)
    # sample i: target return r[lookback+i] = log(close[lookback+i+1] / close[lookback+i])
    return Dataset(
        X=X,
        y=y,
        prev_close=close[lookback : lookback + len(y)],
        actual_close=close[lookback + 1 : lookback + 1 + len(y)],
        dates=df.index[lookback + 1 : lookback + 1 + len(y)],
    )
