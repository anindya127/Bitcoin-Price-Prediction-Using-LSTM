"""Forecast metrics (price level in USD, return level, direction) and a Diebold-Mariano test."""
from __future__ import annotations

import math

import numpy as np


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def directional_accuracy(actual_ret, pred_ret) -> float:
    """Share of days where sign(pred) == sign(actual). NaN if the forecast never takes a side (all zeros)."""
    a, p = np.sign(np.asarray(actual_ret, float)), np.sign(np.asarray(pred_ret, float))
    if not np.any(p != 0):
        return float("nan")
    return float(np.mean(a == p))


def up_share(actual_ret) -> float:
    """Fraction of up-days: the accuracy of the trivial 'always up' forecast."""
    return float(np.mean(np.asarray(actual_ret, float) > 0))


def rmse_ratio(model_rmse: float, naive_rmse: float) -> float:
    return float(model_rmse / naive_rmse)


def diebold_mariano(actual, pred, pred_naive) -> tuple[float, float]:
    """Diebold-Mariano test on squared errors for one-step forecasts (no autocorrelation correction needed
    at horizon 1 beyond the sample variance). Returns (statistic, two-sided normal p-value).
    Negative statistic = model has lower squared error than naive."""
    a = np.asarray(actual, float)
    d = (a - np.asarray(pred, float)) ** 2 - (a - np.asarray(pred_naive, float)) ** 2
    n = len(d)
    var = d.var(ddof=1)
    if n < 2 or var == 0:
        return float("nan"), float("nan")
    stat = d.mean() / math.sqrt(var / n)
    return float(stat), float(math.erfc(abs(stat) / math.sqrt(2)))


def evaluate(actual_ret, pred_ret, prev_close, actual_close) -> dict:
    """All metrics for one set of forecasts (price forecast = prev_close * exp(pred_ret))."""
    pred_price = np.asarray(prev_close, float) * np.exp(np.asarray(pred_ret, float))
    return {
        "rmse_usd": rmse(actual_close, pred_price),
        "mae_usd": mae(actual_close, pred_price),
        "rmse_ret": rmse(actual_ret, pred_ret),
        "mae_ret": mae(actual_ret, pred_ret),
        "dir_acc": directional_accuracy(actual_ret, pred_ret),
        "up_share": up_share(actual_ret),
        "n": int(len(actual_ret)),
    }
