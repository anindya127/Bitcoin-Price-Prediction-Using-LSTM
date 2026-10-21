import numpy as np
import pytest

from btc_forecast import metrics


def test_rmse_mae_known():
    assert metrics.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert metrics.mae([0, 0], [3, -4]) == pytest.approx(3.5)


def test_directional_accuracy():
    assert metrics.directional_accuracy([1, -1, 1, -1], [2, -2, -1, 1]) == 0.5
    assert metrics.directional_accuracy([1, 1], [1, 1]) == 1.0


def test_directional_accuracy_zero_forecast_is_nan():
    assert np.isnan(metrics.directional_accuracy([1, -1], [0, 0]))


def test_up_share():
    assert metrics.up_share([1, 1, -1, 0]) == 0.5


def test_rmse_ratio():
    assert metrics.rmse_ratio(2.0, 4.0) == 0.5


def test_evaluate_perfect_forecast():
    prev = np.array([100.0, 101.0])
    r = np.array([0.01, -0.02])
    m = metrics.evaluate(r, r, prev, prev * np.exp(r))
    assert m["rmse_usd"] == pytest.approx(0) and m["dir_acc"] == 1.0 and m["n"] == 2


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    actual = rng.normal(size=500)
    good = actual + rng.normal(0, 0.1, 500)
    bad = actual + rng.normal(0, 1.0, 500)
    stat, p = metrics.diebold_mariano(actual, good, bad)
    assert stat < 0 and p < 0.01


def test_diebold_mariano_identical_is_nan():
    a = np.arange(5.0)
    stat, p = metrics.diebold_mariano(a, a + 1, a + 1)
    assert np.isnan(stat) and np.isnan(p)


def test_diebold_mariano_no_difference_large_p():
    rng = np.random.default_rng(1)
    actual = rng.normal(size=2000)
    f1 = actual + rng.normal(0, 1, 2000)
    f2 = actual + rng.normal(0, 1, 2000)
    assert metrics.diebold_mariano(actual, f1, f2)[1] > 0.01
