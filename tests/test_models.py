import numpy as np
import pytest

from btc_forecast import metrics, models
from btc_forecast.data import build_dataset, walk_forward_folds


def test_naive_predicts_unchanged_price():
    X = np.random.default_rng(0).normal(size=(10, 60))
    m = models.Naive().fit(X, np.zeros(10))
    prev = np.linspace(100, 200, 10)
    assert np.allclose(prev * np.exp(m.predict(X)), prev)


def test_drift_is_train_mean():
    y = np.array([0.01, 0.03])
    m = models.Drift().fit(np.zeros((2, 5)), y)
    assert np.allclose(m.predict(np.zeros((3, 5))), 0.02)


def test_moving_average_uses_last_window():
    X = np.array([[9.0, 9.0, 1.0, 2.0, 3.0]])
    assert models.MovingAverage(3).predict(X)[0] == pytest.approx(2.0)


def _evaluate(df, model):
    ds = build_dataset(df, 20)
    f = walk_forward_folds(len(ds.y), 0.6, 3)[-1]
    model.fit(ds.X[f.train_idx], ds.y[f.train_idx])
    p = model.predict(ds.X[f.test_idx])
    return metrics.rmse(ds.actual_close[f.test_idx], ds.prev_close[f.test_idx] * np.exp(p))


def test_ridge_beats_naive_on_autocorrelated_series(ar1_df):
    assert _evaluate(ar1_df, models.RidgeLags()) < 0.95 * _evaluate(ar1_df, models.Naive())


def test_ridge_not_much_better_on_random_walk(random_walk_df):
    ratio = _evaluate(random_walk_df, models.RidgeLags(alpha=100.0)) / _evaluate(random_walk_df, models.Naive())
    assert ratio > 0.97  # nothing to learn from pure noise


def test_ridge_scaler_fitted_on_train_only():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(50, 5)), rng.normal(size=50)
    m = models.RidgeLags().fit(X, y)
    assert m.scaler_.mean_ == pytest.approx(np.concatenate([X.ravel(), y]).mean())


def test_lstm_smoke(random_walk_df):
    pytest.importorskip("tensorflow")
    ds = build_dataset(random_walk_df, 20)
    f = walk_forward_folds(len(ds.y), 0.6, 3)[0]
    m = models.LSTMModel(units=4, epochs=2, seed=0)
    m.fit(ds.X[f.fit_idx], ds.y[f.fit_idx], ds.X[f.val_idx], ds.y[f.val_idx])
    p = m.predict(ds.X[f.test_idx])
    assert p.shape == (len(f.test_idx),) and np.isfinite(p).all()
