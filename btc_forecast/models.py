"""Forecasters with a common interface: fit(X, y, X_val, y_val) then predict(X) -> next-day log return.

X has shape (n, lookback): the previous `lookback` daily log returns, oldest first.
"""
from __future__ import annotations

import numpy as np

from .data import Standardizer


class Model:
    name = "model"
    deterministic = True

    def fit(self, X, y, X_val=None, y_val=None) -> "Model":
        return self

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError


class Naive(Model):
    """Tomorrow's price equals today's: predicted log return is zero."""

    name = "Naive (random walk)"

    def predict(self, X):
        return np.zeros(len(X))


class Drift(Model):
    """Mean log return of the training targets."""

    name = "Drift"

    def fit(self, X, y, X_val=None, y_val=None):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)


class MovingAverage(Model):
    """Mean of the most recent `window` returns."""

    def __init__(self, window: int = 5):
        self.window = window
        self.name = f"Moving average ({window}d)"

    def predict(self, X):
        return np.asarray(X)[:, -self.window :].mean(axis=1)


class RidgeLags(Model):
    """Ridge regression on the lagged returns (inputs and target standardised on the train part)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.name = "Ridge (lagged returns)"

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.linear_model import Ridge

        self.scaler_ = Standardizer().fit(X, y)
        self.reg_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(X), self.scaler_.transform(y))
        return self

    def predict(self, X):
        return self.scaler_.inverse(self.reg_.predict(self.scaler_.transform(X)))


class LSTMModel(Model):
    """Single-layer Keras LSTM; early stopping on the validation tail; tensorflow imported lazily."""

    deterministic = False

    def __init__(self, units: int = 32, epochs: int = 30, patience: int = 5, batch_size: int = 64, seed: int = 0):
        self.units, self.epochs, self.patience, self.batch_size, self.seed = units, epochs, patience, batch_size, seed
        self.name = f"LSTM ({units} units)"

    def fit(self, X, y, X_val=None, y_val=None):
        import keras
        from keras import layers

        keras.utils.set_random_seed(self.seed)
        self.scaler_ = Standardizer().fit(X, y)
        Xs = self.scaler_.transform(X)[..., None]
        ys = self.scaler_.transform(y)
        self.model_ = keras.Sequential(
            [layers.Input(shape=(Xs.shape[1], 1)), layers.LSTM(self.units), layers.Dropout(0.1), layers.Dense(1)]
        )
        self.model_.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        kwargs = {}
        callbacks = []
        if X_val is not None and len(X_val):
            kwargs["validation_data"] = (self.scaler_.transform(X_val)[..., None], self.scaler_.transform(y_val))
            callbacks.append(
                keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True, monitor="val_loss")
            )
        hist = self.model_.fit(
            Xs, ys, epochs=self.epochs, batch_size=self.batch_size, shuffle=False, verbose=0,
            callbacks=callbacks, **kwargs,
        )
        self.epochs_run_ = len(hist.history["loss"])
        return self

    def predict(self, X):
        out = self.model_.predict(self.scaler_.transform(X)[..., None], verbose=0).ravel()
        return self.scaler_.inverse(out)
