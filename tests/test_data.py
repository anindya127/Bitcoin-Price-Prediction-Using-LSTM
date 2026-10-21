import numpy as np
import pytest

from btc_forecast import data

from conftest import make_df


def test_log_returns_values():
    r = data.log_returns(np.array([100.0, 110.0, 99.0]))
    assert np.allclose(r, [np.log(1.1), np.log(0.9)])


def test_window_shapes_and_alignment():
    r = np.arange(100, dtype=float)
    X, y = data.make_windows(r, 60)
    assert X.shape == (40, 60) and y.shape == (40,)
    assert np.array_equal(X[0], r[:60]) and y[0] == 60 and y[-1] == 99


def test_window_too_short():
    with pytest.raises(ValueError):
        data.make_windows(np.zeros(10), 60)


def test_return_price_round_trip():
    close = np.array([100.0, 103.0, 98.0, 120.0])
    r = data.log_returns(close)
    assert np.allclose(data.returns_to_prices(close[:-1], r), close[1:])


def test_dataset_alignment(random_walk_df):
    ds = data.build_dataset(random_walk_df, 60)
    assert np.allclose(np.log(ds.actual_close / ds.prev_close), ds.y)
    # each sample's last feature is the previous sample's target: features never contain the target
    assert np.allclose(ds.X[1:, -1], ds.y[:-1])
    assert len(ds.dates) == len(ds.y)


@pytest.mark.parametrize("n,folds,frac", [(1000, 5, 0.6), (500, 3, 0.5), (777, 4, 0.7)])
def test_folds_no_overlap_or_lookahead(n, folds, frac):
    fs = data.walk_forward_folds(n, frac, folds)
    assert len(fs) == folds
    for f in fs:
        assert f.train_idx.max() < f.test_idx.min()
        assert f.val_idx.max() < f.test_idx.min()
        assert not set(f.fit_idx) & set(f.val_idx)
        assert f.fit_idx.max() < f.val_idx.min()
        assert len(f.val_idx) >= 1


def test_folds_consecutive_and_expanding():
    fs = data.walk_forward_folds(1000, 0.6, 5)
    for a, b in zip(fs, fs[1:]):
        assert b.test_idx.min() == a.test_idx.max() + 1
        assert b.n_train > a.n_train
        assert b.train_idx.max() + 1 == b.test_idx.min()
    assert fs[-1].test_idx.max() == 999
    assert fs[0].test_idx.min() == 600


def test_standardizer_train_only():
    train = np.array([1.0, 2.0, 3.0])
    test = np.array([100.0, 200.0])
    s = data.Standardizer().fit(train)
    assert s.mean_ == pytest.approx(2.0)
    s2 = data.Standardizer().fit(train)
    s2.fit(train)  # refit on the same data: unaffected by test values
    assert np.allclose(s.transform(test), s2.transform(test))
    assert np.allclose(s.inverse(s.transform(test)), test)


def test_snapshot_loads_if_present():
    df = data.load_snapshot()
    assert {"Open", "High", "Low", "Close", "Volume"} <= set(df.columns)
    assert df.index.is_monotonic_increasing and (df["Close"] > 0).all()
