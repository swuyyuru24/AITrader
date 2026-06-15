"""Offline tests: synthetic OHLCV exercises features -> model -> backtest.

No network or API keys required.
"""

import numpy as np
import pandas as pd

from aitrader.backtest import backtest
from aitrader.features import FEATURE_COLUMNS, build_dataset, compute_features
from aitrader.model import TradingModel


def synthetic_ohlcv(n=600, seed=0):
    rng = np.random.default_rng(seed)
    # Trending random walk with momentum so an ML model can find *some* signal.
    steps = rng.normal(0.0005, 0.01, n)
    steps[1:] += 0.15 * steps[:-1]
    close = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=idx,
    )


def test_features_have_no_nans_after_warmup():
    df = synthetic_ohlcv()
    feats = compute_features(df)
    assert set(FEATURE_COLUMNS).issubset(feats.columns)
    assert feats[FEATURE_COLUMNS].iloc[40:].notna().all().all()


def test_build_dataset_shapes_align():
    df = synthetic_ohlcv()
    X, y = build_dataset(df, horizon=5, up_threshold=0.0)
    assert len(X) == len(y) and len(X) > 100
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()).issubset({0, 1})


def test_train_predict_and_backtest_run():
    frames = [synthetic_ohlcv(seed=s) for s in range(3)]
    model = TradingModel(horizon=5)
    metrics = model.train(frames)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["train_rows"] > 0 and metrics["test_rows"] > 0

    proba = model.predict_proba_up(compute_features(frames[0]).dropna(subset=FEATURE_COLUMNS).tail(1))
    assert 0.0 <= proba <= 1.0

    result = backtest(model, frames[0], "SYNTH")
    assert result is not None
    assert result.equity_curve.iloc[-1] > 0


def test_model_save_load_roundtrip(tmp_path):
    frames = [synthetic_ohlcv(seed=s) for s in range(2)]
    model = TradingModel(horizon=3)
    model.train(frames)
    path = model.save(tmp_path / "m.joblib")
    loaded = TradingModel.load(path)
    row = compute_features(frames[0]).dropna(subset=FEATURE_COLUMNS).tail(1)
    assert abs(loaded.predict_proba_up(row) - model.predict_proba_up(row)) < 1e-9
