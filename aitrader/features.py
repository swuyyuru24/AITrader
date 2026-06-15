"""Technical-indicator feature engineering.

Indicators are computed with plain pandas/numpy so there is no dependency on a
TA library (which keeps installs reliable across platforms).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# The model is trained on exactly these columns, in this order.
FEATURE_COLUMNS: List[str] = [
    "ret_1",
    "ret_5",
    "ret_10",
    "sma_ratio_10",
    "sma_ratio_30",
    "ema_ratio_12_26",
    "rsi_14",
    "macd_hist",
    "bb_pos_20",
    "volatility_10",
    "volume_change",
    "momentum_10",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns to an OHLCV frame (columns: open, high, low, close, volume)."""
    out = df.copy()
    close = out["close"]
    volume = out["volume"]

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    sma_10 = close.rolling(10).mean()
    sma_30 = close.rolling(30).mean()
    out["sma_ratio_10"] = close / sma_10 - 1
    out["sma_ratio_30"] = close / sma_30 - 1

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["ema_ratio_12_26"] = ema_12 / ema_26 - 1

    out["rsi_14"] = _rsi(close, 14) / 100.0

    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - signal) / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_pos_20"] = (close - bb_mid) / (2 * bb_std).replace(0.0, np.nan)

    out["volatility_10"] = out["ret_1"].rolling(10).std()
    out["volume_change"] = volume.pct_change(5).replace([np.inf, -np.inf], np.nan)
    out["momentum_10"] = close / close.shift(10) - 1

    return out


def make_labels(df: pd.DataFrame, horizon: int, up_threshold: float) -> pd.Series:
    """1 if forward return over `horizon` bars exceeds `up_threshold`, else 0."""
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    return (future_return > up_threshold).astype(int)


def build_dataset(df: pd.DataFrame, horizon: int, up_threshold: float):
    """Return (X, y) ready for training, with rows containing NaNs dropped."""
    feats = compute_features(df)
    feats["label"] = make_labels(feats, horizon, up_threshold)
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["label"])
    X = feats[FEATURE_COLUMNS]
    y = feats["label"]
    return X, y
