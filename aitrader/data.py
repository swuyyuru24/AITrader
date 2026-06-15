"""Market data access.

Historical/training data comes from yfinance (free, keyless). Live execution
uses Alpaca (see broker.py). yfinance also serves as a keyless source of recent
bars for the live loop, so the system is usable before wiring up Alpaca data.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def _to_yf_symbol(symbol: str) -> str:
    """Map our symbols to yfinance tickers ('BTC/USD' -> 'BTC-USD')."""
    if "/" in symbol:
        base, quote = symbol.split("/")
        return f"{base}-{quote}"
    return symbol


def get_history(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV history as a DataFrame with lowercase columns: open/high/low/close/volume."""
    ticker = _to_yf_symbol(symbol)
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {symbol} ({ticker}).")

    # yfinance may return MultiIndex columns for a single ticker; flatten them.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    cols = ["open", "high", "low", "close", "volume"]
    df = raw[[c for c in cols if c in raw.columns]].copy()
    df.index = pd.to_datetime(df.index)
    return df.dropna()


def get_recent(symbol: str, lookback_bars: int = 120, interval: str = "1d") -> pd.DataFrame:
    """Recent bars for live inference. Pulls a generous window then trims."""
    period = "6mo" if interval in ("1d", "1wk") else "60d"
    df = get_history(symbol, period=period, interval=interval)
    return df.tail(lookback_bars)
