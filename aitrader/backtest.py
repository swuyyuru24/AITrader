"""A simple long/flat backtester to validate the model before risking capital.

Decision rule each bar: go (or stay) long when P(up) >= buy_confidence; go flat
when P(up) <= sell_confidence. Positions are entered/exited at the next bar's
close to avoid look-ahead. A per-trade cost models commission + slippage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, compute_features
from .model import TradingModel


@dataclass
class BacktestResult:
    symbol: str
    total_return: float
    buy_hold_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    equity_curve: pd.Series

    def summary(self) -> str:
        return (
            f"{self.symbol:<10} "
            f"strat={self.total_return:+.1%}  buy&hold={self.buy_hold_return:+.1%}  "
            f"sharpe={self.sharpe:.2f}  maxDD={self.max_drawdown:.1%}  "
            f"win={self.win_rate:.0%}  trades={self.n_trades}"
        )


def backtest(
    model: TradingModel,
    df: pd.DataFrame,
    symbol: str,
    buy_confidence: float = 0.55,
    sell_confidence: float = 0.45,
    cost_per_trade: float = 0.0005,
    periods_per_year: int = 252,
) -> Optional[BacktestResult]:
    feats = compute_features(df).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    if len(feats) < 30:
        return None

    proba = model.pipeline.predict_proba(feats[FEATURE_COLUMNS])[:, 1]
    feats = feats.assign(proba=proba)

    # Build target position (1 long / 0 flat) with hysteresis between thresholds.
    position = np.zeros(len(feats))
    holding = 0
    for i, p in enumerate(feats["proba"].to_numpy()):
        if holding == 0 and p >= buy_confidence:
            holding = 1
        elif holding == 1 and p <= sell_confidence:
            holding = 0
        position[i] = holding
    feats["position"] = position

    # Trade on next bar: position decided at close of bar t applies to return t->t+1.
    bar_return = feats["close"].pct_change().shift(-1).fillna(0.0)
    applied_pos = feats["position"]
    trades = applied_pos.diff().abs().fillna(applied_pos.abs())
    strat_return = applied_pos * bar_return - trades * cost_per_trade

    equity = (1 + strat_return).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    buy_hold = float((1 + bar_return).cumprod().iloc[-1] - 1)

    std = strat_return.std()
    sharpe = float(np.sqrt(periods_per_year) * strat_return.mean() / std) if std > 0 else 0.0
    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1).min())

    # Win rate over completed long holdings.
    trade_returns = []
    entry_equity = None
    for pos, eq in zip(applied_pos.to_numpy(), equity.to_numpy()):
        if pos == 1 and entry_equity is None:
            entry_equity = eq
        elif pos == 0 and entry_equity is not None:
            trade_returns.append(eq / entry_equity - 1)
            entry_equity = None
    win_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in trade_returns])) if trade_returns else 0.0

    return BacktestResult(
        symbol=symbol,
        total_return=total_return,
        buy_hold_return=buy_hold,
        sharpe=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        n_trades=int(trades.sum()),
        equity_curve=equity,
    )
