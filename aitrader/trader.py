"""The live trading loop: model predictions -> orders through the broker.

For each symbol it pulls recent bars, computes features, asks the model for
P(up), and:
  * buys (allocating `position_fraction` of buying power) when confident and flat;
  * closes the position when confidence drops below the sell threshold.

Every intended action is logged. In paper mode this is harmless; in live mode the
CLI forces an explicit risk acknowledgement before this ever runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .broker import Broker
from .config import Config
from .data import get_recent
from .features import FEATURE_COLUMNS, compute_features
from .model import TradingModel


@dataclass
class Decision:
    symbol: str
    proba_up: float
    action: str  # "buy" | "close" | "hold"
    detail: str


class Trader:
    def __init__(self, config: Config, model: TradingModel, broker: Broker):
        self.config = config
        self.model = model
        self.broker = broker

    def decide(self, symbol: str, proba: float, holding: bool) -> Decision:
        if not holding and proba >= self.config.buy_confidence:
            return Decision(symbol, proba, "buy", "confidence above buy threshold")
        if holding and proba <= self.config.sell_confidence:
            return Decision(symbol, proba, "close", "confidence below sell threshold")
        return Decision(symbol, proba, "hold", "no threshold crossed")

    def run_once(self, symbols: List[str], dry_run: bool = False) -> List[Decision]:
        """One pass over the universe. Returns the decisions taken."""
        decisions: List[Decision] = []
        positions = self.broker.positions()
        buying_power = self.broker.buying_power()
        alloc = buying_power * self.config.position_fraction

        for symbol in symbols:
            try:
                recent = get_recent(symbol)
                feats = compute_features(recent).dropna(subset=FEATURE_COLUMNS)
                if feats.empty:
                    decisions.append(Decision(symbol, float("nan"), "hold", "insufficient data"))
                    continue
                proba = self.model.predict_proba_up(feats.tail(1))
                holding = symbol.replace("/", "").upper() in positions
                decision = self.decide(symbol, proba, holding)

                if decision.action != "hold" and not self.broker.can_trade_now(symbol):
                    decision = Decision(symbol, proba, "hold", "market closed")

                if not dry_run:
                    if decision.action == "buy":
                        self.broker.buy_notional(symbol, alloc)
                    elif decision.action == "close":
                        self.broker.close(symbol)

                decisions.append(decision)
            except Exception as exc:  # never let one symbol crash the whole loop
                decisions.append(Decision(symbol, float("nan"), "error", str(exc)))

        return decisions
