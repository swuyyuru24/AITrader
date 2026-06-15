"""Alpaca brokerage wrapper covering both US equities and crypto.

Equities and crypto differ in two ways that matter here:
  * crypto orders must use time-in-force GTC (DAY is rejected);
  * crypto trades 24/7 while equities respect market hours.
This wrapper hides those differences behind a small, uniform interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from .config import Config


def _normalize(symbol: str) -> str:
    """Compare symbols ignoring the crypto slash ('BTC/USD' == 'BTCUSD')."""
    return symbol.replace("/", "").upper()


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    market_value: float
    unrealized_pl: float


class Broker:
    def __init__(self, config: Config):
        config.validate_credentials()
        self.config = config
        self.client = TradingClient(
            config.alpaca_api_key,
            config.alpaca_secret_key,
            paper=config.paper,
        )

    @property
    def mode(self) -> str:
        return "PAPER" if self.config.paper else "LIVE"

    def buying_power(self) -> float:
        return float(self.client.get_account().buying_power)

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def positions(self) -> Dict[str, PositionInfo]:
        out: Dict[str, PositionInfo] = {}
        for p in self.client.get_all_positions():
            out[_normalize(p.symbol)] = PositionInfo(
                symbol=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
            )
        return out

    def has_position(self, symbol: str) -> bool:
        return _normalize(symbol) in self.positions()

    def market_is_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def can_trade_now(self, symbol: str) -> bool:
        """Crypto is always tradable; equities only during market hours."""
        if self.config.is_crypto(symbol):
            return True
        return self.market_is_open()

    def buy_notional(self, symbol: str, dollars: float) -> Optional[object]:
        """Buy `dollars` worth of `symbol` with a market order. Returns the order."""
        if dollars <= 0:
            return None
        tif = TimeInForce.GTC if self.config.is_crypto(symbol) else TimeInForce.DAY
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(dollars, 2),
            side=OrderSide.BUY,
            time_in_force=tif,
        )
        return self.client.submit_order(req)

    def close(self, symbol: str) -> Optional[object]:
        """Liquidate the entire position in `symbol`, if any."""
        if not self.has_position(symbol):
            return None
        return self.client.close_position(_normalize(symbol))
