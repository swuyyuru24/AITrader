"""Configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional at runtime
    pass

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass
class Config:
    # --- Broker credentials ---
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    # Safety gate: paper money unless explicitly set to "false".
    paper: bool = field(default_factory=lambda: os.getenv("ALPACA_PAPER", "true").lower() != "false")

    # --- Strategy / model ---
    # Predict whether price will rise more than `up_threshold` over `horizon` bars.
    horizon: int = 5
    up_threshold: float = 0.0
    # Only act when the model's confidence clears this probability.
    buy_confidence: float = 0.55
    sell_confidence: float = 0.45
    # Fraction of buying power to allocate per position.
    position_fraction: float = 0.1

    def is_crypto(self, symbol: str) -> bool:
        """Alpaca crypto symbols look like 'BTC/USD'."""
        return "/" in symbol

    def validate_credentials(self) -> None:
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise RuntimeError(
                "Missing Alpaca credentials. Copy .env.example to .env and fill in "
                "ALPACA_API_KEY / ALPACA_SECRET_KEY (paper keys while testing)."
            )


def default_symbols() -> List[str]:
    """A small default universe spanning both asset classes the user asked for."""
    return ["SPY", "AAPL", "BTC/USD", "ETH/USD"]
