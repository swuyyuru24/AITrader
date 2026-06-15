# AITrader

An ML-driven trading system for **US stocks/ETFs and crypto**, executing through
[Alpaca](https://alpaca.markets). A gradient-boosting model is trained on technical
indicators to predict near-term price direction; a backtester validates the strategy,
and a trading loop turns predictions into orders.

> ⚠️ **Real money is involved when you go live.** The system defaults to Alpaca's
> **paper-trading** endpoint. It will only place real orders when you both set
> `ALPACA_PAPER=false` *and* pass `--i-understand-the-risk`. ML signals on noisy
> markets frequently perform no better than buy-and-hold — **backtest first, paper-trade
> long, and risk only what you can lose.** This is software, not financial advice.

## How it works

```
data.py      OHLCV history (yfinance, keyless) + recent bars for live inference
features.py  ~12 technical indicators (returns, SMA/EMA ratios, RSI, MACD, Bollinger, …)
model.py     StandardScaler + GradientBoostingClassifier -> P(price up over N bars)
backtest.py  long/flat simulation with costs; reports return, Sharpe, drawdown, win rate
broker.py    Alpaca wrapper unifying equities (market-hours) and crypto (24/7, GTC)
trader.py    prediction -> position-sizing -> orders, one symbol at a time
cli.py       `aitrader train | backtest | status | trade`
```

The label is "did price rise more than `--up-threshold` over the next `--horizon`
bars." Validation uses a **chronological** holdout (no shuffling) so the model is
never scored on data from before its training window.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your Alpaca PAPER keys
```

## Usage

```bash
# 1. Train on 2 years of daily bars for the default universe (SPY, AAPL, BTC/USD, ETH/USD)
python -m aitrader train

# 2. Backtest the trained model (validate before risking anything)
python -m aitrader backtest

# 3. Check your Alpaca account + positions
python -m aitrader status

# 4. Paper-trade — dry run first (decides but places no orders)
python -m aitrader trade --once --dry-run
python -m aitrader trade --interval 300          # loop every 5 minutes (paper)
```

Custom universe / horizon:

```bash
python -m aitrader train --symbols SPY,QQQ,BTC/USD,ETH/USD --horizon 10 --up-threshold 0.01
python -m aitrader backtest --symbols SPY,BTC/USD
```

## Going live (only when you mean it)

```bash
# in .env: ALPACA_PAPER=false  + your LIVE keys
python -m aitrader trade --i-understand-the-risk
```

## Tests

```bash
pip install pytest
pytest          # fully offline: synthetic data, no keys needed
```

## Key knobs (`aitrader/config.py`)

| Setting | Meaning |
|---|---|
| `horizon` | bars ahead the model predicts |
| `buy_confidence` / `sell_confidence` | P(up) thresholds to enter / exit |
| `position_fraction` | fraction of buying power per position |
| `paper` | safety gate; `false` = real money |
