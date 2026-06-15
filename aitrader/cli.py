"""Command-line interface: train, backtest, status, and trade."""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

from .config import Config, default_symbols
from .data import get_history, get_recent
from .model import TradingModel


def _symbols_arg(value: str) -> List[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def cmd_train(args) -> int:
    symbols = args.symbols or default_symbols()
    print(f"Fetching history for {len(symbols)} symbols: {', '.join(symbols)}")
    frames = []
    for s in symbols:
        try:
            frames.append(get_history(s, period=args.period, interval=args.interval))
            print(f"  {s}: ok")
        except Exception as exc:
            print(f"  {s}: SKIPPED ({exc})")
    if not frames:
        print("No data fetched; aborting.")
        return 1

    model = TradingModel(horizon=args.horizon, up_threshold=args.up_threshold)
    print("Training model...")
    metrics = model.train(frames)
    path = model.save()
    print("\nValidation metrics (chronological holdout):")
    for k, v in metrics.items():
        print(f"  {k:14}: {v:.4f}" if isinstance(v, float) else f"  {k:14}: {v}")
    print(f"\nSaved model -> {path}")
    print("Tip: accuracy near the base rate means little edge — tune horizon/threshold "
          "or features before trusting it with money.")
    return 0


def cmd_backtest(args) -> int:
    from .backtest import backtest

    cfg = Config()
    symbols = args.symbols or default_symbols()
    model = TradingModel.load()
    print(f"Backtesting {len(symbols)} symbols (cost/trade={args.cost:.2%})\n")
    rows = []
    for s in symbols:
        try:
            df = get_history(s, period=args.period, interval=args.interval)
            res = backtest(
                model, df, s,
                buy_confidence=cfg.buy_confidence,
                sell_confidence=cfg.sell_confidence,
                cost_per_trade=args.cost,
            )
            if res is None:
                print(f"{s:<10} insufficient data")
                continue
            print(res.summary())
            rows.append(res)
        except Exception as exc:
            print(f"{s:<10} ERROR ({exc})")
    if rows:
        avg = sum(r.total_return for r in rows) / len(rows)
        avg_bh = sum(r.buy_hold_return for r in rows) / len(rows)
        print(f"\nAverage: strat={avg:+.1%}  buy&hold={avg_bh:+.1%}")
    return 0


def cmd_status(args) -> int:
    from .broker import Broker

    cfg = Config()
    broker = Broker(cfg)
    print(f"Mode: {broker.mode}")
    print(f"Equity:        ${broker.equity():,.2f}")
    print(f"Buying power:  ${broker.buying_power():,.2f}")
    positions = broker.positions()
    if not positions:
        print("No open positions.")
        return 0
    print("\nPositions:")
    for p in positions.values():
        print(f"  {p.symbol:<10} qty={p.qty:<12.6f} value=${p.market_value:,.2f} "
              f"pl=${p.unrealized_pl:+,.2f}")
    return 0


def cmd_trade(args) -> int:
    from .broker import Broker
    from .trader import Trader

    cfg = Config()

    # --- live-money safety gate ---
    if not cfg.paper:
        if not args.i_understand_the_risk:
            print("REFUSING to trade LIVE money without --i-understand-the-risk.\n"
                  "Set ALPACA_PAPER=true to use paper money, or pass the flag to confirm.")
            return 2
        print("!! LIVE TRADING WITH REAL MONEY !!")

    model = TradingModel.load()
    broker = Broker(cfg)
    trader = Trader(cfg, model, broker)
    symbols = args.symbols or default_symbols()

    print(f"Mode: {broker.mode} | symbols: {', '.join(symbols)} | "
          f"alloc/position: {cfg.position_fraction:.0%} | dry_run: {args.dry_run}")

    iteration = 0
    while True:
        iteration += 1
        decisions = trader.run_once(symbols, dry_run=args.dry_run)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] pass #{iteration}")
        for d in decisions:
            proba = "  nan" if d.proba_up != d.proba_up else f"{d.proba_up:.3f}"
            print(f"  {d.symbol:<10} P(up)={proba}  -> {d.action.upper():<5} ({d.detail})")
        if args.once:
            break
        time.sleep(args.interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aitrader", description="ML-driven trader for stocks + crypto.")
    sub = p.add_subparsers(dest="command", required=True)

    common_period = dict(period="2y", interval="1d")

    t = sub.add_parser("train", help="Train the model on historical data.")
    t.add_argument("--symbols", type=_symbols_arg, help="Comma-separated, e.g. SPY,AAPL,BTC/USD")
    t.add_argument("--period", default="2y")
    t.add_argument("--interval", default="1d")
    t.add_argument("--horizon", type=int, default=5)
    t.add_argument("--up-threshold", type=float, default=0.0, dest="up_threshold")
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("backtest", help="Backtest the trained model.")
    b.add_argument("--symbols", type=_symbols_arg)
    b.add_argument("--period", default="2y")
    b.add_argument("--interval", default="1d")
    b.add_argument("--cost", type=float, default=0.0005, help="Per-trade cost fraction.")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("status", help="Show account + positions.")
    s.set_defaults(func=cmd_status)

    tr = sub.add_parser("trade", help="Run the live/paper trading loop.")
    tr.add_argument("--symbols", type=_symbols_arg)
    tr.add_argument("--once", action="store_true", help="Run a single pass and exit.")
    tr.add_argument("--dry-run", action="store_true", help="Decide but never place orders.")
    tr.add_argument("--interval", type=int, default=300, help="Seconds between passes.")
    tr.add_argument("--i-understand-the-risk", action="store_true", dest="i_understand_the_risk",
                    help="Required to place LIVE (real-money) orders.")
    tr.set_defaults(func=cmd_trade)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
