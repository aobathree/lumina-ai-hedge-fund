# -*- coding: utf-8 -*-
"""
Alpaca（マーケットデータ API）の日足を、Lumina が使う米国の株・ETF について取って控える。

対象は、スクリーナーの米国の銘柄（``MarketScreener.UNIVERSE`` の ``.T`` 以外）と、オプション乱気流の判定に使う SPY・QQQ。
鍵は環境変数 ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY（1Password の参照なら ``op run`` で渡す）。

    op run --env-file ~/kabs-local/n3/env -- .venv/bin/python scripts/alpaca_fetch.py \\
        --from 2024-11-01 --out .trial/alpaca

Basic の契約は直近 15 分の SIP を取れないので、--to を省くと「いまより 20 分前」まで取る。
Alpaca に無い銘柄（OTC の ADR など）は 0 行になり、Lumina は yfinance に落とす。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="frm", required=True)
    ap.add_argument("--to", help="RFC 3339（例 2026-11-20T00:00:00Z）。既定はいまより 20 分前")
    ap.add_argument("--out", default=".trial/alpaca")
    ap.add_argument("--symbols", nargs="*", help="既定はスクリーナーの米国の銘柄 + SPY・QQQ")
    args = ap.parse_args()

    from core.ai_hedge_fund.alpaca_source import AlpacaStore, is_us_equity
    from core.ai_hedge_fund.market_screener import MarketScreener

    store = AlpacaStore(Path(args.out))
    if not store.has_keys():
        print("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY が無い（op run --env-file … -- で呼ぶ）", file=sys.stderr)
        return 2
    symbols = args.symbols or sorted({i["symbol"] for items in MarketScreener.UNIVERSE.values() for i in items
                                      if is_us_equity(i["symbol"])} | {"SPY", "QQQ"})
    got = {}
    for i in range(0, len(symbols), 20):
        got.update(store.download(symbols[i:i + 20], start=args.frm, end=args.to))
    for sym in symbols:
        rows = got.get(sym.upper(), [])
        span = f"{rows[0]['t'][:10]}〜{rows[-1]['t'][:10]}" if rows else "Alpaca に無い（yfinance に落ちる）"
        print(f"{sym:7s} {len(rows):4d} 日  {span}")
    print(f"--- {len(symbols)} 銘柄 / 行のある銘柄 {sum(1 for s in symbols if got.get(s.upper()))} → {Path(args.out) / 'us_daily'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
