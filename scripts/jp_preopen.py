# -*- coding: utf-8 -*-
"""
過去の日の寄前に、Lumina の自律サイクルを 1 回回し、寄りで出す注文（寄前の戦略）を JSON に書く（as-of の再生）。

- 価格は J-Quants の日足（--asof の日まで。scripts/jq_fetch.py で取る）。それより後の値は見せない（asof_replay）
- 立会の判定は --session の時刻の東証の立会中として答える（寄りの成行 = 寄付の注文）
- 板は BRiSK の寄前の最後の気配（scripts/brisk_extract_win.py preopen）。フォークの深層 OFI は寄前の板を隔離する
- 台帳は新しい paper 台帳（--capital の円だけ）。Webull の API も実際の証券会社も呼ばない

    .venv/bin/python scripts/jp_preopen.py --asof 2026-11-19 --session 2026-11-20T09:00 \\
        --jq .trial/jq --preopen .trial/brisk/20261120/preopen.json \\
        --capital 15000000 --out .trial/replay-20261120/rules

LLM を使うときは op run で ANTHROPIC_API_KEY を渡す（TRIAL.md）。--llm off（既定）は鍵を環境から外して縮退モードで回す。
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JST = dt.timezone(dt.timedelta(hours=9))


def build_broker(db_path: str, preopen_path: str):
    from core.ai_hedge_fund.broker_client import PaperTradingEngine
    from core.ai_hedge_fund.deep_ofi_analyzer import DeepOFIAnalyzer

    books = {}
    if preopen_path:
        books = json.loads(Path(preopen_path).read_text(encoding="utf-8")).get("codes", {})

    class ReplayBroker(PaperTradingEngine):
        """paper 台帳に、寄前の板と、出した注文の控えを足したもの。"""

        placed = []

        def get_order_book(self, symbol, depth=10):
            code = symbol.replace(".T", "")
            book = (books.get(code) or {}).get("book")
            if not book:
                return super().get_order_book(symbol, depth)
            bids = [{"price": l["price"], "size": l["size"], "orders": l["orders"]} for l in book["bids"][:depth] if l["price"]]
            asks = [{"price": l["price"], "size": l["size"], "orders": l["orders"]} for l in book["asks"][:depth] if l["price"]]
            ofi = DeepOFIAnalyzer(symbol=symbol).update(bids, asks, timestamp=book["ts_unix"],
                                                        is_special_quote=bool(book.get("quote_flag")))
            return {"available": True, "symbol": symbol, "bids": bids, "asks": asks, "ofi_analysis": ofi,
                    "source": f"BRiSK 寄前 {book['ts_jst']}", "indicative_price": book.get("indicative_price")}

        def place_order(self, symbol, side, quantity, price=None, **kw):
            res = super().place_order(symbol, side, quantity, price, **kw)
            self.placed.append({
                "symbol": symbol, "side": side, "quantity": quantity, "ref_price": price,
                "target_price": kw.get("target_price"), "stop_loss_price": kw.get("stop_loss_price"),
                "notes": kw.get("notes"), "success": bool(res.get("success")),
                "error": res.get("error") or (None if res.get("success") else res.get("message")),
            })
            return res

    return ReplayBroker(db_path=db_path)


def install_council_cache(path: Path, calls: list) -> None:
    """合議（HedgeFundCouncil.run_council）を、保存した判定を返すものに替える。

    同じ判定で、スクリーナーの同点の並びだけを変えて比べるため（Claude の判定は呼ぶたびに揺れうる）。
    控えに無い銘柄は HOLD（0 点）にし、calls に「控えなし」として残す。
    """
    from core.ai_hedge_fund.hedge_fund_council import HedgeFundCouncil

    saved = json.loads(path.read_text(encoding="utf-8")).get("results", {})

    def _run(cls, symbol, *a, **k):
        r = saved.get(symbol)
        calls.append({"symbol": symbol, "cached": r is not None})
        if r is None:
            return {"verdict": {"action": "HOLD", "action_label": "合議の控えなし", "confidence_score": 0},
                    "currency": "JPY", "current_price": None}
        return {"verdict": r.get("verdict") or {k2: r.get(k2) for k2 in ("action", "action_label", "confidence_score",
                                                                            "target_price", "stop_loss_price", "allocation_pct",
                                                                            "thesis_summary")},
                "currency": r.get("currency", "JPY"), "current_price": r.get("current_price")}

    HedgeFundCouncil.run_council = classmethod(_run)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", required=True, help="判断に使う最後の日（YYYY-MM-DD）")
    ap.add_argument("--session", required=True, help="注文を出す時刻（JST、YYYY-MM-DDTHH:MM）")
    ap.add_argument("--news-cutoff", help="これより前のニュースだけ使う（JST）。既定は --session の日の 08:00")
    ap.add_argument("--jq", required=True, help="J-Quants の控えの置き場（scripts/jq_fetch.py の --out）")
    ap.add_argument("--preopen", default="")
    ap.add_argument("--capital", type=float, default=15_000_000)
    ap.add_argument("--max-new-orders", type=int, default=6)
    ap.add_argument("--strategy", default="ALL")
    ap.add_argument("--llm", choices=["off", "on"], default="off")
    ap.add_argument("--tiebreak", default="universe",
                    help="スクリーナーの同点の並び: universe（既定。逐次に評価した順）/ asis（フォークのまま）/ shuffle:<seed>")
    ap.add_argument("--foreign-asof", help="東証以外（米国株・指数・VIX・為替・FRED）の as-of の日。既定は --asof")
    ap.add_argument("--council-cache", default="",
                    help="jp_council_probe.py の council.json。合議を呼ばず、その判定を返す（無い銘柄は HOLD にして記録する）")
    ap.add_argument("--snapshot", default="", help="yfinance の値の控えの置き場（同じ日を何度も回すときに使う）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if (out / "ledger.db").exists():
        print(f"{out} にはもう台帳がある。別の --out を使う", file=sys.stderr)
        return 2
    (out / "appdata").mkdir(parents=True, exist_ok=True)
    os.environ["APPDATA"] = str(out / "appdata")
    os.environ["LUMINA_DB_PATH"] = str(out / "ledger.db")
    os.environ["LUMINA_BROKER"] = "paper"
    if args.llm == "off":
        for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            os.environ.pop(k, None)

    session = dt.datetime.fromisoformat(args.session).replace(tzinfo=JST)
    cutoff = (dt.datetime.fromisoformat(args.news_cutoff).replace(tzinfo=JST) if args.news_cutoff
              else session.replace(hour=8, minute=0))
    asof = dt.date.fromisoformat(args.asof)

    from core.ai_hedge_fund import asof_replay
    asof_replay.install(asof=asof, jq_root=Path(args.jq), news_cutoff=cutoff, session_now=session,
                        snapshot_dir=Path(args.snapshot) if args.snapshot else None,
                        foreign_asof=dt.date.fromisoformat(args.foreign_asof) if args.foreign_asof else None)
    asof_replay.set_screen_tiebreak(args.tiebreak)
    council_calls = []
    if args.council_cache:
        install_council_cache(Path(args.council_cache), council_calls)

    from core.ai_hedge_fund import llm_client
    from core.ai_hedge_fund.auto_trader import AutoTrader

    broker = build_broker(os.environ["LUMINA_DB_PATH"], args.preopen)
    broker.reset_portfolio(initial_jpy=args.capital, initial_usd=0.0)

    started = time.time()
    res = AutoTrader.run_autonomous_cycle(
        broker_engine=broker,
        strategy=args.strategy,
        max_new_orders=args.max_new_orders,
        enforce_market_hours=True,
    )
    took = time.time() - started
    u = llm_client.usage_snapshot()
    llm = {"provider": llm_client.provider() or "なし（縮退モード）", "model": llm_client.claude_model(), **u}

    orders = [o for o in broker.placed if o["success"] and o["side"] == "BUY"]
    strategy = {
        "asof": args.asof,
        "session": session.isoformat(),
        "news_cutoff": cutoff.isoformat(),
        "capital_jpy": args.capital,
        "tiebreak": args.tiebreak,
        "foreign_asof": args.foreign_asof or args.asof,
        "council_cache": args.council_cache or None,
        "council_calls": council_calls,
        "snapshot": args.snapshot or None,
        "llm": llm,
        "seconds": round(took, 1),
        "orders": orders,
        "rejected": [o for o in broker.placed if not o["success"]],
        "cash_after_jpy": broker.get_account_balance().get("JPY"),
    }
    (out / "strategy.json").write_text(json.dumps(strategy, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "cycle.log").write_text("\n".join(res.get("logs", [])) + "\n", encoding="utf-8")
    for line in res.get("logs", []):
        print(line)
    print(f"--- LLM: {llm['provider']} / 呼び出し {u['calls']} 回（失敗 {u['errors']}）/ 入力 {u['input_tokens']:,} + 出力 {u['output_tokens']:,}")
    print(f"--- {took:.0f} 秒 / 寄りの買い {len(orders)} 件 → {out / 'strategy.json'}")
    for o in orders:
        print(f"    {o['symbol']:7s} {o['quantity']:>6,} 株  参考 {o['ref_price']:>9,.1f}  目標 {o['target_price']}  逆指値 {o['stop_loss_price']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
