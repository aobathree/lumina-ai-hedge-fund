# -*- coding: utf-8 -*-
"""
試運転用: 自律サイクルを 1 回だけ回して、ログを出す（trial ブランチ）。

引数なしでは何もしない（ヘルプを出して終わる）。

    python scripts/trial_cycle.py run                 # 1 サイクル
    python scripts/trial_cycle.py run --ignore-hours  # 立会の外でも、スクリーニングと合議まで通す
    python scripts/trial_cycle.py status              # 台帳の残高と建玉、今日の意図の数

ブローカーは LUMINA_BROKER で選ぶ（paper / webull_intents / webull_jp_cash）。Webull の API は呼ばない。
データは yfinance と FRED から取る。LLM の API キーが無いときは、合議は縮退モード（決定的な規則）で動く。
市場データの取り先は core/ai_hedge_fund/data_sources.py が銘柄ごとに決める（LUMINA_DATA_SOURCES=auto が既定）:
東証は J-Quants（JQUANTS_API_KEY、控え LUMINA_JQ_DIR）、米国の株・ETF の日足は Alpaca（ALPACA_API_KEY_ID /
ALPACA_API_SECRET_KEY、控え LUMINA_ALPACA_DIR）、指数・為替と、鍵も控えも無いときは yfinance。
LUMINA_DATA_SOURCES=yfinance で従来どおり yfinance だけにする。
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _data_sources() -> None:
    from core.ai_hedge_fund import data_sources
    for kind, src in data_sources.install().items():
        print(f"データ: {kind} = {src}")


def _broker():
    from core.ai_hedge_fund.webull_intent_broker import create_broker
    return create_broker()


def cmd_status(_args) -> int:
    broker = _broker()
    print(f"broker: {type(broker).__name__}  db: {broker.db_path}")
    print("balance:", json.dumps(broker.get_account_balance(), ensure_ascii=False))
    for p in broker.get_positions():
        print("position:", p.get("symbol"), p.get("quantity"), "@", p.get("avg_entry_price"), p.get("currency"))
    intent_dir = getattr(broker, "intent_dir", None)
    if intent_dir:
        for path in sorted(intent_dir.glob("intents-*.jsonl"))[-3:]:
            with open(path, encoding="utf-8") as f:
                print(f"intents: {path.name}  {sum(1 for _ in f)} 行")
    return 0


def cmd_run(args) -> int:
    from core.ai_hedge_fund.auto_trader import AutoTrader

    broker = _broker()
    print(f"broker: {type(broker).__name__}  db: {broker.db_path}")
    started = datetime.datetime.now()
    res = AutoTrader.run_autonomous_cycle(
        broker_engine=broker,
        strategy=args.strategy,
        max_new_orders=args.max_new_orders,
        enforce_market_hours=not args.ignore_hours,
    )
    for line in res.get("logs", []):
        print(line)
    took = (datetime.datetime.now() - started).total_seconds()
    from core.ai_hedge_fund import llm_client
    u = llm_client.usage_snapshot()
    print(f"--- LLM: {llm_client.provider() or 'なし（縮退モード）'} / Claude の呼び出し {u['calls']} 回（失敗 {u['errors']}）/ "
          f"入力 {u['input_tokens']:,} + 出力 {u['output_tokens']:,} トークン / モデル {llm_client.claude_model()}")
    print(f"--- {took:.0f} 秒 / 新規 {len(res.get('opened_orders', []))} 件 / 決済 {len(res.get('closed_orders', []))} 件")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="1 サイクル回す")
    run.add_argument("--strategy", default="ALL")
    run.add_argument("--max-new-orders", type=int, default=2)
    run.add_argument("--ignore-hours", action="store_true", help="取引時間の判定を外す（発注はブローカーが別に判定する）")
    run.set_defaults(func=cmd_run)
    st = sub.add_parser("status", help="台帳と意図の数を出す")
    st.set_defaults(func=cmd_status)
    args = ap.parse_args()
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    _data_sources()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
