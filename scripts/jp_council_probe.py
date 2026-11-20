# -*- coding: utf-8 -*-
"""
as-of の値で、指定の銘柄の合議（HedgeFundCouncil.run_council）だけを回し、判定を JSON に書く。

自律サイクルは、関門（ステージ 2・ROE・F-Score・RS）を通った銘柄だけを合議にかける。スクリーナーの同点の並びで
どの銘柄が合議に届くかが変わるので、届きうる銘柄の判定を先にまとめて見るための道具。

    op run --env-file .env.op -- .venv/bin/python scripts/jp_council_probe.py --asof 2026-11-19 \\
        --session 2026-11-20T09:00 --jq .trial/jq --out .trial/replay-20261120/council-claude 8001.T 6758.T
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--jq", required=True, help="J-Quants の控えの置き場（scripts/jq_fetch.py の --out）")
    ap.add_argument("--snapshot", default="", help="yfinance の値の控えの置き場（jp_preopen.py と同じものを使う）")
    ap.add_argument("--out", required=True)
    ap.add_argument("symbols", nargs="+")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    (out / "appdata").mkdir(parents=True, exist_ok=True)
    os.environ["APPDATA"] = str(out / "appdata")
    os.environ["LUMINA_DB_PATH"] = str(out / "ledger.db")

    session = dt.datetime.fromisoformat(args.session).replace(tzinfo=JST)
    from core.ai_hedge_fund import asof_replay
    asof_replay.install(asof=dt.date.fromisoformat(args.asof), jq_root=Path(args.jq),
                        news_cutoff=session.replace(hour=8, minute=0), session_now=session,
                        snapshot_dir=Path(args.snapshot) if args.snapshot else None)

    from core.ai_hedge_fund import llm_client
    from core.ai_hedge_fund.hedge_fund_council import HedgeFundCouncil

    results = {}
    for sym in args.symbols:
        t0 = time.time()
        before = llm_client.usage_snapshot()
        res = HedgeFundCouncil.run_council(sym, force_refresh=True)
        after = llm_client.usage_snapshot()
        v = res.get("verdict", {})
        results[sym] = {
            "action": v.get("action"), "action_label": v.get("action_label"), "confidence_score": v.get("confidence_score"),
            "target_price": v.get("target_price"), "stop_loss_price": v.get("stop_loss_price"),
            "allocation_pct": v.get("allocation_pct"), "thesis_summary": v.get("thesis_summary"),
            "current_price": res.get("current_price"), "seconds": round(time.time() - t0, 1),
            "llm_calls": after["calls"] - before["calls"],
            "input_tokens": after["input_tokens"] - before["input_tokens"],
            "output_tokens": after["output_tokens"] - before["output_tokens"],
        }
        r = results[sym]
        print(f"{sym:7s} {r['action']:<11} {r['confidence_score']!s:>4} 点  目標 {r['target_price']}  逆指値 {r['stop_loss_price']}"
              f"  LLM {r['llm_calls']} 回 / {r['input_tokens']:,}+{r['output_tokens']:,}  {r['seconds']} 秒")
    u = llm_client.usage_snapshot()
    doc = {"asof": args.asof, "llm": {"provider": llm_client.provider() or "なし", "model": llm_client.claude_model(), **u},
           "results": results}
    (out / "council.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"--- LLM: {doc['llm']['provider']} / {u['calls']} 回（失敗 {u['errors']}）/ 入力 {u['input_tokens']:,} + 出力 {u['output_tokens']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
