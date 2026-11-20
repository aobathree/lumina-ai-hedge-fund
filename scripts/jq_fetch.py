# -*- coding: utf-8 -*-
"""
J-Quants（V2）の日足を、Lumina のスクリーナーの東証の銘柄について取る（as-of の再生用）。

鍵は環境変数 JQUANTS_API_KEY（1Password の参照なら ``op run`` で渡す）。

    op run --env-file ~/kabs-local/n3/env -- .venv/bin/python scripts/jq_fetch.py \\
        --from 2025-11-01 --to 2026-11-19 --focus-from 2026-11-16 --out .trial/jq

    # 財務・銘柄情報・バリュエーション・決算予定・配当も取る（--to を as-of の日として使う）
    op run --env-file ~/kabs-local/n3/env -- .venv/bin/python scripts/jq_fetch.py \\
        --from 2025-11-01 --to 2026-11-19 --focus-from 2026-11-16 --out .trial/jq --fundamentals

出力（core/ai_hedge_fund/jquants_source.JQuantsStore の置き場。asof_replay と LUMINA_JP_DATA_SOURCE=jquants が読む）:
    <out>/daily/<5 桁のコード>.json          日足（API の行のまま）
    <out>/master|valuation|fins_summary|fins_details|earnings_date|dividend/<コード>.json   --fundamentals のとき
    <out>/focus_<from>_<to>.csv              --focus-from〜--to の日足の一覧（確認用）
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API = "https://api.jquants.com/v2/equities/bars/daily"


def fetch(code: str, frm: str, to: str, key: str) -> list:
    rows, pkey = [], None
    while True:
        q = {"code": code, "from": frm, "to": to}
        if pkey:
            q["pagination_key"] = pkey
        req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(q)}", headers={"x-api-key": key})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 3:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise
        rows.extend(body.get("data", []))
        pkey = body.get("pagination_key")
        if not pkey:
            return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="frm", required=True)
    ap.add_argument("--to", required=True)
    ap.add_argument("--focus-from", required=True, help="一覧の CSV に出す期間の始まり")
    ap.add_argument("--out", default=".trial/jq")
    ap.add_argument("--codes", nargs="*", help="既定はスクリーナーの東証の銘柄すべて")
    ap.add_argument("--fundamentals", action="store_true", help="財務・銘柄情報・バリュエーション・決算予定・配当も取る")
    args = ap.parse_args()

    key = os.environ.get("JQUANTS_API_KEY", "")
    if not key or key.startswith("op://"):
        print("JQUANTS_API_KEY が無い（op run --env-file … -- で呼ぶ）", file=sys.stderr)
        return 2

    from core.ai_hedge_fund.asof_replay import jp_symbols, jq_code
    from core.ai_hedge_fund.market_screener import MarketScreener

    symbols = args.codes or jp_symbols(MarketScreener.UNIVERSE)
    out = Path(args.out)
    (out / "daily").mkdir(parents=True, exist_ok=True)
    focus = []
    for sym in symbols:
        code = jq_code(sym if sym.endswith(".T") else f"{sym}.T")
        rows = fetch(code, args.frm, args.to, key)
        (out / "daily" / f"{code}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        f_rows = [r for r in rows if r["Date"] >= args.focus_from]
        focus.extend(f_rows)
        print(f"{sym:8s} {code}  {len(rows):4d} 日  {rows[0]['Date'] if rows else '-'}〜{rows[-1]['Date'] if rows else '-'}"
              f"  調整係数≠1: {sum(1 for r in rows if r.get('AdjFactor') not in (1, 1.0))}")
        time.sleep(0.2)

    if args.fundamentals:
        import datetime as dt
        from core.ai_hedge_fund.jquants_source import JQuantsStore, fetch_fundamentals
        store = JQuantsStore(out, api_key=key)
        codes = [jq_code(s if s.endswith(".T") else f"{s}.T") for s in symbols]
        fetch_fundamentals(store, codes, dt.date.fromisoformat(args.to))

    cols = ["Date", "Code", "O", "H", "L", "C", "Vo", "Va", "AdjFactor", "AdjC"]
    path = out / f"focus_{args.focus_from}_{args.to}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(focus, key=lambda r: (r["Code"], r["Date"])):
            w.writerow(r)
    print(f"--- {len(symbols)} 銘柄 / 一覧 {path}（{len(focus)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
