# -*- coding: utf-8 -*-
"""
寄前の戦略（scripts/jp_preopen.py の strategy.json）を、その日の BRiSK の約定で寄りから大引けまで再生する。

約定の規則:
- 買い: その銘柄の最初の約定（寄付。特別気配で寄りが遅れた銘柄は寄った時刻）の値段で全量。
  現金が足りなければ、100 株単位で減らす
- 売り（ブラケット）: 寄付より後の約定で、値段が逆指値以下になった最初の約定の値段で全量。
  値段が目標値以上になった最初の約定の値段で全量（指値は約定値段で付く）
- 残りは大引け（15:30:00 以降の約定 = 引けの板寄せ。無ければ最後の約定）の値段で全量
- 手数料は --fee-bps（既定 0）。スリッページは見ない

    .venv/bin/python scripts/jp_intraday_sim.py --strategy .trial/replay-20261120/rules/strategy.json \\
        --trades .trial/brisk/20261120/trades.json --out .trial/replay-20261120/rules/sim
    # 参考: 指定の銘柄を均等に寄りで買い、大引けで売る
    .venv/bin/python scripts/jp_intraday_sim.py --equal-weight 7203,8411 --capital 15000000 \\
        --trades .trial/brisk/20261120/trades.json --out .trial/replay-20261120/ref-eq
"""
import argparse
import bisect
import csv
import datetime as dt
import json
import sys
from pathlib import Path

JST = dt.timezone(dt.timedelta(hours=9))


def ms_at(day: dt.date, hh: int, mm: int, ss: int = 0) -> int:
    return int(dt.datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=JST).timestamp() * 1000)


def hhmmss(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, JST).strftime("%H:%M:%S")


def simulate(orders, trades, capital, day: dt.date, fee_bps: float = 0.0):
    close_ms = ms_at(day, 15, 30)
    fee = fee_bps / 10_000.0
    cash = float(capital)
    events, positions = [], []

    def pay(amount):
        return amount * fee

    # 寄付の時刻の順に買う（同じ時刻なら戦略の順）
    plan = []
    for i, o in enumerate(orders):
        code = o["symbol"].replace(".T", "")
        tr = trades.get(code) or []
        if not tr:
            events.append({"time": None, "code": code, "event": "約定なし（買えない）"})
            continue
        plan.append((tr[0][0], i, o, code, tr))
    plan.sort(key=lambda x: (x[0], x[1]))

    for open_ms, _, o, code, tr in plan:
        price = tr[0][1]
        qty = int(o["quantity"])
        while qty > 0 and qty * price * (1 + fee) > cash:
            qty -= 100
        if qty <= 0:
            events.append({"time": hhmmss(open_ms), "code": code, "event": "現金不足で買えない", "price": price})
            continue
        cost = qty * price
        cash -= cost + pay(cost)
        pos = {"code": code, "qty": qty, "entry_ms": open_ms, "entry": price, "target": o.get("target_price"),
               "stop": o.get("stop_loss_price"), "exit_ms": None, "exit": None, "reason": None}
        events.append({"time": hhmmss(open_ms), "code": code, "event": "寄付で買い", "qty": qty, "price": price,
                       "cash": round(cash)})
        # 寄付の約定（同じ時刻の板寄せの約定）より後を見張る
        j = 0
        while j < len(tr) and tr[j][0] == open_ms:
            j += 1
        for ts, p, _sz, _side in tr[j:]:
            if ts >= close_ms:
                break
            if pos["stop"] and p <= pos["stop"]:
                pos.update(exit_ms=ts, exit=p, reason="逆指値")
                break
            if pos["target"] and p >= pos["target"]:
                pos.update(exit_ms=ts, exit=p, reason="目標値")
                break
        if pos["exit_ms"] is None:
            after = [r for r in tr if r[0] >= close_ms]
            last = after[-1] if after else tr[-1]
            pos.update(exit_ms=last[0], exit=last[1], reason="大引け" if after else "最後の約定")
        positions.append(pos)

    for pos in sorted(positions, key=lambda p: p["exit_ms"]):
        proceeds = pos["qty"] * pos["exit"]
        cash += proceeds - pay(proceeds)
        pos["pnl"] = round(proceeds - pos["qty"] * pos["entry"] - pay(proceeds) - pay(pos["qty"] * pos["entry"]))
        events.append({"time": hhmmss(pos["exit_ms"]), "code": pos["code"], "event": f"売り（{pos['reason']}）",
                       "qty": pos["qty"], "price": pos["exit"], "pnl": pos["pnl"]})
    events.sort(key=lambda e: e["time"] or "")
    return cash, positions, events


def timeline(positions, trades, capital, day: dt.date, fee_bps: float = 0.0):
    """1 分ごと（09:00〜15:30）の現金・投下中の評価額・合計。"""
    fee = fee_bps / 10_000.0
    idx = {c: [r[0] for r in t] for c, t in trades.items()}
    rows = []
    t = ms_at(day, 9, 0)
    end = ms_at(day, 15, 30)
    while t <= end + 60_000:
        stamp = min(t, end + 59_999)  # 最後の行は引けの約定を含める
        cash = float(capital)
        invested = 0.0
        held = 0
        for p in positions:
            if p["entry_ms"] > stamp:
                continue
            cost = p["qty"] * p["entry"]
            cash -= cost * (1 + fee)
            if p["exit_ms"] <= stamp:
                proceeds = p["qty"] * p["exit"]
                cash += proceeds * (1 - fee)
                continue
            k = bisect.bisect_right(idx[p["code"]], stamp) - 1
            invested += p["qty"] * trades[p["code"]][k][1]
            held += 1
        label = dt.datetime.fromtimestamp(min(t, end) / 1000, JST).strftime("%H:%M")
        rows.append({"time": label, "cash": round(cash), "invested_mv": round(invested), "equity": round(cash + invested),
                     "positions": held, "pnl": round(cash + invested - capital)})
        if t >= end:
            break
        t += 60_000
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy")
    ap.add_argument("--equal-weight", help="参考: 銘柄コードをカンマで。均等に寄りで買い、大引けで売る（ブラケット無し）")
    ap.add_argument("--capital", type=float, help="既定は strategy.json の capital_jpy")
    ap.add_argument("--trades", required=True)
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tr_doc = json.loads(Path(args.trades).read_text(encoding="utf-8"))
    day = dt.datetime.strptime(tr_doc["trading_date"], "%Y%m%d").date()
    trades = tr_doc["codes"]

    if args.strategy:
        st = json.loads(Path(args.strategy).read_text(encoding="utf-8"))
        orders = st["orders"]
        capital = args.capital or st["capital_jpy"]
        label = f"strategy {args.strategy}"
    elif args.equal_weight:
        capital = args.capital or 15_000_000
        codes = [c.strip() for c in args.equal_weight.split(",") if c.strip()]
        budget = capital / len(codes)
        orders = []
        for c in codes:
            p = trades[c][0][1]
            orders.append({"symbol": f"{c}.T", "quantity": int(budget // p // 100) * 100,
                           "target_price": None, "stop_loss_price": None})
        label = f"均等 {len(codes)} 銘柄"
    else:
        ap.error("--strategy か --equal-weight")

    cash, positions, events = simulate(orders, trades, capital, day, args.fee_bps)
    rows = timeline(positions, trades, capital, day, args.fee_bps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "timeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    peak = max(rows, key=lambda r: r["equity"])
    low = min(rows, key=lambda r: r["equity"])
    summary = {
        "source": label, "trading_date": str(day), "capital_jpy": capital, "fee_bps": args.fee_bps,
        "final_cash_jpy": round(cash), "pnl_jpy": round(cash - capital),
        "return_pct": round((cash / capital - 1) * 100, 3),
        "max_invested_mv_jpy": max(r["invested_mv"] for r in rows),
        "peak": peak, "low": low, "positions": positions, "events": events,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"== {label} / {day} / 投下資金 ¥{capital:,.0f}")
    for e in events:
        extra = "".join(f" {k}={e[k]:,}" if isinstance(e.get(k), (int, float)) else "" for k in ("qty", "price", "pnl"))
        print(f"  {e['time'] or '--:--:--'} {e['code']} {e['event']}{extra}")
    for r in rows:
        if r["time"] in ("09:00", "09:30", "10:00", "11:00", "11:30", "12:30", "13:30", "14:30", "15:00", "15:25", "15:30"):
            print(f"  {r['time']}  現金 {r['cash']:>13,}  投下中 {r['invested_mv']:>13,}  合計 {r['equity']:>13,}  損益 {r['pnl']:>+11,}")
    print(f"-- 最終 ¥{cash:,.0f}  損益 {cash - capital:+,.0f} 円（{(cash / capital - 1) * 100:+.3f}%）"
          f"  最高 {peak['time']} {peak['pnl']:+,} / 最低 {low['time']} {low['pnl']:+,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
