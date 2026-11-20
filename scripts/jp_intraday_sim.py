# -*- coding: utf-8 -*-
"""
寄前の戦略（scripts/jp_preopen.py の strategy.json）を、BRiSK の約定で再生する（1 日でも、複数日でも）。

--trades に日ごとの約定（scripts/brisk_extract_win.py trades）を日付の順に並べる。最初の日の寄りで買い、
最後の日の大引けまで持つ。手仕舞いは 2 通り:

--exit bracket（既定）: 寄付より後の約定ごとに、逆指値以下の最初の約定の値段で全量、目標値以上の最初の約定の値段で全量
--exit hold:            途中では何もしない
--exit patrol:          フォークの見回り（auto_trader.patrol_positions）の日本株の規則を、--check-minutes ごと（既定 15 分、
                        フォークの常駐の既定の間隔）に、その時点の直近の約定値で判定し、その値で売る:
                        1. 目標株価以上 → 全株
                        2. 含み益 +10% 以上で未利確 → 100 株単位で半分を売り、逆指値を買値へ（200 株未満は売らずに逆指値だけ）
                        3. 含み益か高値の含み益が +10% 以上で、高値から -4% 以下 → 全株（トレイリング）
                        4. 半分利確のあとで買値以下 → 全株（建値撤退）
                        5. 逆指値以下か -15% 以下 → 全株（損切り）
                        米国株だけの規則（Call Wall）と、35 日の時間切れは入れない

どの手仕舞いでも、残りは最後の日の大引け（15:30:00 以降の約定 = 引けの板寄せ。無ければ最後の約定）の値段で全量売る。
買いはその銘柄の最初の約定（寄付）の値段で全量。現金が足りなければ 100 株単位で減らす。手数料は --fee-bps（既定 0）。

    .venv/bin/python scripts/jp_intraday_sim.py --strategy …/strategy.json --trades …/20261120/trades.json --out …/sim
    .venv/bin/python scripts/jp_intraday_sim.py --strategy …/strategy.json --exit patrol \\
        --trades …/20261116/trades.json …/20261117/trades.json … --out …/sim-patrol
    # 参考: 指定の銘柄を均等に寄りで買い、最後の大引けで売る
    .venv/bin/python scripts/jp_intraday_sim.py --equal-weight 7203,8411 --capital 15000000 --trades … --out …
"""
import argparse
import bisect
import csv
import datetime as dt
import json
from pathlib import Path

JST = dt.timezone(dt.timedelta(hours=9))


def ms_at(day: dt.date, hh: int, mm: int, ss: int = 0) -> int:
    return int(dt.datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=JST).timestamp() * 1000)


def hhmmss(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, JST).strftime("%H:%M:%S")


def stamp(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, JST).strftime("%m-%d %H:%M:%S")


def check_times(days, minutes: int):
    """見回りの時刻（各日の 09:00+minutes 〜 15:15。昼休み 11:30〜12:30 は除く。11:30 は含む）。"""
    out = []
    for d in days:
        t = ms_at(d, 9, 0) + minutes * 60_000
        end = ms_at(d, 15, 15)
        while t <= end:
            hm = dt.datetime.fromtimestamp(t / 1000, JST).time()
            if not (dt.time(11, 30) < hm < dt.time(12, 30)):
                out.append(t)
            t += minutes * 60_000
    return out


def _last_price(times, rows, t):
    k = bisect.bisect_right(times, t) - 1
    return rows[k][1] if k >= 0 else None


def simulate(orders, trades, capital, day, fee_bps: float = 0.0, exit_mode: str = "bracket",
             days=None, check_minutes: int = 15):
    """注文を再生する。trades はコード → [[時刻 ms, 値段, 株数, 側], …]（複数日なら続けて並べる）。

    day は最後の日（大引けで手仕舞う日）。days は全日の一覧（見回りの時刻に使う。既定は [day]）。
    戻り値: (最終の現金, 建玉の一覧（売りの明細 fills つき）, 出来事の一覧)
    """
    days = days or [day]
    close_ms = ms_at(day, 15, 30)
    fee = fee_bps / 10_000.0
    cash = float(capital)
    events, positions = [], []

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
        if qty <= 0:
            events.append({"time": stamp(open_ms), "code": code, "event": "数量 0（予算が 1 単元に届かない）", "price": price})
            continue
        while qty > 0 and qty * price * (1 + fee) > cash:
            qty -= 100
        if qty <= 0:
            events.append({"time": stamp(open_ms), "code": code, "event": "現金不足で買えない", "price": price})
            continue
        cost = qty * price
        cash -= cost * (1 + fee)
        target = o.get("target_price")
        stop = o.get("stop_loss_price")
        if exit_mode == "patrol":  # フォークの既定（目標なし → +25%、逆指値なし → -13%）
            target = float(target or price * 1.25)
            stop = float(stop or price * 0.87)
        pos = {"code": code, "qty0": qty, "entry_ms": open_ms, "entry": price, "target": target, "stop": stop,
               "fills": []}
        events.append({"time": stamp(open_ms), "code": code, "event": "寄付で買い", "qty": qty, "price": price})
        times = [r[0] for r in tr]
        if exit_mode == "bracket":
            j = bisect.bisect_right(times, open_ms)
            for ts, p, _sz, _side in tr[j:]:
                if ts >= close_ms:
                    break
                if stop and p <= stop:
                    pos["fills"].append((ts, qty, p, "逆指値"))
                    break
                if target and p >= target:
                    pos["fills"].append((ts, qty, p, "目標値"))
                    break
        elif exit_mode == "patrol":
            left, peak, half = qty, price, 0
            for t in check_times(days, check_minutes):
                if t <= open_ms or left <= 0:
                    continue
                cur = _last_price(times, tr, t)
                if cur is None:
                    continue
                pnl = (cur - price) / price * 100.0
                peak = max(peak, cur)
                dd = (cur - peak) / peak * 100.0
                peak_pnl = (peak - price) / price * 100.0
                if cur >= pos["target"]:
                    pos["fills"].append((t, left, cur, "目標株価")); left = 0; break
                if pnl >= 10.0 and half == 0:
                    if left >= 200:
                        sq = (left // 2 // 100) * 100
                        pos["fills"].append((t, sq, cur, "+10% 半分利確")); left -= sq
                    half, pos["stop"] = 1, price
                if (pnl >= 10.0 or peak_pnl >= 10.0) and dd <= -4.0:
                    pos["fills"].append((t, left, cur, "トレイリング")); left = 0; break
                if half == 1 and cur <= price:
                    pos["fills"].append((t, left, cur, "建値撤退")); left = 0; break
                if cur <= pos["stop"] or pnl <= -15.0:
                    pos["fills"].append((t, left, cur, "損切り")); left = 0; break
        sold = sum(f[1] for f in pos["fills"])
        if sold < qty:
            after = [r for r in tr if r[0] >= close_ms]
            last = after[-1] if after else tr[-1]
            pos["fills"].append((last[0], qty - sold, last[1], "大引け" if after else "最後の約定"))
        positions.append(pos)

    all_fills = sorted(((f, p) for p in positions for f in p["fills"]), key=lambda x: x[0][0])
    for (ts, q, p, why), pos in all_fills:
        proceeds = q * p
        cash += proceeds * (1 - fee)
        pnl = round(proceeds * (1 - fee) - q * pos["entry"] * (1 + fee))
        events.append({"time": stamp(ts), "code": pos["code"], "event": f"売り（{why}）", "qty": q, "price": p, "pnl": pnl})
    for pos in positions:
        pos["qty"] = pos["qty0"]
        pos["pnl"] = round(sum(q * p * (1 - fee) for _, q, p, _ in pos["fills"]) - pos["qty0"] * pos["entry"] * (1 + fee))
        pos["exit_ms"] = pos["fills"][-1][0]
        pos["exit"] = pos["fills"][-1][2]
        pos["reason"] = pos["fills"][-1][3]
    events.sort(key=lambda e: e["time"] or "")
    return cash, positions, events


def timeline(positions, trades, capital, day, fee_bps: float = 0.0, days=None):
    """各日の 1 分ごと（09:00〜15:30）の現金・投下中の評価額・合計。"""
    days = days or [day]
    fee = fee_bps / 10_000.0
    idx = {c: [r[0] for r in t] for c, t in trades.items()}
    rows = []
    for d in days:
        t = ms_at(d, 9, 0)
        end = ms_at(d, 15, 30)
        while True:
            at = end + 59_999 if t >= end else t  # 15:30 の行は引けの約定を含める
            cash = float(capital)
            invested = 0.0
            held = 0
            for p in positions:
                if p["entry_ms"] > at:
                    continue
                cash -= p["qty0"] * p["entry"] * (1 + fee)
                left = p["qty0"]
                for ts, q, px, _ in p["fills"]:
                    if ts <= at:
                        cash += q * px * (1 - fee)
                        left -= q
                if left > 0:
                    k = bisect.bisect_right(idx[p["code"]], at) - 1
                    invested += left * trades[p["code"]][k][1]
                    held += 1
            label = dt.datetime.fromtimestamp(min(t, end) / 1000, JST).strftime("%m-%d %H:%M" if len(days) > 1 else "%H:%M")
            rows.append({"time": label, "cash": round(cash), "invested_mv": round(invested),
                         "equity": round(cash + invested), "positions": held, "pnl": round(cash + invested - capital)})
            if t >= end:
                break
            t += 60_000
    return rows


def split_factors(jq_root, codes, days):
    """J-Quants の日足の AdjFactor から、コード → {日: 最初の日の株数の水準へ戻す係数}。

    例: 11-19 に 1:5 の分割（AdjFactor 0.2）なら、11-19 以降の値段を 0.2 で割ると最初の日の水準になる。
    """
    out = {}
    for code in codes:
        path = Path(jq_root) / "daily" / f"{code}0.json"
        if not path.exists():
            continue
        f_by_day = {}
        for r in json.loads(path.read_text(encoding="utf-8")):
            d = dt.date.fromisoformat(r["Date"])
            f = float(r.get("AdjFactor") or 1.0)
            if days[0] < d <= days[-1] and abs(f - 1.0) > 1e-9:
                f_by_day[d] = f
        if f_by_day:
            cum, table = 1.0, {}
            for d in days:
                cum *= f_by_day.get(d, 1.0)
                table[d] = cum
            out[code] = table
    return out


def load_trades(paths, jq_root=None):
    """日ごとの約定の JSON を読み、コードごとに続けて並べる。戻り値: (日付の一覧, コード → 約定, 分割の係数)

    jq_root を渡すと、期間の途中の分割・併合を J-Quants の AdjFactor で戻し、値段を最初の日の株数の水準にそろえる
    （BRiSK の約定はその日の水準の値段）。
    """
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    days = [dt.datetime.strptime(doc["trading_date"], "%Y%m%d").date() for doc in docs]
    factors = split_factors(jq_root, {c for doc in docs for c in doc["codes"]}, days) if jq_root else {}
    merged = {}
    for d, doc in zip(days, docs):
        for code, rows in doc["codes"].items():
            f = factors.get(code, {}).get(d, 1.0)
            if f != 1.0:
                rows = [[r[0], r[1] / f, r[2] * f, r[3]] for r in rows]
            merged.setdefault(code, []).extend(rows)
    for rows in merged.values():
        rows.sort(key=lambda r: r[0])
    order = sorted(range(len(days)), key=lambda i: days[i])
    if order != list(range(len(days))):
        raise SystemExit("--trades は日付の順に並べる")
    return days, merged, factors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy")
    ap.add_argument("--equal-weight", help="参考: 銘柄コードをカンマで。均等に寄りで買い、最後の大引けで売る")
    ap.add_argument("--capital", type=float, help="既定は strategy.json の capital_jpy")
    ap.add_argument("--trades", nargs="+", required=True, help="日ごとの約定の JSON（日付の順）")
    ap.add_argument("--exit", choices=["bracket", "hold", "patrol"], default="bracket")
    ap.add_argument("--check-minutes", type=int, default=15)
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--jq", default="", help="J-Quants の控えの置き場。期間の途中の分割・併合を戻すのに使う（複数日なら付ける）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    days, trades, factors = load_trades(args.trades, args.jq or None)
    for code, table in factors.items():
        print(f"分割・併合の補正: {code} " + " ".join(f"{d:%m-%d}×{1 / f:g}" for d, f in table.items() if f != 1.0)
              + "（値段を最初の日の株数の水準にそろえた）")
    day = days[-1]

    if args.strategy:
        st = json.loads(Path(args.strategy).read_text(encoding="utf-8"))
        orders = st["orders"]
        capital = args.capital or st["capital_jpy"]
        label = f"strategy {args.strategy}"
    elif args.equal_weight:
        capital = args.capital or 15_000_000
        codes = [c.strip() for c in args.equal_weight.split(",") if c.strip()]
        budget = capital / len(codes)
        orders = [{"symbol": f"{c}.T", "quantity": int(budget // trades[c][0][1] // 100) * 100,
                   "target_price": None, "stop_loss_price": None} for c in codes]
        label = f"均等 {len(codes)} 銘柄"
    else:
        ap.error("--strategy か --equal-weight")

    cash, positions, events = simulate(orders, trades, capital, day, args.fee_bps, args.exit, days, args.check_minutes)
    rows = timeline(positions, trades, capital, day, args.fee_bps, days)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "timeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    closes = [r for r in rows if r["time"].endswith("15:30")]
    peak = max(rows, key=lambda r: r["equity"])
    low = min(rows, key=lambda r: r["equity"])
    summary = {
        "source": label, "days": [str(d) for d in days], "exit": args.exit, "capital_jpy": capital,
        "split_adjusted": {c: {str(d): f for d, f in t.items()} for c, t in factors.items()},
        "fee_bps": args.fee_bps, "final_cash_jpy": round(cash), "pnl_jpy": round(cash - capital),
        "return_pct": round((cash / capital - 1) * 100, 3),
        "max_invested_mv_jpy": max(r["invested_mv"] for r in rows),
        "peak": peak, "low": low, "daily_close": closes,
        "positions": [{k: v for k, v in p.items()} for p in positions], "events": events,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"== {label} / {days[0]}〜{day} / 手仕舞い {args.exit} / 投下資金 ¥{capital:,.0f}")
    for e in events:
        extra = "".join(f" {k}={e[k]:,}" if isinstance(e.get(k), (int, float)) else "" for k in ("qty", "price", "pnl"))
        print(f"  {e['time'] or '--'} {e['code']} {e['event']}{extra}")
    marks = closes if len(days) > 1 else [r for r in rows if r["time"] in
                                          ("09:00", "09:30", "10:00", "11:00", "11:30", "12:30", "13:30", "14:30", "15:00", "15:25", "15:30")]
    for r in marks:
        print(f"  {r['time']}  現金 {r['cash']:>13,}  投下中 {r['invested_mv']:>13,}  合計 {r['equity']:>13,}  損益 {r['pnl']:>+11,}")
    print(f"-- 最終 ¥{cash:,.0f}  損益 {cash - capital:+,.0f} 円（{(cash / capital - 1) * 100:+.3f}%）"
          f"  最高 {peak['time']} {peak['pnl']:+,} / 最低 {low['time']} {low['pnl']:+,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
