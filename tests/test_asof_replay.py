# -*- coding: utf-8 -*-
"""as-of の再生（asof_replay）、J-Quants のデータ源（jquants_source）、場中のシミュレーター（scripts/jp_intraday_sim.py）の試験。

ネットワークは使わない（J-Quants の控えを tmp に作り、yfinance は偽物にする）。
"""
import datetime as dt
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from core.ai_hedge_fund import asof_replay as ar
from core.ai_hedge_fund import jquants_source as jqs
from core.ai_hedge_fund.piotroski import compute_f_score

JST = dt.timezone(dt.timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
ASOF = dt.date(2026, 11, 19)


def _bar(date, c, factor=1.0):
    return {"Date": date, "Code": "72030", "O": c, "H": c + 10, "L": c - 10, "C": c, "Vo": 1000.0,
            "AdjFactor": factor, "AdjO": c, "AdjH": c + 10, "AdjL": c - 10, "AdjC": c, "AdjVo": 1000.0}


def _summary(disc, fye, sales, np_, ta, cfo, sh, doc="FYFinancialStatements_Consolidated_JP", per="FY", **kw):
    return {"DiscDate": disc, "DiscTime": "15:00:00", "DiscNo": disc.replace("-", "") + "0", "DocType": doc,
            "CurPerType": per, "CurPerEn": fye, "CurFYEn": fye, "Sales": str(sales), "OP": str(sales // 10),
            "NP": str(np_), "TA": str(ta), "CFO": str(cfo), "CFI": "-50", "CFF": "-10", "ShOutFY": str(sh),
            "TrShFY": "0", "ShEq": "500", "FDivAnn": "", "DivAnn": "", **kw}


def _details(disc, fye, ca, cl, ltd, gp=None, cos=None):
    fs = {"Current fiscal year end date, DEI": fye, "Current assets": str(ca), "Current liabilities": str(cl),
          "Non-current liabilities": str(ltd * 2)}
    if ltd is not None:
        fs["Long-term borrowings"] = str(ltd)
    if gp is not None:
        fs["Gross profit (loss)"] = str(gp)
    if cos is not None:
        fs["Cost of sales"] = str(cos)
    return {"DiscDate": disc, "DiscTime": "15:00:00", "DiscNo": disc.replace("-", "") + "1",
            "DocType": "FYFinancialStatements_Consolidated_JP", "FS": fs}


def _write(root: Path, kind: str, rows):
    (root / kind).mkdir(parents=True, exist_ok=True)
    (root / kind / "72030.json").write_text(json.dumps(rows), encoding="utf-8")


@pytest.fixture
def jq_root(tmp_path):
    days = ["2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    _write(tmp_path, "daily", [_bar(d, 1000.0 + i * 10) for i, d in enumerate(days)])
    _write(tmp_path, "master", [{"Date": "2026-11-19", "Code": "72030", "CoName": "テスト自動車", "CoNameEn": "Test Motor",
                                 "S17Nm": "自動車・輸送機", "S33Nm": "輸送用機器", "MktNm": "プライム", "ScaleCat": "Core30"}])
    _write(tmp_path, "valuation", [
        {"Date": "2026-11-19", "Code": "72030", "EPS": 104.0, "FwdEPS": 110.0, "BPS": 1000.0, "ROE": 0.125,
         "PER": 10.0, "FwdPER": 9.45, "PBR": 1.04, "MktCap": 2000.0},
        {"Date": "2026-11-20", "Code": "72030", "EPS": 1.0, "ROE": 0.99, "PER": 99.0, "MktCap": 1.0},  # asof より後
    ])
    _write(tmp_path, "fins_summary", [
        _summary("2025-05-10", "2025-03-31", 1000, 50, 2000, 80, 100),
        _summary("2026-05-10", "2026-03-31", 1200, 90, 2100, 120, 100, FDivAnn="20"),
        _summary("2026-11-05", "2026-09-30", 650, 50, 2200, 60, 100, doc="2QFinancialStatements_Consolidated_JP", per="2Q",
                 FDivAnn="30"),
        _summary("2026-11-20", "2027-03-31", 9999, 9999, 9999, 9999, 1, doc="FYFinancialStatements_Consolidated_JP"),  # asof より後
    ])
    _write(tmp_path, "fins_details", [
        _details("2025-05-10", "2025-03-31", ca=500, cl=400, ltd=300, gp=400),
        _details("2026-05-10", "2026-03-31", ca=600, cl=400, ltd=250, gp=None, cos=700),  # 売上総利益は 1200-700
    ])
    _write(tmp_path, "earnings_date", [
        {"PubDate": "2026-10-01", "SchDate": "2026-11-05", "FQName": "2Q", "FYE": "0331", "Code": "72030"},
        {"PubDate": "2026-11-10", "SchDate": "2026-11-21", "FQName": "3Q", "FYE": "0331", "Code": "72030"},
        {"PubDate": "2026-11-20", "SchDate": "2026-11-22", "FQName": "FY", "FYE": "0331", "Code": "72030"},  # asof より後に公表
    ])
    _write(tmp_path, "dividend", [
        {"PubDate": "2026-05-10", "PubTime": "15:00", "RefNo": "A", "CARefNo": "A", "StatCode": "1", "FRCode": "1",
         "DivRate": 25.0, "ExDate": "2026-03-30"},
        {"PubDate": "2026-05-10", "PubTime": "15:00", "RefNo": "B", "CARefNo": "B", "StatCode": "1", "FRCode": "2",
         "DivRate": 30.0, "ExDate": "2026-09-29"},  # 予想は配当の実績に入れない
        {"PubDate": "2026-11-21", "PubTime": "15:00", "RefNo": "C", "CARefNo": "C", "StatCode": "1", "FRCode": "1",
         "DivRate": 30.0, "ExDate": "2026-09-29"},  # asof より後に通知
    ])
    return tmp_path


class _FakeReal:
    def __init__(self, symbol, *a, **k):
        self.symbol = symbol

    @property
    def info(self):
        return {"currentPrice": 999.0, "trailingEps": 10.0, "bookValue": 100.0, "marketCap": 1_000_000.0,
                "dividendYield": 2.0, "longName": "Fake US"}

    @property
    def news(self):
        return [
            {"content": {"pubDate": "2026-11-19T22:00:00Z", "title": "前の晩"}},
            {"content": {"pubDate": "2026-11-20T01:00:00Z", "title": "当日の場中"}},
            {"content": {"title": "日時なし"}},
        ]

    def history(self, period="2y", **kw):
        idx = pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York") for d in ("2026-11-18", "2026-11-19", "2026-11-20")])
        return pd.DataFrame({"Open": [10.0, 11, 12], "High": [10.0, 11, 12], "Low": [10.0, 11, 12],
                             "Close": [10.0, 11, 12], "Volume": [1.0, 1, 1]}, index=idx)


@pytest.fixture
def state(jq_root):
    saved = dict(ar._STATE)
    ar._STATE.clear()
    ar._STATE.update({
        "installed": True, "asof": ASOF, "store": jqs.JQuantsStore(jq_root, fetch=False),
        "news_cutoff": dt.datetime(2026, 11, 20, 8, 0, tzinfo=JST), "foreign_cache": {}, "real_ticker": _FakeReal,
        "foreign_asof": ASOF, "alpaca": None, "used": {},
    })
    yield jq_root
    ar._STATE.clear()
    ar._STATE.update(saved)


def test_jq_code():
    assert ar.jq_code("7203.T") == "72030"
    assert ar.jq_code("285A.T") == "285A0"
    assert ar.jq_code("AAPL") is None


def test_history_stops_at_asof(state):
    t = ar.AsOfTicker("7203.T")
    assert isinstance(t, jqs.JQuantsTicker)
    h = t.history(period="1y")
    assert h.index[-1].date() == ASOF
    assert float(h["Close"].iloc[-1]) == 1040.0
    assert len(t.history(period="2d")) == 2
    assert t.history(period="5d", interval="5m").empty  # 東証の分足は返さない


def test_jp_info_comes_from_jquants(state):
    inf = ar.AsOfTicker("7203.T").info
    assert inf["currentPrice"] == 1040.0 and inf["previousClose"] == 1030.0
    assert inf["longName"] == "テスト自動車" and inf["industry"] == "輸送用機器"
    # バリュエーションは asof の日の行（翌日の行は使わない）
    assert inf["trailingPE"] == 10.0 and inf["returnOnEquity"] == 0.125 and inf["marketCap"] == 2000.0 * 1e6
    # 利益率は最後の通期、配当予想は最後の決算短信（2Q）、利回りは比率
    assert inf["profitMargins"] == pytest.approx(90 / 1200)
    assert inf["dividendYield"] == pytest.approx(30 / 1040.0)
    assert "revenueGrowth" not in inf  # 最後の決算短信（2Q）の前年同期の行が無いので出さない
    assert inf["_source"] == "J-Quants"


def test_jp_statements_and_fscore(state):
    t = ar.AsOfTicker("7203.T")
    fin, bs, cf = t.financials, t.balance_sheet, t.cashflow
    assert list(fin.columns) == [pd.Timestamp("2026-03-31"), pd.Timestamp("2025-03-31")]  # asof 後の通期は入らない
    assert fin.loc["Gross Profit", pd.Timestamp("2026-03-31")] == 500.0  # 1200 - 700
    assert bs.loc["Long Term Debt", pd.Timestamp("2026-03-31")] == 250.0
    assert cf.loc["Operating Cash Flow", pd.Timestamp("2025-03-31")] == 80.0
    res = compute_f_score(fin, bs, cf)
    assert res["missing"] == [] and res["score"] is not None


def test_long_term_debt_falls_back_to_non_current_liabilities(jq_root):
    rows = json.loads((jq_root / "fins_details" / "72030.json").read_text())
    del rows[0]["FS"]["Long-term borrowings"]  # 前の期だけ借入金の科目が無い
    (jq_root / "fins_details" / "72030.json").write_text(json.dumps(rows))
    _, bs, _ = jqs.annual_statements(jqs.JQuantsStore(jq_root, fetch=False), "72030", ASOF)
    assert list(bs.loc["Long Term Debt"]) == [500.0, 600.0]  # 全期を固定負債の合計で揃える


def test_jp_calendar_and_dividends(state):
    t = ar.AsOfTicker("7203.T")
    assert t.calendar == {"Earnings Date": [dt.date(2026, 11, 21)]}  # 11-20 に公表された予定は見えない
    d = t.dividends
    assert list(d.values) == [25.0]
    assert t.splits.empty


def test_news_and_foreign_cut(state):
    t = ar.AsOfTicker("7203.T")
    assert [n["content"]["title"] for n in t.news] == ["前の晩"]  # ニュースは yfinance に回し、締め切りで切る
    us = ar.AsOfTicker("AAPL")
    assert us.history(period="5d").index[-1].date() == ASOF
    assert us.info["currentPrice"] == 11.0 and us.info["longName"] == "Fake US"
    assert ar.news_item_before_cutoff({"pub_date": "Thu, 19 Nov 2026 22:59:00 GMT"})  # 07:59 JST
    assert not ar.news_item_before_cutoff({"pub_date": "Thu, 19 Nov 2026 23:00:00 GMT"})  # 08:00 JST ちょうど
    assert not ar.news_item_before_cutoff({"pub_date": "Fri, 20 Nov 2026 00:30:00 GMT"})  # 09:30 JST
    assert not ar.news_item_before_cutoff({"pub_date": ""})


def test_download_shape(state):
    df = ar.asof_download(["7203.T", "AAPL"], period="1mo")
    assert list(df.columns.names) == ["Price", "Ticker"]
    assert df["Close"]["7203.T"].dropna().index[-1] == pd.Timestamp("2026-11-19")
    assert df["Close"]["AAPL"].dropna().index[-1] == pd.Timestamp("2026-11-19")


def test_live_router(jq_root, tmp_path, monkeypatch):
    import yfinance as yf
    from core.ai_hedge_fund import data_sources

    monkeypatch.setattr(yf, "Ticker", _FakeReal)
    monkeypatch.setattr(yf, "download", yf.download)  # install() が差し替えるので、テストの後に戻す
    # 手元の控え（.trial/alpaca）や鍵に左右されないよう、Alpaca は空の置き場にし、振り分けの状態も毎回作り直す
    monkeypatch.setenv("LUMINA_DATA_SOURCES", "auto")
    monkeypatch.setenv("LUMINA_ALPACA_DIR", str(tmp_path / "no-alpaca"))
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(data_sources, "_STATE", {})
    jqs.install_yfinance_router(jq_root)
    assert isinstance(yf.Ticker("7203.T"), jqs.JQuantsTicker)
    assert isinstance(yf.Ticker("AAPL"), _FakeReal)
    assert yf.Ticker("7203.T").history(period="max").index[-1].date() == dt.date(2026, 11, 20)  # いまの値（切らない）


def _sim():
    spec = importlib.util.spec_from_file_location("jp_intraday_sim", ROOT / "scripts" / "jp_intraday_sim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sim_bracket_and_close():
    sim = _sim()
    day = dt.date(2026, 11, 20)
    t = lambda h, m, s=0: sim.ms_at(day, h, m, s)
    trades = {
        "1111": [[t(9, 0), 1000.0, 5000, "NO_AGGRESSOR"], [t(9, 0), 1000.0, 100, "NO_AGGRESSOR"],
                 [t(9, 5), 960.0, 100, "SELL"], [t(9, 6), 949.0, 100, "SELL"], [t(15, 30), 990.0, 9000, "NO_AGGRESSOR"]],
        "2222": [[t(9, 2), 500.0, 5000, "NO_AGGRESSOR"], [t(10, 0), 520.0, 100, "BUY"],
                 [t(15, 30), 510.0, 9000, "NO_AGGRESSOR"]],
        "3333": [[t(9, 1), 200.0, 5000, "NO_AGGRESSOR"], [t(9, 30), 231.0, 100, "BUY"],
                 [t(15, 30), 240.0, 9000, "NO_AGGRESSOR"]],
    }
    orders = [
        {"symbol": "1111.T", "quantity": 1000, "stop_loss_price": 950.0, "target_price": 1100.0},
        {"symbol": "2222.T", "quantity": 1000, "stop_loss_price": 450.0, "target_price": 600.0},
        {"symbol": "3333.T", "quantity": 1000, "stop_loss_price": 150.0, "target_price": 230.0},
    ]
    cash, pos, _ = sim.simulate(orders, trades, 2_000_000, day)
    by = {p["code"]: p for p in pos}
    assert (by["1111"]["reason"], by["1111"]["exit"]) == ("逆指値", 949.0)
    assert (by["2222"]["reason"], by["2222"]["exit"]) == ("大引け", 510.0)
    assert (by["3333"]["reason"], by["3333"]["exit"]) == ("目標値", 231.0)
    assert cash == 2_000_000 + 1000 * (949 - 1000) + 1000 * (510 - 500) + 1000 * (231 - 200)
    rows = sim.timeline(pos, trades, 2_000_000, day)
    assert rows[0]["time"] == "09:00" and rows[-1]["time"] == "15:30"
    assert rows[-1]["equity"] == cash and rows[-1]["invested_mv"] == 0


def test_sim_scales_down_when_cash_short():
    sim = _sim()
    day = dt.date(2026, 11, 20)
    trades = {"1111": [[sim.ms_at(day, 9, 0), 1000.0, 1, ""], [sim.ms_at(day, 15, 30), 1000.0, 1, ""]]}
    _, pos, _ = sim.simulate([{"symbol": "1111.T", "quantity": 1000}], trades, 450_000, day)
    assert pos[0]["qty"] == 400


def test_snapshot_keeps_first_good_value(state, tmp_path):
    ar._STATE["snapshot_dir"] = tmp_path / "snap"
    calls = []

    def empty():
        calls.append("empty")
        return pd.DataFrame()

    def good():
        calls.append("good")
        return pd.DataFrame({"a": [1]})

    assert ar._snapshot("AAPL", "financials", empty).empty  # 空は控えない
    assert ar._snapshot("AAPL", "financials", good)["a"].iloc[0] == 1
    assert ar._snapshot("AAPL", "financials", empty)["a"].iloc[0] == 1  # 控えを使い、取りに行かない
    assert calls == ["empty", "good"]


def test_sim_patrol_half_profit_then_breakeven_across_days():
    sim = _sim()
    d1, d2 = dt.date(2026, 11, 16), dt.date(2026, 11, 17)
    t = lambda d, h, m: sim.ms_at(d, h, m)
    trades = {"1111": [
        [t(d1, 9, 0), 1000.0, 1, ""], [t(d1, 9, 10), 1050.0, 1, ""],
        [t(d1, 10, 5), 1110.0, 1, ""],   # 10:15 の見回りで +11% → 半分利確、逆指値を買値へ
        [t(d1, 15, 30), 1080.0, 1, ""],
        [t(d2, 9, 0), 995.0, 1, ""],     # 翌日 09:15: 高値 +11% から -10% → トレイリング（フォークは建値撤退より先に判定する）
        [t(d2, 15, 30), 1200.0, 1, ""],
    ]}
    order = [{"symbol": "1111.T", "quantity": 1000, "target_price": 1300.0, "stop_loss_price": 900.0}]
    cash, pos, _ = sim.simulate(order, trades, 2_000_000, d2, exit_mode="patrol", days=[d1, d2])
    fills = [(f[1], f[2], f[3]) for f in pos[0]["fills"]]
    assert fills == [(500, 1110.0, "+10% 半分利確"), (500, 995.0, "トレイリング")]
    assert cash == 2_000_000 + 500 * 110 - 500 * 5
    rows = sim.timeline(pos, trades, 2_000_000, d2, days=[d1, d2])
    closes = [r for r in rows if r["time"].endswith("15:30")]
    assert [r["time"] for r in closes] == ["11-16 15:30", "11-17 15:30"]
    assert closes[0]["equity"] == 2_000_000 + 500 * 110 + 500 * 80  # 半分は 1110 で売り、残りは引け 1080
    assert closes[-1]["equity"] == cash


def test_sim_hold_keeps_until_last_close():
    sim = _sim()
    d1, d2 = dt.date(2026, 11, 16), dt.date(2026, 11, 17)
    trades = {"1111": [[sim.ms_at(d1, 9, 0), 1000.0, 1, ""], [sim.ms_at(d1, 9, 5), 500.0, 1, ""],
                       [sim.ms_at(d2, 15, 30), 1100.0, 1, ""]]}
    order = [{"symbol": "1111.T", "quantity": 100, "stop_loss_price": 900.0}]
    cash, pos, _ = sim.simulate(order, trades, 200_000, d2, exit_mode="hold", days=[d1, d2])
    assert pos[0]["reason"] == "大引け" and cash == 200_000 + 100 * 100


def test_fred_cut_respects_release_lag():
    df = pd.DataFrame({"observation_date": ["2026-11-04", "2026-11-11", "2026-11-18"], "WALCL": [1, 2, 3]})
    assert list(ar.fred_cut(df, "WALCL", dt.date(2026, 11, 12))["WALCL"]) == [1, 2]  # 11-11 の値は 11-12 に公表
    assert list(ar.fred_cut(df, "WALCL", dt.date(2026, 11, 11))["WALCL"]) == [1]
    rrp = pd.DataFrame({"observation_date": ["2026-11-12", "2026-11-13"], "RRPONTSYD": [5, 6]})
    assert list(ar.fred_cut(rrp, "RRPONTSYD", dt.date(2026, 11, 12))["RRPONTSYD"]) == [5]


def test_foreign_asof_options_and_insider(state):
    ar._STATE["foreign_asof"] = dt.date(2026, 11, 18)
    us = ar.AsOfTicker("AAPL")
    assert us.history(period="5d").index[-1].date() == dt.date(2026, 11, 18)
    assert us.options == ()
    with pytest.raises(RuntimeError):
        us.option_chain("2026-12-18")
    us._real.insider_transactions = pd.DataFrame({"Start Date": ["2026-11-10", "2026-11-19"], "Shares": [1, 2]})
    assert list(us.insider_transactions["Shares"]) == [1]
    assert us.insider_purchases.empty


def test_bars_use_the_share_basis_of_the_asof_day(jq_root):
    rows = [_bar("2026-11-13", 100.0), _bar("2026-11-19", 100.0, factor=0.2)]  # 11-19 に 1:5 の分割
    rows[0].update({"AdjO": 20.0, "AdjH": 20.0, "AdjL": 20.0, "AdjC": 20.0})    # 遡って 1/5 に調整された値
    assert float(jqs.bars_frame(rows, dt.date(2026, 11, 13))["Close"].iloc[0]) == 100.0  # 11-13 の水準（分割前）
    assert float(jqs.bars_frame(rows, dt.date(2026, 11, 19))["Close"].iloc[0]) == 20.0   # 分割後に見れば調整済み


def test_sim_split_mid_period(tmp_path):
    sim = _sim()
    (tmp_path / "daily").mkdir()
    (tmp_path / "daily" / "11110.json").write_text(json.dumps([
        {"Date": "2026-11-16", "AdjFactor": 1.0}, {"Date": "2026-11-17", "AdjFactor": 0.2}]))
    paths = []
    for d, px in (("20261116", 1000.0), ("20261117", 210.0)):
        p = tmp_path / f"{d}.json"
        day = dt.datetime.strptime(d, "%Y%m%d").date()
        p.write_text(json.dumps({"trading_date": d, "codes": {"1111": [[sim.ms_at(day, 9, 0), px, 1, ""],
                                                                       [sim.ms_at(day, 15, 30), px, 1, ""]]}}))
        paths.append(p)
    days, trades, factors = sim.load_trades(paths, tmp_path)
    assert trades["1111"][-1][1] == pytest.approx(1050.0)  # 210 円 × 5 = 分割前の水準で 1,050 円
    cash, pos, _ = sim.simulate([{"symbol": "1111.T", "quantity": 100}], trades, 200_000, days[-1],
                                exit_mode="hold", days=days)
    assert cash == pytest.approx(200_000 + 100 * 50)
