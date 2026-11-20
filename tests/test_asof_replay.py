# -*- coding: utf-8 -*-
"""as-of の再生（asof_replay）と、場中のシミュレーター（scripts/jp_intraday_sim.py）の試験。ネットワークは使わない。"""
import datetime as dt
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from core.ai_hedge_fund import asof_replay as ar

JST = dt.timezone(dt.timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]


def _bar(date, c, factor=1.0):
    return {"Date": date, "Code": "72030", "O": c, "H": c + 10, "L": c - 10, "C": c, "Vo": 1000.0,
            "AdjFactor": factor, "AdjO": c, "AdjH": c + 10, "AdjL": c - 10, "AdjC": c, "AdjVo": 1000.0}


class _FakeReal:
    def __init__(self, symbol, *a, **k):
        self.symbol = symbol

    @property
    def info(self):
        return {"currentPrice": 999.0, "trailingEps": 10.0, "bookValue": 100.0, "marketCap": 1_000_000.0,
                "dividendYield": 2.0, "longName": "Fake"}

    @property
    def news(self):
        return [
            {"content": {"pubDate": "2026-11-19T22:00:00Z", "title": "前の晩"}},
            {"content": {"pubDate": "2026-11-20T01:00:00Z", "title": "当日の場中"}},
            {"content": {"title": "日時なし"}},
        ]

    @property
    def financials(self):
        return "FIN"


@pytest.fixture
def state(tmp_path, monkeypatch):
    days = ["2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    rows = [_bar(d, 1000.0 + i * 10) for i, d in enumerate(days)]
    (tmp_path / "72030.json").write_text(json.dumps(rows), encoding="utf-8")
    saved = dict(ar._STATE)
    ar._STATE.clear()
    ar._STATE.update({
        "installed": True, "asof": dt.date(2026, 11, 19), "jq_dir": tmp_path,
        "news_cutoff": dt.datetime(2026, 11, 20, 8, 0, tzinfo=JST), "jq_cache": {}, "foreign_cache": {},
        "real_ticker": _FakeReal,
    })
    yield tmp_path
    ar._STATE.clear()
    ar._STATE.update(saved)


def test_jq_code():
    assert ar.jq_code("7203.T") == "72030"
    assert ar.jq_code("285A.T") == "285A0"
    assert ar.jq_code("AAPL") is None


def test_history_stops_at_asof(state):
    t = ar.AsOfTicker("7203.T")
    h = t.history(period="1y")
    assert h.index[-1].date() == dt.date(2026, 11, 19)
    assert float(h["Close"].iloc[-1]) == 1040.0
    assert len(t.history(period="2d")) == 2
    assert t.history(period="5d", interval="5m").empty  # 東証の分足は返さない


def test_info_uses_asof_close(state):
    inf = ar.AsOfTicker("7203.T").info
    assert inf["currentPrice"] == 1040.0
    assert inf["previousClose"] == 1030.0
    assert inf["trailingPE"] == pytest.approx(104.0)
    assert inf["priceToBook"] == pytest.approx(10.4)
    assert inf["marketCap"] == pytest.approx(1_000_000.0 * 1040.0 / 999.0)
    assert inf["dividendYield"] == pytest.approx(2.0 * 999.0 / 1040.0)
    assert inf["longName"] == "Fake"


def test_news_and_passthrough(state):
    t = ar.AsOfTicker("7203.T")
    assert [n["content"]["title"] for n in t.news] == ["前の晩"]
    assert t.financials == "FIN"
    assert ar.news_item_before_cutoff({"pub_date": "Thu, 19 Nov 2026 22:59:00 GMT"})  # 07:59 JST
    assert not ar.news_item_before_cutoff({"pub_date": "Thu, 19 Nov 2026 23:00:00 GMT"})  # 08:00 JST ちょうど
    assert not ar.news_item_before_cutoff({"pub_date": "Fri, 20 Nov 2026 00:30:00 GMT"})  # 09:30 JST
    assert not ar.news_item_before_cutoff({"pub_date": ""})


def test_download_shape(state):
    df = ar.asof_download(["7203.T"], period="1mo")
    assert list(df.columns.names) == ["Price", "Ticker"]
    assert df["Close"]["7203.T"].index[-1] == pd.Timestamp("2026-11-19")


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

    assert ar._snapshot("8001.T", "financials", empty).empty  # 空は控えない
    assert ar._snapshot("8001.T", "financials", good)["a"].iloc[0] == 1
    assert ar._snapshot("8001.T", "financials", empty)["a"].iloc[0] == 1  # 控えを使い、取りに行かない
    assert calls == ["empty", "good"]
    assert ar.AsOfTicker("7203.T").financials == "FIN"  # 控えに無い銘柄は取りに行く
