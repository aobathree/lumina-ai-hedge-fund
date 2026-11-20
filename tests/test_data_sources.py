# -*- coding: utf-8 -*-
"""データの取り先（data_sources・alpaca_source）と、ウィブル証券の日本株・現物の意図ブローカーの試験。ネットワークは使わない。"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from core.ai_hedge_fund import alpaca_source as als
from core.ai_hedge_fund import asof_replay as ar
from core.ai_hedge_fund import data_sources as ds
from core.ai_hedge_fund import jquants_source as jqs

JST = dt.timezone(dt.timedelta(hours=9))


def _alpaca_rows(days_closes):
    return [{"t": f"{d}T05:00:00Z", "o": c, "h": c, "l": c, "c": c, "v": 100, "n": 1, "vw": c} for d, c in days_closes]


@pytest.fixture
def roots(tmp_path):
    (tmp_path / "jq" / "daily").mkdir(parents=True)
    (tmp_path / "jq" / "daily" / "72030.json").write_text(json.dumps([
        {"Date": "2026-11-19", "AdjFactor": 1.0, "AdjO": 3400.0, "AdjH": 3400.0, "AdjL": 3400.0, "AdjC": 3400.0, "AdjVo": 1.0}]))
    (tmp_path / "al" / "us_daily").mkdir(parents=True)
    (tmp_path / "al" / "us_daily" / "LMT.json").write_text(json.dumps(
        _alpaca_rows([("2026-11-18", 578.19), ("2026-11-19", 571.85), ("2026-11-20", 560.0)])))
    (tmp_path / "al" / "us_daily" / "BRK-B.json").write_text(json.dumps(_alpaca_rows([("2026-11-19", 572.04)])))
    (tmp_path / "al" / "us_daily" / "EADSY.json").write_text("[]")  # Alpaca に無い（OTC）
    return tmp_path


class _FakeYf:
    calls = []

    def __init__(self, symbol, *a, **k):
        self.symbol = symbol
        _FakeYf.calls.append(symbol)

    @property
    def info(self):
        return {"currentPrice": 600.0, "trailingEps": 20.0, "marketCap": 1000.0, "longName": f"yf {self.symbol}"}

    def history(self, period="1mo", **kw):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-11-19", tz="America/New_York")])
        return pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]}, index=idx)


def test_classify():
    assert ds.classify("7203.T") == "jp"
    assert [ds.classify(s) for s in ("AAPL", "BRK-B", "SPY", "F")] == ["us"] * 4
    assert [ds.classify(s) for s in ("^VIX", "USDJPY=X", "HG=F", "SAP.DE")] == ["other"] * 4
    assert als.alpaca_symbol("BRK-B") == "BRK.B"


def test_alpaca_bars_and_info(roots):
    store = als.AlpacaStore(roots / "al", fetch=False)
    t = als.AlpacaTicker("LMT", store, dt.date(2026, 11, 19), real_factory=_FakeYf)
    h = t.history(period="5d")
    assert str(h.index.tz) == "America/New_York" and h.index[-1].date() == dt.date(2026, 11, 19)
    inf = t.info
    assert inf["currentPrice"] == 571.85 and inf["previousClose"] == 578.19  # 価格は Alpaca
    assert inf["longName"] == "yf LMT" and inf["trailingPE"] == pytest.approx(571.85 / 20.0)  # 財務は yfinance
    assert inf["_price_source"] == "Alpaca"


def test_router_falls_back_to_yfinance(roots, monkeypatch):
    monkeypatch.delenv("JQUANTS_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("LUMINA_DATA_SOURCES", "auto")
    jq, al = ds.stores(roots / "jq", roots / "al", fetch=False)
    r = ds.Router(jq, al, _FakeYf)
    assert isinstance(r.ticker("7203.T"), jqs.JQuantsTicker)
    assert isinstance(r.ticker("LMT"), als.AlpacaTicker)
    assert isinstance(r.ticker("BRK-B"), als.AlpacaTicker)
    assert isinstance(r.ticker("EADSY"), _FakeYf)      # Alpaca に行が無い → yfinance
    assert isinstance(r.ticker("^VIX"), _FakeYf)       # 指数は yfinance
    assert isinstance(r.ticker("9999.T"), _FakeYf)     # J-Quants に行が無い → yfinance
    assert r.used == {"7203.T": "J-Quants", "LMT": "Alpaca", "BRK-B": "Alpaca", "EADSY": "yfinance",
                      "^VIX": "yfinance", "9999.T": "yfinance"}
    df = r.download(["LMT", "7203.T"], period="1mo")
    assert df["Close"]["LMT"].dropna().iloc[-1] == 560.0  # いまの値（切らない）


def test_router_policy_yfinance_and_no_keys(roots, tmp_path, monkeypatch):
    monkeypatch.setenv("LUMINA_DATA_SOURCES", "yfinance")
    assert ds.stores(roots / "jq", roots / "al", fetch=False) == (None, None)
    monkeypatch.setenv("LUMINA_DATA_SOURCES", "auto")
    monkeypatch.delenv("JQUANTS_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    assert ds.stores(tmp_path / "none-jq", tmp_path / "none-al", fetch=False) == (None, None)  # 契約も控えも無い → yfinance


def test_asof_routes_us_to_alpaca(roots):
    saved = dict(ar._STATE)
    ar._STATE.clear()
    ar._STATE.update({"installed": True, "asof": dt.date(2026, 11, 19), "foreign_asof": dt.date(2026, 11, 18),
                      "store": jqs.JQuantsStore(roots / "jq", fetch=False), "alpaca": als.AlpacaStore(roots / "al", fetch=False),
                      "news_cutoff": dt.datetime(2026, 11, 20, 8, 0, tzinfo=JST), "foreign_cache": {}, "used": {},
                      "real_ticker": _FakeYf})
    try:
        t = ar.AsOfTicker("LMT")
        assert isinstance(t, als.AlpacaTicker)
        assert t.history(period="5d").index[-1].date() == dt.date(2026, 11, 18)  # 東証以外の as-of で切る
        assert t.info["currentPrice"] == 578.19
        assert isinstance(ar.AsOfTicker("EADSY"), ar.ForeignAsOfTicker)
        assert ar.used_sources()["LMT"] == "Alpaca"
    finally:
        ar._STATE.clear()
        ar._STATE.update(saved)


# ---------------------------------------------------------------- ウィブル証券の日本株・現物


@pytest.fixture
def jp_broker(tmp_path, monkeypatch):
    from core.ai_hedge_fund.market_session import MarketSession
    from core.ai_hedge_fund.webull_intent_broker import WebullJpCashIntentBroker

    monkeypatch.setenv("LUMINA_TEST_MODE", "1")
    state = {"open": True}
    monkeypatch.setattr(MarketSession, "is_tse_open", classmethod(lambda cls, dt_jst=None: state["open"]))
    now = dt.datetime(2026, 11, 20, 9, 0, tzinfo=JST)
    b = WebullJpCashIntentBroker(db_path=str(tmp_path / "l.db"), intent_dir=str(tmp_path / "intents"), mode="paper",
                                 initial_jpy=1_000_000, now_fn=lambda: now.astimezone(dt.timezone.utc))
    return b, state, tmp_path


def test_webull_jp_cash_rules(jp_broker):
    b, state, tmp = jp_broker
    assert b.jp_only is True
    assert not b.place_order("AAPL", "BUY", 1, 200.0)["success"]                 # 米国株は扱わない
    assert "100 株単位" in b.place_order("7203.T", "BUY", 150, 3000.0)["error"]   # 単元未満
    assert "現金が足りない" in b.place_order("7203.T", "BUY", 400, 3000.0)["error"]  # 120 万円 > 100 万円
    assert "空売り" in b.place_order("7203.T", "SELL", 100, 3000.0)["error"]       # 建玉なし
    ok = b.place_order("7203.T", "BUY", 300, 3000.0, target_price=3300.0, stop_loss_price=2800.0)
    assert ok["success"] and ok["intent_written"]
    assert b.get_account_balance()["JPY"] == pytest.approx(100_000)
    assert "超える" in b.place_order("7203.T", "SELL", 400, 3000.0)["error"]
    assert not b.convert_currency("JPY", "USD", 1000)["success"]
    state["open"] = False
    assert b.place_order("7203.T", "SELL", 100, 3000.0)["outside_session"]
    rows = [json.loads(l) for l in (tmp / "intents" / "intents-jp-2026-11-20.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert (r["market"], r["account"], r["currency"], r["order_type"], r["qty"], r["side"]) == \
        ("TSE", "cash", "JPY", "MARKET", 300, "BUY")
    assert r["schema"] == "lumina-intent/1" and r["limit_price"] is None and r["stop_loss_price"] == "2800.0"


def test_create_broker_webull_jp_cash(tmp_path, monkeypatch):
    from core.ai_hedge_fund.webull_intent_broker import WebullJpCashIntentBroker, create_broker
    monkeypatch.setenv("LUMINA_BROKER", "webull_jp_cash")
    monkeypatch.setenv("LUMINA_WEBULL_INTENT_DIR", str(tmp_path / "i"))
    monkeypatch.setenv("LUMINA_DB_PATH", str(tmp_path / "l.db"))
    assert isinstance(create_broker(), WebullJpCashIntentBroker)
