# -*- coding: utf-8 -*-
"""
WebullIntentBroker のテスト。ネットワークには出ない（Webull にも yfinance にも）。

確かめること:
- 米国株以外、立会の外、寄付の直後の新規、空売りを拒み、そのとき意図を書かないこと
- 受け付けた注文 1 件につき、意図が 1 行だけ書かれること
- paper では参考価格で、await では執行機の結果の約定値で、台帳に記帳すること
- await で結果が来ない・拒まれたときは、台帳に記帳しないこと
- このリポジトリに、Webull の発注 API を呼ぶコードが無いこと
"""
import datetime
import json
import pathlib
from decimal import Decimal

import pytest

from core.ai_hedge_fund import broker_client as bc
from core.ai_hedge_fund import webull_intent_broker as wib
from core.ai_hedge_fund.market_data_feed import MarketDataFeed
from core.ai_hedge_fund.webull_intent_broker import (
    UsSessionCalendar, WebullIntentBroker, create_broker, is_us_symbol, marketable_limit,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
UTC = datetime.timezone.utc
ET = wib.ET

# 2026-01-06（火）は立会のある日。10:00 ET = 寄付の 30 分後
T_OPEN_PLUS_30 = datetime.datetime(2026, 1, 6, 10, 0, tzinfo=ET).astimezone(UTC)
T_OPEN_PLUS_2 = datetime.datetime(2026, 1, 6, 9, 32, tzinfo=ET).astimezone(UTC)
T_BEFORE_OPEN = datetime.datetime(2026, 1, 6, 9, 0, tzinfo=ET).astimezone(UTC)
T_AFTER_CLOSE = datetime.datetime(2026, 1, 6, 16, 0, tzinfo=ET).astimezone(UTC)
T_SATURDAY = datetime.datetime(2026, 1, 10, 11, 0, tzinfo=ET).astimezone(UTC)
DAY = datetime.date(2026, 1, 6)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMINA_TEST_MODE", "1")
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(bc.PaperTradingEngine, "get_usdjpy_rate", classmethod(lambda cls: 150.0))
    monkeypatch.setattr(bc.PaperTradingEngine, "get_eurjpy_rate", classmethod(lambda cls: 160.0))

    def fake_stock_data(symbol, force_refresh=False):
        return {"symbol": symbol, "company_name": symbol, "currency": "USD", "current_price": 100.0, "history_metrics": {}}

    monkeypatch.setattr(MarketDataFeed, "get_stock_data", staticmethod(fake_stock_data))
    monkeypatch.setattr(MarketDataFeed, "get_realtime_quote",
                        staticmethod(lambda s, force_refresh=False: {"price": 100.0, "current_price": 100.0}), raising=False)
    MarketDataFeed._cache.clear()
    return tmp_path


def make(env, mode="paper", now=T_OPEN_PLUS_30, **kw):
    clock = Clock(now)
    broker = WebullIntentBroker(
        db_path=str(env / f"ledger-{mode}.db"), intent_dir=str(env / "intents"), mode=mode,
        initial_usd=3200.0, now_fn=clock, sleep_fn=lambda s: None, **kw,
    )
    return broker, clock


def intents(broker, day=DAY):
    path = broker.intent_path(day)
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


# ---------------------------------------------------------------------------
# 小さな部品
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sym, ok", [("AAPL", True), ("BRK-B", True), ("7203.T", False), ("ASML.AS", False), ("^GSPC", False), ("", False)])
def test_is_us_symbol(sym, ok):
    assert is_us_symbol(sym) is ok


def test_marketable_limit_rounds_against_us():
    assert marketable_limit("BUY", 100.0) == Decimal("100.20")
    assert marketable_limit("SELL", 100.0) == Decimal("99.80")
    assert marketable_limit("BUY", 86.56) == Decimal("86.74")   # 86.73312 を切り上げ
    assert marketable_limit("SELL", 86.55) == Decimal("86.37")  # 86.3769 を切り下げ


def test_calendar_without_table_is_weekday_only():
    cal = UsSessionCalendar(None)
    s = cal.session(T_OPEN_PLUS_30)
    assert s["is_open"] is True and s["source"] == "weekday_only"
    assert cal.session(T_AFTER_CLOSE)["is_open"] is False  # 16:00 ちょうどは立会の外
    assert cal.session(T_SATURDAY)["is_open"] is False


def test_calendar_reads_holidays_and_half_days(tmp_path):
    (tmp_path / "nyse-calendar-2026.toml").write_text(
        'year = 2026\n[[holiday]]\ndate = "2026-11-26"\nname = "Thanksgiving Day"\n'
        '[[half_day]]\ndate = "2026-11-27"\nname = "Day after Thanksgiving"\n', encoding="utf-8")
    cal = UsSessionCalendar(str(tmp_path))
    thanksgiving = datetime.datetime(2026, 11, 26, 11, 0, tzinfo=ET).astimezone(UTC)
    assert cal.session(thanksgiving)["is_open"] is False
    assert cal.session(thanksgiving)["source"] == "nyse_calendar"
    half_1230 = datetime.datetime(2026, 11, 27, 12, 30, tzinfo=ET).astimezone(UTC)
    half_1330 = datetime.datetime(2026, 11, 27, 13, 30, tzinfo=ET).astimezone(UTC)
    assert cal.session(half_1230)["is_open"] is True
    assert cal.session(half_1330)["is_open"] is False


# ---------------------------------------------------------------------------
# 台帳の元手
# ---------------------------------------------------------------------------
def test_fresh_ledger_is_seeded_with_usd_only(env):
    broker, _ = make(env)
    bal = broker.get_account_balance()
    assert bal["USD"] == 3200.0 and bal["JPY"] == 0.0


def test_existing_ledger_is_not_reseeded(env):
    broker, _ = make(env)
    assert broker.place_order("AAPL", "BUY", 2, 100.0)["success"] is True
    again, _ = make(env)
    assert again.get_account_balance()["USD"] == pytest.approx(3200.0 - 200.0)


# ---------------------------------------------------------------------------
# 拒むもの（意図を書かない）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs, now, needle", [
    (dict(symbol="7203.T", side="BUY", quantity=100, price=3000.0), T_OPEN_PLUS_30, "米国株以外"),
    (dict(symbol="AAPL", side="BUY", quantity=1, price=100.0), T_BEFORE_OPEN, "立会時間の外"),
    (dict(symbol="AAPL", side="BUY", quantity=1, price=100.0), T_AFTER_CLOSE, "立会時間の外"),
    (dict(symbol="AAPL", side="BUY", quantity=1, price=100.0), T_SATURDAY, "立会時間の外"),
    (dict(symbol="AAPL", side="BUY", quantity=1, price=100.0), T_OPEN_PLUS_2, "寄付から"),
    (dict(symbol="AAPL", side="SELL", quantity=1, price=100.0), T_OPEN_PLUS_30, "空売りはしない"),
    (dict(symbol="AAPL", side="BUY", quantity=0, price=100.0), T_OPEN_PLUS_30, "数量"),
    (dict(symbol="AAPL", side="BUY", quantity=1.5, price=100.0), T_OPEN_PLUS_30, "端株"),
    (dict(symbol="AAPL", side="BUY", quantity=40, price=100.0), T_OPEN_PLUS_30, "余力が足りない"),
    (dict(symbol="AAPL", side="HOLD", quantity=1, price=100.0), T_OPEN_PLUS_30, "向きが不正"),
])
def test_refusals_write_no_intent(env, kwargs, now, needle):
    broker, _ = make(env, now=now)
    res = broker.place_order(**kwargs)
    assert res["success"] is False and res["intent_written"] is False
    assert needle in res["error"]
    assert not list((env / "intents").glob("intents-*.jsonl"))
    assert broker.get_positions() == []


def test_outside_session_is_refused_even_when_hours_are_not_enforced(env):
    broker, _ = make(env, now=T_BEFORE_OPEN)
    res = broker.place_order("AAPL", "BUY", 1, 100.0, enforce_market_hours=False)
    assert res["success"] is False and res.get("outside_session") is True


def test_close_is_allowed_right_after_the_open(env):
    broker, clock = make(env)
    assert broker.place_order("AAPL", "BUY", 3, 100.0)["success"] is True
    clock.now = datetime.datetime(2026, 1, 7, 9, 31, tzinfo=ET).astimezone(UTC)  # 翌日の寄付の 1 分後
    res = broker.place_order("AAPL", "SELL", 3, 101.0)
    assert res["success"] is True
    assert intents(broker, datetime.date(2026, 1, 7))[0]["intent"] == "CLOSE"


def test_sell_more_than_held_is_refused(env):
    broker, clock = make(env)
    broker.place_order("AAPL", "BUY", 3, 100.0)
    clock.now += datetime.timedelta(minutes=10)
    res = broker.place_order("AAPL", "SELL", 4, 100.0)
    assert res["success"] is False and "超える" in res["error"]
    assert len(intents(broker)) == 1


# ---------------------------------------------------------------------------
# paper: 意図を 1 行書き、参考価格で記帳する
# ---------------------------------------------------------------------------
def test_paper_buy_writes_one_intent_and_books_the_ledger(env):
    broker, _ = make(env)
    res = broker.place_order("AAPL", "BUY", 5, 100.0, notes="合議の買い", target_price=125.0, stop_loss_price=90.0)
    assert res["success"] is True and res["intent_written"] is True and res["fill_mode"] == "paper"
    rows = intents(broker)
    assert len(rows) == 1
    it = rows[0]
    assert it["schema"] == "lumina-intent/1" and it["intent_id"] == res["intent_id"]
    assert (it["symbol"], it["side"], it["intent"], it["qty"]) == ("AAPL", "BUY", "OPEN", 5)
    assert it["limit_price"] == "100.20" and it["reference_price"] == "100.0" and it["reference_source"] == "caller"
    assert it["stop_loss_price"] == "90.0" and it["target_price"] == "125.0"
    assert it["session_date_et"] == "2026-01-06" and it["calendar"] == "weekday_only"
    created = datetime.datetime.fromisoformat(it["created_at"])
    until = datetime.datetime.fromisoformat(it["valid_until"])
    assert created == T_OPEN_PLUS_30 and (until - created).total_seconds() == wib.INTENT_TTL_SEC
    pos = broker.get_positions()
    assert len(pos) == 1 and pos[0]["symbol"] == "AAPL" and pos[0]["quantity"] == 5
    assert broker.get_account_balance()["USD"] == pytest.approx(3200.0 - 500.0)


def test_reference_price_falls_back_to_the_feed(env):
    broker, _ = make(env)
    res = broker.place_order("MSFT", "BUY", 1)
    assert res["success"] is True
    assert intents(broker)[0]["reference_source"] == "yfinance"


def test_no_reference_price_means_no_intent(env, monkeypatch):
    broker, _ = make(env)
    monkeypatch.setattr(MarketDataFeed, "get_stock_data", staticmethod(lambda s, force_refresh=False: {"current_price": None}))
    res = broker.place_order("MSFT", "BUY", 1)
    assert res["success"] is False and "参考価格が取れない" in res["error"]
    assert intents(broker) == []


def test_repeat_order_inside_cooldown_is_blocked_before_writing(env):
    broker, _ = make(env)
    assert broker.place_order("AAPL", "BUY", 1, 100.0)["success"] is True
    res = broker.place_order("AAPL", "BUY", 1, 100.0)
    assert res["success"] is False and res.get("idempotency_blocked") is True
    assert len(intents(broker)) == 1


def test_intent_ids_are_unique(env):
    broker, _ = make(env)
    broker.place_order("AAPL", "BUY", 1, 100.0)
    broker.place_order("MSFT", "BUY", 1, 100.0)
    ids = [r["intent_id"] for r in intents(broker)]
    assert len(ids) == 2 and len(set(ids)) == 2


# ---------------------------------------------------------------------------
# await: 執行機の結果を待つ
# ---------------------------------------------------------------------------
def _executor(broker, **result):
    """sleep のたびに、最後の意図への結果を書く「執行機」の代わり。"""
    def sleep(_):
        rows = intents(broker)
        if rows and not broker.result_path(DAY).exists():
            row = {"schema": "lumina-intent-result/1", "intent_id": rows[-1]["intent_id"], **result}
            broker.result_path(DAY).write_text(json.dumps(row) + "\n", encoding="utf-8")
    broker._sleep = sleep


def test_await_books_at_the_executors_fill_price(env):
    broker, _ = make(env, mode="await")
    _executor(broker, status="FILLED", filled_qty=5, avg_price="100.13")
    res = broker.place_order("AAPL", "BUY", 5, 100.0)
    assert res["success"] is True and res["fill_price"] == 100.13 and res["filled_quantity"] == 5
    pos = broker.get_positions()[0]
    assert pos["quantity"] == 5 and pos["avg_entry_price"] == pytest.approx(100.13)
    assert broker.get_account_balance()["USD"] == pytest.approx(3200.0 - 500.65)


def test_await_partial_fill_books_only_what_filled(env):
    broker, _ = make(env, mode="await")
    _executor(broker, status="PARTIAL", filled_qty=2, avg_price="100.10")
    res = broker.place_order("AAPL", "BUY", 5, 100.0)
    assert res["success"] is True and res["filled_quantity"] == 2
    assert broker.get_positions()[0]["quantity"] == 2


@pytest.mark.parametrize("status", ["DENIED", "UNFILLED", "ERROR"])
def test_await_not_filled_leaves_the_ledger_untouched(env, status):
    broker, _ = make(env, mode="await")
    _executor(broker, status=status, reasons=["U6: 枠に無い銘柄"])
    res = broker.place_order("AAPL", "BUY", 5, 100.0)
    assert res["success"] is False and res["status"] == status and "U6" in res["error"]
    assert res["intent_written"] is True
    assert broker.get_positions() == [] and broker.get_account_balance()["USD"] == 3200.0


def test_await_timeout_is_unconfirmed_and_blocks_a_resend(env):
    broker, _ = make(env, mode="await", await_timeout_sec=3.0)
    res = broker.place_order("AAPL", "BUY", 5, 100.0)
    assert res["success"] is False and res["unconfirmed"] is True
    assert broker.get_positions() == []
    again = broker.place_order("AAPL", "BUY", 5, 100.0)
    assert again["success"] is False and again.get("idempotency_blocked") is True
    assert len(intents(broker)) == 1


@pytest.mark.parametrize("bad", [
    dict(status="FILLED", filled_qty=9, avg_price="100.1"),   # 頼んだ数より多い
    dict(status="FILLED", filled_qty=5, avg_price="0"),
    dict(status="FILLED", filled_qty=5),                      # 値段が無い
    dict(status="FILLED", filled_qty="x", avg_price="100.1"),
])
def test_await_malformed_result_is_not_booked(env, bad):
    broker, _ = make(env, mode="await")
    _executor(broker, **bad)
    res = broker.place_order("AAPL", "BUY", 5, 100.0)
    assert res["success"] is False and res["unconfirmed"] is True
    assert broker.get_positions() == []


def test_result_for_another_intent_is_ignored(env):
    broker, _ = make(env, mode="await", await_timeout_sec=2.0)
    broker.result_path(DAY).parent.mkdir(parents=True, exist_ok=True)
    broker.result_path(DAY).write_text(
        json.dumps({"schema": "lumina-intent-result/1", "intent_id": "someone-else", "status": "FILLED",
                    "filled_qty": 5, "avg_price": "1.0"}) + "\n{broken json\n", encoding="utf-8")
    res = broker.place_order("AAPL", "BUY", 5, 100.0)
    assert res["success"] is False and res["unconfirmed"] is True


# ---------------------------------------------------------------------------
# 配線
# ---------------------------------------------------------------------------
def test_create_broker_defaults_to_paper(env, monkeypatch):
    monkeypatch.delenv("LUMINA_BROKER", raising=False)
    broker = create_broker()
    assert type(broker) is bc.PaperTradingEngine and not getattr(broker, "us_only", False)


def test_create_broker_webull_intents(env, monkeypatch):
    monkeypatch.setenv("LUMINA_BROKER", "webull_intents")
    monkeypatch.setenv("LUMINA_WEBULL_INTENT_DIR", str(env / "spool"))
    monkeypatch.setenv("LUMINA_DB_PATH", str(env / "db" / "webull.db"))
    monkeypatch.setenv("LUMINA_WEBULL_INITIAL_USD", "3200")
    broker = create_broker()
    assert isinstance(broker, WebullIntentBroker) and broker.us_only is True and broker.mode == "paper"
    assert broker.db_path == str(env / "db" / "webull.db")
    assert bc.PaperTradingEngine().db_path == broker.db_path  # 引数なしの台帳も同じ DB を見る


def test_create_broker_requires_an_intent_dir(env, monkeypatch):
    monkeypatch.setenv("LUMINA_BROKER", "webull_intents")
    monkeypatch.delenv("LUMINA_WEBULL_INTENT_DIR", raising=False)
    with pytest.raises(ValueError):
        create_broker()


def test_unknown_broker_name_is_an_error(env, monkeypatch):
    monkeypatch.setenv("LUMINA_BROKER", "kabu")
    with pytest.raises(ValueError):
        create_broker()


# ---------------------------------------------------------------------------
# このリポジトリは Webull に発注しない
# ---------------------------------------------------------------------------
def test_repository_has_no_webull_order_api_calls():
    hits = []
    for path in list((ROOT / "core").rglob("*.py")) + list((ROOT / "ui").rglob("*.py")) + [ROOT / "app.py"]:
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if ("import" in low and "webull" in low and "webull_intent_broker" not in low) or "tradeclient" in low:
                hits.append(f"{path.name}:{line_no}")
    assert hits == [], f"Webull の SDK を読み込む箇所がある: {hits}"
    for dep_file in ("requirements.txt", "pyproject.toml"):
        assert "webull" not in (ROOT / dep_file).read_text(encoding="utf-8").lower()
