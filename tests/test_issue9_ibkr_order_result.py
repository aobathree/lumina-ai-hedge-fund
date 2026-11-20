# -*- coding: utf-8 -*-
"""
Issue #9: IBKRClient.place_order が、送っていないブラケット注文を送ったと報告しないこと、
受付を確かめずに success を返さないことの回帰テスト。ib_insync は差し替える（接続しない）。
"""
import pathlib
import sys
import types

import pytest

from core.ai_hedge_fund import broker_client as bc
from core.ai_hedge_fund.broker_client import IBKRClient

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeIB:
    """placeOrder のあと、sleep のたびに statuses を 1 つずつ進める。"""

    def __init__(self, statuses, log_message=""):
        self._statuses = list(statuses)
        self.placed = []
        self.trade = None
        self._log_message = log_message
        self.slept = 0.0

    def isConnected(self):
        return True

    def qualifyContracts(self, contract):
        return [contract]

    def placeOrder(self, contract, order):
        self.placed.append((contract, order))
        self.trade = _Obj(
            order=_Obj(orderId=777),
            orderStatus=_Obj(status="PendingSubmit", filled=0.0, avgFillPrice=0.0),
            log=[_Obj(message=self._log_message)] if self._log_message else [],
        )
        return self.trade

    def sleep(self, secs):
        self.slept += secs
        if self._statuses:
            self.trade.orderStatus.status = self._statuses.pop(0)


@pytest.fixture
def fake_ib_insync(monkeypatch):
    mod = types.ModuleType("ib_insync")
    mod.Stock = lambda *a, **k: _Obj(kind="Stock", args=a, kwargs=k)
    mod.MarketOrder = lambda action, qty: _Obj(kind="MKT", action=action, totalQuantity=qty)
    mod.LimitOrder = lambda action, qty, px: _Obj(kind="LMT", action=action, totalQuantity=qty, lmtPrice=px)
    monkeypatch.setitem(sys.modules, "ib_insync", mod)
    return mod


def _client(monkeypatch, ib):
    client = IBKRClient()
    monkeypatch.setattr(client, "_get_connection", lambda: ib)
    return client


def test_accepted_order_reports_no_bracket(monkeypatch, fake_ib_insync):
    ib = FakeIB(["Submitted"])
    res = _client(monkeypatch, ib).place_order("AAPL", "BUY", 10, stop_loss_price=90.0, target_price=120.0)
    assert res["success"] is True and res["status"] == "Submitted" and res["order_id"] == 777
    assert res["bracket_attached"] is False and res["stop_loss_on_exchange"] is False
    assert "ブラケット" not in res["message"]
    assert "取引所に置いていません" in res["message"]
    assert len(ib.placed) == 1  # 送ったのは親注文 1 本だけ


def test_message_has_no_protection_note_without_stop_or_target(monkeypatch, fake_ib_insync):
    res = _client(monkeypatch, FakeIB(["Filled"])).place_order("AAPL", "BUY", 10)
    assert res["success"] is True and "取引所に置いていません" not in res["message"]


def test_waits_for_acknowledgement(monkeypatch, fake_ib_insync):
    ib = FakeIB(["PendingSubmit", "PendingSubmit", "PreSubmitted"])
    res = _client(monkeypatch, ib).place_order("AAPL", "BUY", 10)
    assert res["success"] is True and res["status"] == "PreSubmitted"
    assert ib.slept == pytest.approx(3 * IBKRClient.ORDER_ACK_POLL_SECONDS)


@pytest.mark.parametrize("status", ["Cancelled", "ApiCancelled", "Inactive"])
def test_rejected_order_is_not_success(monkeypatch, fake_ib_insync, status):
    ib = FakeIB([status], log_message="Order rejected - reason: insufficient funds")
    res = _client(monkeypatch, ib).place_order("AAPL", "BUY", 10)
    assert res["success"] is False and res["status"] == status
    assert "insufficient funds" in res["message"]
    assert "unconfirmed" not in res


def test_unacknowledged_order_is_flagged_unconfirmed(monkeypatch, fake_ib_insync):
    ib = FakeIB([])  # いつまでも PendingSubmit
    res = _client(monkeypatch, ib).place_order("AAPL", "BUY", 10)
    assert res["success"] is False and res["unconfirmed"] is True
    assert res["order_id"] == 777
    assert "再送しないでください" in res["message"]
    assert ib.slept == pytest.approx(IBKRClient.ORDER_ACK_TIMEOUT_SECONDS)


def test_source_and_docs_no_longer_claim_a_bracket():
    src = (ROOT / "core" / "ai_hedge_fund" / "broker_client.py").read_text(encoding="utf-8")
    assert "ブラケット逆指値" not in src
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "OCO Bracket**" not in readme and "IBKR OCA Bracket Order Adapter" not in readme
    en = (ROOT / "docs" / "lumina_fund_master_system_architecture.md").read_text(encoding="utf-8")
    assert "Places entry, take-profit, and stop-loss simultaneously" not in en
    ja = (ROOT / "docs" / "lumina_fund_master_system_architecture.ja.md").read_text(encoding="utf-8")
    assert "IBKR取引所サーバー上へ事前配置" not in ja
