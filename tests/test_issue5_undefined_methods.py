# -*- coding: utf-8 -*-
"""
Issue #5: 定義の無いメソッドを呼んでいた 3 箇所の回帰テスト。

- MarketScreener.get_market_breadth_regime
- Alpha158Engine.compute_symbol_summary
- broker_engine.get_order_book

すべてオフラインで動く（外部通信は差し替える）。
"""
import ast
import importlib
import inspect
import pathlib

import numpy as np
import pandas as pd
import pytest

from core.ai_hedge_fund.alpha158_engine import Alpha158Engine
from core.ai_hedge_fund.broker_client import BaseBrokerClient, IBKRClient, KabuStationClient, PaperTradingEngine
from core.ai_hedge_fund.factor_ontology import FactorOntology
from core.ai_hedge_fund.market_data_feed import MarketDataFeed
from core.ai_hedge_fund.market_screener import MarketScreener

PKG_DIR = pathlib.Path(__file__).resolve().parents[1] / "core" / "ai_hedge_fund"


# ---------------------------------------------------------------------------
# 静的検査: 「クラス名.メソッド(...)」の呼び出し先が、実際に定義されているか
# ---------------------------------------------------------------------------
def _package_classes():
    classes = {}
    for path in sorted(PKG_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            mod = importlib.import_module(f"core.ai_hedge_fund.{path.stem}")
        except Exception:
            continue  # 任意依存が無くて読めないモジュールは対象外
        for name, obj in vars(mod).items():
            if inspect.isclass(obj) and obj.__module__ == mod.__name__:
                classes[name] = obj
    return classes


def test_no_calls_to_undefined_class_methods():
    classes = _package_classes()
    assert "MarketScreener" in classes and "Alpha158Engine" in classes
    missing = []
    for path in sorted(PKG_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id in classes:
                if not hasattr(classes[fn.value.id], fn.attr):
                    missing.append(f"{path.name}:{node.lineno} {fn.value.id}.{fn.attr}")
    assert missing == [], "定義の無いメソッドを呼んでいる: " + ", ".join(missing)


@pytest.mark.parametrize("broker_cls", [BaseBrokerClient, PaperTradingEngine, KabuStationClient, IBKRClient])
def test_every_broker_defines_get_order_book(broker_cls):
    assert callable(getattr(broker_cls, "get_order_book", None))


# ---------------------------------------------------------------------------
# get_order_book: 既定は「板なし」を明示する
# ---------------------------------------------------------------------------
def test_base_order_book_reports_unavailable():
    ob = BaseBrokerClient().get_order_book("AAPL", depth=10)
    assert ob["available"] is False
    assert ob["bids"] == [] and ob["asks"] == []
    assert ob["ofi_analysis"] == {}
    assert ob["reason"]


def test_swing_gate_marks_order_book_unavailable(monkeypatch):
    from core.ai_hedge_fund import auto_trader

    monkeypatch.setattr(auto_trader.yf, "download", lambda *a, **k: None)
    auto_trader.AutoTrader._swing_ofi_gate_retries.pop("AAPL", None)
    logs = []
    allow, mode, ofi = auto_trader.AutoTrader.evaluate_swing_ofi_execution_gate(
        symbol="AAPL", broker_engine=BaseBrokerClient(), symbol_name="Apple", now_str="t", logs=logs
    )
    assert allow is True and mode == "NORMAL"
    assert ofi["order_book_available"] is False


# ---------------------------------------------------------------------------
# get_market_breadth_regime
# ---------------------------------------------------------------------------
def _patch_universe(monkeypatch, above: int, below: int, unknown: int = 0):
    syms = [f"UP{i}" for i in range(above)] + [f"DN{i}" for i in range(below)] + [f"NA{i}" for i in range(unknown)]
    monkeypatch.setattr(MarketScreener, "UNIVERSE", {"TEST": [{"symbol": s, "name": s} for s in syms]})
    monkeypatch.setattr(MarketScreener, "get_custom_watchlist", classmethod(lambda cls: []))

    def fake_stock_data(symbol, force_refresh=False):
        if symbol.startswith("UP"):
            return {"current_price": 110.0, "history_metrics": {"sma_200": 100.0}}
        if symbol.startswith("DN"):
            return {"current_price": 90.0, "history_metrics": {"sma_200": 100.0}}
        return {"current_price": None, "history_metrics": {}}

    monkeypatch.setattr(MarketDataFeed, "get_stock_data", staticmethod(fake_stock_data))


def test_breadth_normal_regime(monkeypatch):
    _patch_universe(monkeypatch, above=12, below=8)
    res = MarketScreener.get_market_breadth_regime()
    assert res["available"] is True
    assert res["breadth_pct"] == 60.0
    assert res["regime"] == "NORMAL"
    assert res["max_slots_cap"] == 6
    assert res["min_rs_required"] == 65
    assert res["sample_size"] == 20


def test_breadth_defensive_regime_below_45pct(monkeypatch):
    _patch_universe(monkeypatch, above=8, below=12)
    res = MarketScreener.get_market_breadth_regime()
    assert res["breadth_pct"] == 40.0
    assert res["regime"] == "DEFENSIVE"
    assert res["max_slots_cap"] == 2


def test_breadth_exactly_45pct_is_not_defensive(monkeypatch):
    _patch_universe(monkeypatch, above=9, below=11)
    res = MarketScreener.get_market_breadth_regime()
    assert res["breadth_pct"] == 45.0
    assert res["max_slots_cap"] == 6


def test_breadth_unknown_when_data_missing_falls_to_defensive(monkeypatch):
    _patch_universe(monkeypatch, above=3, below=0, unknown=30)
    res = MarketScreener.get_market_breadth_regime()
    assert res["available"] is False
    assert res["breadth_pct"] is None
    assert res["regime"] == "UNKNOWN"
    assert res["max_slots_cap"] == 2
    for key in ("regime_label", "min_rs_required"):
        assert key in res


def test_breadth_survives_feed_exception(monkeypatch):
    _patch_universe(monkeypatch, above=0, below=0, unknown=5)

    def boom(symbol, force_refresh=False):
        raise RuntimeError("network down")

    monkeypatch.setattr(MarketDataFeed, "get_stock_data", staticmethod(boom))
    res = MarketScreener.get_market_breadth_regime()
    assert res["available"] is False and res["max_slots_cap"] == 2


# ---------------------------------------------------------------------------
# compute_symbol_summary
# ---------------------------------------------------------------------------
def _ohlcv(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    idx = pd.bdate_range("2025-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def test_alpha158_summary_keys_and_units():
    df = _ohlcv()
    res = Alpha158Engine.compute_symbol_summary("TEST", df=df)
    assert res["available"] is True
    assert res["n_factors"] == 158
    assert res["composite_z_score"] is None  # 合成の定義が無いので作らない
    kf = res["key_factors"]
    close, vol = df["Close"], df["Volume"]
    assert kf["ROC_20"] == pytest.approx((close.iloc[-1] / close.iloc[-21] - 1) * 100, abs=1e-3)
    assert 0.0 <= kf["RSV_20"] <= 1.0
    assert kf["VMA_RATIO_20"] == pytest.approx(vol.iloc[-1] / vol.tail(20).mean(), abs=1e-3)
    assert kf["STD_20"] == pytest.approx(close.pct_change().tail(20).std() * 100, rel=1e-9)
    assert res["as_of"] == df.index[-1].strftime("%Y-%m-%d")


def test_alpha158_summary_feeds_factor_ontology():
    res = Alpha158Engine.compute_symbol_summary("TEST", df=_ohlcv())
    bundle = FactorOntology.build_ontology_bundle(stock_data={"symbol": "TEST"}, alpha158_summary=res)
    text = str(bundle)
    assert "roc_20" in text and "rsv_20" in text


def test_alpha158_summary_short_history_is_unavailable():
    res = Alpha158Engine.compute_symbol_summary("TEST", df=_ohlcv(n=30))
    assert res["available"] is False and "不足" in res["reason"]
