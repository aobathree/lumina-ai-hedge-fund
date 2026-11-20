# -*- coding: utf-8 -*-
"""
Issue #8: データが取れないときの既定値が「強気」「有意」「好成績」になっていた箇所の回帰テスト。

失敗時・実績なしのときに、値を作らず「不明 / 実績なし」を返すことを確かめる。すべてオフライン。
"""
import pathlib

import numpy as np
import pandas as pd
import pytest

from core.ai_hedge_fund import lead_lag_analyzer as lla
from core.ai_hedge_fund.forecast_tracker import ForecastTracker
from core.ai_hedge_fund.lead_lag_analyzer import LeadLagAnalyzer

PKG_DIR = pathlib.Path(__file__).resolve().parents[1] / "core" / "ai_hedge_fund"


# ---------------------------------------------------------------------------
# lead_lag_analyzer
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_lead_lag_cache():
    LeadLagAnalyzer._cache.clear()
    yield
    LeadLagAnalyzer._cache.clear()


def _assert_unknown(res):
    assert res["available"] is False
    assert res["correlation"] is None
    assert res["granger_pvalue"] is None
    assert res["is_statistically_significant"] is False
    assert res["lead_lag_signal"] == "UNKNOWN"
    assert res["unavailable_reason"]
    assert "判定不能" in res["summary_text"]


def test_lead_lag_download_returns_nothing(monkeypatch):
    monkeypatch.setattr(lla.yf, "download", lambda *a, **k: None)
    _assert_unknown(LeadLagAnalyzer.analyze_lead_lag("AAPL"))


def test_lead_lag_download_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(lla.yf, "download", boom)
    _assert_unknown(LeadLagAnalyzer.analyze_lead_lag("AAPL"))


def _pair_frame(target: str, bench: str, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2026-01-05", periods=n)
    a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    b = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.concat({"Close": pd.DataFrame({target: a, bench: b}, index=idx)}, axis=1)


def test_lead_lag_short_history(monkeypatch):
    bench = LeadLagAnalyzer.get_primary_benchmark("AAPL")["symbol"]
    monkeypatch.setattr(lla.yf, "download", lambda *a, **k: _pair_frame("AAPL", bench, 10))
    res = LeadLagAnalyzer.analyze_lead_lag("AAPL")
    _assert_unknown(res)
    assert "データ不足" in res["unavailable_reason"]


def test_lead_lag_unknown_result_is_not_cached(monkeypatch):
    monkeypatch.setattr(lla.yf, "download", lambda *a, **k: None)
    LeadLagAnalyzer.analyze_lead_lag("AAPL")
    assert "AAPL" not in LeadLagAnalyzer._cache


def test_lead_lag_without_granger_test_is_not_significant(monkeypatch):
    bench = LeadLagAnalyzer.get_primary_benchmark("AAPL")["symbol"]
    monkeypatch.setattr(lla.yf, "download", lambda *a, **k: _pair_frame("AAPL", bench, 60))
    monkeypatch.setattr(lla, "HAS_STATSMODELS", False)
    res = LeadLagAnalyzer.analyze_lead_lag("AAPL")
    assert res["available"] is True
    assert res["granger_pvalue"] is None
    assert res["is_statistically_significant"] is False
    assert "p値 未算出" in res["summary_text"]


# ---------------------------------------------------------------------------
# forecast_tracker
# ---------------------------------------------------------------------------
def test_track_record_with_no_verified_forecasts(monkeypatch):
    monkeypatch.setattr(ForecastTracker, "load_history", classmethod(lambda cls: []))
    res = ForecastTracker.get_track_record_summary()
    assert res["verified_count"] == 0
    assert res["accuracy_rate_pct"] is None
    assert res["avg_error_pct"] is None
    assert res["avg_return_pct"] is None
    md = res["note_markdown"]
    assert "累計予測検証数**: 0件" in md
    assert "実績なし" in md
    for fake in ("83.3", "1.9%", "4.2%", "12件"):
        assert fake not in md


def test_track_record_with_verified_forecasts(monkeypatch):
    hist = [
        {"status": "ACCURATE", "is_accurate": True, "error_pct": 1.0, "actual_return_pct": 2.0,
         "ticker": "AAA", "base_price": 10.0, "actual_price": 10.2},
        {"status": "UNDERPERFORM", "is_accurate": False, "error_pct": 3.0, "actual_return_pct": -4.0,
         "ticker": "BBB", "base_price": 20.0, "actual_price": 19.2},
    ]
    monkeypatch.setattr(ForecastTracker, "load_history", classmethod(lambda cls: hist))
    res = ForecastTracker.get_track_record_summary()
    assert res["verified_count"] == 2
    assert res["accuracy_rate_pct"] == 50.0
    assert res["avg_error_pct"] == 2.0
    assert res["avg_return_pct"] == -1.0
    assert "累計予測検証数**: 2件" in res["note_markdown"]
    assert "-1.0%" in res["note_markdown"]


# ---------------------------------------------------------------------------
# quant_reporter / fund_tear_sheet
# ---------------------------------------------------------------------------
# quant_reporter は streamlit、fund_tear_sheet は matplotlib を読み込む。
# どちらも CI の `.[test]` には入っていないので、無ければその部分だけ飛ばす。
@pytest.fixture
def quant_reporter():
    pytest.importorskip("streamlit")
    from core.ai_hedge_fund import quant_reporter as mod
    return mod


@pytest.fixture
def FundTearSheet():
    pytest.importorskip("streamlit")
    pytest.importorskip("matplotlib")
    from core.ai_hedge_fund.fund_tear_sheet import FundTearSheet as cls
    return cls


class FakeEngine:
    def __init__(self, equities=(), trades=()):
        self._eq = list(equities)
        self._trades = list(trades)

    def get_equity_curve(self, limit=200):
        base = pd.Timestamp("2026-01-05 15:00:00")
        return [
            {"timestamp": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S"), "total_equity_jpy": v}
            for i, v in enumerate(self._eq)
        ]

    def get_trade_memories(self, symbol=None, limit=30):
        return list(self._trades)

    def get_portfolio_summary(self):
        return {"total_equity_usd": 6400.0}

    def get_positions(self):
        return []


def test_closed_trades_exclude_entries(quant_reporter):
    eng = FakeEngine(trades=[
        {"trade_type": "BUY_ENTRY", "realized_pnl": 0.0, "currency": "USD"},
        {"trade_type": "STOP_LOSS", "realized_pnl": -12.5, "currency": "USD"},
    ])
    rows = quant_reporter.closed_trades_from_engine(eng)
    assert len(rows) == 1 and rows[0]["trade_type"] == "STOP_LOSS"


def test_quant_metrics_with_no_track_record(quant_reporter):
    m = quant_reporter.QuantReporter.compute_metrics(pd.Series(dtype=float), broker_engine=FakeEngine())
    assert m["is_warmup"] is True
    assert m["closed_trades"] == 0 and m["win_trades"] == 0
    assert m["win_rate"] is None
    assert m["max_drawdown"] is None
    assert m["days_active"] is None
    assert "実績なし" in m["summary_line"]
    for fake in ("4.58", "100%", "1勝"):
        assert fake not in m["summary_line"]


def test_quant_metrics_count_wins_from_recorded_pnl_not_note_text(quant_reporter):
    eng = FakeEngine(
        equities=[1_000_000, 1_020_000, 990_000, 1_005_000],
        trades=[
            {"trade_type": "SELL_EXIT", "realized_pnl": 30.0, "currency": "USD", "lesson_learned": "損切りの反省"},
            {"trade_type": "STOP_LOSS", "realized_pnl": -10.0, "currency": "USD", "lesson_learned": "+ 利確"},
            {"trade_type": "TIMEOUT_EXIT", "realized_pnl": -500.0, "currency": "JPY"},
            {"trade_type": "BUY_ENTRY", "realized_pnl": 0.0, "currency": "USD"},
        ],
    )
    m = quant_reporter.QuantReporter.compute_metrics(pd.Series([0.02, -0.0294, 0.0152]), broker_engine=eng)
    assert (m["closed_trades"], m["win_trades"], m["loss_trades"]) == (3, 1, 2)
    assert m["win_rate"] == pytest.approx(33.3, abs=0.05)
    assert m["realized_pnl_by_currency"] == {"USD": 20.0, "JPY": -500.0}
    assert m["max_drawdown"] == pytest.approx((990_000 / 1_020_000 - 1) * 100, abs=0.01)
    assert "+$20.00" in m["summary_line"] and "-¥500" in m["summary_line"]


def test_monthly_table_without_returns_is_nan_not_a_made_up_number(quant_reporter):
    table = quant_reporter.QuantReporter.compute_monthly_returns_table(pd.Series(dtype=float))
    assert table.isna().all().all()


def test_tear_sheet_with_no_track_record(tmp_path, monkeypatch, FundTearSheet):
    monkeypatch.setattr(FundTearSheet, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(FundTearSheet, "TEAR_SHEET_PNG", str(tmp_path / "t.png"))
    k = FundTearSheet.generate_tear_sheet(FakeEngine(), save_image=True)
    for key in ("cumulative_return_pct", "annualized_volatility_pct", "sharpe_ratio", "sortino_ratio",
                "calmar_ratio", "max_drawdown_pct", "win_rate_pct", "profit_factor"):
        assert k[key] is None, key
    assert k["total_trades_count"] == 0
    assert (tmp_path / "t.png").exists()  # 実績が無くても描画は落ちない


def test_tear_sheet_from_recorded_equity(tmp_path, monkeypatch, FundTearSheet):
    monkeypatch.setattr(FundTearSheet, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(FundTearSheet, "TEAR_SHEET_PNG", str(tmp_path / "t.png"))
    rng = np.random.default_rng(11)
    eq = list(1_000_000 * np.cumprod(1 + rng.normal(0.001, 0.01, 40)))
    eng = FakeEngine(equities=eq, trades=[
        {"trade_type": "SELL_EXIT", "realized_pnl": 300.0, "currency": "USD"},
        {"trade_type": "STOP_LOSS", "realized_pnl": -100.0, "currency": "USD"},
    ])
    k = FundTearSheet.generate_tear_sheet(eng, save_image=True)
    s = pd.Series(eq)
    assert k["cumulative_return_pct"] == pytest.approx((eq[-1] / eq[0] - 1) * 100, abs=0.01)
    assert k["max_drawdown_pct"] == pytest.approx(((s - s.cummax()) / s.cummax()).min() * 100, abs=0.01)
    rets = s.pct_change().dropna()
    assert k["sharpe_ratio"] == pytest.approx(rets.mean() / rets.std() * np.sqrt(252), abs=0.01)
    assert k["win_rate_pct"] == 50.0 and k["profit_factor"] == 3.0
    assert k["total_trades_count"] == 2
    assert (tmp_path / "t.png").exists()


def test_tear_sheet_short_record_gives_no_annualized_ratios(tmp_path, monkeypatch, FundTearSheet):
    monkeypatch.setattr(FundTearSheet, "OUTPUT_DIR", str(tmp_path))
    k = FundTearSheet.generate_tear_sheet(FakeEngine(equities=[1_000_000, 1_010_000, 1_005_000]), save_image=False)
    assert k["cumulative_return_pct"] == 0.5
    assert k["sharpe_ratio"] is None and k["annualized_volatility_pct"] is None


# ---------------------------------------------------------------------------
# 静的検査: 以前の作り物の数値が残っていないこと
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fname, needles", [
    ("lead_lag_analyzer.py", ['"correlation": 0.65', "p_val = 0.04", "p_val = 0.05"]),
    ("fund_tear_sheet.py", ["66.7", "1.85", "1.94", "np.random.seed"]),
    ("forecast_tracker.py", ["83.3", "max(total_v, 12)"]),
    ("quant_reporter.py", ["4.58", "1.83"]),
    ("fund_content_generator.py", ["+$4.58", "勝率: 100% (1勝0敗)"]),
])
def test_fabricated_defaults_are_gone(fname, needles):
    text = (PKG_DIR / fname).read_text(encoding="utf-8")
    for n in needles:
        assert n not in text, f"{fname} に {n!r} が残っている"
