# -*- coding: utf-8 -*-
"""
Issue #6: フィードとエージェントのキー名の不一致の回帰テスト。

- フィードの戻り値が feed_schemas.py の形を満たすこと
- CIO のレジーム判定と CRO の拒否権が、フィードの区分値で動くこと
- エージェントが context_data から読むキーが、合議の渡すキーに含まれること
- RS 順位と F-Score が実際に計算され、取れないときは None になること

すべてオフライン。
"""
import ast
import pathlib

import pandas as pd
import pytest

from core.ai_hedge_fund import feed_schemas as fs
from core.ai_hedge_fund.agents.portfolio_manager_agent import PortfolioManagerAgent
from core.ai_hedge_fund.agents import risk_manager_agent as rma
from core.ai_hedge_fund.macro_liquidity_feed import MacroLiquidityFeed
from core.ai_hedge_fund.market_data_feed import MarketDataFeed
from core.ai_hedge_fund.market_screener import MarketScreener
from core.ai_hedge_fund.options_gex_feed import OptionsGexFeed
from core.ai_hedge_fund.piotroski import compute_f_score

PKG_DIR = pathlib.Path(__file__).resolve().parents[1] / "core" / "ai_hedge_fund"


# ---------------------------------------------------------------------------
# フィードの形
# ---------------------------------------------------------------------------
def _fred_frames(net_change: str):
    walcl = {"up": [6_700_000.0, 6_800_000.0], "down": [6_800_000.0, 6_600_000.0], "flat": [6_700_000.0, 6_700_000.0]}[net_change]

    def fake_read_csv(url, *a, **k):
        if "WALCL" in url:
            return pd.DataFrame({"WALCL": walcl})
        if "WTREGEN" in url:
            return pd.DataFrame({"WTREGEN": [800_000.0, 800_000.0]})
        return pd.DataFrame({"RRPONTSYD": [10.0] * 6})

    return fake_read_csv


@pytest.mark.parametrize("change, regime", [("up", "RISK_ON"), ("down", "RISK_OFF"), ("flat", "NEUTRAL")])
def test_macro_feed_shape_and_regime(monkeypatch, change, regime):
    MacroLiquidityFeed._cache.clear()
    monkeypatch.setattr(pd, "read_csv", _fred_frames(change))
    status = MacroLiquidityFeed.get_macro_status()
    MacroLiquidityFeed._cache.clear()
    assert fs.missing_keys(status, fs.MACRO_REQUIRED_KEYS) == []
    assert status["available"] is True
    assert status["regime"] == regime and regime in fs.MACRO_REGIMES


def test_macro_feed_failure_is_flagged(monkeypatch):
    MacroLiquidityFeed._cache.clear()

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(pd, "read_csv", boom)
    status = MacroLiquidityFeed.get_macro_status()
    MacroLiquidityFeed._cache.clear()
    assert fs.missing_keys(status, fs.MACRO_REQUIRED_KEYS) == []
    assert status["available"] is False
    assert status["regime"] == "NEUTRAL"


def _gex_profiles(monkeypatch, negative: bool):
    prof = {
        "status": "OK", "current_price": 90.0 if negative else 110.0, "gamma_flip": 100.0,
        "regime": "NEGATIVE_GAMMA" if negative else "POSITIVE_GAMMA", "net_gex_millions": 10.0,
    }
    monkeypatch.setattr(OptionsGexFeed, "get_gex_profile", classmethod(lambda cls, sym: dict(prof)))


@pytest.mark.parametrize("negative", [True, False])
def test_turbulence_feed_shape(monkeypatch, negative):
    _gex_profiles(monkeypatch, negative)
    turb = OptionsGexFeed.get_market_turbulence_warning()
    assert fs.missing_keys(turb, fs.TURBULENCE_REQUIRED_KEYS) == []
    assert turb["is_turbulence"] is negative
    ctx = fs.build_gex_context(turb, {"summary_text": "x"})
    assert ctx["is_turbulence"] is negative
    assert ctx["market_turbulence"] == turb["summary_text"]


# ---------------------------------------------------------------------------
# CIO: レジーム判定がフィードの区分値で動く
# ---------------------------------------------------------------------------
def _regime(macro, gex):
    return PortfolioManagerAgent.detect_market_regime({"macro_liquidity": macro, "options_gex": gex})["regime_name"]


def test_cio_regime_follows_feed_values():
    calm = {"is_turbulence": False}
    assert "強気" in _regime({"regime": "RISK_ON"}, calm)
    assert "収縮" in _regime({"regime": "RISK_OFF"}, calm)
    assert "中立" in _regime({"regime": "NEUTRAL"}, calm)


def test_cio_regime_without_data_is_neutral_not_bullish():
    # 以前は status の既定値 "NORMAL" のせいで、データが無いと常に「強気」になっていた
    assert "中立" in PortfolioManagerAgent.detect_market_regime({})["regime_name"]
    assert "中立" in PortfolioManagerAgent.detect_market_regime(None)["regime_name"]


def test_cio_turbulence_uses_real_feed_output(monkeypatch):
    # 実際のフィードの文言（「ネガティブ・ガンマ」「乱気流警報」）で乱流と判定されること
    _gex_profiles(monkeypatch, negative=True)
    turb = OptionsGexFeed.get_market_turbulence_warning()
    assert "乱気流" in turb["summary_text"]
    name = _regime({"regime": "RISK_ON"}, fs.build_gex_context(turb))
    assert "乱流" in name


# ---------------------------------------------------------------------------
# CRO: 拒否権
# ---------------------------------------------------------------------------
def _stock(vol=20.0, **extra):
    data = {"current_price": 100.0, "currency": "USD",
            "history_metrics": {"volatility_annualized_pct": vol, "sma_200": 80.0}}
    data.update(extra)
    return data


def _set_turbulence(monkeypatch, flag: bool):
    monkeypatch.setattr(rma.OptionsGexFeed, "get_market_turbulence_warning",
                        classmethod(lambda cls: {"is_turbulence": flag, "size_multiplier": 0.5 if flag else 1.0, "summary_text": ""}))


def test_cro_liquidity_shock_veto_fires_on_risk_off(monkeypatch):
    _set_turbulence(monkeypatch, True)
    res = rma.RiskManagerAgent.evaluate_risk(_stock(), [], {"macro_liquidity": {"regime": "RISK_OFF"}})
    assert res["stance"] == "VETO"


def test_cro_no_liquidity_veto_when_neutral(monkeypatch):
    _set_turbulence(monkeypatch, True)
    res = rma.RiskManagerAgent.evaluate_risk(_stock(), [], {"macro_liquidity": {"regime": "NEUTRAL"}})
    assert res["stance"] != "VETO"


def test_cro_forecast_contradiction_veto_reads_dual_quantum(monkeypatch):
    _set_turbulence(monkeypatch, False)
    ctx = {"macro_liquidity": {"regime": "NEUTRAL"},
           "dual_quantum": {"consensus_status": "DOUBLE_BULLISH", "ensemble_expected_return_pct": 5.0}}
    res = rma.RiskManagerAgent.evaluate_risk(_stock(piotroski_f_score=3, roe=20.0), [], ctx)
    assert res["stance"] == "VETO"
    res_ok = rma.RiskManagerAgent.evaluate_risk(_stock(piotroski_f_score=8, roe=20.0), [], ctx)
    assert res_ok["stance"] != "VETO"


# ---------------------------------------------------------------------------
# 静的検査: 読む側のキー ⊆ 渡す側のキー
# ---------------------------------------------------------------------------
def _context_keys_built_by_council():
    tree = ast.parse((PKG_DIR / "hedge_fund_council.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if any(isinstance(t, ast.Name) and t.id == "context_data" for t in node.targets):
                return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError("hedge_fund_council.py に context_data の定義が見つからない")


def test_council_context_keys_match_schema():
    assert sorted(_context_keys_built_by_council()) == sorted(fs.COUNCIL_CONTEXT_KEYS)


def _context_keys_read(path: pathlib.Path):
    keys = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and node.args and isinstance(node.args[0], ast.Constant)):
            base = ast.unparse(node.func.value).replace("(", "").replace(")", "")
            if base in ("context_data", "context_data or {}"):
                keys.add(node.args[0].value)
    return keys


def test_agents_only_read_context_keys_that_exist():
    allowed = set(fs.COUNCIL_CONTEXT_KEYS) | set(fs.OPTIONAL_CONTEXT_KEYS)
    files = sorted((PKG_DIR / "agents").glob("*.py")) + [PKG_DIR / "lightgbm_gatekeeper.py"]
    seen = set()
    for path in files:
        keys = _context_keys_read(path)
        seen |= keys
        assert keys <= allowed, f"{path.name} が、合議の渡さないキーを読んでいる: {sorted(keys - allowed)}"
    assert "dual_quantum" in seen and "macro_liquidity" in seen  # 検査が実際に効いていること


@pytest.mark.parametrize("fname, stale", [
    ("agents/portfolio_manager_agent.py", ["summary_headline", "net_liquidity_trend", "ネガティブガンマ"]),
    ("agents/risk_manager_agent.py", ["liquidity_crunch", "net_liquidity_weather", "quantum_forecast"]),
    ("hedge_fund_council.py", ["macro_weather", "summary_headline"]),
    ("reflexion_engine.py", ["macro_weather"]),
    ("lightgbm_gatekeeper.py", ['"SUNNY"']),
])
def test_stale_key_names_are_gone(fname, stale):
    text = (PKG_DIR / fname).read_text(encoding="utf-8")
    for s in stale:
        assert s not in text, f"{fname} に {s} が残っている"


# ---------------------------------------------------------------------------
# RS 順位
# ---------------------------------------------------------------------------
def test_rs_ratings_are_percentile_ranks():
    results = [{"symbol": f"S{i}", "return_1y_pct": float(i)} for i in range(20)] + [{"symbol": "NA", "return_1y_pct": None}]
    MarketScreener.assign_rs_ratings(results)
    by = {r["symbol"]: r["rs_rating"] for r in results}
    assert by["NA"] is None
    assert by["S19"] == 98 and by["S0"] == 2  # (below + 0.5) / 20 * 100 を四捨五入
    assert by["S10"] > by["S9"]
    assert all(1 <= v <= 99 for k, v in by.items() if k != "NA")


def test_rs_ratings_need_enough_symbols():
    results = [{"symbol": f"S{i}", "return_1y_pct": float(i)} for i in range(5)]
    MarketScreener.assign_rs_ratings(results)
    assert all(r["rs_rating"] is None for r in results)


# ---------------------------------------------------------------------------
# F-Score
# ---------------------------------------------------------------------------
CUR, PREV = pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31")


def _statements(strong: bool = True, drop=()):
    fin = {"Net Income": (120.0, 100.0), "Gross Profit": (460.0, 400.0), "Total Revenue": (1100.0, 1000.0)}
    bal = {"Total Assets": (1000.0, 1000.0), "Long Term Debt": (180.0, 200.0), "Current Assets": (330.0, 300.0),
           "Current Liabilities": (200.0, 200.0), "Ordinary Shares Number": (100.0, 100.0)}
    cf = {"Operating Cash Flow": (150.0, 130.0)}
    if not strong:
        fin = {"Net Income": (-10.0, 100.0), "Gross Profit": (350.0, 400.0), "Total Revenue": (900.0, 1000.0)}
        bal = {"Total Assets": (1000.0, 1000.0), "Long Term Debt": (300.0, 200.0), "Current Assets": (250.0, 300.0),
               "Current Liabilities": (200.0, 200.0), "Ordinary Shares Number": (120.0, 100.0)}
        cf = {"Operating Cash Flow": (-20.0, 130.0)}

    def frame(d):
        d = {k: v for k, v in d.items() if k not in drop}
        return pd.DataFrame({CUR: {k: v[0] for k, v in d.items()}, PREV: {k: v[1] for k, v in d.items()}})

    return frame(fin), frame(bal), frame(cf)


def test_f_score_strong_company_scores_nine():
    res = compute_f_score(*_statements(strong=True))
    assert res["score"] == 9 and res["missing"] == []


def test_f_score_weak_company_scores_zero():
    res = compute_f_score(*_statements(strong=False))
    assert res["score"] == 0 and res["missing"] == []


def test_f_score_is_none_when_an_item_is_missing():
    res = compute_f_score(*_statements(drop=("Gross Profit", "Current Assets")))
    assert res["score"] is None
    assert set(res["missing"]) == {"gross_margin_improved", "current_ratio_improved"}


def test_f_score_is_none_without_statements():
    assert compute_f_score(None, None, None)["score"] is None
    assert compute_f_score(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())["score"] is None


def test_f_score_column_order_does_not_matter():
    fin, bal, cf = _statements(strong=True)
    res = compute_f_score(fin[[PREV, CUR]], bal[[PREV, CUR]], cf[[PREV, CUR]])
    assert res["score"] == 9


def test_feed_f_score_failure_returns_none_and_is_not_cached(monkeypatch):
    from core.ai_hedge_fund import market_data_feed as mdf

    def boom(sym):
        raise RuntimeError("offline")

    MarketDataFeed._fscore_cache.pop("ZZZZ", None)
    monkeypatch.setattr(mdf.yf, "Ticker", boom)
    res = MarketDataFeed.get_piotroski_f_score("ZZZZ")
    assert res["score"] is None and "offline" in res["error"]
    assert "ZZZZ" not in MarketDataFeed._fscore_cache


def test_filters_no_longer_default_to_passing_values():
    text = (PKG_DIR / "auto_trader.py").read_text(encoding="utf-8")
    assert 'cand.get("rs_rating", 80)' not in text
    assert 'cand.get("piotroski_f_score", 7)' not in text
