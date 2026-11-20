# -*- coding: utf-8 -*-
"""
LLM の消費を抑える 2 つの変更のテスト（オフライン）。

1. CRO の拒否権が先に確定した銘柄は、総括を LLM に書かせない
2. 投資家役のプロンプトから、重複と使われない項目を外す
"""
import copy
import json

import pytest

from core.ai_hedge_fund.agents.base_agent import BaseInvestmentAgent
from core.ai_hedge_fund.agents.buffett_agent import BuffettAgent
from core.ai_hedge_fund.agents.portfolio_manager_agent import PortfolioManagerAgent


def forecast(label):
    return {
        "ticker": "KO", "status": "SUCCESS", "model": label, "expected_return_pct": 1.23, "median_price": 61.0,
        "lower_bound_80": 58.0, "upper_bound_80": 64.0, "downside_risk_prob": 12.0, "upside_breakout_prob": 18.0,
        "confidence_score": 70, "consensus": {"target_mean": 70.0},
        "trajectory": [{"step": i, "median": 60.0 + i, "lower_80": 58.0, "upper_80": 64.0} for i in range(1, 6)],
        "pinned_config": {"checkpoint": "x", "forecast_config": {"max_context": 1024}},
        "normalized_features": {"expected_move": 0.01}, "summary_text": "長い要約" * 30,
    }


def context():
    tfm, chr_ = forecast("tfm"), forecast("chr")
    return {
        "macro_liquidity": {"regime": "NEUTRAL", "summary_text": "中立"},
        "timesfm_forecast": tfm, "chronos_forecast": chr_,
        "dual_quantum": {"consensus_status": "NEUTRAL", "ensemble_expected_return_pct": 1.2,
                         "models": {"google_timesfm": tfm, "amazon_chronos": chr_}},
        "factor_ontology": {"factors": [{"id": "roe", "value": 20.0, "note": "x" * 400}]},
        "insider_trading": {"signal": "NEUTRAL"},
    }


# ---- 文脈の圧縮 ----

def test_compaction_drops_duplicates_and_keeps_the_numbers():
    src = context()
    before = copy.deepcopy(src)
    out = BaseInvestmentAgent.compact_context_for_prompt(src, has_ontology_section=True)
    assert src == before                                   # 元の辞書は変えない（CRO や記録が同じものを使う）
    assert "models" not in out["dual_quantum"] and out["dual_quantum"]["consensus_status"] == "NEUTRAL"
    assert "factor_ontology" not in out
    for key in ("timesfm_forecast", "chronos_forecast"):
        for dropped in ("trajectory", "pinned_config", "normalized_features", "summary_text"):
            assert dropped not in out[key]
        for kept in ("expected_return_pct", "median_price", "lower_bound_80", "upper_bound_80",
                     "downside_risk_prob", "upside_breakout_prob", "confidence_score", "consensus"):
            assert out[key][kept] == before[key][kept]
    assert out["macro_liquidity"] == before["macro_liquidity"] and out["insider_trading"] == before["insider_trading"]


def test_models_are_kept_when_they_are_the_only_copy():
    src = context()
    del src["timesfm_forecast"]
    out = BaseInvestmentAgent.compact_context_for_prompt(src)
    assert "models" in out["dual_quantum"]                 # 重複でないなら外さない


def test_ontology_is_kept_when_no_formatted_section_exists():
    out = BaseInvestmentAgent.compact_context_for_prompt(context(), has_ontology_section=False)
    assert "factor_ontology" in out


STOCK = {"symbol": "KO", "name": "Coca-Cola", "current_price": 60.0, "currency": "USD", "roe": 20.0, "pe_trailing": 22.0}


def test_prompt_is_smaller_and_still_carries_the_forecast_values(monkeypatch):
    monkeypatch.setenv("LUMINA_COMPACT_PROMPT", "1")
    compact = BuffettAgent._build_evaluation_prompt(STOCK, context())
    monkeypatch.setenv("LUMINA_COMPACT_PROMPT", "0")
    full = BuffettAgent._build_evaluation_prompt(STOCK, context())
    assert len(compact) < len(full) * 0.6
    assert '"trajectory"' in full and '"trajectory"' not in compact
    assert '"google_timesfm"' in full and '"google_timesfm"' not in compact
    for needle in ('"expected_return_pct":1.23', '"downside_risk_prob":12.0', '"consensus_status":"NEUTRAL"', '"regime":"NEUTRAL"'):
        assert needle in compact
    assert "<untrusted_external_market_data>" in compact and "</untrusted_external_market_data>" in compact


def test_compact_context_is_valid_json(monkeypatch):
    monkeypatch.setenv("LUMINA_COMPACT_PROMPT", "1")
    prompt = BuffettAgent._build_evaluation_prompt(STOCK, context())
    start = prompt.index("【市場マクロ・周辺インテル】") + len("【市場マクロ・周辺インテル】")
    end = prompt.index("</untrusted_external_market_data>")
    assert isinstance(json.loads(prompt[start:end].strip()), dict)


# ---- 拒否権が先に確定した銘柄 ----

def test_thesis_is_not_written_by_the_llm_after_a_veto_precheck(monkeypatch):
    def must_not_be_called(*a, **k):
        raise AssertionError("拒否が確定した銘柄で LLM の設定を読みに行った")

    from core.ai_hedge_fund.agents import portfolio_manager_agent as pma
    monkeypatch.setattr(pma, "load_app_settings", must_not_be_called)
    risk = {"name": "リスク管理者 (CRO)", "score": 15, "dialogue": "VETO", "veto_triggered": True,
            "veto_reason": "オプションGamma Flip割れ＋急変ボラティリティ急拡大による暴落警戒", "llm_skipped_by_veto": True}
    text = PortfolioManagerAgent._generate_thesis_summary(STOCK, [], risk, "拒否", 20)
    assert "CRO拒否権を先行判定" in text and "Gamma Flip" in text and "LLMを呼ばずに省略" in text


def test_council_source_runs_the_precheck_before_the_agents():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "core" / "ai_hedge_fund" / "hedge_fund_council.py").read_text(encoding="utf-8")
    pre = src.index('RiskManagerAgent.evaluate_risk(stock_data, [], context_data)')
    agents = src.index("executor.submit(agent_cls.evaluate, stock_data, context_data)")
    debate = src.index("cls._run_cross_debate(stock_data, analyst_results, context_data)")
    assert pre < agents < debate
    assert 'os.getenv("LUMINA_VETO_PRECHECK", "1") != "0"' in src
