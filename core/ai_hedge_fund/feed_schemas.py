# -*- coding: utf-8 -*-
"""
フィードの戻り値と、合議に渡す context_data の形（Issue #6）。

値を作る側（フィード）と読む側（エージェント）でキー名が食い違い、読む側が
既定値で動いていた。ここに形を 1 か所で定義し、両側とテストがこれを参照する。

- 判定に使うのは区分値（regime、is_turbulence）。表示用の文言（summary_text）を
  部分一致で判定に使わない。
"""
from typing import Any, Dict, Iterable, List, Optional, TypedDict

# MacroLiquidityFeed.get_macro_status()["regime"] が取りうる値
MACRO_REGIME_RISK_ON = "RISK_ON"
MACRO_REGIME_NEUTRAL = "NEUTRAL"
MACRO_REGIME_RISK_OFF = "RISK_OFF"
MACRO_REGIMES = (MACRO_REGIME_RISK_ON, MACRO_REGIME_NEUTRAL, MACRO_REGIME_RISK_OFF)


class MacroLiquidityStatus(TypedDict, total=False):
    """MacroLiquidityFeed.get_macro_status() の戻り値。"""
    available: bool          # False = 取得に失敗し、フェイルセーフの値を返している
    regime: str              # MACRO_REGIMES のどれか
    regime_desc: str
    net_liquidity_t: float
    prev_net_liquidity_t: float
    chg_t: float
    chg_pct: float
    emoji: str
    walcl_t: float
    tga_b: float
    rrp_b: float
    summary_text: str
    updated_at: str
    error: str               # 取得に失敗したときだけ


MACRO_REQUIRED_KEYS = ("available", "regime", "regime_desc", "chg_pct", "summary_text")


class MarketTurbulence(TypedDict, total=False):
    """OptionsGexFeed.get_market_turbulence_warning() の戻り値。"""
    is_turbulence: bool
    size_multiplier: float
    spy_regime: Optional[str]
    qqq_regime: Optional[str]
    summary_text: str
    spy_profile: Dict[str, Any]
    qqq_profile: Dict[str, Any]


TURBULENCE_REQUIRED_KEYS = ("is_turbulence", "size_multiplier", "summary_text")


class CouncilGexContext(TypedDict, total=False):
    """context_data["options_gex"]。"""
    is_turbulence: bool      # 判定に使う
    size_multiplier: float
    market_turbulence: str   # 表示用の文言
    ticker_gex: str          # 表示用の文言


def build_gex_context(market_turbulence: Dict[str, Any], gex_profile: Optional[Dict[str, Any]] = None) -> CouncilGexContext:
    """フィードの戻り値から context_data["options_gex"] を作る。"""
    return {
        "is_turbulence": bool(market_turbulence.get("is_turbulence", False)),
        "size_multiplier": float(market_turbulence.get("size_multiplier", 1.0)),
        "market_turbulence": market_turbulence.get("summary_text"),
        "ticker_gex": (gex_profile or {}).get("summary_text"),
    }


# 合議（HedgeFundCouncil.conduct_council）がエージェントに渡す context_data のキー。
# エージェントが context_data から読むキーは、この中に無ければならない。
COUNCIL_CONTEXT_KEYS = (
    "macro_liquidity",
    "insider_trading",
    "options_gex",
    "openbb_bundle",
    "qlib_alpha158",
    "lead_lag_causality",
    "factor_ontology",
    "timesfm_forecast",
    "chronos_forecast",
    "dual_quantum",
    "news_intelligence",
    "recent_news",
    "broker_engine",
    "temporal_graph_lessons",
    "deterministic_metrics",
    "master_mandates_prompt",
)

# 合議が作らず、呼び出し元や途中の処理が足すキー（あれば読む）
OPTIONAL_CONTEXT_KEYS = ("lang", "deep_ofi", "market_regime", "moe_weights")


def missing_keys(data: Dict[str, Any], required: Iterable[str]) -> List[str]:
    return [k for k in required if k not in data]
