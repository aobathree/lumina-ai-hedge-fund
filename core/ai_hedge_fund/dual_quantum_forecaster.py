# -*- coding: utf-8 -*-
"""
dual_quantum_forecaster.py - Google TimesFM × Amazon Chronos 二刀流アンサンブル確率予測エンジン
世界最高峰の2大時系列基盤モデル（Google TimesFM ＆ Amazon Chronos）を同時に実行し、
両モデルの予測確率分布を重ね合わせることで「ダマシ（False Signals）」を排除するコンセンサス判定器。
"""

from typing import Dict, Any, List, Optional
from core.ai_hedge_fund.timesfm_forecaster import TimesFMForecaster
from core.ai_hedge_fund.chronos_forecaster import ChronosForecaster


class DualQuantumForecaster:
    """
    Google TimesFM × Amazon Chronos 統合数理アンサンブル
    """

    @classmethod
    def forecast(cls, ticker: str, horizon_days: int = 5) -> Dict[str, Any]:
        """
        Google TimesFM と Amazon Chronos の両方を実行し、合意シグナルを生成
        """
        # 1. 2大モデルの並行/順次推論
        timesfm_res = TimesFMForecaster.forecast_stock(ticker, horizon_days=horizon_days)
        chronos_res = ChronosForecaster.forecast_stock(ticker, horizon_days=horizon_days)

        tfm_ret = timesfm_res.get("expected_return_pct", 0.0)
        chr_ret = chronos_res.get("expected_return_pct", 0.0)

        tfm_down = timesfm_res.get("downside_risk_prob", 15.0)
        chr_down = chronos_res.get("downside_risk_prob", 15.0)

        tfm_up = timesfm_res.get("upside_breakout_prob", 15.0)
        chr_up = chronos_res.get("upside_breakout_prob", 15.0)

        # 2. コンセンサス状態の数学的判定
        # 閾値: 上昇+1.5%以上, 下落-1.5%以下
        both_bull = (tfm_ret >= 1.5) and (chr_ret >= 1.5)
        both_bear = (tfm_ret <= -1.5) and (chr_ret <= -1.5)
        conflict = (tfm_ret * chr_ret < -1.0) # 符号が逆で乖離大

        # 実際に動いたバックエンドを見て、文言を決める。
        # 近似式（どちらも直近のモメンタムが主な入力）の結果を、基盤モデルの予測・合意と書かない。
        tfm_native = bool(timesfm_res.get("is_foundation_model", False))
        chr_native = bool(chronos_res.get("is_foundation_model", False))
        tfm_label = timesfm_res.get("model_label") or ("Google TimesFM" if tfm_native else "モメンタム近似（TimesFM 未使用）")
        chr_label = chronos_res.get("model_label") or ("Amazon Chronos" if chr_native else "モメンタム近似（Chronos 未使用）")
        both_native = tfm_native and chr_native
        if both_native:
            ensemble_label = "Google TimesFM × Amazon Chronos"
            caveat = ""
        elif tfm_native or chr_native:
            ensemble_label = f"{tfm_label} × {chr_label}"
            caveat = " ※片方は基盤モデルではなく近似式。"
        else:
            ensemble_label = "モメンタム近似 2 本（基盤モデル未使用）"
            caveat = " ※基盤モデルは使っていない。2 本とも直近モメンタムの近似式で、独立した 2 モデルの合意ではない。"

        if both_bull:
            consensus_status = "DOUBLE_BULLISH" # 🚀 両者揃って強気一致
            if both_native:
                consensus_badge = "🚀 DOUBLE BULLISH (Google×Amazon 確率合意)"
                summary_desc = "Google TimesFM と Amazon Chronos の双方が上昇予測で完全合意。最高ランクの確信シグナル。"
            else:
                consensus_badge = f"🚀 DOUBLE BULLISH ({ensemble_label})"
                summary_desc = f"{tfm_label} と {chr_label} がともに上昇を示した。{caveat.strip()}"
        elif both_bear:
            consensus_status = "DOUBLE_BEARISH" # ⚠️ 両者揃って急落警戒一致
            if both_native:
                consensus_badge = "⚠️ DOUBLE BEARISH (急落警戒合意)"
                summary_desc = "両時系列基盤モデルが下落リスクで一致。買いエントリーの厳格見送りを推奨。"
            else:
                consensus_badge = f"⚠️ DOUBLE BEARISH ({ensemble_label})"
                summary_desc = f"{tfm_label} と {chr_label} がともに下落を示した。{caveat.strip()}"
        elif conflict:
            consensus_status = "DISAGREEMENT" # ⚔️ モデル間で予測乖離・対立
            consensus_badge = "⚔️ MODEL CONFLICT (数理予測の対立)"
            if both_native:
                summary_desc = f"Google ({tfm_ret:+.2f}%) と Amazon ({chr_ret:+.2f}%) で方向性が乖離。ボラティリティ急変に警戒。"
            else:
                summary_desc = f"{tfm_label} ({tfm_ret:+.2f}%) と {chr_label} ({chr_ret:+.2f}%) で方向性が乖離。{caveat.strip()}"
        else:
            consensus_status = "NEUTRAL"
            consensus_badge = "⚖️ QUANTUM NEUTRAL (レンジ推移)"
            summary_desc = "両モデルともに小幅レンジ内の推移を予測。トレンド発生待ち。" if both_native else f"2 本とも小幅レンジ内の推移を示した。{caveat.strip()}"

        # 3. アンサンブル合成値（加重平均）
        ensemble_expected_ret = round((tfm_ret * 0.5 + chr_ret * 0.5), 2)
        curr_price = timesfm_res.get("current_price") or chronos_res.get("current_price", 0.0)
        ensemble_median_price = round(curr_price * (1.0 + ensemble_expected_ret / 100.0), 2)
        
        # 信頼区間の重ね合わせ（より安全な最外殻または平均帯）
        ensemble_lower_80 = round(min(timesfm_res.get("lower_bound_80", curr_price), chronos_res.get("lower_bound_80", curr_price)), 2)
        ensemble_upper_80 = round(max(timesfm_res.get("upper_bound_80", curr_price), chronos_res.get("upper_bound_80", curr_price)), 2)

        # 総合確信度スコア (不一致時はペナルティ)
        base_conf = (timesfm_res.get("confidence_score", 50) + chronos_res.get("confidence_score", 50)) / 2.0
        if consensus_status == "DOUBLE_BULLISH" or consensus_status == "DOUBLE_BEARISH":
            final_conf = min(98, int(base_conf + 12))
        elif consensus_status == "DISAGREEMENT":
            final_conf = max(35, int(base_conf - 20))
        else:
            final_conf = int(base_conf)

        return {
            "ticker": ticker,
            "horizon_days": horizon_days,
            "target_date": timesfm_res.get("target_date"),
            "current_price": curr_price,
            "consensus_status": consensus_status,
            "consensus_badge": consensus_badge,
            "confidence_score": final_conf,
            "ensemble_expected_return_pct": ensemble_expected_ret,
            "ensemble_median_price": ensemble_median_price,
            "ensemble_lower_80": ensemble_lower_80,
            "ensemble_upper_80": ensemble_upper_80,
            "avg_downside_risk_prob": round((tfm_down + chr_down) / 2.0, 1),
            "avg_upside_breakout_prob": round((tfm_up + chr_up) / 2.0, 1),
            "summary_desc": summary_desc,
            "ensemble_label": ensemble_label,
            "foundation_models_used": [n for n, used in (("timesfm", tfm_native), ("chronos", chr_native)) if used],
            "backends": {"timesfm": timesfm_res.get("backend"), "chronos": chronos_res.get("backend")},
            "models": {
                "google_timesfm": timesfm_res,
                "amazon_chronos": chronos_res
            }
        }
