# -*- coding: utf-8 -*-
"""
Issue #7: TimesFM / Chronos を使っていないとき、近似式の結果を
「Google TimesFM 予測」「Google × Amazon の合意」と表示しないことの回帰テスト。オフライン。
"""
import pathlib

import numpy as np
import pandas as pd
import pytest

from core.ai_hedge_fund import chronos_forecaster as cf
from core.ai_hedge_fund import dual_quantum_forecaster as dq
from core.ai_hedge_fund import timesfm_forecaster as tf
from core.ai_hedge_fund.chronos_forecaster import ChronosForecaster
from core.ai_hedge_fund.dual_quantum_forecaster import DualQuantumForecaster
from core.ai_hedge_fund.forecast_tracker import ForecastTracker
from core.ai_hedge_fund.timesfm_forecaster import TimesFMForecaster

PKG_DIR = pathlib.Path(__file__).resolve().parents[1] / "core" / "ai_hedge_fund"


def _history(n=130, drift=0.004, seed=5):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    idx = pd.bdate_range("2026-01-05", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": np.full(n, 1_000_000.0)}, index=idx)


class _FakeTicker:
    def __init__(self, df):
        self._df = df
        self.info = {}

    def history(self, *a, **k):
        return self._df


@pytest.fixture
def offline(monkeypatch):
    df = _history()
    monkeypatch.setattr(tf.yf, "Ticker", lambda sym: _FakeTicker(df))
    monkeypatch.setattr(cf.yf, "Ticker", lambda sym: _FakeTicker(df))
    monkeypatch.setattr(tf.yf, "download", lambda *a, **k: df, raising=False)
    monkeypatch.setattr(cf.yf, "download", lambda *a, **k: df, raising=False)
    monkeypatch.setattr(TimesFMForecaster, "get_openbb_consensus", classmethod(lambda cls, t: {}))
    monkeypatch.setattr(ChronosForecaster, "get_pipeline", classmethod(lambda cls: None))
    monkeypatch.delenv("LUMINA_USE_NATIVE_TIMESFM", raising=False)
    for attr in ("_cache", "_forecast_cache"):
        for klass in (TimesFMForecaster, ChronosForecaster):
            if isinstance(getattr(klass, attr, None), dict):
                getattr(klass, attr).clear()
    return df


def test_timesfm_fallback_is_labelled_as_approximation(offline):
    res = TimesFMForecaster.forecast_stock("TEST", horizon_days=5)
    assert res["status"] == "SUCCESS"
    assert res["backend"] == TimesFMForecaster.FALLBACK_BACKEND
    assert res["is_foundation_model"] is False
    assert "Google TimesFM" not in res["summary_text"]
    assert "TimesFM 未使用" in res["summary_text"]
    assert res["model_label"] == TimesFMForecaster.FALLBACK_LABEL


def test_timesfm_native_backend_keeps_the_timesfm_label(offline, monkeypatch):
    traj = [{"step": i, "date": "2026-07-0%d" % i, "median": 101.0 + i, "lower_80": 95.0, "upper_80": 110.0} for i in range(1, 6)]
    monkeypatch.setattr(TimesFMForecaster, "_infer_native_timesfm_trajectory", classmethod(lambda cls, **k: traj))
    res = TimesFMForecaster.forecast_stock("TEST", horizon_days=5)
    assert res["backend"] == TimesFMForecaster.TIMESFM_HF_REPO
    assert res["is_foundation_model"] is True
    assert "【Google TimesFM 5日先予測】" in res["summary_text"]


def test_chronos_fallback_is_labelled_as_approximation(offline):
    res = ChronosForecaster.forecast_stock("TEST", horizon_days=5)
    assert res["status"] == "SUCCESS"
    assert res["backend"] == ChronosForecaster.FALLBACK_BACKEND
    assert res["is_foundation_model"] is False
    assert "Amazon Chronos" not in res["summary_text"] and "Amazon Chronos" not in res["model"]
    assert "Chronos 未使用" in res["summary_text"]


def _dual(monkeypatch, tfm_native, chr_native, tfm_ret=3.0, chr_ret=3.0):
    def res(native, ret, label, backend):
        return {"expected_return_pct": ret, "is_foundation_model": native, "model_label": label, "backend": backend,
                "current_price": 100.0, "confidence_score": 70, "lower_bound_80": 95.0, "upper_bound_80": 106.0}

    t = res(tfm_native, tfm_ret, "Google TimesFM" if tfm_native else TimesFMForecaster.FALLBACK_LABEL,
            "google/timesfm" if tfm_native else TimesFMForecaster.FALLBACK_BACKEND)
    c = res(chr_native, chr_ret, "Amazon Chronos" if chr_native else ChronosForecaster.FALLBACK_LABEL,
            ChronosForecaster.NATIVE_BACKEND if chr_native else ChronosForecaster.FALLBACK_BACKEND)
    monkeypatch.setattr(dq.TimesFMForecaster, "forecast_stock", classmethod(lambda cls, *a, **k: t))
    monkeypatch.setattr(dq.ChronosForecaster, "forecast_stock", classmethod(lambda cls, *a, **k: c))
    for attr in ("_cache",):
        if isinstance(getattr(DualQuantumForecaster, attr, None), dict):
            getattr(DualQuantumForecaster, attr).clear()
    return DualQuantumForecaster.forecast("TEST")


def _dual_entry():
    names = [n for n in dir(DualQuantumForecaster) if not n.startswith("_") and callable(getattr(DualQuantumForecaster, n))]
    return names


@pytest.mark.parametrize("tfm_ret, chr_ret, status", [
    (3.0, 3.0, "DOUBLE_BULLISH"), (-3.0, -3.0, "DOUBLE_BEARISH"), (3.0, -3.0, "DISAGREEMENT"), (0.2, 0.1, "NEUTRAL"),
])
def test_dual_consensus_text_without_foundation_models(monkeypatch, tfm_ret, chr_ret, status):
    out = _dual(monkeypatch, False, False, tfm_ret, chr_ret)
    assert out is not None, f"DualQuantumForecaster の入口が見つからない: {_dual_entry()}"
    assert out["consensus_status"] == status
    assert out["foundation_models_used"] == []
    text = out["summary_desc"] + out["consensus_badge"] + out["ensemble_label"]
    for claim in ("Google TimesFM と Amazon Chronos", "Google×Amazon", "Google (", "Amazon (", "基盤モデルが"):
        assert claim not in text
    assert "基盤モデルは使っていない" in out["summary_desc"]
    assert out["backends"]["timesfm"] == TimesFMForecaster.FALLBACK_BACKEND


def test_dual_consensus_text_with_both_foundation_models(monkeypatch):
    out = _dual(monkeypatch, True, True)
    assert out["foundation_models_used"] == ["timesfm", "chronos"]
    assert out["ensemble_label"] == "Google TimesFM × Amazon Chronos"
    assert "双方が上昇予測で完全合意" in out["summary_desc"]


def test_dual_consensus_text_with_one_foundation_model(monkeypatch):
    out = _dual(monkeypatch, True, False)
    assert out["foundation_models_used"] == ["timesfm"]
    assert "Chronos 未使用" in out["ensemble_label"]
    assert "片方は基盤モデルではなく近似式" in out["summary_desc"]


def test_council_prompts_do_not_hardcode_the_model_names():
    text = (PKG_DIR / "langgraph_council.py").read_text(encoding="utf-8")
    assert "(Google TimesFM × Amazon Chronos)】" not in text
    assert text.count("quantum.get('ensemble_label'") == 3


def test_track_record_text_follows_recorded_backend(monkeypatch):
    def summary(items):
        monkeypatch.setattr(ForecastTracker, "load_history", classmethod(lambda cls: items))
        return ForecastTracker.get_track_record_summary()["note_markdown"]

    base = {"status": "ACCURATE", "is_accurate": True, "error_pct": 1.0, "actual_return_pct": 2.0,
            "ticker": "AAA", "base_price": 10.0, "actual_price": 10.2}
    assert "TimesFM は使っていません" in summary([dict(base, is_foundation_model=False), dict(base)])
    assert "Google開発の時系列基盤モデル（TimesFM）による予測" in summary([dict(base, is_foundation_model=True)])
    mixed = summary([dict(base, is_foundation_model=True), dict(base, is_foundation_model=False)])
    assert "2 件のうち TimesFM によるものは 1 件" in mixed
