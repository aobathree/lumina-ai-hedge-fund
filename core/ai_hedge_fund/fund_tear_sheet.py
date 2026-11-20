# -*- coding: utf-8 -*-
"""
Lumina AI Hedge Fund: Institutional Performance Tear Sheet Engine (pyfolio Inspired)
Generates Wall Street grade performance and risk analytics from trading execution logs.
Produces monthly return heatmaps, underwater drawdown plots, and rolling Sharpe metrics.
"""

import os
import sys
import json
import time
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from core.ai_hedge_fund.broker_client import PaperTradingEngine


class FundTearSheet:
    """
    Institutional Performance & Risk Tear Sheet Generator (pyfolio architecture).
    Evaluates real trading journey health and exports publication-ready graphics for note.com.
    """

    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
    TEAR_SHEET_PNG = os.path.join(OUTPUT_DIR, "lumina_performance_tearsheet.png")

    # 年率の指標（ボラティリティ、シャープなど）を出すのに要る日次リターンの数
    MIN_DAILY_RETURNS_FOR_RATIOS = 20

    @staticmethod
    def _load_equity_series(broker) -> pd.Series:
        """台帳の評価額の記録（equity_snapshots）を時系列で返す。取れなければ空。"""
        try:
            rows = broker.get_equity_curve(limit=5000) or []
            df = pd.DataFrame(rows)
            if df.empty or "total_equity_jpy" not in df.columns or "timestamp" not in df.columns:
                return pd.Series(dtype=float)
            df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["eq"] = pd.to_numeric(df["total_equity_jpy"], errors="coerce")
            df = df.dropna(subset=["ts", "eq"]).sort_values("ts")
            df = df[df["eq"] > 0]
            return pd.Series(df["eq"].values, index=pd.DatetimeIndex(df["ts"].values))
        except Exception as e:
            print(f"[FundTearSheet] Equity curve lookup warning: {e}")
            return pd.Series(dtype=float)

    @classmethod
    def generate_tear_sheet(cls, broker: Optional[PaperTradingEngine] = None, save_image: bool = True) -> Dict[str, Any]:
        """
        Calculates comprehensive fund metrics and renders the institutional tear sheet chart.
        """
        if not broker:
            broker = PaperTradingEngine()

        summary = broker.get_portfolio_summary()
        total_value = float(summary.get("total_equity_usd", summary.get("total_portfolio_value_usd", 0.0)) or 0.0)

        # --- 実際の記録だけから計算する。記録が足りない項目は None（表示は N/A） ---
        equity = cls._load_equity_series(broker)
        daily = equity.resample("D").last().ffill() if len(equity) >= 2 else equity
        rets = daily.pct_change().dropna() if len(daily) >= 2 else pd.Series(dtype=float)

        cumulative_return_pct = None
        max_dd_pct = None
        if len(equity) >= 2 and float(equity.iloc[0]) > 0:
            cumulative_return_pct = round((float(equity.iloc[-1]) / float(equity.iloc[0]) - 1.0) * 100.0, 2)
            max_dd_pct = round(float(((equity - equity.cummax()) / equity.cummax()).min() * 100.0), 2)

        annualized_vol_pct = sharpe_ratio = sortino_ratio = calmar_ratio = None
        if len(rets) >= cls.MIN_DAILY_RETURNS_FOR_RATIOS:
            std = float(rets.std())
            mean = float(rets.mean())
            annualized_vol_pct = round(std * math.sqrt(252) * 100.0, 2)
            if std > 0:
                sharpe_ratio = round(mean / std * math.sqrt(252), 2)
            downside = rets[rets < 0]
            if len(downside) >= 2 and float(downside.std()) > 0:
                sortino_ratio = round(mean / float(downside.std()) * math.sqrt(252), 2)
            if max_dd_pct is not None and max_dd_pct < 0:
                ann_ret_pct = ((1.0 + mean) ** 252 - 1.0) * 100.0
                calmar_ratio = round(ann_ret_pct / abs(max_dd_pct), 2)

        from core.ai_hedge_fund.quant_reporter import closed_trades_from_engine
        try:
            trades = closed_trades_from_engine(broker)
        except Exception as e:
            print(f"[FundTearSheet] Closed-trade lookup warning: {e}")
            trades = []
        pnls = [float(t.get("realized_pnl") or 0.0) for t in trades]
        total_trades = len(pnls)
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        win_rate = round(len(wins) / total_trades * 100.0, 1) if total_trades > 0 else None
        # PF は損失が 0 だと定義できない。通貨が混ざる場合も単純合算になる点に注意（銘柄の通貨建て）。
        profit_factor = round(sum(wins) / abs(sum(losses)), 2) if losses else None

        kpis = {
            "total_portfolio_value_usd": round(total_value, 2),
            "initial_equity_jpy": round(float(equity.iloc[0]), 0) if len(equity) >= 1 else None,
            "latest_equity_jpy": round(float(equity.iloc[-1]), 0) if len(equity) >= 1 else None,
            "equity_snapshots_count": int(len(equity)),
            "daily_returns_count": int(len(rets)),
            "cumulative_return_pct": cumulative_return_pct,
            "annualized_volatility_pct": annualized_vol_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown_pct": max_dd_pct,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "total_trades_count": total_trades,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if save_image:
            try:
                cls._render_matplotlib_tearsheet(kpis, equity)
            except Exception as e:
                print(f"[FundTearSheet] Image rendering warning: {e}")

        return kpis

    @classmethod
    def _render_matplotlib_tearsheet(cls, kpis: Dict[str, Any], equity: Optional[pd.Series] = None):
        """Render high-resolution 3-panel tear sheet chart (Dark Cyberpunk Theme)"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

        fig = plt.figure(figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor("#030712")  # Deep cyber dark

        # Grid setup: 2 rows (Top: Cumulative Return & Underwater Drawdown, Bottom: Monthly Table / KPIs)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

        # 1. Panel 1: Cumulative Equity Curve (Simulated Chronological Curve)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor("#0b0f19")
        
        # 実際に記録された評価額だけを描く。記録が 2 点に満たなければ、曲線を作らずその旨を出す。
        has_curve = equity is not None and len(equity) >= 2
        if has_curve:
            dates = equity.index
            equity_curve = equity.values.astype(float)
            base_val = float(equity_curve[0])
            ax1.plot(dates, equity_curve, color="#00f0ff", linewidth=2.2, label="Lumina Capital NAV (JPY, recorded)")
            ax1.fill_between(dates, base_val, equity_curve, color="#00f0ff", alpha=0.12)
            ax1.axhline(base_val, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.7)
        else:
            ax1.text(0.5, 0.5, "NO TRACK RECORD YET\n(fewer than 2 recorded equity snapshots)",
                     transform=ax1.transAxes, ha="center", va="center", color="#94a3b8", fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
        ax1.set_title("LUMINA CAPITAL — CUMULATIVE PORTFOLIO EQUITY (NAV)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax1.tick_params(colors="#94a3b8", labelsize=9)
        if has_curve:
            ax1.yaxis.set_major_formatter("¥{x:,.0f}")
        for spine in ax1.spines.values():
            spine.set_color("#1e293b")
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        if has_curve:
            ax1.legend(loc="upper left", facecolor="#0b0f19", edgecolor="#1e293b", labelcolor="#f8fafc", fontsize=9)

        # 2. Panel 2: Underwater Drawdown Plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor("#0b0f19")

        if has_curve:
            peaks = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peaks) / peaks * 100.0
            ax2.plot(dates, drawdown, color="#ff0055", linewidth=1.8, label="Underwater DD (%)")
            ax2.fill_between(dates, 0, drawdown, color="#ff0055", alpha=0.25)
        else:
            ax2.text(0.5, 0.5, "N/A", transform=ax2.transAxes, ha="center", va="center", color="#94a3b8", fontsize=12)
            ax2.set_xticks([])
            ax2.set_yticks([])
        ax2.set_title("UNDERWATER DRAWDOWN PROFILE (MAX DD %)", color="#f8fafc", fontsize=10, fontweight="bold", pad=10)
        ax2.tick_params(colors="#94a3b8", labelsize=8)
        if has_curve:
            ax2.yaxis.set_major_formatter("{x:.1f}%")
        for spine in ax2.spines.values():
            spine.set_color("#1e293b")
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        if has_curve:
            ax2.legend(loc="lower left", facecolor="#0b0f19", edgecolor="#1e293b", labelcolor="#f8fafc", fontsize=8)

        # 3. Panel 3: Key Performance Diagnostic Table (KPIs)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor("#0b0f19")
        ax3.axis("off")

        def _fmt(key: str, spec: str, suffix: str = "") -> str:
            val = kpis.get(key)
            return "N/A" if val is None else f"{format(val, spec)}{suffix}"

        kpi_rows = [
            ["Cumulative Return", _fmt("cumulative_return_pct", "+.2f", "%")],
            ["Annualized Volatility", _fmt("annualized_volatility_pct", ".1f", "%")],
            ["Sharpe Ratio", _fmt("sharpe_ratio", ".2f")],
            ["Sortino Ratio", _fmt("sortino_ratio", ".2f")],
            ["Calmar Ratio", _fmt("calmar_ratio", ".2f")],
            ["Max Drawdown", _fmt("max_drawdown_pct", ".1f", "%")],
            ["Win Rate", _fmt("win_rate_pct", ".1f", "%")],
            ["Profit Factor", _fmt("profit_factor", ".2f")],
            ["Closed Trades", _fmt("total_trades_count", "d")],
        ]

        table = ax3.table(
            cellText=kpi_rows,
            colLabels=["Diagnostic Metric", "Fund Score"],
            loc="center",
            cellLoc="left"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#1e293b")
            if row == 0:
                cell.set_facecolor("#0f172a")
                cell.set_text_props(color="#38bdf8", fontweight="bold")
            else:
                cell.set_facecolor("#0b0f19")
                cell.set_text_props(color="#f8fafc" if col == 1 else "#94a3b8")

        plt.savefig(cls.TEAR_SHEET_PNG, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"[FundTearSheet] Institutional Tear Sheet exported to: {cls.TEAR_SHEET_PNG}")
