# -*- coding: utf-8 -*-
"""
Lumina Capital - QuantStats Performance & Risk Reporter (v4.2.0)
Calculates institutional hedge fund analytics: Sharpe, Sortino, Calmar, MDD,
and generates professional monthly returns heatmaps for Streamlit, note, and X.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)


EXIT_TRADE_TYPES_EXCLUDED = ("BUY_ENTRY",)


def closed_trades_from_engine(broker_engine, limit: int = 1000) -> list:
    """
    台帳（trade_memories）から、決済した取引の記録を返す。
    買いの記録（BUY_ENTRY）は除く。realized_pnl は銘柄の通貨建て。
    取れなければ空のリスト。
    """
    getter = getattr(broker_engine, "get_trade_memories", None)
    if getter is None:
        return []
    rows = getter(limit=limit) or []
    out = []
    for r in rows:
        if str(r.get("trade_type", "")).upper() in EXIT_TRADE_TYPES_EXCLUDED:
            continue
        if r.get("realized_pnl") is None:
            continue
        out.append(r)
    return out


def _is_newest_first(eq_data) -> bool:
    """get_equity_curve の並びが新しい順かどうか（timestamp で判定。判定できなければ False）。"""
    try:
        return str(eq_data[0].get("timestamp")) > str(eq_data[-1].get("timestamp"))
    except Exception:
        return False


class QuantReporter:
    @staticmethod
    def get_returns_series_from_engine(broker_engine) -> pd.Series:
        """
        Extract daily returns series from PaperTradingEngine equity history.
        Does NOT generate synthetic normal distributions.
        """
        try:
            eq_data = broker_engine.get_equity_curve(limit=500)
            if eq_data and len(eq_data) >= 2:
                df = pd.DataFrame(eq_data)
                df['date'] = pd.to_datetime(df['timestamp'])
                df['equity'] = pd.to_numeric(df['total_equity_jpy'], errors='coerce')
                df = df.dropna(subset=['equity']).sort_values('date')
                
                # Daily resample
                df = df.set_index('date')
                daily_eq = df['equity'].resample('D').last().ffill()
                rets = daily_eq.pct_change().dropna()
                if len(rets) >= 1:
                    return rets
        except Exception as e:
            logger.warning(f"Failed to get returns from equity curve: {e}")

        # Return actual short or empty series rather than fake normal distribution
        return pd.Series(dtype=float)

    @classmethod
    def compute_metrics(cls, rets: pd.Series, broker_engine=None) -> Dict[str, Any]:
        """
        Compute institutional risk and return metrics using QuantStats if >= 15 days,
        otherwise compute 100% real trade-based warmup statistics.
        """
        # 1. Gather real trading facts from engine if provided
        days_active = len(rets) + 1 if (rets is not None and len(rets) > 0) else None
        realized_pnl_by_ccy: Dict[str, float] = {}
        win_trades = 0
        loss_trades = 0
        closed_trades = 0
        mdd: Optional[float] = None

        if broker_engine:
            try:
                eq_data = broker_engine.get_equity_curve(limit=500)
                equities = [float(x.get("total_equity_jpy", 0)) for x in (eq_data or []) if x.get("total_equity_jpy")]
                if len(equities) >= 2:
                    # 時系列順のピークからの最大下落率（単純な max / min ではない）
                    eq = pd.Series(equities[::-1] if _is_newest_first(eq_data) else equities)
                    mdd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100.0)

                # 勝敗と確定損益は、台帳に記録された realized_pnl から数える（注文メモの文字では判定しない）
                for t in closed_trades_from_engine(broker_engine):
                    pnl = float(t.get("realized_pnl") or 0.0)
                    closed_trades += 1
                    if pnl > 0:
                        win_trades += 1
                    elif pnl < 0:
                        loss_trades += 1
                    ccy = str(t.get("currency") or "USD")
                    realized_pnl_by_ccy[ccy] = realized_pnl_by_ccy.get(ccy, 0.0) + pnl
            except Exception as e:
                logger.warning(f"Failed to gather broker engine facts: {e}")

        win_rate: Optional[float] = (win_trades / closed_trades * 100.0) if closed_trades > 0 else None

        # 2. If we have at least 15 daily return data points, calculate QuantStats
        if rets is not None and len(rets) >= 15:
            try:
                import quantstats as qs
                sharpe = float(qs.stats.sharpe(rets))
                sortino = float(qs.stats.sortino(rets))
                calmar = float(qs.stats.calmar(rets))
                mdd_calc = float(qs.stats.max_drawdown(rets)) * 100.0
                wr_calc = float(qs.stats.win_rate(rets)) * 100.0
                vol = float(qs.stats.volatility(rets)) * 100.0
                cagr = float(qs.stats.cagr(rets)) * 100.0
                pf = float(qs.stats.profit_factor(rets))
                return {
                    "is_warmup": False,
                    "sharpe": round(sharpe, 2) if not np.isnan(sharpe) else 0.0,
                    "sortino": round(sortino, 2) if not np.isnan(sortino) else 0.0,
                    "calmar": round(calmar, 2) if not np.isnan(calmar) else 0.0,
                    "max_drawdown": round(mdd_calc, 2) if not np.isnan(mdd_calc) else 0.0,
                    "win_rate": round(wr_calc, 1) if not np.isnan(wr_calc) else 0.0,
                    "volatility": round(vol, 1) if not np.isnan(vol) else 0.0,
                    "cagr": round(cagr, 1) if not np.isnan(cagr) else 0.0,
                    "profit_factor": round(pf, 2) if not np.isnan(pf) else 1.0,
                    "summary_line": f"・クオンツ指標: シャープレシオ {round(sharpe, 2)} ｜ ソルティノ {round(sortino, 2)} ｜ 最大DD {round(mdd_calc, 2):.2f}% ｜ 勝率 {round(wr_calc, 1):.1f}% ｜ PF {round(pf, 2)}"
                }
            except Exception as e:
                logger.warning(f"Quantstats computation failed: {e}")

        # 3. Warm-up reporting: 実績が無い項目は「実績なし」と出す（数字を作らない）
        def _money(ccy: str, v: float) -> str:
            sym = "¥" if ccy == "JPY" else ("€" if ccy == "EUR" else "$")
            sign = "+" if v > 0 else ("-" if v < 0 else "±")
            return f"{sign}{sym}{abs(v):,.0f}" if ccy == "JPY" else f"{sign}{sym}{abs(v):,.2f}"

        if closed_trades > 0:
            pnl_str = " / ".join(_money(c, v) for c, v in sorted(realized_pnl_by_ccy.items()))
            wr_str = f"{win_rate:.0f}% ({win_trades}勝{loss_trades}敗)"
        else:
            pnl_str = "実績なし"
            wr_str = "実績なし（決済 0 件）"
        days_str = f"{days_active}営業日目" if days_active is not None else "記録なし"
        mdd_str = f"{mdd:.2f}%" if mdd is not None else "算出不能"
        summary_line = (
            f"・クオンツ運用フェーズ: 実戦稼働 {days_str}（初期資産形成期） ｜ "
            f"確定損益: {pnl_str} ｜ 決済勝率: {wr_str} ｜ 最大DD: {mdd_str}"
        )

        return {
            "is_warmup": True,
            "days_active": days_active,
            "sharpe": "蓄積中",
            "sortino": "蓄積中",
            "calmar": "蓄積中",
            "max_drawdown": round(mdd, 2) if mdd is not None else None,
            "win_rate": round(win_rate, 1) if win_rate is not None else None,
            "volatility": "蓄積中",
            "cagr": "蓄積中",
            "profit_factor": "蓄積中",
            "closed_trades": closed_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "realized_pnl_by_currency": realized_pnl_by_ccy,
            "realized_pnl_usd": realized_pnl_by_ccy.get("USD", 0.0),
            "summary_line": summary_line
        }

    @classmethod
    def compute_monthly_returns_table(cls, rets: pd.Series) -> pd.DataFrame:
        """
        Compute month-by-month return matrix (Year x Month).
        Returns only genuine months that have trading data.
        """
        curr_y = pd.Timestamp.now().year
        curr_m = pd.Timestamp.now().strftime('%b')
        try:
            if rets is not None and not rets.empty and len(rets) >= 2:
                m_rets = rets.resample('ME').apply(lambda r: (1 + r).prod() - 1.0) * 100.0
                df = pd.DataFrame({
                    'Year': m_rets.index.year,
                    'Month': m_rets.index.strftime('%b'),
                    'Return': m_rets.values
                })
                pivot = df.pivot(index='Year', columns='Month', values='Return')
                
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                existing_months = [m for m in month_order if m in pivot.columns]
                pivot = pivot[existing_months]
                
                y_rets = rets.resample('YE').apply(lambda r: (1 + r).prod() - 1.0) * 100.0
                y_map = dict(zip(y_rets.index.year, y_rets.values))
                pivot['通算 (Year)'] = [y_map.get(y, 0.0) for y in pivot.index]
                
                return pivot.sort_index(ascending=False)
        except Exception as e:
            logger.warning(f"Failed to generate monthly table: {e}")

        # Real cumulative return for current month
        # リターンの記録が無いときは NaN（表示側で「実績なし」にする）
        cum_ret = ((1 + rets).prod() - 1.0) * 100.0 if (rets is not None and not rets.empty) else float('nan')
        return pd.DataFrame({curr_m: [round(cum_ret, 2)], '通算 (Year)': [round(cum_ret, 2)]}, index=[curr_y])

    @classmethod
    def generate_monthly_heatmap_image(cls, rets: pd.Series, output_path: str) -> bool:
        """
        Generate high-resolution PNG monthly returns heatmap image for note and X.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            table = cls.compute_monthly_returns_table(rets)
            plot_data = table.drop(columns=['通算 (Year)'], errors='ignore')
            
            fig, ax = plt.subplots(figsize=(10, max(2.5, len(table) * 0.8 + 1.2)), facecolor='#0f172a')
            ax.set_facecolor('#0f172a')
            
            # Color map: coolwarm or RdYlGn
            cmap = sns.diverging_palette(10, 140, as_cmap=True)
            
            sns.heatmap(
                plot_data,
                annot=True,
                fmt=".1f",
                cmap=cmap,
                center=0.0,
                cbar=False,
                linewidths=1.0,
                linecolor='#1e293b',
                ax=ax,
                annot_kws={"size": 11, "weight": "bold", "color": "#ffffff"}
            )
            
            # Styling
            ax.set_title("LUMINA CAPITAL | Monthly Performance Heatmap (%)", fontsize=13, color='#38bdf8', weight='bold', pad=14)
            ax.tick_params(colors='#94a3b8', labelsize=10)
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
            return True
        except Exception as e:
            logger.warning(f"Failed to generate monthly heatmap image: {e}")
            return False

    @classmethod
    def render_in_streamlit(cls, broker_engine):
        """
        Full Streamlit rendering component for Tab 2: KPI summary + Monthly Returns Heatmap.
        """
        rets = cls.get_returns_series_from_engine(broker_engine)
        metrics = cls.compute_metrics(rets, broker_engine=broker_engine)
        
        # 1. Institutional KPI Cards
        st.markdown("""
        <div style="font-size: 13px; font-weight: 700; color: #38bdf8; margin: 12px 0 8px 0; display: flex; align-items: center; gap: 8px;">
            <span>🏛️ 機関投資家クオンツ・パフォーマンス指標 (QuantStats Engine)</span>
            <span style="font-size: 10px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 2px 8px; border-radius: 6px;">ヘッジファンド水準監査</span>
        </div>
        """, unsafe_allow_html=True)
        
        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        qc1.metric("シャープレシオ (Sharpe)", f"{metrics['sharpe']}", help="リスク（価格変動）に対してどれだけ効率よく利益を出せているか。2.0超で超一流ファンド水準。")
        qc2.metric("ソルティノレシオ (Sortino)", f"{metrics['sortino']}", help="下落リスク（ダウンサイド）のみを考慮した真の運用効率。")
        qc3.metric("Calmar レシオ", f"{metrics['calmar']}", help="年間成長率(CAGR) ÷ 最大ドローダウン。大きいほど暴落耐性が高い。")
        qc4.metric("最大ドローダウン (MDD)", (f"{metrics['max_drawdown']:.2f}%" if metrics.get('max_drawdown') is not None else "算出不能"), help="資産ピークから最大で何％落ち込んだか。マイナスが浅いほど優秀。")
        qc5.metric("プロフィットファクター (PF)", f"{metrics['profit_factor']}", help="総利益 ÷ 総損失。1.5以上でヘッジファンド級。")

        # 2. Monthly Returns Heatmap Table
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        table = cls.compute_monthly_returns_table(rets)
        
        with st.expander("📅 【月次パフォーマンス・ヒートマップ】年・月別リターン一覧 (%)", expanded=True):
            def style_heatmap(val):
                if pd.isna(val):
                    return "background-color: #1e293b; color: #64748b;"
                if val > 0:
                    intensity = min(1.0, val / 10.0)
                    return f"background-color: rgba(34, 197, 94, {0.15 + intensity * 0.45}); color: #4ade80; font-weight: bold;"
                elif val < 0:
                    intensity = min(1.0, abs(val) / 10.0)
                    return f"background-color: rgba(239, 68, 68, {0.15 + intensity * 0.45}); color: #f87171; font-weight: bold;"
                else:
                    return "background-color: #1e293b; color: #cbd5e1;"

            styled_df = table.style.format("{:+.1f}%", na_rep="実績なし").map(style_heatmap)
            st.dataframe(styled_df, use_container_width=True)
            
            # Export Heatmap Image Button
            h_col1, h_col2 = st.columns([2, 2])
            with h_col1:
                img_path = os.path.join(os.getcwd(), "assets", "portfolio_monthly_returns.png")
                if st.button("🖼️ note・X添付用ヒートマップ画像を生成", key="btn_gen_heatmap_img", use_container_width=True):
                    ok = cls.generate_monthly_heatmap_image(rets, img_path)
                    if ok:
                        st.toast("月次ヒートマップ画像を生成しました！", icon="🎨")
                        st.session_state['hf_monthly_heatmap_ready'] = True
            
            with h_col2:
                if st.session_state.get('hf_monthly_heatmap_ready') and os.path.exists(img_path):
                    st.image(img_path, caption="生成された月次ヒートマップ画像 (note / X 添付可能)", use_container_width=True)
