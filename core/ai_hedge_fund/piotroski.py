# -*- coding: utf-8 -*-
"""
Piotroski F-Score（9 項目）の計算（Issue #6）。

入力は年次の財務諸表（yfinance の financials / balance_sheet / cashflow と同じ形:
行 = 項目名、列 = 決算期、新しい期が左）。直近 2 期が要る。

9 項目のうち 1 つでも計算できなければ score は None を返す（一部だけで点を付けない）。
銀行など、売上総利益や流動資産の項目が無い業種は None になる。
"""
from typing import Any, Dict, Optional, Sequence

import pandas as pd

ROW_ALIASES: Dict[str, Sequence[str]] = {
    "net_income": ("Net Income", "Net Income Common Stockholders"),
    "total_assets": ("Total Assets",),
    "operating_cash_flow": ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
    "long_term_debt": ("Long Term Debt", "Long Term Debt And Capital Lease Obligation", "Total Debt"),
    "current_assets": ("Current Assets",),
    "current_liabilities": ("Current Liabilities",),
    "shares": ("Ordinary Shares Number", "Share Issued"),
    "gross_profit": ("Gross Profit",),
    "revenue": ("Total Revenue", "Operating Revenue"),
}


def _two_periods(df: Optional[pd.DataFrame], key: str):
    """(当期, 前期) を返す。取れなければ (None, None)。"""
    if df is None or getattr(df, "empty", True):
        return None, None
    cols = sorted(df.columns, reverse=True)  # 新しい期を先頭に
    if len(cols) < 2:
        return None, None
    for name in ROW_ALIASES[key]:
        if name in df.index:
            cur, prev = df.loc[name, cols[0]], df.loc[name, cols[1]]
            if pd.notna(cur) and pd.notna(prev):
                return float(cur), float(prev)
    return None, None


def compute_f_score(
    financials: Optional[pd.DataFrame],
    balance_sheet: Optional[pd.DataFrame],
    cashflow: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Returns: {"score": int | None, "criteria": {名前: bool | None}, "missing": [取れなかった項目]}
    """
    v: Dict[str, Any] = {}
    for key, df in (
        ("net_income", financials), ("gross_profit", financials), ("revenue", financials),
        ("total_assets", balance_sheet), ("long_term_debt", balance_sheet),
        ("current_assets", balance_sheet), ("current_liabilities", balance_sheet), ("shares", balance_sheet),
        ("operating_cash_flow", cashflow),
    ):
        v[key] = _two_periods(df, key)

    def ok(*keys: str) -> bool:
        return all(v[k][0] is not None for k in keys)

    c: Dict[str, Optional[bool]] = {}
    ni, ni_p = v["net_income"]
    ta, ta_p = v["total_assets"]
    cfo, _ = v["operating_cash_flow"]

    # 収益性
    c["roa_positive"] = (ni / ta > 0) if ok("net_income", "total_assets") and ta else None
    c["cfo_positive"] = (cfo > 0) if ok("operating_cash_flow") else None
    c["roa_improved"] = (ni / ta > ni_p / ta_p) if ok("net_income", "total_assets") and ta and ta_p else None
    c["cfo_exceeds_net_income"] = (cfo > ni) if ok("operating_cash_flow", "net_income") else None
    # 財務の健全性
    ltd, ltd_p = v["long_term_debt"]
    c["leverage_decreased"] = (ltd / ta < ltd_p / ta_p) if ok("long_term_debt", "total_assets") and ta and ta_p else None
    ca, ca_p = v["current_assets"]
    cl, cl_p = v["current_liabilities"]
    c["current_ratio_improved"] = (
        (ca / cl > ca_p / cl_p) if ok("current_assets", "current_liabilities") and cl and cl_p else None
    )
    sh, sh_p = v["shares"]
    c["no_new_shares"] = (sh <= sh_p) if ok("shares") else None
    # 効率
    gp, gp_p = v["gross_profit"]
    rev, rev_p = v["revenue"]
    c["gross_margin_improved"] = (gp / rev > gp_p / rev_p) if ok("gross_profit", "revenue") and rev and rev_p else None
    c["asset_turnover_improved"] = (rev / ta > rev_p / ta_p) if ok("revenue", "total_assets") and ta and ta_p else None

    missing = [k for k, val in c.items() if val is None]
    score = None if missing else sum(1 for val in c.values() if val)
    return {"score": score, "criteria": c, "missing": missing}
