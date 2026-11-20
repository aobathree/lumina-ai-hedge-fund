# -*- coding: utf-8 -*-
"""
日足の共通の処理（yfinance の日足の形）。J-Quants（jquants_source）・Alpaca（alpaca_source）・as-of の再生（asof_replay）で使う。
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

import pandas as pd

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
INTRADAY = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


def empty_frame(tz: str = "Asia/Tokyo") -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLS + ["Dividends", "Stock Splits"],
                        index=pd.DatetimeIndex([], tz=tz, name="Date"))


def _period_start(end: pd.Timestamp, period: Optional[str]) -> Optional[pd.Timestamp]:
    import re
    if not period or period == "max":
        return None
    if period == "ytd":
        return pd.Timestamp(year=end.year, month=1, day=1, tz=end.tz)
    m = re.fullmatch(r"(\d+)(d|wk|mo|y)", period)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    if unit == "d":
        return None  # yfinance の "5d" は営業日の本数。slice_bars で tail(n) する
    if unit == "wk":
        return end - pd.Timedelta(weeks=n)
    if unit == "mo":
        return end - pd.DateOffset(months=n)
    return end - pd.DateOffset(years=n)


def slice_bars(df: pd.DataFrame, asof: Optional[dt.date], period: Optional[str], start=None, end=None) -> pd.DataFrame:
    """日足を asof の日まで（含む）に切り、yfinance の period / start / end と同じ範囲にする。"""
    import re
    if df is None or df.empty:
        return df
    dates = pd.Index([ts.date() for ts in df.index])
    out = df[dates <= asof] if asof is not None else df
    if start is not None:
        out = out[pd.Index([ts.date() for ts in out.index]) >= pd.Timestamp(start).date()]
    if end is not None:
        out = out[pd.Index([ts.date() for ts in out.index]) < pd.Timestamp(end).date()]  # end は含まない
    if start is None and end is None and period:
        m = re.fullmatch(r"(\d+)d", period)
        if m:
            return out.tail(int(m.group(1)))
        if not out.empty:
            ps = _period_start(out.index[-1], period)
            if ps is not None:
                out = out[out.index > ps]
    return out


def resample_bars(daily: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = {"1wk": "W", "1mo": "ME", "3mo": "QE"}.get(interval)
    if rule is None or daily.empty:
        return daily
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
           "Dividends": "sum", "Stock Splits": "sum"}
    return daily.resample(rule).agg({k: v for k, v in agg.items() if k in daily.columns}).dropna(subset=["Close"])


def price_fields(bars: pd.DataFrame) -> Dict[str, Any]:
    """日足の最後の日から、``info`` の価格の項目と 52 週の高安を作る。"""
    if bars is None or bars.empty:
        return {}
    last = bars.iloc[-1]
    close = float(last["Close"])
    prev = float(bars["Close"].iloc[-2]) if len(bars) >= 2 else None
    year = bars[bars.index > bars.index[-1] - pd.DateOffset(years=1)]
    return {
        "currentPrice": close, "regularMarketPrice": close,
        "previousClose": prev, "regularMarketPreviousClose": prev,
        "open": float(last["Open"]), "regularMarketOpen": float(last["Open"]),
        "dayHigh": float(last["High"]), "dayLow": float(last["Low"]),
        "regularMarketDayHigh": float(last["High"]), "regularMarketDayLow": float(last["Low"]),
        "volume": float(last["Volume"]), "regularMarketVolume": float(last["Volume"]),
        "fiftyTwoWeekHigh": float(year["High"].max()), "fiftyTwoWeekLow": float(year["Low"].min()),
    }


_PRICE_DEPENDENT = ("marketCap", "trailingPE", "forwardPE", "priceToBook", "dividendYield", "trailingAnnualDividendYield")


def rewrite_info(real: Dict[str, Any], daily: pd.DataFrame, asof: Optional[dt.date]) -> Dict[str, Any]:
    """yfinance の ``info`` の価格と、価格から決まる指標を、別の日足（asof の日まで）の終値で置き直す。

    日足が無ければ価格の項目を消す（いまの値を残さない）。
    """
    info = dict(real)
    bars = slice_bars(daily, asof, None) if daily is not None else None
    px = price_fields(bars) if bars is not None else {}
    if not px:
        for k in ("currentPrice", "regularMarketPrice", "previousClose", "regularMarketPreviousClose", "open",
                  "regularMarketOpen", "dayHigh", "dayLow", "fiftyTwoWeekHigh", "fiftyTwoWeekLow") + _PRICE_DEPENDENT:
            info.pop(k, None)
        return info
    close = px["currentPrice"]
    live_price = real.get("currentPrice") or real.get("regularMarketPrice")
    info.update(px)
    if real.get("trailingEps"):
        info["trailingPE"] = close / float(real["trailingEps"]) if float(real["trailingEps"]) > 0 else None
    if real.get("forwardEps"):
        info["forwardPE"] = close / float(real["forwardEps"]) if float(real["forwardEps"]) > 0 else None
    if real.get("bookValue"):
        info["priceToBook"] = close / float(real["bookValue"]) if float(real["bookValue"]) > 0 else None
    if live_price and float(live_price) > 0:
        ratio = close / float(live_price)
        if real.get("marketCap"):
            info["marketCap"] = float(real["marketCap"]) * ratio
        for k in ("dividendYield", "trailingAnnualDividendYield"):
            if real.get(k):
                info[k] = float(real[k]) / ratio
    if asof is not None:
        info["_asof"] = asof.isoformat()
    return info
