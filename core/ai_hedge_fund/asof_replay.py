# -*- coding: utf-8 -*-
"""
過去のある時点（as-of）で Lumina を動かすための差し替え層。

フォークの各部品は yfinance から「いま」の値を取る。過去の日の寄前の判断を再現するには、
その時点より後の値（当日の終値・当日のニュース）が入ってはいけない。この層は、同じプロセスの中で:

- 東証の銘柄（``.T``）は、日足・銘柄情報・財務・決算の予定・配当・分割を J-Quants から、``asof`` の日までに
  開示・公表された分だけで作る（``jquants_source.JQuantsTicker``）。J-Quants に無いもの（ニュース等）だけ yfinance
- それ以外（米国株・指数・為替・VIX）は yfinance のまま、日足を ``asof`` の日で切り、``info`` の価格と
  価格から決まる指標をその日の終値で置き直す
- ニュースは ``news_cutoff`` より前に出た記事だけを残す
- 立会の判定（MarketSession）を ``session_now`` の時刻の東証の立会中として答える

使い方::

    from core.ai_hedge_fund import asof_replay
    asof_replay.install(asof=datetime.date(2026, 11, 19), jq_root=Path(".trial/jq"),
                        news_cutoff=datetime.datetime(2026, 11, 20, 8, 0, tzinfo=JST))
"""
from __future__ import annotations

import datetime as dt
import email.utils
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from core.ai_hedge_fund.jquants_source import (
    INTRADAY, PRICE_COLS, JQuantsStore, JQuantsTicker, empty_frame, jq_code, price_fields, resample_bars, slice_bars,
)

JST = dt.timezone(dt.timedelta(hours=9))

_STATE: Dict[str, Any] = {"installed": False}
_LOCK = threading.Lock()

__all__ = ["install", "set_screen_tiebreak", "installed", "asof_date", "jp_symbols", "jq_code", "AsOfTicker",
           "asof_download", "rewrite_info", "news_item_before_cutoff"]


# ---------------------------------------------------------------- Ticker の差し替え

_PRICE_DEPENDENT = ("marketCap", "trailingPE", "forwardPE", "priceToBook", "dividendYield", "trailingAnnualDividendYield")


def AsOfTicker(symbol: str, *args, **kwargs):
    """``yf.Ticker`` の代わり。東証の銘柄は J-Quants、それ以外は yfinance を asof で切ったもの。"""
    if jq_code(str(symbol)) is not None:
        # J-Quants に無い属性（ニュース等）は、asof・締め切りで切る包みを通して yfinance に回す
        return JQuantsTicker(str(symbol), _STATE["store"], _STATE["asof"], real_factory=ForeignAsOfTicker)
    return ForeignAsOfTicker(symbol, *args, **kwargs)


class ForeignAsOfTicker:
    """東証以外の銘柄の yfinance の ``Ticker`` を包み、``asof`` より後の値を見せない。"""

    def __init__(self, symbol: str, *args, **kwargs):
        self._symbol = str(symbol)
        self._real = _STATE["real_ticker"](symbol, *args, **kwargs)
        self._info: Optional[Dict[str, Any]] = None

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in SNAPSHOT_ATTRS:
            return _snapshot(self._symbol, name, lambda: getattr(self._real, name))
        return getattr(self._real, name)

    @property
    def ticker(self) -> str:
        return self._symbol

    def _daily(self) -> pd.DataFrame:
        with _LOCK:
            cache = _STATE["foreign_cache"]
            if self._symbol not in cache:
                try:
                    cache[self._symbol] = _snapshot(self._symbol, "history_2y",
                                                    lambda: self._real.history(period="2y", auto_adjust=True))
                except Exception:
                    cache[self._symbol] = empty_frame("UTC")
            return cache[self._symbol]

    def history(self, period: Optional[str] = "1mo", interval: str = "1d", start=None, end=None, **kwargs) -> pd.DataFrame:
        asof: dt.date = _STATE["asof"]
        if interval in INTRADAY:
            try:
                df = self._real.history(period=period, interval=interval, start=start, end=end, **kwargs)
            except Exception:
                return empty_frame("UTC")
            if df is None or df.empty:
                return df
            return df[pd.Index([ts.date() for ts in df.index]) <= asof]
        daily = slice_bars(self._daily(), asof, period, start, end).copy()
        return daily if interval in ("1d", "1D") else resample_bars(daily, interval)

    @property
    def info(self) -> Dict[str, Any]:
        if self._info is None:
            try:
                real = dict(_snapshot(self._symbol, "info", lambda: self._real.info) or {})
            except Exception:
                real = {}
            self._info = rewrite_info(real, self._daily(), _STATE["asof"])
        return self._info

    @property
    def fast_info(self) -> Dict[str, Any]:
        inf = self.info
        return {"last_price": inf.get("currentPrice"), "lastPrice": inf.get("currentPrice"),
                "previous_close": inf.get("previousClose"), "previousClose": inf.get("previousClose"),
                "currency": inf.get("currency"), "market_cap": inf.get("marketCap")}

    @property
    def news(self) -> List[Dict[str, Any]]:
        try:
            items = list(self._real.news or [])
        except Exception:
            return []
        return [n for n in items if _yf_news_before_cutoff(n)]

    @property
    def dividends(self) -> pd.Series:
        return _cut_series(_snapshot(self._symbol, "dividends", lambda: self._real.dividends))

    @property
    def splits(self) -> pd.Series:
        return _cut_series(_snapshot(self._symbol, "splits", lambda: self._real.splits))


SNAPSHOT_ATTRS = {
    "financials", "quarterly_financials", "income_stmt", "quarterly_income_stmt",
    "balance_sheet", "quarterly_balance_sheet", "balancesheet", "quarterly_balancesheet",
    "cashflow", "quarterly_cashflow", "cash_flow", "quarterly_cash_flow",
    "calendar", "earnings_dates", "insider_transactions", "insider_purchases",
}


def _is_empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (pd.DataFrame, pd.Series)):
        return v.empty
    if isinstance(v, (dict, list, tuple)):
        return len(v) == 0
    return False


def _snapshot(symbol: str, name: str, fetch):
    """yfinance の値を控えの置き場に控える。控えがあれば取りに行かない。

    取れなかった（例外・空）ときは控えない（取得の制限に当たった回の空の値で固まらないように）。
    控えの置き場が無ければ、毎回取りに行く。
    """
    import pickle

    root = _STATE.get("snapshot_dir")
    if not root:
        return fetch()
    path = Path(root) / re.sub(r"[^A-Za-z0-9._=^-]", "_", symbol) / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    value = fetch()
    if not _is_empty(value):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(value, f)
        tmp.replace(path)
    return value


def _cut_series(s):
    if s is None or len(s) == 0:
        return s
    return s[pd.Index([ts.date() for ts in s.index]) <= _STATE["asof"]]


def rewrite_info(real: Dict[str, Any], daily: pd.DataFrame, asof: dt.date) -> Dict[str, Any]:
    """東証以外の ``info`` の価格と、価格から決まる指標を ``asof`` の終値で置き直す。

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
    info["_asof"] = asof.isoformat()
    return info


# ---------------------------------------------------------------- download の差し替え


def asof_download(tickers, period: Optional[str] = "1mo", interval: str = "1d", start=None, end=None,
                  group_by: str = "column", progress: bool = False, **kwargs) -> pd.DataFrame:
    """``yf.download`` の代わり。列は yfinance 1.x と同じ (Price, Ticker) の 2 段。"""
    if isinstance(tickers, str):
        syms = [t for t in re.split(r"[ ,]+", tickers.strip()) if t]
    else:
        syms = [str(t) for t in tickers]
    frames = {}
    for s in syms:
        df = AsOfTicker(s).history(period=period, interval=interval, start=start, end=end)
        if df is None or df.empty:
            continue
        df = df[[c for c in PRICE_COLS if c in df.columns]].copy()
        if interval not in INTRADAY:
            df.index = pd.DatetimeIndex([pd.Timestamp(ts.date()) for ts in df.index], name="Date")
        frames[s] = df
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)  # (Ticker, Price)
    out = out.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
    out.columns.names = ["Price", "Ticker"]
    return out.sort_index()


# ---------------------------------------------------------------- ニュース


def _parse_when(value: Any) -> Optional[dt.datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
    s = str(value)
    try:
        d = email.utils.parsedate_to_datetime(s)
        if d is not None:
            return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def news_item_before_cutoff(item: Dict[str, Any]) -> bool:
    """日時の分からない記事は残さない（先読みを避ける）。"""
    when = _parse_when(item.get("pub_date") or item.get("published") or item.get("providerPublishTime"))
    return when is not None and when < _STATE["news_cutoff"]


def _yf_news_before_cutoff(n: Dict[str, Any]) -> bool:
    content = n.get("content") or {}
    when = _parse_when(n.get("providerPublishTime") or content.get("pubDate") or content.get("displayTime"))
    return when is not None and when < _STATE["news_cutoff"]


# ---------------------------------------------------------------- install


def install(asof: dt.date, jq_root: Path, news_cutoff: dt.datetime,
            session_now: Optional[dt.datetime] = None, snapshot_dir: Optional[Path] = None,
            fetch_missing: bool = False) -> None:
    """同じプロセスの yfinance・ニュース・立会の判定を、as-of の値に差し替える。

    ``jq_root`` は J-Quants の控えの置き場（``jquants_source.JQuantsStore``、``scripts/jq_fetch.py`` が作る）。
    ``fetch_missing`` が真なら、控えに無い銘柄を J-Quants から取りに行く（鍵が要る）。
    ``snapshot_dir`` を渡すと、東証以外の yfinance の値を控え、次の実行から使い回す。
    """
    import yfinance as yf

    if _STATE.get("installed"):
        raise RuntimeError("asof_replay はすでに入っている")
    _STATE.update({
        "installed": True,
        "asof": asof,
        "store": JQuantsStore(Path(jq_root), fetch=fetch_missing),
        "news_cutoff": news_cutoff,
        "session_now": session_now or dt.datetime.combine(asof + dt.timedelta(days=1), dt.time(9, 0), JST),
        "foreign_cache": {},
        "snapshot_dir": Path(snapshot_dir) if snapshot_dir else None,
        "real_ticker": yf.Ticker,
        "real_download": yf.download,
    })
    yf.Ticker = AsOfTicker
    yf.download = asof_download

    from core.ai_hedge_fund.news_feed import NewsFeed

    real_rss = NewsFeed._fetch_google_news_rss.__func__
    real_yfn = NewsFeed._extract_yfinance_news.__func__ if hasattr(NewsFeed, "_extract_yfinance_news") else None

    def _rss(cls, query, limit=5, lang="ja"):
        items = real_rss(cls, query, limit=max(limit * 4, 20), lang=lang)
        return [i for i in items if news_item_before_cutoff(i)][:limit]

    NewsFeed._fetch_google_news_rss = classmethod(_rss)
    if real_yfn is not None:
        def _yfn(cls, t, limit=5):
            items = real_yfn(cls, t, limit=max(limit * 4, 20))
            return [i for i in items if news_item_before_cutoff(i)][:limit]
        NewsFeed._extract_yfinance_news = classmethod(_yfn)
    NewsFeed._cache.clear()

    from core.ai_hedge_fund.market_session import MarketSession

    now = _STATE["session_now"]
    MarketSession.is_tse_open = classmethod(lambda cls, dt_jst=None: dt.time(9, 0) <= now.timetz().replace(tzinfo=None) < dt.time(15, 30))
    MarketSession.is_us_open = classmethod(lambda cls, dt_utc=None: False)
    MarketSession.is_eu_open = classmethod(lambda cls, dt_utc=None: False)
    MarketSession.is_tse_trading_day = classmethod(lambda cls, target_date=None: True)
    MarketSession.get_market_status = classmethod(lambda cls: {
        "active_market": "JP",
        "status_desc": f"再生: {now:%Y-%m-%d %H:%M} JST の東証の立会中として判定（as-of {asof}）",
    })


def set_screen_tiebreak(mode: str) -> None:
    """スクリーナーの同点の並びを決める。

    フォークの ``scan_opportunities`` は、並列に評価した結果を終わった順に集め、``ai_score`` で安定ソートする。
    ``ai_score`` は 99 で頭打ちになるので、同点の銘柄の並び（= 上位 ``max_results`` に入るか）が実行ごとに変わる。

    - ``asis``: フォークのまま（終わった順）
    - ``universe``: ユニバースの順（逐次に評価したときと同じ並び）
    - ``shuffle:<seed>``: 乱数で並べる（同点の切り方の幅を見る）
    """
    from core.ai_hedge_fund.market_screener import MarketScreener

    if mode == "asis":
        return
    order: Dict[str, int] = {}
    for item in MarketScreener.get_custom_watchlist():
        order.setdefault(item.get("symbol"), len(order))
    for key in ["DEFENSE_ENERGY", "AI_GROWTH", "VALUE_DIVIDEND", "DEEP_VALUE", "EUROPE_GLOBAL"]:
        for item in MarketScreener.UNIVERSE.get(key, []):
            order.setdefault(item["symbol"], len(order))
    if mode.startswith("shuffle:"):
        import random
        rng = random.Random(int(mode.split(":", 1)[1]))
        keys = list(order)
        rng.shuffle(keys)
        order = {k: i for i, k in enumerate(keys)}
    elif mode != "universe":
        raise ValueError(f"tiebreak が不正: {mode!r}（asis / universe / shuffle:<seed>）")

    real = MarketScreener.assign_rs_ratings.__func__

    def _assign(cls, results):
        real(cls, results)
        # この直後に ai_score で安定ソートされるので、ここで並べた順が同点の並びになる
        results.sort(key=lambda r: order.get(r.get("symbol"), len(order)))

    MarketScreener.assign_rs_ratings = classmethod(_assign)
    _STATE["tiebreak"] = mode


def installed() -> bool:
    return bool(_STATE.get("installed"))


def asof_date() -> Optional[dt.date]:
    return _STATE.get("asof")


def jp_symbols(universe: Dict[str, Iterable[Dict[str, Any]]]) -> List[str]:
    seen, out = set(), []
    for items in universe.values():
        for it in items:
            s = it.get("symbol", "")
            if s.endswith(".T") and s not in seen:
                seen.add(s)
                out.append(s)
    return out
