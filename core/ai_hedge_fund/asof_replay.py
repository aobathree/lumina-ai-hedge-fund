# -*- coding: utf-8 -*-
"""
過去のある時点（as-of）で Lumina を動かすための差し替え層。

フォークの各部品は yfinance から「いま」の値を取る。過去の日の寄前の判断を再現するには、
その時点より後の値（当日の終値・当日のニュース）が入ってはいけない。この層は、同じプロセスの中で:

- 東証の銘柄（``.T``）の日足を、J-Quants から取った日足（``asof`` の日まで）に差し替える
- それ以外の銘柄（米国株・指数・為替）の日足は yfinance のまま、``asof`` の日より後を切り捨てる
- ``Ticker.info`` の価格と、価格から決まる指標（PER・PBR・時価総額・配当利回り・52 週の高安）を ``asof`` の終値で計算し直す
- 東証の銘柄の分足は返さない（J-Quants の日足しか持たない）
- ニュースは ``news_cutoff`` より前に出た記事だけを残す
- 立会の判定（MarketSession）を ``session_now`` の時刻の東証の立会中として答える

財務諸表（``financials`` など）と決算の予定（``calendar``）は yfinance のまま渡す（公表済みの過去の値）。

使い方::

    from core.ai_hedge_fund import asof_replay
    asof_replay.install(asof=datetime.date(2026, 11, 19), jq_dir=Path(".trial/jq/daily"),
                        news_cutoff=datetime.datetime(2026, 11, 20, 8, 0, tzinfo=JST))
"""
from __future__ import annotations

import datetime as dt
import email.utils
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

JST = dt.timezone(dt.timedelta(hours=9))

_STATE: Dict[str, Any] = {"installed": False}
_LOCK = threading.Lock()

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
INTRADAY = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


# ---------------------------------------------------------------- J-Quants の日足


def jq_code(symbol: str) -> Optional[str]:
    """``7203.T`` → ``72030``（J-Quants の 5 桁のコード）。東証の銘柄でなければ None。"""
    s = symbol.strip().upper()
    if not s.endswith(".T"):
        return None
    base = s[:-2]
    if len(base) == 4:
        return base + "0"
    return base


def load_jq_bars(path: Path) -> pd.DataFrame:
    """J-Quants の ``/v2/equities/bars/daily`` の行（JSON の配列）を yfinance の形の DataFrame にする。

    値は分割の調整済み（``AdjO`` など）。index は東京時間 0 時の tz 付き日時（yfinance の日足と同じ形）。
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = []
    for r in rows:
        if r.get("AdjC") is None:
            continue  # 売買の無い日
        recs.append({
            "Date": pd.Timestamp(r["Date"]).tz_localize("Asia/Tokyo"),
            "Open": float(r["AdjO"]),
            "High": float(r["AdjH"]),
            "Low": float(r["AdjL"]),
            "Close": float(r["AdjC"]),
            "Volume": float(r["AdjVo"] or 0.0),
        })
    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return pd.DataFrame(columns=PRICE_COLS + ["Dividends", "Stock Splits"],
                            index=pd.DatetimeIndex([], tz="Asia/Tokyo", name="Date"))
    df = df.set_index("Date").sort_index()
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0
    return df


def _period_start(end: pd.Timestamp, period: Optional[str]) -> Optional[pd.Timestamp]:
    if not period or period == "max":
        return None
    if period == "ytd":
        return pd.Timestamp(year=end.year, month=1, day=1, tz=end.tz)
    m = re.fullmatch(r"(\d+)(d|wk|mo|y)", period)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    if unit == "d":
        # yfinance の "5d" は営業日の本数。呼ぶ側で tail(n) する
        return None
    if unit == "wk":
        return end - pd.Timedelta(weeks=n)
    if unit == "mo":
        return end - pd.DateOffset(months=n)
    return end - pd.DateOffset(years=n)


def _slice(df: pd.DataFrame, asof: dt.date, period: Optional[str], start=None, end=None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx_dates = pd.Index([ts.date() for ts in df.index])
    out = df[idx_dates <= asof]
    if start is not None:
        s = pd.Timestamp(start).date()
        out = out[pd.Index([ts.date() for ts in out.index]) >= s]
    if end is not None:
        e = pd.Timestamp(end).date()  # yfinance の end は含まない
        out = out[pd.Index([ts.date() for ts in out.index]) < e]
    if start is None and end is None and period:
        m = re.fullmatch(r"(\d+)d", period)
        if m:
            return out.tail(int(m.group(1)))
        if not out.empty:
            ps = _period_start(out.index[-1], period)
            if ps is not None:
                out = out[out.index > ps]
    return out


def _empty_like(tz: str = "Asia/Tokyo") -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLS + ["Dividends", "Stock Splits"],
                        index=pd.DatetimeIndex([], tz=tz, name="Datetime"))


# ---------------------------------------------------------------- Ticker の差し替え

_PRICE_INFO_KEYS = ("currentPrice", "regularMarketPrice", "previousClose", "regularMarketPreviousClose",
                    "open", "regularMarketOpen", "dayHigh", "dayLow", "regularMarketDayHigh",
                    "regularMarketDayLow", "volume", "regularMarketVolume")


class AsOfTicker:
    """yfinance の ``Ticker`` を包み、``asof`` より後の値を見せない。"""

    def __init__(self, symbol: str, *args, **kwargs):
        self._symbol = str(symbol)
        self._real = _STATE["real_ticker"](symbol, *args, **kwargs)
        self._info: Optional[Dict[str, Any]] = None

    # yfinance の属性で、ここで扱わないものはそのまま渡す（財務諸表・決算の予定など）。
    # 控えの置き場（snapshot_dir）があれば、財務諸表などは最初に取れた値を控えて使い回す。
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
        code = jq_code(self._symbol)
        if code is not None:
            return _jq_frame(code)
        with _LOCK:
            cache = _STATE["foreign_cache"]
            if self._symbol not in cache:
                try:
                    cache[self._symbol] = _snapshot(self._symbol, "history_2y",
                                                    lambda: self._real.history(period="2y", auto_adjust=True))
                except Exception:
                    cache[self._symbol] = _empty_like("UTC")
            return cache[self._symbol]

    def history(self, period: Optional[str] = "1mo", interval: str = "1d", start=None, end=None, **kwargs) -> pd.DataFrame:
        asof: dt.date = _STATE["asof"]
        if interval in INTRADAY:
            if jq_code(self._symbol) is not None:
                return _empty_like()
            try:
                df = self._real.history(period=period, interval=interval, start=start, end=end, **kwargs)
            except Exception:
                return _empty_like("UTC")
            if df is None or df.empty:
                return df
            return df[pd.Index([ts.date() for ts in df.index]) <= asof]
        if interval not in ("1d", "1D"):
            daily = _slice(self._daily(), asof, period, start, end)
            rule = {"1wk": "W", "1mo": "ME"}.get(interval)
            if rule is None or daily.empty:
                return daily
            agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
                   "Dividends": "sum", "Stock Splits": "sum"}
            return daily.resample(rule).agg({k: v for k, v in agg.items() if k in daily.columns}).dropna(subset=["Close"])
        return _slice(self._daily(), asof, period, start, end).copy()

    @property
    def info(self) -> Dict[str, Any]:
        if self._info is not None:
            return self._info
        try:
            real = dict(_snapshot(self._symbol, "info", lambda: self._real.info) or {})
        except Exception:
            real = {}
        self._info = rewrite_info(real, self._daily(), _STATE["asof"])
        return self._info

    @property
    def fast_info(self) -> Dict[str, Any]:
        inf = self.info
        return {
            "last_price": inf.get("currentPrice"),
            "lastPrice": inf.get("currentPrice"),
            "previous_close": inf.get("previousClose"),
            "previousClose": inf.get("previousClose"),
            "currency": inf.get("currency"),
            "market_cap": inf.get("marketCap"),
        }

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

    @property
    def actions(self) -> pd.DataFrame:
        a = self._real.actions
        if a is None or len(a) == 0:
            return a
        return a[pd.Index([ts.date() for ts in a.index]) <= _STATE["asof"]]


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
        _STATE.setdefault("snapshot_writes", 0)
        _STATE["snapshot_writes"] += 1
    return value


def _cut_series(s):
    if s is None or len(s) == 0:
        return s
    return s[pd.Index([ts.date() for ts in s.index]) <= _STATE["asof"]]


def rewrite_info(real: Dict[str, Any], daily: pd.DataFrame, asof: dt.date) -> Dict[str, Any]:
    """``info`` の価格と、価格から決まる指標を ``asof`` の終値で置き直す。

    日足が無ければ価格の項目を消す（いまの値を残さない）。
    """
    info = dict(real)
    bars = _slice(daily, asof, None) if daily is not None else None
    if bars is None or bars.empty:
        for k in _PRICE_INFO_KEYS + ("marketCap", "trailingPE", "forwardPE", "priceToBook",
                                     "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "dividendYield"):
            info.pop(k, None)
        return info
    last = bars.iloc[-1]
    close = float(last["Close"])
    prev = float(bars["Close"].iloc[-2]) if len(bars) >= 2 else None
    live_price = real.get("currentPrice") or real.get("regularMarketPrice")
    info.update({
        "currentPrice": close,
        "regularMarketPrice": close,
        "previousClose": prev,
        "regularMarketPreviousClose": prev,
        "open": float(last["Open"]),
        "regularMarketOpen": float(last["Open"]),
        "dayHigh": float(last["High"]),
        "dayLow": float(last["Low"]),
        "regularMarketDayHigh": float(last["High"]),
        "regularMarketDayLow": float(last["Low"]),
        "volume": float(last["Volume"]),
        "regularMarketVolume": float(last["Volume"]),
    })
    year = bars[bars.index > bars.index[-1] - pd.DateOffset(years=1)]
    info["fiftyTwoWeekHigh"] = float(year["High"].max())
    info["fiftyTwoWeekLow"] = float(year["Low"].min())
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


def _jq_frame(code: str) -> pd.DataFrame:
    with _LOCK:
        cache = _STATE["jq_cache"]
        if code not in cache:
            path = Path(_STATE["jq_dir"]) / f"{code}.json"
            cache[code] = load_jq_bars(path) if path.exists() else _empty_like()
        return cache[code]


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


def install(asof: dt.date, jq_dir: Path, news_cutoff: dt.datetime,
            session_now: Optional[dt.datetime] = None, snapshot_dir: Optional[Path] = None) -> None:
    """同じプロセスの yfinance・ニュース・立会の判定を、as-of の値に差し替える。

    ``snapshot_dir`` を渡すと、yfinance の info・財務諸表・東証以外の日足を控え、次の実行から使い回す
    （何度も回すと yfinance の取得の制限に当たり、財務が欠けて結果が変わるため）。
    """
    import yfinance as yf

    if _STATE.get("installed"):
        raise RuntimeError("asof_replay はすでに入っている")
    _STATE.update({
        "installed": True,
        "asof": asof,
        "jq_dir": Path(jq_dir),
        "news_cutoff": news_cutoff,
        "session_now": session_now or dt.datetime.combine(asof + dt.timedelta(days=1), dt.time(9, 0), JST),
        "jq_cache": {},
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
