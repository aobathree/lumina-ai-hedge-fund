# -*- coding: utf-8 -*-
"""
米国の株・ETF の日足を Alpaca のマーケットデータ API（``/v2/stocks/bars``）から取る。

Alpaca から取るのは日足だけ。財務・事業の説明・インサイダー・オプション・配当の履歴は Alpaca に無いので、
``AlpacaTicker`` はそれらを控えの yfinance（``real_factory``）に回す。``info`` は yfinance の値に、Alpaca の
日足の価格と、価格から決まる指標（PER・PBR・時価総額・配当利回り・52 週の高安）を置き直したもの。

- 鍵: 環境変数 ``ALPACA_API_KEY_ID`` / ``ALPACA_API_SECRET_KEY``（``op://`` の参照なら 1Password で解決）
- 値: SIP（全取引所の統合）、分割・配当の調整 ``adjustment=all``（yfinance の ``auto_adjust`` と同じ扱い）
- Basic の契約は直近 15 分の SIP を取れないので、取得の終わりを「いまより 20 分前」までにする
- 控え: ``<root>/us_daily/<銘柄>.json``（API の行のまま）。銘柄は yfinance の形（``BRK-B``）で持ち、API には ``BRK.B`` で問う
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.ai_hedge_fund.bar_utils import INTRADAY, empty_frame, resample_bars, rewrite_info, slice_bars

API = "https://data.alpaca.markets/v2/stocks/bars"
ET = "America/New_York"
KIND = "us_daily"

_US_EQUITY = re.compile(r"[A-Z]{1,5}(?:[.-][A-Z]{1,2})?")


def is_us_equity(symbol: str) -> bool:
    """米国の株・ETF の形の銘柄か（``AAPL``、``BRK-B``、``SPY``）。指数 ``^VIX``・為替 ``USDJPY=X``・``.T`` などは偽。"""
    s = str(symbol).strip().upper()
    if not _US_EQUITY.fullmatch(s):
        return False
    # 末尾が取引所の接尾辞（.T・.L・.AS など、2 文字以下）でも形は合うので、yfinance の外国の接尾辞を除く
    return not re.search(r"\.(T|L|AS|DE|PA|CO|MI|MC|ST|HK|TO|AX|SW|F)$", s)


def alpaca_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace("-", ".")


def _secret(name: str) -> str:
    raw = (os.getenv(name) or "").strip()
    if raw.startswith("op://"):
        from core.ai_hedge_fund.llm_client import read_op_reference
        raw = read_op_reference(raw)
    return raw


class AlpacaStore:
    """Alpaca の日足を銘柄ごとに控える。"""

    def __init__(self, root: Path, fetch: bool = True, key_id: Optional[str] = None, secret: Optional[str] = None):
        self.root = Path(root)
        self._fetch = fetch
        self._key_id, self._secret = key_id, secret
        self._mem: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def has_keys(self) -> bool:
        if self._key_id is None:
            self._key_id, self._secret = _secret("ALPACA_API_KEY_ID"), _secret("ALPACA_API_SECRET_KEY")
        return bool(self._key_id and self._secret)

    def path(self, symbol: str) -> Path:
        return self.root / KIND / f"{symbol.upper()}.json"

    def has_local(self) -> bool:
        d = self.root / KIND
        return d.is_dir() and any(d.glob("*.json"))

    def rows(self, symbol: str) -> List[Dict[str, Any]]:
        sym = symbol.upper()
        with self._lock:
            if sym in self._mem:
                return self._mem[sym]
        p = self.path(sym)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
        elif self._fetch and self.has_keys():
            try:
                data = self.download([sym], start=(dt.date.today() - dt.timedelta(days=760)).isoformat())[sym]
            except Exception:
                data = []
        else:
            data = []
        with self._lock:
            self._mem[sym] = data
        return data

    def download(self, symbols: List[str], start: str, end: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """日足を取って控える。end は RFC 3339（既定はいまより 20 分前）。戻り値: 銘柄（yfinance の形）→ 行"""
        if not self.has_keys():
            raise RuntimeError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY が無い")
        if end is None:
            end = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_yf = {alpaca_symbol(s): s.upper() for s in symbols}
        out: Dict[str, List[Dict[str, Any]]] = {s.upper(): [] for s in symbols}
        q = {"symbols": ",".join(to_yf), "timeframe": "1Day", "start": start, "end": end,
             "adjustment": "all", "feed": "sip", "limit": 10000, "sort": "asc"}
        headers = {"APCA-API-KEY-ID": self._key_id, "APCA-API-SECRET-KEY": self._secret}
        while True:
            req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(q)}", headers=headers)
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        body = json.load(r)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < 4:
                        time.sleep(2 ** attempt * 2)
                        continue
                    raise
            for a_sym, bars in (body.get("bars") or {}).items():
                out[to_yf.get(a_sym, a_sym)].extend(bars)
            token = body.get("next_page_token")
            if not token:
                break
            q["page_token"] = token
        for sym, rows in out.items():
            p = self.path(sym)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rows), encoding="utf-8")
            with self._lock:
                self._mem[sym] = rows
        return out


def bars_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Alpaca の日足の行を yfinance の日足の形にする（ニューヨーク時間 0 時の tz 付き）。"""
    if not rows:
        return empty_frame(ET)
    recs = []
    for r in rows:
        day = pd.Timestamp(r["t"]).tz_convert(ET).normalize()
        recs.append({"Date": day, "Open": float(r["o"]), "High": float(r["h"]), "Low": float(r["l"]),
                     "Close": float(r["c"]), "Volume": float(r.get("v") or 0.0)})
    df = pd.DataFrame.from_records(recs).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0
    return df


class AlpacaTicker:
    """米国の株・ETF について、日足を Alpaca から、それ以外を控えの yfinance から返す（yfinance の ``Ticker`` の形）。"""

    def __init__(self, symbol: str, store: AlpacaStore, asof: Optional[dt.date] = None,
                 real_factory: Optional[Callable[[str], Any]] = None):
        self._symbol = str(symbol).upper()
        self._store = store
        self._asof = asof
        self._real_factory = real_factory
        self._real = None
        self._info: Optional[Dict[str, Any]] = None

    @property
    def ticker(self) -> str:
        return self._symbol

    def _backup(self):
        if self._real is None:
            if self._real_factory is None:
                import yfinance as yf
                self._real_factory = yf.Ticker
            self._real = self._real_factory(self._symbol)
        return self._real

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._backup(), name)

    def _bars(self) -> pd.DataFrame:
        return bars_frame(self._store.rows(self._symbol))

    def history(self, period: Optional[str] = "1mo", interval: str = "1d", start=None, end=None, **kwargs) -> pd.DataFrame:
        if interval in INTRADAY:
            return self._backup().history(period=period, interval=interval, start=start, end=end, **kwargs)
        daily = slice_bars(self._bars(), self._asof, period, start, end).copy()
        return daily if interval in ("1d", "1D") else resample_bars(daily, interval)

    @property
    def info(self) -> Dict[str, Any]:
        if self._info is None:
            b = self._backup()
            try:
                raw = dict(getattr(b, "raw_info", None) or b.info or {})
            except Exception:
                raw = {}
            info = rewrite_info(raw, self._bars(), self._asof)
            info["_price_source"] = "Alpaca"
            self._info = info
        return self._info

    @property
    def fast_info(self) -> Dict[str, Any]:
        inf = self.info
        return {"last_price": inf.get("currentPrice"), "lastPrice": inf.get("currentPrice"),
                "previous_close": inf.get("previousClose"), "previousClose": inf.get("previousClose"),
                "currency": inf.get("currency", "USD"), "market_cap": inf.get("marketCap")}
