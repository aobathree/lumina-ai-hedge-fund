# -*- coding: utf-8 -*-
"""
市場データの取り先を銘柄ごとに決める（フォークの方針: 公式の API が主、yfinance は控え）。

| 銘柄 | 1 番目 | 控え |
| --- | --- | --- |
| 東証（``.T``） | J-Quants（``jquants_source``） | yfinance |
| 米国の株・ETF（``AAPL``、``BRK-B``、``SPY``） | Alpaca（``alpaca_source``。日足だけ） | yfinance（財務などは常に yfinance） |
| 指数・先物・為替（``^VIX``、``USDJPY=X``、``HG=F``） | yfinance（J-Quants にも Alpaca にも無い） | — |

J-Quants・Alpaca は「鍵がある」か「控えの置き場に行がある」ときに使う。銘柄の行が取れなければ、その銘柄だけ yfinance に落ちる。

``LUMINA_DATA_SOURCES``: ``auto``（既定。上の表）／ ``yfinance``（すべて yfinance。従来どおり）。
控えの置き場は ``LUMINA_JQ_DIR``（既定 ``.trial/jq``）と ``LUMINA_ALPACA_DIR``（既定 ``.trial/alpaca``）。

``install()`` は同じプロセスの ``yfinance.Ticker`` と ``yfinance.download`` を振り分けるものに差し替える
（フォークの各部品は ``yf.Ticker`` / ``yf.download`` を呼ぶので、部品のコードは変えない）。
過去の日の再生（``asof_replay``）は、同じ振り分けを as-of の日で切った形で持つ。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.ai_hedge_fund.alpaca_source import AlpacaStore, AlpacaTicker, is_us_equity
from core.ai_hedge_fund.bar_utils import INTRADAY, PRICE_COLS
from core.ai_hedge_fund.jquants_source import JQuantsStore, JQuantsTicker, jq_code

_STATE: Dict[str, Any] = {}


def classify(symbol: str) -> str:
    """``jp`` / ``us`` / ``other``"""
    s = str(symbol).strip().upper()
    if jq_code(s) is not None:
        return "jp"
    if is_us_equity(s):
        return "us"
    return "other"


def _jq_usable(store: JQuantsStore) -> bool:
    if (os.getenv("JQUANTS_API_KEY") or "").strip():
        return True
    d = store.root / "daily"
    return d.is_dir() and any(d.glob("*.json"))


def _alpaca_usable(store: AlpacaStore) -> bool:
    return store.has_local() or store.has_keys()


class Router:
    """銘柄ごとに Ticker を作る。as-of の日を持つ版は asof_replay が作る。"""

    def __init__(self, jq: Optional[JQuantsStore], alpaca: Optional[AlpacaStore], real_ticker: Callable[..., Any],
                 jp_asof=None, us_asof=None, backup: Optional[Callable[[str], Any]] = None):
        self.jq, self.alpaca, self.real = jq, alpaca, real_ticker
        self.jp_asof, self.us_asof = jp_asof, us_asof
        self.backup = backup or real_ticker  # J-Quants・Alpaca に無い属性と、行が無い銘柄の取り先
        self.used: Dict[str, str] = {}

    def source_of(self, symbol: str) -> str:
        kind = classify(symbol)
        if kind == "jp" and self.jq is not None and self.jq.rows("daily", jq_code(symbol)):
            return "J-Quants"
        if kind == "us" and self.alpaca is not None and self.alpaca.rows(symbol):
            return "Alpaca"
        return "yfinance"

    def ticker(self, symbol, *args, **kwargs):
        sym = str(symbol)
        src = self.source_of(sym)
        self.used[sym.upper()] = src
        if src == "J-Quants":
            return JQuantsTicker(sym, self.jq, self.jp_asof, real_factory=self.backup)
        if src == "Alpaca":
            return AlpacaTicker(sym, self.alpaca, self.us_asof, real_factory=self.backup)
        return self.backup(sym, *args, **kwargs)

    def download(self, tickers, period: Optional[str] = "1mo", interval: str = "1d", start=None, end=None,
                 group_by: str = "column", progress: bool = False, **kwargs) -> pd.DataFrame:
        """``yf.download`` の代わり。列は yfinance 1.x と同じ (Price, Ticker) の 2 段。"""
        syms = [t for t in re.split(r"[ ,]+", tickers.strip()) if t] if isinstance(tickers, str) else [str(t) for t in tickers]
        frames = {}
        for s in syms:
            try:
                df = self.ticker(s).history(period=period, interval=interval, start=start, end=end)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            df = df[[c for c in PRICE_COLS if c in df.columns]].copy()
            if interval not in INTRADAY:
                df.index = pd.DatetimeIndex([pd.Timestamp(ts.date()) for ts in df.index], name="Date")
            frames[s] = df
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
        out.columns.names = ["Price", "Ticker"]
        return out.sort_index()


def stores(jq_root: Optional[Path] = None, alpaca_root: Optional[Path] = None, fetch: bool = True):
    """方針（LUMINA_DATA_SOURCES）に従って、使える控えの置き場を返す。使わないものは None。"""
    policy = (os.getenv("LUMINA_DATA_SOURCES") or "auto").strip().lower()
    if policy == "yfinance":
        return None, None
    if policy != "auto":
        raise ValueError(f"LUMINA_DATA_SOURCES が不正: {policy!r}（auto / yfinance）")
    jq = JQuantsStore(Path(jq_root or os.getenv("LUMINA_JQ_DIR") or Path.cwd() / ".trial" / "jq"), fetch=fetch)
    al = AlpacaStore(Path(alpaca_root or os.getenv("LUMINA_ALPACA_DIR") or Path.cwd() / ".trial" / "alpaca"), fetch=fetch)
    return (jq if _jq_usable(jq) else None), (al if _alpaca_usable(al) else None)


def install(jq_root: Optional[Path] = None, alpaca_root: Optional[Path] = None) -> Dict[str, str]:
    """同じプロセスの ``yf.Ticker`` / ``yf.download`` を振り分けるものにする（いまの値）。戻り値: 種類 → 取り先"""
    import yfinance as yf

    if _STATE.get("router") is not None:
        return _STATE["summary"]
    jq, al = stores(jq_root, alpaca_root)
    router = Router(jq, al, yf.Ticker)
    _STATE.update({"router": router, "real_ticker": yf.Ticker, "real_download": yf.download})
    yf.Ticker = router.ticker
    yf.download = router.download
    _STATE["summary"] = {"東証": "J-Quants（無い銘柄は yfinance）" if jq else "yfinance",
                         "米国の株・ETF": "Alpaca の日足（財務などと無い銘柄は yfinance）" if al else "yfinance",
                         "指数・先物・為替": "yfinance"}
    return _STATE["summary"]


def used_sources() -> Dict[str, str]:
    """install() のあとに作られた Ticker の、銘柄 → 取り先。"""
    r = _STATE.get("router")
    return dict(r.used) if r else {}
