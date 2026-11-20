# -*- coding: utf-8 -*-
"""
東証の銘柄（``.T``）のデータを J-Quants（V2）から取る。

フォークは銘柄の情報・財務・決算の予定・配当を yfinance の ``Ticker`` から読む。yfinance は Yahoo の非公式の
取得口で、取得の制限（429）に当たると財務が欠け、判断が変わる。J-Quants で代わりが取れるものはこちらから取る:

| フォークが読むもの | J-Quants |
| --- | --- |
| 日足 ``history`` | ``/equities/bars/daily``（分割の調整済み ``Adj*``） |
| 社名・業種 ``info`` | ``/equities/master``（``CoName``・``S33Nm``） |
| EPS・BPS・PER・PBR・ROE・時価総額 ``info`` | ``/equities/valuation``（その日の終値で算出、ROE は TTM） |
| 利益率・負債・CF・成長率 ``info`` | ``/fins/summary`` と ``/fins/details`` |
| 年次の財務諸表 ``financials`` / ``balance_sheet`` / ``cashflow``（F-Score） | ``/fins/summary``（通期）と ``/fins/details``（通期） |
| 決算の予定 ``calendar`` | ``/fins/earnings-date``（公表日 ``PubDate`` つき） |
| 配当 ``dividends`` | ``/fins/dividend``（決定、権利落日） |
| 分割 ``splits`` | 日足の ``AdjFactor`` |

J-Quants に無いもの（事業の説明文、ニュース、インサイダー取引、分足）は扱わない（呼ぶ側が yfinance などに回す）。

``asof`` を渡すと、その日までに開示・公表された行だけを使う（過去の日の再生で先読みしない）。

データは ``root/<種類>/<5 桁のコード>.json`` に API の行のまま控える。控えが無く、鍵（環境変数
``JQUANTS_API_KEY``。``op://`` の参照なら 1Password で解決）があれば取りに行く。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

API = "https://api.jquants.com/v2"

# 種類 → (パス, 銘柄ごとに取るときの追加の引数)
KINDS: Dict[str, Tuple[str, Dict[str, str]]] = {
    "daily": ("equities/bars/daily", {}),
    "master": ("equities/master", {}),
    "valuation": ("equities/valuation", {}),
    "fins_summary": ("fins/summary", {}),
    "fins_details": ("fins/details", {}),
    "earnings_date": ("fins/earnings-date", {}),
    "dividend": ("fins/dividend", {}),
}


def jq_code(symbol: str) -> Optional[str]:
    """``7203.T`` → ``72030``（J-Quants の 5 桁のコード）。東証の銘柄でなければ None。"""
    s = symbol.strip().upper()
    if not s.endswith(".T"):
        return None
    base = s[:-2]
    return base + "0" if len(base) == 4 else base


def _num(v: Any) -> Optional[float]:
    if v is None or v == "" or v == "-":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _date(v: Any) -> Optional[dt.date]:
    if not v:
        return None
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------- 取得と控え


class JQuantsStore:
    """J-Quants の行を種類と銘柄ごとに控える。"""

    def __init__(self, root: Path, api_key: Optional[str] = None, fetch: bool = True):
        self.root = Path(root)
        self._key = api_key
        self._fetch = fetch
        self._mem: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _api_key(self) -> str:
        if self._key is None:
            raw = (os.getenv("JQUANTS_API_KEY") or "").strip()
            if raw.startswith("op://"):
                from core.ai_hedge_fund.llm_client import read_op_reference
                raw = read_op_reference(raw)
            self._key = raw
        return self._key

    def get_json(self, path: str, params: Dict[str, str]) -> List[Dict[str, Any]]:
        """API を呼び、ページをたどって全行を返す。"""
        key = self._api_key()
        if not key:
            raise RuntimeError("JQUANTS_API_KEY が無い")
        rows: List[Dict[str, Any]] = []
        q = dict(params)
        while True:
            req = urllib.request.Request(f"{API}/{path}?{urllib.parse.urlencode(q)}", headers={"x-api-key": key})
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
            rows.extend(body.get("data", []))
            pkey = body.get("pagination_key")
            if not pkey:
                return rows
            q["pagination_key"] = pkey

    def path(self, kind: str, code: str) -> Path:
        return self.root / kind / f"{code}.json"

    def rows(self, kind: str, code: str) -> List[Dict[str, Any]]:
        with self._lock:
            if (kind, code) in self._mem:
                return self._mem[(kind, code)]
        p = self.path(kind, code)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
        elif self._fetch and self._api_key():
            data = self.download(kind, code)
        else:
            data = []
        with self._lock:
            self._mem[(kind, code)] = data
        return data

    def download(self, kind: str, code: str, **params: str) -> List[Dict[str, Any]]:
        path, extra = KINDS[kind]
        q = {"code": code, **extra, **{k: v for k, v in params.items() if v}}
        data = self.get_json(path, q)
        p = self.path(kind, code)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        with self._lock:
            self._mem[(kind, code)] = data
        return data


# ---------------------------------------------------------------- as-of の組み立て

# 財務諸表（/fins/details の FS）の科目。日本基準・IFRS・米国基準の順に探す（最初に値のあるもの）
GROSS_PROFIT = ("Gross profit (loss)", "Gross profit", "Gross profit (IFRS)", "Gross profit (US GAAP)")
CURRENT_ASSETS = ("Current assets", "Current assets (IFRS)", "Current assets (US GAAP)")
CURRENT_LIABILITIES = ("Current liabilities", "Current liabilities (IFRS)", "Current liabilities (US GAAP)")
# 長期の有利子負債は足し合わせる（日本基準は社債と長期借入金が別の科目）
LONG_TERM_DEBT_PARTS = (
    ("Bonds payable", "Long-term borrowings", "Long-term loans payable", "Convertible bond-type bonds with share acquisition rights"),
    ("Bonds and borrowings - NCL (IFRS)", "Borrowings - NCL (IFRS)", "Bonds - NCL (IFRS)", "Long-term debt - NCL (IFRS)",
     "Interest-bearing liabilities - NCL (IFRS)"),
    ("Long-term debt (US GAAP)",),
)
# 借入金の科目が無い期（無借金の会社、独自の科目の会社）は、全期を固定負債の合計で揃える
NON_CURRENT_LIABILITIES = ("Non-current liabilities", "Non-current liabilities (IFRS)", "Noncurrent liabilities (US GAAP)")
COST_OF_SALES = ("Cost of sales", "Cost of sales (IFRS)", "Cost of revenue (US GAAP)")
SHORT_TERM_DEBT_PARTS = (
    ("Short-term borrowings", "Short-term loans payable", "Current portion of bonds payable",
     "Current portion of long-term borrowings", "Current portion of long-term loans payable", "Commercial papers"),
    ("Bonds and borrowings - CL (IFRS)", "Borrowings - CL (IFRS)", "Interest-bearing liabilities - CL (IFRS)"),
)
CAPEX = ("Purchase of property, plant and equipment - InvCF", "Purchase of property, plant and equipment - InvCF (IFRS)",
         "Purchase of property, plant and equipment and intangible assets - InvCF (IFRS)")


def _fs_first(fs: Dict[str, Any], names) -> Optional[float]:
    for n in names:
        v = _num(fs.get(n))
        if v is not None:
            return v
    return None


def _fs_sum(fs: Dict[str, Any], groups) -> Optional[float]:
    """基準ごとの科目の組のうち、値のある最初の組を足す。どの組にも値が無ければ None。"""
    for names in groups:
        vals = [_num(fs.get(n)) for n in names]
        vals = [v for v in vals if v is not None]
        if vals:
            return float(sum(vals))
    return None


def _known(rows: List[Dict[str, Any]], asof: Optional[dt.date], key: str = "DiscDate") -> List[Dict[str, Any]]:
    if asof is None:
        return list(rows)
    return [r for r in rows if (_date(r.get(key)) or dt.date.max) <= asof]


def _fy_records(rows: List[Dict[str, Any]], asof: Optional[dt.date]) -> List[Dict[str, Any]]:
    """通期の決算短信の行を、決算期末ごとに最後に開示されたもの 1 つにして、古い順に返す。"""
    by_fy: Dict[str, Dict[str, Any]] = {}
    for r in sorted(_known(rows, asof), key=lambda r: (r.get("DiscDate", ""), r.get("DiscTime", ""), r.get("DiscNo", ""))):
        if not str(r.get("DocType", "")).startswith("FYFinancialStatements"):
            continue
        fye = r.get("CurFYEn") or (r.get("FS") or {}).get("Current fiscal year end date, DEI")
        if fye:
            by_fy[fye] = r
    return [by_fy[k] for k in sorted(by_fy)]


def _latest_record(rows: List[Dict[str, Any]], asof: Optional[dt.date]) -> Optional[Dict[str, Any]]:
    """決算短信（通期・四半期）のうち最後に開示された行。業績予想の修正などは除く。"""
    cands = [r for r in _known(rows, asof) if "FinancialStatements" in str(r.get("DocType", ""))]
    if not cands:
        return None
    return max(cands, key=lambda r: (r.get("DiscDate", ""), r.get("DiscTime", ""), r.get("DiscNo", "")))


def annual_statements(store: JQuantsStore, code: str, asof: Optional[dt.date]):
    """yfinance の ``financials`` / ``balance_sheet`` / ``cashflow`` と同じ形（行 = 科目、列 = 決算期末）の表。"""
    summ = {r.get("CurFYEn"): r for r in _fy_records(store.rows("fins_summary", code), asof)}
    det = {}
    for r in _fy_records(store.rows("fins_details", code), asof):
        fye = (r.get("FS") or {}).get("Current fiscal year end date, DEI")
        if fye:
            det[fye] = r.get("FS") or {}
    periods = sorted(set(summ) | set(det), reverse=True)  # 新しい期が左
    fin, bs, cf = {}, {}, {}
    for fye in periods:
        s = summ.get(fye, {})
        fs = det.get(fye, {})
        col = pd.Timestamp(fye)
        sales = _num(s.get("Sales"))
        gp = _fs_first(fs, GROSS_PROFIT)
        if gp is None and sales is not None and _fs_first(fs, COST_OF_SALES) is not None:
            gp = sales - _fs_first(fs, COST_OF_SALES)  # 売上総利益の行が無い IFRS の会社
        fin[col] = {
            "Total Revenue": sales,
            "Gross Profit": gp,
            "Operating Income": _num(s.get("OP")),
            "Net Income": _num(s.get("NP")),
        }
        sh_out, tr_sh = _num(s.get("ShOutFY")), _num(s.get("TrShFY"))
        bs[col] = {
            "Total Assets": _num(s.get("TA")),
            "Stockholders Equity": _num(s.get("ShEq")) or _num(s.get("Eq")),
            "Current Assets": _fs_first(fs, CURRENT_ASSETS),
            "Current Liabilities": _fs_first(fs, CURRENT_LIABILITIES),
            "Long Term Debt": _fs_sum(fs, LONG_TERM_DEBT_PARTS),
            "Non Current Liabilities": _fs_first(fs, NON_CURRENT_LIABILITIES),
            "Ordinary Shares Number": (sh_out - (tr_sh or 0.0)) if sh_out is not None else None,
            "Share Issued": sh_out,
        }
        capex = _fs_first(fs, CAPEX)
        cfo = _num(s.get("CFO"))
        cf[col] = {
            "Operating Cash Flow": cfo,
            "Investing Cash Flow": _num(s.get("CFI")),
            "Financing Cash Flow": _num(s.get("CFF")),
            "Capital Expenditure": capex,
            "Free Cash Flow": (cfo + capex) if (cfo is not None and capex is not None) else None,
        }
    # 長期の有利子負債は期どうしで同じ定義で比べる: 借入金の科目が全期にそろわなければ、全期を固定負債の合計にする
    if bs and any(v.get("Long Term Debt") is None for v in bs.values()):
        for v in bs.values():
            v["Long Term Debt"] = v.get("Non Current Liabilities")
            v["_long_term_debt_basis"] = "non_current_liabilities"
    for v in bs.values():
        v.pop("_long_term_debt_basis", None)
    to_df = lambda d: pd.DataFrame(d, dtype="float64") if d else pd.DataFrame()
    return to_df(fin), to_df(bs), to_df(cf)


def _same_period_growth(rows: List[Dict[str, Any]], asof: Optional[dt.date], field: str) -> Optional[float]:
    """最後の決算短信の累計の値と、前の年の同じ期の値の比（yfinance の revenueGrowth と同じく比率）。"""
    last = _latest_record(rows, asof)
    if not last:
        return None
    cur = _num(last.get(field))
    end = _date(last.get("CurPerEn"))
    if cur is None or end is None:
        return None
    prev = None
    for r in _known(rows, asof):
        if r.get("CurPerType") != last.get("CurPerType") or "FinancialStatements" not in str(r.get("DocType", "")):
            continue
        e = _date(r.get("CurPerEn"))
        if e and abs((end - e).days - 365) <= 20:
            prev = _num(r.get(field))
    if prev is None or prev == 0:
        return None
    return cur / prev - 1.0


def build_info(store: JQuantsStore, code: str, asof: Optional[dt.date], close: Optional[float]) -> Dict[str, Any]:
    """yfinance の ``info`` の形で、J-Quants から作れる項目を返す（価格は呼ぶ側が日足から足す）。"""
    info: Dict[str, Any] = {"currency": "JPY", "quoteType": "EQUITY", "_source": "J-Quants"}
    master = [r for r in store.rows("master", code) if asof is None or (_date(r.get("Date")) or dt.date.min) <= asof + dt.timedelta(days=3)]
    if master:
        m = max(master, key=lambda r: r.get("Date", ""))
        info.update({"longName": m.get("CoName"), "shortName": m.get("CoName"), "longNameEn": m.get("CoNameEn"),
                     "sector": m.get("S17Nm"), "industry": m.get("S33Nm"), "market": m.get("MktNm"),
                     "scaleCategory": m.get("ScaleCat")})
    val = [r for r in store.rows("valuation", code) if asof is None or (_date(r.get("Date")) or dt.date.max) <= asof]
    if val:
        v = max(val, key=lambda r: r.get("Date", ""))
        mc = _num(v.get("MktCap"))
        info.update({
            "trailingEps": _num(v.get("EPS")), "forwardEps": _num(v.get("FwdEPS")), "bookValue": _num(v.get("BPS")),
            "trailingPE": _num(v.get("PER")), "forwardPE": _num(v.get("FwdPER")), "priceToBook": _num(v.get("PBR")),
            "returnOnEquity": _num(v.get("ROE")), "marketCap": mc * 1e6 if mc is not None else None,
            "_valuation_date": v.get("Date"),
        })
    summ_rows = store.rows("fins_summary", code)
    fy = _fy_records(summ_rows, asof)
    last = _latest_record(summ_rows, asof)
    if fy:
        f = fy[-1]
        sales, op, np_, ta, cfo = (_num(f.get(k)) for k in ("Sales", "OP", "NP", "TA", "CFO"))
        if sales:
            info["operatingMargins"] = op / sales if op is not None else None
            info["profitMargins"] = np_ / sales if np_ is not None else None
            if info.get("marketCap"):
                info["priceToSalesTrailing12Months"] = info["marketCap"] / sales
        if ta and np_ is not None:
            info["returnOnAssets"] = np_ / ta
        info["operatingCashflow"] = cfo
        info["totalRevenue"] = sales
        info["netIncomeToCommon"] = np_
    if last:
        div = _num(last.get("FDivAnn")) or _num(last.get("NxFDivAnn")) or _num(last.get("DivAnn"))
        if div is not None and close:
            info["dividendYield"] = div / close  # 比率（フォークは ×100 して % にする）
            info["trailingAnnualDividendYield"] = info["dividendYield"]
            info["dividendRate"] = div
        info["_latest_disclosure"] = f"{last.get('DiscDate')} {last.get('DocType')}"
    info["revenueGrowth"] = _same_period_growth(summ_rows, asof, "Sales")
    info["earningsGrowth"] = _same_period_growth(summ_rows, asof, "NP")
    det_last = _latest_record(store.rows("fins_details", code), asof)
    if det_last:
        fs = det_last.get("FS") or {}
        lt, st = _fs_sum(fs, LONG_TERM_DEBT_PARTS), _fs_sum(fs, SHORT_TERM_DEBT_PARTS)
        if lt is not None or st is not None:
            info["totalDebt"] = (lt or 0.0) + (st or 0.0)
        ca, cl = _fs_first(fs, CURRENT_ASSETS), _fs_first(fs, CURRENT_LIABILITIES)
        if ca and cl:
            info["currentRatio"] = ca / cl
        eq = _num((last or {}).get("ShEq")) or _num((last or {}).get("Eq"))
        if info.get("totalDebt") is not None and eq:
            info["debtToEquity"] = info["totalDebt"] / eq * 100.0  # yfinance と同じく %
    _, _, cf = annual_statements(store, code, asof)
    if not cf.empty and "Free Cash Flow" in cf.index:
        fcf = cf.iloc[:, 0].get("Free Cash Flow")
        info["freeCashflow"] = None if pd.isna(fcf) else float(fcf)
    return {k: v for k, v in info.items() if v is not None}


def earnings_calendar(store: JQuantsStore, code: str, asof: Optional[dt.date]) -> Dict[str, Any]:
    """yfinance の ``calendar`` の形 ``{"Earnings Date": [date]}``。asof までに公表された予定のうち、asof より後の最も近い日。"""
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in sorted(_known(store.rows("earnings_date", code), asof, key="PubDate"), key=lambda r: r.get("PubDate", "")):
        latest[(r.get("FQName", ""), r.get("FYE", ""))] = r
    today = asof or dt.date.today()
    upcoming = sorted(d for d in (_date(r.get("SchDate")) for r in latest.values()) if d and d > today)
    return {"Earnings Date": upcoming[:1]} if upcoming else {}


def dividends(store: JQuantsStore, code: str, asof: Optional[dt.date]) -> pd.Series:
    """決定した配当を権利落日で並べた Series（yfinance の ``dividends`` と同じ形）。"""
    voided = {r.get("CARefNo") for r in store.rows("dividend", code) if str(r.get("StatCode")) == "3"}
    by_ref: Dict[str, Dict[str, Any]] = {}
    for r in sorted(_known(store.rows("dividend", code), asof, key="PubDate"), key=lambda r: (r.get("PubDate", ""), r.get("PubTime", ""))):
        if str(r.get("FRCode")) != "1" or str(r.get("StatCode")) == "3":
            continue
        by_ref[r.get("CARefNo") or r.get("RefNo")] = r
    pts = {}
    for ref, r in by_ref.items():
        ex, amt = _date(r.get("ExDate")), _num(r.get("DivRate"))
        if ref in voided or ex is None or amt is None or (asof and ex > asof):
            continue
        pts[pd.Timestamp(ex).tz_localize("Asia/Tokyo")] = amt
    s = pd.Series(pts, dtype="float64", name="Dividends").sort_index()
    s.index.name = "Date"
    return s


def splits_from_bars(rows: List[Dict[str, Any]], asof: Optional[dt.date]) -> pd.Series:
    """日足の ``AdjFactor``（例: 1:5 の分割なら 0.2）から、yfinance の ``splits``（5.0）の形を作る。"""
    pts = {}
    for r in rows:
        f = _num(r.get("AdjFactor"))
        d = _date(r.get("Date"))
        if f and d and abs(f - 1.0) > 1e-9 and (asof is None or d <= asof):
            pts[pd.Timestamp(d).tz_localize("Asia/Tokyo")] = 1.0 / f
    s = pd.Series(pts, dtype="float64", name="Stock Splits").sort_index()
    s.index.name = "Date"
    return s


def fetch_fundamentals(store: JQuantsStore, codes: List[str], asof: dt.date, log: Callable[[str], None] = print) -> None:
    """財務などを銘柄ごとに取り直して控える（scripts/jq_fetch.py --fundamentals）。"""
    frm = (asof - dt.timedelta(days=400)).isoformat()
    for code in codes:
        n = {}
        n["master"] = len(store.download("master", code, date=asof.isoformat()))
        n["valuation"] = len(store.download("valuation", code, **{"from": frm, "to": asof.isoformat()}))
        for kind in ("fins_summary", "fins_details", "earnings_date", "dividend"):
            n[kind] = len(store.download(kind, code))
            time.sleep(0.2)
        log(f"{code}  " + "  ".join(f"{k} {v}" for k, v in n.items()))


# ---------------------------------------------------------------- yfinance の Ticker の代わり

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
INTRADAY = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


def bars_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """``/equities/bars/daily`` の行を yfinance の日足の形にする（分割の調整済み、東京時間 0 時の tz 付き）。"""
    recs = []
    for r in rows:
        if r.get("AdjC") is None:
            continue  # 売買の無い日
        recs.append({"Date": pd.Timestamp(r["Date"]).tz_localize("Asia/Tokyo"),
                     "Open": float(r["AdjO"]), "High": float(r["AdjH"]), "Low": float(r["AdjL"]),
                     "Close": float(r["AdjC"]), "Volume": float(r.get("AdjVo") or 0.0)})
    if not recs:
        return empty_frame()
    df = pd.DataFrame.from_records(recs).set_index("Date").sort_index()
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0
    return df


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


class JQuantsTicker:
    """東証の銘柄について、yfinance の ``Ticker`` の代わりに J-Quants の値を返す。

    J-Quants に無い属性（``news``、``insider_*`` など）は、``real_factory`` で作った yfinance の Ticker に回す。
    """

    def __init__(self, symbol: str, store: JQuantsStore, asof: Optional[dt.date] = None,
                 real_factory: Optional[Callable[[str], Any]] = None):
        self._symbol = str(symbol)
        self._code = jq_code(self._symbol)
        if self._code is None:
            raise ValueError(f"東証の銘柄ではない: {symbol}")
        self._store = store
        self._asof = asof
        self._real_factory = real_factory
        self._real = None
        self._info: Optional[Dict[str, Any]] = None

    @property
    def ticker(self) -> str:
        return self._symbol

    def _real_ticker(self):
        if self._real is None:
            if self._real_factory is None:
                import yfinance as yf
                self._real_factory = yf.Ticker
            self._real = self._real_factory(self._symbol)
        return self._real

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._real_ticker(), name)

    def _bars(self) -> pd.DataFrame:
        return bars_frame(self._store.rows("daily", self._code))

    def history(self, period: Optional[str] = "1mo", interval: str = "1d", start=None, end=None, **kwargs) -> pd.DataFrame:
        if interval in INTRADAY:
            return empty_frame()  # J-Quants の日足しか持たない
        daily = slice_bars(self._bars(), self._asof, period, start, end).copy()
        return daily if interval in ("1d", "1D") else resample_bars(daily, interval)

    @property
    def info(self) -> Dict[str, Any]:
        if self._info is None:
            bars = slice_bars(self._bars(), self._asof, None)
            px = price_fields(bars)
            info = build_info(self._store, self._code, self._asof, px.get("currentPrice"))
            info.update(px)
            info["symbol"] = self._symbol
            if self._asof:
                info["_asof"] = self._asof.isoformat()
            self._info = info
        return self._info

    @property
    def fast_info(self) -> Dict[str, Any]:
        inf = self.info
        return {"last_price": inf.get("currentPrice"), "lastPrice": inf.get("currentPrice"),
                "previous_close": inf.get("previousClose"), "previousClose": inf.get("previousClose"),
                "currency": "JPY", "market_cap": inf.get("marketCap")}

    def _statements(self):
        return annual_statements(self._store, self._code, self._asof)

    @property
    def financials(self) -> pd.DataFrame:
        return self._statements()[0]

    income_stmt = financials

    @property
    def balance_sheet(self) -> pd.DataFrame:
        return self._statements()[1]

    balancesheet = balance_sheet

    @property
    def cashflow(self) -> pd.DataFrame:
        return self._statements()[2]

    cash_flow = cashflow

    @property
    def calendar(self) -> Dict[str, Any]:
        return earnings_calendar(self._store, self._code, self._asof)

    @property
    def dividends(self) -> pd.Series:
        return dividends(self._store, self._code, self._asof)

    @property
    def splits(self) -> pd.Series:
        return splits_from_bars(self._store.rows("daily", self._code), self._asof)

    @property
    def actions(self) -> pd.DataFrame:
        return pd.concat([self.dividends, self.splits], axis=1).fillna(0.0)


def install_yfinance_router(root: Path) -> None:
    """同じプロセスの ``yfinance.Ticker`` を、東証の銘柄は J-Quants、それ以外は yfinance に振り分けるものにする（いまの値）。

    ``LUMINA_JP_DATA_SOURCE=jquants`` のときに scripts/trial_cycle.py が呼ぶ。
    """
    import yfinance as yf

    if getattr(yf.Ticker, "_jquants_router", False):
        return
    real = yf.Ticker
    store = JQuantsStore(Path(root))

    def _ticker(symbol, *args, **kwargs):
        if jq_code(str(symbol)) is not None:
            return JQuantsTicker(str(symbol), store, None, real_factory=real)
        return real(symbol, *args, **kwargs)

    _ticker._jquants_router = True
    yf.Ticker = _ticker
