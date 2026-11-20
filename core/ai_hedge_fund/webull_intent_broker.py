# -*- coding: utf-8 -*-
"""
WebullIntentBroker: Webull 証券（米国株の信用口座）向けの「発注の意図」を書き出すブローカー。
WebullJpCashIntentBroker: ウィブル証券の日本株・現物の口座向けの同じもの（ファイルの末尾）。

このクラスは Webull の API を一切呼ばない。発注を送るのは、別のプロセス（執行機）の仕事である。
ここがやるのは次の 3 つだけ:

1. 米国株以外、立会の外、寄付の直後の新規を拒む。
2. 受け付けた注文を、意図（intent）として JSONL に 1 行書く。
3. 台帳（PaperTradingEngine の SQLite）に約定を記帳する。
   - mode="paper": すぐに、参考価格で記帳する（執行機が動いていない段階。紙の上だけ）。
   - mode="await": 執行機が書く結果ファイルを待ち、約定したときだけ、その約定値で記帳する。

意図の形（1 行 1 JSON、schema = "lumina-intent/1"）:
    intent_id, created_at, valid_until（UTC の ISO 8601）, session_date_et,
    symbol, side（BUY/SELL）, intent（OPEN/CLOSE）, qty,
    limit_price, reference_price（文字列の 10 進数）, reference_source,
    stop_loss_price, target_price, strategy_type, reason, calendar

結果の形（執行機が書く。schema = "lumina-intent-result/1"）:
    intent_id, status（FILLED / PARTIAL / DENIED / UNFILLED / ERROR）, filled_qty, avg_price, reasons
"""
import datetime
import json
import os
import threading
import time
import uuid
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from core.ai_hedge_fund.broker_client import PaperTradingEngine
from core.ai_hedge_fund.market_data_feed import MarketDataFeed

ET = ZoneInfo("America/New_York")
UTC = datetime.timezone.utc

INTENT_SCHEMA = "lumina-intent/1"
RESULT_SCHEMA = "lumina-intent-result/1"

# すぐ約定させるための指値の幅。買いは参考価格 +0.2%、売りは -0.2%。
MARKETABLE_PCT = Decimal("0.002")
CENT = Decimal("0.01")
# 寄付からこの秒数が過ぎるまで、新規は出さない。
NO_ENTRY_AFTER_OPEN_SEC = 300
# 意図の有効期限（作ってからの秒数）。過ぎた意図を執行機は捨てる。
INTENT_TTL_SEC = 120

OPEN_TIME = datetime.time(9, 30)
CLOSE_TIME = datetime.time(16, 0)
HALF_CLOSE_TIME = datetime.time(13, 0)

FILLED_STATUSES = ("FILLED", "PARTIAL")
TERMINAL_STATUSES = ("FILLED", "PARTIAL", "DENIED", "UNFILLED", "ERROR")


class UsSessionCalendar:
    """
    米国の立会の時間。calendar_dir に nyse-calendar-<年>.toml があれば休業日と半日立会を読む。
    無ければ「平日 09:30〜16:00 ET」だけで判定し、source を "weekday_only" とする
    （休業日を知らないので、最終的な判定は執行機の暦に任せる）。
    """

    def __init__(self, calendar_dir: Optional[str] = None):
        self._dir = Path(calendar_dir) if calendar_dir else None
        self._years: Dict[int, Tuple[set, set]] = {}

    def _year(self, year: int) -> Optional[Tuple[set, set]]:
        if self._dir is None:
            return None
        if year not in self._years:
            path = self._dir / f"nyse-calendar-{year}.toml"
            if not path.exists():
                return None
            try:
                import tomllib
            except ImportError:  # Python 3.10
                return None
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            self._years[year] = (
                {datetime.date.fromisoformat(h["date"]) for h in raw.get("holiday", [])},
                {datetime.date.fromisoformat(h["date"]) for h in raw.get("half_day", [])},
            )
        return self._years[year]

    def session(self, now_utc: datetime.datetime) -> Dict[str, Any]:
        """now の ET の日付の立会。{"date", "open", "close", "is_open", "source"}。休みの日は open/close が None。"""
        now_et = now_utc.astimezone(ET)
        day = now_et.date()
        year = self._year(day.year)
        source = "nyse_calendar" if year is not None else "weekday_only"
        holidays, half_days = year if year is not None else (set(), set())
        if day.weekday() >= 5 or day in holidays:
            return {"date": day, "open": None, "close": None, "is_open": False, "source": source}
        open_dt = datetime.datetime.combine(day, OPEN_TIME, tzinfo=ET)
        close_dt = datetime.datetime.combine(day, HALF_CLOSE_TIME if day in half_days else CLOSE_TIME, tzinfo=ET)
        return {"date": day, "open": open_dt, "close": close_dt, "is_open": open_dt <= now_et < close_dt, "source": source}


def is_us_symbol(norm_sym: str) -> bool:
    """接尾辞（.T、.AS など）の無い銘柄だけを米国株として扱う。"""
    return bool(norm_sym) and "." not in norm_sym and not norm_sym.startswith("^") and "=" not in norm_sym


def marketable_limit(side: str, reference_price: float) -> Decimal:
    """買いは参考価格 +0.2% を切り上げ、売りは -0.2% を切り下げ（セント単位）。"""
    ref = Decimal(str(reference_price))
    if side == "BUY":
        return (ref * (1 + MARKETABLE_PCT)).quantize(CENT, rounding=ROUND_UP)
    return (ref * (1 - MARKETABLE_PCT)).quantize(CENT, rounding=ROUND_DOWN)


class WebullIntentBroker(PaperTradingEngine):
    """PaperTradingEngine の台帳をそのまま使い、発注の入口だけを差し替える。"""

    us_only = True  # 呼ぶ側（AutoTrader）は、これを見て候補を米国株に絞る

    def __init__(
        self,
        db_path: Optional[str] = None,
        intent_dir: Optional[str] = None,
        mode: Optional[str] = None,
        initial_usd: Optional[float] = None,
        calendar_dir: Optional[str] = None,
        await_timeout_sec: Optional[float] = None,
        now_fn: Optional[Callable[[], datetime.datetime]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        super().__init__(db_path=db_path)
        self.mode = (mode or os.getenv("LUMINA_WEBULL_FILL_MODE", "paper")).lower()
        if self.mode not in ("paper", "await"):
            raise ValueError(f"LUMINA_WEBULL_FILL_MODE は paper か await: {self.mode!r}")
        intent_dir = intent_dir or os.getenv("LUMINA_WEBULL_INTENT_DIR")
        if not intent_dir:
            raise ValueError("意図の書き出し先（LUMINA_WEBULL_INTENT_DIR）が設定されていない")
        self.intent_dir = Path(intent_dir)
        self.intent_dir.mkdir(parents=True, exist_ok=True)
        self.calendar = UsSessionCalendar(calendar_dir or os.getenv("LUMINA_NYSE_CALENDAR_DIR"))
        self.await_timeout_sec = float(
            await_timeout_sec if await_timeout_sec is not None else os.getenv("LUMINA_WEBULL_AWAIT_TIMEOUT_SEC", "60")
        )
        self._now = now_fn or (lambda: datetime.datetime.now(UTC))
        self._sleep = sleep_fn
        self._write_lock = threading.Lock()
        usd = initial_usd if initial_usd is not None else float(os.getenv("LUMINA_WEBULL_INITIAL_USD", "0") or 0)
        if usd > 0:
            self._seed_usd_if_fresh(usd)

    # ------------------------------------------------------------------
    # 台帳の初期化
    # ------------------------------------------------------------------
    def _seed_usd_if_fresh(self, initial_usd: float) -> None:
        """作ったばかりの台帳（既定の 100 万円、注文なし）なら、米ドルだけの元手に置き換える。"""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM orders")
            n_orders = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM positions")
            n_pos = cur.fetchone()[0]
            cur.execute("SELECT currency, cash_balance FROM accounts")
            cash = {r[0]: float(r[1]) for r in cur.fetchall()}
        if n_orders == 0 and n_pos == 0 and cash.get("JPY") == 1000000.0 and cash.get("USD", 0.0) == 0.0:
            self.reset_portfolio(initial_jpy=0.0, initial_usd=float(initial_usd))

    # ------------------------------------------------------------------
    # ファイル
    # ------------------------------------------------------------------
    def intent_path(self, session_date: datetime.date) -> Path:
        return self.intent_dir / f"intents-{session_date.isoformat()}.jsonl"

    def result_path(self, session_date: datetime.date) -> Path:
        return self.intent_dir / f"results-{session_date.isoformat()}.jsonl"

    def _append_intent(self, intent: Dict[str, Any], session_date: datetime.date) -> None:
        line = json.dumps(intent, ensure_ascii=False, sort_keys=True)
        with self._write_lock:
            with open(self.intent_path(session_date), "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _read_result(self, intent_id: str, session_date: datetime.date) -> Optional[Dict[str, Any]]:
        path = self.result_path(session_date)
        if not path.exists():
            return None
        found = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # 書きかけの行
                if row.get("schema") == RESULT_SCHEMA and row.get("intent_id") == intent_id:
                    if str(row.get("status", "")).upper() in TERMINAL_STATUSES:
                        found = row  # 同じ意図の行が複数あれば、最後の終了状態を採る
        return found

    # ------------------------------------------------------------------
    # 発注の入口
    # ------------------------------------------------------------------
    @staticmethod
    def _refuse(message: str, **extra: Any) -> Dict[str, Any]:
        return {"success": False, "error": message, "message": message, "intent_written": False, **extra}

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "MARKET",
        notes: str = "",
        target_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        enforce_market_hours: bool = True,
        strategy_type: str = "SWING",
        signal_timestamp: Optional[float] = None,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        norm_sym = MarketDataFeed.normalize_symbol(symbol)
        side = str(side).upper()
        if side not in ("BUY", "SELL"):
            return self._refuse(f"売買の向きが不正: {side}")
        if not is_us_symbol(norm_sym):
            return self._refuse(f"【Webull】米国株以外は扱わない: {norm_sym}")
        try:
            qty = int(quantity)
        except (TypeError, ValueError):
            return self._refuse(f"数量が不正: {quantity!r}")
        if qty <= 0 or qty != quantity:
            return self._refuse(f"数量は 1 以上の整数（端株は扱わない）: {quantity!r}")

        # 立会の外では出さない。enforce_market_hours=False でも緩めない（Webull は立会の外の注文を次の寄付まで持ち越す）。
        now = self._now()
        sess = self.calendar.session(now)
        if not sess["is_open"]:
            return self._refuse("【Webull】米国の立会時間の外。意図を書かない", outside_session=True)

        held = {p["symbol"]: p for p in self.get_positions()}
        if side == "BUY":
            intent_kind = "OPEN"
            if (now - sess["open"]).total_seconds() < NO_ENTRY_AFTER_OPEN_SEC:
                return self._refuse(f"【Webull】寄付から {NO_ENTRY_AFTER_OPEN_SEC} 秒は新規を出さない", too_early=True)
        else:
            intent_kind = "CLOSE"
            have = int(held.get(norm_sym, {}).get("quantity", 0) or 0)
            if have <= 0:
                return self._refuse(f"【Webull】台帳に {norm_sym} の建玉が無い。空売りはしない")
            if qty > have:
                return self._refuse(f"【Webull】売りの数量 {qty} が台帳の建玉 {have} を超える")

        ref_price = float(price) if price and price > 0 else None
        ref_source = "caller"
        if ref_price is None:
            quote = MarketDataFeed.get_stock_data(norm_sym)
            ref_price = quote.get("current_price")
            ref_source = "yfinance"
        if not ref_price or ref_price <= 0:
            return self._refuse(f"【Webull】{norm_sym} の参考価格が取れない。意図を書かない")
        limit = marketable_limit(side, ref_price)

        if side == "BUY":
            usd_cash = float(self.get_account_balance().get("USD", 0.0))
            need = float(limit) * qty
            if need > usd_cash:
                return self._refuse(f"【Webull】米ドルの余力が足りない（必要 ${need:,.2f} / 余力 ${usd_cash:,.2f}）")

        # 同じ銘柄・同じ向きの連続発注と、古いシグナルは、意図を書く前に止める
        from core.ai_hedge_fund.idempotency_guard import OrderIdempotencyGuard
        guard = OrderIdempotencyGuard(db_path=self.db_path)
        allowed, guard_msg, order_ref = guard.check_and_reserve(
            symbol=norm_sym, side=side, quantity=qty, order_type="LIMIT",
            signal_timestamp=signal_timestamp, strategy_type=strategy_type,
        )
        if not allowed:
            return self._refuse(guard_msg, stale_discarded=("シグナル鮮度ガード" in guard_msg), idempotency_blocked=True)

        intent_id = f"lumina-{sess['date']:%Y%m%d}-{uuid.uuid4().hex[:12]}"
        intent = {
            "schema": INTENT_SCHEMA,
            "intent_id": intent_id,
            "created_at": now.astimezone(UTC).isoformat(),
            "valid_until": (now + datetime.timedelta(seconds=INTENT_TTL_SEC)).astimezone(UTC).isoformat(),
            "session_date_et": sess["date"].isoformat(),
            "calendar": sess["source"],
            "symbol": norm_sym,
            "side": side,
            "intent": intent_kind,
            "qty": qty,
            "limit_price": str(limit),
            "reference_price": str(Decimal(str(ref_price))),
            "reference_source": ref_source,
            "stop_loss_price": None if not stop_loss_price else str(Decimal(str(stop_loss_price))),
            "target_price": None if not target_price else str(Decimal(str(target_price))),
            "strategy_type": strategy_type,
            "reason": notes,
            "fill_mode": self.mode,
            "lumina_order_ref": order_ref,
        }
        try:
            self._append_intent(intent, sess["date"])
        except OSError as e:
            guard.record_completed(order_ref, is_success=False, error_message=f"意図を書けなかった: {e}")
            return self._refuse(f"【Webull】意図を書けなかった: {e}")

        if self.mode == "paper":
            fill_price, fill_qty, result = ref_price, qty, None
        else:
            result = self._await_result(intent_id, sess["date"])
            status = str((result or {}).get("status", "")).upper()
            if result is None:
                # 予約は PENDING のまま残す（注文が生きているかもしれないので、同じ銘柄・向きの出し直しを止める）
                return {
                    "success": False, "unconfirmed": True, "intent_written": True, "intent_id": intent_id,
                    "error": f"【Webull】執行機の結果が {self.await_timeout_sec:.0f} 秒以内に来なかった。台帳には記帳していない。照合するまで出し直さないこと",
                    "message": "執行機の結果が未確認",
                }
            if status not in FILLED_STATUSES:
                reasons = result.get("reasons") or []
                guard.record_completed(order_ref, is_success=False, error_message=f"{status}: {reasons}")
                return {
                    "success": False, "intent_written": True, "intent_id": intent_id, "status": status,
                    "error": f"【Webull】執行機が約定させなかった（{status}）: {'; '.join(map(str, reasons)) or '理由の記載なし'}",
                    "message": f"執行機の結果: {status}", "reasons": reasons,
                }
            try:
                fill_qty = int(result["filled_qty"])
                fill_price = float(Decimal(str(result["avg_price"])))
            except (KeyError, TypeError, ValueError, ArithmeticError) as e:
                return {
                    "success": False, "unconfirmed": True, "intent_written": True, "intent_id": intent_id,
                    "error": f"【Webull】執行機の結果を読めなかった（{e}）。台帳には記帳していない", "message": "結果の形が不正",
                }
            if fill_qty <= 0 or fill_qty > qty or fill_price <= 0:
                return {
                    "success": False, "unconfirmed": True, "intent_written": True, "intent_id": intent_id,
                    "error": "【Webull】執行機の結果の数量か値段が不正。台帳には記帳していない", "message": "結果の値が不正",
                }

        # 意図を書く前の予約は、ここで手放す（状態を「不成立」にすると、次の予約を妨げない）。
        # 台帳への記帳は親クラスの place_order に任せ、親が自分の予約と約定の記録を作る。
        guard.record_completed(order_ref, is_success=False, error_message=f"台帳の記帳へ引き継ぎ（意図 {intent_id}）")
        booked = self._book_fill(
            norm_sym, side, fill_qty, fill_price, f"{notes} [Webull意図 {intent_id} / {self.mode}]".strip(),
            target_price, stop_loss_price, strategy_type, expected_version,
        )
        booked.update({
            "intent_written": True, "intent_id": intent_id, "fill_mode": self.mode,
            "limit_price": float(limit), "filled_quantity": fill_qty, "fill_price": fill_price,
        })
        if not booked.get("success"):
            # 約定は起きている（await のとき）のに記帳できなかった。握りつぶさず、照合が要ることを返す。
            booked["ledger_mismatch"] = self.mode == "await"
        return booked

    def _await_result(self, intent_id: str, session_date: datetime.date) -> Optional[Dict[str, Any]]:
        waited = 0.0
        step = 0.5
        while True:
            row = self._read_result(intent_id, session_date)
            if row is not None:
                return row
            if waited >= self.await_timeout_sec:
                return None
            self._sleep(step)
            waited += step

    def _book_fill(self, norm_sym, side, qty, price, notes, target_price, stop_loss_price, strategy_type, expected_version):
        """台帳に記帳する（親クラスのペーパーの処理）。立会の判定は済んでいるので、親には判定させない。"""
        return super().place_order(
            norm_sym, side, qty, price, order_type="LIMIT", notes=notes,
            target_price=target_price, stop_loss_price=stop_loss_price,
            enforce_market_hours=False, strategy_type=strategy_type,
            signal_timestamp=None, expected_version=expected_version,
        )


JST = ZoneInfo("Asia/Tokyo")
JP_LOT = 100


class WebullJpCashIntentBroker(PaperTradingEngine):
    """
    ウィブル証券の日本株・現物の口座向けに、「発注の意図」を JSONL に書くブローカー（Webull の API は呼ばない）。

    規則:
    - 東証の銘柄（``.T``）だけ。自律サイクルは ``jp_only`` を見て候補を日本株に絞る
    - 現物だけ: 買いは円の現金の範囲（信用・両替はしない）、売りは台帳の建玉の範囲（空売りはしない）
    - 100 株単位（単元未満は扱わない）
    - 東証の立会（平日 09:00〜11:30・12:30〜15:30、休業日はフォークの ``MarketSession``）の外では書かない

    意図は ``lumina-intent/1`` に ``market`` = ``TSE``、``account`` = ``cash``、``currency`` = ``JPY``、
    ``order_type`` = ``MARKET`` を足したもの。指値は付けない（値段の決め方は執行機の仕事）。
    台帳への記帳は ``mode`` = ``paper`` なら参考価格ですぐ、``await`` なら執行機の結果を待つ（米国株の版と同じ）。
    """

    jp_only = True

    def __init__(
        self,
        db_path: Optional[str] = None,
        intent_dir: Optional[str] = None,
        mode: Optional[str] = None,
        initial_jpy: Optional[float] = None,
        await_timeout_sec: Optional[float] = None,
        now_fn: Optional[Callable[[], datetime.datetime]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        super().__init__(db_path=db_path)
        self.mode = (mode or os.getenv("LUMINA_WEBULL_FILL_MODE", "paper")).lower()
        if self.mode not in ("paper", "await"):
            raise ValueError(f"LUMINA_WEBULL_FILL_MODE は paper か await: {self.mode!r}")
        intent_dir = intent_dir or os.getenv("LUMINA_WEBULL_INTENT_DIR")
        if not intent_dir:
            raise ValueError("意図の書き出し先（LUMINA_WEBULL_INTENT_DIR）が設定されていない")
        self.intent_dir = Path(intent_dir)
        self.intent_dir.mkdir(parents=True, exist_ok=True)
        self.await_timeout_sec = float(
            await_timeout_sec if await_timeout_sec is not None else os.getenv("LUMINA_WEBULL_AWAIT_TIMEOUT_SEC", "60")
        )
        self._now = now_fn or (lambda: datetime.datetime.now(UTC))
        self._sleep = sleep_fn
        self._write_lock = threading.Lock()
        jpy = initial_jpy if initial_jpy is not None else float(os.getenv("LUMINA_WEBULL_INITIAL_JPY", "0") or 0)
        if jpy > 0:
            self._seed_jpy_if_fresh(jpy)

    def _seed_jpy_if_fresh(self, initial_jpy: float) -> None:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM orders")
            n_orders = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM positions")
            n_pos = cur.fetchone()[0]
        if n_orders == 0 and n_pos == 0:
            self.reset_portfolio(initial_jpy=float(initial_jpy), initial_usd=0.0)

    def intent_path(self, session_date: datetime.date) -> Path:
        return self.intent_dir / f"intents-jp-{session_date.isoformat()}.jsonl"

    def result_path(self, session_date: datetime.date) -> Path:
        return self.intent_dir / f"results-jp-{session_date.isoformat()}.jsonl"

    _append_intent = WebullIntentBroker._append_intent
    _read_result = WebullIntentBroker._read_result
    _await_result = WebullIntentBroker._await_result
    _refuse = staticmethod(WebullIntentBroker._refuse)

    def _book_fill(self, norm_sym, side, qty, price, notes, target_price, stop_loss_price, strategy_type, expected_version):
        """台帳に記帳する（親クラスのペーパーの処理）。立会の判定は済んでいるので、親には判定させない。"""
        return PaperTradingEngine.place_order(
            self, norm_sym, side, qty, price, order_type="MARKET", notes=notes,
            target_price=target_price, stop_loss_price=stop_loss_price,
            enforce_market_hours=False, strategy_type=strategy_type,
            signal_timestamp=None, expected_version=expected_version,
        )

    def convert_currency(self, *args, **kwargs) -> Dict[str, Any]:
        return {"success": False, "error": "【ウィブル証券・日本株現物】両替はしない（円の現金だけで買う）"}

    @staticmethod
    def _session_open(now_jst: datetime.datetime) -> bool:
        from core.ai_hedge_fund.market_session import MarketSession
        return bool(MarketSession.is_tse_open(now_jst))

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "MARKET",
        notes: str = "",
        target_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        enforce_market_hours: bool = True,
        strategy_type: str = "SWING",
        signal_timestamp: Optional[float] = None,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        norm_sym = MarketDataFeed.normalize_symbol(symbol)
        side = str(side).upper()
        if side not in ("BUY", "SELL"):
            return self._refuse(f"売買の向きが不正: {side}")
        if not norm_sym.endswith(".T"):
            return self._refuse(f"【ウィブル証券・日本株現物】東証の銘柄以外は扱わない: {norm_sym}")
        try:
            qty = int(quantity)
        except (TypeError, ValueError):
            return self._refuse(f"数量が不正: {quantity!r}")
        if qty <= 0 or qty != quantity or qty % JP_LOT != 0:
            return self._refuse(f"【ウィブル証券・日本株現物】数量は {JP_LOT} 株単位（単元未満は扱わない）: {quantity!r}")

        # 立会の外では書かない。enforce_market_hours=False でも緩めない（緩めると寄付前の注文の扱いが執行機ごとに違う）。
        now = self._now()
        now_jst = now.astimezone(JST)
        if not self._session_open(now_jst):
            return self._refuse("【ウィブル証券・日本株現物】東証の立会時間の外。意図を書かない", outside_session=True)
        session_date = now_jst.date()

        held = {p["symbol"]: p for p in self.get_positions()}
        if side == "BUY":
            intent_kind = "OPEN"
        else:
            intent_kind = "CLOSE"
            have = int(held.get(norm_sym, {}).get("quantity", 0) or 0)
            if have <= 0:
                return self._refuse(f"【ウィブル証券・日本株現物】台帳に {norm_sym} の建玉が無い。空売りはしない（現物）")
            if qty > have:
                return self._refuse(f"【ウィブル証券・日本株現物】売りの数量 {qty} が台帳の建玉 {have} を超える")

        ref_price = float(price) if price and price > 0 else None
        ref_source = "caller"
        if ref_price is None:
            quote = MarketDataFeed.get_stock_data(norm_sym)
            ref_price = quote.get("current_price")
            ref_source = "market_data_feed"
        if not ref_price or ref_price <= 0:
            return self._refuse(f"【ウィブル証券・日本株現物】{norm_sym} の参考価格が取れない。意図を書かない")

        if side == "BUY":
            jpy_cash = float(self.get_account_balance().get("JPY", 0.0))
            need = ref_price * qty
            if need > jpy_cash:
                return self._refuse(f"【ウィブル証券・日本株現物】円の現金が足りない（必要 ¥{need:,.0f} / 現金 ¥{jpy_cash:,.0f}）。信用は使わない")

        from core.ai_hedge_fund.idempotency_guard import OrderIdempotencyGuard
        guard = OrderIdempotencyGuard(db_path=self.db_path)
        allowed, guard_msg, order_ref = guard.check_and_reserve(
            symbol=norm_sym, side=side, quantity=qty, order_type="MARKET",
            signal_timestamp=signal_timestamp, strategy_type=strategy_type,
        )
        if not allowed:
            return self._refuse(guard_msg, stale_discarded=("シグナル鮮度ガード" in guard_msg), idempotency_blocked=True)

        intent_id = f"lumina-jp-{session_date:%Y%m%d}-{uuid.uuid4().hex[:12]}"
        intent = {
            "schema": INTENT_SCHEMA,
            "intent_id": intent_id,
            "created_at": now.astimezone(UTC).isoformat(),
            "valid_until": (now + datetime.timedelta(seconds=INTENT_TTL_SEC)).astimezone(UTC).isoformat(),
            "session_date_jst": session_date.isoformat(),
            "market": "TSE",
            "account": "cash",
            "currency": "JPY",
            "symbol": norm_sym,
            "side": side,
            "intent": intent_kind,
            "qty": qty,
            "order_type": "MARKET",
            "limit_price": None,
            "reference_price": str(Decimal(str(ref_price))),
            "reference_source": ref_source,
            "stop_loss_price": None if not stop_loss_price else str(Decimal(str(stop_loss_price))),
            "target_price": None if not target_price else str(Decimal(str(target_price))),
            "strategy_type": strategy_type,
            "reason": notes,
            "fill_mode": self.mode,
            "lumina_order_ref": order_ref,
        }
        try:
            self._append_intent(intent, session_date)
        except OSError as e:
            guard.record_completed(order_ref, is_success=False, error_message=f"意図を書けなかった: {e}")
            return self._refuse(f"【ウィブル証券・日本株現物】意図を書けなかった: {e}")

        if self.mode == "paper":
            fill_price, fill_qty = ref_price, qty
        else:
            result = self._await_result(intent_id, session_date)
            status = str((result or {}).get("status", "")).upper()
            if result is None:
                return {"success": False, "unconfirmed": True, "intent_written": True, "intent_id": intent_id,
                        "error": f"【ウィブル証券・日本株現物】執行機の結果が {self.await_timeout_sec:.0f} 秒以内に来なかった。台帳には記帳していない",
                        "message": "執行機の結果が未確認"}
            if status not in FILLED_STATUSES:
                reasons = result.get("reasons") or []
                guard.record_completed(order_ref, is_success=False, error_message=f"{status}: {reasons}")
                return {"success": False, "intent_written": True, "intent_id": intent_id, "status": status,
                        "error": f"【ウィブル証券・日本株現物】執行機が約定させなかった（{status}）", "reasons": reasons}
            try:
                fill_qty = int(result["filled_qty"])
                fill_price = float(Decimal(str(result["avg_price"])))
            except (KeyError, TypeError, ValueError, ArithmeticError) as e:
                return {"success": False, "unconfirmed": True, "intent_written": True, "intent_id": intent_id,
                        "error": f"【ウィブル証券・日本株現物】執行機の結果を読めなかった（{e}）"}
            if fill_qty <= 0 or fill_qty > qty or fill_qty % JP_LOT != 0 or fill_price <= 0:
                return {"success": False, "unconfirmed": True, "intent_written": True, "intent_id": intent_id,
                        "error": "【ウィブル証券・日本株現物】執行機の結果の数量か値段が不正"}

        guard.record_completed(order_ref, is_success=False, error_message=f"台帳の記帳へ引き継ぎ（意図 {intent_id}）")
        booked = self._book_fill(
            norm_sym, side, fill_qty, fill_price, f"{notes} [ウィブル証券・日本株現物 意図 {intent_id} / {self.mode}]".strip(),
            target_price, stop_loss_price, strategy_type, expected_version,
        )
        booked.update({"intent_written": True, "intent_id": intent_id, "fill_mode": self.mode,
                       "filled_quantity": fill_qty, "fill_price": fill_price})
        if not booked.get("success"):
            booked["ledger_mismatch"] = self.mode == "await"
        return booked


def create_broker() -> PaperTradingEngine:
    """
    環境変数 LUMINA_BROKER でブローカーを選ぶ。既定は paper（従来どおり）。

    - paper:           PaperTradingEngine
    - webull_intents:  WebullIntentBroker（米国株・信用口座。LUMINA_WEBULL_INTENT_DIR が必須）
    - webull_jp_cash:  WebullJpCashIntentBroker（日本株・現物。LUMINA_WEBULL_INTENT_DIR が必須）
    """
    kind = os.getenv("LUMINA_BROKER", "paper").strip().lower()
    if kind in ("", "paper"):
        return PaperTradingEngine()
    if kind == "webull_intents":
        return WebullIntentBroker()
    if kind == "webull_jp_cash":
        return WebullJpCashIntentBroker()
    raise ValueError(f"LUMINA_BROKER が不正: {kind!r}（paper / webull_intents / webull_jp_cash）")
