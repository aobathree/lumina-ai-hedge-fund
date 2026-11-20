# -*- coding: utf-8 -*-
"""
LLM の呼び出し口。Gemini に加えて Claude（Anthropic API）を使えるようにする。

- プロバイダは LUMINA_LLM_PROVIDER で選ぶ（auto / claude / gemini）。既定の auto は、
  Claude の鍵が解決できれば Claude、できなければ Gemini。
- Claude の鍵は ANTHROPIC_API_KEY から受ける。値が 1Password の秘密参照（op://...）なら、
  1Password の CLI（`op read`）で解決する。解決した鍵はプロセスのメモリにだけ持ち、
  ログにもファイルにも出さない。`op run --env-file ... -- <コマンド>` で起動した場合は、
  環境変数に実際の鍵が入っているので、そのまま使う。
- 既存の呼び出し箇所は google-genai の `client.models.generate_content(...)` の形で書かれている。
  ClaudeGenaiShim がその形のまま Claude を呼ぶので、呼び出し箇所は鍵とクライアントの作り方だけを変える。
"""
import os
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

OP_PREFIX = "op://"
CLAUDE_SENTINEL = "__lumina_use_claude__"  # resolve_api_key が「Claude を使う」ことを伝える値。鍵ではない

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_CLAUDE_EFFORT = "medium"
CLAUDE_MAX_TOKENS = 16000
# 拒否（stop_reason = refusal）のとき、API の側で代わりのモデルに回す設定を付けるモデル
SERVER_FALLBACK_MODEL_PREFIXES = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2")
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"
OP_FAILURE_RETRY_SEC = 60.0  # 解決に失敗した参照を、この秒数は引き直さない（承認の画面を何度も出さない）

_lock = threading.Lock()
_op_cache: Dict[str, str] = {}
_op_failed_at: Dict[str, float] = {}
_anthropic_clients: Dict[str, Any] = {}


# Claude の呼び出しの回数とトークン数（このプロセスの中の合計）。費用の見積もりに使う
USAGE: Dict[str, int] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "errors": 0}


def _record_usage(response: Any) -> None:
    usage = getattr(response, "usage", None)
    with _lock:
        USAGE["calls"] += 1
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
            USAGE[key] += int(getattr(usage, key, 0) or 0)


def _count_error() -> None:
    with _lock:
        USAGE["errors"] += 1


def usage_snapshot() -> Dict[str, int]:
    with _lock:
        return dict(USAGE)


class LlmCallError(RuntimeError):
    """Claude の呼び出しの失敗。メッセージに HTTP の状態や種類を入れる（呼ぶ側が文字列で見分けている）。"""


def read_op_reference(ref: str, timeout: Optional[float] = None) -> str:
    """1Password の秘密参照を解決して返す。解決できなければ空文字。値はログに出さない。"""
    with _lock:
        if ref in _op_cache:
            return _op_cache[ref]
        failed = _op_failed_at.get(ref)
        if failed is not None and time.monotonic() - failed < OP_FAILURE_RETRY_SEC:
            return ""
        value, why = "", ""
        if shutil.which("op") is None:
            why = "1Password の CLI（op）が見つからない"
        else:
            try:
                wait = float(timeout if timeout is not None else os.getenv("LUMINA_OP_TIMEOUT_SEC", "60"))
                done = subprocess.run(["op", "read", "--no-newline", ref], capture_output=True, text=True, timeout=wait)
                if done.returncode == 0 and done.stdout.strip():
                    value = done.stdout.strip()
                else:
                    first = (done.stderr or "").strip().splitlines()[:1]
                    why = f"op read が失敗（終了コード {done.returncode}）: {first[0] if first else '出力なし'}"
            except subprocess.TimeoutExpired:
                why = "op read が時間切れ（1Password のアンロックか承認を待っていた）"
            except OSError as e:
                why = f"op を起動できない: {type(e).__name__}"
        if value:
            _op_cache[ref] = value
        else:
            _op_failed_at[ref] = time.monotonic()
            print(f"[llm_client] 1Password の参照を解決できなかった。{why}")
        return value


def _settings() -> Dict[str, Any]:
    try:
        from core.state_manager import load_app_settings
        return load_app_settings() or {}
    except Exception:
        return {}


def anthropic_api_key(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Claude の鍵。環境変数 → 設定の順に見る。op:// の参照は解決する。無ければ空文字。"""
    raw = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not raw:
        raw = str((cfg if cfg is not None else _settings()).get("anthropic_api_key") or "").strip()
    if raw.startswith(OP_PREFIX):
        return read_op_reference(raw)
    return raw


def gemini_api_key(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Gemini の鍵。設定の google_api_key、gemini_api_key、環境変数 GEMINI_API_KEY の順に見る。"""
    cfg = cfg if cfg is not None else _settings()
    raw = str(cfg.get("google_api_key") or cfg.get("gemini_api_key") or os.getenv("GEMINI_API_KEY") or "").strip()
    if raw.startswith(OP_PREFIX):
        return read_op_reference(raw)
    return raw


def provider(cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """使うプロバイダ（"claude" / "gemini"）。鍵が無ければ None。"""
    want = (os.getenv("LUMINA_LLM_PROVIDER") or "auto").strip().lower()
    if want not in ("auto", "claude", "gemini"):
        raise ValueError(f"LUMINA_LLM_PROVIDER が不正: {want!r}（auto / claude / gemini）")
    if want in ("auto", "claude") and anthropic_api_key(cfg):
        return "claude"
    if want in ("auto", "gemini") and gemini_api_key(cfg):
        return "gemini"
    return None


def resolve_api_key(cfg: Optional[Dict[str, Any]] = None) -> str:
    """
    既存の `cfg.get("google_api_key", "")` の置き換え。
    Claude を使うときは CLAUDE_SENTINEL（真の値。鍵ではない）、Gemini を使うときはその鍵、どちらも無ければ空文字。
    """
    which = provider(cfg)
    if which == "claude":
        return CLAUDE_SENTINEL
    if which == "gemini":
        return gemini_api_key(cfg)
    return ""


def make_client(api_key: str):
    """既存の `genai.Client(api_key=api_key)` の置き換え。resolve_api_key の戻り値を渡す。"""
    if api_key == CLAUDE_SENTINEL:
        return ClaudeGenaiShim()
    from google import genai
    return genai.Client(api_key=api_key)


def claude_model() -> str:
    return (os.getenv("LUMINA_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL).strip()


def claude_effort() -> str:
    return (os.getenv("LUMINA_CLAUDE_EFFORT") or DEFAULT_CLAUDE_EFFORT).strip().lower()


def _anthropic_client():
    key = anthropic_api_key()
    if not key:
        raise LlmCallError("Claude の鍵が無い（ANTHROPIC_API_KEY が未設定か、1Password の参照を解決できない）")
    with _lock:
        client = _anthropic_clients.get(key)
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=key, timeout=float(os.getenv("LUMINA_CLAUDE_TIMEOUT_SEC", "180")))
            _anthropic_clients[key] = client
        return client


def claude_generate(prompt: str, system: Optional[str] = None, want_json: bool = False) -> str:
    """Claude に 1 回問い合わせて、本文の文字列を返す。失敗は LlmCallError。"""
    import anthropic

    client = _anthropic_client()
    model = claude_model()
    if want_json:
        prompt = f"{prompt}\n\n出力は JSON だけにしてください。前置き、説明、Markdown のコードブロックは付けないでください。"
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": CLAUDE_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": claude_effort()},
    }
    if system:
        kwargs["system"] = system
    try:
        if model.startswith(SERVER_FALLBACK_MODEL_PREFIXES):
            response = client.beta.messages.create(betas=[SERVER_FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = client.messages.create(**kwargs)
    except anthropic.RateLimitError as e:
        _count_error()
        raise LlmCallError(f"429 rate limit (Claude): {e.message}") from e
    except anthropic.AuthenticationError as e:
        _count_error()
        raise LlmCallError("401 authentication (Claude): 鍵が無効") from e
    except anthropic.APIStatusError as e:
        _count_error()
        kind = "503 unavailable" if e.status_code >= 500 else f"{e.status_code} error"
        raise LlmCallError(f"{kind} (Claude): {e.message}") from e
    except anthropic.APITimeoutError as e:
        _count_error()
        raise LlmCallError("timeout (Claude)") from e
    except anthropic.APIConnectionError as e:
        _count_error()
        raise LlmCallError("unavailable: connection error (Claude)") from e

    _record_usage(response)
    if response.stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        raise LlmCallError(f"refusal (Claude): category={category}")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise LlmCallError(f"empty response (Claude): stop_reason={response.stop_reason}")
    return text


class _ShimModels:
    def generate_content(self, model: Any = None, contents: Any = None, config: Any = None):
        """google-genai の generate_content と同じ呼び方で Claude を呼ぶ。model は使わない（LUMINA_CLAUDE_MODEL が決める）。"""
        if not isinstance(contents, str):
            raise LlmCallError("Claude の経路は、文字列のプロンプトだけを扱う")
        system = getattr(config, "system_instruction", None) if config is not None else None
        mime = getattr(config, "response_mime_type", None) if config is not None else None
        text = claude_generate(contents, system=system if isinstance(system, str) else None, want_json=(mime == "application/json"))
        return SimpleNamespace(text=text)


class ClaudeGenaiShim:
    """`genai.Client` の代わり。使われているのは `.models.generate_content(...)` と戻り値の `.text` だけ。"""

    def __init__(self) -> None:
        self.models = _ShimModels()
