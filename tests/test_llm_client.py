# -*- coding: utf-8 -*-
"""
llm_client のテスト: Claude の経路と、1Password の秘密参照（op://）の解決。
ネットワークにも 1Password にも出ない（SDK のクライアントと op の呼び出しを差し替える）。
"""
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from core.ai_hedge_fund import llm_client as lc

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = "op://Automation/Anthropic API Key/credential"
SECRET = "sk-ant-test-0000-not-a-real-key"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LUMINA_LLM_PROVIDER", "LUMINA_CLAUDE_MODEL", "LUMINA_CLAUDE_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(lc, "_settings", lambda: {})
    lc._op_cache.clear()
    lc._op_failed_at.clear()
    lc._anthropic_clients.clear()
    for k in lc.USAGE:
        lc.USAGE[k] = 0
    yield
    lc._op_cache.clear()
    lc._op_failed_at.clear()
    lc._anthropic_clients.clear()


def fake_op(monkeypatch, returncode=0, stdout=SECRET, stderr="", raises=None, installed=True):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if raises:
            raise raises
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(lc.shutil, "which", lambda name: "/usr/local/bin/op" if installed else None)
    monkeypatch.setattr(lc.subprocess, "run", run)
    return calls


# ---- 1Password の参照 ----

def test_op_reference_is_resolved_once_and_never_printed(monkeypatch, capsys):
    calls = fake_op(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", REF)
    assert lc.anthropic_api_key() == SECRET
    assert lc.anthropic_api_key() == SECRET
    assert calls == [["op", "read", "--no-newline", REF]]  # 2 回めはメモリから
    out = capsys.readouterr()
    assert SECRET not in out.out and SECRET not in out.err


@pytest.mark.parametrize("kw, needle", [
    (dict(returncode=1, stdout="", stderr="[ERROR] authorization timeout\nmore"), "authorization timeout"),
    (dict(returncode=0, stdout="   "), "出力なし"),
    (dict(raises=subprocess.TimeoutExpired(cmd="op", timeout=1)), "時間切れ"),
    (dict(installed=False), "見つからない"),
])
def test_unresolved_reference_yields_no_key_and_is_not_retried_immediately(monkeypatch, capsys, kw, needle):
    calls = fake_op(monkeypatch, **kw)
    monkeypatch.setenv("ANTHROPIC_API_KEY", REF)
    assert lc.anthropic_api_key() == ""       # 参照の文字列そのものを鍵として返さない
    assert lc.anthropic_api_key() == ""
    assert len(calls) <= 1                    # 失敗した参照を続けて引き直さない
    assert needle in capsys.readouterr().out
    assert lc.provider() is None
    assert lc.resolve_api_key() == ""


def test_plain_key_from_environment_or_settings(monkeypatch):
    calls = fake_op(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", f"  {SECRET}  ")
    assert lc.anthropic_api_key() == SECRET and calls == []
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert lc.anthropic_api_key({"anthropic_api_key": REF}) == SECRET  # 設定の値が参照でも解決する


# ---- プロバイダの選択 ----

def test_provider_selection(monkeypatch):
    assert lc.provider() is None
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert lc.provider() == "gemini" and lc.resolve_api_key() == "g-key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    assert lc.provider() == "claude" and lc.resolve_api_key() == lc.CLAUDE_SENTINEL
    monkeypatch.setenv("LUMINA_LLM_PROVIDER", "gemini")
    assert lc.provider() == "gemini"
    monkeypatch.setenv("LUMINA_LLM_PROVIDER", "claude")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert lc.provider() is None              # claude を指定して鍵が無ければ、Gemini に黙って切り替えない
    monkeypatch.setenv("LUMINA_LLM_PROVIDER", "openai")
    with pytest.raises(ValueError):
        lc.provider()


def test_gemini_key_is_found_under_either_settings_name():
    assert lc.gemini_api_key({"google_api_key": "a"}) == "a"
    assert lc.gemini_api_key({"gemini_api_key": "b"}) == "b"   # state_manager の既定の名前


def test_make_client_returns_the_shim_for_claude():
    assert isinstance(lc.make_client(lc.CLAUDE_SENTINEL), lc.ClaudeGenaiShim)


# ---- Claude の呼び出し ----

class FakeAnthropic:
    def __init__(self, text='{"score": 77}', stop_reason="end_turn", raises=None):
        self.calls = []
        outer = self

        def make(kind):
            def create(**kw):
                outer.calls.append((kind, kw))
                if raises:
                    raise raises
                blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)] if text else []
                return SimpleNamespace(content=blocks, stop_reason=stop_reason,
                                       usage=SimpleNamespace(input_tokens=100, output_tokens=40, cache_read_input_tokens=0),
                                       stop_details=SimpleNamespace(category="cyber", explanation="x"))
            return SimpleNamespace(create=create)

        self.messages = make("ga")
        self.beta = SimpleNamespace(messages=make("beta"))


@pytest.fixture
def claude(monkeypatch):
    fake = FakeAnthropic()
    monkeypatch.setattr(lc, "_anthropic_client", lambda: fake)
    return fake


def test_default_request_shape(claude):
    assert lc.claude_generate("質問", system="あなたは審査役") == '{"score": 77}'
    kind, kw = claude.calls[0]
    assert kind == "beta"
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 16000
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["system"] == "あなたは審査役"
    assert kw["messages"] == [{"role": "user", "content": "質問"}]
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    for removed in ("temperature", "top_p", "top_k", "thinking"):  # 今のモデルでは 400 になる、または不要
        assert removed not in kw


def test_model_and_effort_come_from_the_environment(claude, monkeypatch):
    monkeypatch.setenv("LUMINA_CLAUDE_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("LUMINA_CLAUDE_EFFORT", "low")
    lc.claude_generate("q")
    kind, kw = claude.calls[0]
    assert kind == "ga" and "fallbacks" not in kw and "betas" not in kw  # Haiku にはサーバー側の代替が無い
    assert kw["model"] == "claude-haiku-5-5" and kw["output_config"] == {"effort": "low"}


def test_json_request_adds_an_instruction_and_no_prefill(claude):
    lc.claude_generate("q", want_json=True)
    msgs = claude.calls[0][1]["messages"]
    assert len(msgs) == 1 and msgs[0]["role"] == "user" and "JSON だけ" in msgs[0]["content"]


def test_refusal_and_empty_response_are_errors(monkeypatch):
    monkeypatch.setattr(lc, "_anthropic_client", lambda: FakeAnthropic(stop_reason="refusal"))
    with pytest.raises(lc.LlmCallError, match="refusal"):
        lc.claude_generate("q")
    monkeypatch.setattr(lc, "_anthropic_client", lambda: FakeAnthropic(text=""))
    with pytest.raises(lc.LlmCallError, match="empty response"):
        lc.claude_generate("q")


def _status_error(cls, status):
    import httpx
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message="boom", response=httpx.Response(status, request=req), body=None)


def test_sdk_errors_are_mapped_to_messages_the_callers_recognise(monkeypatch):
    import anthropic
    import httpx
    cases = [
        (_status_error(anthropic.RateLimitError, 429), "429"),
        (_status_error(anthropic.AuthenticationError, 401), "401"),
        (_status_error(anthropic.InternalServerError, 500), "503 unavailable"),
        (_status_error(anthropic.BadRequestError, 400), "400 error"),
        (anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")), "timeout"),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), "unavailable"),
    ]
    for exc, needle in cases:
        monkeypatch.setattr(lc, "_anthropic_client", lambda exc=exc: FakeAnthropic(raises=exc))
        with pytest.raises(lc.LlmCallError) as e:
            lc.claude_generate("q")
        assert needle in str(e.value), (type(exc).__name__, str(e.value))


def test_missing_key_is_an_error_not_a_request_with_the_reference(monkeypatch):
    fake_op(monkeypatch, returncode=1, stdout="")
    monkeypatch.setenv("ANTHROPIC_API_KEY", REF)
    with pytest.raises(lc.LlmCallError, match="鍵が無い"):
        lc.claude_generate("q")


def test_shim_passes_system_instruction_and_json_mode(claude):
    cfg = SimpleNamespace(system_instruction="sys", response_mime_type="application/json", temperature=0.7)
    res = lc.ClaudeGenaiShim().models.generate_content(model="gemini-3.8-flash", contents="p", config=cfg)
    assert res.text == '{"score": 77}'
    kw = claude.calls[0][1]
    assert kw["system"] == "sys" and "JSON だけ" in kw["messages"][0]["content"] and kw["model"] == "claude-opus-5-5"


# ---- 既存の呼び出し箇所 ----

def test_investor_agent_evaluates_through_claude(monkeypatch):
    from core.ai_hedge_fund.agents.base_agent import BaseInvestmentAgent
    from core.ai_hedge_fund.agents.buffett_agent import BuffettAgent

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    fake = FakeAnthropic(text='```json\n{"stance": "BULLISH", "score": 81, "dialogue": "ROE 20% は堀の証拠だ。"}\n```')
    monkeypatch.setattr(lc, "_anthropic_client", lambda: fake)
    monkeypatch.setattr(BaseInvestmentAgent, "is_circuit_open", classmethod(lambda cls: False))
    res = BuffettAgent.evaluate({"symbol": "KO", "name": "Coca-Cola", "current_price": 60.0, "currency": "USD", "roe": 20.0}, {})
    assert res["is_degraded_fallback"] is False and res["score"] == 81 and res["stance"] == "BULLISH"
    assert fake.calls and fake.calls[0][1]["system"]  # 役の人格（system）が Claude に渡っている


def test_every_call_site_goes_through_llm_client():
    left = []
    for path in (ROOT / "core").rglob("*.py"):
        if path.name == "llm_client.py":
            continue
        text = path.read_text(encoding="utf-8")
        if 'cfg.get("google_api_key"' in text or "genai.Client(" in text:
            left.append(path.name)
    assert left == [], f"llm_client を通っていない LLM の呼び出しがある: {left}"


def test_no_real_key_or_reference_is_committed():
    sample = (ROOT / ".env.op.sample").read_text(encoding="utf-8")
    assert "op://<vault>/<item>/credential" in sample and "sk-ant-" not in sample
    assert ".env.op" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_usage_is_accumulated(claude, monkeypatch):
    import anthropic
    lc.claude_generate("a")
    lc.claude_generate("b")
    monkeypatch.setattr(lc, "_anthropic_client", lambda: FakeAnthropic(raises=_status_error(anthropic.RateLimitError, 429)))
    with pytest.raises(lc.LlmCallError):
        lc.claude_generate("c")
    assert lc.usage_snapshot() == {"calls": 2, "input_tokens": 200, "output_tokens": 80, "cache_read_input_tokens": 0, "errors": 1}
