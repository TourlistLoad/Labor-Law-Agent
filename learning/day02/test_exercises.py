"""Test tự chấm cho Ngày 2. KHÔNG sửa file này.

Chạy:  uv run pytest learning/day02 -v
Không test nào gọi mạng thật: mọi request tới Gemini đều đi qua
httpx.MockTransport (một "server giả" chạy ngay trong test).
"""

import asyncio
import json
import time

import httpx
import pytest
from pydantic import ValidationError

from learning.day02.exercises import (
    GEMINI_BASE_URL,
    ChatTurn,
    GenerationSettings,
    LLMBlockedError,
    LLMHTTPError,
    LLMResult,
    build_request,
    call_gemini,
    estimate_cost_usd,
    parse_response,
    run_concurrent,
    run_limited,
    run_sequential,
)

QUESTIONS = [f"q{i}" for i in range(5)]


# ---------------------------------------------------------------- Bài 1
class TestSequentialVsConcurrent:
    def test_sequential_answers_in_order(self):
        assert asyncio.run(run_sequential(QUESTIONS, delay=0.01)) == [f"answer: q{i}" for i in range(5)]

    def test_sequential_takes_sum_of_delays(self):
        t0 = time.perf_counter()
        asyncio.run(run_sequential(QUESTIONS, delay=0.1))
        assert time.perf_counter() - t0 >= 0.5, "5 lần x 0.1s tuần tự phải mất >= 0.5s"

    def test_concurrent_answers_in_order(self):
        assert asyncio.run(run_concurrent(QUESTIONS, delay=0.01)) == [f"answer: q{i}" for i in range(5)]

    def test_concurrent_takes_one_delay(self):
        t0 = time.perf_counter()
        asyncio.run(run_concurrent(QUESTIONS, delay=0.1))
        assert time.perf_counter() - t0 < 0.3, "chạy đồng thời phải mất xấp xỉ 1 lần delay"

    def test_returns_list(self):
        assert isinstance(asyncio.run(run_concurrent(["a"], delay=0)), list)


# ---------------------------------------------------------------- Bài 2
class TestRunLimited:
    @staticmethod
    def _tracker(delay: float = 0.05):
        state = {"in_flight": 0, "max": 0}

        async def call(q: str) -> str:
            state["in_flight"] += 1
            state["max"] = max(state["max"], state["in_flight"])
            await asyncio.sleep(delay)
            state["in_flight"] -= 1
            return q.upper()

        return state, call

    def test_never_exceeds_limit(self):
        state, call = self._tracker()
        asyncio.run(run_limited([f"q{i}" for i in range(10)], 3, call))
        assert state["max"] == 3

    def test_keeps_order(self):
        _, call = self._tracker(0.01)
        qs = [f"q{i}" for i in range(7)]
        assert asyncio.run(run_limited(qs, 2, call)) == [q.upper() for q in qs]

    def test_limit_one_is_sequential(self):
        state, call = self._tracker(0.01)
        asyncio.run(run_limited(["a", "b", "c"], 1, call))
        assert state["max"] == 1

    def test_time_is_batched(self):
        _, call = self._tracker(0.1)
        t0 = time.perf_counter()
        asyncio.run(run_limited([f"q{i}" for i in range(6)], 3, call))
        assert 0.2 <= time.perf_counter() - t0 < 0.35, "6 việc, 3 làn -> khoảng 2 lượt x 0.1s"

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_limit(self, bad):
        _, call = self._tracker(0)
        with pytest.raises(ValueError):
            asyncio.run(run_limited(["a"], bad, call))


# ---------------------------------------------------------------- Bài 3
class TestGenerationSettings:
    def test_defaults(self):
        s = GenerationSettings()
        assert (s.temperature, s.max_output_tokens, s.thinking_level) == (None, 1024, None)

    @pytest.mark.parametrize(
        "kwargs",
        [{"temperature": -0.1}, {"temperature": 2.5}, {"max_output_tokens": 0}, {"thinking_level": "extreme"}],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValidationError):
            GenerationSettings(**kwargs)


class TestBuildRequest:
    def test_minimal_shape(self):
        body = build_request("Lương tối thiểu?", GenerationSettings())
        assert body == {
            "contents": [{"role": "user", "parts": [{"text": "Lương tối thiểu?"}]}],
            "generationConfig": {"maxOutputTokens": 1024},
        }

    def test_temperature_only_when_set(self):
        body = build_request("q", GenerationSettings(temperature=0.3))
        assert body["generationConfig"] == {"maxOutputTokens": 1024, "temperature": 0.3}

    def test_system_instruction(self):
        body = build_request("q", GenerationSettings(), system="Bạn là trợ lý luật.")
        assert body["systemInstruction"] == {"parts": [{"text": "Bạn là trợ lý luật."}]}

    def test_thinking_level(self):
        body = build_request("q", GenerationSettings(thinking_level="low"))
        assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}

    def test_history_is_resent_before_question(self):
        history = [ChatTurn(role="user", text="Tôi làm 3 năm."), ChatTurn(role="model", text="Đã ghi nhận.")]
        body = build_request("Tôi được bao nhiêu?", GenerationSettings(), history=history)
        assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
        assert body["contents"][0]["parts"] == [{"text": "Tôi làm 3 năm."}]
        assert body["contents"][-1]["parts"] == [{"text": "Tôi được bao nhiêu?"}]

    def test_history_not_mutated(self):
        history = [ChatTurn(role="user", text="a"), ChatTurn(role="model", text="b")]
        build_request("c", GenerationSettings(), history=history)
        assert len(history) == 2

    def test_json_serializable(self):
        body = build_request("q", GenerationSettings(thinking_level="high"), system="s")
        assert json.loads(json.dumps(body)) == body


# ---------------------------------------------------------------- Bài 4
OK_RESPONSE = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": "tóm tắt suy nghĩ", "thought": True},
                    {"text": "Theo Điều 46, "},
                    {"text": "bạn được trợ cấp."},
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 120,
        "candidatesTokenCount": 40,
        "thoughtsTokenCount": 200,
        "totalTokenCount": 360,
    },
}


class TestParseResponse:
    def test_ok(self):
        r = parse_response(OK_RESPONSE)
        assert isinstance(r, LLMResult)
        assert r.text == "Theo Điều 46, bạn được trợ cấp."
        assert r.finish_reason == "STOP"
        assert (r.prompt_tokens, r.output_tokens, r.thinking_tokens, r.total_tokens) == (120, 40, 200, 360)

    def test_missing_thoughts_count_defaults_to_zero(self):
        data = json.loads(json.dumps(OK_RESPONSE))
        del data["usageMetadata"]["thoughtsTokenCount"]
        assert parse_response(data).thinking_tokens == 0

    def test_max_tokens_is_not_an_error(self):
        data = json.loads(json.dumps(OK_RESPONSE))
        data["candidates"][0]["finishReason"] = "MAX_TOKENS"
        assert parse_response(data).finish_reason == "MAX_TOKENS"

    def test_parts_without_text_are_skipped(self):
        data = json.loads(json.dumps(OK_RESPONSE))
        data["candidates"][0]["content"]["parts"].append({"thoughtSignature": "abc123"})
        assert parse_response(data).text == "Theo Điều 46, bạn được trợ cấp."

    def test_blocked_prompt_raises(self):
        data = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 5}}
        with pytest.raises(LLMBlockedError, match="SAFETY"):
            parse_response(data)


class TestEstimateCost:
    def test_thinking_billed_as_output(self):
        r = parse_response(OK_RESPONSE)
        # 120 * 0.75/1e6 + (40 + 200) * 3.75/1e6 = 0.00009 + 0.0009 = 0.00099
        assert estimate_cost_usd(r, 0.75, 3.75) == pytest.approx(0.00099)

    def test_zero_tokens(self):
        r = LLMResult(text="", finish_reason="STOP", prompt_tokens=0, output_tokens=0, total_tokens=0)
        assert estimate_cost_usd(r, 1, 1) == 0


# ---------------------------------------------------------------- Bài 5
BODY = {"contents": [{"role": "user", "parts": [{"text": "xin chào"}]}]}


def _client(responses: list) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    """Server giả: trả lần lượt từng phần tử trong `responses`; ghi lại mọi request."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


def _run(responses, **kwargs):
    async def go():
        client, seen = _client(responses)
        delays: list[float] = []

        async def fake_sleep(s: float) -> None:
            delays.append(s)

        async with client:
            result = await call_gemini(
                client, api_key="KEY123", model="gemini-test", body=BODY, sleep=fake_sleep, **kwargs
            )
        return result, seen, delays

    return asyncio.run(go())


class TestCallGemini:
    def test_request_is_correct(self):
        result, seen, _ = _run([httpx.Response(200, json=OK_RESPONSE)])
        req = seen[0]
        assert req.method == "POST"
        assert str(req.url) == f"{GEMINI_BASE_URL}/models/gemini-test:generateContent"
        assert req.headers["x-goog-api-key"] == "KEY123"
        assert "key=" not in str(req.url), "không đặt API key trên URL"
        assert json.loads(req.content) == BODY
        assert result.text.startswith("Theo Điều 46")

    def test_retry_429_then_ok(self):
        result, seen, delays = _run(
            [httpx.Response(429, json={"error": {"code": 429}}), httpx.Response(200, json=OK_RESPONSE)]
        )
        assert len(seen) == 2
        assert delays == [1.0]
        assert result.finish_reason == "STOP"

    def test_exponential_backoff(self):
        _, seen, delays = _run(
            [httpx.Response(503), httpx.Response(500), httpx.Response(200, json=OK_RESPONSE)],
            base_delay=0.5,
        )
        assert len(seen) == 3
        assert delays == [0.5, 1.0]

    def test_gives_up_after_max_attempts(self):
        with pytest.raises(LLMHTTPError) as exc:
            _run([httpx.Response(503, text="overloaded")] * 3)
        assert exc.value.status_code == 503

    def test_no_retry_on_400(self):
        with pytest.raises(LLMHTTPError) as exc:
            _run([httpx.Response(400, text="bad request"), httpx.Response(200, json=OK_RESPONSE)])
        assert exc.value.status_code == 400
        assert "bad request" in exc.value.body

    def test_no_retry_on_403(self):
        with pytest.raises(LLMHTTPError) as exc:
            _run([httpx.Response(403, text="key invalid"), httpx.Response(200, json=OK_RESPONSE)])
        assert exc.value.status_code == 403

    def test_retry_on_network_error(self):
        result, seen, delays = _run([httpx.ConnectTimeout("timeout"), httpx.Response(200, json=OK_RESPONSE)])
        assert len(seen) == 2 and delays == [1.0]
        assert result.prompt_tokens == 120
