"""Test tự chấm cho Ngày 3. KHÔNG sửa file này.

Chạy:  uv run pytest learning/day03 -v
"LLM" ở đây là ScriptedLLM: trả lời lần lượt theo kịch bản và ghi lại mọi
request nhận được, để test kiểm tra bạn đã gửi gì.
"""

import asyncio
import copy
import json

import pytest

from learning.day02.exercises import LLMBlockedError
from learning.day03.exercises import (
    OVERTIME_TOOL,
    REGISTRY,
    SEVERANCE_TOOL,
    AgentResult,
    AgentStepLimitError,
    ExtractionError,
    FunctionCall,
    ModelTurn,
    ToolSpec,
    build_agent_request,
    execute_call,
    extract_with_retry,
    function_response_turn,
    parse_model_turn,
    run_agent,
    tool_declaration,
    user_turn,
)
from learning.day01.exercises import SeveranceInput


# ------------------------------------------------------------ dụng cụ test
class ScriptedLLM:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    async def __call__(self, body: dict) -> dict:
        self.requests.append(copy.deepcopy(body))   # chụp lại đúng lúc gửi
        if not self.responses:
            raise AssertionError("LLM bị gọi nhiều hơn kịch bản")
        return self.responses.pop(0)


def text_response(text: str) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]}


def call_response(*calls: tuple, signature: str = "SIG-1") -> dict:
    """calls: (name, args) hoặc (name, args, id). Signature gắn vào part functionCall đầu tiên."""
    parts = []
    for i, c in enumerate(calls):
        fc = {"name": c[0], "args": c[1]}
        if len(c) > 2:
            fc["id"] = c[2]
        part = {"functionCall": fc}
        if i == 0:
            part["thoughtSignature"] = signature
        parts.append(part)
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]}


SEV_ARGS = {"months_worked": 50, "months_unemployment_insurance": 12, "avg_salary_vnd": 20_000_000}
OT_ARGS = {"hourly_wage_vnd": 60_000, "hours": 4, "day_type": "weekly_off"}


# ---------------------------------------------------------------- Bài 1
class TestToolDeclaration:
    def test_shape(self):
        d = tool_declaration(SEVERANCE_TOOL)
        assert set(d) == {"name", "description", "parametersJsonSchema"}
        assert d["name"] == "calc_severance"
        assert d["description"] == SEVERANCE_TOOL.description
        assert d["parametersJsonSchema"] == SeveranceInput.model_json_schema()

    def test_schema_is_json(self):
        d = tool_declaration(OVERTIME_TOOL)
        assert json.loads(json.dumps(d)) == d
        assert d["parametersJsonSchema"]["properties"]["day_type"]["enum"] == ["weekday", "weekly_off", "holiday"]


class TestBuildAgentRequest:
    def test_shape(self):
        contents = [user_turn("xin chào")]
        body = build_agent_request(contents, [SEVERANCE_TOOL, OVERTIME_TOOL])
        assert body["contents"] == contents
        assert len(body["tools"]) == 1
        names = [d["name"] for d in body["tools"][0]["functionDeclarations"]]
        assert names == ["calc_severance", "calc_overtime"]
        assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
        assert "systemInstruction" not in body

    def test_system_and_mode(self):
        body = build_agent_request([user_turn("q")], [SEVERANCE_TOOL], system="Bạn là trợ lý.", mode="ANY")
        assert body["systemInstruction"] == {"parts": [{"text": "Bạn là trợ lý."}]}
        assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"

    def test_contents_is_a_copy(self):
        contents = [user_turn("q")]
        body = build_agent_request(contents, [SEVERANCE_TOOL])
        body["contents"].append(user_turn("thêm"))
        assert len(contents) == 1, "body['contents'] phải là list mới, không phải chính list truyền vào"


# ---------------------------------------------------------------- Bài 2
class TestParseModelTurn:
    def test_text_only(self):
        t = parse_model_turn(text_response("Thử việc tối đa 60 ngày."))
        assert isinstance(t, ModelTurn)
        assert t.text == "Thử việc tối đa 60 ngày."
        assert t.calls == []
        assert t.finish_reason == "STOP"

    def test_function_call_with_id(self):
        t = parse_model_turn(call_response(("calc_severance", SEV_ARGS, "c1")))
        assert t.calls == [FunctionCall(name="calc_severance", args=SEV_ARGS, id="c1")]
        assert t.text == ""

    def test_parallel_calls_without_id(self):
        t = parse_model_turn(call_response(("calc_severance", SEV_ARGS), ("calc_overtime", OT_ARGS)))
        assert [c.name for c in t.calls] == ["calc_severance", "calc_overtime"]
        assert all(c.id is None for c in t.calls)

    def test_missing_args_becomes_empty_dict(self):
        data = {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "calc_overtime"}}]}}]}
        t = parse_model_turn(data)
        assert t.calls[0].args == {}
        assert t.finish_reason == "UNKNOWN"

    def test_content_kept_verbatim_with_signature(self):
        data = call_response(("calc_severance", SEV_ARGS), signature="KEEP-ME")
        t = parse_model_turn(data)
        assert t.content == data["candidates"][0]["content"]
        assert t.content["parts"][0]["thoughtSignature"] == "KEEP-ME"

    def test_text_and_call_together_skips_thought(self):
        data = {"candidates": [{"content": {"role": "model", "parts": [
            {"text": "đang nghĩ", "thought": True},
            {"text": "Để tôi tính. "},
            {"functionCall": {"name": "calc_overtime", "args": OT_ARGS}},
        ]}, "finishReason": "STOP"}]}
        t = parse_model_turn(data)
        assert t.text == "Để tôi tính. "
        assert len(t.calls) == 1

    def test_blocked(self):
        with pytest.raises(LLMBlockedError, match="SAFETY"):
            parse_model_turn({"promptFeedback": {"blockReason": "SAFETY"}})


# ---------------------------------------------------------------- Bài 3
class TestExecuteCall:
    def test_severance_ok(self):
        r = execute_call(FunctionCall(name="calc_severance", args=SEV_ARGS), REGISTRY)
        assert r["amount_vnd"] == 35_000_000
        assert r["eligible"] is True

    def test_overtime_ok(self):
        r = execute_call(FunctionCall(name="calc_overtime", args=OT_ARGS), REGISTRY)
        assert r == {"rate_percent": 200, "amount_vnd": 480_000, "note": r["note"]}

    def test_unknown_tool(self):
        r = execute_call(FunctionCall(name="send_email", args={}), REGISTRY)
        assert set(r) == {"error"}
        assert "send_email" in r["error"]

    def test_invalid_args_become_error_details(self):
        r = execute_call(FunctionCall(name="calc_severance", args={"months_worked": 700}), REGISTRY)
        assert "error" in r
        assert any(d.startswith("months_worked:") for d in r["details"])
        assert any(d.startswith("avg_salary_vnd:") for d in r["details"])

    def test_extra_arg_rejected(self):
        r = execute_call(FunctionCall(name="calc_overtime", args={**OT_ARGS, "bonus": 1}), REGISTRY)
        assert any(d.startswith("bonus:") for d in r["details"])

    def test_tool_crash_becomes_error(self):
        def boom(_inp):
            raise RuntimeError("database down")

        reg = {"boom": ToolSpec(name="boom", description="x", input_model=SeveranceInput, func=boom)}
        r = execute_call(FunctionCall(name="boom", args=SEV_ARGS), reg)
        assert "database down" in r["error"]

    def test_result_is_json_serializable(self):
        r = execute_call(FunctionCall(name="calc_severance", args=SEV_ARGS), REGISTRY)
        assert json.loads(json.dumps(r)) == r


class TestFunctionResponseTurn:
    def test_shape_with_ids(self):
        calls = [FunctionCall(name="calc_severance", args={}, id="c1"), FunctionCall(name="calc_overtime", args={}, id="c2")]
        turn = function_response_turn(calls, [{"a": 1}, {"b": 2}])
        assert turn == {"role": "user", "parts": [
            {"functionResponse": {"name": "calc_severance", "response": {"a": 1}, "id": "c1"}},
            {"functionResponse": {"name": "calc_overtime", "response": {"b": 2}, "id": "c2"}},
        ]}

    def test_no_id_key_when_none(self):
        turn = function_response_turn([FunctionCall(name="calc_overtime", args={})], [{"x": 1}])
        assert turn["parts"][0]["functionResponse"] == {"name": "calc_overtime", "response": {"x": 1}}


# ---------------------------------------------------------------- Bài 4
GOOD_JSON = json.dumps(SEV_ARGS)
BAD_JSON = json.dumps({"months_worked": 700, "avg_salary_vnd": 20_000_000})   # 700 > 600 tháng


class TestExtractWithRetry:
    def test_first_try(self):
        llm = ScriptedLLM([text_response(GOOD_JSON)])
        r = asyncio.run(extract_with_retry("Tôi làm 50 tháng...", SeveranceInput, llm))
        assert r.attempts == 1
        assert r.value == SeveranceInput(**SEV_ARGS)
        body = llm.requests[0]
        assert body["contents"] == [user_turn("Tôi làm 50 tháng...")]
        assert body["generationConfig"]["responseFormat"]["text"]["mimeType"] == "APPLICATION_JSON"

    def test_retry_sends_errors_back(self):
        llm = ScriptedLLM([text_response(BAD_JSON), text_response(GOOD_JSON)])
        r = asyncio.run(extract_with_retry("câu hỏi", SeveranceInput, llm))
        assert r.attempts == 2
        second = llm.requests[1]["contents"]
        assert len(second) == 3
        assert second[1] == {"role": "model", "parts": [{"text": BAD_JSON}]}
        assert second[2]["role"] == "user"
        assert "months_worked" in second[2]["parts"][0]["text"]

    def test_first_request_not_mutated(self):
        llm = ScriptedLLM([text_response(BAD_JSON), text_response(GOOD_JSON)])
        asyncio.run(extract_with_retry("câu hỏi", SeveranceInput, llm))
        assert len(llm.requests[0]["contents"]) == 1

    def test_gives_up(self):
        llm = ScriptedLLM([text_response(BAD_JSON)] * 3)
        with pytest.raises(ExtractionError):
            asyncio.run(extract_with_retry("câu hỏi", SeveranceInput, llm, max_attempts=3))
        assert len(llm.requests) == 3

    def test_broken_json_is_retried(self):
        llm = ScriptedLLM([text_response('{"months_worked": 50,'), text_response(GOOD_JSON)])
        assert asyncio.run(extract_with_retry("q", SeveranceInput, llm)).attempts == 2


# ---------------------------------------------------------------- Bài 5
class TestRunAgent:
    def test_answer_without_tools(self):
        llm = ScriptedLLM([text_response("Thử việc tối đa 60 ngày.")])
        r = asyncio.run(run_agent("Thử việc bao lâu?", llm, REGISTRY))
        assert isinstance(r, AgentResult)
        assert (r.answer, r.steps, r.trace) == ("Thử việc tối đa 60 ngày.", 1, [])
        assert "functionDeclarations" in llm.requests[0]["tools"][0]

    def test_one_tool_round_trip(self):
        llm = ScriptedLLM([call_response(("calc_severance", SEV_ARGS, "c1")), text_response("Bạn được 35 triệu.")])
        r = asyncio.run(run_agent("Tôi được bao nhiêu?", llm, REGISTRY, system="Trợ lý luật"))
        assert r.answer == "Bạn được 35 triệu."
        assert r.steps == 2
        assert r.trace[0].name == "calc_severance" and r.trace[0].result["amount_vnd"] == 35_000_000

        contents = llm.requests[1]["contents"]
        assert [c["role"] for c in contents] == ["user", "model", "user"]
        assert contents[1]["parts"][0]["thoughtSignature"] == "SIG-1", "phải gửi lại nguyên văn lượt của model"
        fr = contents[2]["parts"][0]["functionResponse"]
        assert fr["name"] == "calc_severance" and fr["id"] == "c1"
        assert fr["response"]["amount_vnd"] == 35_000_000
        assert llm.requests[1]["systemInstruction"] == {"parts": [{"text": "Trợ lý luật"}]}

    def test_parallel_calls(self):
        llm = ScriptedLLM([
            call_response(("calc_severance", SEV_ARGS, "a"), ("calc_overtime", OT_ARGS, "b")),
            text_response("xong"),
        ])
        r = asyncio.run(run_agent("hai câu hỏi", llm, REGISTRY))
        assert [t.name for t in r.trace] == ["calc_severance", "calc_overtime"]
        parts = llm.requests[1]["contents"][2]["parts"]
        assert [p["functionResponse"]["id"] for p in parts] == ["a", "b"]
        assert parts[1]["functionResponse"]["response"]["amount_vnd"] == 480_000

    def test_bad_args_then_self_correct(self):
        llm = ScriptedLLM([
            call_response(("calc_severance", {"months_worked": 50})),
            call_response(("calc_severance", SEV_ARGS), signature="SIG-2"),
            text_response("35 triệu"),
        ])
        r = asyncio.run(run_agent("q", llm, REGISTRY))
        assert r.steps == 3
        assert "error" in r.trace[0].result
        assert r.trace[1].result["amount_vnd"] == 35_000_000
        third = llm.requests[2]["contents"]
        assert len(third) == 5
        assert "error" in third[2]["parts"][0]["functionResponse"]["response"]

    def test_unknown_tool_does_not_crash(self):
        llm = ScriptedLLM([call_response(("send_email", {"to": "hr"})), text_response("Tôi không gửi email được.")])
        r = asyncio.run(run_agent("gửi email cho HR", llm, REGISTRY))
        assert r.answer == "Tôi không gửi email được."
        assert "send_email" in r.trace[0].result["error"]

    def test_step_limit(self):
        llm = ScriptedLLM([call_response(("calc_overtime", OT_ARGS))] * 3)
        with pytest.raises(AgentStepLimitError):
            asyncio.run(run_agent("lặp mãi", llm, REGISTRY, max_steps=3))
        assert len(llm.requests) == 3

    def test_history_grows_each_step(self):
        llm = ScriptedLLM([call_response(("calc_overtime", OT_ARGS)), call_response(("calc_overtime", OT_ARGS)), text_response("ok")])
        asyncio.run(run_agent("q", llm, REGISTRY))
        assert [len(b["contents"]) for b in llm.requests] == [1, 3, 5]
