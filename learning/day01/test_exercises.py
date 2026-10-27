"""Test tự chấm cho Ngày 1. KHÔNG sửa file này.

Chạy:  uv run pytest learning/day01 -v
"""

import pytest
from pydantic import ValidationError

from learning.day01.exercises import (
    ChatMessage,
    SeveranceInput,
    SeveranceResult,
    compute_severance,
    parse_llm_output,
    tool_schema,
)


# ---------------------------------------------------------------- Bài 1
class TestChatMessage:
    def test_user_message_ok(self):
        m = ChatMessage(role="user", content="Tôi nghỉ việc có được trợ cấp không?")
        assert m.role == "user"
        assert m.tool_call_id is None

    def test_tool_message_ok(self):
        m = ChatMessage(role="tool", content='{"amount_vnd": 1}', tool_call_id="call_1")
        assert m.tool_call_id == "call_1"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="admin", content="hi")

    def test_empty_content(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")

    def test_tool_without_id(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="tool", content="result")

    def test_non_tool_with_id(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="hi", tool_call_id="call_1")


# ---------------------------------------------------------------- Bài 2
class TestSeveranceInput:
    def test_valid(self):
        x = SeveranceInput(months_worked=24, avg_salary_vnd=10_000_000)
        assert x.months_unemployment_insurance == 0

    def test_coerce_numeric_string(self):
        # Pydantic mặc định "lax mode": "24" -> 24. Biết điều này để không bất ngờ.
        x = SeveranceInput(months_worked="24", avg_salary_vnd=10_000_000)
        assert x.months_worked == 24

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"months_worked": -1, "avg_salary_vnd": 1},
            {"months_worked": 601, "avg_salary_vnd": 1},
            {"months_worked": 12, "avg_salary_vnd": 0},
            {"months_worked": 12, "avg_salary_vnd": 1, "months_unemployment_insurance": -1},
        ],
    )
    def test_out_of_range(self, kwargs):
        with pytest.raises(ValidationError):
            SeveranceInput(**kwargs)

    def test_insurance_exceeds_worked(self):
        with pytest.raises(ValidationError):
            SeveranceInput(months_worked=12, months_unemployment_insurance=13, avg_salary_vnd=1)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            SeveranceInput(months_worked=12, avg_salary_vnd=1, bonus=5)

    def test_every_field_has_description(self):
        props = SeveranceInput.model_json_schema()["properties"]
        assert set(props) == {"months_worked", "months_unemployment_insurance", "avg_salary_vnd"}
        for name, spec in props.items():
            assert len(spec.get("description", "")) >= 15, f"{name} thiếu description đủ rõ"


# ---------------------------------------------------------------- Bài 3
class TestComputeSeverance:
    @pytest.mark.parametrize(
        "worked, bhtn, salary, eligible, years, amount",
        [
            (11, 0, 10_000_000, False, 0.0, 0),
            (12, 0, 10_000_000, True, 1.0, 5_000_000),
            (50, 12, 20_000_000, True, 3.5, 35_000_000),
            (30, 0, 15_000_000, True, 2.5, 18_750_000),
            (31, 0, 15_000_000, True, 3.0, 22_500_000),
            (36, 36, 15_000_000, True, 0.0, 0),
            (25, 0, 9_999_999, True, 2.5, 12_499_998),
        ],
    )
    def test_cases(self, worked, bhtn, salary, eligible, years, amount):
        r = compute_severance(
            SeveranceInput(
                months_worked=worked,
                months_unemployment_insurance=bhtn,
                avg_salary_vnd=salary,
            )
        )
        assert isinstance(r, SeveranceResult)
        assert r.eligible is eligible
        assert r.years_counted == pytest.approx(years)
        assert r.amount_vnd == amount
        assert isinstance(r.amount_vnd, int)
        assert r.note.strip()


# ---------------------------------------------------------------- Bài 4
class TestParseLlmOutput:
    def test_plain_json(self):
        r = parse_llm_output('{"months_worked": 30, "avg_salary_vnd": 15000000}')
        assert isinstance(r, SeveranceInput)
        assert r.months_worked == 30

    def test_fenced_json(self):
        raw = '```json\n{"months_worked": 30, "avg_salary_vnd": 15000000}\n```'
        assert isinstance(parse_llm_output(raw), SeveranceInput)

    def test_fenced_no_lang(self):
        raw = '  ```\n{"months_worked": 30, "avg_salary_vnd": 15000000}\n```  '
        assert isinstance(parse_llm_output(raw), SeveranceInput)

    def test_field_errors_are_reported(self):
        r = parse_llm_output('{"months_worked": 700, "avg_salary_vnd": -5}')
        assert isinstance(r, list)
        assert len(r) == 2
        assert any(e.startswith("months_worked:") for e in r)
        assert any(e.startswith("avg_salary_vnd:") for e in r)

    def test_missing_field(self):
        r = parse_llm_output('{"months_worked": 20}')
        assert isinstance(r, list)
        assert any(e.startswith("avg_salary_vnd:") for e in r)

    def test_cross_field_error_is_root(self):
        r = parse_llm_output(
            '{"months_worked": 12, "months_unemployment_insurance": 20, "avg_salary_vnd": 1}'
        )
        assert isinstance(r, list)
        assert any(e.startswith("<root>:") for e in r)

    def test_broken_json(self):
        r = parse_llm_output('{"months_worked": 12, ')
        assert isinstance(r, list) and r


# ---------------------------------------------------------------- Bài 5
class TestToolSchema:
    def test_shape(self):
        s = tool_schema()
        assert s["name"] == "calc_severance"
        assert len(s["description"]) >= 30
        params = s["parameters"]
        assert params["type"] == "object"
        assert "months_worked" in params["properties"]
        assert set(params["required"]) == {"months_worked", "avg_salary_vnd"}
        assert params.get("additionalProperties") is False
