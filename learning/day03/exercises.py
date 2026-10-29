"""
NGÀY 3 — Tool calling và structured output viết tay (không framework)
=====================================================================

Cách làm:
    1. Đọc tab "Ngày 03" trong tài liệu lộ trình trước.
    2. Thay mọi `raise NotImplementedError` bằng code của bạn.
    3. Chạy từng bài:  uv run pytest learning/day03 -k <TênClassTest> -v
       Chạy tất cả:    uv run pytest learning/day03 -v
    4. Không dùng AI viết hộ phần TODO.

File này dùng lại code của CHÍNH BẠN:
    - Ngày 01: SeveranceInput, compute_severance  (tool tính trợ cấp)
    - Ngày 02: LLMBlockedError
Vì vậy Ngày 01 phải đang xanh 30/30 test.

Test KHÔNG gọi mạng thật: "LLM" trong test là một hàm giả trả lời theo
kịch bản soạn sẵn. live.py mới gọi Gemini thật.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from learning.day01.exercises import SeveranceInput, compute_severance
from learning.day02.exercises import LLMBlockedError

# "LLM" ở đây là MỌI hàm async nhận request body (dict) và trả response (dict).
# Test truyền vào hàm giả; live.py truyền vào hàm gọi Gemini thật.
# Đây là dependency injection bạn đã học ở Ngày 02.
LLM = Callable[[dict], Awaitable[dict]]


# ===========================================================================
# PHẦN CHO SẴN — đọc kỹ, không cần sửa
# ===========================================================================

@dataclass(frozen=True)
class ToolSpec:
    """Mô tả 1 tool: tên, mô tả cho LLM, khuôn input (Pydantic), và hàm Python chạy thật."""
    name: str
    description: str
    input_model: type[BaseModel]
    func: Callable[[BaseModel], BaseModel]


class FunctionCall(BaseModel):
    """1 yêu cầu gọi tool mà model gửi về."""
    name: str
    args: dict = Field(default_factory=dict)
    id: str | None = None          # Gemini có thể gửi kèm id; có thì phải gửi lại đúng id


class ModelTurn(BaseModel):
    """1 lượt trả lời của model, đã bóc tách."""
    content: dict                  # NGUYÊN VĂN content của model, để gửi lại vào history
    text: str                      # phần chữ (bỏ part suy nghĩ)
    calls: list[FunctionCall]      # các yêu cầu gọi tool (có thể rỗng)
    finish_reason: str


class ToolTrace(BaseModel):
    """Ghi lại 1 lần chạy tool, để in ra và debug."""
    name: str
    args: dict
    result: dict


class AgentResult(BaseModel):
    answer: str
    steps: int                     # số lần đã gọi LLM
    trace: list[ToolTrace]


class ExtractionResult(BaseModel):
    value: BaseModel
    attempts: int


class AgentStepLimitError(Exception):
    """Agent gọi LLM quá max_steps lần mà vẫn chưa trả lời xong."""


class ExtractionError(Exception):
    """Hết lượt thử mà model vẫn không trả JSON hợp lệ."""


def user_turn(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


def validation_messages(e: ValidationError) -> list[str]:
    """Giống bài 4 Ngày 01: ValidationError -> list chuỗi 'đường_dẫn: lỗi'."""
    return [f"{'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}" for err in e.errors()]


def retry_message(errors: list[str]) -> dict:
    """Lượt 'user' báo cho model biết JSON vừa rồi sai ở đâu."""
    return user_turn(
        "JSON bạn vừa trả chưa hợp lệ:\n- " + "\n- ".join(errors) + "\nHãy trả lại JSON đã sửa, không giải thích."
    )


def build_extraction_request(contents: list[dict], model_cls: type[BaseModel]) -> dict:
    """Request structured output: ép model trả JSON đúng schema của model_cls."""
    return {
        "contents": list(contents),
        "generationConfig": {
            "responseFormat": {
                "text": {"mimeType": "APPLICATION_JSON", "schema": model_cls.model_json_schema()}
            }
        },
    }


# ---- Tool thứ 2: lương làm thêm giờ (Điều 98 BLLĐ 2019, đơn giản hoá) ----

class OvertimeInput(BaseModel):
    model_config = {"extra": "forbid"}
    hourly_wage_vnd: int = Field(gt=0, description="Tiền lương giờ của ngày làm việc bình thường, đơn vị đồng")
    hours: float = Field(gt=0, le=24, description="Số giờ làm thêm")
    day_type: Literal["weekday", "weekly_off", "holiday"] = Field(
        description="weekday = ngày thường; weekly_off = ngày nghỉ hằng tuần; holiday = ngày lễ, tết"
    )


class OvertimeResult(BaseModel):
    rate_percent: int
    amount_vnd: int
    note: str


OVERTIME_RATE = {"weekday": 150, "weekly_off": 200, "holiday": 300}


def compute_overtime(inp: OvertimeInput) -> OvertimeResult:
    rate = OVERTIME_RATE[inp.day_type]
    return OvertimeResult(
        rate_percent=rate,
        amount_vnd=int(inp.hourly_wage_vnd * inp.hours * rate / 100),
        note=f"Mức tối thiểu {rate}% lương giờ, chưa tính làm đêm.",
    )


SEVERANCE_TOOL = ToolSpec(
    name="calc_severance",
    description=(
        "Tính tiền trợ cấp thôi việc (VND) theo quy tắc đơn giản hoá của hệ thống. "
        "Gọi khi người dùng muốn biết số tiền trợ cấp thôi việc cụ thể và đã cho biết "
        "số tháng làm việc cùng mức lương bình quân 6 tháng gần nhất. "
        "Nếu thiếu một trong hai thông tin đó thì hỏi lại người dùng, không tự giả định. "
        "Không dùng cho trợ cấp mất việc làm hoặc trợ cấp thất nghiệp."
    ),
    input_model=SeveranceInput,
    func=compute_severance,
)

OVERTIME_TOOL = ToolSpec(
    name="calc_overtime",
    description=(
        "Tính tiền lương làm thêm giờ tối thiểu (VND). Gọi khi người dùng hỏi số tiền làm thêm "
        "và đã cho biết lương giờ, số giờ làm thêm, loại ngày. Thiếu thông tin thì hỏi lại."
    ),
    input_model=OvertimeInput,
    func=compute_overtime,
)

REGISTRY: dict[str, ToolSpec] = {t.name: t for t in (SEVERANCE_TOOL, OVERTIME_TOOL)}


# ===========================================================================
# BÀI 1 — Khai báo tool cho Gemini
# ===========================================================================
# (a) tool_declaration(spec) trả về:
#       {"name": ..., "description": ..., "parametersJsonSchema": <JSON Schema của spec.input_model>}
#
# (b) build_agent_request trả về:
#   {
#     "contents": <BẢN SAO list contents, không phải chính list đó>,
#     "tools": [{"functionDeclarations": [<tool_declaration của từng tool>]}],
#     "toolConfig": {"functionCallingConfig": {"mode": mode}},
#     "systemInstruction": {"parts": [{"text": system}]},   # CHỈ khi system khác None
#   }
#   Chú ý: "tools" là list chứa MỘT dict; mọi tool nằm chung trong "functionDeclarations".

def tool_declaration(spec: ToolSpec) -> dict:
    return {
        "name": spec.name,
        "description": spec.description,
        "parametersJsonSchema": spec.input_model.model_json_schema()
    }


def build_agent_request(
    contents: list[dict],
    tools: list[ToolSpec],
    system: str | None = None,
    mode: Literal["AUTO", "ANY", "NONE"] = "AUTO",
) -> dict:
    request = {
        "contents": list(contents),
        "tools": [
            {
                "functionDeclarations": [tool_declaration(tool) for tool in tools]
            }
        ],
        "toolConfig": {
            "functionCallingConfig": {
                "mode": mode
            }
        }
    }
    if system is not None:
        request["systemInstruction"] = {
            "parts": [
                {
                    "text": system
                }
            ]
        }
    return request


# ===========================================================================
# BÀI 2 — Đọc lượt trả lời của model: chữ hay yêu cầu gọi tool?
# ===========================================================================
# Model có thể trả về chữ, hoặc yêu cầu gọi tool, hoặc cả hai. Ví dụ:
#   {"candidates": [{
#       "content": {"role": "model", "parts": [
#           {"functionCall": {"name": "calc_severance", "args": {...}, "id": "c1"},
#            "thoughtSignature": "Abc..."}
#       ]},
#       "finishReason": "STOP"}]}
#
# parse_model_turn(data) -> ModelTurn:
#   - candidates vắng/rỗng -> raise LLMBlockedError(blockReason hoặc "UNKNOWN")   (như Ngày 02)
#   - content  = candidates[0]["content"] NGUYÊN VĂN (không sửa, không copy bớt)
#   - text     = nối các part có "text" và không có "thought": True             (như Ngày 02)
#   - calls    = mỗi part có khoá "functionCall" -> FunctionCall(name, args, id)
#                args vắng -> {} ; id vắng -> None
#   - finish_reason = "finishReason", vắng thì "UNKNOWN"
#
# Vì sao giữ content NGUYÊN VĂN? Part functionCall mang "thoughtSignature".
# Gemini 3 BẮT BUỘC bạn gửi lại đúng signature đó ở lượt sau, thiếu là lỗi 400.

def parse_model_turn(data: dict) -> ModelTurn:
    candidates = data.get("candidates", [])
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
        raise LLMBlockedError(reason)

    candidate = candidates[0]
    content = candidate.get("content", {})
    parts = content.get("parts", [])

    pieces = []
    calls = []
    for p in parts:
        if p.get("text") and not p.get("thought"):
            pieces.append(p.get("text"))
        if p.get("functionCall"):
            func = p.get("functionCall")
            calls.append(FunctionCall(name=func.get("name"), args=func.get("args", {}), id=func.get("id", None)))
    text = "".join(pieces)
    finish_reason = candidate.get("finishReason", "UNKNOWN")
    return ModelTurn(content=content, text=text, calls=calls, finish_reason=finish_reason)



# ===========================================================================
# BÀI 3 — Chạy tool và đóng gói kết quả gửi lại model
# ===========================================================================
# (a) execute_call(call, registry) -> dict.  KHÔNG BAO GIỜ raise: mọi lỗi
#     đều trở thành dict để model đọc và tự sửa.
#       - tên tool không có trong registry ->
#             {"error": "Không có tool tên '<tên>'. ..."}   (chuỗi phải chứa tên tool)
#       - args không qua được spec.input_model.model_validate(call.args) ->
#             {"error": "Tham số không hợp lệ, hãy sửa và gọi lại.",
#              "details": validation_messages(e)}
#       - spec.func(...) tự nó raise lỗi gì đó ->
#             {"error": "Tool gặp lỗi khi chạy: <lỗi>"}
#       - thành công -> kết quả của func, đổi sang dict bằng .model_dump()
#
# (b) function_response_turn(calls, results) -> dict:
#   {"role": "user", "parts": [
#       {"functionResponse": {"name": <call.name>, "response": <result>, "id": <call.id>}},
#       ...                                       # "id" CHỈ có khi call.id khác None
#   ]}
#   Một part cho mỗi call, ĐÚNG thứ tự calls. (Gợi ý: zip(calls, results))

def execute_call(call: FunctionCall, registry: dict[str, ToolSpec]) -> dict:
    # TODO
    raise NotImplementedError


def function_response_turn(calls: list[FunctionCall], results: list[dict]) -> dict:
    # TODO
    raise NotImplementedError


# ===========================================================================
# BÀI 4 — Structured output + vòng tự sửa lỗi (lời hứa của Ngày 01)
# ===========================================================================
# extract_with_retry: nhờ model đọc câu tiếng Việt và trích ra JSON đúng
# model_cls (ví dụ SeveranceInput). Sai thì gửi lỗi lại cho model sửa.
#
#   contents = [user_turn(text)]
#   lặp attempt = 1 .. max_attempts:
#       data = await llm(build_extraction_request(contents, model_cls))
#       turn = parse_model_turn(data)
#       thử  model_cls.model_validate_json(turn.text)
#            thành công -> return ExtractionResult(value=..., attempts=attempt)
#            ValidationError e -> errors = validation_messages(e)
#                                 contents = contents + [turn.content, retry_message(errors)]
#   hết lượt -> raise ExtractionError(errors của lần cuối)
#
# Chú ý: tạo list MỚI (contents + [...]) thay vì append, để không sửa
# list mà llm đã nhận ở lần gọi trước.

async def extract_with_retry(
    text: str,
    model_cls: type[BaseModel],
    llm: LLM,
    max_attempts: int = 3,
) -> ExtractionResult:
    # TODO
    raise NotImplementedError


# ===========================================================================
# BÀI 5 — Agent loop: ghép tất cả lại
# ===========================================================================
#   contents = [user_turn(question)];  trace = []
#   lặp step = 1 .. max_steps:
#       data = await llm(build_agent_request(contents, list(registry.values()), system))
#       turn = parse_model_turn(data)
#       không có turn.calls  -> model đã trả lời xong:
#                               return AgentResult(answer=turn.text, steps=step, trace=trace)
#       có calls -> results = [execute_call(c, registry) cho từng c]
#                   thêm ToolTrace(name, args, result) cho từng cặp vào trace
#                   contents = contents + [turn.content, function_response_turn(turn.calls, results)]
#   hết vòng lặp -> raise AgentStepLimitError(...)
#
# Đây là TOÀN BỘ "agent" mà create_agent của LangChain làm cho bạn ở Ngày 04.

async def run_agent(
    question: str,
    llm: LLM,
    registry: dict[str, ToolSpec],
    system: str | None = None,
    max_steps: int = 5,
) -> AgentResult:
    # TODO
    raise NotImplementedError
