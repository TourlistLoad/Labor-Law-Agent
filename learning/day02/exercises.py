"""
NGÀY 2 — async/await + gọi LLM bằng HTTP thô (không SDK)
=========================================================
 
Cách làm:
    1. Đọc tab "Ngày 02" trong tài liệu lộ trình trước.
    2. Thay mọi `raise NotImplementedError` / `...` bằng code của bạn.
    3. Chạy từng bài:  uv run pytest learning/day02 -k <TênClassTest> -v
       Chạy tất cả:    uv run pytest learning/day02 -v
    4. Không dùng AI viết hộ phần TODO. Được tra docs Python, httpx, Pydantic.
 
Test KHÔNG gọi mạng thật. Sau khi xanh hết, file live.py dùng chính các
hàm bạn viết để gọi Gemini thật.
"""
 
from __future__ import annotations
 
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal
 
import httpx
from pydantic import BaseModel, Field
 
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
 
 
# ---------------------------------------------------------------------------
# BÀI 1 — Tuần tự vs đồng thời
# ---------------------------------------------------------------------------
# fake_llm đã viết sẵn: nó giả lập 1 lần gọi LLM, "chờ mạng" `delay` giây
# rồi trả lời. Dùng asyncio.sleep (KHÔNG chặn), không phải time.sleep.
#
# Yêu cầu:
#   run_sequential: gọi fake_llm cho TỪNG câu, câu sau đợi câu trước xong.
#       Gợi ý: vòng for + await. 5 câu x 0.1s -> mất >= 0.5s.
#   run_concurrent: gọi tất cả CÙNG LÚC.
#       Gợi ý: asyncio.gather(*...). 5 câu x 0.1s -> mất ~0.1s.
#   Cả hai trả về list[str], ĐÚNG thứ tự câu hỏi đầu vào.
#
# Câu hỏi suy nghĩ: vì sao run_concurrent nhanh hơn dù chỉ có 1 luồng (thread)?
 
async def fake_llm(question: str, delay: float = 0.2) -> str:
    """ĐÃ VIẾT SẴN — giả lập 1 lần gọi LLM."""
    await asyncio.sleep(delay)
    return f"answer: {question}"
 
 
async def run_sequential(questions: list[str], delay: float = 0.2) -> list[str]:
    responses = []
    for q in questions:
        resp = await fake_llm(q)
        responses.append(resp)
    return responses
 
 
async def run_concurrent(questions: list[str], delay: float = 0.2) -> list[str]:
    responses = await asyncio.gather(*[fake_llm(q) for q in questions])
    return responses
 
 
# ---------------------------------------------------------------------------
# BÀI 2 — Đồng thời có giới hạn (Semaphore)
# ---------------------------------------------------------------------------
# Free tier Gemini chỉ cho khoảng 10 request/phút. Bắn 100 request cùng lúc
# -> 429 Too Many Requests. Cần "tối đa N việc chạy cùng lúc".
#
# Yêu cầu:
#   - limit < 1 -> raise ValueError
#   - Gọi `call(q)` cho mọi câu hỏi, nhưng không bao giờ quá `limit` lời gọi
#     đang chạy cùng một lúc.
#   - Trả về list kết quả ĐÚNG thứ tự câu hỏi.
#
# Gợi ý:
#   sem = asyncio.Semaphore(limit)
#   async def one(q):              # hàm con bọc 1 lời gọi
#       async with sem:            # chờ tới lượt; ra khỏi khối thì nhả lượt
#           return await call(q)
#   rồi gather tất cả one(q).
#
# `call` là một hàm async nhận 1 câu hỏi. Bài này không quan tâm call làm gì:
# trong test nó là hàm giả, trong live.py nó gọi Gemini thật.
 
async def run_limited(
    questions: list[str],
    limit: int,
    call: Callable[[str], Awaitable[Any]],
) -> list[Any]:
    if limit < 1:
        raise ValueError("Số lượng request/phút phải > 0")
   
    sem = asyncio.Semaphore(limit)
    async def one(q):
        async with sem:
            return await call(q)
 
    responses = await asyncio.gather(*[one(q) for q in questions])
    return responses
 
 
# ---------------------------------------------------------------------------
# BÀI 3 — Tự dựng request body gửi Gemini (generateContent)
# ---------------------------------------------------------------------------
# (a) GenerationSettings — ôn lại Pydantic Ngày 01:
#       temperature: float hoặc None, 0 <= x <= 2, mặc định None
#           (Google khuyên giữ mặc định 1.0 cho Gemini 3, nên None = không gửi)
#       max_output_tokens: int, > 0, mặc định 1024
#       thinking_level: "low" | "medium" | "high" hoặc None, mặc định None
#
# (b) ChatTurn đã viết sẵn: 1 lượt hội thoại cũ. role của Gemini là
#     "user" hoặc "model" (KHÔNG phải "assistant").
#
# (c) build_request trả về dict ĐÚNG tên khoá của Gemini (camelCase):
#   {
#     "systemInstruction": {"parts": [{"text": system}]},   # CHỈ có khi system khác None
#     "contents": [
#         ...mỗi ChatTurn trong history -> {"role": ..., "parts": [{"text": ...}]},
#         {"role": "user", "parts": [{"text": question}]},  # câu hỏi mới LUÔN ở cuối
#     ],
#     "generationConfig": {
#         "maxOutputTokens": ...,
#         "temperature": ...,                                # CHỈ có khi temperature khác None
#         "thinkingConfig": {"thinkingLevel": ...},         # CHỈ có khi thinking_level khác None
#     },
#   }
#   Không được sửa (append vào) list history truyền vào.
#
# Câu hỏi suy nghĩ: vì sao phải gửi lại TOÀN BỘ history mỗi lần gọi?
 
class GenerationSettings(BaseModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_output_tokens: int = Field(default=1024, gt=0)
    thinking_level: Literal["low", "medium", "high"] | None = None
 
 
class ChatTurn(BaseModel):
    """ĐÃ VIẾT SẴN."""
    role: Literal["user", "model"]
    text: str = Field(min_length=1)
 
 
def build_request(
    question: str,
    settings: GenerationSettings,
    system: str | None = None,
    history: list[ChatTurn] | None = None,
) -> dict:
    request_dict = {}
    if system is not None:
        request_dict["systemInstruction"] = {
            "parts": [
                {
                    "text": system
                }
            ]
        }


    request_dict["contents"] = []
    if history is not None:
        for h in history:
            request_dict["contents"].append({
                "role": h.role,
                "parts": [
                    {
                        "text": h.text
                    }
                ]
            })

    request_dict["contents"].append({
        "role": "user",
        "parts": [
            {
                "text": question
            }
        ]
    })
 
    request_dict["generationConfig"] = {
        "maxOutputTokens": settings.max_output_tokens
    }
    if settings.temperature is not None:
        request_dict["generationConfig"]["temperature"] = settings.temperature
    if settings.thinking_level is not None:
        request_dict["generationConfig"]["thinkingConfig"] = {
            "thinkingLevel": settings.thinking_level
        }
 
    return request_dict
 
 
# ---------------------------------------------------------------------------
# BÀI 4 — Đọc response của Gemini + tính chi phí
# ---------------------------------------------------------------------------
# Response thành công trông như sau (rút gọn):
#   {
#     "candidates": [{
#         "content": {"role": "model", "parts": [{"text": "..."}, ...]},
#         "finishReason": "STOP"             # hoặc "MAX_TOKENS", "SAFETY", ...
#     }],
#     "usageMetadata": {
#         "promptTokenCount": 120,           # token bạn gửi đi
#         "candidatesTokenCount": 40,        # token câu trả lời
#         "thoughtsTokenCount": 200,         # token "suy nghĩ" — CÓ THỂ VẮNG
#         "totalTokenCount": 360
#     }
#   }
#
# parse_response(data) -> LLMResult:
#   - candidates vắng hoặc rỗng -> prompt bị chặn:
#       raise LLMBlockedError(<data["promptFeedback"]["blockReason"]>, hoặc "UNKNOWN")
#   - text = nối các part của candidates[0] mà:
#       có khoá "text"  VÀ  không có "thought": True  (đó là tóm tắt suy nghĩ)
#     (part không có "text", ví dụ {"thoughtSignature": "..."}, bỏ qua)
#   - finish_reason: lấy "finishReason", vắng thì "UNKNOWN"
#   - token: khoá nào vắng thì coi là 0  (gợi ý: dict.get(key, 0))
#   - finishReason == "MAX_TOKENS" KHÔNG phải lỗi: vẫn trả LLMResult.
#
# estimate_cost_usd(result, input_per_million, output_per_million):
#   giá tính theo 1 triệu token; token suy nghĩ tính giá như token output.
#   cost = prompt * giá_in / 1e6 + (output + thinking) * giá_out / 1e6
 
class LLMResult(BaseModel):
    """ĐÃ VIẾT SẴN."""
    text: str
    finish_reason: str
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int = 0
    total_tokens: int
 
 
class LLMBlockedError(Exception):
    """ĐÃ VIẾT SẴN — prompt bị Gemini chặn, không có câu trả lời."""
 
 
def parse_response(data: dict) -> LLMResult:
    candidates = data.get("candidates", [])
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
        raise LLMBlockedError(reason)
 
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
   
    pieces = []
    for p in parts:
        if "text" in p and not p.get("thought"):
            pieces.append(p["text"])
    text = "".join(pieces)
 
    finish_reason = "UNKNOWN"
    if "finishReason" in candidate:
        finish_reason = candidate["finishReason"]
 
    usageMetadata = data["usageMetadata"]
    prompt_tokens = usageMetadata.get("promptTokenCount", 0)
    output_tokens = usageMetadata.get("candidatesTokenCount", 0)
    thinking_tokens = usageMetadata.get("thoughtsTokenCount", 0)
    total_tokens = usageMetadata.get("totalTokenCount", 0)
 
    return LLMResult(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        total_tokens=total_tokens
    )
 
 
def estimate_cost_usd(result: LLMResult, input_per_million: float, output_per_million: float) -> float:
    cost = result.prompt_tokens * input_per_million / 1e6 + (result.output_tokens + result.thinking_tokens) * output_per_million / 1e6
    return cost
 
# ---------------------------------------------------------------------------
# BÀI 5 — Gọi Gemini bằng httpx, có retry + exponential backoff
# ---------------------------------------------------------------------------
# Yêu cầu:
#   - POST tới  f"{GEMINI_BASE_URL}/models/{model}:generateContent"
#     header  {"x-goog-api-key": api_key}   (KHÔNG đặt key trên URL)
#     body    json=body
#   - 200 -> return parse_response(resp.json())
#   - status trong RETRYABLE_STATUS (429, 500, 503) -> chờ rồi thử lại
#   - lỗi mạng (httpx.TransportError, ví dụ timeout) -> chờ rồi thử lại
#   - status khác (400, 401, 403, 404...) -> raise LLMHTTPError NGAY, không thử lại
#   - Tối đa `max_attempts` lần gửi. Hết lượt:
#       lỗi HTTP -> raise LLMHTTPError(resp.status_code, resp.text)
#       lỗi mạng -> raise lại chính lỗi đó (`raise` trống trong except)
#   - Thời gian chờ trước lần thử thứ n+1:  base_delay * 2 ** (n - 1)
#       base_delay=1 -> chờ 1s, rồi 2s, rồi 4s...
#   - Dùng tham số `sleep` để chờ (await sleep(giây)), KHÔNG gọi thẳng
#     asyncio.sleep. Test truyền vào hàm sleep giả để chạy nhanh và ghi lại
#     bạn đã chờ bao lâu.
#
# Khung gợi ý:
#   for attempt in range(1, max_attempts + 1):
#       try:
#           resp = await client.post(...)
#       except httpx.TransportError:
#           ...
#       ...
 
RETRYABLE_STATUS = {429, 500, 503}
 
 
class LLMHTTPError(Exception):
    """ĐÃ VIẾT SẴN."""
 
    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body[:300]}")
        self.status_code = status_code
        self.body = body
 
 
async def call_gemini(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    model: str,
    body: dict,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> LLMResult:
    for attempt in range(1, max_attempts + 1):
        try:
            url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"
            resp = await client.post(url=url, headers={"x-goog-api-key": api_key}, json=body)
        except httpx.TransportError:
            if attempt == max_attempts:
                raise 
            else:
                await sleep(base_delay * 2 ** (attempt - 1))
                continue

        if resp.status_code == 200:
            return parse_response(resp.json())

        if resp.status_code in RETRYABLE_STATUS:
            if attempt == max_attempts:
                raise LLMHTTPError(resp.status_code, resp.text)
            else:
                await sleep(base_delay * 2 ** (attempt - 1))
                continue

        raise LLMHTTPError(resp.status_code, resp.text)

 