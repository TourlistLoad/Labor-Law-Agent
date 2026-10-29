"""
NGÀY 3 — Chạy agent và structured output với Gemini THẬT, bằng code của bạn.

Chỉ chạy sau khi `uv run pytest learning/day03` đã xanh hết.

    uv run --env-file .env python -m learning.day03.live extract   # trích JSON từ câu tiếng Việt
    uv run --env-file .env python -m learning.day03.live agent     # agent tự chọn tool

Biến môi trường giống Ngày 02: GEMINI_API_KEY, GEMINI_MODEL, GEMINI_THINKING.
Gặp 503 thì script tự thử lại (2s, 4s, 8s, 16s); vẫn lỗi thì đổi GEMINI_MODEL.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx

from learning.day01.exercises import SeveranceInput, compute_severance
from learning.day02.exercises import GEMINI_BASE_URL, RETRYABLE_STATUS, LLMHTTPError
from learning.day03.exercises import (
    REGISTRY,
    AgentStepLimitError,
    ExtractionError,
    extract_with_retry,
    run_agent,
)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
_thinking = os.environ.get("GEMINI_THINKING", "low").strip().lower()
THINKING = None if _thinking == "none" else _thinking
if THINKING not in (None, "low", "medium", "high"):
    sys.exit(f"GEMINI_THINKING={_thinking!r} không hợp lệ. Dùng: low | medium | high | none")

MAX_ATTEMPTS, BASE_DELAY = 5, 2.0

SYSTEM = (
    "Bạn là trợ lý giải đáp Bộ luật Lao động Việt Nam 2019. Trả lời bằng tiếng Việt, ngắn gọn. "
    "Mọi con số tiền phải lấy từ kết quả tool, không tự tính. "
    "Nếu thiếu thông tin để gọi tool thì hỏi lại người dùng."
)

AGENT_QUESTIONS = [
    "Tôi làm ở công ty 4 năm 2 tháng, trong đó 2 năm cuối có đóng bảo hiểm thất nghiệp, "
    "lương bình quân 6 tháng gần nhất là 18 triệu. Nghỉ việc thì trợ cấp thôi việc bao nhiêu?",
    "Lương giờ của tôi là 60.000 đồng. Chủ nhật vừa rồi là ngày nghỉ hằng tuần, tôi làm thêm 4 tiếng. "
    "Tôi được trả ít nhất bao nhiêu tiền làm thêm?",
    "Tôi sắp nghỉ việc, được trợ cấp thôi việc bao nhiêu?",
    "Người lao động có phải báo trước khi nghỉ việc không?",
]

EXTRACT_TEXT = (
    "Em vào công ty từ tháng 3/2021, nghỉ cuối tháng 4/2025. Công ty đóng bảo hiểm thất nghiệp "
    "cho em được 30 tháng. Lương 6 tháng cuối trung bình khoảng 16 triệu rưỡi."
)


def make_llm(client: httpx.AsyncClient, key: str):
    """Trả về hàm llm(body) -> dict gọi Gemini thật, có retry 429/500/503 và lỗi mạng."""
    url = f"{GEMINI_BASE_URL}/models/{MODEL}:generateContent"
    state = {"calls": 0, "legacy_schema": False}

    def add_thinking(body: dict) -> dict:
        if THINKING is None:
            return body
        body = {**body, "generationConfig": {**body.get("generationConfig", {})}}
        body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": THINKING}
        return body

    def to_legacy_schema(body: dict) -> dict:
        # Kiểu khai báo cũ (đã bị đánh dấu deprecated), dùng khi model/endpoint chưa nhận responseFormat
        gen = dict(body["generationConfig"])
        fmt = gen.pop("responseFormat")["text"]
        gen["responseMimeType"] = "application/json"
        gen["responseJsonSchema"] = fmt["schema"]
        return {**body, "generationConfig": gen}

    async def post(body: dict) -> dict:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            state["calls"] += 1
            try:
                resp = await client.post(url, headers={"x-goog-api-key": key}, json=body)
            except httpx.TransportError:
                if attempt == MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(BASE_DELAY * 2 ** (attempt - 1))
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                wait = BASE_DELAY * 2 ** (attempt - 1)
                print(f"    (HTTP {resp.status_code}, chờ {wait:.0f}s rồi thử lại)")
                await asyncio.sleep(wait)
                continue
            raise LLMHTTPError(resp.status_code, resp.text)
        raise AssertionError("unreachable")

    async def llm(body: dict) -> dict:
        body = add_thinking(body)
        if state["legacy_schema"] and "responseFormat" in body.get("generationConfig", {}):
            body = to_legacy_schema(body)
        try:
            return await post(body)
        except LLMHTTPError as e:
            if e.status_code == 400 and "responseFormat" in e.body and not state["legacy_schema"]:
                print("    (endpoint chưa nhận responseFormat, chuyển sang responseJsonSchema kiểu cũ)")
                state["legacy_schema"] = True
                return await post(to_legacy_schema(body))
            raise

    llm.state = state
    return llm


# ---------------------------------------------------------------- extract
async def mode_extract(llm) -> None:
    print(f"\nCâu gốc:\n  {EXTRACT_TEXT}\n")
    try:
        result = await extract_with_retry(EXTRACT_TEXT, SeveranceInput, llm)
    except ExtractionError as e:
        print(f"Model không trả được JSON hợp lệ sau nhiều lần: {e}")
        return
    inp = result.value
    print(f"Trích được sau {result.attempts} lần gọi:")
    print("  " + json.dumps(inp.model_dump(), ensure_ascii=False))
    out = compute_severance(inp)
    print("\nĐưa vào compute_severance (code của bạn, Ngày 01):")
    print("  " + json.dumps(out.model_dump(), ensure_ascii=False))
    print(
        "\nTự kiểm tra: tháng 3/2021 đến hết tháng 4/2025 là bao nhiêu tháng? "
        "Model có quy đổi đúng không? '16 triệu rưỡi' có thành 16500000 không?"
    )


# ---------------------------------------------------------------- agent
async def mode_agent(llm) -> None:
    for i, q in enumerate(AGENT_QUESTIONS, 1):
        print("\n" + "=" * 78)
        print(f"[{i}] {q}")
        try:
            r = await run_agent(q, llm, REGISTRY, system=SYSTEM)
        except AgentStepLimitError as e:
            print(f"  Dừng: {e}")
            continue
        for t in r.trace:
            print(f"  -> gọi {t.name}({json.dumps(t.args, ensure_ascii=False)})")
            print(f"     kết quả: {json.dumps(t.result, ensure_ascii=False)}")
        if not r.trace:
            print("  (không gọi tool nào)")
        print(f"  Trả lời sau {r.steps} lần gọi LLM:\n  {r.answer.strip()}")
    print(
        "\nGhi vào notes: câu nào gọi tool, câu nào không, câu nào model hỏi lại? "
        "Có câu nào model chọn sai, hoặc tự bịa con số không lấy từ tool?"
    )


# ---------------------------------------------------------------- main
MODES = {"extract": mode_extract, "agent": mode_agent}


async def main(mode: str) -> None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("Chưa có GEMINI_API_KEY. Chạy: uv run --env-file .env python -m learning.day03.live " + mode)
    print(f"Model: {MODEL}   thinking: {THINKING or 'không gửi'}")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            llm = make_llm(client, key)
            await MODES[mode](llm)
            print(f"\nTổng số request đã gửi (kể cả retry): {llm.state['calls']}")
    except LLMHTTPError as e:
        print(f"\nDừng vì lỗi: {e}")
        if e.status_code == 503:
            print("Model quá tải. Thử: GEMINI_MODEL=gemini-3.5-flash-lite uv run --env-file .env "
                  f"python -m learning.day03.live {mode}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        sys.exit("Dùng: python -m learning.day03.live [extract|agent]")
    asyncio.run(main(sys.argv[1]))
