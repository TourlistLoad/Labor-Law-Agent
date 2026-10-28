"""
NGÀY 2 — Gọi Gemini THẬT bằng chính các hàm bạn viết trong exercises.py.

Chỉ chạy sau khi `uv run pytest learning/day02` đã xanh hết.

Cách chạy (từ thư mục gốc repo, file .env có GEMINI_API_KEY=...):
    uv run --env-file .env python -m learning.day02.live raw       # xem request/response thô
    uv run --env-file .env python -m learning.day02.live speed     # tuần tự vs đồng thời
    uv run --env-file .env python -m learning.day02.live history   # LLM không có trí nhớ

Mỗi chế độ gửi khoảng 1-8 request (chưa tính retry). Free tier giới hạn
khoảng 10 request/phút: gặp lỗi 429 thì đợi 1 phút rồi chạy lại.
Gặp 503 (model quá tải) thì script tự thử lại, chờ 2s, 4s, 8s, 16s.
Vẫn 503 thì đợi vài phút, hoặc đổi sang model khác (xem bên dưới).

Biến môi trường:
    GEMINI_API_KEY    bắt buộc
    GEMINI_MODEL      tuỳ chọn, mặc định gemini-3.8-flash
    GEMINI_THINKING   tuỳ chọn: low (mặc định) | medium | high | none
                      none = không gửi thinkingConfig, dùng cho model không
                      nhận thinkingLevel (ví dụ dòng 2.5)

Ví dụ đổi model khi gemini-3.8-flash quá tải:
    GEMINI_MODEL=gemini-3.5-flash-lite uv run --env-file .env python -m learning.day02.live raw
    GEMINI_MODEL=gemini-2.5-flash GEMINI_THINKING=none uv run --env-file .env python -m learning.day02.live raw
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import httpx

from learning.day02.exercises import (
    GEMINI_BASE_URL,
    RETRYABLE_STATUS,
    ChatTurn,
    GenerationSettings,
    LLMHTTPError,
    LLMResult,
    build_request,
    call_gemini,
    estimate_cost_usd,
    run_limited,
)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
_thinking = os.environ.get("GEMINI_THINKING", "low").strip().lower()
THINKING = None if _thinking == "none" else _thinking

# Chạy thật thì chờ lâu hơn trong test: 5 lần gửi, chờ 2s, 4s, 8s, 16s.
# Model quá tải (503) thường cần vài chục giây để hồi lại.
RETRY = {"max_attempts": 5, "base_delay": 2.0}

# Giá trả phí của gemini-3.8-flash, USD / 1 triệu token, áp dụng đến 31/12/2026
# (nguồn: ai.google.dev/gemini-api/docs/pricing). Free tier không mất tiền;
# con số này để bạn hình dung chi phí khi lên production.
PRICE_INPUT_PER_M = 0.75
PRICE_OUTPUT_PER_M = 3.75

SYSTEM = (
    "Bạn là trợ lý giải đáp Bộ luật Lao động Việt Nam. "
    "Trả lời bằng tiếng Việt, tối đa 3 câu, nêu số Điều nếu bạn chắc chắn."
)
# Không đặt temperature: Google khuyên giữ mặc định 1.0 cho dòng Gemini 3.
# thinking_level="low": câu hỏi tra cứu ngắn không cần suy nghĩ lâu, đỡ tốn token.
try:
    SETTINGS = GenerationSettings(max_output_tokens=1024, thinking_level=THINKING)
except ValueError:
    sys.exit(f"GEMINI_THINKING={_thinking!r} không hợp lệ. Dùng: low | medium | high | none")

QUESTIONS = [
    "Thời gian thử việc tối đa với công việc cần trình độ đại học là bao lâu?",
    "Người lao động có được đơn phương nghỉ việc mà không cần báo trước không?",
    "Làm thêm giờ vào ngày nghỉ hằng tuần được trả ít nhất bao nhiêu phần trăm lương?",
    "Lao động nữ được nghỉ thai sản bao nhiêu tháng?",
]


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit(
            "Chưa có GEMINI_API_KEY.\n"
            "Tạo file .env ở gốc repo với dòng: GEMINI_API_KEY=...\n"
            "rồi chạy: uv run --env-file .env python -m learning.day02.live raw"
        )
    return key


def _short(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------- raw
async def mode_raw(client: httpx.AsyncClient, key: str) -> None:
    """Gửi 1 request bằng httpx thuần và in nguyên văn cả hai chiều."""
    body = build_request(QUESTIONS[0], SETTINGS, system=SYSTEM)
    url = f"{GEMINI_BASE_URL}/models/{MODEL}:generateContent"

    print("=" * 70)
    print(f"REQUEST  POST {url}")
    print("Header   x-goog-api-key: " + key[:4] + "…(ẩn)")
    print("Body:")
    print(json.dumps(body, ensure_ascii=False, indent=2))

    print("=" * 70)
    # Vòng thử lại viết thẳng ở đây (không qua call_gemini) để bạn nhìn thấy
    # từng lần gửi: status nào, mất bao lâu, chờ bao lâu trước lần sau.
    for attempt in range(1, RETRY["max_attempts"] + 1):
        t0 = time.perf_counter()
        resp = await client.post(url, headers={"x-goog-api-key": key}, json=body)
        elapsed = time.perf_counter() - t0
        print(f"Lần gửi {attempt}: status {resp.status_code}  ({elapsed:.2f}s)")
        if resp.status_code not in RETRYABLE_STATUS or attempt == RETRY["max_attempts"]:
            break
        wait = RETRY["base_delay"] * 2 ** (attempt - 1)
        print(f"          lỗi tạm thời, chờ {wait:.0f}s rồi thử lại...")
        await asyncio.sleep(wait)

    print("=" * 70)
    print(f"RESPONSE status {resp.status_code}  (lần gửi cuối mất {elapsed:.2f}s)")
    try:
        data = resp.json()
    except ValueError:
        print(resp.text)
        return
    print(json.dumps(data, ensure_ascii=False, indent=2)[:4000])
    print("=" * 70)
    print("Đọc kỹ: candidates[0].content.parts, finishReason, usageMetadata.")
    print("Có thoughtsTokenCount không? Có part nào mang thoughtSignature không?")


# ---------------------------------------------------------------- speed
def _print_table(title: str, results: list[LLMResult], latencies: list[float], wall: float) -> None:
    print(f"\n{title}  (tổng thời gian thực: {wall:.2f}s)")
    print(f"{'#':>2} {'giây':>6} {'finish':>10} {'prompt':>7} {'output':>7} {'think':>6}  trả lời")
    for i, (r, lat) in enumerate(zip(results, latencies), 1):
        print(
            f"{i:>2} {lat:>6.2f} {r.finish_reason:>10} {r.prompt_tokens:>7} "
            f"{r.output_tokens:>7} {r.thinking_tokens:>6}  {_short(r.text, 60)}"
        )
    cost = sum(estimate_cost_usd(r, PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M) for r in results)
    print(f"Ước tính nếu trả phí: ${cost:.6f}  (≈ {cost * 26_000:.0f} VND, tỉ giá giả định 26.000)")


async def mode_speed(client: httpx.AsyncClient, key: str) -> None:
    async def ask(q: str) -> tuple[LLMResult, float]:
        t0 = time.perf_counter()
        r = await call_gemini(
            client, api_key=key, model=MODEL, body=build_request(q, SETTINGS, system=SYSTEM), **RETRY
        )
        return r, time.perf_counter() - t0

    t0 = time.perf_counter()
    seq = [await ask(q) for q in QUESTIONS]
    seq_wall = time.perf_counter() - t0
    _print_table("TUẦN TỰ", [r for r, _ in seq], [t for _, t in seq], seq_wall)

    t0 = time.perf_counter()
    con = await run_limited(QUESTIONS, 3, ask)
    con_wall = time.perf_counter() - t0
    _print_table("ĐỒNG THỜI, tối đa 3 cùng lúc", [r for r, _ in con], [t for _, t in con], con_wall)

    print(f"\nNhanh hơn {seq_wall / con_wall:.1f} lần. Ghi bảng này vào notes/day02.md.")


# ---------------------------------------------------------------- history
async def mode_history(client: httpx.AsyncClient, key: str) -> None:
    first_q = "Tôi làm việc ở công ty được 3 năm 7 tháng, lương 15 triệu một tháng."
    follow_q = "Vậy nếu tôi nghỉ việc đúng luật thì trợ cấp thôi việc khoảng bao nhiêu?"

    first = await call_gemini(
        client, api_key=key, model=MODEL, body=build_request(first_q, SETTINGS, system=SYSTEM), **RETRY
    )
    print(f"\nLượt 1  > {first_q}\n        < {_short(first.text, 200)}")

    no_mem = await call_gemini(
        client, api_key=key, model=MODEL, body=build_request(follow_q, SETTINGS, system=SYSTEM), **RETRY
    )
    print(f"\nLượt 2 KHÔNG gửi history  > {follow_q}")
    print(f"        < {_short(no_mem.text, 200)}")
    print(f"        prompt_tokens = {no_mem.prompt_tokens}")

    history = [ChatTurn(role="user", text=first_q), ChatTurn(role="model", text=first.text or "(trống)")]
    with_mem = await call_gemini(
        client,
        api_key=key,
        model=MODEL,
        body=build_request(follow_q, SETTINGS, system=SYSTEM, history=history),
        **RETRY,
    )
    print(f"\nLượt 2 CÓ gửi history     > {follow_q}")
    print(f"        < {_short(with_mem.text, 200)}")
    print(f"        prompt_tokens = {with_mem.prompt_tokens}")

    print(
        "\nSo sánh 2 câu trả lời lượt 2 và 2 con số prompt_tokens.\n"
        "Trợ lý chỉ 'nhớ' những gì bạn gửi lại, và bạn trả tiền cho mọi token gửi lại đó."
    )


# ---------------------------------------------------------------- main
MODES = {"raw": mode_raw, "speed": mode_speed, "history": mode_history}


async def main(mode: str) -> None:
    key = _api_key()
    print(f"Model: {MODEL}   thinking: {THINKING or 'không gửi'}")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            await MODES[mode](client, key)
    except LLMHTTPError as e:
        hints = {
            400: "Request sai. Đọc 'message' ở trên. Nếu nói về thinking: thử GEMINI_THINKING=none.",
            403: "Key sai hoặc chưa bật Gemini API. Tạo lại key trong AI Studio.",
            404: "Không có model này. Kiểm tra GEMINI_MODEL.",
            429: "Hết quota theo phút/ngày. Đợi 1 phút rồi chạy lại.",
            503: "Model đang quá tải. Đợi vài phút, hoặc thử model khác, ví dụ:\n"
            "  GEMINI_MODEL=gemini-3.5-flash-lite uv run --env-file .env python -m learning.day02.live "
            + mode,
        }
        print(f"\nDừng vì lỗi sau khi đã thử lại: {e}")
        print(hints.get(e.status_code, "Xem bảng 'Gặp lỗi thì xem ở đây' trong tài liệu Ngày 02."))
        sys.exit(1)
    except httpx.TransportError as e:
        print(f"\nLỗi mạng sau khi đã thử lại: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        sys.exit("Dùng: python -m learning.day02.live [raw|speed|history]")
    asyncio.run(main(sys.argv[1]))
