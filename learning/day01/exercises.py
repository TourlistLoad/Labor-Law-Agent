"""
NGÀY 1 — Type hints + Pydantic v2 cho agent stack
==================================================

Cách làm:
    1. Đọc tab "Ngày 01" trong tài liệu lộ trình trước.
    2. Thay mọi `raise NotImplementedError` / `...` bằng code của bạn.
    3. Chạy:  uv run pytest learning/day01 -v
       Xanh hết = qua bài. Đỏ = đọc kỹ thông báo lỗi, đó là bài học.
    4. Không dùng LLM/AI để viết hộ phần TODO. Được tra docs Pydantic.

Bối cảnh: đây là các "khuôn dữ liệu" mà agent Luật Lao động của bạn
sẽ dùng thật ở các tuần sau (tool tính trợ cấp, parse output LLM...).
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)


# ---------------------------------------------------------------------------
# BÀI 1 — ChatMessage: khuôn của 1 tin nhắn trong hội thoại với LLM
# ---------------------------------------------------------------------------
# Yêu cầu:
#   - role: chỉ nhận đúng 4 giá trị "system", "user", "assistant", "tool"
#     (gợi ý: Literal)
#   - content: str, BẮT BUỘC, không được rỗng (min_length=1)
#   - tool_call_id: str hoặc None, mặc định None
#   - Ràng buộc chéo: nếu role == "tool" thì tool_call_id BẮT BUỘC có giá trị;
#     nếu role khác "tool" thì tool_call_id PHẢI là None.
#     (gợi ý: @model_validator(mode="after"), raise ValueError("..."))
#
# Câu hỏi suy nghĩ (ghi vào notes/day01.md): vì sao API LLM cần tool_call_id?

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = Field(description="Cho biết message của role nào")
    content: str = Field(min_length=1, description="Nội dung của message")
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def cross_validation(self):
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError('role "tool" phải có tool_call_id')
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError('chỉ có role "tool" mới được có tool_call_id')
        return self

# ---------------------------------------------------------------------------
# BÀI 2 — SeveranceInput: input của tool "tính trợ cấp thôi việc"
# ---------------------------------------------------------------------------
# Đây là schema mà LLM sẽ NHÌN THẤY khi quyết định gọi tool. Viết
# `description` cho mỗi field như đang viết prompt cho LLM: rõ đơn vị,
# rõ ý nghĩa.
#
# Yêu cầu:
#   - model_config = ConfigDict(extra="forbid")   # LLM bịa thêm field -> lỗi
#   - months_worked: int, 0 <= x <= 600, có description
#       (tổng số tháng thực tế làm việc cho người sử dụng lao động)
#   - months_unemployment_insurance: int, >= 0, mặc định 0, có description
#       (số tháng đã đóng bảo hiểm thất nghiệp trong thời gian đó)
#   - avg_salary_vnd: int, > 0, có description
#       (tiền lương bình quân 6 tháng liền kề trước khi nghỉ, đơn vị VND)
#   - Ràng buộc chéo: months_unemployment_insurance <= months_worked
#
# Gợi ý: Field(ge=..., le=..., gt=..., default=..., description="...")

class SeveranceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    months_worked: int = Field(ge=0, le=600, description="Tổng số tháng thực tế làm việc cho người sử dụng lao động. Đổi năm sang tháng, ví dụ 3 năm rưỡi = 42")
    months_unemployment_insurance: int = Field(default=0, ge=0, description="Số tháng đã đóng bảo hiểm thất nghiệp trong thời gian làm việc cho người sử dụng lao động này")
    avg_salary_vnd: int = Field(gt=0, description="Mức lương bình quân mỗi tháng của 6 tháng liền kề trước khi nghỉ, tính bằng đồng. Ví dụ 15 triệu/tháng thì điền 15000000")

    @model_validator(mode="after")
    def cross_validation(self):
        if self.months_unemployment_insurance > self.months_worked:
            raise ValueError("months_unemployment_insurance phải <= months_worked")
        return self

# ---------------------------------------------------------------------------
# BÀI 3 — compute_severance: logic thuần, KHÔNG gọi LLM
# ---------------------------------------------------------------------------
# Nguyên tắc agent quan trọng: phép tính tiền phải do CODE làm, không để LLM
# tự nhẩm. LLM chỉ trích xuất input; code tính; LLM diễn giải kết quả.
#
# Quy tắc ĐƠN GIẢN HOÁ (sẽ đối chiếu với Điều 46 BLLĐ 2019 và Nghị định
# hướng dẫn ở tuần 2 — đừng coi đây là tư vấn pháp lý):
#   a) Nếu months_worked < 12  -> không đủ điều kiện:
#        eligible=False, years_counted=0.0, amount_vnd=0
#   b) Ngược lại:
#        tháng tính trợ cấp = months_worked - months_unemployment_insurance
#        đổi ra năm: số năm tròn + phần tháng lẻ:
#            lẻ 0 tháng      -> +0
#            lẻ 1..6 tháng   -> +0.5 năm
#            lẻ 7..11 tháng  -> +1 năm
#        amount_vnd = years_counted * 0.5 * avg_salary_vnd, làm tròn xuống
#        về số nguyên (int(...) hoặc math.floor)
#   c) note: 1 câu tiếng Việt giải thích ngắn gọn (tự do).
#
# Ví dụ: worked=50, bhtn=12 -> 38 tháng = 3 năm lẻ 2 tháng -> 3.5 năm
#        lương 20_000_000 -> 3.5 * 0.5 * 20tr = 35_000_000

class SeveranceResult(BaseModel):
    eligible: bool
    years_counted: float = Field(ge=0)
    amount_vnd: int = Field(ge=0)
    note: str


def compute_severance(inp: SeveranceInput) -> SeveranceResult:
    if inp.months_worked < 12:
        return SeveranceResult(eligible=False, years_counted=0.0, amount_vnd=0, note="Không đủ điều kiện nhận trợ cấp")

    months_counted = inp.months_worked - inp.months_unemployment_insurance
    years_counted = months_counted // 12
    months_remainder = months_counted - 12*years_counted

    if 0 < months_remainder and months_remainder <= 6:
        years_counted += 0.5
    elif months_remainder > 6:
        years_counted += 1.0

    amount_vnd = int(years_counted * 0.5 * inp.avg_salary_vnd)
    return SeveranceResult(eligible=True, years_counted=years_counted, amount_vnd=amount_vnd, note="Đủ điều kiện nhận trợ cấp")

# ---------------------------------------------------------------------------
# BÀI 4 — parse_llm_output: pattern "validate -> báo lỗi -> LLM tự sửa"
# ---------------------------------------------------------------------------
# LLM hay trả JSON bọc trong markdown fence, ví dụ:
#     ```json
#     {"months_worked": 30, "avg_salary_vnd": 15000000}
#     ```
# Viết hàm:
#   - Bỏ fence nếu có (```json ... ``` hoặc ``` ... ```), strip khoảng trắng.
#   - Dùng SeveranceInput.model_validate_json(...) để parse + validate.
#   - Thành công -> trả về object SeveranceInput.
#   - Thất bại (ValidationError) -> trả về list[str], mỗi phần tử dạng
#         "<đường dẫn field>: <thông báo lỗi>"
#     ví dụ "months_worked: Input should be less than or equal to 600"
#     Lỗi không gắn với field nào (loc rỗng, vd JSON hỏng hoặc lỗi
#     model_validator) -> dùng "<root>" làm đường dẫn.
#     Gợi ý: e.errors() trả list dict có key "loc" (tuple) và "msg".
#
# Vì sao trả list lỗi chứ không raise? Vì tuần 2 bạn sẽ gửi list này
# NGƯỢC LẠI cho LLM: "Output của bạn sai ở các chỗ sau, hãy sửa".

def parse_llm_output(raw: str) -> SeveranceInput | list[str]:
    cleanRaw = raw.strip()

    if(cleanRaw.startswith("```")):
        lines = cleanRaw.splitlines()
        cleanRaw = "\n".join(lines[1:-1])

    try:
        return SeveranceInput.model_validate_json(cleanRaw)
    except ValidationError as e:
        errorList = []
        for err in e.errors():
            loc = err["loc"]
            if not loc:
                path = "<root>"
            else:
                path = ".".join(map(str,loc))
            errorList.append(f"{path}: {err["msg"]}")

        return errorList

# ---------------------------------------------------------------------------
# BÀI 5 — tool_schema: thứ LLM thực sự nhìn thấy
# ---------------------------------------------------------------------------
# Trả về dict theo đúng dạng "function declaration" mà hầu hết API LLM dùng:
#   {
#     "name": "calc_severance",
#     "description": "<1-2 câu: tool làm gì, KHI NÀO nên gọi>",
#     "parameters": <JSON Schema của SeveranceInput>,
#   }
# Gợi ý: SeveranceInput.model_json_schema()
#
# Sau khi xong: chạy  uv run python -m learning.day01.exercises
# để in schema ra màn hình và ĐỌC nó như thể bạn là LLM.

def tool_schema() -> dict:
    return {
        "name": "calc_severance",
        "description": (
            "Tính tiền trợ cấp thôi việc (VND) theo quy tắc đơn giản hoá của hệ thống. "
            "Gọi khi người dùng muốn biết số tiền trợ cấp thôi việc cụ thể và đã cho biết "
            "số tháng làm việc cùng mức lương bình quân 6 tháng gần nhất. "
            "Nếu thiếu một trong hai thông tin đó thì hỏi lại người dùng, không tự giả định. "
            "Không dùng cho trợ cấp mất việc làm hoặc trợ cấp thất nghiệp."
        ),
        "parameters": SeveranceInput.model_json_schema()
    }


if __name__ == "__main__":
    import json

    print(json.dumps(tool_schema(), ensure_ascii=False, indent=2))
