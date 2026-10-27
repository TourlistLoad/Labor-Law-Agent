Câu hỏi suy nghĩ (ghi vào notes/day01.md): vì sao API LLM cần tool_call_id?
Vì hệ thống có thể có rất nhiều tool nên nếu message là của "tool" thì cần biết tool_call_id để xác định đó là tool nào.

1. Type hint trong Python có tự kiểm tra kiểu lúc chạy không? Vậy Pydantic dùng chúng để làm gì?
2. Nếu bạn xoá hết description trong SeveranceInput, hệ thống sẽ hỏng ở đâu, dù test bài 2 không còn kiểm tra?
3. Vì sao không để LLM tự tính tiền trợ cấp mà phải có compute_severance()?
4. parse_llm_output() trả list lỗi thay vì raise. Ai sẽ đọc list lỗi đó, và đọc xong thì làm gì?
5. Khi nào dùng field_validator, khi nào dùng model_validator?
6. extra="forbid" bảo vệ bạn khỏi loại lỗi nào của LLM?

Answer:
1. Type hint trong Python không tự kiểm tra lúc chạy. Pydantic dùng nó để tạo schema và validate
2. Nếu xóa hết description LLM sẽ không biết điền thông tin nó lấy được vào đúng field.
3. Công thức tính có thể không phổ biến để LLM biết nên sẽ tính sai
4. LLM sẽ đọc list lỗi, LLM đọc xong sẽ sửa lại
5. field_validator dùng khi validate rõ hơn cho 1 trường nào đó, còn nếu validate có liên quan nhiều hơn 1 thì dùng model_validator
6. extra="forbid" bảo vệ khỏi lỗi khi LLM tự tạo field lạ mà không có trong list field nhưng Pydantic bỏ qua