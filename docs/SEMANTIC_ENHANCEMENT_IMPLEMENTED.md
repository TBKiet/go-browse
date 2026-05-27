# Semantic Enhancement: Những Thay Đổi Đã Thực Hiện

Tài liệu này ghi lại các thay đổi đã triển khai từ
`docs/SEMANTIC_ENHANCEMENT_PROPOSAL.md`.

## 1. Structured Summary Cho Trajectory

File:

- `webexp/agents/task_summarization_agent.py`

Đã thêm `TaskSummary` với các field:

- `accomplished_goal`
- `completion_evidence`
- `summary_differs_from_original`

`TaskSummarizationAgent` hiện có thêm method `summarize_details()` để trả về
summary có cấu trúc. Method cũ `summarize()` vẫn được giữ và tiếp tục trả về
string để không phá code/test cũ.

Prompt summarizer đã được chỉnh để:

- suy ra task từ action history và screenshot trước
- chỉ dùng original goal như hint phụ
- tránh copy original goal nếu action history không chứng minh
- không mô tả các thao tác procedural như click, scroll, wait
- output JSON trong code fence

Metadata mới được lưu trong `traj.misc["semantic_summary"]`; field cũ
`traj.misc["summarized_goal"]` vẫn được ghi bằng `accomplished_goal`.

## 2. Semantic Verifier

File mới:

- `webexp/agents/semantic_verifier_agent.py`

Đã thêm `SemanticVerifierAgent` để kiểm tra chất lượng semantic của trajectory.
Verifier kiểm tra ba điều:

- `is_aligned`: trajectory và accomplished goal còn khớp với original goal
- `is_grounded`: action_nl khớp raw action và element target
- `is_complete`: final state/action history chứng minh task đã hoàn thành

Kết quả được lưu trong:

```json
traj.misc["semantic_verification"]
```

Nếu verifier lỗi API hoặc parse lỗi, quá trình summarize không crash; trajectory
chỉ thiếu metadata verification.

## 3. Tích Hợp Summarizer Và Verifier Vào Node Summarization

File:

- `webexp/explore/algorithms/web_explore.py`

`_summarize_node_trajectories()` hiện:

1. gọi `summarize_details()` nếu trajectory chưa có `semantic_summary`
2. lưu `semantic_summary` và `summarized_goal`
3. gọi `SemanticVerifierAgent.verify()` nếu chưa có `semantic_verification`
4. lưu lại `traj_info.json`

Logic chọn task-level summary cũng đã đổi:

- ưu tiên positive trajectory
- ưu tiên summary đã pass semantic verification
- ưu tiên summary có evidence
- ưu tiên summary khác original goal nếu có bằng chứng
- không còn chọn chỉ vì summary xuất hiện nhiều nhất

Task-level `summarized_goal` vẫn được ghi bằng `Task.update_summarized_goal()`.
Ngoài ra task misc có thêm:

```json
task.misc["semantic_summary_selected"]
```

## 4. Structured Task State Trong Solver

File:

- `webexp/agents/solver_agent.py`
- `webexp/agents/prompt_builders/solver_prompt_builder.py`

Solver prompt hiện yêu cầu LLM trả thêm `task_state`:

```json
{
  "original_goal": "...",
  "current_objective": "...",
  "confirmed_entities": [],
  "constraints": [],
  "negative_observations": [],
  "completion_condition": null,
  "infeasibility_reason": null
}
```

Parser trong `solver_agent.py` đọc `task_state` nếu model trả về. Nếu model
không trả field này, code tự tạo fallback task state từ original goal và
`refined_goal`.

Task state được lưu trong:

```json
step.misc["task_state"]
```

Field cũ `refined_goal` vẫn giữ nguyên để backward compatible.

## 5. Action Grounding Validation

File:

- `webexp/agents/solver_agent.py`

Đã thêm `_validate_action_grounding()` để kiểm tra heuristic:

- action có `action_nl` không
- với `type(...)`, text được type có xuất hiện trong `action_nl` không
- với action có bid, `action_nl` có overlap với text của target element không
- action không gắn element như `send_msg_to_user`, `noop`, `go_back` được coi là
  grounded ở mức action type

Kết quả lưu trong step misc:

```json
{
  "grounding_valid": true,
  "grounding_reason": "element_text_overlap"
}
```

Ngoài ra `_extract_element_metadata()` đã được mở rộng để parse format a11y phổ
biến của BrowserGym:

```text
[42] button 'Add to Cart'
```

## 6. Bật SoM Trong Config

File:

- `configs/go_browse_config.yaml`

Đã thêm:

```yaml
use_som: true
```

cho:

- `page_explorers`
- `nav_explorers`
- `feasibility_checkers`
- `solvers`

Điều này bật Set-of-Mark processing cho các agent chính trong các run tiếp theo.

## Những Phần Chưa Triển Khai

Các phần sau vẫn còn là việc tiếp theo:

- Persist riêng raw screenshot và SoM screenshot cùng lúc.
- Ghi `bounding_box` thành field chính thức cho từng step.
- Semantic training export script.
- Loại trajectory fail semantic verification khỏi training export thực tế.
- Schema JSON mới cho PageExplorer/NavExplorer task proposal.
- Báo cáo tổng hợp các trajectory bị semantic verifier reject.

## Lưu Ý Tương Thích

Các thay đổi hiện tại cố ý giữ backward compatibility:

- `summarize()` vẫn trả string.
- `summarized_goal` vẫn tồn tại.
- `refined_goal` vẫn tồn tại.
- Các field mới được lưu trong `misc` thay vì đổi schema bắt buộc.

Vì vậy dữ liệu cũ vẫn load được, còn dữ liệu mới có thêm metadata semantic để
verify và export training tốt hơn.
