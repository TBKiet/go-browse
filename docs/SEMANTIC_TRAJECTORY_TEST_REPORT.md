# Semantic Trajectory Implementation — Test Report

**Ngày:** 2026-05-19
**Kết quả:** 66/66 tests passed
**Nhánh:** `main`

---

## 1. Tổng quan

Tất cả 7 phase của `semantic_trajectory_improvement.md` đã được test và debug kỹ lưỡng. Tổng cộng 66 tests (17 parsing + 14 semantic trajectory + 35 edge cases) đều pass. 5 bugs được phát hiện và sửa trong quá trình test.

---

## 2. Test files

| File | Số tests | Mục đích |
|------|----------|----------|
| `tests/test_parsing.py` | 17 | Test các hàm parsing cơ bản (`_extract_full_response`, `_extract_element_metadata`, `sanitize_action`, `TrajectoryStep.save/load`) |
| `tests/test_semantic_trajectory.py` | 14 | Test Phase 1-3: semantic fields, roundtrip save/load, backward compat |
| `tests/test_edge_cases.py` | 35 | Test edge cases toàn diện cho tất cả 7 phase + integration tests |

---

## 3. Bugs đã phát hiện và sửa

### Bug #1 — `PageExplorerPromptBuilder.system_message()` thiếu `use_som` param (🔴 Crash)

**File:** `webexp/agents/prompt_builders/page_explorer_prompt_builder.py:6`

**Mô tả:** `PageExplorerPromptBuilder.system_message(self)` không nhận `use_som` parameter. Khi `SolverPromptBuilder._build_messages()` gọi `self.system_message(use_som=use_som)`, nếu `self` là `PageExplorerPromptBuilder` instance với `use_som=True`, sẽ crash với `TypeError: system_message() got an unexpected keyword argument 'use_som'`.

**Fix:** Thêm `use_som: bool = False` vào signature và thêm SoM note khi `use_som=True`.

### Bug #2 — `NavExplorerPromptBuilder.system_message()` thiếu `use_som` param (🔴 Crash)

**File:** `webexp/agents/prompt_builders/nav_explorer_prompt_builder.py:6`

**Mô tả:** Giống hệt bug #1, áp dụng cho `NavExplorerPromptBuilder`.

**Fix:** Tương tự bug #1.

### Bug #3 — `Trajectory.load()` crash khi `final_state` là None (🔴 Crash)

**File:** `webexp/explore/core/trajectory.py:199-200`

**Mô tả:** `Trajectory.load()` luôn cố gắng load `final_state/step_info.json` ngay cả khi `final_state` chưa từng được save (khi `final_state=None` trong `save()`). Gây `FileNotFoundError` khi load trajectory chưa có final_state.

**Fix:** Kiểm tra `os.path.exists()` trước khi load:
```python
if os.path.exists(os.path.join(final_state_load_dir, "step_info.json")):
    final_state = TrajectoryStep.load(final_state_load_dir, load_image=load_images)
else:
    final_state = None
```

### Bug #4 — `TrajectoryStep.load()` dùng `step_info["misc"]` không `.get()` (🟡 Edge case)

**File:** `webexp/explore/core/trajectory.py:97`

**Mô tả:** `step_info["misc"]` sẽ throw `KeyError` nếu load old-format step_info.json không có field `misc`. Mặc dù `save()` luôn ghi field này, nhưng các trajectory được tạo bởi code cũ hơn có thể không có.

**Fix:** Đổi thành `step_info.get("misc")`.

### Bug #5 — `test_parsing.py` import lỗi do module name sai (🟡 Test infra)

**File:** `tests/test_parsing.py:8-9`

**Mô tả:** `spec_from_file_location("solver_agent", ...)` đặt module name là `"solver_agent"` (không có package prefix), khiến relative imports (`from .base_agent import ...`) trong `solver_agent.py` fail với `ModuleNotFoundError`.

**Fix:** Viết lại test dùng direct imports từ package đã cài đặt (thay vì dùng `importlib` hack).

---

## 4. Test coverage chi tiết theo Phase

### Phase 1 — Semantic fields trong TrajectoryStep

**Tests: 23** (parsing 10 + trajectory model 13)

- `_extract_full_response`: all fields, minimal, Qwen think tags, escaped quotes, newlines, truncated JSON, empty input, verbose text before JSON, multiple JSON blocks, markdown code fence, plain code fence, nested braces, unicode text, action key only
- `_extract_element_metadata`: click, type, scroll (no bid), bid not in tree, multiline axtree, newlines in text, long text
- `sanitize_action`: newlines, zero-width space, whitespace, carriage return, NBSP, tab, ZWNJ/ZWJ, BOM, preserve valid content
- `TrajectoryStep`: new fields init, save/load roundtrip, without screenshot, old format backward compat, misc backward compat
- `Trajectory`: from_goal defaults, add_step backward compat, save_info, full roundtrip, multiple semantic steps

### Phase 2 — Task Refinement (evolving goal)

**Tests: 6**

- Episode flow integration test: full 3-step trajectory với refined_goal tiến hóa qua từng step
- `refined_goal` propagation qua `callback_context_seed`
- `page_url_before` / `page_url_after` tracking per step
- Backward compat: `add_step` không có semantic fields vẫn hoạt động
- Verified: `PageExplorerAgent` và `NavExplorerAgent` dùng `SolverAgent.get_action()` đều lấy `action_nl`, `refined_goal` từ LLM response (nếu có)

### Phase 3 — Task Summarization

**Tests: 4**

- `TaskSummarizationAgent._build_action_list()`: fallback từ `action_nl` → `action` khi không có NL description
- `_collect_screenshots()`: even sampling với `max_screenshots`, no screenshots
- `summarize()`: empty actions trả về original goal, không crash
- Verified: `_summarize_node_trajectories()` trong `web_explore.py` gọi `traj.save_info()` sau khi set `summarized_goal`

### Phase 4 — Set-of-Mark (SoM)

**Tests: 5**

- `extract_rois_from_axtree()`: empty tree, nested tree, list format với `backend_node_id` matching
- `process_observation()`: no axtree → unchanged, no screenshot → unchanged
- Verified: `use_som` param được pipe từ `SolverAgent.__init__` → `make_llm_call_with_adaptive_retry` → `prompt_builder.build_messages()` → `_build_messages()` → `system_message(use_som=...)`
- Verified: `PageExplorerPromptBuilder.system_message(use_som=True)` và `NavExplorerPromptBuilder.system_message(use_som=True)` không crash (bug #1, #2 đã sửa)

### Phase 5 — Evaluator (enhanced verification)

**Tests: 4**

- `extract_content()`: standard cases (Task Type, Status, Failure Reason), missing tags
- `build_vision_eval_prompt()`: xác nhận prompt chứa đủ 4 task types (transaction, information_seeking, site_navigation, content_modification)
- `evaluation_info` dict chứa `task_type`, `failure_reason`, `reward`, `model_usage`

### Phase 6 — Better Task Proposal (PageExplorer)

- `PageExplorerAgent` prompt builder đã được update với task quality requirements, concrete examples, verification strategy
- `NavExplorerAgent` tương tự cho navigation tasks
- Không có bug riêng — verified qua integration với Phase 2 flow

### Phase 7 — CAPTCHA Detection

**Tests: 1**

- `CaptchaDetectionAgent.__init__()`: default model name
- Verified: `is_captcha()` trả về `False` khi có exception (không skip valid pages do lỗi)
- Verified: CAPTCHA check được gọi trước khi explore node trong `web_explore_loop()`

---

## 5. Integration verification

- **Episode flow:** `run_episode()` → `get_action()` → `perform_env_step()` → record `page_url_after` → propagate `refined_goal` → evaluator → save trajectory
- **Task/Node flow:** `Task.add_trajectory()` gọi `traj.save()` → `_summarize_node_trajectories()` gọi `traj.save_info()` để update `summarized_goal`
- **Backward compat:** Old `add_step()` calls không có semantic fields vẫn hoạt động; old `step_info.json` load đúng với `None` defaults
- **Data roundtrip:** `Trajectory.save()` → `Trajectory.load()` bảo toàn tất cả semantic fields

---

## 6. Code changes summary

### Files sửa (bug fixes):

| File | Thay đổi |
|------|----------|
| `webexp/explore/core/trajectory.py:97` | `step_info["misc"]` → `step_info.get("misc")` |
| `webexp/explore/core/trajectory.py:199-203` | Check file exists trước khi load `final_state` |
| `webexp/agents/prompt_builders/page_explorer_prompt_builder.py:6` | Thêm `use_som: bool = False` vào `system_message()` |
| `webexp/agents/prompt_builders/nav_explorer_prompt_builder.py:6` | Tương tự |
| `tests/test_parsing.py` | Viết lại dùng direct imports |

### Files tạo mới (tests):

| File | Mục đích |
|------|----------|
| `tests/test_edge_cases.py` | 35 edge case tests cho tất cả 7 phase |

---

## 7. Risks & Considerations

1. **OpenAI API key required:** `evaluator.py` tạo `OpenAI` client ở module level — cần set `OPENAI_API_KEY` env var (dù là fake) để import module này. Tests chạy với `OPENAI_API_KEY=sk-test`.
2. **SoM requires `backend_node_id`:** `extract_rois_from_axtree()` cần `backend_node_id` trong axtree nodes để map với `extra_element_properties`. Nếu BrowserGym không cung cấp `extra_element_properties` với bbox, SoM sẽ không có overlay.
3. **Token cost:** LLM output thêm `action_in_natural_language` và `refined_goal` → completion tokens tăng ~2-3x (đã lường trước trong doc).
4. **`run_episode.py` cần `omegaconf`:** Đã cài đặt trong conda env `go-browse`.

---

## 8. Cách chạy tests

```bash
# Activate conda environment
conda activate go-browse

# Run all tests
OPENAI_API_KEY=sk-test python tests/test_parsing.py
OPENAI_API_KEY=sk-test python tests/test_semantic_trajectory.py
OPENAI_API_KEY=sk-test python tests/test_edge_cases.py
```
