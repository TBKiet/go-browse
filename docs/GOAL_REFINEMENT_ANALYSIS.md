# Phân Tích Vấn Đề Goal Refinement Trong Go-Browse

> Tài liệu này phân tích 11 vấn đề về goal refinement dựa trên code hiện tại,
> kèm giải pháp đề xuất cho từng vấn đề.

---

## Mục Lục

1. [Goal Drift (Trôi mục tiêu)](#1-goal-drift-trôi-mục-tiêu)
2. [Refined Goal Mô Tả Action Thay Vì Objective](#2-refined-goal-mô-tả-action-thay-vì-objective)
3. [Refined Goal Overwrite Toàn Bộ Context](#3-refined-goal-overwrite-toàn-bộ-context)
4. [Local-Optimum Bias](#4-local-optimum-bias)
5. [Không Encode Progress State](#5-không-encode-progress-state)
6. [Không Encode Negative Observations](#6-không-encode-negative-observations)
7. [Refined Goal Quá Procedural](#7-refined-goal-quá-procedural)
8. [Không Có Termination-Aware Refinement](#8-không-có-termination-aware-refinement)
9. [Không Phân Biệt Environment State vs Intended Outcome](#9-không-phân-biệt-environment-state-vs-intended-outcome)
10. [Không Chống Repetition](#10-không-chống-repetition)
11. [Summarized Goal Không Được Ghi Lại Vào Task](#11-summarized-goal-không-được-ghi-lại-vào-task)
12. [Tổng Kết: Kiến Trúc Đề Xuất](#12-tổng-kết-kiến-trúc-đề-xuất)

---

## 1. Goal Drift (Trôi mục tiêu)

### Hiện trạng trong code

Trong `solver_agent.py` (dòng ~270):

```python
self._refined_goal: str | None = None  # evolving task description
```

Và trong `get_action()` (dòng ~360-365):

```python
if refined_goal:
    self._refined_goal = refined_goal
    logger.info(f"Refined goal updated: {refined_goal[:200]}")
```

`self._refined_goal` bị **overwrite hoàn toàn** mỗi step. Không có cơ chế nào giữ original goal làm anchor.

Trong `solver_prompt_builder.py` (dòng ~149-151):

```python
if refined_goal:
    user_content.append(self.refined_goal_message(refined_goal))
```

`refined_goal_message` chỉ hiển thị `# Current Task Understanding (evolving)` — không có original goal bên cạnh để so sánh.

### Vấn đề

- Original goal chỉ xuất hiện ở `goal_message` (dòng ~139), nhưng `refined_goal` lại xuất hiện sau đó (dòng ~149-151) và không có chỉ dẫn rằng nó phải **phục vụ** original goal.
- LLM dễ dàng "quên" original goal khi refined goal bị thu hẹp thành subtask.

### Giải pháp đề xuất

1. **Dual-goal context**: Luôn hiển thị cả original goal và refined goal, với chỉ dẫn rõ:
   ```
   # Original Goal (immutable)
   Add product X to shopping cart

   # Current Progress (evolving from original goal)
   - Phase: Adding to cart
   - Found: Product X on PDP
   - Next: Click "Add to Cart"
   ```

2. **Alignment check instruction**: Thêm instruction cho LLM:
   ```
   Before outputting refined_goal, verify it is still aligned with the original goal.
   If refined_goal deviates from the original goal, correct it.
   ```

---

## 2. Refined Goal Mô Tả Action Thay Vì Objective

### Hiện trạng trong code

Trong `solver_prompt_builder.py` (dòng ~417-419), các CoT examples:

```python
{"thought": "...", "action": "click('12')", "refined_goal": "Submit the contact form with the provided information"}
```

Ví dụ này tương đối ổn, nhưng instruction ở system message (dòng ~372-378) lại không đủ chặt:

```
"refined_goal": "<updated task description incorporating new details learned from the current page>"
```

Không có guidance rõ ràng rằng refined goal phải là **desired outcome/state**, không phải **action mô tả**.

### Vấn đề

- LLM được instruction "update the task description", nhưng không được hướng dẫn phân biệt:
  - **Goal**: trạng thái mong muốn (cart contains product)
  - **Action**: hành động để đạt trạng thái đó (click button)
- Action-based goal → mất strategic flexibility.

### Giải pháp đề xuất

1. **Cấm action trong refined goal**: Thêm instruction:
   ```
   "refined_goal" must describe the DESIRED STATE, not the action.
   GOOD: "Cart contains product X"
   BAD:  "Click Add to Cart button"
   BAD:  "Scroll to reveal button"
   ```

2. **Dùng action_nl cho action description**: `action_in_natural_language` đã có sẵn — không cần refined goal làm việc đó.

---

## 3. Refined Goal Overwrite Toàn Bộ Context

### Hiện trạng trong code

Trong `solver_agent.py`:

```python
self._refined_goal = refined_goal  # Overwrite hoàn toàn
```

Trong `episode.py` (dòng ~239-242):

```python
refined_goal = action_extras.get("refined_goal")
if refined_goal:
    callback_context_seed["refined_goal"] = refined_goal
```

**Chỉ có 1 string** được propagate. Không có:
- Original goal
- Progress history
- Known facts

### Vấn đề

- Khi refined goal = "Scroll down to load products", original intent "Add V8 to cart" biến mất hoàn toàn khỏi prompt state.
- `past_refined_goals` (8 items gần nhất) được lưu nhưng chỉ dùng làm history, không phải context structure.

### Giải pháp đề xuất

1. **Restructure thành `TaskState` object** thay vì string:

```python
@dataclass
class TaskState:
    original_goal: str
    current_objective: str  # refined từ original, nhưng vẫn aligned
    completed_phases: list[str]
    known_facts: list[str]
    failed_attempts: list[str]
    current_phase: str  # e.g., "search", "locate_product", "add_to_cart", "verify"
```

2. **Immutable original goal**: Không bao giờ cho phép refined goal overwrite original goal trong context.

---

## 4. Local-Optimum Bias

### Hiện trạng trong code

Không có cơ chế nào trong code hiện tại để ngăn agent khóa vào local obstacle. Khi refined goal thành "Make button visible", agent chỉ tập trung giải subproblem đó.

### Vấn đề

- Agent thấy button chưa visible → scroll → refined goal = "Make Add to Cart visible"
- Agent không còn cân nhắc: search lại, mở PDP, kiểm tra stock, dùng alternative interaction
- Refined goal hiện tại "khóa" agent vào local obstacle

### Giải pháp đề xuất

1. **Periodic re-alignment step**: Cứ N steps, thêm message:
   ```
   # Re-alignment Check
   Original goal: {original_goal}
   Current refined goal: {refined_goal}
   
   Are you still making progress toward the original goal?
   If not, consider a fundamentally different approach instead of continuing current trajectory.
   ```

2. **Backtrack trigger**: Nếu refined goal không thay đổi trong K steps và không có progress (URL, cart state), force backtrack.

---

## 5. Không Encode Progress State

### Hiện trạng trong code

Trong `solver_prompt_builder.py`, `past_refined_goals` chỉ là list các string:

```python
past_refined_goals = [step.misc.get('refined_goal') if step.misc else None for step in history]
```

Không có structured progress tracking.

### Vấn đề

Sau 10 steps, agent không biết:
- Đã search chưa?
- Đã tìm thấy product chưa?
- Đã mở PDP chưa?
- Stock có available không?

→ Dễ repeat actions vì không biết phase hiện tại.

### Giải pháp đề xuất

1. **Structured progress tracker**:

```python
@dataclass
class ProgressState:
    phases: dict[str, bool] = field(default_factory=lambda: {
        "searched": False,
        "product_located": False,
        "pdp_opened": False,
        "stock_verified": False,
        "added_to_cart": False,
        "cart_verified": False,
    })
    current_phase: str = "search"
```

2. **Hiển thị progress bar trong prompt**:
```
# Progress
[x] Searched for product
[x] Located product in results
[x] Opened PDP
[ ] Verified stock availability
[ ] Added to cart
[ ] Verified cart
```

---

## 6. Không Encode Negative Observations

### Hiện trạng trong code

Trong `solver_prompt_builder.py`, `last_action_error_message` chỉ hiển thị error hiện tại:

```python
def last_action_error_message(self, last_action_error: str):
    return {
        "type": "text",
        "text": (
            "# Error message from last action\n"
            f"{last_action_error}\n\n"
            "..."
        )
    }
```

Không có persistent fact storage cho negative observations.

### Vấn đề

- Agent thấy "unavailable" ở step 5
- Step 6: refined goal vẫn là "Add product to cart"
- Step 7-10: retry add-to-cart → fail → loop
- Không có "Known facts: product is out of stock" để ngăn retry

### Giải pháp đề xuất

1. **Persistent facts section**:
```
# Known Facts
- Product "V8 Energy Drink 12oz" is currently OUT OF STOCK (observed at step 5)
- Add-to-cart attempt failed with error: "Requested quantity unavailable"
- No alternative sellers found on this page
```

2. **Auto-extract negative facts**: Khi có error, tự động extract fact và append vào `known_facts` list.

---

## 7. Refined Goal Quá Procedural

### Hiện trạng trong code

Instruction hiện tại (dòng ~376-378):

```
The "refined_goal" key should update the overall task description with new constraints/details you've discovered
```

Không có ràng buộc về mức độ abstract.

### Vấn đề

- LLM sinh refined goal kiểu: "Scroll down to see if button appears"
- Đây là chain-of-thought bị expose thành goal
- Làm prompt noisier, tăng action imitation, tăng looping tendency

### Giải pháp đề xuất

1. **Abstract level enforcement**:
   ```
   "refined_goal" must be at the TASK level, not the ACTION level.
   
   TASK-level: "Add product X to cart"
   ACTION-level: "Click button" ✗
   ACTION-level: "Scroll down" ✗
   ACTION-level: "Type in search box" ✗
   ```

2. **Validation rule**: Nếu refined goal bắt đầu bằng action verb (click, scroll, type, press, etc.), reject và yêu cầu rewrite ở task level.

---

## 8. Không Có Termination-Aware Refinement

### Hiện trạng trong code

Trong `solver_agent.py`, không có logic nào chuyển refined goal sang terminal reasoning khi gặp infeasible condition.

### Vấn đề

- Agent phát hiện "out of stock"
- refined goal vẫn: "Add item to cart"
- → Agent oscillate giữa retry, infeasible, retry again
- Không có goal kiểu: "Determine whether task is infeasible due to stock unavailability"

### Giải pháp đề xuất

1. **Termination conditions**: Thêm rules:
   ```
   If you have confirmed a product is out of stock across all available options:
   - Set refined_goal to: "Confirm unavailability and report infeasible"
   - Use report_infeasible() action
   ```

2. **Infeasible confirmation loop**: Trước khi retry, kiểm tra:
   - Đã confirm infeasible condition chưa?
   - Đã thử alternative approaches chưa?
   - Nếu đã confirm → force termination

---

## 9. Không Phân Biệt Environment State vs Intended Outcome

### Hiện trạng trong code

Refined goal hiện tại không phân biệt:
- **Intended interaction**: "Click Add to Cart button"
- **Desired state**: "Cart contains target product"

### Vấn đề

- Agent optimize action completion thay vì state completion
- "Tôi đã click → done" thay vì "Tôi cần verify cart có product chưa"
- Thiếu verification step sau action

### Giải pháp đề xuất

1. **State-based goal format**:
   ```
   "refined_goal": "<desired state after completing current phase>"
   
   GOOD: "Cart contains product X with quantity 1"
   BAD:  "Click Add to Cart"
   
   GOOD: "Search results show product X"
   BAD:  "Type 'V8' in search box"
   ```

2. **State verification instruction**: Sau mỗi action, yêu cầu verify state transition:
   ```
   After executing an action, verify whether the desired state was achieved.
   If not, diagnose why and try an alternative approach.
   ```

---

## 10. Không Chống Repetition

### Hiện trạng trong code

Trong `solver_agent.py`, có `_recent_action_errors` để detect retry loop (dòng ~380-395):

```python
self._recent_action_errors: list[tuple[str, str]] = []
```

Và stuck detection (dòng ~398-411):

```python
if len(unique_actions) <= 2 and len(recent_actions) >= 4:
    logger.warning("Stuck detection...")
```

Nhưng các cơ chế này **không được feed vào refined goal**. Chúng chỉ log warning hoặc raise error.

### Vấn đề

- Agent không biết action nào đã ineffective
- Agent không biết action nào đã fail nhiều lần
- Retry loops vẫn xảy ra vì refined goal không có memory về failures

### Giải pháp đề xuất

1. **Failed attempts trong prompt**:
```
# Failed Attempts (past 5)
1. click('42') → "Element not found" (step 3)
2. scroll(0, 200) → "No new content loaded" (step 4)
3. click('42') → "Element not found" (step 5)
```

2. **Auto-suggest alternatives**: Khi detect retry loop, inject vào prompt:
   ```
   You have tried click('42') 3 times and it has failed each time.
   The element with bid 42 may not exist on this page.
   Consider: searching for a different product page, using navigation, or reporting infeasible.
   ```

---

## 11. Summarized Goal Không Được Ghi Lại Vào Task

### Hiện trạng trong code

Trong `task.py` — `__post_init__()`:

```python
def __post_init__(self):
    if not os.path.exists(self.exp_dir):
        os.makedirs(self.exp_dir)
        ...
        task_info = {"goal": self.goal, "misc": self.misc}
        with open(os.path.join(self.exp_dir, "task_info.json"), "w") as f:
            json.dump(task_info, f, indent=4)
```

`task_info.json` chỉ được ghi **đúng 1 lần** khi task được tạo. Không có method nào update file này sau đó.

Trong `web_explore.py` — `_summarize_node_trajectories()`:

```python
summarized = summarizer.summarize(traj)
if summarized:
    traj.misc["summarized_goal"] = summarized  # Lưu trong trajectory
    traj.save_info()                           # Ghi vào traj_info.json
```

Summarized goal chỉ được lưu trong **trajectory** (`traj_info.json`), không được đẩy ngược lên **task** (`task_info.json`).

### Dữ liệu trên đĩa

```
node_1/tasks/task_7/
├── task_info.json                  ← {"goal": "original goal gốc"}  ❌ KHÔNG đổi
├── positive_trajs/0/
│   └── traj_info.json
│       └── misc.summarized_goal    ← "refined goal"  ✅ Có nhưng ở trajectory
└── negative_trajs/0/
    └── traj_info.json
        └── misc.summarized_goal    ← "refined goal"  ✅ Có nhưng ở trajectory
```

### Vấn đề

| Stage | File được ghi | Nội dung goal |
|-------|--------------|---------------|
| Task creation | `task_info.json` | `goal`: original proposed goal |
| Feasibility check | `traj_info.json` | `goal`: original goal (không đổi) |
| Solving | `traj_info.json` | `goal`: original goal (không đổi) |
| Summarization | `traj_info.json` | `misc.summarized_goal`: refined (chỉ ở trajectory) |
| **Task info** | `task_info.json` | **KHÔNG BAO GIỜ được cập nhật** |

### Hậu quả

1. **Training data không dùng được summarized goal**: Khi load task để train, `task.goal` vẫn là goal gốc — summarized goal bị bỏ quên.
2. **Không có single source of truth**: Phải đọc từng trajectory để biết summarized goal, không có tổng quan ở cấp task.
3. **Task evaluation sai lệch**: Task được đánh giá dựa trên goal gốc, không phải goal thực tế agent đã thực hiện.
4. **Traceability kém**: Khi nhìn vào `task_info.json` không biết task đã được refine thành gì.

### Giải pháp đề xuất

1. **Cập nhật `task_info.json` sau summarization**: Thêm method `Task.update_summarized_goal()`:

```python
def update_summarized_goal(self, summarized_goal: str):
    """Ghi summarized goal vào task_info.json mà không làm mất goal gốc."""
    task_info_path = os.path.join(self.exp_dir, "task_info.json")
    with open(task_info_path, "r") as f:
        task_info = json.load(f)
    task_info["summarized_goal"] = summarized_goal
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=4)
```

2. **Chọn summarized goal tốt nhất từ các trajectories**: Nếu có nhiều trajectory, chọn cái tốt nhất (hoặc phổ biến nhất):

```python
# Trong _summarize_node_trajectories()
best_summary = _select_best_summary(task.positive_trajs)
if best_summary:
    task.update_summarized_goal(best_summary)
```

3. **Dual-goal trong task_info.json**:
```json
{
    "goal": "List all 5-star reviews for the V8 +Energy",
    "summarized_goal": "Find and display 5-star customer reviews for V8 Energy Drink on the product page",
    "misc": { ... }
}
```

4. **Training pipeline đọc summarized goal**: Khi load task để train, ưu tiên dùng `summarized_goal` nếu có.

---

## 12. Tổng Kết: Kiến Trúc Đề Xuất

### Hiện tại

```
[Step N]                    [Step N+1]
goal: str                   goal: str (bị overwrite)
refined_goal: str           refined_goal: str (bị overwrite)
                            past_refined_goals: list[str] (8 items, chỉ để tham khảo)
```

### Đề xuất

```
[Step N]                    [Step N+1]
original_goal: str (immutable)
task_state: TaskState {
    current_objective: str (aligned với original)
    completed_phases: list[str]
    current_phase: str
    known_facts: list[str]
    failed_attempts: list[FailedAttempt]
    negative_observations: list[str]
}
```

### Thay đổi cụ thể cần implement

| File | Thay đổi |
|------|----------|
| `solver_agent.py` | Thay `_refined_goal: str` bằng `_task_state: TaskState` |
| `solver_prompt_builder.py` | Restructure prompt: thêm progress section, known facts, failed attempts |
| `solver_prompt_builder.py` | Thêm instruction về state-based goal, cấm action-based goal |
| `solver_prompt_builder.py` | Thêm re-alignment check periodic message |
| `episode.py` | Propagate `TaskState` thay vì `refined_goal: str` |
| `trajectory.py` | Thêm `TaskState` vào `TrajectoryStep` |
| `solver_agent.py` | Thêm validation rules cho refined goal |
| `solver_prompt_builder.py` | Thêm termination-aware refinement logic |

### Priority

1. **Critical** (gây ra lỗi nghiêm trọng nhất): Vấn đề 1 (Goal Drift), 3 (Overwrite), 6 (Negative Observations)
2. **High** (gây kém hiệu quả): Vấn đề 2 (Action vs Objective), 5 (Progress State), 10 (Repetition)
3. **Medium** (cải thiện độ ổn định): Vấn đề 4 (Local-Optimum Bias), 7 (Procedural Goal), 8 (Termination-Aware), 9 (State vs Outcome)
