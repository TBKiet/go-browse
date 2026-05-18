# Semantic Trajectory Improvement Proposal

## Mục tiêu

Cải tiến quy trình thu thập dữ liệu của Go-Browse để sinh ra **semantic trajectory** — trajectory mà mỗi action đều đi kèm với ngữ nghĩa (natural language description), intent đang tiến hóa (refined goal), visual grounding (Set-of-Mark), và post-hoc verification — giống như cách Explorer thực hiện.

---

## 1. So sánh kiến trúc hai dự án

### 1.1 Tổng quan

| Khía cạnh | Go-Browse (hiện tại) | Explorer |
|-----------|---------------------|----------|
| **Framework** | BrowserGym + Playwright | Playwright trực tiếp |
| **LLM API** | OpenAI Python client | REST API trực tiếp (`requests.post`) |
| **Config** | OmegaConf YAML + dataclass | argparse |
| **Agent types** | PageExplorer, NavExplorer, Solver | TaskProposal, TaskRefiner, Summarizer, Verifier |
| **Exploration** | Multi-node graph traversal | Single-site depth exploration |
| **Output** | Graph nodes + tasks + trajectories | Single `task_trajectory_data.json` |
| **Grounding** | BrowserGym bid-based (`click('bid_42')`) | Set-of-Mark (`click [42]`) |
| **Training** | SFT/LoRA với TRL | Qwen2-VL fine-tuning với DeepSpeed |

### 1.2 Quy trình thu thập dữ liệu

**Go-Browse (hiện tại):**
```
Seed URL → Graph
  └─ For each node:
      1. PageExplorerAgent: khám phá page → đề xuất tasks (add_tasks_to_dataset)
      2. NavExplorerAgent: tìm navigation tasks
      3. Feasibility Checker: kiểm tra task khả thi
      4. Solver: sinh trajectory (prefixed + non-prefixed variants)
      5. Evaluator (GPT-4V): đánh giá success/failure
```

**Explorer:**
```
Seed URL
  └─ For step in range(MAX_STEPS):
      1. Get state: SoM screenshot + accessibility tree
      2. step=0: TaskProposalAgent → propose task + first action
      3. step>0: TaskRefinerAgent → refine task + next action
      4. Execute action via Playwright
      5. Record action metadata (NL, grounded, bbox, refined_goal)
  └─ Post-trajectory:
      6. TaskSummarizationAgent: tóm tắt intent từ action history + screenshots
      7. TrajectoryVerifierAgent: phân loại success/failure
  └─ Output: task_trajectory_data.json
```

---

## 2. Điều gì làm cho Explorer trajectory "semantic"?

### 2.1 Dual action representation

Mỗi action trong Explorer có **hai dạng biểu diễn**:
- **Natural language**: `"Click on the search bar and type 'coffee maker'"`
- **Grounded form**: `"type [7] [coffee maker]"`

Go-Browse hiện tại chỉ có raw action (`click('bid_42')`) và parsed action, **không có** natural language description.

### 2.2 Evolving task refinement

Trong Explorer, task description được **cập nhật ở mỗi bước** bởi TaskRefinerAgent:
- Step 0: `"Find a product on Amazon"` (mơ hồ)
- Step 3: `"Find a coffee maker under $100 with 4+ stars on Amazon"` (cụ thể hơn)

Go-Browse: task goal là **static** — được propose một lần và giữ nguyên suốt trajectory.

### 2.3 Visual grounding qua Set-of-Mark (SoM)

Explorer sử dụng SoM để tạo **cầu nối ngữ nghĩa** giữa pixel và text:
- Screenshot được overlay với numbered tags (mỗi interactive element có một số)
- Agent reference element qua `[idx]` trong grounded action
- `som_id_info` map từ element ID → bounding box coordinates

Go-Browse dùng BrowserGym `bid_XX` nhưng **không có visual overlay** trên screenshot.

### 2.4 Post-hoc task summarization

Sau khi trajectory kết thúc, Explorer gọi **TaskSummarizationAgent** để tạo ra một câu mô tả task hoàn chỉnh từ toàn bộ action history + screenshots. Điều này đảm bảo task description:
- Nhất quán với những gì thực sự đã làm
- Có đầy đủ context (constraints, budget, product details)
- Format chuẩn: `"<task> on <website>"`

Go-Browse: **không có bước này**. Task goal được dùng trực tiếp từ lúc propose.

### 2.5 Trajectory verification

Explorer có **TrajectoryVerifierAgent** phân loại:
- Task type: Transaction / Information seeking / Site navigation / Content modification
- Status: success / failure (với reasoning)

Go-Browse có Evaluator (GPT-4V) nhưng chỉ trả về success/failure, **không phân loại task type**.

### 2.6 Rich per-step metadata

Mỗi step trong Explorer chứa:
```json
{
  "step_action_nl": "Click on the 'Add to Cart' button",
  "new_action_grounded": "click [42]",
  "bounding_box_coord": {"x": 100, "y": 200, "width": 80, "height": 30},
  "step_refined_goal": "Purchase a coffee maker under $100 on Amazon",
  "step_reasoning_response": "...GPT-4o raw output...",
  "acc_tree_before": "...",
  "URL_after": "https://..."
}
```

Go-Browse mỗi step chứa:
```json
{
  "action": "click('bid_42')",
  "parsed_action": "click('bid_42')",
  "thought": "...LLM reasoning...",
  "observation": { "screenshot": ..., "axtree_txt": "...", ... },
  "misc": { "model_usage": ..., "agent_config": ... }
}
```

**Thiếu**: action_nl, refined_goal, bounding_box, element_metadata, URL tracking per step.

---

## 3. Phân tích chi tiết: Những gì Go-Browse còn thiếu

### 3.1 Data model — `TrajectoryStep` (`webexp/explore/core/trajectory.py`)

| Field | Hiện có | Cần thêm |
|-------|---------|----------|
| `action` | raw action string | ✅ |
| `parsed_action` | parsed action | ✅ |
| `thought` | LLM reasoning | ✅ |
| `observation` | full obs dict | ✅ |
| **`action_nl`** | ❌ | Natural language mô tả action |
| **`refined_goal`** | ❌ | Task description đã được refine ở step này |
| **`bounding_box`** | ❌ | Tọa độ element được tương tác |
| **`element_metadata`** | ❌ | Tag, text, attributes của element |
| **`page_url_before`** | ❌ | URL trước khi thực hiện action |
| **`page_url_after`** | ❌ | URL sau khi thực hiện action |
| **`action_reasoning`** | ❌ | Tại sao chọn action này (cho mục đích training) |

### 3.2 Data model — `Trajectory` (`webexp/explore/core/trajectory.py`)

| Field | Hiện có | Cần thêm |
|-------|---------|----------|
| `goal` | static goal | ✅ |
| `reward` | binary 0/1 | ✅ |
| `success` | bool | ✅ |
| **`summarized_goal`** | ❌ | Task description được summarize từ full trajectory |
| **`task_type`** | ❌ | Transaction / Info-seeking / Navigation / Modification |
| **`verification_reasoning`** | ❌ | Lý do success/failure (không chỉ binary) |
| **`task_tags`** | chỉ parse từ goal string | Cần được agent phân loại chính xác |

### 3.3 Agent pipeline — Thiếu các agent

| Agent | Explorer có | Go-Browse có |
|-------|------------|-------------|
| Task Proposal | ✅ TaskProposalAgent | ✅ PageExplorerAgent + NavExplorerAgent |
| Task Refinement | ✅ TaskRefinerAgent (mỗi step) | ❌ |
| Task Summarization | ✅ TaskSummarizationAgent | ❌ |
| Trajectory Verification | ✅ TrajectoryVerifierAgent | ✅ Evaluator (đơn giản hơn) |
| CAPTCHA Detection | ✅ CaptchaDetectionAgent | ❌ |

### 3.4 Visual grounding

- Explorer: **Set-of-Mark** — numbered tags overlay trên screenshot, element ID → bbox mapping
- Go-Browse: **Không có visual grounding** — agent dùng `bid_XX` nhưng screenshot không có annotation

---

## 4. Đề xuất cải tiến

### Phase 1: Làm giàu TrajectoryStep với semantic fields

**Mục tiêu**: Mỗi step có đủ thông tin để training model hiểu được ngữ nghĩa của action.

**Thay đổi trong `webexp/explore/core/trajectory.py`:**

```python
@dataclass
class TrajectoryStep:
    action: str | None
    parsed_action: str | None
    thought: str | None
    observation: dict
    # NEW fields:
    action_nl: str | None = None           # "Click on the search button"
    refined_goal: str | None = None        # evolving task description
    action_reasoning: str | None = None    # why this action was chosen
    bounding_box: dict | None = None       # {"x": ..., "y": ..., "width": ..., "height": ...}
    element_metadata: dict | None = None   # {"tag": "button", "text": "Search", "bid": "42"}
    page_url_before: str | None = None
    page_url_after: str | None = None
    misc: dict | None = None
```

**Thay đổi trong `SolverAgent.get_action()` (`webexp/agents/solver_agent.py`):**
- Yêu cầu LLM output thêm field `action_in_natural_language` và `refined_goal` trong JSON response
- Extract thông tin element từ axtree dựa trên bid được chọn

**Thay đổi trong prompt builder (`webexp/agents/prompt_builders/solver_prompt_builder.py`):**
- Thêm instruction yêu cầu LLM trả về JSON với format:
```json
{
  "thought": "...",
  "action": "click('bid_42')",
  "action_in_natural_language": "Click on the 'Add to Cart' button for the first search result",
  "refined_goal": "Purchase a Sony WH-1000XM5 headphones under $350 on Amazon"
}
```

### Phase 2: Thêm Task Refinement trong quá trình thực thi

**Mục tiêu**: Task description được cập nhật liên tục khi agent tương tác với website.

**Cách làm:**
- Ở mỗi step, `SolverAgent` không chỉ chọn action mà còn refine task description
- Prompt builder thêm context: "Your current understanding of the task is: {current_goal}. Based on what you see now, refine this task description if needed."
- Refined goal được lưu vào `TrajectoryStep.refined_goal`

**Implementation:**
- Sửa `SolverPromptBuilder.build_messages()` để include evolving goal
- `SolverAgent.get_action()` parse thêm field `refined_goal` từ LLM response
- Goal được truyền qua `callback_context` và cập nhật qua các step

### Phase 3: Thêm Task Summarization post-trajectory

**Mục tiêu**: Tạo ra một task description chất lượng cao từ toàn bộ trajectory.

**Tạo agent mới: `TaskSummarizationAgent`**

```python
# File mới: webexp/agents/task_summarization_agent.py
# Hoặc thêm function vào webexp/explore/core/evaluator.py

class TaskSummarizationAgent:
    """
    Sau khi trajectory hoàn thành, summarize task description 
    từ action history + screenshots.
    """
    def summarize(self, trajectory: Trajectory) -> str:
        # Gọi GPT-4V với:
        # - Toàn bộ action history (action_nl nếu có, nếu không thì parsed_action)
        # - Screenshots của các step quan trọng
        # - Yêu cầu output: task description format chuẩn
```

**Tích hợp vào `web_explore_loop()`:**
- Sau `sample_task_solving_trajectories_for_node()`, chạy summarization cho mỗi trajectory
- Lưu summarized goal vào `Trajectory.misc["summarized_goal"]`

### Phase 4: Thêm Set-of-Mark visual grounding

**Mục tiêu**: Screenshot có numbered tags để agent có thể reference element bằng số.

**Cách làm:**
- Port `set_of_mark.py` từ Explorer vào Go-Browse
- Tạo `ImageObservationProcessor` wrapper cho BrowserGym observations
- Thêm SoM-annotated screenshot như một field trong observation

**Implementation plan:**
1. Copy `Explorer/traj_gen/set_of_mark.py` → `webexp/explore/grounding/set_of_mark.py`
2. Copy `Explorer/traj_gen/page_script.js` → `webexp/explore/grounding/page_script.js`
3. Tạo `webexp/explore/grounding/som_processor.py` — wrapper để thêm SoM vào BrowserGym obs
4. Sửa `SolverAgent.obs_preprocessor()` để thêm SoM image vào observation
5. Sửa prompt builder để include instruction về cách dùng SoM tags

**Lưu ý**: BrowserGym đã có `bid_XX` trong axtree. SoM sẽ map từ `bid_XX` → numbered tag trên ảnh. Cần sync giữa bid và SoM tag number.

### Phase 5: Cải thiện Trajectory Verification

**Mục tiêu**: Verification không chỉ binary success/failure mà còn phân loại task type và đưa ra reasoning.

**Thay đổi trong `Evaluator` (`webexp/explore/core/evaluator.py`):**

- Thêm prompt yêu cầu LLM phân loại task type:
  - `Transaction`: mua hàng, book, add to cart...
  - `Information seeking`: tìm thông tin, so sánh...
  - `Site navigation`: điều hướng đến page cụ thể
  - `Content modification`: thay đổi nội dung, cấu hình
- Output format:
```
Thoughts: <reasoning>
Task Type: <transaction|information_seeking|site_navigation|content_modification>
Status: "success" or "failure"
Failure Reason: <nếu failure, giải thích lý do>
```

- Lưu vào `Trajectory.misc["evaluation_info"]`:
  - `task_type`
  - `failure_reason` (nếu có)

### Phase 6: Cải thiện Task Proposal (PageExplorer)

**Mục tiêu**: Task được propose có chất lượng cao hơn, concrete hơn, và có format chuẩn.

**Thay đổi trong `PageExplorerAgent`:**
- Prompt thêm yêu cầu về format task: `"<action verb> <specific details> on <website>"`
- Thêm yêu cầu về constraints: budget, rating, location, date...
- Agent không chỉ propose task mà còn **thực thi thử first action** (giống Explorer's TaskProposalAgent)

### Phase 7: Thêm CAPTCHA detection

**Mục tiêu**: Tự động phát hiện và skip CAPTCHA pages.

**Tạo agent mới: `CaptchaDetectionAgent`**
- Gọi GPT-4V với screenshot để check "does this page contain a captcha?"
- Nếu có → skip node, không lãng phí LLM calls

---

## 5. Implementation Priority & Effort

| Phase | Impact | Effort | Dependencies |
|-------|--------|--------|-------------|
| **Phase 1**: Semantic fields in TrajectoryStep | 🔴 HIGH | 🟢 Low (2-3 files) | None |
| **Phase 2**: Task Refinement | 🔴 HIGH | 🟡 Medium (prompt + agent) | Phase 1 |
| **Phase 3**: Task Summarization | 🟡 MEDIUM | 🟢 Low (1 new agent) | Phase 1 |
| **Phase 4**: Set-of-Mark | 🟡 MEDIUM | 🔴 High (port code, sync bid/tag) | None |
| **Phase 5**: Enhanced Verification | 🟡 MEDIUM | 🟢 Low (prompt change) | None |
| **Phase 6**: Better Task Proposal | 🟢 LOW | 🟡 Medium | None |
| **Phase 7**: CAPTCHA Detection | 🟢 LOW | 🟢 Low (1 new agent) | None |

**Khuyến nghị**: Bắt đầu với Phase 1 + Phase 2 + Phase 3 vì đây là core của "semantic trajectory" và effort thấp nhất. Phase 4 (SoM) có impact lớn nhất về visual grounding nhưng effort cao nhất.

---

## 6. Code changes summary

### Files cần sửa:

| File | Phase | Thay đổi |
|------|-------|----------|
| `webexp/explore/core/trajectory.py` | 1 | Thêm fields: `action_nl`, `refined_goal`, `action_reasoning`, `bounding_box`, `element_metadata`, `page_url_before`, `page_url_after` |
| `webexp/agents/solver_agent.py` | 1, 2 | Parse thêm fields từ LLM response; thêm task refinement logic |
| `webexp/agents/prompt_builders/solver_prompt_builder.py` | 1, 2 | Thêm output format yêu cầu `action_nl` + `refined_goal` |
| `webexp/explore/core/episode.py` | 1, 2 | Truyền refined_goal qua các step; lưu URL per step |
| `webexp/explore/core/evaluator.py` | 5 | Thêm task type classification + failure reason |

### Files cần tạo mới:

| File | Phase | Mục đích |
|------|-------|----------|
| `webexp/agents/task_summarization_agent.py` | 3 | Post-hoc task summarization |
| `webexp/agents/captcha_detection_agent.py` | 7 | CAPTCHA detection |
| `webexp/explore/grounding/set_of_mark.py` | 4 | SoM overlay (port từ Explorer) |
| `webexp/explore/grounding/som_processor.py` | 4 | SoM wrapper cho BrowserGym |

---

## 7. Ví dụ: TrajectoryStep trước và sau cải tiến

### Trước (hiện tại):
```json
{
  "action": "click('bid_42')",
  "parsed_action": "click('bid_42')",
  "thought": "I need to click on the search result to view the product details.",
  "observation": {
    "screenshot": "<numpy_array>",
    "axtree_txt": "[42] [LINK] [Sony WH-1000XM5 ...]",
    ...
  },
  "misc": {
    "model_usage": {"prompt_tokens": 1500, "completion_tokens": 50},
    "agent_config": {"name": "SolverAgent", "model_id": "gpt-4o-mini"}
  }
}
```

### Sau (với semantic fields):
```json
{
  "action": "click('bid_42')",
  "parsed_action": "click('bid_42')",
  "thought": "I need to click on the first search result which is the Sony WH-1000XM5 to view its product details.",
  "action_nl": "Click on the 'Sony WH-1000XM5 Wireless Headphones' link in the search results",
  "refined_goal": "Find and purchase Sony WH-1000XM5 wireless headphones under $350 with free shipping on Amazon",
  "action_reasoning": "This is the first search result matching the target product, and the price appears to be within budget",
  "bounding_box": {"x": 245, "y": 380, "width": 600, "height": 24},
  "element_metadata": {"tag": "link", "text": "Sony WH-1000XM5 Wireless...", "bid": "42", "role": "link"},
  "page_url_before": "https://www.amazon.com/s?k=sony+wh-1000xm5",
  "page_url_after": "https://www.amazon.com/s?k=sony+wh-1000xm5",
  "observation": { ... },
  "misc": {
    "model_usage": {"prompt_tokens": 1800, "completion_tokens": 120},
    "agent_config": {"name": "SolverAgent", "model_id": "gpt-4o-mini"}
  }
}
```

---

## 8. Risks & Considerations

1. **Token cost tăng**: Yêu cầu LLM output thêm fields (`action_nl`, `refined_goal`) → completion tokens tăng ~2-3x
2. **Latency tăng**: Thêm summarization step sau mỗi trajectory → thêm 1 LLM call
3. **Prompt engineering**: Cần điều chỉnh prompt cẩn thận để LLM output đúng format
4. **Backward compatibility**: Cần migrate trajectory data cũ hoặc hỗ trợ cả old + new format
5. **SoM porting**: Code Explorer dùng Playwright trực tiếp, Go-Browse dùng BrowserGym → cần adapter
