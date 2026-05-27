# Đề Xuất Cải Tiến Semantic Enhancement Cho Go-Browse

## Mục Tiêu

Tài liệu này đề xuất các chỉnh sửa cụ thể để phần semantic enhancement hiện tại
thực sự cải thiện chất lượng dữ liệu Go-Browse, thay vì chỉ lưu thêm metadata.

Đề xuất dựa trên:

- `docs/papers/explorer.pdf`: pipeline semantic trajectory synthesis của Explorer.
- `docs/papers/go-browse.pdf`: pipeline structured exploration của Go-Browse.
- Code hiện tại trong `webexp/`.
- Dữ liệu run hiện tại trong `runs/go_browse_shopping/graph`.

Kết luận chính: semantic enhancement hiện tại đã có schema và một số hook đúng
hướng, nhưng chưa tạo thành vòng kiểm soát chất lượng dữ liệu như Explorer.
Nó cần được dùng để relabel task, verify trajectory, kiểm tra grounding, và tạo
training export tốt hơn.

## Chẩn Đoán Hiện Trạng

### Phần Đã Có Tác Dụng

Code hiện tại đã có một số thành phần semantic:

- Solver có thể trả về `action_in_natural_language` và `refined_goal`.
- `TrajectoryStep` lưu `action_nl`, `refined_goal`, `element_metadata`,
  `page_url_before`, và `page_url_after`.
- Evaluator ưu tiên `action_nl` thay vì raw action khi dựng action history.
- `TaskSummarizationAgent` chạy sau khi explore node và ghi `summarized_goal`.
- `task_info.json` có field `summarized_goal`.
- Prompt solver đã có cơ chế giữ original goal bất biến và current objective
  dạng evolving.

### Phần Còn Yếu

Dữ liệu run hiện tại cho thấy semantic layer chưa thay đổi chất lượng task nhiều:

- Kiểm tra 27 task.
- 18 task có `summarized_goal: null`.
- 6 task có `summarized_goal` y hệt `goal`.
- Chỉ 3 task có `summarized_goal` khác `goal`, và khác biệt chủ yếu là rewrite
  nhẹ.
- `bounding_box` chưa được populate trong run đã kiểm tra.
- SoM đã có code nhưng chưa được bật trong config hiện tại.
- CAPTCHA detection đã implement nhưng đang bị tắt trong config.

Điều này cho thấy semantic enhancement hiện tại chủ yếu là cải tiến logging và
schema. Nó chưa trở thành vòng `propose -> refine -> summarize -> verify ->
filter` như Explorer.

## Định Hướng Thiết Kế

Không nên thay Go-Browse bằng Explorer. Go-Browse vẫn nên giữ thế mạnh chính:

- graph exploration theo các URL đã phát hiện
- reset về node đã biết
- PageExplorer và NavExplorer để đề xuất task
- FeasibilityChecker để lọc task có thể làm được
- prefixed và unprefixed trajectory sampling

Phần nên học từ Explorer là semantic data-quality layer:

- action label giàu ngữ nghĩa hơn
- relabel task ở cấp trajectory
- lưu lịch sử intent refinement
- kiểm tra action grounding
- semantic verification sau khi trajectory kết thúc
- export training data dùng semantic fields

## Đề Xuất Chỉnh Sửa

## 1. Thay `refined_goal` Dạng Chuỗi Bằng Task State Có Cấu Trúc

### Vấn Đề

`refined_goal` hiện tại chỉ là một chuỗi và bị overwrite qua từng step. Nó
thường rơi vào hai trường hợp:

- lặp lại gần giống original goal
- mô tả trạng thái cục bộ như `Cart contains product X`

Cách này làm mất nhiều thông tin quan trọng:

- entity hoặc sản phẩm đã xác nhận
- ràng buộc phát hiện trên trang
- quan sát phủ định
- bằng chứng infeasible
- điều kiện hoàn thành
- việc current objective có bị lệch khỏi original goal hay không

### Đề Xuất

Thêm `task_state` có cấu trúc, đồng thời vẫn giữ `refined_goal` để backward
compatible.

Schema đề xuất:

```json
{
  "original_goal": "Add product X to cart on One Stop Market",
  "current_objective": "Cart contains product X with quantity 1",
  "confirmed_entities": ["product X", "One Stop Market"],
  "constraints": ["quantity 1"],
  "negative_observations": [],
  "completion_condition": "product X visible in shopping cart",
  "infeasibility_reason": null
}
```

### Cách Triển Khai

- Giai đoạn đầu lưu `task_state` trong `TrajectoryStep.misc`.
- Sau khi ổn định mới promote thành field chính thức của `TrajectoryStep`.
- Prompt solver yêu cầu chỉ cập nhật field có thay đổi.
- Luôn hiển thị original goal như immutable context.
- Không cho `current_objective` trở thành action-level goal như `click`,
  `scroll`, `type`, `wait`.

### Tác Dụng Kỳ Vọng

Refinement sẽ trở thành bộ nhớ có cấu trúc cho trajectory, dùng được cho verify,
debug và training export, thay vì chỉ là một chuỗi dễ trùng lặp.

## 2. Sửa Summarization Để Sinh Task Đã Thực Sự Hoàn Thành

### Vấn Đề

Summarizer hiện nhận original goal và dễ copy lại goal. Ở cấp task,
`summarized_goal` vì vậy thường gần như y hệt `goal`.

Điều này trái với tinh thần Explorer: task summarizer phải nhìn toàn bộ action
và screenshot history để suy ra task tổng thể đã được thực hiện, không chỉ lặp
lại task ban đầu.

### Đề Xuất

Tách output của summarizer thành các field rõ nghĩa hơn:

```json
{
  "accomplished_goal": "Add the Pre-baked Gingerbread House Kit to the cart on One Stop Market",
  "completion_evidence": "Final cart page contains the target product",
  "summary_differs_from_original": false
}
```

Trong đó:

- `goal`: task được đề xuất ban đầu.
- `accomplished_goal`: task thực sự được action history chứng minh.
- `completion_evidence`: bằng chứng ngắn gọn từ final state hoặc action history.

### Sửa Prompt Summarizer

Prompt nên yêu cầu:

- Suy ra task từ action history và screenshots trước.
- Chỉ dùng original goal như hint phụ khi action history mơ hồ.
- Không copy original goal nếu action history không chứng minh đầy đủ.
- Không mô tả cách làm như `click My Cart`, `wait`, `scroll`.
- Giữ task ở mức high-level nhưng đủ cụ thể.
- Thêm constraint đã xác nhận trong quá trình chạy.

### Sửa Logic Chọn Summary Ở Cấp Task

Không nên chọn summary chỉ vì nó xuất hiện nhiều nhất. Thay vào đó chọn summary
dựa trên:

- có bằng chứng hoàn thành rõ
- không procedural
- đủ cụ thể
- khớp final state
- không bị copy máy móc từ original goal nếu không cần thiết

### Tác Dụng Kỳ Vọng

`summarized_goal` hoặc `accomplished_goal` sẽ trở thành nhãn task đáng tin cậy
cho training, thay vì metadata gần như trùng với `goal`.

## 3. Thêm Semantic Trajectory Verifier

### Vấn Đề

Evaluator hiện chủ yếu quyết định success/failure theo `trajectory.goal`. Nó
chưa kiểm tra mạnh các điểm sau:

- `action_nl` có khớp raw action không
- raw action có khớp element được click/type không
- `refined_goal` có bị drift khỏi original goal không
- `summarized_goal` có được action history hỗ trợ không
- final page có bằng chứng hoàn thành task không

Explorer discard trajectory bị incoherent hoặc misaligned. Go-Browse cũng nên có
bước này nếu muốn dữ liệu semantic có chất lượng cao.

### Đề Xuất

Thêm `SemanticVerifierAgent` chạy sau summarization.

Input:

- original goal
- accomplished/summarized goal
- action history gồm raw action và `action_nl`
- task state hoặc refined goal history
- final URL
- final a11y tree hoặc markdown
- sampled screenshots trong trajectory

Output đề xuất:

```json
{
  "is_aligned": true,
  "is_grounded": true,
  "is_complete": true,
  "failure_reason": null,
  "evidence": "Cart page lists the target product"
}
```

### Chính Sách Lọc

Trajectory chỉ nên được đưa vào positive training export nếu:

- `is_aligned == true`
- `is_grounded == true`
- `is_complete == true`

Trajectory bị reject vẫn nên lưu lại cùng lý do để debug, nhưng không dùng như
positive demonstration.

### Tác Dụng Kỳ Vọng

Đây là chỉnh sửa quan trọng nhất. Nó biến semantic enhancement từ annotation
thành data curation.

## 4. Kiểm Tra `action_nl` Với Grounded Action Và Element Metadata

### Vấn Đề

Explorer yêu cầu natural-language action phải nhất quán với grounded action.
Go-Browse hiện lưu cả hai nhưng chưa validate.

Ví dụ lỗi:

- raw action click bid `123`
- bid `123` là `Add to Compare`
- `action_nl` lại ghi `Click Add to Cart`

Nếu đưa dữ liệu này vào training, model sẽ học sai grounding.

### Đề Xuất

Thêm validator nhẹ sau mỗi solver action:

1. Parse target bid từ raw action.
2. Lấy role/name/text từ `element_metadata`.
3. Kiểm tra `action_nl` có mô tả đúng target element hoặc text được type không.
4. Ghi `grounding_valid: true/false` vào step.
5. Nếu invalid, có thể repair bằng LLM hoặc để semantic verifier reject
   trajectory.

### Tác Dụng Kỳ Vọng

Giảm nhiễu trong action supervision, đặc biệt quan trọng khi train model có khả
năng grounding trên UI.

## 5. Bật Và Sửa Set-of-Mark Grounding

### Vấn Đề

SoM đã được implement nhưng config hiện tại chưa bật `use_som`. Ngoài ra
`bounding_box` chưa được populate trong run đã kiểm tra, nên phần visual
grounding chưa đạt format giống Explorer.

### Đề Xuất

Bật `use_som: true` trong config dùng cho semantic data collection:

```yaml
page_explorers:
  - agent_factory_args:
      use_som: true

nav_explorers:
  - agent_factory_args:
      use_som: true

feasibility_checkers:
  - agent_factory_args:
      use_som: true

solvers:
  - agent_factory_args:
      use_som: true
```

Nên persist thêm:

- raw screenshot
- SoM screenshot
- element bbox
- mapping từ SoM tag sang BrowserGym bid

### Cách Triển Khai

- Kiểm tra lại key mapping giữa `extra_element_properties` và a11y node.
- Nếu không lấy được bbox, ghi `grounding_missing_reason`.
- Không silently coi step là grounded nếu bbox thiếu.

### Tác Dụng Kỳ Vọng

Dataset sẽ phù hợp hơn với training multimodal web agent: model thấy ảnh có
mark, a11y tree, action_nl và grounded action cùng lúc.

## 6. Cải Thiện Context Cho Evaluator

### Vấn Đề

Evaluator hiện dùng final screenshot và final a11y tree. Với nhiều task, final
state không đủ để chứng minh quá trình đã đúng:

- chọn option sản phẩm
- checkout flow
- review page
- success message chỉ xuất hiện tạm thời
- modal interaction

Explorer verifier dùng action history, final markdown và screenshots của các
webpage tương ứng với action.

### Đề Xuất

Evaluator nên nhận thêm:

- sampled screenshots trong trajectory
- final a11y tree
- final DOM hoặc markdown
- final URL
- raw action
- `action_nl`
- `task_state` history
- accomplished/summarized goal

### Tác Dụng Kỳ Vọng

Nhãn success/failure đáng tin hơn, failure reason rõ hơn, và ít accept nhầm các
trajectory chỉ “trông có vẻ đúng” ở final page.

## 7. Export Training Data Từ Semantic Fields

### Vấn Đề

Nếu training vẫn chỉ dùng original `goal` và raw action, semantic enhancement sẽ
không cải thiện model nhiều. Explorer có tác dụng vì dữ liệu training chứa task,
ảnh/SoM, a11y, action natural language và grounded action.

### Đề Xuất

Tạo semantic export format:

```json
{
  "instruction": "accomplished_goal or summarized_goal or original goal",
  "observation": {
    "a11y_tree": "...",
    "screenshot_som": "...",
    "url": "..."
  },
  "history": [
    {
      "action_nl": "Click Add to Cart",
      "grounded_action": "click('1735')"
    }
  ],
  "target": {
    "action_nl": "Click My Cart",
    "grounded_action": "click('246')"
  },
  "quality": {
    "semantic_verified": true,
    "grounding_valid": true
  }
}
```

Training nên ưu tiên:

- verified positive trajectories
- steps có grounding hợp lệ
- SoM screenshot nếu có
- `accomplished_goal` thay vì original `goal` khi đã verify

Không nên train model predict `refined_goal` nếu field này vẫn noisy. Trước mắt
nên dùng nó làm context hoặc filter signal.

## 8. Siết Chất Lượng Task Proposal

### Vấn Đề

Go-Browse đã có PageExplorer/NavExplorer, nhưng semantic enhancement hiện chủ
yếu xảy ra sau khi task đã được đề xuất. Trong Explorer, task proposal có rule
chặt hơn về feasibility, specificity, no-login, action consistency và realistic
constraints.

### Đề Xuất

Update PageExplorer/NavExplorer prompt để task proposal có schema rõ:

```json
{
  "task": "Add product X to cart on One Stop Market",
  "type": "TRANSACTION",
  "evidence": "Product card has Add to Cart button",
  "completion_condition": "Cart contains product X",
  "requires_login": false
}
```

Yêu cầu task:

- một task ứng với một user intent cụ thể
- không yêu cầu login, payment, CAPTCHA hoặc thông tin ẩn bên ngoài
- có constraint thực tế nếu cần
- có task type: `INFO`, `NAV`, `MOD`, `TRANSACTION`
- có evidence từ page hiện tại
- có completion condition rõ

### Tác Dụng Kỳ Vọng

Giảm task mơ hồ hoặc infeasible trước khi tốn chi phí chạy feasibility checker
và solver.

## Roadmap Triển Khai

### Phase 1: Làm Semantic Fields Hiện Có Trở Nên Hữu Ích

- Thêm `accomplished_goal` và `completion_evidence` vào summarizer output.
- Sửa logic chọn task-level summary.
- Thêm validator cơ bản cho `action_nl` và `element_metadata`.
- Thêm test cho summary selection và action grounding validation.

### Phase 2: Thêm Semantic Verification

- Implement `SemanticVerifierAgent`.
- Lưu `semantic_verification` trong `traj.misc`.
- Loại trajectory không đạt semantic verification khỏi positive training export.
- Tạo debug report cho trajectory bị reject.

### Phase 3: Structured Task State

- Thêm `task_state` vào step misc.
- Update solver prompt để emit structured task state.
- Giữ `refined_goal` cũ để backward compatible.
- Thêm guardrail chống drift và action-level objective.

### Phase 4: Multimodal Grounding

- Bật `use_som` trong config semantic collection.
- Persist raw screenshot và SoM screenshot.
- Sửa bbox extraction và SoM tag mapping.
- Ghi metric về grounding completeness.

### Phase 5: Semantic Training Export

- Tạo script export dataset theo semantic format.
- Chỉ export positive trajectories đã verify.
- Ưu tiên `accomplished_goal` khi có.
- Update training docs để dùng semantic export.

## Metrics Nên Theo Dõi

Các metric nên ghi sau mỗi run:

- `% task có accomplished_goal không null`
- `% accomplished_goal khác original goal khi action history chứng minh được`
- `% summary pass semantic verification`
- `% step có action_nl`
- `% step có valid action grounding`
- `% step có bbox`
- `% trajectory có SoM screenshot`
- tỷ lệ task trùng lặp giữa các node
- phân bố task type
- số positive trajectory đã verify

## Thứ Tự Ưu Tiên

Thứ tự triển khai nên là:

1. Semantic verifier.
2. Sửa summarizer và logic chọn summary.
3. Action grounding validation.
4. Structured task state.
5. Bật và sửa SoM/bbox.
6. Semantic training export.
7. Siết task proposal schema.

Nếu không có verification và training export, semantic enhancement sẽ vẫn chủ
yếu là metadata mô tả. Khi có hai phần đó, Go-Browse có thể giữ lợi thế graph
exploration, đồng thời đạt chất lượng semantic trajectory gần hơn với Explorer.
