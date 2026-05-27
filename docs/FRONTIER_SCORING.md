# Frontier Scoring — Hướng dẫn chi tiết theo code hiện tại

## Tổng quan

`FrontierScorer` là module dùng để ưu tiên chọn node/URL tiếp theo trong frontier của Go-Browse. Thay vì luôn chọn node theo FIFO, mỗi node được gán một điểm tổng hợp:

$$
S = \alpha U + \beta V + \theta D
$$

Trong đó:

- `U` — **Uncertainty**: độ bất định ước lượng từ các action candidate do `LookaheadPredictor` dự đoán.
- `V` — **Value**: giá trị khai thác dựa trên success rate của các trajectory đã chạy từ node.
- `D` — **Diversity**: độ đa dạng UI/DOM dựa trên khoảng cách cosine giữa các embedding trạng thái liên tiếp.

Node có score cao nhất sẽ được chọn để explore trước.

Điểm quan trọng của phiên bản code hiện tại là: **không phải mọi thành phần đều được tính từ dữ liệu exploration thật ngay từ đầu**. Với node mới, hệ thống dùng một phần giá trị mặc định hoặc giá trị lookahead để có thể chấm điểm trước khi chạy PageExplorer/NavExplorer thật.

---

## 1. Luồng hoạt động tổng quát

### Bước 1 — Node được thêm vào frontier

Khi một URL mới được phát hiện và thêm vào frontier, node đó có thể chưa có trajectory, chưa có success rate và chưa có embedding.

Nếu có accessibility tree snippet, hệ thống có thể dùng `LookaheadPredictor` để dự đoán trước một số action khả thi trên trang:

```python
node.lookahead_candidates = lookahead_predictor.propose(axtree_snippet)
```

Mỗi candidate có dạng:

```python
{
    "action": "Search for a product in the search bar",
    "confidence": 0.95,
}
```

Các confidence score này được dùng để tính `U`.

### Bước 2 — Tính score cho từng node trong frontier

Với mỗi node, `FrontierScorer.compute(node)` gọi:

```python
U = self.uncertainty(node)
V = self.value(node)
D = self.diversity(node)
S = self.alpha * U + self.beta * V + self.theta * D
```

### Bước 3 — Chọn node có score cao nhất

Graph sẽ sort các node chưa explore theo score giảm dần:

```python
scored_nodes = [(node, self.scorer.compute(node))
                for node in self.unexplored_nodes]
scored_nodes.sort(key=lambda x: x[1], reverse=True)
best_node, best_score = scored_nodes[0]
```

Node `best_node` sẽ được lấy ra để chạy vòng explore tiếp theo.

---

## 2. `LookaheadPredictor` — dự đoán action trước khi explore thật

### Mục đích

`LookaheadPredictor` là một predictor nhẹ, dùng LLM rẻ để nhìn vào accessibility tree của trang và đề xuất một số action mà người dùng có thể thực hiện.

Mục tiêu của nó là tạo dữ liệu prior cho node mới, trước khi tốn chi phí chạy PageExplorer/NavExplorer hoặc solver thật.

### Input

```python
axtree_snippet: str
```

Đây là đoạn accessibility tree, thường bị cắt còn khoảng 3500 ký tự đầu:

```python
axtree_snippet[:3500]
```

### Output

```python
list[dict]
```

Ví dụ:

```python
[
    {"action": "Search for a product in the search bar", "confidence": 0.95},
    {"action": "Click on a category link", "confidence": 0.70},
    {"action": "Open the shopping cart", "confidence": 0.40},
]
```

### Fallback

Nếu LLM call lỗi, parse JSON lỗi, hoặc output không đúng format, predictor trả về các candidate trung lập:

```python
[
    {"action": "generic_interaction_0", "confidence": 0.5},
    {"action": "generic_interaction_1", "confidence": 0.5},
    ...
]
```

Khi đó uncertainty sẽ thấp vì các confidence đều bằng nhau.

### Lưu ý kỹ thuật

Trong code hiện tại, prompt yêu cầu model trả về JSON array, nhưng API lại dùng:

```python
response_format={"type": "json_object"}
```

Vì vậy model có thể trả về object dạng:

```python
{
    "candidates": [
        {"action": "...", "confidence": 0.9}
    ]
}
```

Code hiện tại đã xử lý các wrapper key phổ biến:

```python
"candidates", "actions", "predictions"
```

---

## 3. Công thức tổng hợp

```python
def compute(self, node: Node) -> float:
    U = self.uncertainty(node)
    V = self.value(node)
    D = self.diversity(node)
    return self.alpha * U + self.beta * V + self.theta * D
```

Và hàm debug:

```python
def breakdown(self, node: Node) -> dict:
    U = self.uncertainty(node)
    V_sr, V_sr_source = self._estimate_success_rate_with_source(node)
    V = self.value(node)
    D = self.diversity(node)
    S = self.alpha * U + self.beta * V + self.theta * D
    return {
        "score": S,
        "U": U,
        "V": V,
        "V_sr": V_sr,
        "V_sr_source": V_sr_source,
        "D": D,
        "alpha": self.alpha,
        "beta": self.beta,
        "theta": self.theta,
    }
```

---

## 4. Uncertainty (`U`) — độ bất định lookahead

### Công thức trong code hiện tại

$$
U = \frac{\sigma}{\mu + \varepsilon}
$$

Trong code hiện tại:

- `μ` là trung bình các confidence score do `LookaheadPredictor` sinh ra.
- `σ` trong comment đang được gọi là variance, nhưng thực tế biến `sigma` đang chứa **variance**, không phải standard deviation.
- `ε` là hằng số nhỏ để tránh chia cho 0, mặc định `1e-5`.

Code hiện tại:

```python
def uncertainty(self, node: Node) -> float:
    candidates = node.lookahead_candidates
    if not candidates:
        return 1.0

    confidences = [c["confidence"] for c in candidates if isinstance(c, dict)]
    if not confidences:
        return 1.0

    n = len(confidences)
    mu = sum(confidences) / n
    sigma = sum((c - mu) ** 2 for c in confidences) / n  # variance
    return sigma / (mu + self.epsilon)
```

### Ý nghĩa

`U` hiện tại **không còn được tính từ success rate của các exploration task thật**. Nó được tính từ độ phân tán của các confidence score mà LLM lookahead dự đoán.

Nói cách khác:

```text
U cao  → các action candidate có confidence phân hóa mạnh.
U thấp → các action candidate có confidence gần giống nhau.
```

Ví dụ:

```python
confidences = [0.95, 0.80, 0.30, 0.10]
```

Các confidence rất khác nhau, nên variance cao hơn → `U` cao hơn.

```python
confidences = [0.80, 0.78, 0.82, 0.79]
```

Các confidence gần nhau, nên variance thấp → `U` thấp hơn.

### Với node mới

Nếu node chưa có `lookahead_candidates`, code trả về:

```python
U = 1.0
```

Đây là cold-start default để khuyến khích node mới được thử ít nhất một lần.

### Điểm cần ghi nhớ

Trong tài liệu cũ, `U` được mô tả là variance của success rate giữa các task. Điều đó **không khớp với code hiện tại**. Theo code hiện tại, `U` phải được hiểu là:

> Độ bất định ước lượng trước exploration, tính từ variance của confidence score trong các action candidate do `LookaheadPredictor` sinh ra.

---

## 5. Value (`V`) — giá trị khai thác

### Công thức

$$
V = SR \cdot \frac{1}{\log(n + 2)}
$$

Trong đó:

- `SR` là success rate của các trajectory đã chạy từ node.
- `n` là `exploration_count`, số lần node đã được ghé thăm/lấy mẫu.
- `1 / log(n + 2)` là penalty giảm dần để tránh chọn mãi một node.

Code hiện tại:

```python
def value(self, node: Node) -> float:
    sr, _ = self._estimate_success_rate_with_source(node)
    n = max(node.exploration_count, 0)
    penalty = 1.0 / math.log(n + 2)
    return sr * penalty
```

Success rate dùng trong `V` được lấy theo thứ tự:

1. `self`: nếu chính node có `total_trajs > 0`, dùng `node.success_rate`.
2. `ancestor:<url>`: nếu node chưa có trajectory, dùng ancestor gần nhất có `total_trajs > 0`.
3. `prior`: nếu không có ancestor hợp lệ hoặc không tìm thấy parent, dùng prior trung lập `0.5`.

Helper debug:

```python
def _estimate_success_rate_with_source(self, node: Node) -> tuple[float, str]:
    ...
```

`breakdown()` log thêm `V_sr` và `V_sr_source` để biết `V` đang dùng success rate từ đâu.

### Ý nghĩa

`V` là phần exploitation của score:

```text
V cao  → node từng tạo ra nhiều trajectory thành công và chưa bị khai thác quá nhiều.
V thấp → node ít thành công hoặc đã bị explore nhiều lần.
```

Ví dụ:

```text
Node A: SR = 0.8, exploration_count = 1  → V tương đối cao
Node B: SR = 0.8, exploration_count = 30 → V thấp hơn vì bị phạt theo log
```

### Với node chưa có trajectory

Nếu node chưa có trajectory:

```python
node.total_trajs = 0
```

thì code dùng:

```python
sr, source = self._estimate_success_rate_with_source(node)
```

Nghĩa là `V` lúc này dùng estimate cho cold-start node:

- nếu parent/ancestor gần nhất có trajectory, `sr` kế thừa từ ancestor đó;
- nếu không có parent, không tìm thấy parent, hoặc ancestor chưa có dữ liệu, `sr = 0.5`.

Với `n = 0`:

```python
V = sr * (1 / math.log(2))
```

Vì `math.log` là log tự nhiên, nên:

```text
1 / log(2) ≈ 1.4427
V ≈ 0.7213
```

Do đó, ở node mới, `V` có thể lớn hơn 0.5. Ví dụ fallback prior `sr=0.5` cho `V≈0.7213`; nếu kế thừa `sr=0.8` từ parent thì `V≈1.1542`.

### Điểm cần ghi nhớ

`V` phản ánh success rate thật của chính node sau khi node đã có trajectory. Trước đó, nó là estimate kế thừa từ parent/ancestor gần nhất có dữ liệu, hoặc prior `0.5` nếu không có dữ liệu để kế thừa.

---

## 6. Diversity (`D`) — độ đa dạng UI/DOM

### Công thức

$$
D = \frac{1}{m} \sum_{i=1}^{m} \text{cos\_distance}(embedding_i, embedding_{i+1})
$$

Trong đó:

$$
\text{cos\_distance}(a,b) = 1 - \frac{a \cdot b}{\|a\|\|b\|}
$$

Code hiện tại:

```python
@staticmethod
def cosine_distance(a: List[float], b: List[float], eps: float = 1e-10) -> float:
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    cos_sim = dot / (norm_a * norm_b + eps)
    return 1.0 - cos_sim


def diversity(self, node: Node) -> float:
    if node.embedding is None or len(node.embedding) < 2:
        return 0.5

    distances = []
    for i in range(len(node.embedding) - 1):
        dist = self.cosine_distance(node.embedding[i], node.embedding[i + 1])
        distances.append(dist)
    return sum(distances) / len(distances) if distances else 0.5
```

### Ý nghĩa

`D` đo xem trong quá trình tương tác với node, trạng thái UI/DOM có thay đổi nhiều không.

```text
D cao  → các state liên tiếp khác nhau nhiều, node có thể có nhiều chức năng/luồng tương tác.
D thấp → UI ít thay đổi, node có thể tĩnh hoặc ít chức năng.
```

Ví dụ node có nhiều tab, dropdown, form, filter, modal:

```text
state 1: product edit page
state 2: mở tab Inventory
state 3: mở dropdown Stock Status
state 4: mở tab Advanced Pricing
```

Các state khác nhau nhiều → cosine distance cao hơn → `D` cao hơn.

### Với node chưa có embedding

Nếu node chưa có ít nhất 2 embedding:

```python
D = 0.5
```

Đây là neutral default. Nó không phải diversity thật.

### Điểm cần ghi nhớ

`D` chỉ có ý nghĩa sau khi node đã được tương tác qua nhiều state và `node.embedding` đã được populate. Với node mới, `D = 0.5` chỉ là giá trị mặc định.

---

## 7. Cách hiểu score với node mới và node đã explore

### Node mới chưa explore

Với node mới, thường chưa có trajectory và embedding.

Nếu chưa có lookahead candidates:

```text
U = 1.0
V = inherited_or_prior_sr / log(2)
D = 0.5
```

Nếu có lookahead candidates:

```text
U = variance(confidence_scores) / mean(confidence_scores)
V = inherited_or_prior_sr / log(2)
D = 0.5
```

Nghĩa là node mới được chấm chủ yếu bằng:

- độ bất định lookahead `U`, nếu có;
- inherited value `V` từ parent/ancestor gần nhất có dữ liệu, hoặc prior `0.5`;
- neutral diversity `D`.

### Node đã explore

Sau khi node đã có trajectory và embedding:

```text
U = vẫn tính từ lookahead_candidates theo code hiện tại
V = success_rate thật, có penalty theo exploration_count
D = mean cosine distance thật giữa các embedding state
```

Lưu ý: code hiện tại **không tự chuyển `U` sang dùng task success rate thật**. Nếu muốn `U` dùng observed task uncertainty, cần sửa code riêng.

---

## 8. Files đã thay đổi

### 8.1 `webexp/explore/core/scoring.py`

Module chính chứa:

```python
class LookaheadPredictor:
    def propose(axtree_snippet: str) -> list[dict]
    def _fallback() -> list[dict]

class FrontierScorer:
    def compute(node) -> float
    def breakdown(node) -> dict
    def uncertainty(node) -> float
    def value(node) -> float
    def _estimate_success_rate_with_source(node) -> tuple[float, str]
    def diversity(node) -> float
    def cosine_distance(a, b) -> float
    def to_dict() -> dict
    def from_dict(d) -> FrontierScorer
```

Trong đó:

- `LookaheadPredictor` tạo `lookahead_candidates` từ accessibility tree.
- `FrontierScorer` tính score tổng hợp `S = αU + βV + θD`.

### 8.2 `webexp/explore/core/node.py`

Theo logic hiện tại, node cần có các trường sau để scoring hoạt động đầy đủ:

| Trường | Kiểu | Mục đích |
|---|---|---|
| `lookahead_candidates` | `list[dict]` | Danh sách action candidate và confidence do `LookaheadPredictor` sinh ra |
| `exploration_count` | `int` | Số lần node đã được explore/lấy mẫu |
| `success_rate` | `float` | Tỷ lệ trajectory thành công từ node |
| `total_trajs` | `int` | Tổng số trajectory đã chạy từ node |
| `successful_trajs` | `int` | Số trajectory thành công |
| `embedding` | `list[list[float]]` hoặc `None` | Danh sách embedding của các UI/DOM state |
| `parent_url` | `str` hoặc `None` | URL cha, persist trong `node_info.json` để debug/load graph |
| `parent` | `Node` hoặc `None` | Runtime reference tới node cha, không serialize trực tiếp |

Phương thức cập nhật outcome:

```python
def record_trajectory_outcome(self, success: bool):
    self.total_trajs += 1
    if success:
        self.successful_trajs += 1
    self.success_rate = self.successful_trajs / max(self.total_trajs, 1)
    self.exploration_count += 1
```

### 8.3 `webexp/explore/core/graph.py`

Graph giữ một instance của scorer:

```python
self.scorer = FrontierScorer(
    alpha=frontier_alpha,
    beta=frontier_beta,
    theta=frontier_theta,
)
```

Khi chọn node tiếp theo:

```python
def get_next_node(self):
    scored_nodes = [(node, self.scorer.compute(node))
                    for node in self.unexplored_nodes]
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    best_node, best_score = scored_nodes[0]
    breakdown = self.scorer.breakdown(best_node)
    logger.info(
        f"Selected '{best_node.url}' score={best_score:.4f} "
        f"(U={breakdown['U']:.4f}, V={breakdown['V']:.4f}, D={breakdown['D']:.4f})"
    )
    return best_node
```

### 8.4 `webexp/explore/algorithms/web_explore.py`

Config mới trong `WebExploreConfig`:

```python
frontier_alpha: float = 1.0
frontier_beta: float = 1.0
frontier_theta: float = 1.0
```

Gọi cập nhật outcome sau mỗi trajectory hoặc feasibility check:

```python
node.record_trajectory_outcome(success)
```

Nếu dùng lookahead, cần có bước tạo candidate khi node được thêm vào frontier hoặc trước khi scoring:

```python
node.lookahead_candidates = lookahead_predictor.propose(axtree_snippet)
```

### 8.5 `configs/go_browse_config.yaml`

```yaml
# Frontier scoring: S = α·U + β·V + θ·D
# α: weight cho Lookahead Uncertainty
# β: weight cho Value / success-rate exploitation
# θ: weight cho UI/DOM Diversity
frontier_alpha: 1.0
frontier_beta: 1.0
frontier_theta: 1.0
```

---

## 9. Embedding cho Diversity

### 9.1 Module `webexp/explore/core/embedding.py`

Có thể dùng:

```python
from webexp.explore.core.embedding import (
    compute_screenshot_embedding,
    compute_dom_embedding,
)
```

### 9.2 `compute_screenshot_embedding(screenshot)`

Cách hoạt động:

1. Dùng GPT-4o-mini mô tả screenshot thành text caption.
2. Dùng `text-embedding-3-small` để embed caption thành vector.

### 9.3 `compute_dom_embedding(dom_text)`

Cách hoạt động:

1. Embed DOM/accessibility tree text trực tiếp bằng embedding model.

### 9.4 Tích hợp trong post-step callback

```python
if "screenshot" in obs:
    emb = compute_screenshot_embedding(obs["screenshot"])
    if emb:
        if node.embedding is None:
            node.embedding = []
        node.embedding.append(emb)
```

### 9.5 Khi nào `D` có hiệu lực

- Cần ít nhất 2 embedding vector.
- Nếu không có embedding hoặc chỉ có 1 embedding: `D = 0.5`.
- Có thể tắt ảnh hưởng của diversity bằng:

```yaml
frontier_theta: 0.0
```

---

## 10. Cách điều chỉnh hành vi

### Balanced

```yaml
frontier_alpha: 1.0
frontier_beta: 1.0
frontier_theta: 1.0
```

Dùng khi chưa biết nên ưu tiên thành phần nào.

### Ưu tiên lookahead exploration

```yaml
frontier_alpha: 2.0
frontier_beta: 0.5
frontier_theta: 0.5
```

Dùng khi muốn ưu tiên node có action candidate phân hóa mạnh, tức là page có vẻ còn nhiều điều chưa chắc chắn.

### Ưu tiên exploitation

```yaml
frontier_alpha: 0.5
frontier_beta: 2.0
frontier_theta: 0.5
```

Dùng khi muốn thu thêm trajectory từ node đã từng tạo kết quả tốt.

### Ưu tiên UI diversity

```yaml
frontier_alpha: 0.5
frontier_beta: 0.5
frontier_theta: 2.0
```

Dùng cho website có nhiều form, dropdown, modal, tab hoặc dashboard nhiều tương tác.

### Gần giống FIFO / baseline

```yaml
frontier_alpha: 0.0
frontier_beta: 0.0
frontier_theta: 0.0
```

Khi tất cả score bằng 0, thứ tự chọn phụ thuộc vào thứ tự node trong `unexplored_nodes` và cách sort ổn định của Python. Trường hợp này có thể dùng để debug hoặc so sánh với baseline.

---

## 11. Bảng chọn hyperparameter

### 11.1 Theo mục tiêu

| Mục tiêu | α | β | θ | Khi nào dùng |
|---|:-:|:-:|:-:|---|
| Balanced | 1.0 | 1.0 | 1.0 | Chưa chắc nên chọn gì |
| FIFO-like | 0.0 | 0.0 | 0.0 | Debug hoặc baseline |
| Lookahead exploration | 2.0 | 0.5 | 0.5 | Muốn ưu tiên node có confidence phân hóa mạnh |
| Exploitation | 0.5 | 2.0 | 0.5 | Muốn thu thêm trajectory từ node có success rate cao |
| UI diversity focus | 0.5 | 0.5 | 2.0 | Website nhiều tương tác ẩn |

### 11.2 Theo loại website

| Loại website | α | β | θ | Lý do |
|---|:-:|:-:|:-:|---|
| E-commerce | 1.5 | 1.0 | 1.0 | Nhiều category/product page, cần khám phá rộng |
| Admin dashboard | 0.5 | 1.0 | 2.0 | Ít URL hơn nhưng mỗi trang nhiều control |
| Wiki / tài liệu | 2.0 | 0.5 | 0.0 | Nội dung tĩnh, diversity UI ít quan trọng |
| Social media/forum | 1.0 | 1.0 | 1.5 | Nhiều trạng thái động và tương tác |
| Form-heavy website | 0.5 | 1.0 | 2.0 | Form/dropdown/modal quan trọng |

---

## 12. Cách đọc log để điều chỉnh

Log mẫu:

```text
Selected 'https://...' score=1.5234 (U=0.8712, V=0.4311, D=0.5000)
```

| Log pattern | Ý nghĩa | Cách xử lý |
|---|---|---|
| `U` luôn = 1.0 | Nhiều node không có `lookahead_candidates` | Kiểm tra nơi gọi `LookaheadPredictor.propose()` |
| `U` luôn gần 0.0 | Lookahead confidence quá giống nhau hoặc fallback toàn 0.5 | Kiểm tra prompt/API/parse output |
| `V` cao ở node mới | Do prior `sr=0.5` và `1/log(2)` | Đây là hành vi hiện tại của code; giảm `β` nếu không muốn prior value ảnh hưởng mạnh |
| `V` luôn ≈ 0.0 | Hầu như không có trajectory thành công | Giảm `β`, tăng solver retries, kiểm tra reward model |
| `D` luôn = 0.5 | Chưa có đủ embedding | Kiểm tra tích hợp `compute_screenshot_embedding` hoặc `compute_dom_embedding` |
| Một node được chọn lặp lại nhiều | `β` hoặc `θ` quá cao, penalty chưa đủ mạnh | Giảm `β`, tăng exploration pressure bằng `α`, hoặc thêm novelty bonus nếu cần |
| Score các node quá gần nhau | Các prior/default giống nhau | Bảo đảm lookahead chạy cho node mới hoặc thêm discovery score |

---

## 13. Lưu ý kỹ thuật

- Code không phụ thuộc numpy; các phép tính dùng `math` và pure Python.
- `U` hiện tại dùng `node.lookahead_candidates`, không dùng success rate giữa các task thật.
- `V` dùng success rate thật nếu node có trajectory; node mới kế thừa success rate từ parent/ancestor gần nhất có dữ liệu; nếu không có thì dùng prior `0.5`.
- `D` cần ít nhất 2 embedding vector, nếu không trả về `0.5`.
- `math.log` trong code là log tự nhiên, nên `1 / log(2) > 1`.
- `sigma` trong `uncertainty()` hiện đang là variance, dù tên biến là `sigma`.
- `FrontierScorer.to_dict()` và `from_dict()` dùng để serialize/restore `alpha`, `beta`, `theta`, `epsilon`.
- Nếu load graph cũ không có field mới, cần đảm bảo `Node` có default cho `lookahead_candidates`, `exploration_count`, `success_rate`, `total_trajs`, `successful_trajs`, `embedding`, `parent_url`.

---

## 14. Tóm tắt ngắn

```text
U = độ bất định từ lookahead confidence, dùng được cả trước khi explore thật nếu đã có axtree snippet.
V = giá trị khai thác từ success rate trajectory; node mới kế thừa SR từ parent/ancestor gần nhất, rồi mới fallback prior 0.5.
D = độ đa dạng UI/DOM từ embedding state liên tiếp, node mới dùng default 0.5.
S = αU + βV + θD, node có S cao nhất được chọn từ frontier.
```
