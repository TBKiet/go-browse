# Frontier Scoring — Hướng dẫn chi tiết

## Tổng quan

Frontier Scoring là cơ chế ưu tiên hóa các node (URL) trong hàng đợi khám phá (frontier) dựa trên công thức đa mục tiêu:

$$S = \alpha U + \beta V + \theta D$$

Thay vì chọn node theo FIFO như trước, hệ thống giờ đây tính điểm cho tất cả node chưa khám phá và chọn node có điểm cao nhất ở mỗi bước lặp.

## Công thức chi tiết

### 1. Uncertainty ($U$) — Sự bất định

$$U = \frac{\sigma}{\mu + \varepsilon}$$

- **Ý nghĩa**: Đo lường tiềm năng khám phá các tác vụ mới tại node.
- **$\sigma$**: Phương sai (variance) của tỷ lệ thành công (success rate) giữa các exploration task tại node.
- **$\mu$**: Giá trị trung bình của các tỷ lệ thành công đó.
- **$\varepsilon$**: Hằng số rất nhỏ (mặc định $10^{-5}$) để tránh chia cho 0.
- **Giá trị mặc định cho node mới**: `1.0` — khuyến khích khám phá node chưa từng được ghé thăm.

**Cách tính** (pure Python, không numpy):

```python
task_scores = []
for task in node.exploration_tasks.values():
    total = len(task.positive_trajs) + len(task.negative_trajs)
    if total > 0:
        task_scores.append(len(task.positive_trajs) / total)
    else:
        task_scores.append(0.5)

n = len(task_scores)
mu = sum(task_scores) / n
sigma = sum((s - mu) ** 2 for s in task_scores) / n  # variance (σ)
U = sigma / (mu + epsilon)
```

### 2. Value ($V$) — Giá trị khai thác

$$V = SR \cdot \frac{1}{\log(n + 2)}$$

- **Ý nghĩa**: Ưu tiên node có tỷ lệ thành công cao nhưng giảm dần khi đã khai thác nhiều lần.
- **$SR$ (Success Rate)**: Tỷ lệ thành công của các trajectory đã thực thi từ node này.
- **$n$**: Số lần node đã được ghé thăm/lấy mẫu (exploration count).
- **$\frac{1}{\log(n + 2)}$**: Hàm suy giảm — càng khai thác nhiều, giá trị càng giảm mượt mà. Dùng `n+2` thay vì `n+1` để đảm bảo mẫu số luôn dương ngay cả khi `n=0`.

**Cách tính**:

```python
sr = node.success_rate if node.total_trajs > 0 else 0.5
n = max(node.exploration_count, 0)
penalty = 1.0 / math.log(n + 2)
V = sr * penalty
```

### 3. Diversity ($D$) — Tính đa dạng

$$D = \frac{1}{m} \sum_{i=1}^{m} \text{cos\_distance}(\text{embedding}_i, \text{embedding}_{i+1})$$

- **Ý nghĩa**: Đo lường sự thay đổi giao diện/DOM khi tương tác tại node. Các node có biến đổi UI lớn thường chứa nhiều chức năng phong phú.
- **$\text{embedding}_i$**: Vector đại diện cho trạng thái giao diện tại bước $i$.
- **$\text{cos\_distance}$**: Khoảng cách Cosine = $1 - \frac{a \cdot b}{\|a\| \|b\|}$.
- **Giá trị mặc định khi không có embedding**: `0.5`.

**Cách tính**:

```python
def _cosine_distance(a, b, eps=1e-10):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    cos_sim = dot / (norm_a * norm_b + eps)
    return 1.0 - cos_sim

# Tính D
if node.embedding is None or len(node.embedding) < 2:
    D = 0.5
else:
    distances = []
    for i in range(len(node.embedding) - 1):
        distances.append(_cosine_distance(node.embedding[i], node.embedding[i + 1]))
    D = sum(distances) / len(distances)
```

## Files đã thay đổi

### 1. `webexp/explore/core/node.py`

**Thêm các trường dữ liệu:**

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `exploration_count` | `int` | Số lần node đã được ghé thăm/lấy mẫu (n) |
| `success_rate` | `float` | Tỷ lệ thành công của các trajectory (SR) |
| `total_trajs` | `int` | Tổng số trajectory đã chạy từ node này |
| `successful_trajs` | `int` | Số trajectory thành công |
| `embedding` | `list` | Danh sách các embedding vector cho diversity |

**Phương thức mới:**

```python
def record_trajectory_outcome(self, success: bool):
    """Ghi nhận kết quả trajectory và cập nhật success_rate + exploration_count."""
    self.total_trajs += 1
    if success:
        self.successful_trajs += 1
    self.success_rate = self.successful_trajs / max(self.total_trajs, 1)
    self.exploration_count += 1
```

**Cập nhật:** `__post_init__`, `update_save()`, `load()` — persist/restore các trường mới.

### 2. `webexp/explore/core/scoring.py` (module mới)

Module `FrontierScorer` chứa toàn bộ logic tính điểm, tách biệt hoàn toàn khỏi `Graph`.

```python
class FrontierScorer:
    def compute(node) -> float       # S = α·U + β·V + θ·D
    def breakdown(node) -> dict      # Trả về {score, U, V, D, alpha, beta, theta}
    def uncertainty(node) -> float   # U = σ / (μ + ε)
    def value(node) -> float         # V = SR · (1 / log(n+2))
    def diversity(node) -> float     # D = mean cosine distance
    def to_dict() -> dict            # Serialize params
    def from_dict(d) -> FrontierScorer  # Deserialize
```

### 3. `webexp/explore/core/graph.py`

**Thay đổi:** `Graph` giờ chỉ giữ một instance `self.scorer = FrontierScorer(...)` và gọi nó.

```python
# get_next_node() đơn giản hóa:
def get_next_node(self):
    scored_nodes = [(node, self.scorer.compute(node))
                    for node in self.unexplored_nodes]
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    best_node, best_score = scored_nodes[0]
    breakdown = self.scorer.breakdown(best_node)
    logger.info(f"Selected '{best_node.url}' score={best_score:.4f} "
                f"(U={breakdown['U']:.4f}, V={breakdown['V']:.4f}, D={breakdown['D']:.4f})")
    return best_node
```

Tất cả logic scoring (`_compute_uncertainty`, `_compute_value`, `_compute_diversity`, `_cosine_distance`) đã được chuyển sang `scoring.py`.

### 4. `webexp/explore/algorithms/web_explore.py`

**Config mới** trong `WebExploreConfig`:

```python
frontier_alpha: float = 1.0
frontier_beta: float = 1.0
frontier_theta: float = 1.0
```

**Tích hợp:** Truyền scoring params vào `Graph()` và gọi `node.record_trajectory_outcome()` tại:
- `filter_to_feasible_tasks_for_node()` — sau mỗi feasibility check
- `sample_task_solving_trajectories_for_node()` — sau mỗi trajectory (có prefix và không prefix)

### 5. `configs/go_browse_config.yaml`

```yaml
# Frontier scoring: S = α·U + β·V + θ·D
# α (alpha): weight cho Uncertainty — ưu tiên node có task score phân hóa cao
# β (beta):   weight cho Value — ưu tiên node có success rate cao, giảm dần theo số lần visit
# θ (theta):  weight cho Diversity — ưu tiên node có UI/DOM thay đổi nhiều
frontier_alpha: 1.0
frontier_beta: 1.0
frontier_theta: 1.0
```

## Cách điều chỉnh hành vi

### Tăng cường khám phá (Exploration)
```yaml
frontier_alpha: 2.0   # Tăng trọng số Uncertainty
frontier_beta: 0.5    # Giảm trọng số Value
frontier_theta: 0.5
```

### Tăng cường khai thác (Exploitation)
```yaml
frontier_alpha: 0.5
frontier_beta: 2.0    # Tăng trọng số Value
frontier_theta: 0.5
```

### Ưu tiên đa dạng UI
```yaml
frontier_alpha: 0.5
frontier_beta: 0.5
frontier_theta: 2.0   # Tăng trọng số Diversity
```

### FIFO (hành vi cũ)
```yaml
frontier_alpha: 0.0
frontier_beta: 0.0
frontier_theta: 0.0   # Tất cả node có điểm 0 → chọn node đầu tiên
```

---

## 2. Embedding cho Diversity

### 2.1 Module `webexp/explore/core/embedding.py`

```python
from webexp.explore.core.embedding import (
    compute_screenshot_embedding,
    compute_dom_embedding,
)
```

**`compute_screenshot_embedding(screenshot)`** — 2 bước:
1. GPT-4o-mini mô tả screenshot → text caption
2. `text-embedding-3-small` embed caption → vector

**`compute_dom_embedding(dom_text)`** — 1 bước: embed DOM text trực tiếp.

### 2.2 Tích hợp vào exploration

Trong callback post_step, khi có screenshot:

```python
if "screenshot" in obs:
    emb = compute_screenshot_embedding(obs["screenshot"])
    if emb:
        if node.embedding is None:
            node.embedding = []
        node.embedding.append(emb)
```

### 2.3 Chi phí

Mỗi lần gọi: ~$0.00015 (GPT-4o-mini + embedding). Với 1000 bước ~ $0.15.

### 2.4 Khi nào Diversity có hiệu lực

- Cần ≥ 2 embedding vector để tính D
- Không có embedding → D = 0.5 (neutral)
- Có thể tắt: `frontier_theta: 0.0`

---

## 3. Bảng chọn Hyperparameter

### 3.1 Chiến lược cơ bản

| Mục tiêu | α | β | θ | Khi nào dùng |
|----------|:-:|:-:|:-:|-------------|
| **Balanced** (mặc định) | 1.0 | 1.0 | 1.0 | Không chắc nên chọn gì |
| **FIFO** (tắt scoring) | 0.0 | 0.0 | 0.0 | Debug, so sánh baseline |
| **Pure exploration** | 2.0 | 0.0 | 0.0 | Website mới, cần khám phá hết |
| **Pure exploitation** | 0.0 | 2.0 | 0.0 | Chỉ muốn thu thêm trajectory từ node đã biết |
| **UI diversity focus** | 0.5 | 0.5 | 2.0 | Website nhiều tính năng ẩn (form, menu) |

### 3.2 Chiến lược theo đặc thù website

| Loại website | α | β | θ | Lý do |
|-------------|:-:|:-:|:-:|-------|
| **E-commerce** | 1.5 | 1.0 | 1.0 | Nhiều category, cần khám phá trước |
| **Admin dashboard** | 0.5 | 1.0 | 2.0 | Ít URL, mỗi trang nhiều tương tác |
| **Wiki / tài liệu** | 2.0 | 0.5 | 0.0 | Nội dung tĩnh, cần phủ rộng |
| **Social media** | 1.0 | 1.0 | 1.5 | Nội dung động, diversity quan trọng |
| **Form-heavy** | 0.5 | 1.0 | 2.0 | Form giống nhau, khai thác luồng đã biết |

### 3.3 Cách đọc log để điều chỉnh

Log mẫu:
```
Frontier scoring: selected 'https://...' with score=1.5234
  (U=0.8712, V=0.4311, D=0.5000, α=1.00, β=1.00, θ=1.00) [explored=3]
```

| Log pattern | Vấn đề | Giải pháp |
|-------------|--------|-----------|
| `U` luôn ≈ 1.0 với mọi node | Quá nhiều node mới, không có đủ exploration tasks | Giảm α hoặc tăng `max_feasible_page_explorer_tasks_per_node` |
| `V` luôn ≈ 0.0 | Không có trajectory nào thành công | Giảm β, tăng solver retries |
| `D` luôn = 0.5 | Chưa có embedding | Tích hợp `compute_screenshot_embedding` |
| Một node được chọn liên tục | α, β không cân bằng | Dùng schedule `explore_first` |
| Chọn node ngẫu nhiên | Điểm quá gần nhau | Tăng ε hoặc θ=0 |

---

## 5. Lưu ý kỹ thuật

- **Không phụ thuộc numpy**: Toàn bộ tính toán dùng `math` và pure Python.
- **Diversity cần embedding**: `node.embedding` cần được populate qua `embedding.py`. Mặc định D = 0.5.
- **Logging**: Log đầy đủ U, V, D, α, β, θ và số node đã explore.
- **Persistent**: Scoring fields lưu trong `node_info.json`, scorer params lưu trong `graph_info.json`.
- **Backward compatible**: Load graph cũ → dùng giá trị mặc định.
