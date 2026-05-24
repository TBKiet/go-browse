# Hướng dẫn kích hoạt và kiểm tra Frontier Scoring

## 1. Kích hoạt

### Cách 1: Dùng config mặc định (đã được bật sẵn)

Chỉ cần chạy exploration bình thường — metric đã được tích hợp:

```bash
python -m webexp.explore.algorithms.web_explore -c configs/go_browse_config.yaml
```

File `configs/go_browse_config.yaml` đã có sẵn:

```yaml
frontier_alpha: 1.0
frontier_beta: 1.0
frontier_theta: 1.0
```

### Cách 2: Tùy chỉnh trọng số

```yaml
frontier_alpha: 2.0   # Tăng Uncertainty → ưu tiên khám phá node mới lạ
frontier_beta: 0.5    # Giảm Value → bớt khai thác node cũ
frontier_theta: 0.0   # Tắt Diversity (nếu chưa có embedding)
```

### Cách 3: Tắt scoring (quay về FIFO)

```yaml
frontier_alpha: 0.0
frontier_beta: 0.0
frontier_theta: 0.0
```

---

## 2. Kiểm tra kết quả

### Cách 1: Đọc log (dễ nhất)

Khi exploration chạy, log sẽ hiển thị dòng như thế này:

```
Frontier scoring: selected 'https://...' with score=1.5234 (U=0.8712, V=0.4311, D=0.5000)
```

Giải thích:
- `U=0.8712` → Uncertainty cao, node này có nhiều task với success rate phân hóa mạnh (đáng khám phá)
- `V=0.4311` → Value trung bình, node có success rate vừa phải
- `D=0.5000` → Diversity ở mức default (chưa có embedding)

### Cách 2: Chạy test unit

```bash
python -m pytest tests/test_frontier_scoring.py -v
```

File test này cần được tạo (xem mục 3).

### Cách 3: So sánh kết quả với FIFO

Chạy 2 lần exploration với cùng config, khác nhau ở tham số scoring:

```bash
# Lần 1 — scoring ON
python -m webexp.explore.algorithms.web_explore \
  -c configs/go_browse_config.yaml

# Lần 2 — scoring OFF (FIFO)
python -m webexp.explore.algorithms.web_explore \
  -c configs/go_browse_config_fifo.yaml
```

Với `configs/go_browse_config_fifo.yaml` copy từ `go_browse_config.yaml` và sửa:

```yaml
frontier_alpha: 0.0
frontier_beta: 0.0
frontier_theta: 0.0
```

Sau đó so sánh:
- Số lượng node khám phá được
- Số lượng feasible tasks tìm được
- Độ đa dạng của các URL trong explored_nodes

---

## 4. Kiểm tra tích hợp (Graph + Scorer)

Kiểm tra `Graph.get_next_node()` thực sự chọn node đúng:

```python
"""Kiểm tra Graph.get_next_node() với FrontierScorer."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))

from webexp.explore.core.graph import Graph
from webexp.explore.core.node import Node

# Tạo graph với 2 node chưa explore
with tempfile.TemporaryDirectory() as tmpdir:
    graph = Graph(
        root_url="http://root.com",
        exp_dir=tmpdir,
        alpha=1.0, beta=2.0, theta=0.0,
    )
    # Thêm node con
    node_a = graph.add_url("http://a.com", graph.root, [])
    node_b = graph.add_url("http://b.com", graph.root, [])

    # Gán dữ liệu cho node_a: SR=0.9, n=10 → V=0.9/log(12)≈0.362
    node_a.success_rate = 0.9
    node_a.total_trajs = 10
    node_a.successful_trajs = 9
    node_a.exploration_count = 10

    # Gán dữ liệu cho node_b: SR=0.5, n=0 → V=0.5/log(2)≈0.721
    node_b.success_rate = 0.5
    node_b.total_trajs = 0

    # Với beta=2.0: S_a = 2*0.362 = 0.724, S_b = 2*0.721 = 1.442
    # → Phải chọn node_b (value cao hơn vì chưa khai thác)
    chosen = graph.get_next_node()
    print(f"Chosen: {chosen.url}")
    assert chosen.url == "http://b.com", f"Expected b.com, got {chosen.url}"
    print("✅ Graph tích hợp FrontierScorer hoạt động đúng")
```

---

## 5. Checklist kích hoạt

- [ ] **Config**: `configs/go_browse_config.yaml` đã có `frontier_alpha`, `frontier_beta`, `frontier_theta`
- [ ] **Code**: `webexp/explore/core/scoring.py` tồn tại và import được
- [ ] **Graph**: `webexp/explore/core/graph.py` import `FrontierScorer` và dùng `self.scorer`
- [ ] **Node**: `webexp/explore/core/node.py` có các trường `exploration_count`, `success_rate`, `total_trajs`, `successful_trajs`, `embedding`
- [ ] **Recording**: `webexp/explore/algorithms/web_explore.py` gọi `node.record_trajectory_outcome()` sau mỗi trajectory
- [ ] **Test**: `python test_scoring_manual.py` chạy không lỗi
