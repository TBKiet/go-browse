# Go-Browse: Hướng dẫn chạy eval pretrained model trên WebArena Shopping

> **Ngày:** 09/05/2026
> **Mục tiêu:** Đánh giá 2 pretrained model (`Llama-3.1-70B-Instruct` và `Qwen3.5-9B`) trên tập test Shopping của WebArena, chạy trên vast.ai A100 80GB.

---

## 1. Kiến trúc tổng quan

### 1.1 Tổ chức codebase

```
Go-Browse/
├── webexp/
│   ├── agents/
│   │   ├── base_agent.py          # AgentFactory registry + BaseAgent protocol
│   │   ├── solver_agent.py        # SolverAgent (gọi LLM qua OpenAI API)
│   │   ├── nav_explorer_agent.py  # NavExplorerAgent (khám phá navigation)
│   │   ├── page_explorer_agent.py # PageExplorerAgent (khám phá page)
│   │   ├── run_episode.py         # Chạy 1 episode đơn lẻ
│   │   └── prompt_builders/       # Build prompt cho từng loại agent
│   ├── benchmark/
│   │   ├── run_webarena.py        # Benchmark toàn bộ WebArena (dùng agentlab)
│   │   └── run_webarena_shopping.py  # Custom: chỉ chạy shopping tasks
│   ├── explore/                   # Go-Browse exploration loop
│   │   ├── core/                  # episode, graph, node, evaluator, trajectory
│   │   └── algorithms/web_explore.py  # WebExplore algorithm
│   └── train/                     # Training pipeline
│       ├── sft_policy.py          # Full SFT
│       ├── sft_lora.py            # LoRA fine-tuning
│       └── demo_dataset.py        # Demo data pipeline
├── configs/
│   ├── benchmark_webarena.yaml    # Config benchmark agent đơn
│   ├── go_browse_config.yaml      # Config multi-agent Go-Browse exploration
│   └── agent_run_episode.yaml     # Config chạy 1 episode
├── run_shopping_eval.sh           # Script bash chạy eval shopping
└── projects/go-browse/data/       # Data generation & processing
```

### 1.2 Cách model được "cắm" vào framework

Tất cả agent (`SolverAgent`, `NavExplorerAgent`, `PageExplorerAgent`) đều dùng **OpenAI-compatible API**:

```python
# solver_agent.py
from openai import OpenAI

self.client = OpenAI(base_url=base_url, api_key=api_key)
response = self.client.chat.completions.create(
    model=self.model_id,
    messages=messages,
    temperature=self.temperature
)
```

→ `base_url` có thể trỏ tới **bất kỳ server OpenAI-compatible nào** (vLLM, sglang, OpenAI, Anthropic proxy...).

**Không có** wrapper load model HuggingFace trực tiếp — bắt buộc phải serve model qua inference server.

### 1.3 Agent Factory pattern

```python
# base_agent.py
@AgentFactory.register
class SolverAgent(BaseAgent):
    ...

# Tạo agent từ config
AgentFactory.create_agent(name="SolverAgent", model_id="...", base_url="...")
```

Config YAML → `AgentFactory.create_agent()` → Agent instance với OpenAI client.

---

## 2. Cách benchmark hoạt động

### 2.1 Entry point: `run_webarena.py`

```
CLI arg → YAML config → OmegaConf load → RunBenchmarkConfig
    → AgentLabAgentArgsWrapper → agentlab.make_study(benchmark="webarena")
    → study.run(n_jobs, n_relaunch=8)
```

- `agentlab` quản lý toàn bộ: workers, browser env, logging, retry, đánh giá
- File `run_webarena.py` chỉ làm nhiệm vụ "dịch" từ config Go-Browse sang format agentlab
- Mặc định chạy **toàn bộ 812 tasks** WebArena (không có filter domain)

### 2.2 WebArena tasks breakdown

| Domain         | Số task | Task ID    |
| -------------- | ------- | ---------- |
| **shopping**   | 192     | 21-26, ... |
| shopping_admin | 182     | 0-6, ...   |
| gitlab         | 204     | 44-45, ... |
| map            | 128     | 7-15, ...  |
| reddit         | 106     | 27-43, ... |

URL được kiểm soát qua biến môi trường:
- `WA_SHOPPING` → Shopping URL
- `WA_SHOPPING_ADMIN` → Shopping Admin URL
- `WA_REDDIT`, `WA_GITLAB`, `WA_MAP`, `WA_WIKIPEDIA`, `WA_HOMEPAGE`

---

## 3. Chuẩn bị chạy eval

### 3.1 Yêu cầu phần cứng

| Model                               | VRAM (BF16) | Trên A100 80GB? | Giải pháp                                                                         |
| ----------------------------------- | ----------- | --------------- | --------------------------------------------------------------------------------- |
| `Qwen/Qwen3.5-9B`                   | ~18 GB      | ✅ Vừa           | Dùng trực tiếp                                                                    |
| `meta-llama/Llama-3.1-70B-Instruct` | ~140 GB     | ❌ OOM           | Dùng bản AWQ INT4 (~35 GB): `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` |

### 3.2 File config cho từng model

**`configs/benchmark_qwen.yaml`:**
```yaml
agent_factory_args:
  name: SolverAgent
  model_id: "Qwen/Qwen3.5-9B"
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"
  temperature: 0.
  char_limit: 80000

exp_dir: "./results/qwen3.5-9b_shopping"
n_jobs: 1
resume_dir: null
```

**`configs/benchmark_llama.yaml`:**
```yaml
agent_factory_args:
  name: SolverAgent
  model_id: "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"
  temperature: 0.
  char_limit: 80000

exp_dir: "./results/llama3.1-70b_shopping"
n_jobs: 1
resume_dir: null
```

### 3.3 Shopping URL

Ngrok tunnel: `https://setting-legibly-implicate.ngrok-free.dev/`

Script `run_shopping_eval.sh` tự động set `WA_SHOPPING` tới URL này nếu chưa được export.

---

## 4. Các bước thực thi trên vast.ai

```bash
# === Bước 1: Clone & cài đặt ===
git clone <repo-url> && cd Go-Browse
pip install -r requirements.txt && pip install -e .
pip install vllm
playwright install chromium

# === Bước 2: Serve model 1 — Qwen3.5-9B ===
vllm serve Qwen/Qwen3.5-9B \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32000 \
    --gpu-memory-utilization 0.90

# === Bước 3: Chạy eval Qwen (mở terminal khác) ===
bash run_shopping_eval.sh configs/benchmark_qwen.yaml

# === Bước 4: Kill Qwen, serve model 2 — Llama-3.1-70B AWQ ===
# Ctrl+C terminal vLLM, rồi:
vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32000 \
    --gpu-memory-utilization 0.90 \
    --quantization awq

# === Bước 5: Chạy eval Llama ===
bash run_shopping_eval.sh configs/benchmark_llama.yaml
```

---

## 5. Script `run_shopping_eval.sh`

Script bash chạy từng task shopping một qua `run_episode.py`:

```
┌─────────────────────────────────────┐
│  1. Lấy 192 shopping task IDs       │
│     từ webarena test.raw.json       │
├─────────────────────────────────────┤
│  2. Với mỗi task ID:                │
│     - Tạo temp config YAML          │
│     - Gọi run_episode.py            │
│     - Log kết quả vào file riêng    │
│     - Đếm PASS / FAIL              │
├─────────────────────────────────────┤
│  3. In tổng kết:                    │
│     Total / Success / Failed / Rate │
└─────────────────────────────────────┘
```

---

## 6. Output và cách đánh giá

### 6.1 Cấu trúc output mỗi task

```
results/qwen3.5-9b_shopping/
├── task_21.log           # Log text toàn bộ quá trình
└── <task_dir>/           # BrowserGym tự tạo
    ├── exp_args.pkl      # Config experiment
    ├── step_0.pkl.gz     # Info step 0 (obs, action, reward)
    ├── step_N.pkl.gz     # Step cuối + reward
    ├── screenshot_*.jpg  # Ảnh chụp màn hình
    └── summary_info.json
```

### 6.2 Cơ chế đánh giá

WebArena dùng **native evaluator** cho từng task:

| Loại task                | Cách đánh giá                                | Ví dụ                                            |
| ------------------------ | -------------------------------------------- | ------------------------------------------------ |
| **Information seeking**  | So khớp câu trả lời với đáp án (fuzzy match) | "Giá iPhone 13?" → phải trả lời đúng             |
| **Site navigation**      | Kiểm tra URL cuối cùng                       | "Đi tới giỏ hàng" → URL phải là `/checkout/cart` |
| **Content modification** | Kiểm tra DOM/state thay đổi                  | "Thêm PS5 vào giỏ" → giỏ phải có item đó         |

Kết quả: **nhị phân** — `0.0` (fail) hoặc `1.0` (success).

```
Agent thực thi → send_msg_to_user("câu trả lời")
    → Episode kết thúc
    → WebArena Evaluator so sánh với ground truth
    → Score: 0 (fail) / 1 (pass)
```

### 6.3 Kết quả cuối cùng

Script in ra bảng tổng kết:

```
========================================
RESULTS for Qwen/Qwen3.5-9B on Shopping:
  Total:   192
  Success: 47
  Failed:  145
  Rate:    24.5%
Logs saved to: ./results/qwen3.5-9b_shopping/
```

**So sánh 2 model:** Chỉ cần so sánh **Success Rate** — model nào cao hơn là tốt hơn trên shopping domain.

### 6.4 Phân tích sâu (Python)

```python
from browsergym.experiments import get_exp_result

result = get_exp_result("results/qwen3.5-9b_shopping/<task_dir>/")
record = result.get_exp_record()

for key, val in record.items():
    print(f"{key}: {val}")
# Output:
#   task_id: 21
#   reward: 1.0        ← final score
#   steps: 12           ← số step đã dùng
#   truncated: False    ← có bị hết step không
```

---

## 7. Các file đã tạo/thay đổi

| File                                        | Mục đích                                 |
| ------------------------------------------- | ---------------------------------------- |
| `configs/benchmark_qwen.yaml`               | Config benchmark cho Qwen3.5-9B          |
| `configs/benchmark_llama.yaml`              | Config benchmark cho Llama-3.1-70B AWQ   |
| `run_shopping_eval.sh`                      | Script bash chạy eval 192 shopping tasks |
| `webexp/benchmark/run_webarena_shopping.py` | Script Python alternative dùng agentlab  |

---

## 8. Lưu ý quan trọng

1. **Llama-3.1-70B BF16 gốc (~140 GB) sẽ OOM trên 1x A100 80GB** → bắt buộc dùng bản AWQ INT4.
2. **Docker shopping phải accessible từ vast.ai instance** → dùng ngrok tunnel hoặc chạy Docker trực tiếp trên instance.
3. **Mỗi lần chỉ chạy 1 model** (1 vLLM server trên port 8000), chạy tuần tự.
4. **192 task × ~1-3 phút/task ≈ 4-10 giờ** cho mỗi model (tùy tốc độ model và network).
5. **Có thể resume nếu gián đoạn** — script ghi log từng task riêng, task đã chạy sẽ bị skip nếu chạy lại (tùy chỉnh).
6. **Biến môi trường `WA_SHOPPING`** được script tự động set tới ngrok URL, không cần export thủ công.
