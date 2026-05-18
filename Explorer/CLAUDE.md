# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Explorer

This is the codebase for **Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents** (ACL 2025 Findings). It has three components: trajectory synthesis, model training, and benchmark evaluation.

## Key commands

```bash
# Trajectory synthesis (requires Xvfb + OPENAI_API_KEY)
Xvfb :99 -screen 0 1920x1280x16 &
export DISPLAY=:99
export OPENAI_API_KEY=xxxxxxxxxxxx
python -m traj_gen.main --model-dir OUTPUT_DIR --init-url https://example.com --max-steps 10

# Training (Mind2Web-Live)
cd train/
torchrun --nproc_per_node=4 train_qwen2vl.py \
  --use_flash_attention --bf16 \
  --train_dir <HF_DATASET_PATH> --train_data_dir <RAW_TRAJS_ROOT> \
  --output_dir <OUTPUT_DIR> --num_train_epochs 10 --batch_size 64 --use-google-search

# Training (Multimodal-Mind2Web)
torchrun --nproc_per_node=4 train_qwen2vl.py \
  --use_flash_attention --bf16 \
  --train_dir <HF_DATASET_PATH> --train_dir_order <EMPTY_DIR> --train_data_dir <RAW_TRAJS_ROOT> \
  --output_dir <OUTPUT_DIR> --num_train_epochs 2 --batch_size 64 \
  --model_name_or_path Qwen/Qwen2-VL-7B-Instruct --use-nogoto-gs-format --order_all_steps --learning_rate 1e-5

# Evaluation (Mind2Web-Live)
python -m evals.mind2web_live_eval.evaluate_model --index -1 \
  --planning_text_model {qwen2-vl-7b|phi-3.5v} --toml-path evals/mind2web_live_eval/configs/setting.toml \
  --use-flash-attention --ckpt-path CKPT_PATH --temp 0.01 --log-dir LOG_DIR --viewport-width 1280

# Evaluation (Multimodal-Mind2Web)
cd evals
python -m mind2web_orig_eval.eval --use-flash-attention --ckpt-path CKPT_PATH \
  --log-dir LOG_DIR --score-file DEBERTA_SCORE_FILE --split {test_domain|test_task|test_website} --model {qwen-7b|phi-3.5}

# Evaluation (In-domain)
python -u -m evals.in_domain_eval.eval --input-file in_domain_test.json --ckpt-path CKPT_PATH \
  --use-flash-attention --log-dir LOG_DIR --use-spiral

# Evaluation (MiniWoB++)
bash evals/miniwob/eval-explorer.sh
```

Each eval/training component has its own `requirements.txt`. Create a fresh conda env before each (Python 3.12.5).

## Architecture

### `traj_gen/` — Trajectory synthesis pipeline

The **Explorer** algorithm generates web interaction trajectories via a multi-agent pipeline:

1. **`main.py`** — `Explorer` class orchestrates the loop (max `max_steps` per site, up to 2 retries on errors)
2. **`task_proposal_agent.py`** — Proposes initial task + first action from the seed page
3. **`task_refiner_agent.py`** — Refines the task description and selects the next action on subsequent steps
4. **`task_summarization_flow.py`** — After the trajectory ends, summarizes the full task from action history + screenshots
5. **`trajectory_verifier.py`** — Verifies the trajectory successfully completed the summarized intent
6. **`captcha_detection_agent.py`** — Checks if the seed page is a CAPTCHA (skip if so)

All agents call GPT-4o via `llm_utils.py` (direct REST API, not OpenAI client library). Grounding uses **Set-of-Mark** (`set_of_mark.py`) — numeric tags overlayed on screenshots, matched to accessibility tree element IDs. Browser automation via Playwright (`browser_env.py` → `ScriptBrowserEnv`). Accessibility tree parsing in `processors.py` → `ImageObservationProcessor`.

### `train/` — Training

- **`train_qwen2vl.py`** — Main training script for Qwen2-VL (7B). Uses DeepSpeed ZeRO-3 config. Two modes: Mind2Web-Live (Google search enabled, 10 epochs, `--use-google-search`) and Multimodal-Mind2Web (no search, ordered steps, 2 epochs, `--use-nogoto-gs-format --order_all_steps`).
- **`train_utils.py`** — Dataset creation (`create_stepwise_dataset_qwen`) and collator (`WebTrajDataOrderedQwenCollator`). Includes system prompt templates for web agent actions.

### `evals/` — Evaluation benchmarks

Four independent benchmarks, each with its own env setup and requirements:

| Benchmark | Entry Point | Model |
|---|---|---|
| Mind2Web-Live | `mind2web_live_eval/evaluate_model.py` | Qwen2-VL / Phi-3.5V |
| Multimodal-Mind2Web | `mind2web_orig_eval/eval.py` | Qwen-7B / Phi-3.5 |
| In-domain | `in_domain_eval/eval.py` | Qwen2-VL / Phi-3.5V / GPT-4o |
| MiniWoB++ | `miniwob/main_slm_qwen.py` | Qwen2-VL / Phi-3.5V |

## Conventions

- **No shared package** — `traj_gen/`, `train/`, and each `evals/*/` subdirectory are self-contained with duplicated utility code (browser_env, processors, set_of_mark, actions). Changes to one module do NOT automatically propagate to others.
- **Browser automation** — All browser interaction uses Playwright via `ScriptBrowserEnv` (synchronous, Chrome-based). Requires `Xvfb` for headless display.
- **Element grounding** — Set-of-Mark: accessibility tree nodes are numbered, numbers are overlaid on screenshots, agents reference elements by `[idx]`.
- **GPT-4o agents** (`traj_gen/`) call the OpenAI REST API directly (`requests.post`) with retry logic, not via the OpenAI Python client.
- **Checkpoint-based models** (`evals/`, `train/`) use HuggingFace Transformers with `--use-flash-attention` and `--ckpt-path` for fine-tuned Qwen2-VL or Phi-3.5V checkpoints.
- **Config** — argparse-based (not OmegaConf), each component parses its own args in `main()`.
- **Environment** — Requires `OPENAI_API_KEY` for GPT-4o agents; `DISPLAY=:99` with Xvfb for browser rendering. Each subproject has its own conda env and `requirements.txt`.
