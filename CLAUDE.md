# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project: Go-Browse

This is a **Python** research project (not Go). It implements the Go-Browse algorithm for unsupervised web agent training data collection via structured website exploration. It also includes WebArena benchmarking and SFT/LoRA training pipelines.

### Key commands

```bash
# Install (editable)
pip install -e .

# Data collection (Go-Browse algorithm)
python -m webexp.explore.algorithms.web_explore -c configs/go_browse_config.yaml

# Run a single episode
python -m webexp.agents.run_episode -c configs/agent_run_episode.yaml

# Benchmark (WebArena)
python -m webexp.benchmark.run_webarena -c configs/benchmark_webarena.yaml

# Training
python webexp/train/sft_policy.py      # Full SFT
python webexp/train/sft_lora.py        # LoRA
```

### Package structure

- `webexp/` — main package, installed via `setup.py`
- `webexp/agents/` — LM agents (Solver, PageExplorer, NavExplorer) with factory-based registration
- `webexp/explore/core/` — exploration primitives (Graph, Node, Task, Trajectory, Trace)
- `webexp/explore/algorithms/` — the Go-Browse algorithm loop
- `webexp/benchmark/` — WebArena evaluation via agentlab
- `webexp/train/` — SFT/LoRA training with HuggingFace TRL

### Conventions

- All config via OmegaConf YAML + `@dataclass` config classes
- Agents registered via `@AgentFactory.register` decorator in module-level `AGENT_FACTORY_REGISTRY`
- Data models are `@dataclass` with `.save()` / `.load()` JSON serialization
- LLM calls use the OpenAI-compatible API format (`openai.OpenAI` client) with JSON response format
- Relative imports throughout the `webexp` package
- `tenacity` for retry logic, adaptive char limit reduction on failure
