# Goal Refinement Fixes

This document summarizes the goal-refinement changes implemented after
`docs/GOAL_REFINEMENT_ANALYSIS.md`.

## Scope

The implementation focuses on the highest-impact issues without replacing the
whole trajectory schema with a new `TaskState` object. It keeps backward
compatibility with existing trajectory JSON while adding stronger prompt memory,
goal-alignment guardrails, and task-level persistence for summarized goals.

## Implemented Changes

### 1. Dual-goal task state in solver prompts

File: `webexp/agents/prompt_builders/solver_prompt_builder.py`

The solver prompt now includes a `# Task State` section with:

- Original goal marked as immutable
- Current objective, derived from `refined_goal` when available
- Lightweight progress counters
- Recent actions
- Known facts and negative observations from previous refined goals and errors

This prevents `refined_goal` from replacing the user's original goal in the
model context.

### 2. Persistent failed-attempt memory

File: `webexp/agents/prompt_builders/solver_prompt_builder.py`

The prompt now includes a `# Failed Attempts` section when past actions produced
browser errors. The section keeps the most recent failed actions and tells the
solver not to retry them unless the page state or target element has changed.

This addresses retry loops where the agent repeatedly attempts actions that have
already failed.

### 3. Periodic re-alignment checks

File: `webexp/agents/prompt_builders/solver_prompt_builder.py`

Every 4 actions, if a refined goal exists, the prompt injects a
`# Re-alignment Check` section:

- Shows the original goal
- Shows the current objective
- Asks the model to verify progress toward the original goal
- Encourages a different approach if progress has stalled

This reduces local-optimum bias caused by over-focusing on local page obstacles.

### 4. State-based refined goal instruction

File: `webexp/agents/prompt_builders/solver_prompt_builder.py`

The system instruction for `refined_goal` now requires task-level desired state
or outcome, not browser actions.

Examples:

- Good: `Cart contains product X with quantity 1`
- Bad: `Click Add to Cart`
- Bad: `Scroll down`
- Bad: `Type in the search box`

The instruction also says confirmed unavailability should become a task-level
infeasibility objective followed by `report_infeasible("reason")`.

### 5. Runtime guardrail for action-level refined goals

File: `webexp/agents/solver_agent.py`

Added `_looks_like_action_level_goal()` and an action-verb heuristic. If the LLM
returns a procedural `refined_goal` such as `Click ...`, `Scroll ...`, or
`Type ...`, the agent rejects it and keeps the prior objective, falling back to
the original goal when needed.

This prevents action-level text from overwriting the current task objective.

### 6. Task-level summarized goal persistence

Files:

- `webexp/explore/core/task.py`
- `webexp/explore/algorithms/web_explore.py`

`Task` now has a `summarized_goal` field and `update_summarized_goal()` method.
`task_info.json` now stores both:

```json
{
  "goal": "Original proposed goal",
  "summarized_goal": "Refined semantic goal",
  "misc": {}
}
```

During trajectory summarization, WebExplore selects the best available
summarized goal and writes it back to the task:

1. Prefer summaries from successful trajectories.
2. If none exist, fall back to all available trajectory summaries.
3. Break ties by frequency, then by summary length.

This keeps `task_info.json` as the task-level source of truth without
overwriting the original goal.

### 7. More robust task directory initialization

File: `webexp/explore/core/task.py`

`Task.__post_init__()` now always ensures these paths exist:

- task directory
- `positive_trajs/`
- `negative_trajs/`
- `task_info.json` when missing

Previously, if `exp_dir` already existed but subdirectories did not, loading or
round-tripping a task could fail.

## Covered Analysis Items

These changes directly address or mitigate:

- 1. Goal Drift
- 2. Refined Goal Describes Action Instead Of Objective
- 3. Refined Goal Overwrites Context
- 4. Local-Optimum Bias
- 5. Missing Progress State
- 6. Missing Negative Observations
- 7. Refined Goal Too Procedural
- 8. Missing Termination-Aware Refinement
- 9. Environment State vs Intended Outcome Confusion
- 10. Missing Repetition Awareness
- 11. Summarized Goal Not Written Back To Task

## Compatibility

Existing trajectory files remain compatible:

- `TrajectoryStep.refined_goal` is unchanged.
- `Trajectory.misc["summarized_goal"]` is unchanged.
- `Task.load()` reads old `task_info.json` files that do not contain
  `summarized_goal`.
- New task saves include `summarized_goal`, with `null` when not available.

## Tests Added

File: `tests/test_edge_cases.py`

Added tests for:

- Action-level refined goal detection
- `Task.summarized_goal` save/load roundtrip

## Verification

Run with the conda `agent` environment Python:

```powershell
C:\Users\Admin\anaconda3\envs\agent\python.exe -m py_compile webexp\agents\prompt_builders\solver_prompt_builder.py webexp\agents\solver_agent.py webexp\explore\core\task.py webexp\explore\algorithms\web_explore.py tests\test_edge_cases.py
```

```powershell
C:\Users\Admin\anaconda3\envs\agent\python.exe tests\test_edge_cases.py
```

Observed result:

```text
37 passed, 0 failed out of 37 tests
```

Note: the `agent` environment did not have `pytest` installed during
verification, so the existing direct test runner in `tests/test_edge_cases.py`
was used.

## Remaining Work

The current implementation is intentionally incremental. It does not yet add a
first-class serialized `TaskState` dataclass to trajectory steps or callback
context. That larger change would be the next step if stronger structured state
tracking is needed across agents, callbacks, and training pipelines.
