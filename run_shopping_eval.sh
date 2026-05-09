#!/bin/bash
# =============================================================================
# Simple script to evaluate a model on ALL WebArena Shopping tasks
# Uses run_episode.py for each task individually (no agentlab dependency).
#
# Usage:
#   bash run_shopping_eval.sh ./configs/benchmark_qwen.yaml
#
# Shopping URL (ngrok tunnel):
#   https://setting-legibly-implicate.ngrok-free.dev/
# =============================================================================

# NOTE: Do NOT use 'set -e' — run_episode.py exits with code 1 when task fails
# (reward=0), which is normal and expected. We check the actual reward from the
# log file instead of relying on exit codes for pass/fail.

# Auto-set WA_SHOPPING if not already defined
if [ -z "$WA_SHOPPING" ]; then
    export WA_SHOPPING="https://setting-legibly-implicate.ngrok-free.dev"
    echo "WA_SHOPPING auto-set to: $WA_SHOPPING"
fi

# BrowserGym requires ALL WebArena env vars to exist, even if we only run Shopping.
# Set dummy URLs for domains we don't use.
export WA_SHOPPING_ADMIN="${WA_SHOPPING_ADMIN:-http://localhost:7780}"
export WA_REDDIT="${WA_REDDIT:-http://localhost:7781}"
export WA_GITLAB="${WA_GITLAB:-http://localhost:7782}"
export WA_WIKIPEDIA="${WA_WIKIPEDIA:-http://localhost:7783}"
export WA_MAP="${WA_MAP:-http://localhost:7784}"
export WA_HOMEPAGE="${WA_HOMEPAGE:-http://localhost:7785}"

# WebArena evaluator uses its own OpenAI client internally for llm_ua_match
# (LLM-based answer matching). Set these to point to your local vLLM server
# so the evaluator can score tasks that require open-ended answer comparison.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"

# Model used by the WebArena evaluator for fuzzy answer matching (llm_fuzzy_match).
# Must be a model available at OPENAI_BASE_URL. Defaults to gpt-4.1-mini when
# hitting OpenAI directly; set to a vLLM-served model if using local endpoint.
export WEBARENA_EVAL_MODEL="${WEBARENA_EVAL_MODEL:-gpt-4.1-mini}"

CONFIG_FILE="${1:?Usage: $0 <config.yaml>}"
MODEL_NAME=$(python3 -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG_FILE').agent_factory_args.model_id)")
EXP_DIR=$(python3 -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG_FILE').exp_dir)")
echo "Model: $MODEL_NAME"
echo "Output: $EXP_DIR"

# Get all shopping task IDs
SHOPPING_IDS=$(python3 -c "
import json, importlib.resources, webarena
all_str = importlib.resources.files(webarena).joinpath('test.raw.json').read_text()
all_configs = json.loads(all_str)
ids = sorted(set(c['task_id'] for c in all_configs if '__SHOPPING__' in json.dumps(c) and '__SHOPPING_ADMIN__' not in json.dumps(c)))
print(' '.join(str(i) for i in ids))
")

TOTAL=$(echo "$SHOPPING_IDS" | wc -w | tr -d ' ')
echo "Total shopping tasks: $TOTAL"

# Ensure output directory exists
mkdir -p "$EXP_DIR"

echo "========================================"

COUNT=0
SUCCESS=0
FAIL=0

for TID in $SHOPPING_IDS; do
    COUNT=$((COUNT + 1))
    TASK_NAME="webarena.${TID}"

    echo "[$COUNT/$TOTAL] Running task: $TASK_NAME"

    # Create a temp config for this specific task
    TMP_CONFIG="/tmp/_shopping_eval_${TID}.yaml"
    python3 -c "
from omegaconf import OmegaConf
conf = OmegaConf.load('$CONFIG_FILE')
conf.env_args = {'task_name': '${TASK_NAME}', 'max_steps': 30, 'headless': True, 'viewport': {'width': 1280, 'height': 1440}}
OmegaConf.save(conf, '$TMP_CONFIG')
"

    # Run the episode
    python3 -m webexp.agents.run_episode -c "$TMP_CONFIG" > "$EXP_DIR/task_${TID}.log" 2>&1
    EXIT_CODE=$?

    # Extract actual reward from log (primary check, more reliable than exit code)
    REWARD=$(grep 'reward:' "$EXP_DIR/task_${TID}.log" | tail -1 | sed 's/.*reward:[[:space:]]*//')

    if [ "$REWARD" = "1.0" ]; then
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ PASS (reward=1.0)"
    elif [ "$REWARD" = "0.0" ]; then
        FAIL=$((FAIL + 1))
        echo "  ❌ FAIL (reward=0.0)"
    else
        # Fallback: no reward found (crash or unexpected output), use exit code
        if [ $EXIT_CODE -eq 0 ]; then
            SUCCESS=$((SUCCESS + 1))
            echo "  ⚠️  PASS? (exit=0, no reward found — see log)"
        else
            FAIL=$((FAIL + 1))
            echo "  ❌ FAIL (exit=$EXIT_CODE, no reward found)"
        fi
    fi

    rm -f "$TMP_CONFIG"
done

echo "========================================"
echo "RESULTS for $MODEL_NAME on Shopping:"
echo "  Total:   $TOTAL"
echo "  Success: $SUCCESS"
echo "  Failed:  $FAIL"
echo "  Rate:    $(python3 -c "print(f'{$SUCCESS/$TOTAL*100:.1f}%')" 2>/dev/null || echo "N/A")"
echo "Logs saved to: $EXP_DIR/"
