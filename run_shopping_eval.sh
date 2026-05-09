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

set -e

# Auto-set WA_SHOPPING if not already defined
if [ -z "$WA_SHOPPING" ]; then
    export WA_SHOPPING="https://setting-legibly-implicate.ngrok-free.dev"
    echo "WA_SHOPPING auto-set to: $WA_SHOPPING"
fi

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
    if python3 -m webexp.agents.run_episode -c "$TMP_CONFIG" > "$EXP_DIR/task_${TID}.log" 2>&1; then
        SUCCESS=$((SUCCESS + 1))
        echo "  ✅ PASS"
    else
        FAIL=$((FAIL + 1))
        echo "  ❌ FAIL (see $EXP_DIR/task_${TID}.log)"
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
