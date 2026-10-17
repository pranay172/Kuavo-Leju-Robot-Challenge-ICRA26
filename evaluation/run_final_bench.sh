#!/usr/bin/env bash
# Score a final model on the BENCH seeds (100-129) in three 10-episode blocks.
#
#   LEJU_ROOT=/path/to/workspace evaluation/run_final_bench.sh <task 1|2|3> <checkpoint dir>
#
# The checkpoint dir holds model.safetensors, config.json and the processor files, and must be
# under $LEJU_ROOT/recovery. The re-plan interval is the one each final model was chosen with:
# Task 1 every 10 steps, Task 2 every 5, Task 3 every 10 (no temporal ensembling).
# Prints each block's mean and observation-latency check (threshold set on Task 3; see
# docs/REPRODUCE.md for Task 2), then the pooled mean.
# Container names default to those of docker/start_containers.sh (POLICY_CONTAINER, SIM_CONTAINER).
# Took about 20 minutes for Task 1/2 and about 35 minutes for Task 3.
set -euo pipefail
: "${LEJU_ROOT:?set LEJU_ROOT to the workspace root}"
TASK=${1:?task 1, 2 or 3}; CKPT=$(realpath "${2:?checkpoint directory}")
case $TASK in
    1) NAS=10; BUDGET=30; WALL=120 ;;
    2) NAS=5;  BUDGET=30; WALL=120 ;;
    3) NAS=10; BUDGET=45; WALL=200 ;;
    *) echo "task must be 1, 2 or 3" >&2; exit 2 ;;
esac
[[ "$CKPT" == "$LEJU_ROOT/recovery/"* ]] || { echo "Checkpoint must be under $LEJU_ROOT/recovery" >&2; exit 2; }
cd "$LEJU_ROOT"
STAMP=$(date -u +%Y%m%d_%H%M%S); DIRS=()
# Scored episodes: valid ones plus controller safety exits. The harness marks a safety exit
# ("Safety check failed") invalid and leaves it out of its own mean_score, but it is a policy
# outcome (a jump in the commanded arm motion trips the whole-body controller), so the README
# scores and results/evaluations.csv count it at the score it reached.
SCORED='
import json, sys
def scored(dirs):
    result = []
    for d in dirs:
        for e in json.load(open(d + "/summary.json"))["episodes"]:
            safety_exit = any("Safety check failed" in str(c) for c in e.get("controller_diagnostics", []))
            if (e.get("valid") or safety_exit) and e.get("score") is not None:
                result.append((e["score"], safety_exit))
    return result
def mean(result):
    if not result: return "no scored episodes"
    exits = sum(s for _, s in result)
    return f"{sum(x for x, _ in result) / len(result):.1f} over {len(result)} episodes ({exits} safety exits)"
'
for SEEDS in 100-109 110-119 120-129; do
    LABEL="bench_task${TASK}_nas${NAS}_s${SEEDS}_${STAMP}"
    OUT="recovery/experiments/eval/$LABEL"
    python3 recovery/tools/eval_seeds.py --checkpoint "$CKPT" --task "$TASK" --seeds "$SEEDS" \
        --n-action-steps "$NAS" --group-size 10 --time-budget-minutes "$BUDGET" \
        --episode-wall-timeout "$WALL" --label "$LABEL" --output "$OUT" \
        --policy-container "${POLICY_CONTAINER:-kuavo-icra-policy}" \
        --sim-container "${SIM_CONTAINER:-kuavo-icra-sim}" > "$OUT.log" 2>&1 || true
    DIRS+=("$OUT")
    python3 - "$OUT" <<PY
$SCORED
for d in sys.argv[1:]:
    label = json.load(open(d + '/summary.json'))['label']
    print(f"{label}: mean {mean(scored([d]))}")
PY
    python3 recovery/tools/latency_report.py "$OUT"
done
python3 - "${DIRS[@]}" <<PY
$SCORED
print(f"BENCH pooled: {mean(scored(sys.argv[1:]))}")
PY
