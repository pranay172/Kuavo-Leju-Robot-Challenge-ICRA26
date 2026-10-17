#!/usr/bin/env bash
# Install the evaluation harness into the runtime directory both containers mount at /recovery.
#
#   LEJU_ROOT=/path/to/workspace evaluation/install_runtime.sh
#
# The harness locates itself through this layout, so it must run from $LEJU_ROOT/recovery/tools:
#   $LEJU_ROOT/recovery/tools/         run_sim_evaluation.py, eval_seeds.py, observation_timing.py, ...
#   $LEJU_ROOT/recovery/environment/   headless_render_fix.c (compiled in the simulator container)
#   $LEJU_ROOT/recovery/experiments/   checkpoints to evaluate and evaluation outputs
# Checkpoints must live under $LEJU_ROOT/recovery, because the containers only see that directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
: "${LEJU_ROOT:?set LEJU_ROOT to the workspace root}"
RUNTIME="$LEJU_ROOT/recovery"
mkdir -p "$RUNTIME/tools" "$RUNTIME/environment" "$RUNTIME/experiments/eval" "$RUNTIME/cache/torch"
install -m 0755 "$HERE"/run_sim_evaluation.py "$HERE"/eval_seeds.py "$HERE"/aggregate_sim_evaluations.py \
    "$HERE"/screen_cloud_snapshot.sh "$HERE"/cloud_snapshot_transfer.py "$RUNTIME/tools/"
# observation_timing is always used; the other three back optional diagnostic flags.
install -m 0644 "$HERE"/observation_timing.py "$HERE"/action_selection_trace.py \
    "$HERE"/object_pose_trace.py "$HERE"/controlled_task1_evaluator.py "$RUNTIME/tools/"
install -m 0755 "$HERE"/scripts/latency_report.py "$RUNTIME/tools/"
install -m 0644 "$REPO/docker/headless_render_fix.c" "$RUNTIME/environment/"
echo "Harness installed in $RUNTIME"
