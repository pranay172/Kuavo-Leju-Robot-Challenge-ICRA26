#!/usr/bin/env bash
# Record camera videos of a model on a few consecutive seeds, for demos.
#
#   LEJU_ROOT=/path/to/workspace evaluation/record_demo.sh <task 1|2|3> <checkpoint dir> <first seed> [episodes]
#
# Every step is recorded (10 fps) from the three policy cameras. Videos land in
# $LEJU_ROOT/recovery/experiments/eval/demo_task<N>_<stamp>/videos/episode_*_seed*/, next to the
# scores in summary.json. The checkpoint's config.json must already hold the re-plan interval
# (the final model configs do). Turn the best episode into a GIF with evaluation/make_demo_gif.py.
#
# FRONT_VIEW=1 also saves the simulator's MuJoCo viewer, which follows the robot from the front,
# to <run>/front_view/ (about 10 fps, timestamped JPEGs; make_demo_gif.py --front). The viewer stays
# hidden: docker/viewer_dump.cpp, preloaded instead of the headless shim, copies its frames.
set -euo pipefail
: "${LEJU_ROOT:?set LEJU_ROOT to the workspace root}"
TASK=${1:?task 1, 2 or 3}; CKPT=$(realpath "${2:?checkpoint directory}"); SEED=${3:?first seed}
EPISODES=${4:-3}
[[ "$CKPT" == "$LEJU_ROOT/recovery/"* ]] || { echo "Checkpoint must be under $LEJU_ROOT/recovery" >&2; exit 2; }
STEPS=300; [[ $TASK == 3 ]] && STEPS=600
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LEJU_ROOT"
OUT="recovery/experiments/eval/demo_task${TASK}_$(date -u +%Y%m%d_%H%M%S)"
SIM_ENV=(--sim-env LD_PRELOAD=/recovery/environment/libheadless_render_fix.so)
if [[ "${FRONT_VIEW:-0}" == 1 ]]; then
    if [[ ! -f recovery/environment/libviewer_dump.so ]]; then
        install -m 0644 "$HERE/../docker/viewer_dump.cpp" recovery/environment/
        docker exec "${SIM_CONTAINER:-kuavo-icra-sim}" bash -lc 'g++ -shared -fPIC -O2 \
            -o /recovery/environment/libviewer_dump.so /recovery/environment/viewer_dump.cpp \
            $(pkg-config --cflags --libs opencv4) -ldl -lGL'
    fi
    SIM_ENV=(--sim-env LD_PRELOAD=/recovery/environment/libviewer_dump.so
             --sim-env "MUJOCO_VIEWER_DUMP=/$OUT/front_view"
             --sim-env MUJOCO_VIEWER_DUMP_FPS=10 --sim-env MUJOCO_VIEWER_DUMP_CROP=0.317,0.683)
fi
python3 recovery/tools/run_sim_evaluation.py prepare --checkpoint "$CKPT" --label "$(basename "$OUT")" \
    --output "$OUT" --task "$TASK" --episodes "$EPISODES" --seed-start "$SEED" \
    --max-episode-steps "$STEPS" --record-episodes "$(seq -s, 0 $((EPISODES - 1)))" --video-stride 1 \
    --disable-rosbag "${SIM_ENV[@]}" --sim-ready-timeout 120
[[ "${FRONT_VIEW:-0}" == 1 ]] && mkdir "$OUT/front_view"
python3 recovery/tools/run_sim_evaluation.py run --manifest "$OUT/manifest.json" \
    --policy-container "${POLICY_CONTAINER:-kuavo-icra-policy}" --sim-container "${SIM_CONTAINER:-kuavo-icra-sim}"
python3 recovery/tools/run_sim_evaluation.py summarize "$OUT" | python3 -c '
import json, sys
s = json.load(sys.stdin)
print(json.dumps({k: s.get(k) for k in ("label", "mean_score", "episodes_with_scores")}))'
echo "Videos: $LEJU_ROOT/$OUT/videos"
