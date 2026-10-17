#!/usr/bin/env bash
# Lay out a training machine's /workspace the way training/launchers/*.sh expect it.
#
#   training/stage_workspace.sh            (run from a checkout of this repo on the GPU machine)
#
# Result:
#   /workspace/kuavo_data_challenge/          public repo at the pinned commit, patched
#   /workspace/recovery/tools/                trainer, resume-state helpers, frame-cache builder
#   /workspace/tools/act_train_supervisor.py
#   /workspace/recovery/experiments/          ACT config and augmentation config
#   /workspace/recovery/datasets/             (you add) task<N>_source plus the data_prep outputs
#   /workspace/setup/                         venv setup script and the pip freeze it installs
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
W=/workspace
mkdir -p "$W/recovery/tools" "$W/tools" "$W/recovery/experiments/act_task1_reference" \
    "$W/recovery/datasets" "$W/setup/tmp" "$W/runs" "$W/frame_cache"
LEJU_ROOT="$W" SKIP_SIMULATOR=1 "$REPO/upstream/setup_upstream.sh"
install -m 0644 "$HERE"/train_act_v2.py "$HERE"/train_act_experiment.py "$HERE"/act_resume_state.py \
    "$HERE"/build_task1_jpeg_cache.py "$W/recovery/tools/"
install -m 0644 "$HERE"/act_train_supervisor.py "$W/tools/"
install -m 0644 "$HERE"/configs/act_policy_config.json "$W/recovery/experiments/act_task1_reference/config.json"
install -m 0644 "$HERE"/configs/augmentation.json "$W/recovery/experiments/act_task1_full900_augmentation.json"
install -m 0755 "$HERE"/launchers/setup_venv.sh "$W/setup/"
install -m 0644 "$HERE"/launchers/venv_pip_freeze_20261005.txt "$W/setup/"
install -m 0755 "$HERE"/launchers/launch_task*_act_wsd_*.sh "$W/setup/"
echo "Staged $W. Next: /workspace/setup/setup_venv.sh, then the dataset steps in docs/REPRODUCE.md."
