#!/usr/bin/env bash
# Task 3 (ConveyorBeltSorting): same recipe as Task 1 run1 / Task 2 (fresh ACT, WSD, EMA), no frame cache (AV1 decode, 32 workers). 60k updates or eight hours.
# Supervisor retries up to three failures only from committed full states.
set -euo pipefail
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export XDG_CACHE_HOME=/workspace/.cache
export TMPDIR=/workspace/setup/tmp
TRAINER=/workspace/recovery/tools/train_act_v2.py

[[ -f "$TRAINER" ]] || { echo "Trainer is missing" >&2; exit 2; }
RUN="/workspace/runs/task3_act_fresh_wsd_60000_$(date -u +%Y%m%d_%H%M%S)"
mkdir "$RUN"
# Fresh output RUN/train is created by the trainer. Stop gracefully with:
# touch "$RUN/train/STOP"
# ImageNet ResNet18 + fresh ACT/VAE: deliberately omit --init-checkpoint.
# Gradient clip=10 is applied by the trainer.
nohup /workspace/venv/bin/python -u /workspace/tools/act_train_supervisor.py \
 --log "$RUN/train.log" --pid-file "$RUN/processes.json" --max-restarts 3 -- \
 /workspace/venv/bin/python -u "$TRAINER" \
 --source /workspace/kuavo_data_challenge \
 --manifest /workspace/recovery/datasets/task3_experiment_manifest.json \
 --policy-config /workspace/recovery/experiments/act_task1_reference/config.json \
 --augmentation-config /workspace/recovery/experiments/act_task1_full900_augmentation.json \
 --variant repaired --mode ab --seed 2025 --device cuda \
 --worker-transport tensor \
 --cudnn-benchmark --channels-last --amp --autocast-dtype float16 \
 --workers 32 --microbatch 16 --accumulation 2 --prefetch 2 --loader-timeout 120 --log-every 25 \
 --output "$RUN/train" --max-steps 60000 \
 --lr-schedule constant --warmup-steps 500 \
 --lr 3e-5 --lr-backbone 1e-5 --weight-decay 1e-4 --ema-decay 0.9995 \
 --resume-every 2500 --snapshot-every 5000 \
 --validation-every 1000 --validation-frames-per-episode 5 \
 --max-wall-seconds 28800 > "$RUN/supervisor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN/supervisor.pid"
printf 'Parent run: %s\nTraining run_dir (STOP, snapshots, resume states): %s/train\n' "$RUN" "$RUN"
