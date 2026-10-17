#!/usr/bin/env bash
# Copy one cloud EMA snapshot to a fresh directory, then screen serial DEV seeds.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  printf '%s\n' 'Usage: screen_cloud_snapshot.sh RUN_NAME SNAP_STEP [SEEDS]' 'Default seeds: 44-48,64-68. SCREEN_DEADLINE_UNIX may set an outer stage deadline.'
  exit 0
fi
if (( $# < 2 || $# > 3 )); then
  printf '%s\n' 'Usage: screen_cloud_snapshot.sh RUN_NAME SNAP_STEP [SEEDS]' >&2
  exit 64
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
RUN_NAME=$1
SNAP_ARGUMENT=${2#step}
SEEDS=${3:-44-48,64-68}
[[ $RUN_NAME =~ ^[A-Za-z0-9_-]+$ && $SNAP_ARGUMENT =~ ^[0-9]{1,6}$ ]] || { printf '%s\n' 'Invalid run name or snapshot step' >&2; exit 64; }
SNAP_STEP=$((10#$SNAP_ARGUMENT))
(( SNAP_STEP > 0 )) || { printf '%s\n' 'Snapshot step must be positive' >&2; exit 64; }
printf -v SNAP_DIR 'step%06d' "$SNAP_STEP"
for dependency in python3 rsync ssh docker timeout; do
  command -v "$dependency" >/dev/null || { printf 'Missing dependency: %s\n' "$dependency" >&2; exit 69; }
done
SEED_COUNT=$(python3 - "$SCRIPT_DIR/eval_seeds.py" "$SEEDS" <<'PY'
import importlib.util,sys
spec=importlib.util.spec_from_file_location('eval_seeds',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
values=module.seeds(sys.argv[2])
if any(seed < 44 or seed > 68 for seed in values):raise SystemExit('Screening accepts DEV seeds44-68 only')
print(len(values))
PY
)
POLICY_CONTAINER=${SCREEN_POLICY_CONTAINER:-${POLICY_CONTAINER:-kuavo-icra-policy}}
SIM_CONTAINER=${SCREEN_SIM_CONTAINER:-${SIM_CONTAINER:-kuavo-icra-sim}}
for container in "$POLICY_CONTAINER" "$SIM_CONTAINER"; do
  [[ $(docker inspect --format '{{.State.Running}}' "$container") == true ]] || { printf 'Container is not running: %s\n' "$container" >&2; exit 69; }
done
# The caller must serialize this script with every other laptop GPU/simulator job.
STAMP=$(date -u +%Y%m%d_%H%M%S)_$$
DEST_PARENT="$ROOT/recovery/experiments/cloud_snapshots/$RUN_NAME/${SNAP_DIR}_${STAMP}"
LABEL="cloud_${RUN_NAME}_${SNAP_DIR}_${STAMP}"
EVAL_DIR="$ROOT/recovery/experiments/eval/$LABEL"
mkdir -p -- "$DEST_PARENT"
python3 "$SCRIPT_DIR/cloud_snapshot_transfer.py" --run "$RUN_NAME" --step "$SNAP_STEP" \
  --destination "$DEST_PARENT/ema" --max-seconds 900 > "$DEST_PARENT/transfer.json"
python3 - "$DEST_PARENT/ema" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
required=['model.safetensors','config.json','policy_preprocessor.json','policy_postprocessor.json']
for name in required:
 if not (p/name).is_file() or (p/name).stat().st_size==0:raise SystemExit('Missing checkpoint file: '+name)
for name in ['config.json','policy_preprocessor.json','policy_postprocessor.json']:
 json.loads((p/name).read_text())
if not list(p.glob('*normalizer*.safetensors')):raise SystemExit('Normalizer weights missing')
PY
if (( SEED_COUNT == 1 )); then BUDGET=8; else BUDGET=20; fi
EVAL_ARGS=(--checkpoint "$DEST_PARENT/ema" --seeds "$SEEDS" --label "$LABEL" \
  --output "$EVAL_DIR" --policy-container "$POLICY_CONTAINER" --sim-container "$SIM_CONTAINER" \
  --group-size "$SEED_COUNT" --time-budget-minutes "$BUDGET" \
  --episode-wall-timeout 120 --cleanup-reserve-seconds 20)
EXTRA_FILE=${SCREEN_EXTRA_ARGS_FILE:-$ROOT/orchestration/run1/screen_extra_args}
if [[ -f $EXTRA_FILE ]]; then read -r -a EXTRA <<< "$(cat "$EXTRA_FILE")"; EVAL_ARGS+=("${EXTRA[@]}"); LABEL_NOTE="$(tr -d ' -' < "$EXTRA_FILE")"; fi
if [[ -n ${SCREEN_DEADLINE_UNIX:-} ]]; then
  EVAL_ARGS+=(--deadline-unix "$SCREEN_DEADLINE_UNIX")
fi
printf 'Checkpoint: %s\nEvaluation: %s\n' "$DEST_PARENT/ema" "$EVAL_DIR" >&2
set +e
python3 "$SCRIPT_DIR/eval_seeds.py" "${EVAL_ARGS[@]}" > "$DEST_PARENT/evaluation.log" 2>&1
EVAL_EXIT=$?
set -e
python3 - "$EVAL_DIR/summary.json" "$SNAP_STEP" "$SEED_COUNT" "$EVAL_EXIT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);step,count,code=map(int,sys.argv[2:])
if not p.exists():raise SystemExit('Evaluation did not produce summary.json; logs retained')
s=json.loads(p.read_text());mean=s.get('mean_score');text='NA' if mean is None else f'{mean:.4f}'
print(f"SNAP {step} mean {text} full {s.get('full_success_count',0)}/{count} valid {s.get('valid_count',0)}")
if code or s.get('status')!='complete' or s.get('valid_count')!=count or s.get('completed_count')!=count:
 print('Incomplete/invalid evaluation; summary and logs retained: '+str(p),file=sys.stderr)
 raise SystemExit(2)
PY
