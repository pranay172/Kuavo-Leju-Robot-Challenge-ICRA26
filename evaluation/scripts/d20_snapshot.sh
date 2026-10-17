#!/usr/bin/env bash
# Second DEV half (49-58) for one Task 3 snapshot, under the shared simulator lock.
s=$1; ROOTD=${LEJU_ROOT:?set LEJU_ROOT to the workspace root}; cd $ROOTD; DIR=$ROOTD/orchestration/task3; RUN=$(cat $DIR/RUN_NAME)
printf -v SD 'step%06d' "$s"
busy(){ pgrep -f '^python3 (.*/)?recovery/tools/eval_seeds[.]py' >/dev/null; }
ck=$(ls -d recovery/experiments/cloud_snapshots/$RUN/${SD}_*/ema | tail -1)
base=$(ls -td recovery/experiments/eval/cloud_${RUN}_${SD}_* | head -1)
exec 9>$ROOTD/orchestration/sim.lock; flock 9; while busy; do sleep 30; done
lbl=dev2_${RUN}_${SD}_$(date -u +%Y%m%d_%H%M%S)
python3 recovery/tools/eval_seeds.py --checkpoint "$ck" --seeds 49-58 --label $lbl --output recovery/experiments/eval/$lbl \
  --group-size 10 --task 3 --n-action-steps 10 --time-budget-minutes 45 --episode-wall-timeout 200 > $DIR/d20_$SD.log 2>&1
r=$(python3 - $base/summary.json recovery/experiments/eval/$lbl/summary.json <<'PY'
import json,sys
e=[x for f in sys.argv[1:] for x in json.load(open(f))['episodes']]; sc=[x['score'] for x in e if x['score'] is not None]
print(f"meanAll {sum(sc)/max(len(sc),1):.1f} n{len(sc)} full {sum(x>=100 for x in sc)} safetyExits {sum(bool(x.get('controller_diagnostics')) for x in e)}")
PY
)
echo "SNAP d20_$s $r at $(date +%H:%M)" >> $DIR/screen_results.txt
