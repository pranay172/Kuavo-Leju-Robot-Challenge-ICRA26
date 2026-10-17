#!/usr/bin/env bash
# Task 2 inference-only sweep on 20 DEV seeds (44-58,64-68); baseline 35k EMA nas10 d20 = 71.0.
# Serializes with the Task 3 screener: waits until no evaluation is running before each variant.
ROOTD=${LEJU_ROOT:?set LEJU_ROOT to the workspace root}; cd $ROOTD
DIR=$ROOTD/orchestration/task2; RES=$DIR/sweep_results.txt
CK35=recovery/experiments/cloud_snapshots/task2_act_fresh_wsd_40000_20261005_060359/step035000_20261005_103407_120519/ema
SOUP=recovery/experiments/task2_soup_ema_30k_35k_40k
busy(){ pgrep -f '^python3 (.*/)?recovery/tools/eval_seeds[.]py' >/dev/null; }
waitfree(){ while true; do sleep 45; busy && continue; sleep 5; busy || return 0; done; }
summ(){ python3 - "$1" <<'PY'
import json,sys
try:
    e=json.load(open(sys.argv[1]))['episodes']; sc=[x['score'] for x in e if x['score'] is not None]
    safety=sum(bool(x.get('controller_diagnostics')) for x in e)
    print(f"meanAll {sum(sc)/max(len(sc),1):.1f} n{len(sc)} full {sum(x>=97 for x in sc)} reachedScale {sum(x>=40 for x in sc)} safetyExits {safety}")
except Exception as ex: print("summary_error",ex)
PY
}
run(){ local name=$1 ck=$2; shift 2
  exec 9>$ROOTD/orchestration/sim.lock; flock 9; while busy; do sleep 30; done
  lbl=sweep2_${name}_$(date -u +%Y%m%d_%H%M%S)
  echo "$(date +%H:%M) start $name" >> $DIR/sweep.log
  python3 recovery/tools/eval_seeds.py --checkpoint "$ck" --seeds 44-58,64-68 --label "$lbl" \
     --output recovery/experiments/eval/$lbl --group-size 10 --task 2 --time-budget-minutes 40 \
     --episode-wall-timeout 120 "$@" > $DIR/sweep_$name.log 2>&1
  echo "SWEEP $name $(summ recovery/experiments/eval/$lbl/summary.json) at $(date +%H:%M)" >> $RES
  flock -u 9; exec 9>&-; sleep 40
}
while kill -0 472419 2>/dev/null; do sleep 20; done
run nas3 $CK35 --n-action-steps 3
run nas7 $CK35 --n-action-steps 7
echo "SWEEP-B done at $(date +%H:%M)" >> $RES
