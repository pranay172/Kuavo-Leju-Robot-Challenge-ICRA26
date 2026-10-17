#!/usr/bin/env bash
# Task 2 clean re-check: 35k EMA, nas10 vs nas5 on BENCH 100-129, interleaved per 10-seed block, latency reported.
ROOTD=${LEJU_ROOT:?set LEJU_ROOT to the workspace root}; cd $ROOTD; DIR=$ROOTD/orchestration/task2; RES=$DIR/clean_recheck_results.txt
CK35=recovery/experiments/cloud_snapshots/task2_act_fresh_wsd_40000_20261005_060359/step035000_20261005_103407_120519/ema
exec 9>$ROOTD/orchestration/sim.lock; flock 9
declare -A dirs
for rng in 100-109 110-119 120-129; do for nas in 10 5; do
  lbl=clean2_task2_step035000_nas${nas}_s${rng}_$(date -u +%Y%m%d_%H%M%S)
  echo "$(date +%H:%M) start nas$nas $rng" >> $DIR/clean_recheck.log
  python3 recovery/tools/eval_seeds.py --checkpoint $CK35 --seeds $rng --label $lbl --output recovery/experiments/eval/$lbl \
    --group-size 10 --task 2 --n-action-steps $nas --time-budget-minutes 30 --episode-wall-timeout 120 > $DIR/clean2_nas${nas}_$rng.log 2>&1
  dirs[$nas]="${dirs[$nas]} recovery/experiments/eval/$lbl"
  echo "BLOCK nas$nas $rng $(python3 $ROOTD/orchestration/task3/latency_report.py recovery/experiments/eval/$lbl | awk '{print $2,$3,$4,$5,$6}') at $(date +%H:%M)" >> $RES
done; done
for nas in 10 5; do
  r=$(python3 - ${dirs[$nas]} <<'PY'
import json,sys
e=[x for d in sys.argv[1:] for x in json.load(open(d+'/summary.json'))['episodes']]; sc=[x['score'] for x in e if x['score'] is not None]
print(f"meanAll {sum(sc)/max(len(sc),1):.1f} n{len(sc)} completed {sum(x>=90 for x in sc)} zeros {sum(x==0 for x in sc)} safetyExits {sum(bool(x.get('controller_diagnostics')) for x in e)}")
PY
)
  echo "CLEAN nas$nas $r at $(date +%H:%M)" >> $RES
done
