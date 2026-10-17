#!/usr/bin/env bash
# After sweep B: BENCH 100-129 on the 35k EMA with the better of nas5 / nas7 (by DEV d20 meanAll).
ROOTD=${LEJU_ROOT:?set LEJU_ROOT to the workspace root}; cd $ROOTD; DIR=$ROOTD/orchestration/task2; RES=$DIR/sweep_results.txt
CK35=recovery/experiments/cloud_snapshots/task2_act_fresh_wsd_40000_20261005_060359/step035000_20261005_103407_120519/ema
busy(){ pgrep -f '^python3 (.*/)?recovery/tools/eval_seeds[.]py' >/dev/null; }
until grep -q '^SWEEP-B done' $RES; do sleep 30; done
m5=$(grep '^SWEEP nas5 ' $RES | tail -1 | awk '{print $4}'); m7=$(grep '^SWEEP nas7 ' $RES | tail -1 | awk '{print $4}')
best=5; python3 -c "import sys; sys.exit(0 if float('${m7:-0}')>float('${m5:-0}') else 1)" && best=7
sleep 50
exec 9>$ROOTD/orchestration/sim.lock; flock 9; while busy; do sleep 30; done
lbl=bench_task2_step035000_nas${best}_$(date -u +%Y%m%d_%H%M%S)
echo "$(date +%H:%M) start BENCH nas$best (nas5 $m5, nas7 $m7)" >> $DIR/sweep.log
python3 recovery/tools/eval_seeds.py --checkpoint $CK35 --seeds 100-129 --label $lbl --output recovery/experiments/eval/$lbl \
  --group-size 10 --task 2 --n-action-steps $best --time-budget-minutes 50 --episode-wall-timeout 120 > $DIR/bench_nas$best.log 2>&1
r=$(python3 - recovery/experiments/eval/$lbl/summary.json <<'PY'
import json,sys
e=json.load(open(sys.argv[1]))['episodes']; sc=[x['score'] for x in e if x['score'] is not None]
print(f"meanAll {sum(sc)/max(len(sc),1):.1f} n{len(sc)} completed {sum(x>=90 for x in sc)} zeros {sum(x==0 for x in sc)} safetyExits {sum(bool(x.get('controller_diagnostics')) for x in e)}")
PY
)
echo "BENCH nas$best $r (baseline nas10 meanAll 67.8) at $(date +%H:%M)" >> $RES
flock -u 9
