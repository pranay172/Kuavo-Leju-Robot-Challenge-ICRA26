#!/usr/bin/env bash
# Task 3 BENCH 100-129 on the 35k EMA snapshot (nas10), under the shared simulator lock.
ROOTD=${LEJU_ROOT:?set LEJU_ROOT to the workspace root}; cd $ROOTD; DIR=$ROOTD/orchestration/task3; RUN=$(cat $DIR/RUN_NAME)
busy(){ pgrep -f '^python3 (.*/)?recovery/tools/eval_seeds[.]py' >/dev/null; }
ck=$(ls -d recovery/experiments/cloud_snapshots/$RUN/step035000_*/ema | tail -1)
exec 9>$ROOTD/orchestration/sim.lock; flock 9; while busy; do sleep 30; done
lbls=""
for rng in 100-109 110-119 120-129; do
  lbl=bench2_${RUN}_step035000_s${rng}_$(date -u +%Y%m%d_%H%M%S); lbls="$lbls recovery/experiments/eval/$lbl/summary.json"
  echo "$(date +%H:%M) start BENCH-rerun 35k seeds $rng" >> $DIR/auto_screen.log
  python3 recovery/tools/eval_seeds.py --checkpoint "$ck" --seeds $rng --label $lbl --output recovery/experiments/eval/$lbl \
    --group-size 10 --task 3 --n-action-steps 10 --time-budget-minutes 45 --episode-wall-timeout 200 > $DIR/bench2_35k_$rng.log 2>&1
done
r=$(python3 - $lbls <<'PY'
import json,sys
e=[x for f in sys.argv[1:] for x in json.load(open(f))['episodes']]; sc=[x['score'] for x in e if x['score'] is not None]
print(f"meanAll {sum(sc)/max(len(sc),1):.1f} n{len(sc)} full {sum(x>=100 for x in sc)} zeros {sum(x==0 for x in sc)} safetyExits {sum(bool(x.get('controller_diagnostics')) for x in e)}")
PY
)
echo "BENCH35k_rerun $r at $(date +%H:%M)" >> $DIR/screen_results.txt
python3 $DIR/latency_report.py $(for f in $lbls; do dirname $f; done) >> $DIR/screen_results.txt
