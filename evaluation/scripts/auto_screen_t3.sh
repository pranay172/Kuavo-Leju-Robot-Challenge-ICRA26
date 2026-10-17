#!/usr/bin/env bash
# Laptop-side Task 3 screener: each cloud EMA snapshot on DEV seeds 44-48,64-68 (600-step episodes), serially.
# When training ends (supervisor_complete), archive the final snapshot + main_end_resume (if present) and exit.
# The pod is NOT stopped by this script.
DIR=${LEJU_ROOT:?set LEJU_ROOT to the workspace root}/orchestration/task3
source $DIR/pod.env; RUN=$(cat $DIR/RUN_NAME)
export POD_HOST=root@$POD_IP POD_PORT SCREEN_EXTRA_ARGS_FILE=$DIR/screen_extra_args
P=/workspace/runs/$RUN
RES=$DIR/screen_results.txt
ARCH=${LEJU_ROOT:?set LEJU_ROOT to the workspace root}/recovery/cloud_snapshots/$RUN
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 root@$POD_IP -p $POD_PORT -i $HOME/.ssh/id_ed25519"
RSH="ssh -p $POD_PORT -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15"
ROOTD=${LEJU_ROOT:?set LEJU_ROOT to the workspace root}
log(){ echo "$(date +%H:%M) $*" >> "$DIR/auto_screen.log"; }
touch "$RES"
snap_steps(){ $SSH "grep -h snapshot_written $P/train.log" 2>/dev/null | python3 -c "import sys,json
for l in sys.stdin:
    try: print(json.loads(l)['snapshot_written']['optimizer_step'])
    except Exception: pass"; }
fetch(){ for a in 1 2 3 4; do rsync -a --partial --timeout=60 -e "$RSH" "root@$POD_IP:$1" "$2" && return 0; sleep 10; done; return 1; }
# Wait for the laptop to be free of other evaluations (Task 2 BENCH) before each screen.
busy(){ pgrep -f '^python3 (.*/)?recovery/tools/eval_seeds[.]py' >/dev/null; }
while true; do
  for s in $(snap_steps); do
    grep -q "^SNAP $s " "$RES" && continue
    exec 9>${LEJU_ROOT:?set LEJU_ROOT to the workspace root}/orchestration/sim.lock; flock 9; while busy; do sleep 30; done
    log "screening step $s"
    out=$(cd $ROOTD && bash recovery/tools/screen_cloud_snapshot.sh "$RUN" "$s" 2>>"$DIR/auto_screen.log" | grep '^SNAP')
    [ -n "$out" ] && echo "$out at $(date +%H:%M)" >> "$RES" || echo "SNAP $s FAILED at $(date +%H:%M)" >> "$RES"
    flock -u 9; exec 9>&-
  done
  if $SSH "grep -q supervisor_complete $P/train.log" 2>/dev/null; then
    pending=0; for s in $(snap_steps); do grep -q "^SNAP $s " "$RES" || pending=1; done
    if [ $pending = 0 ]; then
      mkdir -p "$ARCH"; last=$(snap_steps | tail -1); printf -v LD 'step%06d' "$last"
      log "archiving final snapshot $LD"; ok=1
      fetch "$P/train/$LD/" "$ARCH/$LD/" || ok=0
      $SSH "test -d $P/main_end_resume" && { fetch "$P/main_end_resume/" "$ARCH/main_end_resume/" || ok=0; }
      fetch "$P/train/summary.json" "$ARCH/"; fetch "$P/train.log" "$ARCH/"
      [ $ok = 1 ] && echo "ARCHIVE ok final=$LD at $(date +%H:%M)" >> "$RES" || echo "ARCHIVE FAILED at $(date +%H:%M)" >> "$RES"
      exit 0
    fi
  fi
  sleep 60
done
