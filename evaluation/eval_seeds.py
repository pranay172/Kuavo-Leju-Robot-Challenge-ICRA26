#!/usr/bin/env python3
"""Serial local Task1 evaluation with arbitrary seeds and a compact result."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / 'recovery'
RUNNER = RECOVERY / 'tools/run_sim_evaluation.py'


def read(path):
    return json.loads(Path(path).read_text())


def seeds(value):
    result = []
    for token in value.split(','):
        token = token.strip()
        if re.fullmatch(r'\d+-\d+', token):
            first, last = map(int, token.split('-'))
            if last < first: raise argparse.ArgumentTypeError('Descending seed range')
            result.extend(range(first, last+1))
        elif token.isdigit(): result.append(int(token))
        else: raise argparse.ArgumentTypeError('Use comma-separated integers or inclusive ranges')
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError('Seeds must be nonempty and unique')
    return result


def coefficient(value):
    if value.lower() in ('none', 'off'): return None
    parsed = float(value)
    if not math.isfinite(parsed): raise argparse.ArgumentTypeError('Coefficient must be finite')
    return parsed


def command_logged(command, path, timeout):
    def stop_group(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
    with path.open('w') as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout), False
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Let the existing coordinator restore XML and stop only its workers.
            stop_group(process, signal.SIGINT)
            try: process.wait(timeout=100)
            except subprocess.TimeoutExpired:
                stop_group(process, signal.SIGTERM)
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    stop_group(process, signal.SIGKILL); process.wait()
            return process.returncode, True


def outcome(directory, seed, returncode, timed_out, index=0):
    policy_file = directory/f'policy_episode_{index:04d}.json'
    policy = read(policy_file) if policy_file.exists() else {}
    summary_file = directory/'summary.json'
    summary = read(summary_file) if summary_file.exists() else {}
    episode = next((row for row in summary.get('episodes', []) if row.get('seed') == seed), {})
    sim = episode.get('simulator') or {}
    reasons = list(episode.get('invalid_reasons', []))
    # A later cohort failure cannot invalidate an earlier completed episode.
    if returncode != 0 and not episode.get('comparable', False):
        reasons.append(f'coordinator_exit_{returncode}')
    if timed_out and not episode.get('comparable', False): reasons.append('wall_time_limit')
    if not episode.get('comparable', False) or not sim.get('score_valid', False):
        reasons.append('incomplete_or_invalid_episode_receipts')
    if policy.get('seed') != seed: reasons.append('policy_seed_missing_or_mismatched')
    if policy and policy.get('index') != index: reasons.append('policy_index_mismatched')
    log_file = directory/f'simulator_{index:04d}.log'
    log = log_file.read_text(errors='replace') if log_file.exists() else ''
    diagnostics = [line for line in log.splitlines() if any(marker in line for marker in
        ('Nan, Inf or huge value', 'The simulation is unstable',
         'Safety check failed, stopping the controller', 'controllerExit called'))]
    if not log_file.exists(): reasons.append('simulator_log_missing')
    if diagnostics: reasons.append('simulator_or_controller_failure_cause_unassigned')
    raw_score = sim.get('score')
    if not isinstance(raw_score, (int, float)) or not math.isfinite(raw_score):
        reasons.append('numeric_score_missing'); raw_score = None
    # Task 3: ~20 s after a safety stop the controller nodelet times out and the REQUIRED nodelet_manager
    # dies, ending roslaunch. The simulator's own evaluator still writes this episode's score (as in the
    # official harness), so keep that score, flagged, instead of counting an infrastructure failure.
    safety_crash, original_reasons = False, []
    score_txt = directory/'scores'/f'episode_{index:04d}_seed{seed}.txt'
    if (raw_score is None and 'Safety check failed, stopping the controller' in log
            and 'REQUIRED process [nodelet_manager' in log and score_txt.exists()):
        try:
            raw_score = int(score_txt.read_text().strip())
            detail_file = score_txt.with_suffix('.json')
            sim = {'detail': read(detail_file)} if detail_file.exists() else {}
            safety_crash, original_reasons, reasons = True, list(dict.fromkeys(reasons)), []
        except ValueError:
            pass
    valid = not reasons
    return {'seed':seed, 'score':raw_score, 'components':(sim.get('detail') or {}).get('components',{}),
        'safety_exit_crash':safety_crash, 'original_invalid_reasons':original_reasons,
        'valid':valid, 'invalid_reasons':list(dict.fromkeys(reasons)),
        'full_success':bool(valid and raw_score >= 97), 'finished':bool(valid and policy.get('success')),
        'steps':policy.get('steps'), 'actual_action_hz':policy.get('actual_action_hz'),
        'controller_diagnostics':diagnostics, 'directory':str(directory),
        'cohort_index':index, 'coordinator_returncode':returncode, 'cohort_timed_out':timed_out,
        'attempted':policy_file.exists() or log_file.exists() or bool(sim),
        'sdk_startup_failure':policy.get('steps') == 0 and
            'Failed to connect to motion control topics' in str(policy.get('failure'))}


def aggregate(rows, requested):
    valid = [r for r in rows if r['valid']]
    scores = [r['score'] for r in valid]
    finished_steps = [r['steps'] for r in valid if r.get('finished') and r.get('steps') is not None]
    recorded = [r['score'] for r in rows if r['score'] is not None]
    return {'requested_count':len(requested),'completed_count':len(rows),'valid_count':len(valid),
        'invalid_count':len(rows)-len(valid),'mean_score':statistics.mean(scores) if scores else None,
        'standard_error':statistics.stdev(scores)/math.sqrt(len(scores)) if len(scores)>1 else None,
        'full_success_count':sum(r['full_success'] for r in valid),
        'finished_count':sum(r.get('finished',False) for r in valid),
        'mean_steps_to_finish':statistics.mean(finished_steps) if finished_steps else None,
        'mean_all_recorded_scores':statistics.mean(recorded) if recorded else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seeds',type=seeds,required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--task',type=int,default=1,choices=[1,2,3])
    p.add_argument('--rate-limit-file',type=Path,default=None,help='JSON with max_step: 16 per-step limits (null = unlimited)')
    p.add_argument('--rate-limit-scale',type=float,default=1.0)
    p.add_argument('--ensemble-coeff',type=coefficient,default=argparse.SUPPRESS)
    p.add_argument('--n-action-steps',type=int,default=argparse.SUPPRESS)
    p.add_argument('--record',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--policy-container',default=os.environ.get('POLICY_CONTAINER','kuavo-icra-policy'))
    p.add_argument('--sim-container',default=os.environ.get('SIM_CONTAINER','kuavo-icra-sim'))
    p.add_argument('--time-budget-minutes',type=float,default=55)
    p.add_argument('--deadline-unix',type=float)
    p.add_argument('--episode-wall-timeout',type=float,default=None)
    p.add_argument('--cleanup-reserve-seconds',type=float,default=125,
                   help='Reserve inside the evaluation deadline; use a smaller value only with an outer stage cleanup buffer')
    p.add_argument('--group-size',type=int,default=1,
                   help='Reuse a policy worker across this many serial episodes; policy.reset runs for each seed')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.label): p.error('Label must be filesystem-safe')
    if not 0 < args.time_budget_minutes < 60: p.error('Evaluation budget must be below60 minutes')
    if not 1 <= args.group_size <= 100: p.error('Group size must be between1 and100')
    if args.episode_wall_timeout is None: args.episode_wall_timeout = 200.0 if args.task==3 else 120.0
    if args.episode_wall_timeout <= 0: p.error('Episode wall timeout must be positive')
    if args.cleanup_reserve_seconds < 0: p.error('Cleanup reserve must be nonnegative')
    checkpoint = args.checkpoint.resolve(); config = read(checkpoint/'config.json'); overrides = {}
    if hasattr(args,'ensemble_coeff'): overrides['temporal_ensemble_coeff'] = args.ensemble_coeff
    if hasattr(args,'n_action_steps'):
        overrides['n_action_steps'] = args.n_action_steps
        if args.n_action_steps>1 and not hasattr(args,'ensemble_coeff'):
            overrides['temporal_ensemble_coeff'] = None
    effective = {k:overrides.get(k,config[k]) for k in ('temporal_ensemble_coeff','n_action_steps')}
    rate_limit = None
    if args.rate_limit_file is not None:
        rate_limit = [None if v is None else v*args.rate_limit_scale for v in read(args.rate_limit_file)['max_step']]
        effective['action_rate_limit'] = rate_limit
    if not 1 <= effective['n_action_steps'] <= config['chunk_size']: p.error('Action count exceeds chunk')
    if effective['temporal_ensemble_coeff'] is not None and effective['n_action_steps'] != 1:
        p.error('Averaging requires one action step; use --ensemble-coeff none for a queue')
    root = (args.output or RECOVERY/'experiments/eval'/args.label).resolve()
    root.mkdir(parents=True,exist_ok=False)
    started = time.time(); deadline = started+args.time_budget_minutes*60
    if args.deadline_unix: deadline = min(deadline,args.deadline_unix)
    state = {'label':args.label,'checkpoint':str(checkpoint),'seeds':args.seeds,'effective_policy':effective,
             'started_at_unix':started,'deadline_unix':deadline,'status':'running','episodes':[],
             'group_size':args.group_size,
             'policy_reuse_note':'Each cohort loads one policy worker. Existing policy.reset and set_seed execute separately for every seed; simulator launches and scoring remain per episode.',
             'validity_note':'Task failures remain scored. Missing/inconsistent receipts and simulator/controller faults are flagged; original recorded scores are retained separately.'}
    def save():
        state.update(aggregate(state['episodes'],args.seeds),elapsed_seconds=time.time()-started)
        target = root/'summary.json'; temporary = root/'summary.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n'); temporary.replace(target)
    def run_cohort(cohort, tag):
        directory = root/tag
        prepare = [sys.executable,str(RUNNER),'prepare','--checkpoint',str(checkpoint),
            '--label',args.label+'_'+tag,'--output',str(directory),
            '--task',str(args.task),'--episodes',str(len(cohort)),'--seed-start',str(cohort[0]),
            '--max-episode-steps','600' if args.task==3 else '300',
            '--record-episodes',','.join(map(str,range(len(cohort)))) if args.record else '',
            '--disable-rosbag','--sim-env','LD_PRELOAD=/recovery/environment/libheadless_render_fix.so',
            '--sim-ready-timeout','120']
        rc,_ = command_logged(prepare,root/(tag+'_prepare.log'),30)
        if rc != 0:
            return [{'seed':seed,'score':None,'components':{},'valid':False,
                'full_success':False,'finished':False,'sdk_startup_failure':False,
                'attempted':True,'invalid_reasons':['prepare_failed'],'directory':str(directory)}
                for seed in cohort]
        manifest_path = directory/'manifest.json'; manifest = read(manifest_path)
        # The runner accepts an explicit list; prepare's contiguous range is a default.
        manifest.update(policy_overrides=overrides,seeds=cohort)
        if rate_limit is not None: manifest['action_rate_limit'] = rate_limit
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        run = [sys.executable,str(RUNNER),'run','--manifest',str(manifest_path),
               '--policy-container',args.policy_container,'--sim-container',args.sim_container]
        timeout = min(args.episode_wall_timeout*len(cohort),
                      max(1,deadline-time.time()-args.cleanup_reserve_seconds))
        rc,timed_out = command_logged(run,root/(tag+'_run.log'),timeout)
        return [outcome(directory,seed,rc,timed_out,index) for index,seed in enumerate(cohort)]

    save()
    pending = list(args.seeds)
    cohort_number = 0
    while pending:
        if time.time()+180 >= deadline:
            state['status'] = 'time_budget_exhausted'; break
        cohort = pending[:args.group_size]; del pending[:len(cohort)]
        tag = f'cohort{cohort_number:03d}_seeds{cohort[0]}_{cohort[-1]}'
        state['active_cohort'] = {'seeds':cohort,'directory':str(root/tag)}; save()
        rows = run_cohort(cohort,tag)
        deferred = []
        for row in rows:
            seed = row['seed']
            if not row.get('attempted',True):
                # Cohort ended before this seed started: continue it once in a new
                # cohort. Completed/attempted task failures are never rerun.
                deferred.append(seed); continue
            attempts = [{'attempt':0,'valid':row['valid'],'invalid_reasons':row['invalid_reasons'],
                         'directory':row['directory'],'cohort_index':row.get('cohort_index')}]
            if row.get('sdk_startup_failure') and time.time()+180<deadline:
                retry = run_cohort([seed],tag+f'_sdkretry_seed{seed}')[0]
                attempts.append({'attempt':1,'valid':retry['valid'],
                    'invalid_reasons':retry['invalid_reasons'],'directory':retry['directory']})
                row = retry
            row['attempts'] = attempts; state['episodes'].append(row)
            state['episodes'].sort(key=lambda item: args.seeds.index(item['seed']))
            save()
            print(json.dumps({'seed':seed,'score':row['score'],'valid':row['valid'],
                              'completed':state['completed_count'],'mean':state['mean_score']}),flush=True)
        pending = deferred + pending
        cohort_number += 1
        if deferred == cohort:
            # A shared startup fault cannot trigger an endless series of cohorts.
            state['status'] = 'cohort_startup_failure'; break
    else: state['status'] = 'complete'
    state['unattempted_seeds'] = pending
    state.pop('active_cohort',None)
    state['finished_at_unix'] = time.time(); save()
    print(json.dumps({k:state[k] for k in ('status','valid_count','requested_count','mean_score',
                                         'standard_error','full_success_count','elapsed_seconds')}),flush=True)


if __name__ == '__main__': main()
