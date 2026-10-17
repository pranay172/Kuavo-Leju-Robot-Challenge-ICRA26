#!/usr/bin/env python3
"""Serial ACT trainer supervisor: restart failed children from committed full states.

No torch imports. Logs share one append-only file with trainer stdout/stderr.
The output directory is inferred from trainer --output. STOP always ends retries.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

stop_signal = None


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def option_value(command, name):
    for index, value in enumerate(command):
        if value == name:
            if index + 1 >= len(command):
                raise ValueError(f"Missing value after {name}")
            return command[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def without_options(command, names):
    result, index = [], 0
    while index < len(command):
        value = command[index]
        if value in names:
            index += 2
        elif any(value.startswith(name + "=") for name in names):
            index += 1
        else:
            result.append(value)
            index += 1
    return result


def committed_state(directory):
    directory = Path(directory)
    metadata = read_json(directory / "resume.json")
    if not metadata or metadata.get("status", "complete") not in ("complete", "ready", "committed"):
        return None
    state_names = []
    for key in ("training_state_file", "state_file", "payload_file"):
        if isinstance(metadata.get(key), str):
            state_names.append(metadata[key])
    state_names += ["training_state.pt", "state.pt"]
    if not any((directory / name).is_file() and (directory / name).stat().st_size > 0
               for name in state_names):
        return None
    if not any(key in metadata for key in ("optimizer_step", "step", "global_step")):
        return None
    return {"directory": str(directory.resolve()), "metadata": metadata}


def latest_good_state(output):
    output = Path(output)
    candidates = []
    for pointer in (output / "latest_resume.json", output / "resume_states/latest_resume.json"):
        descriptor = read_json(pointer)
        if not descriptor:
            continue
        for key in ("path", "state_dir", "directory", "resume_dir", "resume_path",
                    "resume_state", "checkpoint_dir", "state"):
            value = descriptor.get(key)
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates += [output / path, pointer.parent / path]
    states = output / "resume_states"
    if states.is_dir():
        candidates += sorted((path for path in states.iterdir() if path.is_dir()
                              and not path.name.startswith(".")),
                             key=lambda path: path.stat().st_mtime, reverse=True)
    seen = set()
    for path in candidates:
        identity = str(path.resolve())
        if identity in seen:
            continue
        seen.add(identity)
        state = committed_state(path)
        if state:
            return state
    return None


def saved_phase(metadata):
    phase = metadata.get("phase")
    if isinstance(phase, dict):
        return phase.get("kind")
    return phase or metadata.get("phase_kind")


def saved_elapsed(metadata):
    for key in ("phase_training_seconds", "phase_elapsed_seconds", "cumulative_wall_seconds", "cumulative_elapsed_seconds",
                "elapsed_training_seconds", "training_wall_seconds", "total_training_seconds", "elapsed_seconds"):
        try:
            if key in metadata:
                return float(metadata[key])
        except (TypeError, ValueError):
            pass
    return None


def kill_group(child, sig):
    try:
        os.killpg(child.pid, sig)
    except ProcessLookupError:
        pass


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--log", type=Path, required=True)
    result.add_argument("--pid-file", type=Path, required=True,
                        help="Atomic JSON containing supervisor_pid, child_pid and current attempt")
    result.add_argument("--max-restarts", type=int, default=3)
    result.add_argument("--poll-seconds", type=float, default=0.5)
    result.add_argument("--restart-delay-seconds", type=float, default=2)
    result.add_argument("--grace-seconds", type=float, default=150)
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def main():
    args = parser().parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Pass trainer argv after --")
    if args.max_restarts < 0 or min(args.poll_seconds, args.grace_seconds) <= 0:
        raise ValueError("Invalid restart or polling limits")
    output_flag = option_value(command, "--output")
    if not output_flag:
        raise ValueError("Trainer command needs --output PATH")
    output = Path(output_flag).resolve()
    stop_file = output / "STOP"
    wall_flag = option_value(command, "--max-wall-seconds")
    wall_limit = float(wall_flag) if wall_flag else None
    # Do not create a fresh trainer output merely to hold supervisor logs/PIDs.
    if not output.exists():
        for path in (args.log.resolve(), args.pid_file.resolve()):
            if path == output or output in path.parents:
                raise ValueError("Keep supervisor log and PID files outside the fresh trainer --output")
    prior = read_json(args.pid_file) or {}
    if prior.get("output") != str(output):
        prior = {}
    prior_pid = prior.get("supervisor_pid")
    if isinstance(prior_pid, int) and prior_pid != os.getpid():
        try:
            previous_command = Path(f"/proc/{prior_pid}/cmdline").read_bytes()
        except OSError:
            previous_command = b""
        if b"act_train_supervisor.py" in previous_command:
            raise RuntimeError("Another supervisor still owns this output/PID file")
    started_unix = float(prior.get("supervisor_started_unix", time.time()))
    candidates = []
    if wall_limit is not None:
        candidates.append(started_unix + wall_limit)
    original_deadline = option_value(command, "--deadline-unix")
    if original_deadline is not None:
        candidates.append(float(original_deadline))
    if prior.get("deadline_unix") is not None:
        candidates.append(float(prior["deadline_unix"]))
    deadline_unix = min(candidates) if candidates else None
    if deadline_unix is not None:
        command = without_options(command, ("--deadline-unix",))
        # Trainer stops 120 s before the supervisor hard-kill so its final save completes.
        command += ["--deadline-unix", str(deadline_unix - 120)]
    initial_resume = None
    if prior and not option_value(command, "--resume") and not option_value(command, "--resume-state"):
        initial_resume = latest_good_state(output)
        if initial_resume:
            names = ["--resume", "--resume-state", "--init-checkpoint"]
            if saved_phase(initial_resume["metadata"]) == "decay":
                names.append("--decay-steps")
            command = without_options(command, names) + ["--resume", initial_resume["directory"]]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    status = {"supervisor_pid": os.getpid(), "child_pid": None, "attempt": 0,
              "status": "starting", "output": str(output), "updated_utc": utc(),
              "supervisor_started_unix": started_unix, "deadline_unix": deadline_unix}
    atomic_json(args.pid_file, status)

    def receive(sig, frame):
        global stop_signal
        stop_signal = sig

    signal.signal(signal.SIGTERM, receive)
    signal.signal(signal.SIGINT, receive)
    with args.log.open("ab", buffering=0) as log:
        def event(kind, **values):
            row = {"event": kind, "time_utc": utc(), "supervisor_pid": os.getpid(),
                   "attempt": attempts, **values}
            log.write((json.dumps(row, sort_keys=True) + "\n").encode())

        def update(**values):
            status.update(values, attempt=attempts, updated_utc=utc())
            atomic_json(args.pid_file, status)

        event("supervisor_start", command=command, max_restarts=args.max_restarts,
              stop_file=str(stop_file), deadline_unix=deadline_unix,
              supervisor_started_unix=started_unix)
        if initial_resume:
            event("manual_resume", resume_path=initial_resume["directory"],
                  resume_phase=saved_phase(initial_resume["metadata"]))
        while True:
            if deadline_unix is not None and time.time() >= deadline_unix - 20:
                event("supervisor_stop", reason="absolute_wall_budget", deadline_unix=deadline_unix)
                update(status="budget_exhausted", child_pid=None, reason="absolute_wall_budget")
                return 0
            if stop_file.exists() or stop_signal:
                reason = "STOP" if stop_file.exists() else f"signal_{stop_signal}"
                event("supervisor_stop", reason=reason)
                update(status="stopped", child_pid=None, reason=reason)
                return 0
            try:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as error:
                event("launch_error", error=str(error))
                update(status="launch_error", child_pid=None, error=str(error))
                return 1
            update(status="running", child_pid=child.pid, command=command)
            event("trainer_start", child_pid=child.pid, command=command)
            terminating_at = None
            forced_stop = False
            absolute_budget_stop = False
            while child.poll() is None:
                absolute_budget_stop = deadline_unix is not None and time.time() >= deadline_unix - 20
                if (stop_file.exists() or stop_signal or absolute_budget_stop) and terminating_at is None:
                    terminating_at = time.monotonic()
                    event("stop_requested", child_pid=child.pid,
                          reason="STOP" if stop_file.exists() else ("absolute_wall_budget" if absolute_budget_stop else f"signal_{stop_signal}"))
                    # Signal the trainer, which can commit a final resumable state.
                    try:
                        child.send_signal(signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if absolute_budget_stop and time.time() >= deadline_unix:
                    event("forced_stop", child_pid=child.pid, reason="absolute_wall_budget")
                    kill_group(child, signal.SIGKILL)
                    forced_stop = True
                    break
                if terminating_at is not None and time.monotonic() - terminating_at >= args.grace_seconds:
                    event("forced_stop", child_pid=child.pid, grace_seconds=args.grace_seconds)
                    kill_group(child, signal.SIGKILL)
                    forced_stop = True
                    break
                time.sleep(args.poll_seconds)
            returncode = child.wait()
            # A killed trainer may leave CPU DataLoader descendants behind.
            kill_group(child, signal.SIGKILL)
            event("trainer_exit", child_pid=child.pid, returncode=returncode,
                  stop_file_present=stop_file.exists(), forced_stop=forced_stop)
            update(status="child_exited", child_pid=None, last_returncode=returncode)
            if stop_file.exists() or stop_signal or forced_stop or absolute_budget_stop:
                reason = "STOP" if stop_file.exists() else ("absolute_wall_budget" if absolute_budget_stop else "supervisor_signal")
                update(status="budget_exhausted" if absolute_budget_stop else "stopped", reason=reason)
                event("supervisor_stop", reason=status["reason"])
                return 0
            if returncode == 0:
                update(status="complete")
                event("supervisor_complete", returncode=0)
                return 0
            state = latest_good_state(output)
            if not state:
                existing_resume = option_value(command, "--resume") or option_value(command, "--resume-state")
                if existing_resume:
                    state = committed_state(existing_resume) or latest_good_state(existing_resume)
            if not state:
                update(status="failed_no_committed_resume")
                event("supervisor_failed", reason="no_committed_resume", returncode=returncode)
                return 1
            elapsed = saved_elapsed(state["metadata"])
            phase = saved_phase(state["metadata"])
            # A new decay phase has its own budget; do not charge old plateau elapsed.
            if option_value(command, "--decay-steps") is not None and phase != "decay":
                elapsed = None
            if wall_limit is not None and elapsed is not None and elapsed >= wall_limit - 20:
                update(status="budget_exhausted", resume_path=state["directory"])
                event("supervisor_stop", reason="cumulative_wall_budget",
                      resume_path=state["directory"], elapsed_seconds=elapsed, wall_limit=wall_limit)
                return 0
            if attempts >= args.max_restarts:
                update(status="failed_restart_limit", resume_path=state["directory"])
                event("supervisor_failed", reason="restart_limit", returncode=returncode,
                      resume_path=state["directory"])
                return 1
            attempts += 1
            names = ["--resume", "--resume-state", "--init-checkpoint"]
            if phase == "decay":
                names.append("--decay-steps")
            command = without_options(command, names)
            command += ["--resume", state["directory"]]
            update(status="restarting", resume_path=state["directory"], command=command)
            event("restart", returncode=returncode, resume_path=state["directory"],
                  optimizer_step=state["metadata"].get("optimizer_step", state["metadata"].get("step")),
                  phase_elapsed_seconds=elapsed, resume_phase=phase,
                  total_training_seconds=state["metadata"].get("total_training_seconds"))
            end_delay = time.monotonic() + args.restart_delay_seconds
            while time.monotonic() < end_delay and not stop_file.exists() and not stop_signal:
                time.sleep(min(args.poll_seconds, max(0, end_delay-time.monotonic())))


if __name__ == "__main__":
    sys.exit(main())
