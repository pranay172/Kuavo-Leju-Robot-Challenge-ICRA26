#!/usr/bin/env python3
"""Prepare and run measured, matched-seed evaluations in recovery-owned containers.

The coordinator needs only Python's standard library and Docker. The policy worker
uses the existing Kuavo policy classes, processors and Gym environment. The simulator
worker uses the existing evaluator without changing its scoring rules. No container
is stopped or removed by this script. Only process groups started by its workers are
stopped, and a separate ROS master port prevents contact with another ROS session.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import hashlib
import json
import logging
import math
import os
import re
import runpy
from pathlib import Path
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
import uuid
import xmlrpc.client


STOP = False


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def processor_directory(checkpoint):
    """Support self-contained v2 snapshots and the older parent-level layout."""
    checkpoint = Path(checkpoint)
    for directory in (checkpoint, checkpoint.parent):
        if all((directory / name).is_file() for name in
               ("policy_preprocessor.json", "policy_postprocessor.json")):
            return directory
    raise FileNotFoundError(f"Policy processor configs not found in {checkpoint} or its parent")


def check_stop(directory):
    if STOP or (directory / "stop.request").exists():
        raise KeyboardInterrupt("Recovery evaluation stop requested")


def wait_file(path, directory, timeout):
    deadline = time.monotonic() + timeout
    while not path.exists():
        check_stop(directory)
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {path.name}")
        time.sleep(0.1)
    return read_json(path)


def interrupted(signum, frame):
    global STOP
    STOP = True
    raise KeyboardInterrupt(f"Received signal {signum}")


def install_signals():
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)


def guarded_call(function, timeout):
    def expired(signum, frame):
        raise TimeoutError(f"Operation exceeded {timeout} seconds")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(int(timeout))
    try:
        return function()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    # Popen(start_new_session=True) is used for every process passed here.
    for sig, timeout in [(signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 3)]:
        if process.poll() is not None:
            break
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def owned_fingerprint(pid, run_id, role, episode=None):
    """Read ownership inside the worker's private container PID namespace."""
    try:
        environment = set(Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"))
        required = {f"LEJU_RECOVERY_RUN_ID={run_id}".encode(), f"LEJU_RECOVERY_ROLE={role}".encode()}
        if episode is not None:
            required.add(f"LEJU_RECOVERY_EPISODE={episode}".encode())
        if not required <= environment:
            return None
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return stat[19]  # Linux /proc stat field22, process start time; protects against PID reuse.
    except (OSError, IndexError):
        return None


def stop_owned_descendants(run_id, role, episode=None):
    """Stop only descendants carrying this run's inherited UUID/role/episode token."""
    targets = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and int(entry.name) != os.getpid():
            pid = int(entry.name)
            fingerprint = owned_fingerprint(pid, run_id, role, episode)
            if fingerprint is not None:
                targets[pid] = fingerprint
    for sig, grace in [(signal.SIGINT, 1), (signal.SIGTERM, 1), (signal.SIGKILL, .5)]:
        active = {pid: fp for pid, fp in targets.items()
                  if owned_fingerprint(pid, run_id, role, episode) == fp}
        if not active:
            break
        for pid in active:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if all(owned_fingerprint(pid, run_id, role, episode) != fp for pid, fp in active.items()):
                break
            time.sleep(.05)


@contextlib.contextmanager
def workspace_lock(workspace):
    """Prevent two recovery jobs from writing the same simulator model files."""
    path = Path(workspace) / ".leju_recovery_sim.lock"
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
    except FileExistsError:
        descriptor = os.open(path, os.O_RDWR)
        created = False
    with os.fdopen(descriptor, "a+") as stream:
        if created:
            owner = Path(workspace).stat()
            os.fchown(stream.fileno(), owner.st_uid, owner.st_gid)
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another recovery evaluation owns simulator workspace {workspace}") from exc
        yield


def atomic_model_write(path, content, mode, run_id, uid=None, gid=None):
    temporary = path.with_name(path.name + f".leju-{run_id}.tmp")
    try:
        temporary.write_bytes(content)
        if uid is not None and gid is not None:
            os.chown(temporary, uid, gid)
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def restore_owned_models(directory):
    ownership_path = directory / "model_xml_ownership.json"
    if not ownership_path.exists():
        return
    ownership = read_json(ownership_path)
    conflicts = []
    for record in ownership["models"]:
        path = Path(record["path"])
        current = sha256(path) if path.exists() else None
        if current == record["original_sha256"]:
            continue
        if current != record["expected_sha256"]:
            conflicts.append({"path": str(path), "reason": "External modification/deletion preserved"})
            continue
        backup = directory / record["backup"]
        if sha256(backup) != record["original_sha256"]:
            conflicts.append({"path": str(path), "reason": "Original XML backup hash mismatch"})
            continue
        atomic_model_write(path, backup.read_bytes(), record["mode"], ownership["run_id"],
                           record.get("uid"), record.get("gid"))
    if conflicts:
        write_json(directory / "model_restore_conflict.json", {"conflicts": conflicts})


def restore_models_after_worker_exit(directory):
    ownership_path = directory / "model_xml_ownership.json"
    if not ownership_path.exists():
        return
    ownership = read_json(ownership_path)
    workspace = ownership.get("sim_workspace")
    if not workspace:
        raise RuntimeError("Cannot restore simulator models without the original workspace ownership lock")
    with workspace_lock(workspace):
        restore_owned_models(directory)


def simulator_score_errors(simulator):
    if not simulator:
        return ["missing simulator receipt"]
    errors = []
    if simulator.get("evaluator_returncode") != 0:
        errors.append("evaluator did not exit successfully")
    if simulator.get("launch_returncode_before_cleanup") is not None:
        errors.append("simulator exited before owned cleanup")
    score, detail = simulator.get("score"), simulator.get("detail")
    if type(score) is not int or not 0 <= score <= 100:
        errors.append("missing/invalid numeric score")
    if not isinstance(detail, dict) or detail.get("total") != score or not isinstance(detail.get("components"), dict):
        errors.append("missing or inconsistent score detail")
    else:
        values = list(detail["components"].values())
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            errors.append("invalid score component")
        elif type(score) is int and int(sum(values) + 1e-6) != score:
            errors.append("score components do not sum to total")
    if simulator.get("score_read_failure"):
        errors.append("score receipt could not be read")
    return errors


def evaluator_bootstrap(args):
    # The original Task2 yaw sampler uses module-global random, not self.rng.
    # Catkin setup prepends other packages with a top-level `utils` module.
    # The evaluator itself prepends examples/, one level short of its package.
    package = str(Path(args.script).resolve().parents[2])
    sys.path = [package] + [path for path in sys.path if path != package]
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    sys.argv = [args.script, "--seed", str(args.seed)]
    if getattr(args, "task1_control", None):
        from controlled_task1_evaluator import run
        run(args.script, args.seed, read_json(args.task1_control), args.control_receipt)
    else:
        runpy.run_path(args.script, run_name="__main__")


def readiness_requirements(task):
    topics = ["/sensors_data_raw", "/gripper/state", "/odom", "/humanoid_mpc_observation"]
    topics.extend(f"/cam_{camera}/color/image_raw/compressed" for camera in ["h", "l", "r"])
    objects = {1: ["basket1", "basket2"], 2: ["bag_place_root", "bag_drop_root"],
               3: ["box_red1", "box_black1", "box_red2", "box_black2"]}[task]
    topics.extend(f"/mujoco/{name}/pose" for name in objects)
    services = ["/set_object_position", "/arm_traj_change_mode", "/humanoid_get_arm_ctrl_mode", "/ik/fk_srv"]
    return topics, services


def readiness_missing(topics, services, received, available_services, now, max_age=2.0):
    missing_topics = [name for name in topics if received.get(name, {}).get("count", 0) < 2
                      or now - received[name]["last_received"] > max_age]
    missing_services = [name for name in services if name not in available_services]
    return missing_topics, missing_services


def simulator_readiness(args):
    """Wait for live data and advertised services; never issue robot commands."""
    import rospy
    import threading
    directory = Path(args.manifest).parent
    manifest = read_json(args.manifest)
    install_signals()
    rospy.init_node("leju_recovery_sim_readiness", anonymous=True, disable_signals=True)
    topics, services = readiness_requirements(manifest["task_id"])
    received, subscriptions = {}, []
    lock = threading.Lock()
    started = time.monotonic()
    receipt = {"index": args.episode, "seed": manifest["seeds"][args.episode], "ready": False,
               "required_topics": topics, "required_services": services}
    path = directory / f"simulator_ready_episode_{args.episode:04d}.json"
    def callback(message, topic):
        with lock:
            prior = received.get(topic, {})
            received[topic] = {"count": prior.get("count", 0) + 1, "last_received": time.monotonic()}
    try:
        for topic in topics:
            subscriptions.append(rospy.Subscriber(topic, rospy.AnyMsg, callback, callback_args=topic, queue_size=1))
        deadline = started + manifest.get("simulator_ready_timeout_sec", 600)
        last_progress = 0
        while True:
            check_stop(directory)
            if rospy.is_shutdown():
                raise RuntimeError("ROS shut down before simulator readiness")
            available_services = []
            try:
                with xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"]) as master:
                    code, _, state = master.getSystemState(rospy.get_name())
                    if code == 1:
                        available_services = [name for name, nodes in state[2] if nodes]
            except OSError:
                pass
            now = time.monotonic()
            with lock:
                snapshot = {name: value.copy() for name, value in received.items()}
            missing_topics, missing_services = readiness_missing(topics, services, snapshot, available_services, now)
            receipt.update(elapsed_sec=now - started, received=snapshot,
                           missing_topics=missing_topics, missing_services=missing_services)
            if not missing_topics and not missing_services:
                receipt["ready"] = True
                write_json(path, receipt)
                print(json.dumps(receipt), flush=True)
                return
            if now > deadline:
                raise TimeoutError(f"Simulator readiness timed out: topics={missing_topics}, services={missing_services}")
            if now - last_progress >= 5:
                write_json(directory / f"simulator_readiness_progress_{args.episode:04d}.json", receipt)
                print(json.dumps({"elapsed_sec": receipt["elapsed_sec"], "missing_topics": missing_topics,
                                  "missing_services": missing_services}), flush=True)
                last_progress = now
            time.sleep(.1)
    except BaseException:
        receipt["failure"] = traceback.format_exc()
        write_json(path, receipt)
        raise
    finally:
        for subscription in subscriptions:
            subscription.unregister()
        rospy.signal_shutdown("Recovery simulator readiness check finished")


def worker_environment(manifest, directory):
    environment = os.environ.copy()
    environment.update({
        "ROS_MASTER_URI": f"http://127.0.0.1:{manifest['ros_port']}",
        "ROS_IP": "127.0.0.1",
        "ROS_HOME": str(directory / "ros_home"),
        "ROS_LOG_DIR": str(directory / "ros_logs"),
        "ROBOT_VERSION": "45",
        "LEJU_RECOVERY_RUN_ID": manifest["run_id"],
        "PYTHONUNBUFFERED": "1",
    })
    environment.pop("ROS_HOSTNAME", None)
    return environment


def policy_environment(manifest, directory, ros_bin="/opt/ros/noetic/bin"):
    environment = worker_environment(manifest, directory)
    # Kuavo's SDK invokes rosnode through PATH; preserve the policy virtualenv order.
    environment["PATH"] = ros_bin + os.pathsep + environment.get("PATH", "")
    return environment


def simulator_environment(manifest, directory, xvfb_display=None):
    environment = worker_environment(manifest, directory)
    environment["LD_LIBRARY_PATH"] = "/opt/drake/lib" + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    if xvfb_display:
        if not re.fullmatch(r":\d+", xvfb_display):
            raise ValueError("Owned Xvfb display must have the form :99")
        environment["DISPLAY"] = xvfb_display
    elif environment.get("DISPLAY"):
        # Hybrid laptops need explicit offload to use NVIDIA's GLX renderer.
        # Preserve an existing choice; manifest overrides support other hosts.
        environment.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        environment.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    environment.update(manifest.get("sim_environment", {}))
    return environment


def worker_identity(role, manifest, directory):
    try:
        os.setsid()
    except PermissionError:
        pass
    write_json(directory / f"{role}_worker_identity.json", {
        "pid": os.getpid(), "pgid": os.getpgrp(), "role": role,
        "run_id": manifest["run_id"], "script": str(Path(__file__).resolve()),
    })


def prepare(args):
    checkpoint = Path(args.checkpoint).resolve()
    config_path = checkpoint / "config.json"
    if not (checkpoint / "model.safetensors").is_file() or not config_path.is_file():
        raise ValueError("Checkpoint requires config.json and model.safetensors")
    config = read_json(config_path)
    if config.get("type") != "custom_act":
        raise ValueError("This first recovery comparison requires an ACT checkpoint")
    if config.get("custom", {}).get("use_depth", False):
        raise ValueError("This evaluation configuration is for the RGB ACT baseline")
    if config["output_features"]["action"]["shape"] != [16]:
        raise ValueError("Expected the simulation's 16-dimensional joint/gripper action")
    if args.capture_action_chunks and args.action_selection == "checkpoint" and config.get("n_action_steps") != 1:
        raise ValueError("Per-cycle chunk tracing requires the checkpoint to replan every cycle")
    action_horizon = getattr(args, "action_horizon", 1)
    gripper_selection = getattr(args, "gripper_selection", "policy")
    if gripper_selection == "latest":
        if (args.action_selection != "checkpoint" or config.get("n_action_steps") != 1
                or config.get("temporal_ensemble_coeff") is None or not args.capture_action_chunks):
            raise ValueError("Newest grippers require saved per-cycle averaging and --capture-action-chunks")
    if args.action_selection == "cached":
        if not 2 <= action_horizon <= config["chunk_size"]:
            raise ValueError("Cached execution needs an action horizon between 2 and chunk_size")
    elif action_horizon != 1:
        raise ValueError("An action horizon above 1 requires cached selection")
    task1_control = read_json(args.task1_control) if args.task1_control else None
    if task1_control and args.task != 1:
        raise ValueError("Task 1 layout controls cannot be used for another task")
    directory = Path(args.output).resolve()
    if not 1 <= args.episodes <= 100 or args.video_stride < 1 or not 1 <= args.max_episode_steps <= 1000:
        raise ValueError("Use 1–100 episodes, 1–1000 episode steps, and a positive video stride")
    if not 30 <= args.sim_ready_timeout <= 3600:
        raise ValueError("Simulator readiness timeout must be between 30 and 3600 seconds")
    sim_environment = {}
    for assignment in args.sim_env:
        name, separator, value = assignment.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("Simulator environment overrides must have the form NAME=VALUE")
        if name.startswith("ROS_") or name.startswith("LEJU_RECOVERY_"):
            raise ValueError("Simulator overrides cannot change ROS isolation or process ownership variables")
        sim_environment[name] = value
    if not 1024 <= args.ros_port <= 65535 or args.ros_port == 11311:
        raise ValueError("Use a dedicated non-default ROS port between 1024 and 65535")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Use a new, empty output directory to preserve earlier results")
    recovery = Path(__file__).resolve().parents[1]
    if args.checkpoint_container:
        checkpoint_container = args.checkpoint_container
    else:
        checkpoint_container = str(Path(args.mount_root) / checkpoint.relative_to(recovery))
    image_topics = {
        "head_cam_h": ["/cam_h/color/image_raw/compressed", "CompressedImage", 30, [640, 480]],
        "wrist_cam_l": ["/cam_l/color/image_raw/compressed", "CompressedImage", 30, [640, 480]],
        "wrist_cam_r": ["/cam_r/color/image_raw/compressed", "CompressedImage", 30, [640, 480]],
        "joint_q": ["/sensors_data_raw", "sensorsData", 500],
        "rq2f85": ["/gripper/state", "JointState", 500],
    }
    runtime_config = {
        "env": {
            "env_name": "Kuavo-Sim", "control_mode": "joint", "eef_type": "rq2f85",
            "which_arm": "both", "head_init": [0, 0.209], "ros_rate": 10,
            "only_arm": True, "image_size": [640, 480], "depth_range": [0, 1500],
            "obs_key_map": image_topics, "arm_state_keys": ["joint_q", "gripper"],
            "frame_alignment": False, "ratio": 1.0, "is_binary": False,
        },
        "inference": {
            "policy_type": "act", "device": "cuda", "eval_episodes": args.episodes,
            "seed": args.seed_start, "start_seed": args.seed_start,
            "task": {1: "Task1_ToySorting", 2: "Task2_ParcelWeighing", 3: "Task3_ConveyorBeltSorting"}[args.task],
            "method": args.label, "timestamp": directory.name, "epoch": "best",
            "max_episode_steps": args.max_episode_steps,
        },
    }
    manifest = {
        "schema_version": 1, "run_id": uuid.uuid4().hex, "label": args.label,
        "task_id": args.task, "seeds": list(range(args.seed_start, args.seed_start + args.episodes)),
        "seed_note": "44 is the first scored seed after the original deploy script's discarded seed-43 reset round",
        "checkpoint_host": str(checkpoint), "checkpoint_container": checkpoint_container,
        "output_host": str(directory), "output_container": str(Path(args.mount_root) / directory.relative_to(recovery)),
        "mount_root": args.mount_root, "ros_port": args.ros_port,
        "max_episode_steps": args.max_episode_steps, "ros_rate": 10, "robot_version": 45,
        "record_episodes": [int(x) for x in args.record_episodes.split(",") if x],
        "video_stride": args.video_stride, "startup_timeout_sec": 120,
        "episode_timeout_sec": 180, "runtime_config": "runtime_config.json",
        "simulator_ready_timeout_sec": args.sim_ready_timeout,
        "sim_environment": sim_environment,
        "action_selection": args.action_selection,
        "gripper_selection": gripper_selection,
        "action_horizon": action_horizon,
        "capture_action_chunks": args.capture_action_chunks,
        "record_full_telemetry": not args.disable_rosbag,
        "task1_control": task1_control,
        "track_objects": args.track_objects,
        "policy_constructor_note": "ImageNet downloads are disabled during checkpoint construction; strict checkpoint loading restores all learned weights",
        "checkpoint_files_sha256": {
            str(path.relative_to(checkpoint.parent)): sha256(path)
            for path in [checkpoint / "model.safetensors", checkpoint / "config.json"]
            + sorted(processor_directory(checkpoint).glob("*processor*.json"))
            + sorted(processor_directory(checkpoint).glob("*processor*.safetensors"))
        },
    }
    write_json(directory / "runtime_config.json", runtime_config)
    if task1_control:
        write_json(directory / "task1_control.json", task1_control)
    write_json(directory / "manifest.json", manifest)
    print(directory / "manifest.json")


def policy_worker(args):
    manifest_path = Path(args.manifest)
    directory = manifest_path.parent
    manifest = read_json(manifest_path)
    os.environ.pop("ROS_HOSTNAME", None)
    os.environ.update(policy_environment(manifest, directory))
    ros_cli = {"rosnode": shutil.which("rosnode")}
    if not all(ros_cli.values()):
        raise RuntimeError("Kuavo SDK requires rosnode on the policy worker PATH")
    install_signals()
    worker_identity("policy", manifest, directory)
    os.chdir(args.policy_repository)
    sys.path.insert(0, args.policy_repository)
    import rospy
    import torch
    import numpy as np
    import gymnasium as gym
    from lerobot_patches import custom_patches  # registers RGB/DEPTH feature types
    from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
    from std_msgs.msg import Bool
    from kuavo_deploy.config import load_kuavo_config
    from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv  # registers existing env
    from kuavo_deploy.utils.obs_buffer import ObsBuffer
    from kuavo_train.wrapper.policy.act.ACTConfigWrapper import CustomACTConfigWrapper
    from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.random_utils import set_seed
    from observation_timing import ObservationTimingMonitor, summarize_timing
    import threading

    rospy.init_node("leju_recovery_policy", anonymous=True, disable_signals=True)
    init_event = threading.Event()
    success_event = threading.Event()
    def initialized(request):
        init_event.set()
        return TriggerResponse(success=True, message="Recovery policy ready")
    init_service = rospy.Service("/simulator/init", Trigger, initialized)
    success_sub = rospy.Subscriber("/simulator/success", Bool,
                                   lambda msg: success_event.set() if msg.data else None)
    config = load_kuavo_config(str(directory / manifest["runtime_config"]))
    checkpoint = Path(manifest["checkpoint_container"])
    policy_config = CustomACTConfigWrapper.from_pretrained(checkpoint)
    if manifest.get("action_selection", "checkpoint") == "latest":
        policy_config.temporal_ensemble_coeff = None
        policy_config.n_action_steps = 1
    elif manifest.get("action_selection") == "cached":
        horizon = manifest["action_horizon"]
        if not 2 <= horizon <= policy_config.chunk_size:
            raise ValueError("Cached action horizon exceeds the trained chunk")
        policy_config.temporal_ensemble_coeff = None
        policy_config.n_action_steps = horizon
    # Evaluation-only settings; never rewrite the saved training checkpoint.
    overrides = manifest.get("policy_overrides", {})
    if set(overrides) - {"temporal_ensemble_coeff", "n_action_steps"}:
        raise ValueError("Unsupported evaluation policy override")
    for name, value in overrides.items():
        setattr(policy_config, name, value)
    if not 1 <= policy_config.n_action_steps <= policy_config.chunk_size:
        raise ValueError("n_action_steps must be within the trained chunk")
    coefficient = policy_config.temporal_ensemble_coeff
    if coefficient is not None:
        import math
        if not math.isfinite(coefficient) or policy_config.n_action_steps != 1:
            raise ValueError("Temporal averaging requires a finite coefficient and one action step")
    # Optional per-dimension cap on |applied action change| per control step (None = unlimited).
    # Smooths chunk-boundary jumps that can trip the whole-body controller's safety check.
    rate_limit = manifest.get("action_rate_limit")
    if rate_limit is not None:
        rate_limit = np.array([np.inf if v is None else float(v) for v in rate_limit])
        if rate_limit.shape != (16,) or (rate_limit <= 0).any():
            raise ValueError("action_rate_limit must list 16 positive limits or nulls")
    policy_config.pretrained_backbone_weights = None
    if hasattr(policy_config, "pretrained_depth_backbone_weights"):
        policy_config.pretrained_depth_backbone_weights = None
    policy = CustomACTPolicyWrapper.from_pretrained(checkpoint, config=policy_config, strict=True)
    policy.eval().to(torch.device(config.inference.device))
    processors = processor_directory(checkpoint)
    preprocessor, postprocessor = make_pre_post_processors(None, processors)
    chunk_trace = None
    gripper_selection = manifest.get("gripper_selection", "policy")
    if gripper_selection not in ("policy", "latest"):
        raise ValueError("Unknown gripper selection")
    if gripper_selection == "latest" and (policy_config.temporal_ensemble_coeff is None
            or policy_config.n_action_steps != 1 or not manifest.get("capture_action_chunks")):
        raise ValueError("Gripper-only control must preserve per-cycle arm averaging and capture predictions")
    if manifest.get("capture_action_chunks", False):
        from action_selection_trace import ActionSelectionTrace, apply_gripper_selection
        chunk_trace = ActionSelectionTrace(policy)
    # The existing deployment modules log each tensor/state by default.
    for name in ["model", "robot", "env"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    loaded_hashes = {name: sha256(checkpoint.parent / name)
                     for name in manifest["checkpoint_files_sha256"]}
    if loaded_hashes != manifest["checkpoint_files_sha256"]:
        raise RuntimeError("Loaded checkpoint/processor files do not match the prepared manifest")
    write_json(directory / "policy_ready.json", {
        "pid": os.getpid(), "checkpoint": str(checkpoint), "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "checkpoint_files_sha256": loaded_hashes,
        "processor_directory": str(processors),
        "ros_cli": ros_cli,
        "action_selection": manifest.get("action_selection", "checkpoint"),
        "temporal_ensemble_coeff": policy_config.temporal_ensemble_coeff,
        "n_action_steps": policy_config.n_action_steps,
        "capture_action_chunks": chunk_trace is not None,
        "gripper_selection": gripper_selection,
        "policy_overrides": overrides,
        "action_rate_limit": None if rate_limit is None else [None if np.isinf(v) else v for v in rate_limit.tolist()],
    })
    observation_monitor = ObservationTimingMonitor(ObsBuffer, lambda: rospy.Time.now().to_sec())
    env = None
    try:
        for index, seed in enumerate(manifest["seeds"]):
            check_stop(directory)
            success_event.clear()
            init_event.clear()
            write_json(directory / f"policy_next_ready_{index:04d}.json", {"index": index, "seed": seed})
            ready = wait_file(directory / f"simulator_ready_episode_{index:04d}.json", directory,
                              manifest.get("simulator_ready_timeout_sec", 600) + 15)
            if not ready.get("ready") or ready.get("index") != index or ready.get("seed") != seed:
                raise RuntimeError(f"Simulator readiness failed for seed {seed}: {ready.get('failure')}")
            deadline = time.monotonic() + manifest["startup_timeout_sec"]
            while not init_event.wait(0.1):
                check_stop(directory)
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Simulator initialization timed out for seed {seed}")
            result = {"index": index, "seed": seed, "success": False, "steps": 0,
                      "failure": None, "videos": [], "recorded": index in manifest["record_episodes"],
                      "observation_timing": [], "first_steps": []}
            previous_monitor_errors = dict(observation_monitor.errors)
            observations, actions, frames, frame_times = [], [], {}, []
            previous_applied, rate_limited_steps = None, 0
            action_traces = []
            pose_trace, object_snapshots = None, []
            inference_times, step_times = [], []
            started = started_unix = None
            try:
                set_seed(seed)
                policy.reset()
                if manifest.get("track_objects", False):
                    from object_pose_trace import ObjectPoseTrace
                    if manifest["task_id"] == 1:
                        import random
                        rng = random.Random(seed)
                        control = manifest.get("task1_control")
                        toy = rng.choice(["dino", "dog", "elephant", "shark", "sheep"])
                        car = rng.choice(["fire", "roller", "school", "boat", "fire_engine"])
                        names = [control["toy"], control["car"]] if control else [toy, car]
                        names += ["basket1", "basket2"]
                    elif manifest["task_id"] == 3:
                        names = ["box_red1", "box_black1", "box_red2", "box_black2"]
                    else:
                        names = ["bag_place_root", "bag_drop_root"]
                    pose_trace = ObjectPoseTrace(names)
                    result["tracked_object_names"] = names
                env = guarded_call(lambda: gym.make(config.env.env_name, config=config,
                                  max_episode_steps=manifest["max_episode_steps"]), 60)
                observation, _ = guarded_call(lambda: env.reset(seed=seed), 30)
                if chunk_trace is not None:
                    chunk_trace.attach_environment(env.unwrapped)
                result["initial_state"] = observation["observation.state"].detach().cpu().reshape(-1).tolist()
                if pose_trace is not None:
                    result["initial_object_poses"] = pose_trace.snapshot()
                result["initial_observation_timing"] = observation_monitor.snapshot(
                    env.unwrapped.obs_buffer, rospy.Time.now().to_sec())
                rospy.wait_for_service("/simulator/start", timeout=30)
                success_event.clear()
                response = guarded_call(lambda: rospy.ServiceProxy("/simulator/start", Trigger)(TriggerRequest()), 30)
                if not response.success:
                    raise RuntimeError(f"Simulator rejected start: {response.message}")
                started, started_unix = time.monotonic(), time.time()
                while result["steps"] < manifest["max_episode_steps"]:
                    check_stop(directory)
                    if time.monotonic() - started > manifest["episode_timeout_sec"]:
                        raise TimeoutError("Policy episode exceeded its wall-clock watchdog")
                    step_start = time.monotonic()
                    timing = observation_monitor.snapshot(env.unwrapped.obs_buffer, rospy.Time.now().to_sec())
                    timing["step"] = result["steps"]
                    result["observation_timing"].append(timing)
                    if result["recorded"]:
                        observations.append(observation["observation.state"].detach().cpu().numpy().reshape(-1))
                    batch = preprocessor(observation)
                    if chunk_trace is not None:
                        chunk_trace.begin_step()
                    with torch.inference_mode():
                        normalized_action = policy.select_action(batch)
                        action = postprocessor(normalized_action)
                        if chunk_trace is not None:
                            trace = chunk_trace.prediction_record(normalized_action, postprocessor)
                    array = action.squeeze(0).cpu().numpy()
                    policy_array = array.copy()
                    if chunk_trace is not None:
                        array = apply_gripper_selection(array, trace, gripper_selection)
                    if rate_limit is not None:
                        flat = array.reshape(-1)
                        if previous_applied is not None:
                            limited = previous_applied + np.clip(flat - previous_applied, -rate_limit, rate_limit)
                            if not np.array_equal(limited, flat):
                                rate_limited_steps += 1
                            array = limited.reshape(array.shape).astype(array.dtype)
                        previous_applied = array.reshape(-1).astype(np.float64).copy()
                    inference_times.append(time.monotonic() - step_start)
                    if result["steps"] < 5:
                        result["first_steps"].append({"step": result["steps"],
                            "state": observation["observation.state"].detach().cpu().reshape(-1).tolist(),
                            "postprocessed_policy_action": policy_array.reshape(-1).tolist(),
                            "applied_action": array.reshape(-1).tolist(),
                            "expected_box_clipped_action": np.clip(array, env.unwrapped.action_space.low,
                                                                  env.unwrapped.action_space.high).reshape(-1).tolist(),
                            "action_note": "Expected default environment box clipping; ROS command dispatch is not intercepted."})
                    observation, _, terminated, truncated, _ = env.step(array)
                    if pose_trace is not None:
                        snapshot = pose_trace.snapshot()
                        snapshot["elapsed_sec"] = time.monotonic() - started
                        object_snapshots.append(snapshot)
                    if chunk_trace is not None:
                        action_traces.append(chunk_trace.finish_step(trace, array, time.monotonic() - started))
                    if result["recorded"]:
                        actions.append(array.copy())
                        if result["steps"] % manifest["video_stride"] == 0:
                            frame_times.append(time.monotonic() - started)
                            for key, value in observation.items():
                                if "images" in key:
                                    frame = (value.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                                    frames.setdefault(key, []).append(frame)
                    step_times.append(time.monotonic() - step_start)
                    result["steps"] += 1
                    if terminated or truncated or success_event.is_set():
                        result["termination"] = "success" if success_event.is_set() else "time_limit"
                        break
                result.setdefault("termination", "time_limit")
                result["success"] = success_event.is_set()
            except (Exception, SystemExit):
                result["failure"] = traceback.format_exc()
                result["termination"] = "exception"
            finally:
                # End scoring before disk writes or video encoding.
                result["episode_wall_sec"] = time.monotonic() - started if started else None
                try:
                    response = guarded_call(lambda: rospy.ServiceProxy("/simulator/reset", Trigger)(TriggerRequest()), 15)
                    if not response.success:
                        raise RuntimeError(f"Simulator rejected reset: {response.message}")
                except Exception:
                    result["reset_failure"] = traceback.format_exc()
                if env is not None:
                    env.close()
                    env = None
                if pose_trace is not None:
                    pose_trace.close()
            duration = result["episode_wall_sec"]
            result["actual_action_hz"] = result["steps"] / duration if duration else None
            if rate_limit is not None:
                result["rate_limited_steps"] = rate_limited_steps
            result["mean_inference_sec"] = statistics.mean(inference_times) if inference_times else None
            result["p95_inference_sec"] = float(np.percentile(inference_times, 95)) if inference_times else None
            result["mean_step_sec"] = statistics.mean(step_times) if step_times else None
            result["p95_step_sec"] = float(np.percentile(step_times, 95)) if step_times else None
            result["observation_timing_summary"] = summarize_timing(result["observation_timing"])
            result["observation_monitor_errors"] = {
                name: count - previous_monitor_errors.get(name, 0)
                for name, count in observation_monitor.errors.items()
                if count > previous_monitor_errors.get(name, 0)}
            if frames:
                import imageio.v2 as imageio
                video_directory = directory / "videos" / f"episode_{index:04d}_seed{seed}"
                video_directory.mkdir(parents=True, exist_ok=True)
                for key, camera_frames in frames.items():
                    path = video_directory / (key + ".mp4")
                    imageio.mimsave(path, camera_frames, fps=max(0.1, len(camera_frames) / duration))
                    result["videos"].append(str(path))
                write_json(video_directory / "frame_times.json", frame_times)
                # frame_times are seconds since this moment; lets other recordings be aligned
                write_json(video_directory / "episode_start.json", {"unix_time": started_unix})
                np.savez_compressed(video_directory / "state_action_trace.npz", state=np.array(observations), action=np.array(actions))
            if action_traces:
                trace_directory = directory / "videos" / f"episode_{index:04d}_seed{seed}"
                trace_directory.mkdir(parents=True, exist_ok=True)
                trace_path = trace_directory / "action_selection_trace.npz"
                np.savez_compressed(trace_path, **{
                    key: np.asarray([row[key] for row in action_traces]) for key in action_traces[0]
                })
                result["action_selection_trace"] = str(trace_path)
                result["action_selection_trace_steps"] = len(action_traces)
            if object_snapshots:
                trace_directory = directory / "videos" / f"episode_{index:04d}_seed{seed}"
                trace_directory.mkdir(parents=True, exist_ok=True)
                trace_path = trace_directory / "object_pose_trace.npz"
                np.savez_compressed(trace_path, object_names=np.asarray(result["tracked_object_names"]), **{
                    key: np.asarray([row[key] for row in object_snapshots]) for key in object_snapshots[0]
                })
                result["object_pose_trace"] = str(trace_path)
            write_json(directory / f"policy_episode_{index:04d}.json", result)
            print(json.dumps({key: result.get(key) for key in ["index", "seed", "success", "steps", "actual_action_hz", "termination"]}), flush=True)
            if result["failure"] or result.get("reset_failure"):
                raise RuntimeError(f"Infrastructure/policy failure for seed {seed}; see episode receipt")
    finally:
        if chunk_trace is not None:
            chunk_trace.close()
        observation_monitor.close()
        if env is not None:
            env.close()
        init_service.shutdown()
        success_sub.unregister()
        rospy.signal_shutdown("Recovery evaluation finished")


def simulator_worker(args):
    with workspace_lock(args.sim_workspace):
        return _simulator_worker_locked(args)


def _simulator_worker_locked(args):
    directory = Path(args.manifest).parent
    manifest = read_json(args.manifest)
    install_signals()
    worker_identity("simulator", manifest, directory)
    environment = simulator_environment(manifest, directory, args.xvfb_display)
    write_json(directory / "simulator_environment.json", {
        key: environment.get(key) for key in ["DISPLAY", "LD_LIBRARY_PATH", "__NV_PRIME_RENDER_OFFLOAD",
                                             "__GLX_VENDOR_LIBRARY_NAME", "ROS_MASTER_URI", "ROS_IP"]})
    workspace = Path(args.sim_workspace)
    package = workspace / "src/data_challenge_simulator"
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location("leju_original_deploy", package / "examples/deploy/deploy.py")
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    setup = "source /opt/ros/noetic/setup.bash; source " + shlex.quote(args.sim_setup) + "; exec "
    def launch(command, log_name, extra=None):
        log = (directory / log_name).open("wb")
        child_environment = {**environment, **(extra or {})}
        return subprocess.Popen(["bash", "-c", setup + shlex.join(command)],
                                stdout=log, stderr=subprocess.STDOUT, env=child_environment,
                                start_new_session=True)
    # This port must be free: do not attach to or shut down another ROS master.
    with socket.socket() as test:
        # Old XML-RPC connections can be in TIME_WAIT after our master exits.
        # Reuse permits those closed connections, while a listener still rejects bind.
        test.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test.bind(("127.0.0.1", manifest["ros_port"]))
    roscore = simulation = evaluator = readiness = xvfb = None
    originals = {}
    ownership = {"run_id": manifest["run_id"], "sim_workspace": str(workspace), "models": []}
    task = manifest["task_id"]
    if task in (1, 3):
        model_paths = [package / "models/biped_s45/xml/scene1.xml",
                       package / "models/assets/basket1/model.xml",
                       package / "models/assets/basket2/model.xml"] if task == 1 else \
                      [package / "models/biped_s45/xml/scene3.xml"]
        originals = {path: path.read_bytes() for path in model_paths}
        backup_directory = directory / "original_xml"
        backup_directory.mkdir(exist_ok=True)
        for path, content in originals.items():
            original_stat = path.stat()
            digest = hashlib.sha256(content).hexdigest()
            backup = backup_directory / (hashlib.sha256(str(path).encode()).hexdigest()[:12] + ".xml")
            backup.write_bytes(content)
            ownership["models"].append({"path": str(path), "backup": str(backup.relative_to(directory)),
                "original_sha256": digest, "expected_sha256": digest,
                "mode": original_stat.st_mode & 0o777,
                "uid": original_stat.st_uid, "gid": original_stat.st_gid})
        write_json(directory / "model_xml_ownership.json", ownership)
    try:
        if args.xvfb_display:
            xvfb = launch(["Xvfb", args.xvfb_display, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"], "xvfb.log")
            time.sleep(1)
            if xvfb.poll() is not None:
                raise RuntimeError("Owned Xvfb failed to start; inspect xvfb.log")
        roscore = launch(["roscore", "-p", str(manifest["ros_port"])], "roscore.log")
        deadline = time.monotonic() + 30
        while True:
            check_stop(directory)
            try:
                with xmlrpc.client.ServerProxy(environment["ROS_MASTER_URI"]) as master:
                    if master.getPid("/leju_recovery_preflight")[0] == 1:
                        break
            except OSError:
                pass
            if roscore.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Dedicated recovery ROS master failed to start")
            time.sleep(0.1)
        write_json(directory / "master_ready.json", {"pid": roscore.pid, "uri": environment["ROS_MASTER_URI"]})
        wait_file(directory / "policy_ready.json", directory, manifest["startup_timeout_sec"])
        for index, seed in enumerate(manifest["seeds"]):
            check_stop(directory)
            wait_file(directory / f"policy_next_ready_{index:04d}.json", directory, 120)
            conveyor, table, basket1, basket2 = deploy.generate_cfg(seed)
            if task in (1, 3):
                generated_directory = directory / "generated_xml" / f"episode_{index:04d}"
                generated_directory.mkdir(parents=True, exist_ok=True)
                for record, config in zip(ownership["models"], [table, basket1, basket2] if task == 1 else [conveyor]):
                    path = Path(record["path"])
                    current = sha256(path) if path.exists() else None
                    if current != record["expected_sha256"]:
                        write_json(directory / "model_restore_conflict.json", {
                            "conflicts": [{"path": str(path), "reason": "External change detected before XML generation; preserved"}]})
                        raise RuntimeError(f"Simulator XML changed outside this run: {path}")
                    backup = directory / record["backup"]
                    generated = generated_directory / backup.name
                    deploy.randomize_mjcf(in_path=str(backup), out_path=str(generated), config=config, seed=seed)
                    record["expected_sha256"] = sha256(generated)
                    # Persist restoration ownership before atomic replacement so forced cleanup can recover.
                    write_json(directory / "model_xml_ownership.json", ownership)
                    atomic_model_write(path, generated.read_bytes(), record["mode"], manifest["run_id"],
                                       record["uid"], record["gid"])
                write_json(directory / f"scene_episode_{index:04d}.json", {
                    "index": index, "seed": seed,
                    "model_hashes": {r["path"]: r["expected_sha256"] for r in ownership["models"]}})
            score_file = directory / "scores" / f"episode_{index:04d}_seed{seed}.txt"
            score_file.parent.mkdir(exist_ok=True)
            launch_arguments = ["roslaunch", "data_challenge_simulator", f"load_kuavo_mujoco_sim{task}.launch", "coredump:=false"]
            if not manifest.get("record_full_telemetry", True):
                launch_arguments.append("record_full_telemetry:=false")
            simulation = launch(launch_arguments,
                                f"simulator_{index:04d}.log", {"MUJOCO_HEADLESS": environment.get("MUJOCO_HEADLESS", "1"), "LEJU_RECOVERY_EPISODE": str(index)})
            readiness = launch([args.sim_python, str(Path(__file__).resolve()), "_sim_readiness",
                                "--manifest", str(args.manifest), "--episode", str(index)],
                               f"readiness_{index:04d}.log", {"LEJU_RECOVERY_EPISODE": str(index)})
            deadline = time.monotonic() + manifest.get("simulator_ready_timeout_sec", 600) + 15
            while readiness.poll() is None:
                check_stop(directory)
                if simulation.poll() is not None:
                    raise RuntimeError(f"ROS launch exited before readiness for seed {seed}; see simulator log")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Simulator readiness watchdog expired for seed {seed}")
                time.sleep(.1)
            if readiness.returncode != 0:
                raise RuntimeError(f"Simulator readiness failed for seed {seed}; inspect readiness log/receipt")
            ready = read_json(directory / f"simulator_ready_episode_{index:04d}.json")
            if not ready.get("ready") or ready.get("index") != index or ready.get("seed") != seed:
                raise RuntimeError(f"Simulator readiness receipt mismatch for seed {seed}")
            readiness = None
            evaluator_arguments = [args.sim_python, str(Path(__file__).resolve()), "_evaluator_bootstrap",
                                   "--script", str(package / f"examples/deploy/eval{task}.py"), "--seed", str(seed)]
            if manifest.get("task1_control"):
                evaluator_arguments += ["--task1-control", str(directory / "task1_control.json"),
                                        "--control-receipt", str(directory / f"controlled_layout_{index:04d}.json")]
            evaluator = launch(evaluator_arguments,
                               f"evaluator_{index:04d}.log", {"SCORE_FILE": str(score_file), "LEJU_RECOVERY_EPISODE": str(index),
                                "PYTHONPATH": str(package) + os.pathsep + environment.get("PYTHONPATH", "")})
            deadline = time.monotonic() + manifest["startup_timeout_sec"] + manifest["episode_timeout_sec"]
            while evaluator.poll() is None:
                check_stop(directory)
                if simulation.poll() is not None:
                    raise RuntimeError(f"Simulator exited during seed {seed}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Evaluator watchdog expired for seed {seed}")
                time.sleep(0.1)
            score, detail, score_read_failure = None, None, None
            try:
                score = int(score_file.read_text().strip()) if score_file.exists() else None
                detail = read_json(score_file.with_suffix(".json")) if score_file.with_suffix(".json").exists() else None
            except (OSError, ValueError):
                score_read_failure = traceback.format_exc()
            receipt = {
                "index": index, "seed": seed, "evaluator_returncode": evaluator.returncode,
                "score": score, "detail": detail, "score_read_failure": score_read_failure,
                "launch_returncode_before_cleanup": simulation.poll(),
            }
            receipt["score_errors"] = simulator_score_errors(receipt)
            receipt["score_valid"] = not receipt["score_errors"]
            write_json(directory / f"simulator_episode_{index:04d}.json", receipt)
            stop_process(simulation)
            stop_owned_descendants(manifest["run_id"], "_sim_worker", str(index))
            simulation = evaluator = None
            if not receipt["score_valid"]:
                raise RuntimeError(f"Invalid evaluator outcome for seed {seed}: {receipt['score_errors']}")
            # Wait for post-scoring video encoding before launching the next episode.
            wait_file(directory / f"policy_episode_{index:04d}.json", directory, 120)
    finally:
        stop_process(readiness)
        stop_process(evaluator)
        stop_process(simulation)
        stop_process(roscore)
        stop_process(xvfb)
        stop_owned_descendants(manifest["run_id"], "_sim_worker")
        restore_owned_models(directory)


def stop_worker(args):
    directory = Path(args.manifest).parent
    manifest = read_json(args.manifest)
    identity_path = directory / f"{args.worker}_worker_identity.json"
    if not identity_path.exists():
        return
    identity = read_json(identity_path)
    if identity["run_id"] != manifest["run_id"]:
        raise RuntimeError("Worker ownership mismatch")
    pid = identity["pid"]
    role = "_policy_worker" if args.worker == "policy" else "_sim_worker"
    def verified():
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
            if not command:  # An exited, unreaped worker has an empty command line.
                return None
            if b"run_sim_evaluation.py" not in command or str(args.manifest).encode() not in command:
                raise RuntimeError("PID no longer belongs to this recovery worker")
            if os.getpgid(pid) != identity["pgid"] or identity["pgid"] != pid:
                raise RuntimeError("Worker process group ownership mismatch")
            fingerprint = owned_fingerprint(pid, manifest["run_id"], role)
            if fingerprint is None:
                raise RuntimeError("Worker UUID/role ownership token mismatch")
            return fingerprint
        except (ProcessLookupError, FileNotFoundError):
            return None
    fingerprint = verified()
    for sig, grace in [(signal.SIGINT, 20), (signal.SIGTERM, 3), (signal.SIGKILL, 1)]:
        if fingerprint is None or verified() != fingerprint:
            break
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and verified() == fingerprint:
            time.sleep(.1)
    stop_owned_descendants(manifest["run_id"], role)
    if args.worker == "simulator":
        restore_models_after_worker_exit(directory)


def summarize(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    rows = []
    for index, seed in enumerate(manifest["seeds"]):
        policy_file = directory / f"policy_episode_{index:04d}.json"
        simulator_file = directory / f"simulator_episode_{index:04d}.json"
        policy = read_json(policy_file) if policy_file.exists() else None
        simulator = read_json(simulator_file) if simulator_file.exists() else None
        errors = simulator_score_errors(simulator)
        if not policy:
            errors.append("missing policy receipt")
        elif policy.get("failure") or policy.get("reset_failure"):
            errors.append("policy/reset infrastructure failure")
        for role, receipt in [("policy", policy), ("simulator", simulator)]:
            if receipt and (receipt.get("index") != index or receipt.get("seed") != seed):
                errors.append(f"{role} receipt episode/seed mismatch")
        rows.append({"seed": seed, "policy": policy, "simulator": simulator,
                     "comparable": not errors, "invalid_reasons": errors})
    valid_rows = [row for row in rows if row["comparable"]]
    scores = [r["simulator"]["score"] for r in valid_rows]
    frequencies = [r["policy"]["actual_action_hz"] for r in valid_rows
                   if r["policy"].get("actual_action_hz") is not None]
    component_names = {1: ["toy_pos", "car_pos", "time"],
                       2: ["intermediate_pos", "final_pos", "time"],
                       3: ["box_red1_pos", "box_black1_pos", "box_red2_pos", "box_black2_pos", "bonus", "time"]}
    components = {name: 0 for name in component_names[manifest["task_id"]]}
    for row in valid_rows:
        simulator = row["simulator"]
        if simulator and simulator.get("score") is not None:
            for name, value in (simulator.get("detail") or {}).get("components", {}).items():
                components[name] = components.get(name, 0) + value
    result = {
        "run_id": manifest["run_id"], "label": manifest["label"],
        "requested_episodes": len(rows), "episodes_with_scores": len(scores),
        "score_files_present": sum(bool(r["simulator"] and r["simulator"].get("score") is not None) for r in rows),
        "infrastructure_failure_count": len(rows) - len(valid_rows),
        "mean_score": statistics.mean(scores) if scores else None,
        "components_mean": {k: v / len(scores) for k, v in components.items()} if scores else {},
        "mean_actual_action_hz": statistics.mean(frequencies) if frequencies else None,
        "success_count": sum(bool(r["policy"].get("success")) for r in valid_rows),
        "policy_failure_count": sum(bool(r["policy"] and (r["policy"].get("failure") or r["policy"].get("reset_failure"))) for r in rows),
        "complete": len(valid_rows) == len(rows),
        "episodes": rows,
    }
    write_json(directory / "summary.json", result)
    return result


def coordinate(args):
    directory = Path(args.manifest).resolve().parent
    manifest = read_json(directory / "manifest.json")
    if (directory / "run_status.json").exists():
        raise RuntimeError("This run already started; prepare a new directory to preserve results")
    for container in [args.policy_container, args.sim_container]:
        data = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
        if not data["State"]["Running"] or data["Config"].get("Labels", {}).get("leju.recovery") != "true":
            raise RuntimeError(f"{container} must be a running container created for recovery with label leju.recovery=true")
        if data["HostConfig"].get("NetworkMode") != "host":
            raise RuntimeError("Both recovery containers require host networking for the dedicated ROS master")
        if data["HostConfig"].get("PidMode") == "host":
            raise RuntimeError("Recovery process ownership requires a private container PID namespace")
    container_manifest = str(Path(manifest["output_container"]) / "manifest.json")
    script = str(Path(manifest["mount_root"]) / "tools/run_sim_evaluation.py")
    processes = []
    def command(container, python, role, extra):
        return ["docker", "exec", "-e", f"LEJU_RECOVERY_RUN_ID={manifest['run_id']}",
                "-e", f"LEJU_RECOVERY_ROLE={role}", container, python, script,
                role, "--manifest", container_manifest] + extra
    install_signals()
    write_json(directory / "run_status.json", {"status": "running", "started_at_unix": time.time(), "run_id": manifest["run_id"]})
    try:
        sim_options = ["--sim-workspace", args.sim_workspace, "--sim-setup", args.sim_setup, "--sim-python", args.sim_python]
        if args.xvfb_display:
            sim_options += ["--xvfb-display", args.xvfb_display]
        sim_command = command(args.sim_container, args.sim_python, "_sim_worker", sim_options)
        with (directory / "sim_worker.log").open("wb") as log:
            processes.append(subprocess.Popen(sim_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        wait_file(directory / "master_ready.json", directory, 45)
        policy_command = command(args.policy_container, args.policy_python, "_policy_worker", ["--policy-repository", args.policy_repository])
        with (directory / "policy_worker.log").open("wb") as log:
            processes.append(subprocess.Popen(policy_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while any(process.poll() is None for process in processes):
            check_stop(directory)
            for process in processes:
                if process.poll() not in [None, 0]:
                    raise RuntimeError(f"Recovery worker exited with status {process.returncode}; inspect worker logs")
            time.sleep(0.25)
        result = summarize(directory)
        write_json(directory / "run_status.json", {"status": "complete" if result["complete"] else "incomplete", "finished_at_unix": time.time(), "run_id": manifest["run_id"]})
        print(json.dumps({key: result[key] for key in ["label", "requested_episodes", "episodes_with_scores", "mean_score", "mean_actual_action_hz", "complete"]}, indent=2))
        if not result["complete"]:
            raise RuntimeError("Not all requested episodes completed; infrastructure failures are not zero task scores")
    except BaseException:
        write_json(directory / "run_status.json", {
            "status": "interrupted" if STOP else "failed",
            "finished_at_unix": time.time(), "run_id": manifest["run_id"],
            "failure": traceback.format_exc(),
        })
        raise
    finally:
        (directory / "stop.request").touch()
        for container, python, role in [(args.policy_container, args.policy_python, "policy"), (args.sim_container, args.sim_python, "simulator")]:
            try:
                subprocess.run(command(container, python, "_stop_worker", ["--worker", role]), timeout=30, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                stop_process(process)
        summarize(directory)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-container")
    p.add_argument("--label", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--action-selection", choices=["checkpoint", "latest", "cached"], default="checkpoint",
                   help="Keep saved averaging, apply each newest first action, or execute a cached chunk prefix")
    p.add_argument("--action-horizon", type=int, default=1,
                   help="Number of cached actions to execute before replanning; used with cached selection")
    p.add_argument("--gripper-selection", choices=["policy", "latest"], default="policy",
                   help="Keep selected grippers or combine saved averaged arms with newest gripper predictions")
    p.add_argument("--capture-action-chunks", action="store_true",
                   help="Record raw chunks, selected actions and commands at the dispatch boundary")
    p.add_argument("--disable-rosbag", action="store_true",
                   help="Disable optional full ROS telemetry; retain scores, videos and compact action traces")
    p.add_argument("--task1-control", help="JSON identity/side controls for diagnostic Task 1 layouts")
    p.add_argument("--track-objects", action="store_true", help="Record object positions alongside policy actions")
    p.add_argument("--sim-env", action="append", default=[], metavar="NAME=VALUE",
                   help="Simulator-only environment override; repeat for host-specific rendering settings")
    p.add_argument("--sim-ready-timeout", type=int, default=600,
                   help="Maximum wait for live simulator data/control services before starting the evaluator")
    p.add_argument("--seed-start", type=int, default=44)
    p.add_argument("--ros-port", type=int, default=11315)
    p.add_argument("--mount-root", default="/recovery")
    p.add_argument("--record-episodes", default="0,19")
    p.add_argument("--video-stride", type=int, default=2)
    p.set_defaults(function=prepare)
    p = commands.add_parser("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--policy-container", required=True)
    p.add_argument("--sim-container", required=True)
    p.add_argument("--policy-python", default="/root/kuavo_data_challenge/myenv/bin/python")
    p.add_argument("--sim-python", default="/usr/bin/python3")
    p.add_argument("--policy-repository", default="/root/kuavo_data_challenge")
    p.add_argument("--sim-workspace", default="/root/kuavo_ws")
    p.add_argument("--sim-setup", default="/root/kuavo_ws/devel/setup.bash")
    p.add_argument("--xvfb-display", help="Start an owned Xvfb process in the simulator container, for example :99")
    p.set_defaults(function=coordinate)
    p = commands.add_parser("summarize")
    p.add_argument("directory")
    p.set_defaults(function=lambda a: print(json.dumps(summarize(a.directory), indent=2)))
    p = commands.add_parser("_policy_worker")
    p.add_argument("--manifest", required=True)
    p.add_argument("--policy-repository", required=True)
    p.set_defaults(function=policy_worker)
    p = commands.add_parser("_sim_worker")
    p.add_argument("--manifest", required=True)
    p.add_argument("--sim-workspace", required=True)
    p.add_argument("--sim-setup", required=True)
    p.add_argument("--sim-python", required=True)
    p.add_argument("--sim-startup-delay", type=float, default=5)
    p.add_argument("--xvfb-display")
    p.set_defaults(function=simulator_worker)
    p = commands.add_parser("_stop_worker")
    p.add_argument("--manifest", required=True)
    p.add_argument("--worker", choices=["policy", "simulator"], required=True)
    p.set_defaults(function=stop_worker)
    p = commands.add_parser("_evaluator_bootstrap")
    p.add_argument("--script", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--task1-control")
    p.add_argument("--control-receipt")
    p.set_defaults(function=evaluator_bootstrap)
    p = commands.add_parser("_sim_readiness")
    p.add_argument("--manifest", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.set_defaults(function=simulator_readiness)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
