#!/usr/bin/env python3
"""Fresh, controlled ACT experiments. Importing this module does not import torch.

Use the full LeRobot dataset and global-index Subsets: the pinned LeRobot reader's
temporal queries use global episode bounds, so filtering its episodes is unsafe.
Fresh experiments are the default. Optional continuation loads only matching
recovery-generated weights, optimizer, schedule, scaler, sampler and RNG state.
All checkpoints carry processors.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
RGB_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
)
JOINT_INDICES = tuple(range(7)) + tuple(range(8, 15))
GRIPPER_INDICES = (7, 15)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_contract(source):
    source = Path(source).resolve()
    paths = (
        "kuavo_train/wrapper/policy/act/ACTConfigWrapper.py",
        "kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py",
        "kuavo_train/wrapper/policy/act/ACTModelWrapper.py",
        "kuavo_train/utils/transforms.py",
        "lerobot_patches/custom_patches.py",
        "third_party/lerobot/src/lerobot/datasets/lerobot_dataset.py",
        "third_party/lerobot/src/lerobot/policies/act/modeling_act.py",
        "third_party/lerobot/src/lerobot/policies/act/processor_act.py",
        "third_party/lerobot/src/lerobot/processor/normalize_processor.py",
    )
    try:
        result = subprocess.run(["git", "-c", f"safe.directory={source}", "-C", str(source),
                                 "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        head = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        head = None
    return {"git_head": head,
            "source_sha256": {name: sha256_file(source / name) for name in paths}}


def load_manifest(path):
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("task") not in ("Task1_ToySorting", "Task2_ParcelWeighing", "Task3_ConveyorBeltSorting"):
        raise ValueError("This RGB ACT driver supports the audited Task1 and Task2 manifests")
    lists = {}
    for key in ("train_episodes", "validation_episodes", "small_subset_episodes"):
        values = manifest.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty episode list")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"Invalid episode IDs in {key}")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate episodes in {key}")
        lists[key] = set(values)
    if lists["train_episodes"] & lists["validation_episodes"]:
        raise ValueError("Training and validation episodes overlap")
    if not lists["small_subset_episodes"] <= lists["train_episodes"]:
        raise ValueError("Small overfit subset must be contained in training episodes")
    if len(lists["train_episodes"]) != 900 or len(lists["validation_episodes"]) != 100:
        raise ValueError("The controlled split must contain 900/100 episodes")
    if lists["train_episodes"] | lists["validation_episodes"] != set(range(1000)):
        raise ValueError("The split must partition all 1000 episodes")
    if not 10 <= len(lists["small_subset_episodes"]) <= 20:
        raise ValueError("The overfit subset must contain 10–20 complete episodes")
    for key, expected in (("original_stats_episodes", lists["train_episodes"]),
                          ("repaired_stats_episodes", lists["train_episodes"]),
                          ("overfit_stats_episodes", lists["small_subset_episodes"])):
        values = manifest.get(key)
        if not isinstance(values, list) or len(values) != len(expected) or set(values) != expected:
            raise ValueError(f"Statistics provenance does not match the selected episodes: {key}")
    for key in ("original_root", "corrected_root", "original_train_stats", "repaired_train_stats",
                "original_overfit_stats", "repaired_overfit_stats"):
        if key in manifest:
            candidate = Path(manifest[key])
            manifest[key] = str(candidate if candidate.is_absolute() else path.parent / candidate)
    for key in ("original_root", "corrected_root", "original_train_stats", "repaired_train_stats"):
        if key not in manifest or not Path(manifest[key]).exists():
            raise ValueError(f"Missing prepared artifact: {key}")
    original = read_json(manifest["original_train_stats"])
    repaired = read_json(manifest["repaired_train_stats"])
    if set(original) != set(repaired):
        raise ValueError("A/B statistics do not have identical feature keys")
    for key in original:
        if key not in RGB_KEYS and original[key] != repaired[key]:
            raise ValueError(f"A/B changes a non-RGB statistic: {key}")
    for key in RGB_KEYS:
        if key not in original:
            raise ValueError(f"Missing RGB statistics: {key}")
    if not any(original[key]["std"] != repaired[key]["std"] for key in RGB_KEYS):
        raise ValueError("The repaired RGB standard deviations do not differ from the control")
    manifest["_path"] = str(path)
    return manifest


def indices_for_episodes(episode_rows, episode_ids, samples_per_episode=None):
    """Return original GLOBAL frame indices, never filtered-reader row indices."""
    indices = []
    sampled_by_episode = []
    for episode_id in sorted(episode_ids):
        row = episode_rows[episode_id]
        if int(row["episode_index"]) != episode_id:
            raise ValueError("Episode metadata must use contiguous, ordered IDs")
        start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        if end <= start or int(row["length"]) != end - start:
            raise ValueError(f"Invalid episode bounds: {episode_id}")
        if samples_per_episode is None or samples_per_episode >= end - start:
            indices.extend(range(start, end))
        else:
            if samples_per_episode < 2:
                raise ValueError("At least two validation frames per episode are required")
            sampled_by_episode.append([round(start + i * (end - start - 1) / (samples_per_episode - 1))
                                       for i in range(samples_per_episode)])
    if sampled_by_episode:
        # A bounded smoke evaluation must sample several episodes and trajectory
        # positions, rather than spending its whole prefix on episode0's start.
        for offset in range(samples_per_episode):
            for episode_position, samples in enumerate(sampled_by_episode):
                indices.append(samples[(episode_position + offset) % len(samples)])
    return indices


def import_runtime(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "third_party/lerobot/src"))
    import lerobot_patches.custom_patches  # FeatureType extensions must precede policy imports.
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import make_pre_post_processors
    from kuavo_train.wrapper.policy.act.ACTConfigWrapper import CustomACTConfigWrapper
    from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
    return locals()


def configure_policy(runtime, reference, metadata, device, amp):
    path = Path(reference).resolve()
    if path.name != "config.json":
        raise ValueError("Policy reference must be a saved config.json, not weights")
    config = runtime["CustomACTConfigWrapper"].from_pretrained(path.parent)
    if read_json(path).get("type") != "custom_act":
        raise ValueError("Expected the historical custom_act reference")
    features = runtime["dataset_to_policy_features"](metadata.features)
    config.input_features = {key: features[key] for key in ("observation.state", *RGB_KEYS)}
    config.output_features = {"action": features["action"]}
    config.device = device
    config.use_amp = bool(amp and device.startswith("cuda"))
    config.push_to_hub = False
    config.custom["use_depth"] = False
    config.use_depth = False
    if config.chunk_size != 40 or config.n_action_steps != 1 or config.temporal_ensemble_coeff != 0.01:
        raise ValueError("The reference must retain chunk40, one action, temporal coefficient .01")
    if tuple(config.output_features["action"].shape) != (16,):
        raise ValueError("ACT requires the audited 16-dimensional action contract")
    return config


def prepare_data(runtime, manifest, variant, mode, config, validation_frames):
    root = manifest["original_root" if variant == "original" else "corrected_root"]
    selected = manifest["small_subset_episodes" if mode == "overfit" else "train_episodes"]
    stats_key = ("original" if variant == "original" else "repaired") + (
        "_overfit_stats" if mode == "overfit" and "original_overfit_stats" in manifest
        and "repaired_overfit_stats" in manifest else "_train_stats")
    stats = read_json(manifest[stats_key])
    # This full reader retains global frame indexing for future action chunks.
    dataset = runtime["LeRobotDataset"](
        repo_id=f"lerobot/{manifest['task']}", root=root, episodes=None,
        delta_timestamps={"action": [index / 10 for index in range(config.chunk_size)]},
        image_transforms=None, video_backend="pyav", download_videos=False,
    )
    if dataset.fps != 10 or dataset.meta.total_episodes != 1000:
        raise ValueError("Expected the audited 1000-episode, 10 Hz dataset")
    if set(dataset.meta.camera_keys) != set(RGB_KEYS):
        raise ValueError("Use the prepared RGB-only dataset views to avoid decoding unused depth")
    rows = dataset.meta.episodes
    train_indices = indices_for_episodes(rows, selected)
    validation_indices = indices_for_episodes(rows, manifest["validation_episodes"], validation_frames)
    if set(train_indices) & set(validation_indices):
        raise ValueError("Global training/validation frame indices overlap")
    stats_episodes = selected if "_overfit_stats" in stats_key else manifest["train_episodes"]
    stats_frames = sum(int(rows[episode]["length"]) for episode in stats_episodes)
    if int(stats["action"]["count"][0]) != stats_frames:
        raise ValueError("Selected training-only statistics have an incorrect frame count")
    return dataset, train_indices, validation_indices, stats, stats_key


def seed_everything(torch, np, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_augmentation(runtime, configuration):
    if configuration is None:
        return None
    from kuavo_train.utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig
    return ImageTransforms(ImageTransformsConfig(
        enable=True, max_num_transforms=configuration["max_num_transforms"],
        random_order=configuration["random_order"],
        tfs={name: ImageTransformConfig(**definition) for name, definition in configuration["tfs"].items()},
    ))


def process_batch(torch, batch, preprocessor, device, augmentation=None):
    # Original training augments RGB after device placement and before normalization.
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    if augmentation is not None:
        batch = dict(batch)
        for key in RGB_KEYS:
            batch[key] = augmentation(batch[key])
    return preprocessor(batch)


def accumulation_groups(loader, size):
    iterator = iter(loader)
    while True:
        group = list(itertools.islice(iterator, size))
        if not group:
            return
        yield group


def capture_training_state(torch, np, policy, device):
    """Save forward randomness and buffers before a possibly skipped AMP step."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(),
             "buffers": {name: value.detach().clone() for name, value in policy.named_buffers()}}
    if device.startswith("cuda"):
        state["cuda"] = torch.cuda.get_rng_state(torch.device(device))
    return state


def restore_training_state(torch, np, policy, device, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], torch.device(device))
    with torch.no_grad():
        for name, value in policy.named_buffers():
            value.copy_(state["buffers"][name])


def optimize_group(runtime, policy, raw_group, preprocessor, device, augmentation,
                   amp, scaler, optimizer, skip_budget):
    """Retry the same samples/random draws after overflow, keeping the lower scale."""
    torch, np = runtime["torch"], runtime["np"]
    sample_count = sum(len(raw["action"]) for raw in raw_group)
    state = capture_training_state(torch, np, policy, device)
    randomness = hashlib.sha256(state["torch"].numpy().tobytes())
    randomness.update(repr(state["python"]).encode())
    randomness.update(state["numpy"][1].tobytes())
    randomness.update(repr((state["numpy"][0], *state["numpy"][2:])).encode())
    if "cuda" in state:
        randomness.update(state["cuda"].cpu().numpy().tobytes())
    skipped = 0
    while True:
        if skipped:
            restore_training_state(torch, np, policy, device, state)
        optimizer.zero_grad(set_to_none=True)
        metrics = {"loss": 0.0, "l1_loss": 0.0, "kld_loss": 0.0}
        for raw in raw_group:
            weight = len(raw["action"]) / sample_count
            batch = process_batch(torch, raw, preprocessor, device, augmentation)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp):
                loss, losses = policy.forward(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss in an accumulation group")
            scaler.scale(loss * weight).backward()
            metrics["loss"] += float(loss.detach()) * weight
            for key in ("l1_loss", "kld_loss"):
                metrics[key] += float(losses.get(key, 0.0)) * weight
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= old_scale:
            return metrics, sample_count, skipped, randomness.hexdigest()
        skipped += 1
        if skipped > skip_budget:
            raise RuntimeError("AMP overflow retry budget exhausted; investigate scaling")


def validate(runtime, policy, preprocessor, postprocessor, loader, device, max_batches=None,
             posterior_loss=True):
    torch = runtime["torch"]
    policy.eval()
    dim_error = torch.zeros(16, dtype=torch.float64, device=device)
    t0_error = torch.zeros(16, dtype=torch.float64, device=device)
    norm_error = 0.0
    posterior_totals = {"act_loss": 0.0, "l1_loss": 0.0, "kld_loss": 0.0}
    valid_count = frame_count = grip_correct = chunk_grip_correct = 0
    devices = [torch.device(device).index if torch.device(device).index is not None
               else torch.cuda.current_device()] if device.startswith("cuda") else []
    # DataLoader iteration also consumes torch RNG for its worker seed. Preserve
    # the whole evaluation's RNG state, not only the posterior sampling call.
    with torch.inference_mode(), torch.random.fork_rng(devices=devices):
        for batch_index, raw in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = process_batch(torch, raw, preprocessor, device)
            # eval forward() cannot compute KL: this ACT model returns None for its
            # posterior parameters in deployment mode. Predict with the zero prior.
            normalized = policy.predict_action_chunk(batch)
            if not torch.isfinite(normalized).all():
                raise RuntimeError("Validation predicted nonfinite actions")
            valid = ~batch["action_is_pad"]
            difference = (normalized - batch["action"]).abs()
            norm_error += float((difference * valid.unsqueeze(-1)).sum().item())
            predicted = postprocessor(normalized).to(device)
            truth = raw["action"].to(device)
            physical_error = (predicted - truth).abs()
            dim_error += (physical_error * valid.unsqueeze(-1)).sum((0, 1)).double()
            t0_error += physical_error[:, 0].sum(0).double()
            valid_count += int(valid.sum().item())
            frame_count += len(truth)
            grip_correct += int(((predicted[:, 0, GRIPPER_INDICES] >= .5)
                                 == (truth[:, 0, GRIPPER_INDICES] >= .5)).sum().item())
            chunk_grip_correct += int((((predicted[:, :, GRIPPER_INDICES] >= .5)
                == (truth[:, :, GRIPPER_INDICES] >= .5)) * valid.unsqueeze(-1)).sum().item())
            if posterior_loss:
                # The model's top-level flag enables its VAE posterior; child
                # modules remain in eval mode. This preserves dropout/backbone
                # state, unlike calling policy.train() during validation.
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(2025 + batch_index)
                    policy.model.training = bool(policy.config.use_vae)
                    try:
                        act_loss, loss_terms = policy.forward(batch)
                    finally:
                        policy.model.training = False
                if not torch.isfinite(act_loss):
                    raise RuntimeError("Nonfinite posterior ACT validation loss")
                posterior_totals["act_loss"] += float(act_loss) * len(truth)
                for key in ("l1_loss", "kld_loss"):
                    posterior_totals[key] += float(loss_terms.get(key, 0.0)) * len(truth)
    if not valid_count or not frame_count:
        raise ValueError("Validation produced no observations")
    all_mae = (dim_error / valid_count).cpu().tolist()
    current_mae = (t0_error / frame_count).cpu().tolist()
    result = {
        "observations": frame_count, "valid_chunk_actions": valid_count,
        "chunk_l1_normalized": norm_error / (valid_count * 16),
        "joint_mae_rad": sum(current_mae[i] for i in JOINT_INDICES) / 14,
        "gripper_mae_unit": sum(current_mae[i] for i in GRIPPER_INDICES) / 2,
        "gripper_threshold_accuracy": grip_correct / (frame_count * 2),
        "chunk_joint_mae_rad": sum(all_mae[i] for i in JOINT_INDICES) / 14,
        "chunk_gripper_mae_unit": sum(all_mae[i] for i in GRIPPER_INDICES) / 2,
        "chunk_gripper_threshold_accuracy": chunk_grip_correct / (valid_count * 2),
        "current_action_mae_per_dimension": current_mae,
        "chunk_action_mae_per_dimension": all_mae,
        "current_action_selection_score": sum(current_mae[i] / max(
            float(preprocessor.steps[-1].stats["action"]["std"][i]), 1e-8)
            for i in range(16)) / 16,
        "prediction_mode": "deployment_zero_vae_prior; no temporal ensembling offline",
    }
    if posterior_loss:
        result["vae_posterior_validation"] = {
            key: value / frame_count for key, value in posterior_totals.items()}
        result["vae_posterior_validation"]["mode"] = (
            "deterministic sampled posterior; child modules eval; historical padding denominator")
    return result


def save_checkpoint(path, policy, preprocessor, postprocessor, metrics):
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    write_json(path / "validation_metrics.json", metrics)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=WORKSPACE / "recovery/datasets/task1_experiment_manifest.json")
    result.add_argument("--source", type=Path, default=WORKSPACE / "kuavo_data_challenge")
    result.add_argument("--policy-config", type=Path, default=WORKSPACE / "recovery/experiments/act_task1_reference/config.json")
    result.add_argument("--variant", choices=("original", "repaired"), required=True)
    result.add_argument("--mode", choices=("overfit", "ab"), default="overfit")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, default=2025)
    result.add_argument("--epochs", type=int, default=4)
    result.add_argument("--max-steps", type=int, help="Successful optimizer updates, not microbatches")
    result.add_argument("--schedule-steps", type=int,
                        help="Planned cosine schedule length; can exceed this run's update budget")
    result.add_argument("--checkpoint-every-steps", type=int,
                        help="Save full continuation state at this successful-update interval")
    result.add_argument("--resume-state", type=Path,
                        help="Continue a recovery-generated training_state.pt in a fresh output directory")
    result.add_argument("--microbatch", type=int, default=4)
    result.add_argument("--accumulation", type=int, default=8)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--device", default="cuda")
    result.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--warmup-steps", type=int, default=500)
    result.add_argument("--validation-frames-per-episode", type=int, default=20)
    result.add_argument("--max-validation-batches", type=int)
    result.add_argument("--max-diagnostic-batches", type=int,
                        help="Optional cap for the 20-episode training diagnostic; default evaluates all probes")
    result.add_argument("--validation-posterior-loss", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--augmentation-config", type=Path, default=WORKSPACE / "recovery/experiments/act_task1_augmentation.json")
    result.add_argument("--check-contract", action="store_true", help="Check JSON contracts only; no ML imports")
    return result


def main():
    args = parser().parse_args()
    stop_requested = False
    def requested_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, requested_stop)
    signal.signal(signal.SIGTERM, requested_stop)
    def check_training_stop():
        if stop_requested:
            raise KeyboardInterrupt("Training stop requested at a consistent optimizer boundary")
    manifest = load_manifest(args.manifest)
    if args.check_contract:
        print(json.dumps({"train_episodes": 900, "validation_episodes": 100,
                          "overfit_episodes": len(manifest["small_subset_episodes"]),
                          "non_rgb_statistics_identical": True}, indent=2))
        return
    if min(args.epochs, args.microbatch, args.accumulation) < 1:
        raise ValueError("Epoch, microbatch and accumulation settings must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.schedule_steps is not None and args.schedule_steps < 1:
        raise ValueError("schedule-steps must be positive")
    if args.checkpoint_every_steps is not None and args.checkpoint_every_steps < 1:
        raise ValueError("checkpoint-every-steps must be positive")
    if args.resume_state and args.workers != 0:
        raise ValueError("Strict training continuation currently requires workers=0")
    if any(value is not None and value < 1 for value in
           (args.max_validation_batches, args.max_diagnostic_batches)):
        raise ValueError("Evaluation batch caps must be positive")
    if args.mode == "overfit" and args.max_steps is None:
        raise ValueError("Overfit mode needs an explicit --max-steps; start with 500")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be fresh and empty: {output}")
    runtime = import_runtime(args.source)
    torch, np = runtime["torch"], runtime["np"]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; complete the GPU preflight first")
    seed_everything(torch, np, args.seed)
    root = manifest["original_root" if args.variant == "original" else "corrected_root"]
    metadata = runtime["LeRobotDatasetMetadata"](f"lerobot/{manifest['task']}", root=root)
    config = configure_policy(runtime, args.policy_config, metadata, args.device, args.amp)
    dataset, train_indices, validation_indices, stats, stats_key = prepare_data(
        runtime, manifest, args.variant, args.mode, config, args.validation_frames_per_episode)
    policy = runtime["CustomACTPolicyWrapper"](config)
    initialization_hash = model_digest(policy)
    policy.to(args.device)
    preprocessor, postprocessor = runtime["make_pre_post_processors"](config, dataset_stats=stats)
    optimizer = config.get_optimizer_preset().build(policy.parameters())
    batches_per_epoch = math.ceil(len(train_indices) / args.microbatch)
    updates_per_epoch = math.ceil(batches_per_epoch / args.accumulation)
    total_updates = args.max_steps or args.epochs * updates_per_epoch
    schedule_updates = args.schedule_steps or total_updates
    if schedule_updates < total_updates:
        raise ValueError("The learning-rate schedule cannot be shorter than the update budget")
    epoch_limit = max(args.epochs, math.ceil((total_updates + 50) / updates_per_epoch)) if args.max_steps else args.epochs
    from diffusers.optimization import get_scheduler
    scheduler = get_scheduler("cosine", optimizer=optimizer,
                              num_warmup_steps=min(args.warmup_steps, max(1, schedule_updates // 10)),
                              num_training_steps=schedule_updates)
    scaler_type = torch.amp.GradScaler if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler
    scaler = scaler_type("cuda", enabled=config.use_amp) if scaler_type is getattr(getattr(torch, "amp", None), "GradScaler", None) else scaler_type(enabled=config.use_amp)
    augmentation = build_augmentation(runtime, read_json(args.augmentation_config)) if args.mode == "ab" else None
    from training_resume import save_state, load_state, restore_rng
    continuation_contract = {
        "task": manifest["task"],
        "variant": args.variant, "mode": args.mode, "seed": args.seed,
        "microbatch": args.microbatch, "accumulation": args.accumulation, "amp": config.use_amp,
        "manifest_sha256": sha256_file(args.manifest), "stats_sha256": sha256_file(manifest[stats_key]),
        "policy_config_sha256": sha256_file(args.policy_config),
        "source_sha256": source_contract(args.source)["source_sha256"],
        "schedule_updates": schedule_updates,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "warmup_updates": min(args.warmup_steps, max(1, schedule_updates // 10)),
        "validation_frames_per_episode": args.validation_frames_per_episode,
        "max_validation_batches": args.max_validation_batches,
        "max_diagnostic_batches": args.max_diagnostic_batches,
        "validation_posterior_loss": args.validation_posterior_loss,
        "augmentation": read_json(args.augmentation_config) if augmentation else None,
        "runner_sha256": sha256_file(__file__),
        "training_resume_sha256": sha256_file(Path(__file__).with_name("training_resume.py")),
        "sampler_resume_sha256": sha256_file(Path(__file__).with_name("sampler_resume.py")),
    }
    resumed = load_state(torch, args.resume_state, continuation_contract) if args.resume_state else None
    if resumed:
        if total_updates <= resumed["global_step"]:
            raise ValueError("Continuation budget must exceed the saved global update count")
        from safetensors.torch import load_file
        policy.load_state_dict(load_file(str(Path(resumed["model_checkpoint"]) / "model.safetensors")), strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
    from torch.utils.data import DataLoader, Subset
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=args.microbatch,
                                   shuffle=False, num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "training_resume.py").write_bytes(Path(__file__).with_name("training_resume.py").read_bytes())
    (output / "sampler_resume.py").write_bytes(Path(__file__).with_name("sampler_resume.py").read_bytes())
    write_json(output / "worker_identity.json", {
        "pid": os.getpid(), "run_id": os.environ.get("LEJU_RECOVERY_TRAIN_ID"),
        "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
        "output": str(output),
    })
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)
    receipt = {
        "task": manifest["task"],
        "variant": args.variant, "mode": args.mode, "seed": args.seed,
        "train_episode_ids": manifest["small_subset_episodes" if args.mode == "overfit" else "train_episodes"],
        "validation_episode_ids": manifest["validation_episodes"],
        "selected_train_frames": len(train_indices), "validation_frames": len(validation_indices),
        "microbatch": args.microbatch, "accumulation": args.accumulation,
        "effective_batch": args.microbatch * args.accumulation,
        "planned_optimizer_updates": total_updates, "updates_per_epoch": updates_per_epoch,
        "planned_schedule_updates": schedule_updates,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "maximum_epochs": epoch_limit,
        "warmup_updates": min(args.warmup_steps, max(1, schedule_updates // 10)),
        "initial_weights_sha256": initialization_hash,
        "policy_config_sha256": sha256_file(args.policy_config),
        "stats_file": manifest[stats_key], "stats_sha256": sha256_file(manifest[stats_key]),
        "manifest_sha256": sha256_file(args.manifest), "runner_sha256": sha256_file(__file__),
        "source": str(args.source.resolve()), "dataset_root": root,
        "source_contract": source_contract(args.source),
        "augmentation": read_json(args.augmentation_config) if augmentation else None,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
        "amp": config.use_amp, "fresh_training": resumed is None,
        "resume_state": str(args.resume_state.resolve()) if args.resume_state else None,
        "continuation_state": "training_state.pt contains optimizer/scheduler/scaler/RNG and sampler position",
        "max_validation_batches": args.max_validation_batches,
        "max_diagnostic_batches": args.max_diagnostic_batches,
        "validation_posterior_loss": args.validation_posterior_loss,
        "checkpoint_selection": "held-out current-action MAE divided by training action std",
        "overfit_uses_900_training_episode_stats": args.mode == "overfit" and "_overfit_stats" not in stats_key,
        "amp_skip_policy": "Replay identical accumulation group and random draws; restore model buffers; retain lowered scaler",
        "sampler_resume": "Skip consumed batch indices before dataset decoding; preserve DataLoader and RandomSampler generator order",
        "sampler_resume_sha256": continuation_contract["sampler_resume_sha256"],
    }
    write_json(output / "experiment_receipt.json", receipt)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(output))
    global_step = skipped_updates = 0
    consumed_frames = 0
    sample_digest, randomness_digest = hashlib.sha256(), hashlib.sha256()
    best_score = math.inf
    best_checkpoint = None
    epoch_history = []
    frame_history, randomness_history = bytearray(), bytearray()
    resume_epoch = resume_group = 0
    if resumed:
        global_step = resumed["global_step"]
        auxiliary = resumed["auxiliary"]
        skipped_updates, consumed_frames = auxiliary["skipped_updates"], auxiliary["consumed_frames"]
        best_score, epoch_history = auxiliary["best_score"], auxiliary["epoch_history"]
        best_checkpoint = auxiliary.get("best_checkpoint")
        frame_history.extend(auxiliary["frame_history"])
        randomness_history.extend(auxiliary["randomness_history"])
        sample_digest.update(frame_history)
        randomness_digest.update(randomness_history)
        resume_epoch, resume_group = resumed["next_epoch"], resumed["next_group"]
        # A stop during initial validation or before the first new update must
        # export the saved randomness and cursor, not initialization randomness.
        restore_rng(torch, np, args.device, resumed["rng"])
    current_epoch = resume_epoch
    next_group = resume_group
    epoch_loss_sum = resumed["auxiliary"]["epoch_loss_sum"] if resumed and resume_group else 0
    epoch_samples = resumed["auxiliary"]["epoch_samples"] if resumed and resume_group else 0
    def save_continuation(checkpoint, epoch, group):
        save_state(torch, np, output / "training_state.pt", checkpoint, optimizer, scheduler, scaler,
                   args.device, continuation_contract, global_step, epoch, group, {
                       "skipped_updates": skipped_updates, "consumed_frames": consumed_frames,
                       "best_score": best_score, "epoch_history": epoch_history,
                       "best_checkpoint": best_checkpoint,
                       "frame_history": bytes(frame_history), "randomness_history": bytes(randomness_history),
                       "epoch_loss_sum": epoch_loss_sum if group else 0,
                       "epoch_samples": epoch_samples if group else 0})
    started = time.time()
    try:
        initial_metrics = {"validation": validate(runtime, policy, preprocessor, postprocessor,
            validation_loader, args.device, args.max_validation_batches, args.validation_posterior_loss)}
        diagnostic_loader = None
        if args.mode == "overfit":
            diagnostic_indices = indices_for_episodes(dataset.meta.episodes,
                manifest["small_subset_episodes"], 10)
            diagnostic_loader = DataLoader(Subset(dataset, diagnostic_indices), batch_size=args.microbatch,
                shuffle=False, num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
            initial_metrics["overfit_train_deployment_metrics"] = validate(runtime, policy, preprocessor,
                postprocessor, diagnostic_loader, args.device, args.max_diagnostic_batches,
                args.validation_posterior_loss)
        write_json(output / "initial_deployment_metrics.json", initial_metrics)
        print(json.dumps({"initial_deployment_metrics": initial_metrics}), flush=True)
        for epoch in range(resume_epoch, epoch_limit):
            check_training_stop()
            current_epoch = epoch
            policy.train()
            from sampler_resume import training_loader
            starting_group = resume_group if epoch == resume_epoch else 0
            loader = training_loader(torch, Subset(dataset, train_indices), args.seed + epoch,
                args.microbatch, args.workers, args.device.startswith("cuda"),
                args.accumulation, starting_group)
            epoch_loss_sum = resumed["auxiliary"]["epoch_loss_sum"] if resumed and epoch == resume_epoch and resume_group else 0
            epoch_samples = resumed["auxiliary"]["epoch_samples"] if resumed and epoch == resume_epoch and resume_group else 0
            next_group = starting_group
            for group_index, raw_group in enumerate(accumulation_groups(loader, args.accumulation), start=starting_group):
                check_training_stop()
                group_metrics, group_samples, group_skips, random_hash = optimize_group(
                    runtime, policy, raw_group, preprocessor, args.device, augmentation,
                    config.use_amp, scaler, optimizer, 50 - skipped_updates)
                skipped_updates += group_skips
                epoch_loss_sum += group_metrics["loss"] * group_samples
                epoch_samples += group_samples
                consumed_frames += group_samples
                for raw in raw_group:
                    indices_bytes = raw["index"].detach().cpu().numpy().astype("<i8").tobytes()
                    sample_digest.update(indices_bytes)
                    frame_history.extend(indices_bytes)
                random_bytes = bytes.fromhex(random_hash)
                randomness_digest.update(random_bytes)
                randomness_history.extend(random_bytes)
                scheduler.step()
                global_step += 1
                next_group = group_index + 1
                check_training_stop()
                for key, value in group_metrics.items():
                    writer.add_scalar("train/" + key, value, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                if global_step == 1 or global_step % 25 == 0:
                    print(json.dumps({"epoch": epoch + 1, "optimizer_step": global_step,
                                      "train": group_metrics, "lr": scheduler.get_last_lr()[0],
                                      "skipped_amp_updates": skipped_updates}), flush=True)
                    write_json(output / "progress.json", {"epoch": epoch + 1, "optimizer_step": global_step,
                        "planned_optimizer_updates": total_updates, "planned_schedule_updates": schedule_updates,
                        "learning_rate": scheduler.get_last_lr()[0], "train": group_metrics,
                        "elapsed_seconds": time.time() - started, "updated_at_unix": time.time()})
                if global_step >= total_updates:
                    break
                if args.checkpoint_every_steps and global_step % args.checkpoint_every_steps == 0:
                    checkpoint = output / f"step{global_step}"
                    save_checkpoint(checkpoint, policy, preprocessor, postprocessor,
                                    {"optimizer_step": global_step, "intermediate_checkpoint": True})
                    save_continuation(checkpoint, epoch, next_group)
            metrics = validate(runtime, policy, preprocessor, postprocessor, validation_loader,
                               args.device, args.max_validation_batches, args.validation_posterior_loss)
            metrics.update(epoch=epoch + 1, optimizer_step=global_step,
                           train_loss=epoch_loss_sum / epoch_samples, skipped_amp_updates=skipped_updates)
            if args.mode == "overfit":
                metrics["overfit_train_deployment_metrics"] = validate(runtime, policy, preprocessor,
                    postprocessor, diagnostic_loader, args.device, args.max_diagnostic_batches,
                    args.validation_posterior_loss)
            epoch_history.append(metrics)
            for key in ("chunk_l1_normalized", "joint_mae_rad", "gripper_mae_unit", "gripper_threshold_accuracy"):
                writer.add_scalar("validation/" + key, metrics[key], global_step)
            save_checkpoint(output / f"epoch{epoch + 1}", policy, preprocessor, postprocessor, metrics)
            if metrics["current_action_selection_score"] < best_score:
                best_score = metrics["current_action_selection_score"]
                save_checkpoint(output / "epochbest", policy, preprocessor, postprocessor, metrics)
                best_checkpoint = str((output / "epochbest").resolve())
            save_checkpoint(output, policy, preprocessor, postprocessor, metrics)
            # The mutable root export must not invalidate the previous resume state.
            immutable_checkpoint = output / f"epoch{epoch + 1}"
            if next_group >= updates_per_epoch:
                save_continuation(immutable_checkpoint, epoch + 1, 0)
                current_epoch, next_group = epoch + 1, 0
            else:
                save_continuation(immutable_checkpoint, epoch, next_group)
            write_json(output / "epoch_history.json", epoch_history)
            print(json.dumps({"validation": metrics}), flush=True)
            check_training_stop()
            if global_step >= total_updates:
                break
    except KeyboardInterrupt:
        save_checkpoint(output / "interrupted", policy, preprocessor, postprocessor,
                        {"optimizer_step": global_step, "interrupted": True})
        save_continuation(output / "interrupted", current_epoch, next_group)
        raise
    finally:
        writer.close()
        if best_checkpoint:
            write_json(output / "best_checkpoint_reference.json", {
                "checkpoint": best_checkpoint, "selection_score": best_score,
                "model_sha256": sha256_file(Path(best_checkpoint) / "model.safetensors")})
        write_json(output / "run_status.json", {"optimizer_steps": global_step,
                   "planned_optimizer_updates": total_updates, "completed_budget": global_step >= total_updates,
                   "planned_schedule_updates": schedule_updates,
                   "final_learning_rates": scheduler.get_last_lr(),
                   "training_state": "training_state.pt",
                   "interrupted": stop_requested,
                   "skipped_amp_updates": skipped_updates, "elapsed_seconds": time.time() - started,
                   "consumed_training_frames": consumed_frames,
                   "training_frame_order_sha256": sample_digest.hexdigest(),
                   "training_randomness_stream_sha256": randomness_digest.hexdigest(),
                   "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated()
                   if args.device.startswith("cuda") else None})


if __name__ == "__main__":
    main()
