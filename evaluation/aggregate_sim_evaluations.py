#!/usr/bin/env python3
"""Assemble completed seeds from interrupted runs, preserving source receipts.

This creates an analysis artifact, not a new simulator run. Incomplete episodes
are excluded explicitly and never assigned a zero score. Source files stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from run_sim_evaluation import summarize, simulator_score_errors


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def link(source, target):
    target.symlink_to(os.path.relpath(source, target.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=44)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--failed-attempts", type=Path,
                        help="Evidence records for controller exits; never replace them with zero scores")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a fresh analysis directory")
    parts = [p.resolve() for p in args.part]
    manifests = [read(p / "manifest.json") for p in parts]
    reference = manifests[0]
    for manifest in manifests:
        for key in ("task_id", "checkpoint_files_sha256", "max_episode_steps", "ros_rate",
                    "robot_version", "sim_environment", "video_stride", "checkpoint_host"):
            if manifest[key] != reference[key]:
                raise ValueError(f"Part configurations differ: {key}")
        if manifest["record_episodes"] != list(range(len(manifest["seeds"]))):
            raise ValueError("Every episode must be recorded for this aggregation")
    selected, excluded = {}, []
    for part, manifest in zip(parts, manifests):
        ready = read(part / "policy_ready.json")
        if ready["checkpoint_files_sha256"] != reference["checkpoint_files_sha256"]:
            raise ValueError("Loaded checkpoint hashes differ")
        for index, seed in enumerate(manifest["seeds"]):
            pp, sp = part / f"policy_episode_{index:04d}.json", part / f"simulator_episode_{index:04d}.json"
            if not pp.exists() or not sp.exists():
                excluded.append({"run": str(part), "seed": seed, "reason": "Missing completed policy/simulator receipt"})
                continue
            policy, simulator = read(pp), read(sp)
            errors = simulator_score_errors(simulator)
            if policy.get("failure") or policy.get("reset_failure"):
                errors.append("Policy/reset failure")
            if policy["seed"] != seed or simulator["seed"] != seed:
                errors.append("Seed mismatch")
            if errors:
                excluded.append({"run": str(part), "seed": seed, "reason": errors})
                continue
            if seed in selected:
                raise ValueError("Duplicate completed seed; do not cherry-pick outcomes")
            selected[seed] = (part, manifest, index)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    failed = {}
    for record in read(args.failed_attempts) if args.failed_attempts else []:
        if record["seed"] in selected:
            continue  # A separately reported extra attempt was repeated once.
        evidence = Path(record["evidence"])
        if hashlib.sha256(evidence.read_bytes()).hexdigest() != record["evidence_sha256"]:
            raise ValueError("Controller-exit evidence changed")
        if record["status"] != "controller_exit" or f"Simulator exited during seed {record['seed']}" not in evidence.read_text():
            raise ValueError("Failed-seed record is not supported by its simulator log")
        failed[record["seed"]] = record
    if set(selected) | set(failed) != set(seeds):
        raise ValueError(f"Completed seeds differ from requested cohort: {sorted(selected)}")
    output.mkdir()
    (output / "scores").mkdir()
    (output / "videos").mkdir()
    provenance = []
    for combined_index, seed in enumerate(seeds):
        if seed in failed:
            write(output / f"failed_episode_{combined_index:04d}.json", failed[seed])
            continue
        part, manifest, source_index = selected[seed]
        record = {"seed": seed, "combined_index": combined_index, "source_run": str(part),
                  "source_run_id": manifest["run_id"], "source_index": source_index, "files": {}}
        for stem in ("policy_episode", "simulator_episode", "scene_episode"):
            source = part / f"{stem}_{source_index:04d}.json"
            if stem == "scene_episode" and reference["task_id"] == 2 and not source.exists():
                continue
            value = read(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            value["index"] = combined_index
            value["source_receipt"] = {"path": str(source), "sha256": digest,
                "source_run_id": manifest["run_id"], "source_index": source_index}
            write(output / f"{stem}_{combined_index:04d}.json", value)
            record["files"][stem] = value["source_receipt"]
        link(part / f"evaluator_{source_index:04d}.log", output / f"evaluator_{combined_index:04d}.log")
        for extension in ("json", "txt"):
            link(part / "scores" / f"episode_{source_index:04d}_seed{seed}.{extension}",
                 output / "scores" / f"episode_{combined_index:04d}_seed{seed}.{extension}")
        link(part / "videos" / f"episode_{source_index:04d}_seed{seed}",
             output / "videos" / f"episode_{combined_index:04d}_seed{seed}")
        provenance.append(record)
    manifest = dict(reference)
    manifest.update(run_id=uuid.uuid4().hex, label=output.name, seeds=seeds,
                    record_episodes=list(range(len(seeds))), output_host=str(output),
                    output_container=str(Path(reference["mount_root"]) / output.relative_to(Path(__file__).resolve().parents[1])),
                    analysis_only=True, source_runs=[str(p) for p in parts])
    write(output / "manifest.json", manifest)
    write(output / "policy_ready.json", {"analysis_only": True,
        "checkpoint_files_sha256": reference["checkpoint_files_sha256"],
        "source_receipts": [str(p / "policy_ready.json") for p in parts]})
    write(output / "aggregation_provenance.json", {"episodes": provenance, "excluded": excluded,
        "failed_outcomes": list(failed.values()), "all_requested_seeds_attempted": True,
        "note": "Only completed receipts selected. Sources preserved; indices translated for analysis."})
    result = summarize(output)
    if not result["complete"] and result["infrastructure_failure_count"] != len(failed):
        raise ValueError("Assembled summary is incomplete")
    write(output / "run_status.json", {"status": "complete_with_controller_failures" if failed else "complete", "analysis_only": True,
        "all_requested_seeds_attempted": True,
        "note": "Completed seed results assembled from source runs; no new simulation executed."})
    (output / "README.md").write_text(
        "# Assembled evaluation evidence\n\nThis directory is an analysis artifact. "
        "Completed seeds are assembled from the source runs in `aggregation_provenance.json`; "
        "the original runs, interruptions and files remain intact. Receipt indices are translated "
        "with source SHA-256 hashes. Scores, videos and evaluator logs link to the original files.\n")
    print({key: result[key] for key in ("requested_episodes", "episodes_with_scores", "mean_score", "complete")})


if __name__ == "__main__":
    main()
