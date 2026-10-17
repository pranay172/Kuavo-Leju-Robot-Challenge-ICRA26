#!/usr/bin/env python3
"""Copy one cloud EMA snapshot to a fresh laptop folder over SSH/rsync only (with retries)."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HOST = os.environ["POD_HOST"]  # e.g. root@<pod-ip>
PORT = os.environ["POD_PORT"]
KEY = os.path.expanduser(os.environ.get("POD_KEY", "~/.ssh/id_ed25519"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()
    remote = f"/workspace/runs/{args.run}/train/step{args.step:06d}/ema/"
    args.destination.mkdir(parents=True, exist_ok=True)
    ssh = f"ssh -p {PORT} -i {KEY} -o BatchMode=yes -o ConnectTimeout=15"
    start = time.monotonic()
    for attempt in range(1, args.attempts + 1):
        left = args.max_seconds - (time.monotonic() - start)
        if left <= 5:
            break
        command = ["rsync", "-a", "--partial", "--timeout=60", "-e", ssh,
                   f"{HOST}:{remote}", str(args.destination) + "/"]
        try:
            code = subprocess.run(command, timeout=left).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        print(f"Transfer attempt {attempt}: rsync exit {code}", file=sys.stderr, flush=True)
        if code == 0:
            files = sorted(p.name for p in args.destination.iterdir())
            size = sum(p.stat().st_size for p in args.destination.iterdir())
            print(json.dumps({"transport": "rsync", "attempts": attempt, "bytes": size,
                              "files": files, "elapsed_seconds": time.monotonic() - start}))
            return 0
        time.sleep(5)
    print(json.dumps({"transport": "rsync", "status": "failed",
                      "elapsed_seconds": time.monotonic() - start}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
