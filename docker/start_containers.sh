#!/usr/bin/env bash
# Create the simulator and policy containers that evaluation/eval_seeds.py drives with docker exec.
#
#   LEJU_ROOT=/path/to/workspace docker/start_containers.sh
#
# Needs: the simulator image loaded, the policy image built (docker/policy/Dockerfile), the
# harness installed (evaluation/install_runtime.sh), a running X server in $DISPLAY, and X
# access for the container's root user:  xhost +SI:localuser:root
# After a host reboot, run that xhost command again and `docker start` both containers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LEJU_ROOT:?set LEJU_ROOT to the workspace root}"
: "${DISPLAY:?DISPLAY must name the host X server (MuJoCo renders the cameras through it)}"
# shellcheck source=../upstream/pins.env
source "$HERE/../upstream/pins.env"
POLICY_IMAGE=${POLICY_IMAGE:-kuavo-icra-policy:cu128}
SIM_CONTAINER=${SIM_CONTAINER:-kuavo-icra-sim}
POLICY_CONTAINER=${POLICY_CONTAINER:-kuavo-icra-policy}
RUNTIME="$LEJU_ROOT/recovery"
[[ -f "$RUNTIME/tools/run_sim_evaluation.py" ]] || { echo "Run evaluation/install_runtime.sh first" >&2; exit 1; }

# X cookie readable by root inside the container (harmless if xhost already allows root).
XAUTH_FILE="$RUNTIME/environment/xauthority"
touch "$XAUTH_FILE" && chmod 600 "$XAUTH_FILE"
if command -v xauth >/dev/null; then
    xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH_FILE" nmerge - || true
fi

# The harness only drives containers carrying this label, with host networking and their own PID namespace.
docker run -d --name "$SIM_CONTAINER" --label leju.recovery=true --net=host --gpus all \
    -e DISPLAY="$DISPLAY" -e XAUTHORITY=/recovery/environment/xauthority \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$RUNTIME:/recovery" -v "$LEJU_ROOT/kuavo-ros-opensource:/root/kuavo_ws" \
    -v /tmp/.X11-unix:/tmp/.X11-unix -w /root/kuavo_ws \
    --entrypoint /bin/bash "$SIM_IMAGE" -c "exec sleep infinity"

docker run -d --name "$POLICY_CONTAINER" --label leju.recovery=true --net=host --gpus all \
    -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e TORCH_HOME=/recovery/cache/torch \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
    -v "$RUNTIME:/recovery" -v "$LEJU_ROOT:$LEJU_ROOT" -w /root/kuavo_data_challenge \
    --entrypoint /bin/bash "$POLICY_IMAGE" -c "exec sleep infinity"

# Lets offscreen camera rendering skip window presentation (loaded with LD_PRELOAD by the harness).
docker exec "$SIM_CONTAINER" gcc -shared -fPIC -O2 -o /recovery/environment/libheadless_render_fix.so \
    /recovery/environment/headless_render_fix.c -ldl
echo "Containers running: $SIM_CONTAINER, $POLICY_CONTAINER"
