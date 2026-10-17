#!/usr/bin/env bash
# Clone the public challenge repositories at the pinned commits and apply this repo's patches.
#
#   LEJU_ROOT=/path/to/workspace upstream/setup_upstream.sh
#
# Creates $LEJU_ROOT/kuavo_data_challenge and $LEJU_ROOT/kuavo-ros-opensource. An existing
# checkout is reused only if it is at the pinned commit; patches already applied are skipped.
# Set KDC_URL / KUAVO_ROS_URL to use a local mirror instead of GitHub. SKIP_SIMULATOR=1 sets up
# only kuavo_data_challenge (enough for training).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LEJU_ROOT:?set LEJU_ROOT to the workspace root}"
_kdc_url="${KDC_URL:-}"; _ros_url="${KUAVO_ROS_URL:-}"
# shellcheck source=pins.env
source "$HERE/pins.env"
KDC_URL="${_kdc_url:-$KDC_URL}"; KUAVO_ROS_URL="${_ros_url:-$KUAVO_ROS_URL}"
mkdir -p "$LEJU_ROOT"

checkout() {  # url branch commit directory
    local url=$1 branch=$2 commit=$3 dir=$4
    if [[ ! -d "$dir/.git" ]]; then
        git clone --branch "$branch" "$url" "$dir"
    fi
    if ! git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
        git -C "$dir" fetch origin "$branch"
    fi
    local head; head="$(git -C "$dir" rev-parse HEAD)"
    if [[ "$head" != "$commit" ]]; then
        if [[ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]]; then
            echo "$dir has local changes and is not at $commit; refusing to touch it" >&2
            exit 1
        fi
        git -C "$dir" checkout --detach "$commit"
    fi
}

apply_patches() {  # directory patch-dir
    local dir=$1 patch
    for patch in "$2"/*.patch; do
        if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
            echo "already applied: ${patch##*/}"
        else
            git -C "$dir" apply --check --whitespace=nowarn "$patch"
            git -C "$dir" apply --whitespace=nowarn "$patch"
            echo "applied: ${patch##*/}"
        fi
    done
}

KDC="$LEJU_ROOT/kuavo_data_challenge"
checkout "$KDC_URL" "$KDC_BRANCH" "$KDC_COMMIT" "$KDC"
# LeRobot keeps only test artifacts in Git LFS; they are not needed.
GIT_LFS_SKIP_SMUDGE=1 git -C "$KDC" submodule update --init third_party/lerobot
[[ "$(git -C "$KDC/third_party/lerobot" rev-parse HEAD)" == "$LEROBOT_COMMIT" ]] \
    || { echo "third_party/lerobot is not at $LEROBOT_COMMIT" >&2; exit 1; }
apply_patches "$KDC" "$HERE/patches/kuavo_data_challenge"

[[ "${SKIP_SIMULATOR:-0}" == 1 ]] && { echo "Skipping kuavo-ros-opensource"; exit 0; }
ROS="$LEJU_ROOT/kuavo-ros-opensource"
checkout "$KUAVO_ROS_URL" "$KUAVO_ROS_BRANCH" "$KUAVO_ROS_COMMIT" "$ROS"
apply_patches "$ROS" "$HERE/patches/kuavo-ros-opensource"

echo "Upstream checkouts ready under $LEJU_ROOT"
