#!/usr/bin/env bash
# Build the simulator workspace inside the simulator container, following steps 3-7 of the
# kuavo-ros-opensource readme (branch opensource/kuavo-data-challenge-icra).
#
#   docker exec -i kuavo-icra-sim bash -s < docker/build_simulator.sh
#
# The workspace is the host checkout mounted at /root/kuavo_ws, so build/ and devel/ persist
# on the host; the apt packages and the SDK install live in the container.
set -eo pipefail
export ROBOT_VERSION=45
cd /root/kuavo_ws
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys AD19BAB3CBF125EA
apt-get update
apt-get install -y ros-noetic-geographic-msgs
catkin config -DCMAKE_ASM_COMPILER=/usr/bin/as -DCMAKE_BUILD_TYPE=Release
source installed/setup.bash
catkin build humanoid_controllers data_challenge_simulator
source devel/setup.bash
(cd src/kuavo_humanoid_sdk && ./install.sh)
echo "Simulator workspace built"
