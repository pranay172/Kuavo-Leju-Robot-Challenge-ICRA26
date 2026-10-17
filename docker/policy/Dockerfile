# Policy (inference) image for the local simulator evaluation.
#
# Built from the pinned, patched public kuavo_data_challenge checkout (upstream/setup_upstream.sh),
# passed as the named build context "kdc":
#
#   docker build -t kuavo-icra-policy:cu128 \
#       --build-context kdc="$LEJU_ROOT/kuavo_data_challenge" docker/policy
#
# Same layout as the upstream Dockerfile: code in /root/kuavo_data_challenge, Python 3.10
# environment in /root/kuavo_data_challenge/myenv, ROS Noetic from the base image. The upstream
# Dockerfile unpacks a conda-pack archive; this one creates the environment from
# requirements.txt, the exact package list of the environment that produced the results.

FROM ros:noetic-ros-core-focal AS source
COPY --from=kdc . /src
RUN rm -rf /src/.git /src/third_party/lerobot/.git

FROM ros:noetic-ros-core-focal
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential bzip2 \
        ros-noetic-cv-bridge ros-noetic-apriltag-ros \
    && rm -rf /var/lib/apt/lists/*

ARG MINIFORGE_VERSION=25.3.1-0
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh

WORKDIR /root/kuavo_data_challenge
# ffmpeg 6.1.1 from conda-forge, as the upstream README recommends for torchcodec.
RUN /opt/conda/bin/conda create -y -p /root/kuavo_data_challenge/myenv -c conda-forge \
        python=3.10.18 ffmpeg=6.1.1 \
    && /opt/conda/bin/conda clean -afy

COPY requirements.txt /tmp/requirements.txt
# The list is a complete freeze, so dependency resolution is skipped. torch/torchvision come from
# the PyTorch index in a separate step, so no other package picks up a PyTorch-index build.
RUN /root/kuavo_data_challenge/myenv/bin/python -m pip install --no-cache-dir --no-deps \
        -r /tmp/requirements.txt \
    && /root/kuavo_data_challenge/myenv/bin/python -m pip install --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128 torchvision==0.22.1+cu128

COPY --from=source /src /root/kuavo_data_challenge
RUN /root/kuavo_data_challenge/myenv/bin/python -m pip install --no-cache-dir --no-deps \
        -e /root/kuavo_data_challenge -e /root/kuavo_data_challenge/third_party/lerobot \
    && chmod -R 777 /root/kuavo_data_challenge/kuavo_deploy

ENV PATH="/root/kuavo_data_challenge/myenv/bin:/opt/conda/bin:${PATH}" \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    ROS_VERSION=1 ROS_DISTRO=noetic ROS_PYTHON_VERSION=3 \
    ROS_ROOT=/opt/ros/noetic/share/ros ROS_PACKAGE_PATH=/opt/ros/noetic/share \
    ROS_ETC_DIR=/opt/ros/noetic/etc/ros CMAKE_PREFIX_PATH=/opt/ros/noetic \
    PKG_CONFIG_PATH=/opt/ros/noetic/lib/pkgconfig \
    PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/ros/noetic/lib \
    ROS_IP=127.0.0.1 ROS_MASTER_URI=http://127.0.0.1:11311

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

CMD ["bash"]
