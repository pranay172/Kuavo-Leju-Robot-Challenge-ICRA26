#!/usr/bin/env bash
set -euo pipefail
cd /workspace/setup
export PIP_CACHE_DIR=/workspace/setup/pip-cache TMPDIR=/workspace/setup/tmp
python3 -c "import torch; assert torch.__version__==\"2.8.0+cu128\""
python3 -m venv --system-site-packages /workspace/venv
grep -E "^[A-Za-z0-9_.-]+==" venv_pip_freeze_20261005.txt | grep -v -E "^(torch|torchvision|torchaudio|triton|nvidia-)" > req_filtered.txt
/workspace/venv/bin/pip install -q --upgrade pip
/workspace/venv/bin/pip install -r req_filtered.txt
/workspace/venv/bin/pip install --no-deps -e /workspace/kuavo_data_challenge/third_party/lerobot -e /workspace/kuavo_data_challenge
/workspace/venv/bin/python -c "import torch,torchvision,lerobot,av,cv2;print(\"ENV_OK\",torch.__version__,torchvision.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
