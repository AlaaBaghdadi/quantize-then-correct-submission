#!/usr/bin/env bash
# [REV-R1] Create the two Python environments on the network volume (once).
#   $W/venv_gen  : DynamiCtrl generation + quantization (pins from quant_common.DYNAMICTRL_CORE_PINS)
#   $W/venv_gate : phase-1 kinematic pipeline (mediapipe==0.10.21 legacy API, numpy<2)
set -euo pipefail
W="${W:-/workspace}"
apt-get install -y -q --no-install-recommends python3.10-venv ffmpeg libgl1 libglib2.0-0 >/dev/null || true
if [ ! -f "$W/venv_gen/.ok" ]; then
  python3.10 -m venv "$W/venv_gen"
  "$W/venv_gen/bin/pip" install -q --upgrade pip wheel
  "$W/venv_gen/bin/pip" install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  "$W/venv_gen/bin/pip" install -q torchao==0.10.0 transformers==4.49.0 diffusers==0.32.0 "numpy>=2.0.0,<2.1.0" \
      accelerate sentencepiece imageio imageio-ffmpeg opencv-python-headless decord einops \
      onnxruntime-gpu==1.20.2 huggingface_hub gdown botocore scikit-image lpips==0.1.4 scipy pandas \
      protobuf matplotlib
  if [ -f "$W/content/DynamiCtrl/requirements.txt" ]; then
    grep -viE '^(torch|torchvision|numpy|transformers|diffusers|onnxruntime)' "$W/content/DynamiCtrl/requirements.txt" \
      > /tmp/dc_req.txt || true
    "$W/venv_gen/bin/pip" install -q -r /tmp/dc_req.txt || echo "[env] some DynamiCtrl extras failed (non-fatal)"
    "$W/venv_gen/bin/pip" install -q torch==2.6.0 torchvision==0.21.0 "numpy>=2.0.0,<2.1.0" transformers==4.49.0 diffusers==0.32.0 \
      --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
  fi
  "$W/venv_gen/bin/pip" freeze > "$W/logs/venv_gen_freeze.txt"
  touch "$W/venv_gen/.ok"
fi
if [ ! -f "$W/venv_gate/.ok" ]; then
  python3.10 -m venv "$W/venv_gate"
  "$W/venv_gate/bin/pip" install -q --upgrade pip wheel
  "$W/venv_gate/bin/pip" install -q "numpy>=1.23,<2" mediapipe==0.10.21 opencv-python==4.9.0.80 \
      scipy pandas
  "$W/venv_gate/bin/pip" freeze > "$W/logs/venv_gate_freeze.txt"
  touch "$W/venv_gate/.ok"
fi
echo "[env] ok"
