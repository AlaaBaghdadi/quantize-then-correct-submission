#!/usr/bin/env bash
# [REV-R1] Pod entry point.
# Start command (pod create):
#   bash -lc 'git clone -q -b "$GH_BRANCH" "https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git" /workspace/code_$RUNPOD_POD_ID
#             && bash /workspace/code_$RUNPOD_POD_ID/revision/bootstrap.sh; sleep infinity'
# Env: GH_TOKEN GH_REPO GH_BRANCH  JOBS (e.g. WI2,WI3,WI5 | BENCH | SETUP)  SHARD (k/n)  GATE_PROCS  AUTO_STOP
set -uo pipefail
export W=/workspace
CODE="$(cd "$(dirname "$0")/.." && pwd)"; REV="$CODE/revision"
POD="${RUNPOD_POD_ID:-local}"; JOBS="${JOBS:-SETUP}"; SHARD="${SHARD:-0/1}"; GATE_PROCS="${GATE_PROCS:-4}"
mkdir -p "$W/logs" "$W/runs"
exec > >(tee -a "$W/logs/bootstrap_${POD}.log") 2>&1
echo "== $(date -u) pod=$POD jobs=$JOBS shard=$SHARD"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
apt-get update -q >/dev/null
# system libs every pod needs (OpenPose runtime deps are re-installed; the build itself is on the volume)
apt-get install -y -q --no-install-recommends python3.10 python3.10-venv ffmpeg libgl1 libglib2.0-0 \
  libprotobuf-dev libgoogle-glog-dev libgflags-dev libhdf5-dev libopenblas-dev libboost-all-dev \
  libopencv-dev libatlas-base-dev rsync git curl >/dev/null || true
# /content -> volume (the original Colab code hard-codes /content paths)
mkdir -p "$W/content"; [ -e /content ] || ln -s "$W/content" /content

(
  flock 9
  [ -d "$W/content/DynamiCtrl/.git" ] || git clone -q https://github.com/gulucaptain/DynamiCtrl.git "$W/content/DynamiCtrl"
  bash "$REV/setup_env.sh" && bash "$REV/fetch_data.sh" && bash "$REV/setup_openpose.sh"
  REPO="$W/content/DynamiCtrl"
  if [ ! -f "$REPO/configs/quant/quantize_layers.txt" ]; then
    mkdir -p "$REPO/scripts" "$REPO/configs"
    cp -r "$CODE/scripts/quant" "$REPO/scripts/"; cp -r "$CODE/configs/quant" "$REPO/configs/"
    (cd "$REPO" && "$W/venv_gen/bin/python" scripts/quant/list_layers.py --repo-root . --model-path ./checkpoints/DynamiCtrl)
  fi
) 9>"$W/.setup.lock"
[ -x "$W/openpose/build/examples/openpose/openpose.bin" ] || { echo "SETUP FAILED"; sleep infinity; }

# refresh code inside the DynamiCtrl clone on every start (quantize_layers.txt is kept)
REPO="$W/content/DynamiCtrl"; export REPO
cp -r "$CODE/scripts/quant/." "$REPO/scripts/quant/"
cp "$CODE/configs/quant/vidit_ptq.json" "$REPO/configs/quant/"
mkdir -p "$REPO/assets"; cp -n "$W/drive/assets/"*.{jpg,mp4} "$REPO/assets/" 2>/dev/null || true
cp -n "$CODE/assets/"*.{jpg,mp4} "$REPO/assets/" 2>/dev/null || true
ls "$REPO/assets"

export DONE_FLAG="$W/runs/GEN_DONE_POD_$POD"; rm -f "$DONE_FLAG"
"$REV/push_results.sh" loop &
PUSH_PID=$!
case "$JOBS" in
  SETUP) echo "setup only";;
  BENCH) "$W/venv_gen/bin/python" "$REV/bench.py" ;;
  GATE)  touch "$DONE_FLAG"; "$W/venv_gate/bin/python" "$REV/gate_worker.py" --procs "$GATE_PROCS"
         "$W/venv_gen/bin/python" "$REV/recon_worker.py" ;;
  *)
    "$W/venv_gate/bin/python" "$REV/gate_worker.py" --procs "$GATE_PROCS" --follow > "$W/logs/gate_${POD}.log" 2>&1 &
    GATE_PID=$!
    "$W/venv_gen/bin/python" "$REV/gen.py" --wi "$JOBS" --shard "$SHARD"
    "$W/venv_gen/bin/python" "$REV/recon_worker.py" > "$W/logs/recon_${POD}.log" 2>&1
    touch "$DONE_FLAG"; wait "$GATE_PID"
    "$W/venv_gate/bin/python" "$REV/gate_worker.py" --procs "$GATE_PROCS"   # final sweep
    "$W/venv_gen/bin/python" "$REV/recon_worker.py" >> "$W/logs/recon_${POD}.log" 2>&1 ;;
esac
kill "$PUSH_PID" 2>/dev/null; "$REV/push_results.sh" once
echo "== $(date -u) finished"
if [ "${AUTO_STOP:-1}" = "1" ] && command -v runpodctl >/dev/null; then runpodctl stop pod "$POD"; fi
sleep infinity
