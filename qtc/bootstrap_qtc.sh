#!/usr/bin/env bash
# Pod entry point for the Quantize-then-Correct study.
# Start command (pod create) with a repo-scoped deploy key (env QTC_DEPLOY_KEY_B64 = base64 of the private key):
#   bash -lc 'mkdir -p ~/.ssh && echo "$QTC_DEPLOY_KEY_B64" | base64 -d > ~/.ssh/qtc && chmod 600 ~/.ssh/qtc
#             && export GIT_SSH_COMMAND="ssh -i ~/.ssh/qtc -o StrictHostKeyChecking=accept-new"
#             && git clone -q -b "$GH_BRANCH" "git@github.com:${GH_REPO}.git" /workspace/code_$RUNPOD_POD_ID
#             && bash /workspace/code_$RUNPOD_POD_ID/qtc/bootstrap_qtc.sh; sleep infinity'
# Alternative start command with a GitHub token (RunPod secret) that has access to the repo:
#   bash -lc 'git clone -q -b "$GH_BRANCH" "https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git" /workspace/code_$RUNPOD_POD_ID
#             && bash /workspace/code_$RUNPOD_POD_ID/qtc/bootstrap_qtc.sh; sleep infinity'
# Env: GH_TOKEN GH_REPO GH_BRANCH  JOBS (SETUP | PILOT | CORE | CORE_OPT | REVISION | BENCH | SCORE | ANALYZE)  SHARD (k/n)
#      RESULTS_BRANCH (default results/<pod>)  MAX_HOURS (watchdog: stop the pod after this many hours)
#      GATE_PROCS (CPU skeleton workers, default 6)  AUTO_STOP (1 = stop the pod when finished)
set -uo pipefail
export W=/workspace
CODE="$(cd "$(dirname "$0")/.." && pwd)"; QTC="$CODE/qtc"; REV="$CODE/revision"
POD="${RUNPOD_POD_ID:-local}"; JOBS="${JOBS:-SETUP}"; SHARD="${SHARD:-0/1}"; GATE_PROCS="${GATE_PROCS:-6}"
export SMOOTHNET_ROOT="$W/SmoothNet" GATE_THREADS="${GATE_THREADS:-2}"
mkdir -p "$W/logs" "$W/runs/qtc"
exec > >(tee -a "$W/logs/qtc_bootstrap_${POD}.log") 2>&1
echo "== $(date -u) pod=$POD jobs=$JOBS shard=$SHARD code=$(git -C "$CODE" rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
apt-get update -q >/dev/null
apt-get install -y -q --no-install-recommends python3.10 python3.10-venv ffmpeg libgl1 libglib2.0-0 \
  libprotobuf-dev libgoogle-glog-dev libgflags-dev libhdf5-dev libopenblas-dev libboost-all-dev \
  libopencv-dev libatlas-base-dev rsync git curl >/dev/null || true
mkdir -p "$W/content"; [ -e /content ] || ln -s "$W/content" /content

(
  flock 9
  [ -d "$W/content/DynamiCtrl/.git" ] || git clone -q https://github.com/gulucaptain/DynamiCtrl.git "$W/content/DynamiCtrl"
  bash "$REV/setup_env.sh" || exit 1
  # CPU torch + plotting stack for SmoothNet post-correction and analysis in the gate venv
  if [ ! -f "$W/venv_gate/.qtc_ok" ]; then
    "$W/venv_gate/bin/pip" install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    "$W/venv_gate/bin/pip" install -q matplotlib "scipy>=1.10" pandas
    touch "$W/venv_gate/.qtc_ok"
  fi
  # QTC_RECORD_CALIB=1: do not fetch the Drive calibration files; record them during the seed-42 BF16 runs,
  # as in the submitted run (meta calib_source = "recorded in this run")
  if [ "${QTC_RECORD_CALIB:-0}" = "1" ] && [ ! -f "$W/drive/calib_data/.ok" ]; then
    mkdir -p "$W/drive/calib_data"; touch "$W/drive/calib_data/.unavailable"; fi
  bash "$QTC/fetch_qtc.sh" || exit 1
  bash "$REV/setup_openpose.sh" || exit 1
  REPO="$W/content/DynamiCtrl"
  if [ ! -f "$REPO/configs/quant/quantize_layers.txt" ]; then
    mkdir -p "$REPO/scripts" "$REPO/configs"
    cp -r "$CODE/scripts/quant" "$REPO/scripts/"; cp -r "$CODE/configs/quant" "$REPO/configs/"
    (cd "$REPO" && "$W/venv_gen/bin/python" scripts/quant/list_layers.py --repo-root . --model-path ./checkpoints/DynamiCtrl)
  fi
) 9>"$W/.setup.lock"
[ -x "$W/openpose/build/examples/openpose/openpose.bin" ] || { echo "SETUP FAILED"; sleep infinity; }

# revision runs on a fresh pod: restore calibration files, reference videos and skeleton cache first
[ -n "${STATE_BRANCH:-}" ] && { bash "$QTC/state_restore.sh" || { echo "STATE RESTORE FAILED"; sleep infinity; }; }
REPO="$W/content/DynamiCtrl"; export REPO
cp -r "$CODE/scripts/quant/." "$REPO/scripts/quant/"
cp "$CODE/configs/quant/vidit_ptq.json" "$REPO/configs/quant/"
mkdir -p "$REPO/assets"; cp -n "$W/drive/assets/"*.{jpg,mp4} "$REPO/assets/" 2>/dev/null || true
cp -n "$CODE/assets/"*.{jpg,mp4} "$REPO/assets/" 2>/dev/null || true
ls "$REPO/assets"
export QTC_ASSETS="$REPO/assets"
# cuDNN 9 / cuBLAS from the torch wheels, so onnxruntime-gpu runs DWPose on the GPU instead of the CPU
NVLIBS=$(ls -d "$W"/venv_gen/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"

GEN="$W/venv_gen/bin/python"; GATE="$W/venv_gate/bin/python"
if [ -n "${MAX_HOURS:-}" ]; then   # hard stop
  ( sleep "$(python3 -c "print(int(float('$MAX_HOURS')*3600))")"; echo "== MAX_HOURS reached"; "$QTC/push_qtc.sh" once
    command -v runpodctl >/dev/null && runpodctl stop pod "$POD" ) &
fi
export DONE_FLAG="$W/runs/qtc/GEN_DONE_POD_$POD"; rm -f "$DONE_FLAG"
"$QTC/push_qtc.sh" loop & PUSH_PID=$!

precorrect() {   # driving-pose correction fields (CPU, ~minutes); needed by drive=pre units
  for m in motion1 motion2 motion3; do
    (cd "$CODE" && OPENPOSE_BIN="$W/openpose/build/examples/openpose/openpose.bin" QTC_OP_ROOT="$W/op_work/pre_$m" \
       "$GATE" -m qtc.precorrect build --motion "$m") > "$W/logs/qtc_precorrect_${m}.log" 2>&1 &
  done
  wait
  ls -la "$W/runs/qtc/driving_correction/"*.json
}

generate() {     # $1 = unit sets
  (cd "$CODE" && "$GATE" -m qtc.score kin --sets "$1" --procs "$GATE_PROCS" --follow) > "$W/logs/qtc_kin_${POD}.log" 2>&1 &
  local KIN_PID=$!
  precorrect &
  local PRE_PID=$!
  (cd "$CODE" && "$GEN" -m qtc.gen_qtc --sets "$1" --shard "$SHARD" --skip-pre)   # raw-drive units first
  wait "$PRE_PID"
  (cd "$CODE" && "$GEN" -m qtc.gen_qtc --sets "$1" --shard "$SHARD")              # then drive=pre units
  (cd "$CODE" && "$GEN" -m qtc.score recon --sets "$1") > "$W/logs/qtc_recon_${POD}.log" 2>&1
  touch "$DONE_FLAG"; wait "$KIN_PID"
  (cd "$CODE" && "$GATE" -m qtc.score kin --sets "$1" --procs "$GATE_PROCS")        # final sweep
  (cd "$CODE" && "$GATE" -m qtc.analyze --runs "$W/runs" --out "$W/runs/qtc/analysis") || true
}

case "$JOBS" in
  SETUP)    echo "setup only";;
  PUSH)     echo "push only (set PUSH_KEYPOINTS=1 to include runs/keypoints)";;
  PILOT)    generate pilot ;;
  CORE)     generate core ;;
  CORE_OPT) generate core,optional ;;
  FALLBACK) generate fallback ;;
  REVISION) generate refs,heldout,seeds ;;                # IJIES R1: BF16 refs (skipped if present) + held-out calibration + extra seeds
  BENCH)    (cd "$CODE" && "$GEN" -m qtc.bench_qtc) ;;
  SCORE)    touch "$DONE_FLAG"; (cd "$CODE" && "$GATE" -m qtc.score kin --sets pilot,core,optional,fallback --procs "$GATE_PROCS")
            (cd "$CODE" && "$GEN" -m qtc.score recon --sets pilot,core,optional,fallback) ;;
  ANALYZE)  (cd "$CODE" && "$GATE" -m qtc.analyze --runs "$W/runs" --out "$W/runs/qtc/analysis") ;;
  *) echo "unknown JOBS=$JOBS" ;;
esac
kill "$PUSH_PID" 2>/dev/null; "$QTC/push_qtc.sh" once
echo "== $(date -u) finished"
if [ "${AUTO_STOP:-1}" = "1" ] && command -v runpodctl >/dev/null; then runpodctl stop pod "$POD"; fi
sleep infinity
