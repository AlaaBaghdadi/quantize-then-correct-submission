#!/usr/bin/env bash
# Data staging for the QTC study (idempotent). Only what Article 3 needs:
#   Drive (link-shared by the authors): per-clip ViDiT calibration files, reference images + driving videos
#   HuggingFace: DynamiCtrl checkpoint, DWPose weights
#   SmoothNet: source (GitHub) + pretrained checkpoints (authors' Google Drive folder)
set -euo pipefail
W="${W:-/workspace}"; HERE="$(cd "$(dirname "$0")" && pwd)"; CODE="$(cd "$HERE/.." && pwd)"
PY="$W/venv_gen/bin/python"
mkdir -p "$W/drive"
"$PY" - <<PY
import json, subprocess, pathlib, sys
W = pathlib.Path("$W"); ids = json.load(open("$HERE/drive_folders_qtc.json"))
for name, fid in ids.items():
    dst = W / "drive" / name; ok = dst / ".ok"
    if ok.exists() or (dst / ".unavailable").exists():
        continue
    dst.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        r = subprocess.run([str(W / "venv_gen/bin/gdown"), "--folder", fid, "-O", str(dst), "-q"])
        if r.returncode == 0:
            ok.write_text("ok"); print("[drive] ok", name, flush=True); break
        print("[drive] retry", name, attempt, flush=True)
    else:
        # non-fatal: assets ship with the DynamiCtrl repository and calibration statistics are recorded
        # during the seed-42 FP16 runs (qtc/gen_qtc.py) when Google Drive refuses the pod's address
        (dst / ".unavailable").write_text("gdown refused from this pod")
        print("[drive] unavailable", name, "- falling back (see fetch_qtc.sh)", flush=True)
PY
if [ ! -d "$W/content/DynamiCtrl/.git" ]; then
  git clone -q https://github.com/gulucaptain/DynamiCtrl.git "$W/content/DynamiCtrl"
fi
if [ ! -f "$W/content/DynamiCtrl/checkpoints/DynamiCtrl/.ok" ]; then
  "$W/venv_gen/bin/huggingface-cli" download gulucaptain/DynamiCtrl --local-dir "$W/content/DynamiCtrl/checkpoints/DynamiCtrl" >/dev/null
  touch "$W/content/DynamiCtrl/checkpoints/DynamiCtrl/.ok"
fi
mkdir -p "$W/content/checkpoints/DWPose"
for f in yolox_l.onnx dw-ll_ucoco_384.onnx; do
  [ -s "$W/content/checkpoints/DWPose/$f" ] || "$W/venv_gen/bin/huggingface-cli" download yzd-v/DWPose "$f" --local-dir "$W/content/checkpoints/DWPose" >/dev/null
done
# SmoothNet (Zeng et al., ECCV 2022) source + pretrained windows 8/16/32/64
SN="$W/SmoothNet"
[ -d "$SN/.git" ] || git clone -q https://github.com/cure-lab/SmoothNet.git "$SN"
if [ ! -f "$SN/.ckpt_ok" ]; then
  mkdir -p "$SN/data/checkpoints"
  "$W/venv_gen/bin/gdown" --folder 1AsOm10AReDKt4HSVAQ0MsZ1Fp-_18IV3 -O "$SN/data/checkpoints" -q || true
  CK=$(find "$SN/data/checkpoints" -path "*h36m_fcn_3D*" -name "checkpoint_32.pth.tar" -size +1M | head -1)
  if [ -z "$CK" ] && ls "$CODE"/third_party_ckpt/smoothnet_h36m_fcn_3D_w32.pth.tar.part* >/dev/null 2>&1; then
    # Drive refused the pod: rebuild the checkpoint from the copy committed in this repository
    mkdir -p "$SN/data/checkpoints/h36m_fcn_3D"
    cat $(ls "$CODE"/third_party_ckpt/smoothnet_h36m_fcn_3D_w32.pth.tar.part* | sort -V) > "$SN/data/checkpoints/h36m_fcn_3D/checkpoint_32.pth.tar"
    echo "ce79cbfda85979f3ae7eeda150ded759286642c3c805c0811814c8f3242c1128  $SN/data/checkpoints/h36m_fcn_3D/checkpoint_32.pth.tar" | sha256sum -c - || exit 4
    CK="$SN/data/checkpoints/h36m_fcn_3D/checkpoint_32.pth.tar"
  fi
  if [ -z "$CK" ]; then echo "[smoothnet] checkpoint h36m_fcn_3D/checkpoint_32.pth.tar NOT found"; ls -R "$SN/data/checkpoints" | head -50; exit 4; fi
  mkdir -p "$SN/data/checkpoints/h36m_fcn_3D"
  [ -f "$SN/data/checkpoints/h36m_fcn_3D/checkpoint_32.pth.tar" ] || cp "$CK" "$SN/data/checkpoints/h36m_fcn_3D/checkpoint_32.pth.tar"
  touch "$SN/.ckpt_ok"
fi
echo "[data] ok"
