#!/usr/bin/env bash
# [REV-R1] One-time data staging on the network volume (flock-protected by bootstrap.sh).
#   Drive (link-shared by the authors): published videos, per-clip calibration files, assets, logs
#   HuggingFace: DynamiCtrl checkpoint (gulucaptain/DynamiCtrl), DWPose weights (yzd-v/DWPose)
set -euo pipefail
W="${W:-/workspace}"; REV="$(cd "$(dirname "$0")" && pwd)"
PY="$W/venv_gen/bin/python"
mkdir -p "$W/drive/outputs"
"$PY" - <<PY
import json, subprocess, pathlib, sys
W = pathlib.Path("$W"); ids = json.load(open("$REV/drive_folders.json"))
for name, fid in ids.items():
    dst = W / "drive" / ("outputs/" + name if name not in ("calib_data", "assets", "logs") else name)
    ok = dst / ".ok"
    if ok.exists():
        continue
    dst.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        r = subprocess.run([str(W / "venv_gen/bin/gdown"), "--folder", fid, "-O", str(dst), "--remaining-ok", "-q"])
        if r.returncode == 0:
            ok.write_text("ok"); print("[drive] ok", name, flush=True); break
        print("[drive] retry", name, attempt, flush=True)
    else:
        print("[drive] FAILED", name, "- is the folder link-shared?", flush=True); sys.exit(3)
PY
# DynamiCtrl source + checkpoint
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
echo "[data] ok"
