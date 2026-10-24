#!/usr/bin/env bash
# IJIES R1 revision: export the state of the submitted run (volume of pod wo0ny5hy0vo2vt) that the revision
# runs need, to an orphan branch $STATE_BRANCH of $GH_REPO, so that a fresh pod can continue without the old
# host. Runs CPU-only. Contents: per-clip calibration files (runs/calib), BF16 reference videos (seeds 42/7),
# cached skeletons (runs/keypoints), qtc metrics/post/gate/recon/driving_correction. Files > 90 MB are split.
set -uo pipefail
W=/workspace; B="${STATE_BRANCH:-state/qtc-submitted}"; D="$W/state_export"
REMOTE="https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git"
rm -rf "$D"; mkdir -p "$D/runs"; cd "$D" && git init -q && git checkout -q --orphan "$B"
mkdir -p runs/calib runs/videos runs/keypoints runs/qtc
cp "$W"/runs/calib/*_vidit.pt runs/calib/
cp "$W"/runs/videos/fp16-none-na-*-s42-t25-r0.mp4 "$W"/runs/videos/fp16-none-na-*-s7-t25-r0.mp4 runs/videos/
find "$W/runs/keypoints" -maxdepth 1 -name '*.json' -exec cp -t runs/keypoints/ {} +
for s in metrics post gate recon driving_correction; do
  [ -d "$W/runs/qtc/$s" ] && cp -a "$W/runs/qtc/$s" runs/qtc/
done
find runs/qtc -type d -name 'work_*' -prune -exec rm -rf {} +
find runs -type f -size +90M | while read -r f; do split -b 90m -d -a 2 "$f" "$f.part"; rm "$f"; done
( cd runs && find . -type f ! -name '*.part*' -print0 | xargs -0 sha256sum ) > MANIFEST.sha256
for f in $(find runs -name '*.part00'); do o="${f%.part00}"; echo "$(cat "${o}".part* | sha256sum | cut -d' ' -f1)  ./${o#runs/}" >> MANIFEST.sha256; done
du -sh runs; ls runs/calib runs/videos | head -40; wc -l MANIFEST.sha256
git add -A && git -c user.name=runpod -c user.email=runpod@local commit -qm "state of the submitted QTC run (pod wo0ny5hy0vo2vt)" \
  && git push -q -f "$REMOTE" "HEAD:$B" && echo "STATE PUSHED $B"
