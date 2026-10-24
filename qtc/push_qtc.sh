#!/usr/bin/env bash
# Push small QTC result files (JSON/TXT/CSV/logs) to branch results/<pod> of $GH_REPO. Videos stay on the volume.
set -uo pipefail
W=/workspace; POD="${RUNPOD_POD_ID:-local}"; B="${RESULTS_BRANCH:-results/$POD}"   # revision runs: RESULTS_BRANCH
# a repo-scoped deploy key (QTC_DEPLOY_KEY_B64, with GIT_SSH_COMMAND set by the pod start command) takes precedence
if [ -n "${QTC_DEPLOY_KEY_B64:-}" ]; then REMOTE="git@github.com:${GH_REPO}.git"
else REMOTE="https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git"; fi
push() {
  local D="$W/push_$POD"; [ -n "${RESULTS_BRANCH:-}" ] && D="$W/push_${POD}_${RESULTS_BRANCH//\//_}"
  if [ ! -d "$D/.git" ]; then
    git clone -q "$REMOTE" "$D" || return 0
    (cd "$D" && (git checkout -q -b "$B" origin/"$B" 2>/dev/null || git checkout -q --orphan "$B") && git rm -rq --cached . 2>/dev/null; true)
  fi
  mkdir -p "$D/runs"
  DIRS="meta errors qtc"; [ "${PUSH_KEYPOINTS:-0}" = "1" ] && DIRS="$DIRS keypoints"   # raw phase-1 skeletons (Zenodo)
  [ -d "$W/runs/calib_heldout" ] && DIRS="$DIRS calib_heldout"
  for s in $DIRS; do
    [ -d "$W/runs/$s" ] && rsync -a --include='*/' --include='*.json' --include='*.txt' --include='*.csv' \
        --exclude='post/**' --exclude='work_*/**' --exclude='*' "$W/runs/$s/" "$D/runs/$s/"
  done
  mkdir -p "$D/logs"; cp "$W"/logs/*"$POD"*.log "$W"/logs/*.txt "$D/logs/" 2>/dev/null
  (cd "$D" && git add -A . >/dev/null && git -c user.name=runpod -c user.email=runpod@local commit -qm "qtc results $(date -u +%FT%TZ)" \
     && git push -q origin "HEAD:$B") || true
}
if [ "${1:-once}" = "loop" ]; then while true; do sleep 600; push; done; else push; fi
