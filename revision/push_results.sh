#!/usr/bin/env bash
# [REV-R1] Push small result files (meta, gate, recon, bench, calib-K lists, logs) to branch results/<pod>.
set -uo pipefail
W=/workspace; POD="${RUNPOD_POD_ID:-local}"; B="results/$POD"
push() {
  local D="$W/push_$POD"
  if [ ! -d "$D/.git" ]; then
    git clone -q "https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git" "$D" || return 0
    (cd "$D" && (git checkout -q -b "$B" origin/"$B" 2>/dev/null || git checkout -q --orphan "$B") && git rm -rq --cached . 2>/dev/null; true)
  fi
  mkdir -p "$D/runs"
  for s in meta gate recon bench errors gate_errors; do
    [ -d "$W/runs/$s" ] && rsync -a --include='*/' --include='*.json' --include='*.txt' --exclude='*' "$W/runs/$s/" "$D/runs/$s/"
  done
  mkdir -p "$D/logs"; cp "$W"/logs/*"$POD"*.log "$W"/logs/*.txt "$D/logs/" 2>/dev/null
  cp "$W"/logs/openpose_build.err "$D/logs/" 2>/dev/null
  (cd "$D" && git add -A . >/dev/null && git -c user.name=runpod -c user.email=runpod@local commit -qm "results $(date -u +%FT%TZ)" \
     && git push -q origin "HEAD:$B") || true
}
if [ "${1:-once}" = "loop" ]; then while true; do sleep 900; push; done; else push; fi
