#!/usr/bin/env bash
# Restore the exported state (qtc/state_export.sh) into $W/runs on a fresh pod. Idempotent.
set -euo pipefail
W=/workspace; B="${STATE_BRANCH:-state/qtc-submitted}"; S="$W/state"
[ -f "$W/runs/.state_restored" ] && { echo "[state] already restored"; exit 0; }
REMOTE="https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git"
[ -d "$S/.git" ] || git clone -q --depth 1 -b "$B" "$REMOTE" "$S"
cd "$S/runs"
for p in $(find . -name '*.part00'); do o="${p%.part00}"; cat "$o".part* > "$o"; done
sha256sum -c --quiet ../MANIFEST.sha256
mkdir -p "$W/runs"
find . -name '*.part[0-9][0-9]' -delete
cp -a ./. "$W/runs/"
touch "$W/runs/.state_restored"; echo "[state] restored from $B"
