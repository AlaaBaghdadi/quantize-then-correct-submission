# kinematic_cache/

> Cached pose keypoints extracted during kinematic gate runs. **Safe to delete** — will be regenerated on next gate run.

## Contents

```
kinematic_cache/
  keypoints/
    <clip_id>/
      <clip_id>__fp16__<hash>_smoothed_keypoints.json
      <clip_id>__quant__<hash>_smoothed_keypoints.json
      <clip_id>__quant__<hash>_angles.json
      *.meta.json
```

Each `<clip_id>` subdirectory corresponds to a grid clip (e.g. `human2_motion1`).

## File naming

| Pattern | Meaning |
|---------|---------|
| `__fp16__` | Keypoints from FP16 baseline video |
| `__quant__` | Keypoints from quantized video |
| `_smoothed_keypoints.json` | Kalman-smoothed pose keypoints |
| `_angles.json` | Joint angle time series |
| `.meta.json` | Extraction metadata (video path hash, timestamps) |

The `<hash>` suffix identifies the source video file so cache entries invalidate when videos change.

## How caching works

When `kinematic_gate.skip_if_cached` is `true` in [`configs/quant/vidit_ptq.json`](../configs/quant/vidit_ptq.json) (default), gate scripts reuse existing keypoint files instead of re-running OpenPose + MediaPipe extraction.

To force re-extraction:

1. Set `skip_if_cached: false` in config, or
2. Delete the relevant `keypoints/<clip_id>/` subdirectory, or
3. Delete the entire `kinematic_cache/` folder

## How to use

This directory is populated automatically by [scripts/quant/kinematic_gate/](../scripts/quant/kinematic_gate/README.md). No manual setup required.

Cache location is configured via `kinematic_gate.cache_dir` (default: `./kinematic_cache`).

## See also

- [../README.md](../README.md) — project overview
- [../scripts/quant/kinematic_gate/README.md](../scripts/quant/kinematic_gate/README.md) — gate pipeline that writes cache
- [../logs/README.md](../logs/README.md) — gate pass/fail logs
- [../outputs/README.md](../outputs/README.md) — source videos for extraction
- [../configs/quant/README.md](../configs/quant/README.md) — `kinematic_gate` config section
