# Low-Bit Post-Training Quantization and Kinematic Pose Correction of a Pose-Driven Video Diffusion Transformer: An Evaluation Calibrated Against the Seed-to-Seed Noise Floor

Code, per-video records and statistics for the article by A. A. Albaghdadi and A. R. Naghsh-Nilchi
(University of Isfahan), submitted to the *International Journal of Intelligent Engineering and Systems*
(paper 20266618, revision R1). Repository: `AlaaBaghdadi/quantize-then-correct-submission`.

The study quantizes the DynamiCtrl diffusion transformer (5.77 B parameters) to W8A8 and W4A8. It tests
whether the kinematic statistics of the generated human motion differ from a bfloat16 (BF16) reference by
more than the seed-to-seed variation of the BF16 model itself. It also evaluates a pose-correction pipeline
(KAPT followed by SmoothNet) applied before generation (pre-correction) and after it (post-correction). No
model is trained.

## Arms

| Arm | Weights / activations | Kernels | Driving pose | Clips x seeds |
|---|---|---|---|---|
| BF16 (reference) | bfloat16 | native | raw, pre-corrected | 12 x {42, 7}; revision: + seeds {1, 2, 3} |
| W8A8 | INT8 per-channel / INT8 per-token, SmoothQuant, 24 protected layers | torchao integer kernels | raw, pre-corrected | 12 x {42, 7} |
| W4A8 | INT4 per-group (g = 128, RTN) / INT8 per-token, same SmoothQuant and protected set | **simulated** (fake quantization) | raw, pre-corrected | 12 x {42, 7} |

**W4A8 is simulated.**
- The de-quantized weights are held in bfloat16 and the activations are fake-quantized. This reproduces the
  numerics of an integer kernel up to accumulation order.
- The W4A8 latency and peak memory are those of a bfloat16 execution. They do not measure a W4A8 kernel.
- The W4A8 weight memory (5.71 GB) is the packed 4-bit footprint computed from the parameter census; it was
  not measured.
- No latency or memory saving is claimed for W4A8.

The revision adds two sets of runs:
- W8A8 and W4A8 with held-out-by-subject calibration (96 videos);
- three extra BF16 seeds per clip for the noise floor (36 videos).

## Layout

| Path | Content |
|---|---|
| `qtc/` | study code: generation units, pre/post-correction, scoring, analysis, revision statistics |
| `scripts/quant/`, `configs/quant/` | DynamiCtrl post-training quantization (ViDiT-Q W8A8, W4A8 simulation), kinematic gate |
| `revision/` | generator, skeleton extractor, environment and OpenPose build scripts |
| `phase1/` | OpenPose + MediaPipe fusion, Kalman filter, KAPT, SmoothNet bridge |
| `paper/data/` | per-video tables of the submitted runs, input of `qtc.revision_stats` |
| `revision/ijies-r1/` | `RESULTS.md` (every number of the revision with script, input and commit) and `stats/` (outputs) |
| `data/submitted_runs_wo0ny5hy0vo2vt/runs/` | per-video records of the submitted runs: metrics, gate verdicts, reconstructions, layer sets, metadata |
| `data/r1_runs_9ykozyhz1vzsv8/runs/` | per-video records of the revision runs (held-out calibration, extra BF16 seeds) |
| `data/correction_fields/` | the driving-pose correction fields used for pre-correction |
| `tests/` | CPU tests (`python -m pytest -q tests`) |

Not included:
- the DynamiCtrl, ViDiT-Q, SmoothNet and OpenPose weights, which are available from their authors under their own licences;
- the generated videos;
- the driving videos.

## Reproducing the statistics (CPU)

```bash
pip install numpy pandas scipy statsmodels matplotlib
python -m qtc.revision_stats                       # clustering, equivalence, gate stability, pilot-clip sensitivity
python -m qtc.revision_gpu_stats \
    --new data/r1_runs_9ykozyhz1vzsv8 --sub data/submitted_runs_wo0ny5hy0vo2vt   # held-out calibration, extended noise floor
```

Random seeds are fixed. The outputs are written to `revision/ijies-r1/stats/` and summarised in
`revision/ijies-r1/RESULTS.md`.

The figures under `data/*/runs/qtc/analysis/` are the analysis outputs as produced at run time. Some of them
label the reference arm "FP16", the earlier name of the BF16 arm. The cost-plausibility frontier figure there
is exploratory and is not part of the manuscript.

## Regenerating the videos (GPU)

The runs used one A100 80 GB on RunPod. The pod start command is below. Set the environment variables
`GH_TOKEN`, `GH_REPO=AlaaBaghdadi/quantize-then-correct-submission` and `GH_BRANCH=main`.

```
bash -lc 'git clone -q -b "$GH_BRANCH" "https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git" /workspace/code_$RUNPOD_POD_ID && bash /workspace/code_$RUNPOD_POD_ID/qtc/bootstrap_qtc.sh; sleep infinity'
```

`JOBS` selects the stage:

| `JOBS` value | Stage |
|---|---|
| `SETUP` | environment setup |
| `PILOT` | pilot run |
| `CORE` | core arms |
| `CORE_OPT` | core arms plus optional arms |
| `REVISION` | BF16 references, held-out calibration and extra seeds |
| `BENCH` | parameter census and latency/memory measurement |
| `SCORE` | scoring |
| `ANALYZE` | analysis |

Small result files are pushed to a branch `results/<pod id>`. The driving videos, reference images and
calibration statistics must be provided separately; their sources are listed in `qtc/drive_folders_qtc.json`.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). Third-party models and code
keep their own licences.
