# P2 evaluation entry + diagnostics — evidence receipts

Small receipts for the two files this task owns (`src/owac/evaluation/n02_full/v12.py`,
`src/owac/evaluation/n02_full/v12_viz.py`) and the LIBERO protocol confirmation. Raw figures,
checkpoints and logs stay on the server; the numbers below are the real outputs.

## Commands actually run

Local (checkout `/home/jeong/zeno/owac/worktrees/n02-measurement-completion`):

```
uv run --locked ruff check .                     -> All checks passed!
uv run --locked ruff format --check .            -> 323 files already formatted
uv run --locked python -m compileall -q src      -> ok
uv run --locked pytest tests/encoders/n02_full -q -> 588 passed, 6 skipped (CUDA), 0 failed
  (run twice back to back; one earlier run showed a single flaky timing failure in
   tests/encoders/n02_full/test_p1_model.py::test_score_hot_path_beats_revalidating_every_call,
   which passes on its own and in both later full runs and touches no file of this task)
uv run --locked pytest tests/encoders/n02_full/test_v12_eval.py -q -> 14 passed
```

Remote (GPU, `outputs/n02-full/runtime/py310-cu128/bin/python`). Two invocation shapes were run
and both produced the same numbers: (a) cwd `<worktree>/src` so `owac` resolves to `src/owac`,
all paths absolute, and (b) the documented shape from the worktree root
`--config configs/encoders/n02_v12/raw_n02_robotwin.json` (the receipt records
`module_file` so the copy that ran is never ambiguous - the worktree root carries an `owac/`
mirror of `src/owac`, and the mirror was byte-identical in this run):

```
CUDA_VISIBLE_DEVICES=0 python -m owac.evaluation.n02_full.v12 \
  --dataset-format robotwin --calibration-samples 6 --dev-samples 3 --intervention-samples 1 \
  --anchors-dir $W/outputs/n02-full/p0-independent-audit/frozen-data \
  --manifest $W/outputs/n02-v12/P2/measurement-completion/manifest-robotwin.json \
  --output $W/outputs/n02-v12/P2/raw-n02-robotwin-uninit/eval --action-check auto
```

```
cd $W && PYTHONPATH="$W/src:/workspace/owac/rawvla/source/starVLA" CUDA_VISIBLE_DEVICES=0 \
  outputs/n02-full/runtime/py310-cu128/bin/python -m owac.evaluation.n02_full.v12 \
  --config configs/encoders/n02_v12/raw_n02_robotwin.json \
  --calibration-samples 6 --dev-samples 3 --intervention-samples 1 \
  --anchors-dir $W/outputs/n02-full/p0-independent-audit/frozen-data \
  --manifest $W/outputs/n02-v12/P2/measurement-completion/manifest-robotwin.json \
  --output $W/outputs/n02-v12/P2/raw-n02-robotwin-uninit/eval --action-check auto
```
(that config resolves to cameras `robotwin_head/left/right`, `k=4`, `d_model=512`,
`dims=[128,256,512,1024]`, `n_queries=256`, `out_dim=2048`, config sha256
`a7cad7e61218476761962a9a425801b941a11039193f921f499e441678d71701`)

Two interface hazards found in the hand-written config (reported, not edited):
`policy.checkpoint` is the **pi0.5 backend** path, so the eval must not treat a wildcard
`checkpoint` key as the N02 model weights (`model.checkpoint` / `paths.checkpoint` are used, and a
non-torch suffix is refused instead of handed to `torch.load`); and the worktree-root `owac/`
mirror shadows `src/owac` for `python -m` runs started from the worktree root.

## Real numbers (RoboTwin, dev/calibration partitions of the frozen manifest)

* model: 106,324,683 parameters, cameras `robotwin_head/left/right`, **no trained checkpoint
  existed yet**, so the weights are the freshly built ones (`checkpoint.loaded = false`,
  `reason = "no checkpoint given: the weights are the freshly built ones"`).
* calibration (independent `calibration` partition): threshold **0.91736** at quantile 0.95 over
  **6** sequences (6 finite), rule hash `49276cf6f03d44f7…`, statistic min 0.015968 / median
  0.309657 / max 0.91736.
* three-state counts on the 3 fixed dev anchors (`dev_samples.json`):
  **total 3, valid 3, invalid 0, compatible 3, insufficient_information 0, model_inadequate 0**;
  all 24 causal rows: compatible 24, invalid 0. Invalid rows are counted separately by contract.
* dev flow-matching action error through the real π0.5 RoboTwin backend, fixed noise
  (`Generator(seed=1103)`) and fixed flow time 0.5 on the same fixed samples:
  **mean 2.813329, median 2.295501, values [4.261521, 1.882966, 2.295501]**;
  tokens `(1, 768, 2048)`, actions `(1, 50, 32)`, control dim 14, audit 374 loaded keys / 0
  unexpected / 0 active-missing.
* real interventions on the same dev anchor (history rows included, `applied_to_history = true`):

| intervention | raw_diff_norm | pixels changed | energy delta | support ΔL1 | verdict change |
| --- | --- | --- | --- | --- | --- |
| cue_swap (eps=1.0) | 206.347756 | 691200 | −0.344727 | 0.000972 | compatible → compatible |
| no_information (matched noise) | 458.186631 | 5529599 | −372.916260 | 0.020274 | compatible → compatible |
| exposure_shift (gain=2.0) | 893.285036 | 5529597 | +6105.289062 | 0.024925 | **compatible → model_inadequate** |
| noise_shift (sigma_scale=1.0) | 23.987062 | 5529591 | +0.459717 | 0.000703 | compatible → compatible |
| translation_shift (8, −10) | 255.407013 | 5529592 | −1137.067627 | 0.006680 | compatible → compatible |

* preview: 33 SVG files + `figures.json` in `<run>/preview/`; the run is published in
  `<run>/status.json` with 12 structured entries, and the monitor serves them
  (`/api/preview?...` → `http=200 type=image/svg+xml bytes=322071`), 9 tiles with
  raw/rgb/attention/residual slots.
* every figure carries `episode=open_laptop/episode_000044 | camera=… | frame=36 |
  iteration=inner round 3 (final accepted) | model step=unknown` (unknown: no checkpoint file).

## LIBERO real forward/backward + protocol (GPU1, `libero_protocol_probe.py`, 51.8 s, peak 21.68 GiB)

* cameras `libero_agentview` + `libero_wrist` (2), measurement keys `agentview_raw_uint8`,
  `wrist_raw_uint8`; loaded RAW is float32 in **[0, 1]** (uint8/255 by the loader), shape
  `(8, 3, 256, 256)` per camera, causal window frames `[112…119]`.
* proprio **8-dim** (`robot_states[t]`, `proprio_available = true`), supervision actions
  `(10, 7)`, `action_dim = 7`, `action_horizon = 10`, `action_offset = 0`; window options
  `observation_stride 1, rnn_frames 8, burst_frames 1`; the cached 92-dim `states` array is
  never a policy input (loader `state_notes`, `state_source_key = robot_states`).
* replan cadence measured: `needs_replan(0, 0/4/5) = [false, false, true]` (every 5 steps).
* real forward/backward over the model's own tokens (no detach): flow-matching action loss
  **0.4956688**, `tokens.requires_grad = true`, per-role gradient L1 —
  encoder 36370.63 (348 tensors), readout 55737.64 (72), proposal 43.30 (9), corrector 75.58 (17),
  predictor 112.18 (32), decoder 0.0254 (29), measurement 0 (no parameters).
  A first pass that detaches the tokens only exercises the frozen bridge and reported 0
  everywhere; that run is kept in the receipt as the negative control.
* one real sample: `control_actions_shape [1, 10, 7]` == declared shape, 10 denoise steps,
  finite; first row vs the loader reference row both recorded in the JSON.

## Not run / open

* no trained N02 v1.2 checkpoint existed, so the numbers are from freshly initialised weights;
  the receipt says so and no capability claim is made.
* `--dev-samples 3` / `--calibration-samples 6` / `--intervention-samples 1` were used to keep
  the first evidence run bounded; the CLI accepts larger counts.
* the module-level `make check` and pytest subsets are the only automated checks run here.
