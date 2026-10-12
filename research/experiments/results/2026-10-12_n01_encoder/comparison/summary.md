# N01 evaluation summary

- checkpoint: `outputs/n01-smoke/checkpoint.pt`
- config fingerprint: `f93a9d2ba0e25638`
- git sha (evaluation run): `6fc354fff103dc77a19c68bca59b9769e7e5e8cd
--untracked-files=no`
- checkpoint git sha: `6fc354fff103dc77a19c68bca59b9769e7e5e8cd
--untracked-files=no`
- git sha matches: `True`
- config fingerprint matches: `True`
- suites: `comparison`
- device: `cpu`, torch `2.9.1+cpu`

## Falsification criteria (declared before the run)

- **F1_no_static_average_tradeoff** — On the narrow-tolerance, high-SNR slice the candidate must not trade fast thin-edge accuracy for static averaging: mean position error on 'fast' must exceed the mean on 'still' by more than 1.5 px.
  - measure: position_error_px (pixels)
  - threshold: delta(fast - still) > 1.5 px on the high/narrow slice
- **F2_not_max_window_average** — The candidate must not degenerate into a maximum-window average: the effective history length must not sit within 5% of max_history, and the mean weighted support count must stay below the full (2r+1)^2 window mass.
  - measure: effective_history_frames (frames), support_count_mean (weighted samples)
  - threshold: history within 5% of max_history AND support_count >= 0.9 * window area
- **F3_fast_edge_not_worse_than_static_baseline** — The candidate must not be worse than the strongest static-history baseline (fixed average over the same history) on the fast+motion slice by more than 2 px.
  - measure: position_error_px on motion=fast
  - threshold: candidate_error - best_static_error > 2.0 px
- **F4_support_actually_reorganises** — The support graph must actually reorganise: either merge_count or split_count must be non-zero in the mechanism matrix, otherwise the hysteresis band is inert.
  - measure: merge/split counts from the encoder diagnostics
  - threshold: merge_count == 0 AND split_count == 0 on every condition

## Falsification results (measured)

| id | triggered | measured |
| --- | --- | --- |

`triggered = true` means the candidate's claim is *not* supported by that measurement; `null` means the comparison could not be computed from this run and is reported as such.

## Controlled-budget comparison

- tuning status: UNTUNED: these are the starting parameters named in the implementation prompt (AdamW, lr=3e-4, identical step count). No per-method hyper-parameter search was run.
- identical budget: {"aux_measurement_prediction": false, "batch_size": 4, "grad_clip": 1.0, "identical_batches": true, "length": 12, "location_mode": "huber", "location_sigma_px": 1.5, "lr": 0.0003, "optimizer": "AdamW", "training_steps": 40, "weight_decay": 0.0001}

| method | params | pos err (px) | over tol | first stable | step p50/p95 (ms) | train loss (first -> last) |
| --- | --- | --- | --- | --- | --- | --- |
| single_frame_raw_cnn | 10496 | 4.748 | 0.909 | per-condition | 4.04/8.46 | 3.07971 -> 3.44972 |
| fixed_average | 10496 | 4.680 | 0.920 | per-condition | 9.81/12.11 | 2.94359 -> 3.39531 |
| ema | 10496 | 4.722 | 0.909 | per-condition | 8.72/10.11 | 2.98936 -> 3.43750 |
| kalman | 10496 | 4.665 | 0.909 | per-condition | 9.51/10.59 | 3.06986 -> 3.42482 |
| motion_compensated | 10496 | 4.317 | 0.875 | per-condition | 32.84/35.07 | 3.10678 -> 3.15441 |
| conv_gru | 75168 | 4.812 | 0.920 | per-condition | 50.92/123.56 | 2.99737 -> 3.45717 |
| n01 | 116430 | 3.687 | 0.830 | per-condition | 1627.03/2787.80 | 3.24142 -> 2.55862 |
| n01_no_merge | 116430 | 3.307 | 0.670 | per-condition | 1562.49/2655.03 | 3.18475 -> 2.42890 |
| n01_fixed_horizon | 116430 | 3.458 | 0.807 | per-condition | 457.89/41026.99 | 3.66230 -> 2.70668 |

## Units

- `location_heatmap`: logits over the HxW measurement grid; cross-entropy target is a Gaussian heatmap with sigma in pixels
- `location_huber`: pixel units: mean over the diagonal of |pred_xy - target_xy|
- `position_error`: pixels, row/col convention matching target_xy (normalised in [0,1])
- `measurement_prediction`: normalised measurement units (x=(raw-black)/(white-black)); target is the held-out pixel block of frame t-1, input replacement is a neighbourhood mean
- `budget`: effective update count per step minus the declared target update count
- `latency_ms`: milliseconds of wall time on this machine; the sample age (algorithmic lag) is reported separately and is never mixed into this number
- `memory_bytes`: bytes

## Not measured in this run

- closed-loop robot success rate (no robot or simulator in this run)
- real camera RAW data (synthetic weak-edge task only)
- hard real-time guarantees: --deadline_ms is an optional soft budget, not polled

