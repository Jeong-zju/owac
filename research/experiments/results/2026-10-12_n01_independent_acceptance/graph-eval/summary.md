# N01 evaluation summary

- checkpoint: `outputs/n01-independent-audit/graph-smoke/checkpoint.pt`
- config fingerprint: `f93a9d2ba0e25638`
- git sha (evaluation run): `e33f478e72ca0f3720f4cc4a43ae22bc36b0ae01`
- checkpoint git sha: `e33f478e72ca0f3720f4cc4a43ae22bc36b0ae01`
- git sha matches: `True`
- config fingerprint matches: `True`
- suites: `all`
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
| F1_no_static_average_tradeoff | False | `{"delta_px": 0.7474110424518585, "fast_narrow_error_px": 2.9588179290294647, "still_narrow_error_px": 2.211406886577606}` |
| F2_not_max_window_average | False | `{"effective_history_frames": 4.147119858048179, "max_history": 8, "support_count_mean": 1.0781169008992997, "window_area": 25}` |
| F3_fast_edge_not_worse_than_static_baseline | False | `{"best_static_fast_error_px": 4.79825218518575, "candidate_fast_error_px": 2.742953300476074, "delta_px": -2.055298884709676}` |
| F4_support_actually_reorganises | False | `{"merge_count_sum": 20195.75, "split_count_sum": 16411.0}` |

`triggered = true` means the candidate's claim is *not* supported by that measurement; `null` means the comparison could not be computed from this run and is reported as such.

## Mechanism matrix (candidate)

| condition | seed | pos err (px) | over tol | first stable | recovery | proc p50/p95 (ms) | sample age | support width | history age | merge/split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_still_wide | 1235 | 2.187 | 0.333 | 0 | None | 27.42/30.56 | 2.063 | 1.128 | 3.004 | 879.500/730.000 |
| low_still_narrow | 1236 | 2.016 | 0.354 | 0 | None | 25.77/27.39 | 2.057 | 1.177 | 2.991 | 906.500/744.500 |
| low_slow_wide | 1237 | 2.159 | 0.438 | 0 | None | 27.41/28.61 | 2.059 | 1.161 | 2.994 | 920.500/760.000 |
| low_slow_narrow | 1238 | 1.320 | 0.208 | 0 | None | 27.34/28.86 | 2.052 | 1.144 | 2.974 | 854.000/692.500 |
| low_fast_wide | 1239 | 1.864 | 0.333 | 2 | None | 27.67/28.88 | 2.003 | 1.158 | 2.886 | 884.500/721.000 |
| low_fast_narrow | 1240 | 1.611 | 0.250 | 0 | None | 26.96/27.78 | 2.001 | 1.156 | 2.906 | 877.250/718.500 |
| medium_still_wide | 1241 | 3.248 | 0.479 | 0 | None | 24.93/26.70 | 1.250 | 1.007 | 1.499 | 906.750/734.750 |
| medium_still_narrow | 1242 | 1.449 | 0.250 | 0 | None | 25.39/26.66 | 1.276 | 1.008 | 1.548 | 866.500/709.500 |
| medium_slow_wide | 1243 | 1.712 | 0.250 | 0 | None | 25.26/26.33 | 1.256 | 1.006 | 1.514 | 902.000/740.500 |
| medium_slow_narrow | 1244 | 2.069 | 0.354 | 0 | None | 26.09/27.10 | 1.244 | 1.010 | 1.482 | 939.500/769.750 |
| medium_fast_wide | 1245 | 1.657 | 0.271 | 2 | None | 27.88/28.43 | 1.206 | 1.017 | 1.427 | 888.250/716.500 |
| medium_fast_narrow | 1246 | 1.881 | 0.354 | 0 | None | 27.64/28.80 | 1.209 | 1.019 | 1.412 | 910.750/726.750 |
| high_still_wide | 1247 | 5.010 | 0.604 | 9 | None | 27.65/29.46 | 0.338 | 1.000 | 0.350 | 984.000/782.000 |
| high_still_narrow | 1248 | 2.211 | 0.292 | 0 | None | 29.14/32.23 | 0.354 | 1.001 | 0.377 | 970.500/788.750 |
| high_slow_wide | 1249 | 3.155 | 0.396 | 0 | None | 28.89/30.42 | 0.345 | 1.001 | 0.364 | 964.500/778.500 |
| high_slow_narrow | 1250 | 4.548 | 0.646 | 0 | None | 28.10/29.30 | 0.326 | 1.000 | 0.326 | 995.500/797.500 |
| high_fast_wide | 1251 | 2.733 | 0.521 | 1 | None | 27.06/27.82 | 0.342 | 1.001 | 0.369 | 1001.000/806.750 |
| high_fast_narrow | 1252 | 2.959 | 0.438 | 0 | None | 26.32/27.71 | 0.350 | 1.001 | 0.357 | 938.000/763.250 |
| event_sudden_appear | 1253 | 1.106 | 0.146 | 0 | None | 25.40/27.63 | 1.256 | 1.006 | 1.504 | 908.250/739.500 |
| event_direction_reversal | 1254 | 2.454 | 0.292 | 0 | 1 | 29.05/41.91 | 1.243 | 1.012 | 1.461 | 887.000/722.250 |
| event_occlusion_release | 1255 | 1.629 | 0.333 | 1 | None | 28.07/28.64 | 1.242 | 1.006 | 1.502 | 896.000/725.250 |
| event_crossing | 1256 | 1.795 | 0.250 | 0 | None | 28.07/28.92 | 1.252 | 1.008 | 1.520 | 915.000/743.000 |

### Candidate aggregate

```json
{
  "conditions": 22,
  "history_age_mean_frames": 1.5803169350732456,
  "merge_count_mean": 917.9886363636364,
  "observed_frames_mean": 4.147119858048179,
  "over_tolerance_rate_mean": 0.35416666486046533,
  "position_error_px_mean": 2.3078293691981924,
  "processing_p50_ms_mean": 27.15929734098278,
  "processing_p95_ms_mean": 29.097309226596156,
  "sample_age_mean_frames": 1.2146924070336602,
  "split_count_mean": 745.9545454545455,
  "state_bytes_max": 1508176,
  "state_bytes_mean": 1508176.0,
  "support_count_mean": 1.0781169008992997,
  "support_width_mean_px": 1.0466974431818181
}
```

## Controlled-budget comparison

- tuning status: UNTUNED: these are the starting parameters named in the implementation prompt (AdamW, lr=3e-4, identical step count). No per-method hyper-parameter search was run.
- identical budget: {"aux_measurement_prediction": false, "batch_size": 4, "grad_clip": 1.0, "identical_batches": true, "length": 12, "location_mode": "huber", "location_sigma_px": 1.5, "lr": 0.0003, "optimizer": "AdamW", "training_steps": 40, "weight_decay": 0.0001}

| method | params | pos err (px) | over tol | first stable | step p50/p95 (ms) | train loss (first -> last) |
| --- | --- | --- | --- | --- | --- | --- |
| single_frame_raw_cnn | 10496 | 4.766 | 0.909 | per-condition | 2.90/2.97 | 3.23891 -> 3.43484 |
| fixed_average | 10496 | 4.733 | 0.909 | per-condition | 7.88/8.03 | 3.23918 -> 3.41944 |
| ema | 10496 | 4.728 | 0.909 | per-condition | 7.18/7.30 | 3.23915 -> 3.41771 |
| kalman | 10496 | 4.767 | 0.909 | per-condition | 7.39/7.45 | 3.23917 -> 3.41874 |
| motion_compensated | 10496 | 4.724 | 0.909 | per-condition | 23.47/23.64 | 3.23916 -> 3.41644 |
| conv_gru | 75168 | 4.815 | 0.920 | per-condition | 47.50/49.62 | 2.90062 -> 3.49122 |
| n01 | 116622 | 3.026 | 0.625 | per-condition | 538.80/579.59 | 2.86217 -> 1.51711 |
| n01_no_merge | 116622 | 3.026 | 0.636 | per-condition | 540.50/576.71 | 2.85053 -> 1.50507 |
| n01_fixed_horizon | 116622 | 3.027 | 0.625 | per-condition | 389.78/395.73 | 2.86217 -> 1.51711 |

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

