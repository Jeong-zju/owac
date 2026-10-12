# N01 evaluation summary

- checkpoint: `outputs/n01-smoke/checkpoint.pt`
- config fingerprint: `f93a9d2ba0e25638`
- git sha (evaluation run): `ef647d9e8cb11b75b4f932ae5f3383341edfd157
--untracked-files=no`
- checkpoint git sha: `ef647d9e8cb11b75b4f932ae5f3383341edfd157
--untracked-files=no`
- git sha matches: `True`
- config fingerprint matches: `True`
- suites: `mechanism`
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
| F1_no_static_average_tradeoff | False | `{"delta_px": 0.2674805174271264, "fast_narrow_error_px": 2.3858518401781716, "still_narrow_error_px": 2.1183713227510452}` |
| F2_not_max_window_average | False | `{"effective_history_frames": 1.0, "max_history": 8, "support_count_mean": 56049.72669566761, "window_area": 25}` |
| F3_fast_edge_not_worse_than_static_baseline | None | `{"best_static_fast_error_px": null, "candidate_fast_error_px": 2.3489816735188165, "delta_px": null}` |
| F4_support_actually_reorganises | False | `{"merge_count_sum": 9585.25, "split_count_sum": 42175.25}` |

`triggered = true` means the candidate's claim is *not* supported by that measurement; `null` means the comparison could not be computed from this run and is reported as such.

## Mechanism matrix (candidate)

| condition | seed | pos err (px) | over tol | first stable | recovery | proc p50/p95 (ms) | sample age | support width | history age | merge/split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_still_wide | 1235 | 2.314 | 0.396 | 0 | None | 168.87/178.90 | 0.000 | 2.519 | 0.541 | 427.000/1916.000 |
| low_still_narrow | 1236 | 2.147 | 0.375 | 0 | None | 167.43/170.24 | 0.000 | 2.507 | 0.535 | 420.500/1904.250 |
| low_slow_wide | 1237 | 2.809 | 0.479 | 10 | None | 166.37/168.45 | 0.000 | 2.509 | 0.543 | 425.750/1913.500 |
| low_slow_narrow | 1238 | 1.493 | 0.208 | 0 | None | 84.77/164.97 | 0.000 | 2.500 | 0.538 | 432.000/1920.000 |
| low_fast_wide | 1239 | 2.198 | 0.438 | 0 | None | 85.48/89.09 | 0.000 | 2.504 | 0.546 | 439.500/1922.750 |
| low_fast_narrow | 1240 | 1.923 | 0.292 | 0 | None | 83.06/88.63 | 0.000 | 2.515 | 0.544 | 424.000/1925.000 |
| medium_still_wide | 1241 | 2.017 | 0.396 | 0 | None | 85.26/101.61 | 0.000 | 2.523 | 0.554 | 454.250/1935.250 |
| medium_still_narrow | 1242 | 1.873 | 0.312 | 0 | None | 82.05/89.36 | 0.000 | 2.502 | 0.533 | 430.000/1930.250 |
| medium_slow_wide | 1243 | 2.045 | 0.375 | 1 | None | 82.59/92.65 | 0.000 | 2.508 | 0.547 | 456.750/1925.750 |
| medium_slow_narrow | 1244 | 2.602 | 0.438 | 0 | None | 83.40/87.91 | 0.000 | 2.509 | 0.548 | 421.500/1907.750 |
| medium_fast_wide | 1245 | 1.704 | 0.312 | 0 | None | 83.61/86.09 | 0.000 | 2.505 | 0.551 | 441.250/1932.750 |
| medium_fast_narrow | 1246 | 2.351 | 0.417 | 0 | None | 81.22/86.41 | 0.000 | 2.513 | 0.548 | 450.250/1930.000 |
| high_still_wide | 1247 | 4.641 | 0.708 | 10 | None | 80.50/82.59 | 0.000 | 1.434 | 0.550 | 429.250/1890.000 |
| high_still_narrow | 1248 | 2.335 | 0.354 | 0 | None | 81.65/86.95 | 0.000 | 1.455 | 0.541 | 430.250/1907.250 |
| high_slow_wide | 1249 | 3.384 | 0.396 | 0 | None | 79.86/81.71 | 0.000 | 1.483 | 0.547 | 458.500/1912.750 |
| high_slow_narrow | 1250 | 4.379 | 0.708 | 3 | None | 84.91/97.53 | 0.000 | 1.442 | 0.551 | 451.000/1902.250 |
| high_fast_wide | 1251 | 3.033 | 0.500 | 0 | None | 82.38/95.88 | 0.000 | 1.080 | 0.554 | 438.000/1914.000 |
| high_fast_narrow | 1252 | 2.883 | 0.479 | 4 | None | 77.80/82.53 | 0.000 | 1.099 | 0.555 | 431.750/1912.250 |
| event_sudden_appear | 1253 | 1.184 | 0.167 | 0 | None | 81.77/85.66 | 0.000 | 2.511 | 0.536 | 425.250/1913.500 |
| event_direction_reversal | 1254 | 2.530 | 0.375 | 0 | 1 | 78.01/82.13 | 0.000 | 2.508 | 0.553 | 439.000/1921.000 |
| event_occlusion_release | 1255 | 2.139 | 0.333 | 0 | None | 77.70/81.42 | 0.000 | 2.506 | 0.548 | 429.750/1917.750 |
| event_crossing | 1256 | 2.082 | 0.396 | 0 | None | 77.61/79.83 | 0.000 | 2.509 | 0.540 | 429.750/1921.250 |

### Candidate aggregate

```json
{
  "conditions": 22,
  "history_age_mean_frames": 0.5455718230117451,
  "merge_count_mean": 435.6931818181818,
  "observed_frames_mean": 1.0,
  "over_tolerance_rate_mean": 0.40246211940591986,
  "position_error_px_mean": 2.4575923078439454,
  "processing_p50_ms_mean": 93.46745863633095,
  "processing_p95_ms_mean": 102.75261863634196,
  "sample_age_mean_frames": 0.0,
  "split_count_mean": 1917.0568181818182,
  "state_bytes_max": 1049424,
  "state_bytes_mean": 1049424.0,
  "support_count_mean": 56049.72669566761,
  "support_width_mean_px": 2.188232421875
}
```

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

