# N01 evaluation summary

- checkpoint: `outputs/n01-source-repair-20261013/smoke200/checkpoint.pt`
- config fingerprint: `f93a9d2ba0e25638`
- git sha (evaluation run): `d4bf84c5a73a4fa97bcb9319f2aa845836c38994`
- checkpoint git sha: `d4bf84c5a73a4fa97bcb9319f2aa845836c38994`
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
| F1_no_static_average_tradeoff | False | `{"delta_px": 0.18470078706741333, "fast_narrow_error_px": 1.9356844872236252, "still_narrow_error_px": 1.7509837001562119}` |
| F2_not_max_window_average | False | `{"effective_history_frames": 3.550200115550648, "max_history": 8, "support_count_mean": 1.0784579795543026, "window_area": 25}` |
| F3_fast_edge_not_worse_than_static_baseline | None | `{"best_static_fast_error_px": null, "candidate_fast_error_px": 2.8955009058117867, "delta_px": null}` |
| F4_support_actually_reorganises | False | `{"merge_count_sum": 21613.75, "split_count_sum": 17605.5}` |

`triggered = true` means the candidate's claim is *not* supported by that measurement; `null` means the comparison could not be computed from this run and is reported as such.

## Mechanism matrix (candidate)

| condition | seed | pos err (px) | over tol | first stable | recovery | proc p50/p95 (ms) | sample age | support width | history age | merge/split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_still_wide | 1235 | 4.099 | 0.688 | 11.0 | None | 35.88/40.09 | 1.832 | 1.175 | 2.528 | 936.000/758.500 |
| low_still_narrow | 1236 | 3.913 | 1.000 | None | None | 28.45/33.07 | 1.823 | 1.183 | 2.512 | 970.750/786.500 |
| low_slow_wide | 1237 | 4.306 | 0.688 | 10.0 | None | 39.77/57.88 | 1.825 | 1.177 | 2.522 | 964.000/784.250 |
| low_slow_narrow | 1238 | 3.939 | 0.854 | None | None | 38.47/57.01 | 1.830 | 1.181 | 2.531 | 976.000/794.250 |
| low_fast_wide | 1239 | 4.251 | 0.667 | 10.0 | None | 37.21/58.24 | 1.835 | 1.167 | 2.541 | 938.750/774.000 |
| low_fast_narrow | 1240 | 3.642 | 0.812 | None | None | 40.56/50.99 | 1.837 | 1.146 | 2.557 | 948.750/794.750 |
| medium_still_wide | 1241 | 3.313 | 0.500 | 11.0 | None | 46.76/55.90 | 1.030 | 1.017 | 1.202 | 949.000/776.000 |
| medium_still_narrow | 1242 | 1.933 | 0.771 | 11.0 | None | 37.17/54.16 | 1.032 | 1.020 | 1.210 | 965.750/784.250 |
| medium_slow_wide | 1243 | 3.052 | 0.500 | 8.0 | None | 48.85/62.07 | 1.019 | 1.015 | 1.190 | 980.500/797.750 |
| medium_slow_narrow | 1244 | 2.847 | 0.938 | None | None | 44.07/53.42 | 1.017 | 1.021 | 1.177 | 965.250/775.500 |
| medium_fast_wide | 1245 | 2.259 | 0.354 | 9.0 | None | 40.10/55.41 | 0.975 | 1.018 | 1.115 | 959.000/784.000 |
| medium_fast_narrow | 1246 | 3.543 | 0.938 | None | None | 37.81/41.53 | 0.971 | 1.028 | 1.108 | 956.000/763.250 |
| high_still_wide | 1247 | 2.574 | 0.479 | 11.0 | None | 33.68/39.29 | 0.282 | 1.000 | 0.268 | 1024.500/841.750 |
| high_still_narrow | 1248 | 1.751 | 0.750 | 4.0 | None | 31.23/39.48 | 0.302 | 1.000 | 0.295 | 1041.750/847.000 |
| high_slow_wide | 1249 | 2.063 | 0.333 | 4.0 | None | 37.12/40.37 | 0.267 | 1.000 | 0.261 | 1032.000/827.500 |
| high_slow_narrow | 1250 | 2.339 | 0.792 | None | None | 37.30/43.56 | 0.258 | 1.000 | 0.239 | 1028.500/836.250 |
| high_fast_wide | 1251 | 1.743 | 0.250 | 4.0 | None | 39.57/42.07 | 0.228 | 1.000 | 0.208 | 1050.500/861.250 |
| high_fast_narrow | 1252 | 1.936 | 0.833 | None | None | 33.40/38.74 | 0.228 | 1.000 | 0.212 | 1068.000/857.500 |
| event_sudden_appear | 1253 | 1.105 | 0.042 | 6.5 | None | 32.20/41.75 | 1.105 | 1.011 | 1.141 | 1008.750/836.000 |
| event_direction_reversal | 1254 | 2.310 | 0.396 | 6.333333333333333 | 3.3333333333333335 | 36.07/40.24 | 1.025 | 1.015 | 1.185 | 937.250/760.250 |
| event_occlusion_release | 1255 | 1.740 | 0.219 | 8.75 | None | 31.08/33.92 | 0.857 | 1.017 | 1.134 | 982.250/796.750 |
| event_crossing | 1256 | 2.184 | 0.396 | 7.0 | None | 27.93/32.23 | 0.932 | 1.013 | 1.055 | 930.500/768.250 |

### Candidate aggregate

```json
{
  "conditions": 22,
  "history_age_mean_frames": 1.281470319086855,
  "merge_count_mean": 982.4431818181819,
  "observed_frames_mean": 3.550200115550648,
  "over_tolerance_rate_mean": 0.5999053044413979,
  "position_error_px_mean": 2.7654446878216485,
  "processing_p50_ms_mean": 37.0312290456654,
  "processing_p95_ms_mean": 45.97435836463278,
  "sample_age_mean_frames": 1.0232735079797832,
  "split_count_mean": 800.25,
  "state_bytes_max": 1508176,
  "state_bytes_mean": 1508176.0,
  "support_count_mean": 1.0784579795543026,
  "support_width_mean_px": 1.0547318892045454
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
