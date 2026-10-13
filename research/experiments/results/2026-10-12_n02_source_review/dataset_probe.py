import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from owac.frameworks.n02.synthetic import (
    SyntheticGenerator,
    SyntheticOptions,
    marker_xy,
    values_for_task,
)

Path("outputs/n02-source-review").mkdir(parents=True, exist_ok=True)

rows = []
for seed in (100, 101, 102, 103):
    options = SyntheticOptions(height=64, width=64, noise_sigma=0.01)
    gen = SyntheticGenerator(options, seed=seed, split="validation")
    ep = gen.generate(seed=seed, num_frames=8, prior_condition="correct")
    truth = ep.geometry_before.value
    other = values_for_task(ep.task)[1 - values_for_task(ep.task).index(truth)]
    other_scene = ep.geometry_before.with_value(other)
    alternate = replace(ep, geometry_before=other_scene, geometry_after=other_scene)
    clean_true = gen.clean_measurements(ep)
    clean_other = gen.clean_measurements(alternate)
    old_atten = ep.occluder_mask[None] * ep.transmission[:, None, None]
    corrected_atten = 1 - ep.occluder_mask[None] * (1 - ep.transmission[:, None, None])
    delta = clean_true - clean_other
    old_gain = (delta * old_atten[:, None]) ** 2 / (2 * options.noise_sigma**2)
    corrected_gain = (delta * corrected_atten[:, None]) ** 2 / (2 * options.noise_sigma**2)
    y, x = marker_xy(ep.geometry_before)
    rr, cc = np.mgrid[:64, :64]
    disk = (rr + 0.5 - y) ** 2 + (cc + 0.5 - x) ** 2 <= (3 * ep.geometry_before.cue_sigma) ** 2
    clear = replace(ep, cue_strength=np.zeros_like(ep.cue_strength))
    actual_cue = (clean_true - gen.clean_measurements(clear)) * old_atten[:, None]
    old_snr = float(old_gain.sum())
    new_snr = float(corrected_gain.sum())
    old_true = clean_true * old_atten[:, None]
    old_other = clean_other * old_atten[:, None]
    observed_gain = float(
        (
            ((ep.frames - old_other) ** 2 - (ep.frames - old_true) ** 2)
            / (2 * options.noise_sigma**2)
        ).sum()
    )
    actual_var = (old_atten[:, None] * options.noise_sigma) ** 2
    actual_mask = actual_var > 0
    actual_variance_gain = float(
        np.where(
            actual_mask,
            ((ep.frames - old_other) ** 2 - (ep.frames - old_true) ** 2)
            / (2 * np.maximum(actual_var, 1e-300)),
            0,
        ).sum()
    )

    def posterior(gain):
        odds = -math.log(19) + gain
        return 1 / (1 + math.exp(-min(odds, 700)))

    rows.append(
        {
            "seed": seed,
            "identifiable_flag": ep.identifiable,
            "arrival": ep.evidence_arrival_step,
            "cue_strength": ep.cue_strength.tolist(),
            "raw_sha256": hashlib.sha256(ep.frames.tobytes()).hexdigest(),
            "marker_yx": [y, x],
            "marker_disk_sample_count": int(disk.sum()),
            "mask_in_disk_range": [
                float(ep.occluder_mask[disk].min()),
                float(ep.occluder_mask[disk].max()),
            ],
            "last_frame_marker_absmax": float(np.abs(ep.frames[-1, :, disk]).max()),
            "declared_cue_last_peak": float(clean_true[-1, :, disk].max()),
            "surviving_cue_absmax": float(np.abs(actual_cue).max()),
            "expected_full_sequence_log_likelihood_gain_actual_attenuation": old_snr,
            "expected_full_sequence_log_likelihood_gain_corrected_attenuation": new_snr,
            "true_mass_at_expected_log_odds_actual": posterior(old_snr),
            "true_mass_at_expected_log_odds_corrected": posterior(new_snr),
            "observed_log_likelihood_gain_declared_variance": observed_gain,
            "oracle_true_mass_declared_variance": posterior(observed_gain),
            "observed_log_likelihood_gain_actual_spatial_variance": actual_variance_gain,
            "oracle_true_mass_actual_spatial_variance": posterior(actual_variance_gain),
            "variance_declared": options.noise_sigma**2,
            "actual_noise_variance_formula": (
                "(mask * transmission * noise_sigma)^2; zero in cue disk"
            ),
        }
    )
report = {
    "scope": (
        "independent dataset signal probe; corrected attenuation is a mathematical "
        "counterfactual, not a trained model evaluation"
    ),
    "expected_gain_definition": (
        "sum_{t,c,pixel} (mu_true - mu_other)^2 / (2 * declared_sensor_variance); "
        "noise expectation, known geometry used by evaluator only"
    ),
    "rows": rows,
}
Path("outputs/n02-source-review/dataset.json").write_text(json.dumps(report, indent=2) + "\n")
for row in rows:
    print(
        {
            k: row[k]
            for k in (
                "seed",
                "marker_disk_sample_count",
                "last_frame_marker_absmax",
                "declared_cue_last_peak",
                "surviving_cue_absmax",
                "expected_full_sequence_log_likelihood_gain_actual_attenuation",
                "expected_full_sequence_log_likelihood_gain_corrected_attenuation",
                "observed_log_likelihood_gain_declared_variance",
                "oracle_true_mass_declared_variance",
                "oracle_true_mass_actual_spatial_variance",
            )
        }
    )
