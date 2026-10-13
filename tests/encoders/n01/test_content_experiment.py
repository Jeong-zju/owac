"""Input, causal-oracle, checkpoint-comparison and transient metric contracts."""

import json
from pathlib import Path

import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.evaluation.n01_content import (
    METHODS,
    make_matched_models,
    oracle_translation,
    recovery_delays,
    run_sequence,
    summarize_sequence,
)
from owac.evaluation.n01_content_data import CONDITIONS, make_batch, split_seed


def protocol():
    path = Path(__file__).parents[3] / "configs/encoders/n01/content_mechanism.json"
    # tests live three parents below checkout root.
    result = json.loads(path.read_text())
    result.update(batch_size=1, evaluation_batch_size=1, methods=list(METHODS))
    return result


def test_prefix_does_not_change_when_future_length_or_event_changes():
    first = make_batch("local_change_stale_history", seed=12, length=24, event_frame=12)
    longer = make_batch("local_change_stale_history", seed=12, length=28, event_frame=15)
    assert torch.equal(first.values[:, :12], longer.values[:, :12])
    assert torch.equal(first.target_xy[:, :12], longer.target_xy[:, :12])


def test_noise_is_independent_across_cfa_samples_and_matched_across_conditions():
    noisy = make_batch("static_weak", seed=5, noise_sigma=0.1)
    clean = make_batch("static_weak", seed=5, noise_sigma=0)
    moving = make_batch("translation_oracle_alignment", seed=5, noise_sigma=0.1)
    moving_clean = make_batch("translation_oracle_alignment", seed=5, noise_sigma=0)
    noise = noisy.values - clean.values
    assert not torch.equal(noise[:, :, 1], noise[:, :, 2])
    assert torch.allclose(noise, moving.values - moving_clean.values, atol=3e-8)
    assert not torch.equal(clean.values[:, :, 1], clean.values[:, :, 2])
    neighbor = make_batch("static_weak", seed=6, noise_sigma=0.1)
    neighbor_clean = make_batch("static_weak", seed=6, noise_sigma=0)
    assert not torch.allclose(noise[:, 1:], (neighbor.values - neighbor_clean.values)[:, :-1])


def test_events_are_observed_physical_changes_not_oracle_invalid_masks():
    occluded = make_batch("occlusion_release", seed=7)
    changed = make_batch("local_change_stale_history", seed=7)
    assert occluded.valid.all() and changed.valid.all()
    assert not occluded.visible[:, 8:12].any()
    assert occluded.visible[:, 12:].all()
    assert changed.visible.all()
    assert ((changed.target_xy[:, 12] - changed.target_xy[:, 11]).norm(dim=1) > 4).all()


def test_split_seeds_are_disjoint_and_do_not_depend_on_method():
    seeds = {
        split_seed(seed, split, index)
        for seed in (17, 29, 43)
        for split in ("train", "validation", "test", "calibration")
        for index in range(8)
    }
    assert len(seeds) == 3 * 4 * 8


def test_recovery_is_per_episode_requires_consecutive_good_and_counts_censoring():
    errors = torch.tensor([[3.0, 0.2, 0.3, 0.4, 2.0], [3.0, 0.1, 2.0, 0.1, 0.1]])
    visible = torch.ones_like(errors, dtype=torch.bool)
    assert recovery_delays(errors, visible, event=1, tolerance=0.75, consecutive=3) == [0, None]
    visible[0, 2] = False
    assert recovery_delays(errors, visible, event=1, tolerance=0.75, consecutive=3) == [None, None]


def test_shared_initial_weights_calibration_and_capacity_modes_are_matched():
    models, calibration = make_matched_models(protocol(), 17)
    reference = dict(models["D"].named_parameters())
    for method, model in models.items():
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter, reference[name]), (method, name)
        assert torch.equal(
            model.encoder.normalization_scales, models["D"].encoder.normalization_scales
        )
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    assert counts["A"] == counts["B"] < counts["C"]
    assert len({counts[name] for name in "CDEF"}) == 1
    assert calibration["reference_time_depths"] == [1, 4, 8]


def test_oracle_alignment_uses_past_translation_and_restores_estimator():
    models, _ = make_matched_models(protocol(), 17)
    encoder = models["A"].encoder
    batch = make_batch("translation_oracle_alignment", seed=4, batch_size=1, noise_sigma=0)
    state = encoder.init_state(1, {"camera_ids": ["camera"], "shape": (16, 16)})
    previous = batch.values[:, 7]
    state.values[0] = previous
    state.valid[0] = True
    state.observed[0] = True
    state.timestamps[0] = 7
    frame = MeasurementFrame(values=batch.values[:, 8], timestamp=torch.tensor([8.0]))
    assert encoder.correspondence is None
    with oracle_translation(encoder, batch):
        fields = encoder._estimate_motion(state, frame)
        warped, _, valid = encoder.correspondence.align(
            previous, batch.variance[:, 7], batch.valid[:, 7], fields[1]
        )
        assert torch.allclose(warped[valid], batch.values[:, 8][valid], atol=1e-7)
        assert not valid.all()  # boundary mask, never wraparound
        # Check the actual integrator dispatch too: a field alone does not prove
        # its alignment is enabled by the core's configuration guard.
        assert encoder.integrator.config.alignment == "fixed"
        shifted = encoder.integrator.aligned_slots(
            state, [fields[1]] * state.num_slots, encoder.correspondence
        )
        shifted_valid = shifted[2][0].bool()
        assert torch.allclose(
            shifted[0][0][shifted_valid], batch.values[:, 8][shifted_valid], atol=1e-7
        )
    assert encoder.correspondence is None
    assert encoder.integrator.config.alignment == "none"


def test_transient_metric_excludes_burnin_and_keeps_failed_recovery():
    batch = make_batch("local_change_stale_history", seed=3, batch_size=2)
    errors = torch.full((2, 24), 2.0)
    errors[0, 12:15] = 0.2
    result = summarize_sequence(errors, batch.visible, [8.0] * 24, batch, protocol())
    assert result["recovery"]["censored_count"] == 1
    assert result["recovery"]["denominator"] == 2
    assert result["recovery"]["capped_delay_score"] == 6
    assert not torch.tensor(result["supervised_visible_per_member_per_frame"])[:, :8].any()


def test_full_causal_training_sequence_reaches_content_branch_without_scale_drift():
    settings = protocol()
    settings["methods"] = ["D"]
    model = make_matched_models(settings, 17)[0]["D"]
    before = model.encoder.normalization_scales.clone()
    batch = make_batch(CONDITIONS[0], seed=4, batch_size=1)
    loss, summary = run_sequence(model, batch, settings, training=True)
    loss.backward()
    gradients = [p.grad for name, p in model.named_parameters() if "content_update" in name]
    assert gradients and any(grad is not None and grad.abs().sum() > 0 for grad in gradients)
    assert torch.equal(before, model.encoder.normalization_scales)
    assert len(summary["error_per_frame"]) == 24
