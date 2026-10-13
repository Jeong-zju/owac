"""Behavioral regressions for the seven grouped 2026-10-13 source findings."""

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.evaluation.n01 import build_conditions, evaluate_probe
from owac.evaluation.n01_trajectories import TRAJECTORIES, build_encoder, run_trajectory
from owac.frameworks.n01.config import N01Config
from owac.frameworks.n01.model import N01Encoder, calibrate_statistics, encode_evidence_statistics
from owac.frameworks.n01.motion import MotionField
from owac.frameworks.n01.sequence1d import HeteroscedasticSequence1DTask
from owac.frameworks.n01.support import SupportDomainIntegrator
from owac.frameworks.n01.synthetic import _SNR_SETTINGS, SyntheticWeakEdgeTask
from owac.runtime import n01_feature_service as service_module
from owac.training.n01_action import build_bridge


def config(**kwargs):
    return N01Config.tiny(width=8, num_blocks=1, readout_tokens=8, readout_width=16, **kwargs)


def test_gate_units_are_bounded_differentiable_and_invertible():
    raw = torch.tensor([0.02, 0.0001, 2, 100000, 5, 9]).reshape(1, 6, 1, 1).requires_grad_()
    scales = raw.detach().flatten() * 2
    coded = encode_evidence_statistics(raw, scales)
    assert torch.allclose(coded, torch.full_like(raw, 1 / 3))
    restored = coded * scales.reshape(1, 6, 1, 1) / (1 - coded.abs())
    assert torch.allclose(restored, raw)
    coded.sum().backward()
    assert bool((raw.grad > 0).all())


def test_real_gate_uses_all_statistics_without_precision_saturation():
    torch.manual_seed(0)
    cfg = config()
    encoder = N01Encoder(cfg)
    training = SyntheticWeakEdgeTask(cfg, seed=500000).sample(2, 8)
    calibrate_statistics(encoder, training)
    sample = SyntheticWeakEdgeTask(cfg, seed=50).sample(2, 8)
    captures = []
    hook = encoder.blocks[0].gate.register_forward_hook(
        lambda m, a, out: captures.append(out.detach())
    )
    from owac.baselines.n01 import N01Probe

    probe = N01Probe(cfg)
    probe.encoder = encoder
    with torch.no_grad():
        original = probe.encode(sample.values, sample.valid, sample.variance, sample.timestamps)
    hook.remove()
    assert all(not bool(((out <= 1e-6) | (out >= 1 - 1e-6)).any()) for out in captures)
    raw_mass = encoder.last_statistics.support_count.clone()

    def erase(module, args):
        feature, statistics, graph, mask = args
        altered = torch.zeros_like(statistics)
        altered[:, 3] = statistics[:, 3]
        return feature, altered, graph, mask

    hook = encoder.blocks[0].register_forward_pre_hook(erase)
    probe.reset_state()
    with torch.no_grad():
        erased = probe.encode(sample.values, sample.valid, sample.variance, sample.timestamps)
    hook.remove()
    assert float((original - erased).abs().max()) > 1e-4
    assert torch.equal(raw_mass, encoder.last_statistics.support_count)


def test_trajectory_calibrates_independent_training_and_keeps_frozen_scales():
    cfg = config(alignment="candidate", fractional_alignment=True)
    encoder = build_encoder(cfg, seed=11)
    weights = {name: parameter.clone() for name, parameter in encoder.named_parameters()}
    first = run_trajectory(TRAJECTORIES[0], config=cfg, encoder=encoder)
    assert first.config["calibration_source"] == "independent_training_low_slow_seed_500000"
    assert int(encoder.calibration_count) == 8
    fitted = encoder.normalization_scales.clone()
    second = run_trajectory(TRAJECTORIES[1], config=cfg, encoder=encoder)
    assert second.config["calibration_source"] == "retained_encoder_training_scales"
    assert torch.equal(fitted, encoder.normalization_scales)
    assert all(
        torch.equal(parameter, weights[name]) for name, parameter in encoder.named_parameters()
    )


def test_motion_gate_consumes_nearest_history_and_tracks_missing_per_batch():
    integrator = SupportDomainIntegrator(config(learnable_gate=False))
    identity = MotionField(
        torch.zeros(2, 2), torch.zeros(2), torch.ones(2), torch.ones(2, dtype=torch.bool)
    )
    history = MotionField(
        torch.zeros(2, 2), torch.full((2,), 100.0), torch.ones(2), torch.tensor([True, False])
    )
    gate, missing = integrator._motion_gate([identity, history], 2, torch.zeros(2, 1, 4, 4))
    assert float(gate[0].mean()) == pytest.approx(math.exp(-35), rel=1e-5)
    assert float(gate[1].mean()) == 1
    assert missing == 0.5
    gate, missing = integrator._motion_gate([identity], 2, torch.zeros(2, 1, 4, 4))
    assert missing == 1 and bool((gate == 1).all())


@pytest.mark.parametrize(
    "event", ["sudden_appear", "direction_reversal", "occlusion_release", "crossing"]
)
def test_augmented_events_change_actual_measurements_and_geometry(event):
    kwargs = dict(snr="medium", motion="slow", seed=42, dead_pixel_rate=0)
    reference = SyntheticWeakEdgeTask(config(), **kwargs).sample(2, 12)
    batch = SyntheticWeakEdgeTask(config(), augment=event, **kwargs).sample(2, 12)
    assert not torch.equal(batch.values, reference.values)
    assert batch.meta["event"] == event
    if event == "sudden_appear":
        assert not bool(batch.target_visible[:, :6].any())
        assert bool(batch.target_visible[:, 6:].all())
    elif event == "occlusion_release":
        assert not bool(batch.target_visible[:, 4:8].any())
        assert bool(batch.target_visible[:, :4].all() & batch.target_visible[:, 8:].all())
    elif event == "direction_reversal":
        velocity = batch.target_xy[:, 1:, 1] - batch.target_xy[:, :-1, 1]
        assert bool((velocity[:, 0] * velocity[:, -1] < 0).all())
        assert float(velocity.abs().max()) < 0.02  # Continuous turn, no jump.
    else:
        relative = batch.target_xy[..., 1] - batch.meta["second_target_xy"][..., 1]
        assert bool((relative[:, 0] * relative[:, -1] < 0).all())


def test_declared_and_observed_snr_order_in_both_input_structures():
    declared, observed = [], []
    for index, (name, (photon, read, contrast)) in enumerate(_SNR_SETTINGS.items()):
        declared.append(contrast / math.sqrt(photon * contrast + read**2))
        task = SyntheticWeakEdgeTask(config(), snr=name, dead_pixel_rate=0)
        truth = torch.full((2, 2, 64, 64), contrast)
        values, _, _, _ = task._measure(truth, torch.Generator().manual_seed(index), photon, read)
        observed.append(contrast / float((values[:, :, 0] - truth).square().mean().sqrt()))
    assert declared[0] < declared[1] < declared[2]
    assert observed[0] < observed[1] < observed[2]
    sequence_snr = [
        c / math.sqrt(p * c + r**2)
        for p, r, c in HeteroscedasticSequence1DTask.SNR_SETTINGS.values()
    ]
    assert sequence_snr[0] < sequence_snr[1] < sequence_snr[2]


def test_condition_tolerances_and_explicit_override():
    for condition in build_conditions():
        assert condition.tolerance_px == (2.5 if condition.tolerance == "wide" else 0.75)
        assert condition.tolerance_source == "condition_specific"
    assert all(
        c.tolerance_px == 2 and c.tolerance_source == "explicit_override"
        for c in build_conditions(tolerance_px=2)
    )


def test_compact_cfa_has_real_spatial_neighbours_without_extra_measurements():
    cfg = config(
        input_mode="bayer_cfa",
        channels=4,
        planes=4,
        learnable_gate=False,
        spatial_radii=(0, 1),
        neighbourhood_radius=1,
    )
    task = SyntheticWeakEdgeTask(cfg, image_size=8, dead_pixel_rate=0)
    batch = task.sample(1, 1)
    assert batch.values.shape == (1, 1, 4, 4, 4)
    assert int(batch.valid.sum()) == 64
    assert batch.meta["spec"]["layout"] == "compact_2x2_phase_planes"
    encoder = N01Encoder(cfg)
    state = encoder.init_state(1, {"camera_ids": ["camera"], "shape": (4, 4)})
    frame = MeasurementFrame(
        batch.values[:, 0],
        valid_mask=batch.valid[:, 0],
        camera_id="camera",
        timestamp=torch.zeros(1),
        metadata={"variance": batch.variance[:, 0]},
        sample_xy=task.sample_xy(),
    )
    output = encoder.step(frame, state=state)
    assert int(torch.count_nonzero(output.state.support["camera"].edges)) > 0
    with pytest.raises(ValueError, match="even"):
        SyntheticWeakEdgeTask(cfg, image_size=7)


def test_endpoint_stability_is_unmeasured_and_batch_order_invariant(monkeypatch):
    import owac.evaluation.n01 as evaluation

    cfg = config()
    batch = SyntheticWeakEdgeTask(cfg, image_size=8).sample(4, 2)
    batch.values.fill_(-30)
    batch.values[:, -1, :, 0, 0] = 30
    batch.values[0, -1].fill_(-30)
    batch.values[0, -1, :, 7, 7] = 30
    batch.target_xy.zero_()
    batch.target_visible.fill_(True)
    source = SimpleNamespace(
        SyntheticWeakEdgeTask=lambda *a, **kw: SimpleNamespace(sample=lambda *a, **kw: batch)
    )
    monkeypatch.setattr(evaluation, "_require", lambda *a: source)
    probe = SimpleNamespace(
        reset_state=lambda: None, encode=lambda x, *a: x[:, -1, :1], diagnostics=lambda: {}
    )

    class Heads:
        def build(self, *args):
            pass

        def __call__(self, x):
            return x[:, 0], None

    a = evaluate_probe(probe, Heads(), cfg, build_conditions()[:1], 4, 2, "cpu")
    batch.values = batch.values.flip(0)
    b = evaluate_probe(probe, Heads(), cfg, build_conditions()[:1], 4, 2, "cpu")
    assert (
        a["conditions"][0]["first_stable_frame"] is b["conditions"][0]["first_stable_frame"] is None
    )
    assert a["conditions"][0]["position_error_px"] == b["conditions"][0]["position_error_px"]


@pytest.mark.parametrize(
    "suite,horizon,control,replan", [("libero", 10, 7, 5), ("robotwin2", 50, 14, 50)]
)
def test_service_main_selects_the_requested_suite(
    tmp_path, monkeypatch, suite, horizon, control, replan
):
    cfg = config()
    policy = torch.nn.Linear(1, 1)
    policy.action_horizon, policy.action_dim = horizon, 32
    encoder, bridge = build_bridge(policy, cfg, ["camera"], "cpu", suite=suite)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "suite": suite,
            "config": cfg.to_dict(),
            "camera_order": ["camera"],
            "encoder": encoder.state_dict(),
            "postprocessor": bridge.postprocessor.state_dict(),
        },
        checkpoint,
    )
    monkeypatch.setattr(service_module, "load_backend", lambda *args: (policy, {}))
    captured = []

    class Server:
        def __init__(self, **kwargs):
            captured.append(kwargs["metadata"])

        def serve_forever(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "deployment.model_server.tools.websocket_policy_server",
        SimpleNamespace(WebsocketPolicyServer=Server),
    )
    service_module.main(
        [
            "--suite",
            suite,
            "--weights",
            "unused",
            "--tokenizer",
            "unused",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ]
    )
    assert captured[0]["control_dim"] == control
    assert captured[0]["replan_steps"] == replan
    assert captured[0]["action_horizon"] == horizon
    wrong = "robotwin2" if suite == "libero" else "libero"
    with pytest.raises(ValueError, match="checkpoint suite"):
        service_module.main(
            [
                "--suite",
                wrong,
                "--weights",
                "unused",
                "--tokenizer",
                "unused",
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cpu",
            ]
        )


def test_service_rejects_policy_protocol_dimension_mismatch():
    policy = torch.nn.Linear(1, 1)
    policy.action_horizon, policy.action_dim = 50, 32
    encoder, bridge = build_bridge(policy, config(), ["camera"], "cpu", suite="libero")
    with pytest.raises(ValueError, match="dimensions"):
        service_module.N01FeatureService(policy, encoder, bridge, device="cpu")


def test_checkpoint_cannot_silently_reinterpret_old_gate_units():
    encoder = N01Encoder(config())
    old = {
        key: value
        for key, value in encoder.state_dict().items()
        if key != "evidence_encoding_version"
    }
    with pytest.raises(RuntimeError, match="evidence_encoding_version"):
        encoder.load_state_dict(old)
    restored = N01Encoder(config())
    restored.load_state_dict(encoder.state_dict())
    assert int(restored.evidence_encoding_version) == 1


def test_temporal_stability_is_computed_per_trajectory():
    from owac.evaluation.n01 import condition_metrics

    logits = torch.full((2, 6, 5, 5), -30.0)
    logits[0, :, 2, 2] = 30  # Always within tolerance.
    logits[1, :3, 4, 4] = 30
    logits[1, 3:, 2, 2] = 30
    result = SimpleNamespace(
        logits=logits,
        target_xy=torch.full((2, 6, 2), 0.5),
        target_visible=torch.ones(2, 6, dtype=torch.bool),
        step_times_ms=[],
        sample_age=None,
    )
    condition = build_conditions()[1]
    a = condition_metrics(
        result,
        condition,
        max_history=8,
        window_area=25,
        state_bytes=0,
        diagnostics={},
        observed_frames=6,
    )
    a = a["metrics"]
    assert a["first_stable_frame_per_trajectory"] == [0, 3]
    assert a["first_stable_frame"] == 1.5
    result.logits = result.logits.flip(0)
    b = condition_metrics(
        result,
        condition,
        max_history=8,
        window_area=25,
        state_bytes=0,
        diagnostics={},
        observed_frames=6,
    )
    b = b["metrics"]
    assert b["first_stable_frame_per_trajectory"] == [3, 0]
    assert a["first_stable_frame"] == b["first_stable_frame"]
