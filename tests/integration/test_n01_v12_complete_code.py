"""Full code-path controls, with no optimizer construction or parameter updates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.frameworks.n01_full.learned_motion import (
    DomainWindowMotion,
    MotionConfig,
    known_transform_loss,
)
from owac.frameworks.n01_full.model import N01V12Config, N01V12Model, arm_matrix
from owac.frameworks.n01_full.observation import ObservationConfig, prepare_frame
from owac.frameworks.n01_full.support import SupportConfig
from owac.frameworks.n01_full.tensor_gls import conservative_source_mask, fit_aligned_innovations
from owac.frameworks.n01_full.training_plan import BackendSchedule, TrainingPlan


class Translation(nn.Module):
    def __init__(self):
        super().__init__()
        self.dx = nn.Parameter(torch.tensor(0.35))
        self.seen = []

    def forward(self, first, second, **kwargs):
        self.seen.append((first.detach().clone(), second.detach().clone()))
        flow = first.new_zeros(first.shape[0], 2, *first.shape[-2:])
        return {"final": flow + torch.stack([self.dx, self.dx * 0])[None, :, None, None]}


def _model(arm="raw_n01", window=4):
    config = N01V12Config(
        arm=arm,
        feature_width=8,
        stem_width=4,
        residual_blocks=1,
        window_frames=window,
        observation=ObservationConfig(),
        support=SupportConfig(
            implementation="gls_window",
            region_grid=2,
            multiscale_radii=(0, 1),
            receptive_field_radius=0,
            large_residual_threshold=3.0,
        ),
    )
    model = N01V12Model(config)
    model.encoder.set_motion_provider(
        DomainWindowMotion(
            MotionConfig(iters=0, cycle_check=False),
            config.spec.input_domain,
            network=Translation(),
        )
    )
    return model


def _push(model, state, index, value, compute=True):
    return model.step(
        MeasurementFrame(value, camera_id="agentview", frame_ids=[index], episode_ids=["a"]),
        state=state,
        budget=ComputeBudget(max_history=model.config.window_frames, max_updates=int(compute)),
    )


def test_full_matrix_has_real_modules_and_matched_scopes():
    configs = arm_matrix()
    assert all(c.motion and c.observation and c.training for c in configs)
    assert all(c.support.implementation == "gls_window" for c in configs)
    assert len({str(c.training.to_dict()) for c in configs}) == 1
    for c in configs:
        assert N01V12Config.from_dict(c.to_dict()).to_dict() == c.to_dict()


def test_gls_coefficients_agree_with_independent_numpy_wls():
    yy, xx = np.mgrid[0:12, 0:12]
    x = (xx + 0.5) * 2 / 12 - 1
    y = (yy + 0.5) * 2 / 12 - 1
    field = 0.7 + 0.13 * x - 0.22 * y
    variance = 0.01 + (0.04 * (x + 1))
    result = fit_aligned_innovations(
        torch.tensor(field)[None, None, None].float(),
        torch.ones(1, 1, 1, 12, 12, dtype=torch.bool),
        torch.tensor(variance)[None, None, None].float(),
        rows=1,
        cols=1,
        radii=(0,),
        affine=True,
    )
    design = np.stack([np.ones_like(x), x, y], -1).reshape(-1, 3)
    weight = 1 / variance.reshape(-1)
    expected = np.linalg.solve(
        design.T @ (weight[:, None] * design), design.T @ (weight * field.reshape(-1))
    )
    np.testing.assert_allclose(
        result.coefficients[0, 0, 0, 0, 0].numpy(), expected, rtol=1e-4, atol=1e-5
    )
    assert result.valid.all() and result.rank.min() == 3


def test_affine_fit_and_multiscale_candidates_are_active():
    field = torch.linspace(-1, 1, 12)[None, None, None, None, :].expand(1, 1, 1, 12, 12)
    valid = torch.ones(1, 1, 1, 12, 12, dtype=torch.bool)
    constant = fit_aligned_innovations(
        field, valid, torch.ones_like(field) * 0.01, rows=2, cols=2, radii=(0,), affine=False
    )
    affine = fit_aligned_innovations(
        field, valid, torch.ones_like(field) * 0.01, rows=2, cols=2, radii=(0, 1), affine=True
    )
    assert affine.residual_rms[0].max() < 1e-2
    assert constant.residual_rms[0].min() > 1
    assert affine.coefficients.shape[0] == 2
    valid[:] = False
    valid[..., 5:7, 5:7] = True
    result = fit_aligned_innovations(
        field,
        valid,
        torch.ones_like(field) * 0.01,
        rows=4,
        cols=4,
        radii=(0, 1),
        minimum_cells=4,
        affine=False,
    )
    assert (result.valid & (result.radius == 1)).any()


def test_observation_isp_precedes_full_field_resize_and_rejects_crop():
    from owac.raw.n01_paired_views import raw_unit_to_srgb

    raw = torch.rand(1, 3, 24, 32)
    config = ObservationConfig(native_shape=(24, 32), output_shape=(16, 16))
    frame = MeasurementFrame(raw, metadata={"variance": torch.ones_like(raw) * 1000})
    rgb = prepare_frame(frame, config, "RGB")
    expected = torch.nn.functional.interpolate(
        raw_unit_to_srgb(raw), size=(16, 16), mode="bilinear", align_corners=False
    )
    assert torch.equal(rgb.values, expected)
    assert rgb.metadata["variance"].max() < 1e-2
    assert prepare_frame(rgb, config, "RGB") is rgb
    with pytest.raises(ValueError, match="client resize/crop"):
        prepare_frame(MeasurementFrame(raw[..., :16, :16]), config, "RAW")


def test_motion_receives_own_domain_and_has_action_path_gradient():
    image = torch.rand(1, 3, 16, 16)
    for arm in ("raw_base", "rgb_base"):
        model = _model(arm)
        state = model.init_state(1, {"camera_ids": ["agentview"], "shape": (16, 16)})
        first = _push(model, state, 0, image, compute=False)
        result = _push(model, first.state, 1, image + 0.01)
        loss = model.readout(result).values.square().mean()
        loss.backward()
        provider = model.encoder.motion_provider
        assert provider.network.dx.grad is not None
        assert provider.network.dx.grad.abs() > 0
        from owac.raw.n01_paired_views import raw_unit_to_srgb

        expected = image + 0.01 if arm.startswith("raw") else raw_unit_to_srgb(image + 0.01)
        assert torch.equal(provider.network.seen[-1][0][..., :16, :16], expected)


def test_known_transform_supervision_is_differentiable_without_optimization():
    provider = DomainWindowMotion(MotionConfig(iters=0, cycle_check=False), "RAW", Translation())
    before = provider.network.dx.detach().clone()
    loss = known_transform_loss(provider, torch.rand(1, 3, 16, 16))
    loss.backward()
    assert provider.network.dx.grad.abs() > 0
    assert torch.equal(before, provider.network.dx)


def test_window_posterior_excludes_evicted_history():
    model = _model(window=4)
    image = torch.rand(1, 3, 16, 16)

    def sequence(prefix):
        state = model.init_state(1, {"camera_ids": ["agentview"], "shape": (16, 16)})
        for i, value in enumerate(prefix):
            state = _push(model, state, i, value).state
        return state

    shared = [image + 0.001 * i for i in range(4)]
    left = sequence([image + 5, image - 5] + shared)
    right = sequence([image * 0.1, image * 7] + shared)
    assert torch.equal(
        left.support["agentview"]["detector"].log_message,
        right.support["agentview"]["detector"].log_message,
    )
    assert torch.equal(left.tokens["agentview"].values, right.tokens["agentview"].values)
    assert left.support["agentview"]["replay_sources"] == 4


def test_reverse_bilinear_mask_covers_every_positive_tap():
    mask = torch.ones(1, 2, 1, 8, 8, dtype=torch.bool)
    decision = mask.clone()
    decision[:, 1, :, 3, 3] = False
    offsets = torch.zeros(1, 2, 2, 8, 8)
    offsets[:, 1, 0] = 0.5
    offsets[:, 1, 1] = 0.5
    source = conservative_source_mask(decision, mask, offsets)
    assert not source[0, 1, 0, 3:5, 3:5].any()
    assert source[0, 0].all()


def test_backend_scope_changes_and_serializes_without_any_step():
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = nn.Linear(2, 2)
            self.language = nn.Linear(2, 2)

    policy = Policy()
    model = nn.Linear(2, 2)
    schedule = BackendSchedule(model, policy, TrainingPlan(warmup_steps=2, total_steps=10))
    schedule.apply(0)
    assert not any(p.requires_grad for p in policy.parameters())
    schedule.apply(2)
    assert all(p.requires_grad for p in policy.action_head.parameters())
    assert not any(p.requires_grad for p in policy.language.parameters())
    saved = schedule.checkpoint()
    schedule.restore(saved)
    assert len(schedule.parameter_groups()) == 2
    with pytest.raises(ValueError, match="RGB tower"):
        TrainingPlan(backend_prefixes=("vlm.",))


def test_full_source_deletion_is_independent_and_never_backfills():
    model = _model("raw_base", window=4).eval()
    image = torch.rand(1, 3, 16, 16)

    def build(deleted_value):
        state = model.init_state(1, {"camera_ids": ["agentview"], "shape": (16, 16)})
        for index in range(6):
            value = deleted_value if index == 4 else image + 0.01 * index
            state = _push(model, state, index, value).state
        return model.delete_sources(state, "agentview", {4})

    first = build(image + 1)
    second = build(image - 3)
    assert first.state.cameras["agentview"].frame_ids[:3, 0].tolist() == [5, 3, 2]
    assert first.state.cameras["agentview"].filled.item() == 3
    assert torch.equal(model.readout(first).values, model.readout(second).values)
    assert first.diagnostics["new_independent_observations"] == 0


def test_fixed_decision_and_full_counterfactual_have_distinct_scopes():
    model = _model().eval()
    image = torch.rand(1, 3, 16, 16)
    state = model.init_state(1, {"camera_ids": ["agentview"], "shape": (16, 16)})
    for index in range(4):
        state = _push(model, state, index, image + 0.001 * index).state
    fixed = model.fixed_decision_readout(state, "agentview", {1})
    full = model.delete_sources(state, "agentview", {1})
    assert fixed.state.cameras["agentview"].filled.item() == 4
    assert full.state.cameras["agentview"].filled.item() == 3
    assert "fixed_motion_noise_support" in fixed.diagnostics["recompute_scope"]
    assert "all_dependencies" in full.diagnostics["recompute_scope"]


def _service(monkeypatch):
    from owac.backends.rawvla_features import ActionChunkCache
    from owac.runtime.n01_v12_service import N01V12Service
    from owac.training import n01_action

    model = _model().eval()

    class Bridge:
        protocol = SimpleNamespace(action_horizon=10, model_action_dim=32, replan_steps=8)
        action_cache = ActionChunkCache(8)
        calls = 0

        def sample_actions_with_features(self, observation, visual, **kwargs):
            self.calls += 1
            return torch.ones(1, 10, 32) * visual.values.mean()

    monkeypatch.setattr(n01_action, "observation_from_batch", lambda *args: None)
    bridge = Bridge()
    return N01V12Service(
        model, SimpleNamespace(action_horizon=10, action_dim=32), bridge, device="cpu"
    )


def _request(ids=(0, 1, 2), episode="a", camera="agentview", valid=True):
    values = (
        np.random.default_rng(1).uniform(0.05, 0.25, (1, len(ids), 3, 16, 16)).astype(np.float32)
    )
    return {
        "frames": {
            camera: {
                "values": values,
                "frame_ids": np.asarray(ids)[None],
                "timestamp": np.asarray(ids, dtype=np.float32)[None],
                "episode_ids": [episode],
                "valid_mask": np.full((1, len(ids), 3, 16, 16), valid, dtype=bool),
                "value_units": "normalized_raw_measurement",
            }
        },
        "normalized_state": np.zeros((1, 8)),
        "language": ["test task"],
    }


def test_service_ingests_every_real_frame_and_deduplicates_transport_overlap(monkeypatch):
    service = _service(monkeypatch)
    service.infer(_request(ids=(0, 0, 1, 2)))
    assert service.states["agentview"].cameras["agentview"].filled.item() == 3
    service.infer(_request(ids=(1, 2, 3)))
    buffer = service.states["agentview"].cameras["agentview"]
    assert buffer.frame_ids[:, 0].tolist() == [3, 2, 1, 0]
    assert service.states["agentview"].unique_arrivals["agentview"].item() == 4


def test_service_does_not_commit_half_a_multicamera_request(monkeypatch):
    service = _service(monkeypatch)
    service.infer(_request())
    original = service.states
    snapshot = service.snapshots["latest"]
    payload = _request(ids=(3,), episode="b")
    bad = _request(ids=(3,), episode="b", camera="wrist")["frames"]["wrist"]
    bad["values"] = bad["values"][:, :, :1]
    payload["frames"]["wrist"] = bad
    with pytest.raises(ValueError):
        service.infer(payload)
    assert service.states is original
    assert service.snapshots["latest"] is snapshot
    assert service.episodes["agentview"] == ["a"]


def test_service_all_invalid_inputs_never_fabricate_an_action(monkeypatch):
    service = _service(monkeypatch)
    result = service.infer(_request(valid=False))
    assert "unavailable" in result
    assert "normalized_actions" not in result
    assert service.bridge.calls == 0


def test_native_openpi_transport_requires_and_preserves_actual_exposure_ids(monkeypatch):
    from owac.runtime.n01_v12_service import N01V12OpenPIAdapter, OpenPIAdapterConfig

    service = _service(monkeypatch)
    adapter = N01V12OpenPIAdapter(service, OpenPIAdapterConfig(cameras=tuple(service.camera_order)))
    example = {
        "raw_burst_float32": [np.zeros((1, 4, 16, 16, 3), np.float32)] * 2,
        "raw_frame_ids": [np.asarray([0, 0, 1, 2])] * 2,
        "episode_id": "real/case",
        "lang": "test",
        "state": np.zeros((1, 8)),
    }
    query = adapter.to_query([example])
    assert query["frames"]["agentview"]["values"].shape == (1, 4, 3, 16, 16)
    assert query["frames"]["agentview"]["frame_ids"].tolist() == [[0, 0, 1, 2]]
    del example["raw_frame_ids"]
    with pytest.raises(ValueError, match="real raw_frame_ids"):
        adapter.to_query([example])


def test_robo_native_client_collects_between_replans_and_keeps_mean_std():
    from owac.runtime.n01_v12_robotwin import NativeRoboTwinClient

    class RPC:
        payloads = []

        def predict_action(self, payload):
            self.payloads.append(payload)
            return {"ok": True, "data": {"normalized_actions": np.ones((1, 50, 32), np.float32)}}

    stats = {
        "state": {"mean": [2] * 14, "std": [2] * 14},
        "actions": {"mean": [3] * 14, "std": [2] * 14},
    }
    rpc = RPC()
    client = NativeRoboTwinClient({"norm_stats": stats, "episode_id": "task/seed"}, rpc)
    example = {
        "image": [np.zeros((24, 32, 3), np.float32)] * 3,
        "state": np.full(14, 4),
        "lang": "task",
    }
    for step in range(51):
        action = client.step(example, step)
    assert len(rpc.payloads) == 2
    assert np.allclose(rpc.payloads[-1]["normalized_state"], 1)
    assert np.all(action == 5)
    for frame in rpc.payloads[-1]["frames"].values():
        assert frame["frame_ids"].tolist() == [list(range(19, 51))]
        assert frame["values"].shape == (1, 32, 3, 24, 32)


def test_content_mapping_discovers_permuted_initial_states_and_rejects_tampering(tmp_path):
    from owac.evaluation.n01_v12_identity import verified_split_index, verify_demo_content

    actions = np.arange(15, dtype=np.float32).reshape(5, 3)
    states = np.arange(20, dtype=np.float64).reshape(5, 4)
    initial = np.stack([states[1], states[0], states[2]])
    result = verify_demo_content(actions[[1, 3, 4]], states[[1, 3, 4]], actions, states, initial)
    assert result["verified"] and result["matching_init_state_indices"] == [1]
    altered = states[[1, 3, 4]].copy()
    altered[1, 0] += 1
    assert not verify_demo_content(actions[[1, 3, 4]], altered, actions, states, initial)[
        "verified"
    ]
    import json

    metadata = {"suite": "libero_10", "task_id": 0, "demo_index": 17}
    np.savez(
        tmp_path / "episode.npz",
        actions=actions[[1, 3, 4]],
        states=states[[1, 3, 4]],
        metadata_json=json.dumps(metadata),
    )
    manifest = {"rows": [{"relative_path": "episode.npz", "split": "test"}]}
    row = {**result, "relative_path": "episode.npz", **metadata}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "proof.json").write_text(
        json.dumps({"schema": "n01-v12-content-identity-v1", "rows": [row]})
    )
    index = verified_split_index(tmp_path / "manifest.json", tmp_path, tmp_path / "proof.json")
    assert index == {("libero_10", 0, 1): "test"}
    np.savez(
        tmp_path / "episode.npz",
        actions=actions[[1, 3, 4]],
        states=altered,
        metadata_json=json.dumps(metadata),
    )
    with pytest.raises(ValueError, match="cache content changed"):
        verified_split_index(tmp_path / "manifest.json", tmp_path, tmp_path / "proof.json")


def test_persistent_hold_blocks_direct_train_before_model_or_data_load(tmp_path):
    from owac.training.n01_v12_guard import assert_training_allowed

    hold = tmp_path / "configs/encoders/n01_full/v1_2/TRAINING_HOLD.json"
    hold.parent.mkdir(parents=True)
    hold.write_text("{}")
    with pytest.raises(RuntimeError, match="training is disabled"):
        assert_training_allowed(tmp_path, tmp_path / "new-run")


def test_validation_sampling_covers_distribution_and_does_not_consume_global_rng():
    import random

    from owac.training.n01_v12_sampling import validation_indices

    before = random.getstate()
    assert validation_indices(100, 4, 101) == [0, 25, 50, 75]
    first = validation_indices(100, 4, 101, mode="random", step=1000)
    assert first == validation_indices(100, 4, 101, mode="random", step=1000)
    assert first != validation_indices(100, 4, 101, mode="random", step=2000)
    assert random.getstate() == before


def test_sampling_coverage_survives_resume_and_discloses_missing_conditions():
    from owac.training.n01_v12_sampling import Coverage

    coverage = Coverage()
    coverage.observe(3, {"episode_key": "a", "condition": "dark"})
    coverage = Coverage.from_dict(coverage.to_dict())
    coverage.observe(3, {"episode_key": "a"})
    coverage.observe(9, {"episode_key": "b", "condition": "bright"})
    report = coverage.report(10)
    assert report["distinct_windows"] == 2 and report["distinct_trajectories"] == 2
    assert report["repeat_fraction"] == pytest.approx(1 / 3)
    assert report["missing_condition_metadata"] == 1


def test_identity_content_jointly_matches_repeated_actions_and_states():
    from owac.evaluation.n01_v12_identity import verify_demo_content

    actions = np.zeros((3, 2))
    states = np.arange(6).reshape(3, 2)
    result = verify_demo_content(actions[1:], states[1:], actions, states, states[[2, 0, 1]])
    assert result["verified"]
    assert result["matched_action_indices"] == [1, 2]
    assert result["matching_init_state_indices"] == [1]


def test_labelled_calibration_uses_actual_source_and_region_decisions():
    from owac.evaluation.n01_v12_calibrate import label_counts

    model = _model()
    state = model.init_state(1, {"camera_ids": ["agentview"], "shape": (16, 16)})
    for i in range(4):
        output = _push(model, state, i, torch.full((1, 3, 16, 16), 0.2))
        state = output.state
    camera = model.config.cameras[0]
    counts = label_counts(
        state, camera, [{"source_frame_id": 1, "region": 0, "expected": "stable"}]
    )
    assert sum(counts.values()) == 1
    with pytest.raises(ValueError, match="retained historical"):
        label_counts(state, camera, [{"source_frame_id": 3, "region": 0, "expected": "stable"}])


def test_motion_never_reads_bad_or_saturated_measurement_values():
    model = _model().eval()
    state = model.init_state(1, {"camera_ids": ["agentview"], "shape": (16, 16)})
    for i in range(2):
        image = torch.full((1, 3, 16, 16), 0.2)
        image[..., 4:8, 4:8] = 500.0 * (i + 1)
        bad = torch.zeros_like(image, dtype=torch.bool)
        bad[..., 4:8, 4:8] = True
        output = model.step(
            MeasurementFrame(image, camera_id="agentview", frame_ids=[i], bad_mask=bad),
            state=state,
            budget=ComputeBudget(max_history=4, max_updates=1),
        )
        state = output.state
    seen = model.encoder.motion_provider.network.seen
    assert seen
    for reference, source in seen:
        assert not reference[..., 4:8, 4:8].any()
        assert not source[..., 4:8, 4:8].any()


def test_deleting_every_source_marks_camera_unavailable_without_fake_measurement():
    model = _model().eval()
    state = model.init_state(1, {"camera_ids": ["agentview"], "shape": (16, 16)})
    state = _push(model, state, 7, torch.rand(1, 3, 16, 16)).state
    output = model.delete_sources(state, "agentview", {7})
    tokens = model.readout(output)
    assert not tokens.mask.any()
    assert not tokens.values.any()
    assert output.state.cameras["agentview"].filled.item() == 0
    assert output.diagnostics["unavailable_camera"] == "agentview"


def test_camera_checkpoint_preserves_full_fusion_and_action_gradients():
    import copy

    direct = _model("raw_base").train()
    recompute = copy.deepcopy(direct)
    direct.encoder.checkpoint_content = False
    recompute.encoder.checkpoint_content = True
    values = torch.rand(1, 3, 3, 16, 16)
    masks = torch.ones(1, 3, 1, 16, 16, dtype=torch.bool)
    masks[:, 2, :, 3:7, 3:7] = False
    offsets = torch.zeros(1, 3, 2, 16, 16)
    offsets[:, 1, 0] = 0.25
    a, available, weights = direct.encoder._encode_readout(values, masks, offsets)
    b, present, summary = recompute.encoder._encode_readout(values, masks, offsets)
    assert torch.equal(available, present)
    assert torch.allclose(a, b, atol=1e-7, rtol=1e-5)
    assert torch.allclose(weights.mean(2, keepdim=True), summary, atol=1e-7)
    a.square().sum().backward()
    b.square().sum().backward()
    for (name, first), (_, second) in zip(
        direct.encoder.named_parameters(), recompute.encoder.named_parameters(), strict=True
    ):
        if first.grad is not None:
            assert second.grad is not None, name
            assert torch.allclose(first.grad, second.grad, atol=1e-6, rtol=1e-4), name
