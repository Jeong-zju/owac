"""Adversarial coverage, execution and fixed-weight revision acceptance."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from owac.backends.rawvla_language import canonical_task_prompt
from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame
from owac.frameworks.n02_full.calibration_v12 import (
    pack_calibration,
    parameter_fingerprint,
    restore_calibration,
)
from owac.frameworks.n02_full.checks_v12 import (
    AbsoluteCheckResult,
    calibrate_windowed,
    v12_viewing_rule,
)
from owac.frameworks.n02_full.data_v12 import read_task_prompts
from owac.frameworks.n02_full.inference_v12 import V12InferenceInput
from owac.frameworks.n02_full.model_v12 import build_n02_v12_model
from owac.frameworks.n02_full.sequence_v12 import replay_inputs, snapshot
from owac.frameworks.n02_full.state import ObservationBundle
from owac.runtime.n02_control import ActionExecution, ControlProtocol
from owac.training.n02_v12_task import (
    REPAIR_PROTOCOL,
    BalancedBlocks,
    LegacyBalancedBlocks,
    control_flow_loss,
    select_repaired_checkpoint,
    validate_repair_config,
)


def records():
    result = []
    for episode, length in enumerate((93, 98, 87, 7)):
        for t in range(length):
            result.append(
                dict(
                    index=len(result),
                    path=f"episode{episode}",
                    t=t,
                    frames=length,
                    light="Normal" if episode % 2 else "Over",
                    phase="opening" if t < 10 else "approach" if t < 0.6 * length else "late",
                    valid=min(50, max(0, length - t - 1)),
                )
            )
    return result


def test_variable_episode_sampler_has_no_unreachable_or_mislabelled_valid_rows():
    rows = records()
    sampler = BalancedBlocks(rows, 8, 1103)
    expected = {i for i, row in enumerate(rows) if row["valid"] > 0}
    observed = []
    for key, blocks in sampler.groups.items():
        for start, stop in blocks:
            assert 1 <= stop - start <= 8
            assert len({r["path"] for r in rows[start:stop]}) == 1
            assert {(r["light"], r["phase"]) for r in rows[start:stop]} == {key}
            observed.extend(range(start, stop))
    assert set(observed) == expected and len(observed) == len(expected)
    reached = set()
    for cursor in range(len(sampler.keys) * max(map(len, sampler.groups.values()))):
        start, stop = sampler.bounds(cursor)
        reached.update(range(start, stop))
        assert (start, stop) == BalancedBlocks(rows, 8, 1103).bounds(cursor)
    assert reached == expected
    assert sampler.record()["complete_valid_coverage"]
    legacy = LegacyBalancedBlocks(rows, 8, 1103)
    old = {i for blocks in legacy.groups.values() for a, b in blocks for i in range(a, b)}
    assert expected - old, "fixture must expose the historical boundary bug"


def test_canonical_text_uses_the_same_map_for_cache_and_live(tmp_path):
    instructions = {"grab_roller": "Grab the light brown cylindrical roller on the table."}
    path = tmp_path / "data/robotwin2/raw/grab_roller/episode_000008.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, metadata_json=json.dumps({"task_name": "grab_roller"}))
    offline = read_task_prompts(
        [str(path.relative_to(tmp_path))], str(tmp_path), instructions=instructions
    )
    assert offline == [
        canonical_task_prompt("grab_roller", "different live sentence", instructions)
    ]
    with pytest.raises(ValueError, match="canonical"):
        canonical_task_prompt(None, "grab roller", instructions)
    with pytest.raises(ValueError, match="canonical"):
        read_task_prompts(["missing.npz"], str(tmp_path), instructions=instructions)


def test_native_flow_retains_gradient_in_all_internal_dimensions():
    velocity = torch.zeros(1, 50, 32, requires_grad=True)
    target = torch.ones_like(velocity)
    control_flow_loss(velocity, target, np.ones((1, 50), bool), 32).backward()
    assert torch.all(velocity.grad[..., 14:] != 0)


def test_resume_rejects_losing_the_original_selection_baseline(tmp_path):
    from owac.training.n02_v12 import (
        build_parser,
        resolve_plan,
        training_contract,
        validate_run_directory,
    )

    checkpoint = tmp_path / "old/checkpoint_0000050.pt"
    checkpoint.parent.mkdir()
    config = "configs/encoders/n02_v12/raw_task_r11_robotwin.json"
    plan = resolve_plan(
        build_parser().parse_args(
            ["--config", config, "--resume", str(checkpoint), "--output", str(tmp_path / "new")]
        )
    )
    torch.save(dict(step=50, training_contract=training_contract(plan)), checkpoint)
    with pytest.raises(ValueError, match="original run directory"):
        validate_run_directory(plan)
    plan.output = checkpoint.parent
    with pytest.raises(ValueError, match="missing its frozen"):
        validate_run_directory(plan)


def test_live_clients_share_execution_across_replans(monkeypatch):
    from owac.runtime.n02_v12_robotwin import ModelClient, reset_model
    from owac.runtime.rawvla_robotwin import ModelClient as NativeClient

    protocol = ControlProtocol()
    chunks = [np.full((50, 14), value, dtype=np.float32) for value in (2, -1)]

    class RPC:
        def __init__(self):
            self.calls = 0

        def predict_action(self, request):
            value = chunks[self.calls]
            self.calls += 1
            return {
                "ok": True,
                "data": {"actions": value, "execution_contract": protocol.as_dict()},
            }

    monkeypatch.setenv("RAWVLA_CASE_ID", "test")
    monkeypatch.setenv("RAWVLA_FLOW_SEED", "12")
    raw = ModelClient(RPC(), protocol=protocol.as_dict())
    native = object.__new__(NativeClient)
    native.client, native.actions, native.chunk_start = RPC(), None, 0
    native.execution = ActionExecution(protocol)
    state = np.zeros(14, dtype=np.float32)
    state[[6, 13]] = 1
    for i in range(20):
        example = {"image": [], "state": state, "lang": "task"}
        a, b = raw.step(example, i), native.step(example, i)
        np.testing.assert_array_equal(a, b)
        assert np.abs(a - state).max() <= 0.050001
        state = a
    assert raw.client.calls == native.client.calls == 2
    reset_model(raw)
    assert raw.execution.previous is None and raw.last_action is None


def test_execution_contract_mismatch_is_rejected(monkeypatch):
    from owac.runtime.n02_v12_robotwin import ModelClient

    monkeypatch.setenv("RAWVLA_CASE_ID", "test")
    monkeypatch.setenv("RAWVLA_FLOW_SEED", "12")
    rpc = SimpleNamespace(
        predict_action=lambda _: {"ok": True, "data": {"actions": np.zeros((50, 14))}}
    )
    with pytest.raises(ValueError, match="contract"):
        ModelClient(rpc, protocol=ControlProtocol().as_dict()).step(
            {"image": [], "state": np.zeros(14)}, 0
        )


def test_native_execution_preserves_every_requested_float_and_full_horizon():
    protocol = ControlProtocol(
        replan_interval=50, max_joint_delta=None, max_gripper_delta=None, clip_grippers=False
    )
    chunks = np.random.default_rng(42).normal(size=(50, 14)).astype(np.float32)
    state = np.random.default_rng(5).normal(size=14).astype(np.float32)
    actual = ActionExecution(protocol).prefix(chunks, state)
    np.testing.assert_array_equal(actual, chunks)


def test_full_horizon_selection_cannot_hide_late_joint_regression(tmp_path):
    protocol = ControlProtocol(
        replan_interval=50, max_joint_delta=None, max_gripper_delta=None, clip_grippers=False
    ).as_dict()
    config = {"task_adaptation": {"protocol": REPAIR_PROTOCOL}, "execution": protocol}
    (tmp_path / "run_config.json").write_text(json.dumps(config))

    def row(step, primary, full_joint):
        opening = dict(
            first10_mse=primary,
            prefix_joint_mse=primary,
            prefix_gripper_mse=primary,
            mse=primary,
            joint_mse=full_joint,
            gripper_mse=primary,
        )
        return dict(
            step=step,
            sampled_control_mse=primary,
            task_dev_manifest_sha256="fixture",
            execution_contract=protocol,
            calibration_weights_sha256="fixture",
            task_control=dict(
                strata={"Normal/opening": opening},
                prefix_joint_mse=primary,
                prefix_gripper_mse=primary,
                joint_mse=full_joint,
                gripper_mse=primary,
            ),
            executed_prefix_proxy=dict(balanced_prefix_mse=primary, balanced_mse=primary),
        )

    (tmp_path / "dev_baseline.json").write_text(json.dumps(row(0, 1, 1)))
    for step, joint in ((1, 1.1), (2, 0.9)):
        (tmp_path / f"checkpoint_{step:07d}.metrics.json").write_text(
            json.dumps(row(step, 0.9, joint))
        )
        (tmp_path / f"checkpoint_{step:07d}.pt").touch()
    choice = select_repaired_checkpoint(tmp_path)
    assert choice["selected"]["step"] == 2
    assert "all_full_joints" in choice["candidates"][0]["failed_guards"]


def model_and_input(batch=1):
    torch.manual_seed(42)
    model = build_n02_v12_model(
        dict(
            camera_ids=["head"],
            dims=[8, 16, 32, 64],
            depths=[1, 1, 1, 1],
            d_model=8,
            slot_dim=8,
            n_slots=2,
            k=4,
            n_latents=4,
            n_queries=4,
            out_dim=16,
            decoder_hidden=16,
            rounds=1,
            query_stride=8,
            max_queries_per_camera=64,
            preserve_spatial_tokens=True,
            scale_balance=True,
            learned_global_fusion=True,
            revision_policy="reset_explanations",
        )
    ).eval()

    def inp(index):
        frame = MeasurementFrame(
            values=torch.rand(batch, 3, 20, 28),
            camera_id="head",
            episode_ids=tuple(f"ep{i}" for i in range(batch)),
            frame_ids=tuple(f"f{index}" for _ in range(batch)),
            valid_mask=torch.ones(batch, 20, 28, dtype=torch.bool),
        )
        return V12InferenceInput(
            ObservationBundle(
                frames={"head": frame}, context=EmbodimentContext(), episode_ids=frame.episode_ids
            ),
            frame_index=[index] * batch,
        )

    return model, inp


def test_global_image_content_survives_constant_spatial_features_and_gets_gradients():
    model, inp = model_and_input()
    readout = model.inference.readout
    spatial = model.inference.encoder(inp(0).observation)
    for key, value in spatial.levels.items():
        ramp = torch.arange(value.shape[1], dtype=value.dtype)[None, :, None, None]
        spatial.levels[key] = ramp.expand_as(value).clone()
    before = readout.spatial_output_tokens(spatial)
    assert before.abs().sum() > 0
    changed = replace(spatial, levels={k: v.clone() for k, v in spatial.levels.items()})
    for value in changed.levels.values():
        value[:, 0] += 3
    after = readout.spatial_output_tokens(changed)
    assert not torch.allclose(before, after)
    (before * torch.randn_like(before)).sum().backward()
    assert readout.global_gain.grad.abs() > 0
    assert readout.scale_logits.grad.abs().sum() > 0
    assert torch.isfinite(readout.spatial_skip.weight.grad).all()


@pytest.mark.parametrize("batch", [1, 2])
def test_valid_contradiction_rebuilds_only_rejected_explanations_and_keeps_raw_history(batch):
    model, inp = model_and_input(batch)
    first, second = inp(0), inp(1)
    with torch.no_grad():
        state = model.step(first, model.build_state(batch)).state
        state.slot_hidden += 20
        verdict = AbsoluteCheckResult(
            "model_inadequate",
            True,
            {},
            "fixture",
            per_sequence=[{"status": "model_inadequate"}]
            + [{"status": "compatible"}] * (batch - 1),
        )
        model.inference._absolute_check = lambda *a, **kw: verdict
        raw = model.inference.step(second, snapshot(state))
        fixed = model.step(second, snapshot(state))
        fresh_input = (
            second
            if batch == 1
            else replace(
                second,
                observation=__import__(
                    "owac.frameworks.n02_full.sequence_v12", fromlist=["select_observation"]
                ).select_observation(second.observation, [0]),
                frame_index=[1],
            )
        )
        independent = model.inference.step(fresh_input, model.build_state(1))
        torch.testing.assert_close(fixed.tokens[:1], independent.tokens)
        assert not torch.allclose(raw.tokens[:1], fixed.tokens[:1])
        assert fixed.check is verdict and fixed.causal is not None
        assert fixed.diagnostics["revision"]["rows"] == [0]
        assert [e.frame_index for e in fixed.state.geometry[0].entries] == [0, 1]
        torch.testing.assert_close(fixed.state.geometry[0].tokens, fixed.tokens[0])
        if batch == 2:
            torch.testing.assert_close(raw.tokens[1:], fixed.tokens[1:])


def test_fast_prefix_preserves_current_tokens_and_causal_check_inputs():
    model, inp = model_and_input()
    inputs = [inp(i) for i in range(3)]
    with torch.no_grad():
        slow = replay_inputs(model, inputs)["rows"][-1]
        calls = []
        hook = model.inference.readout.register_forward_hook(lambda *args: calls.append(1))
        model.inference.fast_prefix = True
        fast = replay_inputs(model, inputs)["rows"][-1]
        hook.remove()
        torch.testing.assert_close(fast.tokens, slow.tokens, rtol=0, atol=0)
        torch.testing.assert_close(
            fast.causal["residuals"], slow.causal["residuals"], rtol=0, atol=0
        )
        assert len(calls) == 1


def test_fixed_weight_calibration_roundtrip_and_staleness():
    from owac.frameworks.n02_full.checks import ResidualView

    model = torch.nn.Linear(2, 2)
    rule = v12_viewing_rule(1, ("head",))
    groups = [
        [
            ResidualView(
                standardised_residual=torch.full((2, 2), v),
                valid=torch.ones(2, 2, dtype=torch.bool),
                scale=rule.scales[0],
                frame_index=0,
                region_id=rule.regions[0],
            )
        ]
        for v in (1.0, 2.0, 3.0)
    ]
    calibration = calibrate_windowed(groups, partition="calibration", rule=rule)
    identity = {"partition": "calibration", "manifest": "fixture"}
    digest = parameter_fingerprint(model)
    payload = pack_calibration(calibration, fingerprint=digest, source_identity=identity)
    restored = restore_calibration(
        json.loads(json.dumps(payload)), fingerprint=digest, source_identity=identity
    )
    assert restored.threshold == calibration.threshold
    assert restored.rule.rule_hash() == rule.rule_hash()
    with torch.no_grad():
        model.weight[0, 0] += 0.1
    with pytest.raises(ValueError, match="weights"):
        restore_calibration(
            payload, fingerprint=parameter_fingerprint(model), source_identity=identity
        )
    with pytest.raises(ValueError, match="partition identity"):
        restore_calibration(payload, fingerprint=digest, source_identity={"partition": "train"})


def test_selection_rejects_prefix_joint_gripper_or_execution_regression(tmp_path):
    protocol = ControlProtocol().as_dict()
    config = {"task_adaptation": {"protocol": REPAIR_PROTOCOL}, "execution": protocol}
    (tmp_path / "run_config.json").write_text(json.dumps(config))

    def row(step, prefix=1.0, joint=1.0, gripper=1.0, proxy=1.0):
        opening = {"first10_mse": prefix, "prefix_joint_mse": joint, "prefix_gripper_mse": gripper}
        return dict(
            step=step,
            sampled_control_mse=prefix,
            sampled_full_control_mse=0.01,
            task_dev_manifest_sha256="fixture",
            execution_contract=protocol,
            calibration_weights_sha256="fixture",
            task_control={
                "strata": {"Normal/opening": opening},
                "prefix_joint_mse": joint,
                "prefix_gripper_mse": gripper,
            },
            executed_prefix_proxy={"balanced_prefix_mse": proxy},
        )

    (tmp_path / "dev_baseline.json").write_text(json.dumps(row(0)))
    for i, record in enumerate(
        [
            row(1, 0.9, joint=1.01),
            row(2, 0.8, gripper=1.01),
            row(3, 0.7, proxy=1.01),
            row(4, 0.95, 0.9, 0.9, 0.9),
        ],
        1,
    ):
        (tmp_path / f"checkpoint_{i:07d}.pt").touch()
        (tmp_path / f"checkpoint_{i:07d}.metrics.json").write_text(json.dumps(record))
    decision = select_repaired_checkpoint(tmp_path)
    assert decision["selected"]["step"] == 4
    assert decision["physical_success"] is None
    assert [r["eligible"] for r in decision["candidates"]] == [False, False, False, True]


def test_repair_configs_require_matching_native_objective_and_protocol():
    from pathlib import Path

    config = json.loads(Path("configs/encoders/n02_v12/raw_task_r11_robotwin.json").read_text())
    validate_repair_config(config)
    modified = copy.deepcopy(config)
    modified["task_adaptation"]["objective"] = "control_flow_14"
    with pytest.raises(ValueError, match="native"):
        validate_repair_config(modified)
