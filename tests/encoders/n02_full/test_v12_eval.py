"""N02 v1.2 P2 evaluation-entry tests (CPU, tiny real modules, no dataset, no GPU).

They check the parts the acceptance depends on and that a green import cannot show:

* the declared viewing rule really matches the views the frozen inference builds, so the
  three-state check is *valid* instead of silently invalid;
* a mismatched label is an explicit failure, and a missing calibrator is counted separately,
  never as a zero and never as the string ``None``;
* settings resolution tolerates a missing/hand-written config, and the fixed dev sample receipt
  is reused instead of silently re-sampling;
* the diagnostic figures are produced from the real tensors and carry the required labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame
from owac.evaluation.n02_full import v12 as v12_eval
from owac.evaluation.n02_full import v12_viz
from owac.frameworks.n02_full.checks import STATUSES, ViewingRule, select_views
from owac.frameworks.n02_full.checks_v12 import (
    AbsoluteCheckResult,
    calibrate_windowed,
    summarise_results,
)
from owac.frameworks.n02_full.model_v12 import build_n02_v12_model
from owac.frameworks.n02_full.state import ObservationBundle

CAMERAS = ("cam_a", "cam_b")
SIDE = 32
ROWS = 3


def tiny_settings(**overrides) -> dict:
    settings = {
        "camera_ids": CAMERAS,
        "dims": (8, 16, 32, 64),
        "depths": (1, 1, 1, 1),
        "d_model": 32,
        "k": 3,
        "n_slots": 2,
        "slot_dim": 8,
        "rounds": 2,
        "history": 8,
        "query_stride": 8,
        "alignment": 32,
        "input_mode": "RAW3",
        "measurement_channels": 3,
        "decoder_hidden": 16,
        "n_latents": 4,
        "n_queries": 8,
        "out_dim": 32,
        "strict_check": True,
    }
    settings.update(overrides)
    return settings


def tiny_model(**overrides):
    torch.manual_seed(1103)
    return build_n02_v12_model(tiny_settings(**overrides))


def bundle(
    seed: int = 0, *, rows: int = ROWS, side: int = SIDE, cameras=CAMERAS
) -> ObservationBundle:
    generator = torch.Generator().manual_seed(seed)
    frames = {}
    for camera in cameras:
        frames[camera] = MeasurementFrame(
            values=torch.randn(rows, 3, side, side, generator=generator) * 0.25 + 0.5,
            valid_mask=torch.ones(rows, side, side, dtype=torch.bool),
            camera_id=camera,
            frame_ids=tuple(f"{1000 + row}" for row in range(rows)),
            episode_ids=tuple("episode_007" for _ in range(rows)),
            metadata={"declared_units": "unit_test"},
        )
    return ObservationBundle(
        frames=frames,
        context=EmbodimentContext(proprio=np.zeros((rows, 8), np.float32), proprio_available=True),
        episode_ids=tuple("episode_007" for _ in range(rows)),
        anchor_ids=tuple(f"episode_007#anchor{1000 + row:06d}" for row in range(rows)),
    )


class FakeBatch:
    """The three fields ``v12`` reads off a served window batch."""

    def __init__(
        self, observation=None, *, rows: int = ROWS, frame_start: int = 1000, seed: int = 0
    ):
        self.observation = bundle(seed, rows=rows) if observation is None else observation
        self.frame_index = np.arange(frame_start, frame_start + rows, dtype=np.int64)
        self.measurement_ids = tuple(
            (f"episode_{index:03d}", str(camera), str(frame_start + row))
            for row in range(rows)
            for index, camera in enumerate(CAMERAS)
        )
        self.declared_prior = None
        self.partition = "dev"


class FakeRecord:
    def __init__(self, payload: dict):
        self._payload = payload

    def as_dict(self) -> dict:
        return dict(self._payload)


class FakeSource:
    def __init__(self, total: int):
        self.anchor_count = total

    def anchor(self, index: int) -> FakeRecord:
        return FakeRecord(
            {
                "relative_path": f"libero/raw/task_{index:03d}.npz",
                "current_frame_index": 10 + index,
                "episode_group_id": f"episode_{index:03d}",
            }
        )


def _contains_none_string(payload) -> bool:
    if isinstance(payload, str):
        return payload == "None"
    if isinstance(payload, dict):
        return any(_contains_none_string(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_contains_none_string(value) for value in payload)
    return False


# ---------------------------------------------------------------------------------------
# the check rule really matches the frozen views
# ---------------------------------------------------------------------------------------


def test_check_rule_matches_without_changing_decoder_time() -> None:
    model = tiny_model()
    batch = FakeBatch()
    replay = v12_eval.replay_window(model, batch, device="cpu", check_label=0)
    locked = v12_eval.locked_residual_groups(
        model.inference, replay["check_input"], replay["check_state"], replay["check_backend"].q
    )
    assert locked["frame_index"] == [1002]
    assert locked["labels"][0] == [("sensor_stride8", 0, camera) for camera in CAMERAS]
    rule = ViewingRule(scales=("sensor_stride8",), frames=(0,), regions=CAMERAS)
    assert len(select_views(locked["groups"][0], rule=rule)) == 2
    assert replay["rows"][0].check.valid is False  # no prefix is not a valid check


def _fit_causal_check(model):
    rule = ViewingRule(
        scales=("sensor_stride8",), frames=(0,), regions=CAMERAS, window=4, use_local_maximum=False
    )
    groups, ids = [], []
    for seed in range(6):
        replay = v12_eval.replay_window(model, FakeBatch(seed=seed), device="cpu", check_label=0)
        locked = v12_eval.locked_residual_groups(
            model.inference, replay["check_input"], replay["check_state"], replay["check_backend"].q
        )
        groups.extend(locked["groups"])
        ids.extend(locked["measurement_ids"])
    return calibrate_windowed(
        groups, partition="calibration", rule=rule, calibration_ids=ids, test_ids=[]
    )


def test_calibrated_check_is_valid_and_uncalibrated_is_counted_separately() -> None:
    model = tiny_model()
    frozen = _fit_causal_check(model)
    assert np.isfinite(frozen.threshold)
    model.inference.calibrator = frozen
    checked = v12_eval.replay_window(model, FakeBatch(seed=99), device="cpu", check_label=0)[
        "check_backend"
    ]
    assert checked.check.valid is True and checked.check.status in STATUSES
    model.inference.calibrator = None
    uncalibrated = v12_eval.replay_window(model, FakeBatch(seed=99), device="cpu", check_label=0)[
        "check_backend"
    ]
    assert uncalibrated.check.valid is False
    record = v12_eval.check_record(uncalibrated.check)
    assert record["statistics"]["reason"] == "no_fitted_calibrator"
    summary = summarise_results([checked.check, uncalibrated.check])
    assert summary["valid"] == 1 and summary["invalid"] == 1
    assert not _contains_none_string(record)


def test_a_label_that_does_not_match_the_frozen_rule_is_an_explicit_failure() -> None:
    model = tiny_model()
    frozen = _fit_causal_check(model)
    frozen.rule = ViewingRule(scales=("sensor_stride8",), frames=(4242,), regions=CAMERAS)
    model.inference.calibrator = frozen
    with pytest.raises(Exception, match="viewing rule"):
        v12_eval.replay_window(model, FakeBatch(), device="cpu", check_label=0)


def test_invalid_results_never_enter_the_status_counts() -> None:
    valid = AbsoluteCheckResult(
        status="compatible",
        valid=True,
        statistics={"sequence_statistic": 0.2},
        calibration_version="test",
    )
    invalid = AbsoluteCheckResult.invalid(reason="no_fitted_calibrator")
    summary = summarise_results([valid, invalid])
    assert summary["compatible"] == 1 and summary["invalid"] == 1
    assert invalid.counted_as_false_positive is False
    assert invalid.counted_as_detection is False
    assert not _contains_none_string(summary)


# ---------------------------------------------------------------------------------------
# replay and settings
# ---------------------------------------------------------------------------------------


def test_replay_window_chains_the_state_and_checks_the_anchor_row() -> None:
    model = tiny_model()
    batch = FakeBatch()
    replay = v12_eval.replay_window(model, batch, device="cpu", check_label=0, check=True)
    assert len(replay["rows"]) == ROWS
    assert replay["check_backend"] is not None
    first_state = replay["rows"][0].state
    last_state = replay["rows"][-1].state
    assert last_state.version >= first_state.version
    assert int(last_state.slot_hidden.shape[0]) == 1
    # the check step declares the constant window label, and the real frame identity survives in
    # the measurement ids of the check step
    observation = v12_eval.row_observation(batch.observation, ROWS - 1)
    inp = v12_eval.inference_input(batch, observation=observation, rows=[ROWS - 1], frame_label=0)
    assert [int(item) for item in inp.frame_index.tolist()] == [int(batch.frame_index[-1])]
    assert all(item[2] == str(int(batch.frame_index[-1])) for item in inp.measurement_ids), (
        inp.measurement_ids
    )
    real_inp = v12_eval.inference_input(batch, observation=observation, rows=[ROWS - 1])
    assert [int(item) for item in real_inp.frame_index.tolist()] == [int(batch.frame_index[-1])]


def test_resolve_settings_tolerates_missing_and_nested_configs(tmp_path: Path) -> None:
    parser = v12_eval.build_parser()
    missing = tmp_path / "not_here.json"
    settings = v12_eval.resolve_settings(
        parser.parse_args(["--config", str(missing), "--output", str(tmp_path / "eval")])
    )
    assert settings.config_present is False
    assert settings.dataset_format == "robotwin"
    assert settings.camera_ids() == ("robotwin_head", "robotwin_left", "robotwin_right")
    assert settings.preview_dir == str(tmp_path / "preview")

    nested = tmp_path / "config.json"
    nested.write_text(
        json.dumps(
            {
                "run": {"name": "raw-n02"},
                "data": {"dataset_format": "libero", "data_root": "/data/rawvla"},
                "model": {"k": 2, "query_stride": 16, "n_queries": 4},
            }
        ),
        encoding="utf-8",
    )
    settings = v12_eval.resolve_settings(
        parser.parse_args(["--config", str(nested), "--output", str(tmp_path / "eval")])
    )
    assert settings.config_present is True and settings.config_sha256
    assert settings.dataset_format == "libero"
    assert settings.data_root == "/data/rawvla"
    assert settings.camera_ids() == ("libero_agentview", "libero_wrist")
    assert settings.query_stride == 16
    assert settings.model["k"] == 2 and settings.model["n_queries"] == 4

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"dataset_format": "nope"}), encoding="utf-8")
    settings = v12_eval.resolve_settings(parser.parse_args(["--config", str(unknown)]))
    assert settings.dataset_format == "robotwin"
    assert any("not a known cache format" in note for note in settings.notes)


def test_config_checkpoints_are_not_confused(tmp_path: Path) -> None:
    """The config's ``policy.checkpoint`` is the pi0.5 backend, never the N02 model weights."""
    parser = v12_eval.build_parser()
    config = tmp_path / "raw_n02_robotwin.json"
    config.write_text(
        json.dumps(
            {
                "platform": "robotwin",
                "model": {"k": 2},
                "policy": {"checkpoint": "/backend/pi05/model.safetensors"},
                "paths": {"output_root": "/tmp/runs"},
            }
        ),
        encoding="utf-8",
    )
    settings = v12_eval.resolve_settings(parser.parse_args(["--config", str(config)]))
    assert settings.checkpoint is None
    assert settings.backend_checkpoint == "/backend/pi05/model.safetensors"

    model_config = tmp_path / "with_model_config.json"
    model_config.write_text(json.dumps({"model": {"checkpoint": "/runs/x.pt"}}), encoding="utf-8")
    settings = v12_eval.resolve_settings(parser.parse_args(["--config", str(model_config)]))
    assert settings.checkpoint == "/runs/x.pt"


def test_a_safetensors_file_is_never_torch_loaded(tmp_path: Path) -> None:
    model = tiny_model()
    bad = tmp_path / "model.safetensors"
    bad.write_bytes(b"not a torch checkpoint")
    with pytest.raises(v12_eval.EvalError, match="not a torch checkpoint"):
        v12_eval.load_checkpoint_into(model, bad)
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a torch checkpoint either")
    with pytest.raises(v12_eval.EvalError, match="cannot load"):
        v12_eval.load_checkpoint_into(model, broken)
    with pytest.raises(v12_eval.EvalError, match="missing"):
        v12_eval.load_checkpoint_into(model, tmp_path / "absent.pt")


def test_model_step_prefers_the_file_then_the_flag_then_stays_unknown(tmp_path: Path) -> None:
    """The label names the real step: the file's step, else ``--model-step``, else unknown."""
    model = tiny_model()
    weights = dict(model.state_dict())

    with_step = tmp_path / "with_step.pt"
    torch.save({"model": {"step": 1000, "global_step": 4242, **weights}}, with_step)
    receipt = v12_eval.load_checkpoint_into(model, with_step)
    assert receipt["loaded"] is True
    assert receipt["training_step_recorded_in_file"] == 1000
    assert v12_eval.resolve_model_step(receipt["training_step_recorded_in_file"], 7) == 1000

    bare = tmp_path / "bare_state_dict.pt"
    torch.save(weights, bare)
    receipt = v12_eval.load_checkpoint_into(model, bare)
    assert receipt["loaded"] is True
    assert receipt["training_step_recorded_in_file"] is None
    assert v12_eval.resolve_model_step(receipt["training_step_recorded_in_file"], 1000) == 1000

    assert v12_eval.resolve_model_step(None, None) is None
    identity = {"episode_group_id": "episode_007"}
    unknown = v12_viz._labels(None, identity, CAMERAS[0], 0, 0, None)
    assert "model step=unknown" in unknown
    assert "model step=1000" in v12_viz._labels(None, identity, CAMERAS[0], 0, 0, 1000)

    for parser in (v12_eval.build_parser(), v12_viz.build_parser()):
        assert parser.parse_args([]).model_step is None
        assert parser.parse_args(["--model-step", "1000"]).model_step == 1000


def test_anchor_indices_and_frozen_sample_receipt(tmp_path: Path) -> None:
    source = FakeSource(101)
    indices = v12_eval.anchor_indices(source, 5)
    assert indices == sorted(set(indices))
    assert all(0 <= item < 101 for item in indices)
    assert v12_eval.anchor_indices(source, 5) == indices

    path = tmp_path / "dev_samples.json"
    first_indices, first = v12_eval.fixed_sample_identities(
        source, 3, path=path, partition="dev", frozen=None
    )
    assert first["source"] == "frozen_now" and path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    second_indices, second = v12_eval.fixed_sample_identities(
        source, 3, path=path, partition="dev", frozen=payload["identities"]
    )
    assert second_indices == first_indices
    assert second["source"] == "frozen_receipt"
    assert [item["anchor_index"] for item in second["identities"]] == first_indices


# ---------------------------------------------------------------------------------------
# report and figure plumbing
# ---------------------------------------------------------------------------------------


def test_jsonable_is_json_safe_and_compact() -> None:
    payload = {
        "nan": float("nan"),
        "inf": float("inf"),
        "tensor": torch.zeros(3),
        "long": list(range(100)),
        "path": Path("/tmp/x"),
    }
    safe = v12_eval._jsonable(payload)
    text = json.dumps(safe, allow_nan=False)
    assert safe["nan"] is None and safe["inf"] is None
    assert safe["long"]["count"] == 100 and safe["long"]["min"] == 0
    assert safe["tensor"] == [0.0, 0.0, 0.0]
    assert isinstance(text, str) and "None" not in text.replace('"None"', "")


def test_png_bytes_is_a_real_png() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[..., 0] = 255
    data = v12_viz.png_bytes(image)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in data and b"IEND" in data
    with pytest.raises(ValueError):
        v12_viz.png_bytes(np.zeros((2, 2, 2), dtype=np.uint8))


def test_figures_carry_identity_labels_and_real_tensors(tmp_path: Path) -> None:
    model = tiny_model()
    batch = FakeBatch()
    replay = v12_eval.replay_window(model, batch, device="cpu", check_label=0, check=True)
    capture = v12_viz._anchor_capture(model, batch, device="cpu", replay=replay)
    capture["model"] = model
    capture["stride"] = 8
    attention = v12_viz._attention_maps(capture, None)
    assert attention["available"] is True
    for camera in CAMERAS:
        entry = attention[camera]
        assert entry["coords"].shape[0] == entry["attention"].shape[0]
        assert entry["coords"].shape[0] > 0
    labels = v12_viz._labels(None, {"episode_group_id": "episode_007"}, CAMERAS[0], 1234, 1, 1000)
    for key in ("episode=", "camera=", "frame=", "iteration=", "model step="):
        assert key in labels

    frame = v12_eval.row_observation(batch.observation, ROWS - 1).frames[CAMERAS[0]]
    svg = v12_viz.figure_raw_display(
        frame,
        camera_id=CAMERAS[0],
        labels=labels,
        title="unit test raw",
        white_level=1.0,
        gamma=2.2,
        gains=[1.0, 1.0, 1.0],
        zoom_box=(0, 8, 0, 8),
        max_side=48,
    )
    assert svg.startswith(b"<svg") and b"<image" in svg and b"same-source RGB" in svg
    assert b"episode=episode_007" in svg

    svg = v12_viz.figure_attention(
        frame,
        attention,
        camera_id=CAMERAS[0],
        labels=labels,
        title="unit test attention",
        white_level=1.0,
        max_side=48,
    )
    assert b"cross-attention" in svg and b"<image" in svg

    svg = v12_viz.figure_prediction(
        capture,
        frame,
        camera_id=CAMERAS[0],
        camera_slot=0,
        labels=labels,
        title="unit test prediction",
        max_side=48,
    )
    assert b"candidate RAW prediction" in svg

    svg = v12_viz.figure_rounds(
        replay["rows"][-1], labels=labels, title="unit test rounds", rounds=2
    )
    assert b"final support" in svg

    svg = v12_viz.figure_actions(
        None,
        np.zeros((10, 7), dtype=np.float64),
        labels=labels,
        title="unit test actions",
        control_dim=7,
        reason="unit test backend absent",
    )
    assert b"prediction unavailable" in svg and b"reference dim 0" in svg


def test_attention_figure_says_so_when_the_tensor_is_absent(tmp_path: Path) -> None:
    frame = MeasurementFrame(values=np.zeros((3, 8, 8), np.float32), camera_id="cam_a")
    svg = v12_viz.figure_attention(
        frame,
        {"available": False, "reason": "no tensor"},
        camera_id="cam_a",
        labels="episode=x camera=cam_a frame=0 iteration=0 model step=none",
        title="missing attention",
        white_level=1.0,
        max_side=32,
    )
    assert b"attention tensor unavailable" in svg


def test_preview_files_are_named_relative_to_the_run_dir(tmp_path: Path) -> None:
    run = tmp_path / "run"
    preview = run / "preview"
    preview.mkdir(parents=True)
    (preview / "01_raw_rgb_zoom_000_cam_a.svg").write_text("<svg/>", encoding="utf-8")
    (preview / "figures.json").write_text("{}", encoding="utf-8")
    files = v12_eval._preview_files(preview)
    assert files == ["preview/01_raw_rgb_zoom_000_cam_a.svg"]


def test_real_window_tokens_match_evaluation_with_two_step_truncated_gradients():
    from owac.backends.n02_v12_tokens import N02V12TokenSource
    from owac.frameworks.n02_full.sequence_v12 import replay_inputs, window_inputs

    model = tiny_model(n_queries=256, out_dim=2048)
    batch = FakeBatch()
    batch.history = ROWS
    for frame in batch.observation.frames.values():
        frame.values.requires_grad_()
    source = N02V12TokenSource(model, camera_ids=CAMERAS, history=ROWS)
    tokens = source(batch).values
    reference = replay_inputs(model, window_inputs(batch))["rows"][-1].tokens
    torch.testing.assert_close(tokens, reference, rtol=0, atol=0)
    tokens.square().mean().backward()
    for frame in batch.observation.frames.values():
        assert frame.values.grad[: ROWS - 2].abs().sum() == 0
        assert frame.values.grad[ROWS - 2].abs().sum() > 0
        assert frame.values.grad[ROWS - 1].abs().sum() > 0
    changed = FakeBatch()
    changed.history = ROWS
    for camera in CAMERAS:
        changed.observation.frames[camera].values[-1] = (
            batch.observation.frames[camera].values[-1].detach()
        )
        changed.observation.frames[camera].values[:-1] += 3
    changed_tokens = source(changed).values
    assert float((changed_tokens - tokens.detach()).detach().abs().max()) > 1e-7


def test_causal_loss_reaches_the_previous_corrector_without_current_pixel_leakage():
    from owac.frameworks.n02_full.sequence_v12 import replay_inputs, window_inputs

    batch = FakeBatch()
    batch.history = ROWS
    model = tiny_model()
    causal = replay_inputs(model, window_inputs(batch), grad_anchor=True)["rows"][-1].causal
    energy = model.inference.measurement.energy(
        mean=causal["mean"],
        variance=causal["variance"],
        masks=causal["valid"],
        target=causal["queries"].target,
    ).sum()
    energy.backward()
    corrector = list(model.inference.corrector.parameters())
    assert sum(float(p.grad.abs().sum()) for p in corrector if p.grad is not None) > 0


def test_no_measurement_preserves_prior_and_slots_even_with_known_saturation_level():
    import dataclasses

    from owac.frameworks.n02_full.inference_v12 import V12InferenceInput

    model = tiny_model(saturation_level=1.0)
    observation = bundle(rows=1)
    for camera, frame in observation.frames.items():
        observation.frames[camera] = dataclasses.replace(
            frame,
            saturated_mask=torch.ones_like(frame.valid_mask),
        )
    state = model.build_state()
    prior = torch.tensor([[0.0, 0.25, 0.75]])
    output = model.step(V12InferenceInput(observation, prior, [1000]), state)
    torch.testing.assert_close(output.q, prior)
    torch.testing.assert_close(output.state.slot_hidden, state.slot_hidden)
    assert not output.energies.any()
    assert not any(bool(r["accepted"].any()) for r in output.rounds)
    assert output.diagnostics["no_information_rows"] == [True]


def test_causal_prediction_is_independent_of_current_pixels_and_candidate_selection():
    first, changed = FakeBatch(), FakeBatch()
    for frame in changed.observation.frames.values():
        frame.values[-1] += 10
    model = tiny_model()
    a = v12_eval.replay_window(model, first, device="cpu", check_label=0)["rows"][-1]
    b = v12_eval.replay_window(model, changed, device="cpu", check_label=0)["rows"][-1]
    for name in ("mean", "variance", "chosen"):
        torch.testing.assert_close(a.causal[name], b.causal[name], rtol=0, atol=0)
    assert not torch.equal(a.causal["residuals"], b.causal["residuals"])
    assert not torch.equal(a.mean, b.mean)


def test_dense_queries_preserve_all_channels_last_rows_and_pixel_masks():
    from owac.frameworks.n02_full.inference_v12 import build_query_sets

    obs = bundle(rows=1, side=32, cameras=(CAMERAS[0],))
    frame = obs.frames[CAMERAS[0]]
    frame.valid_mask = torch.ones_like(frame.values, dtype=torch.bool)
    frame.valid_mask[0, 1, 31, 1] = False
    queries = build_query_sets(obs, stride=1, max_queries_per_camera=0, channel_reduction="select")
    assert queries.coverage == (1.0,)
    assert queries.target.numel() == 3 * 32 * 32
    torch.testing.assert_close(queries.target, frame.values.flatten(1))
    assert int(queries.masks["valid"].sum()) == 3 * 32 * 32 - 1
    assert float(queries.coords[0, :, 1].max()) == 31.5
    assert set(queries.query_channel.flatten().tolist()) == {0, 1, 2}


def test_retraction_replay_matches_a_clean_run_without_the_removed_measurement():
    from owac.frameworks.n02_full.sequence_v12 import window_inputs

    model = tiny_model()
    inputs = window_inputs(FakeBatch())
    with torch.no_grad():
        first = model.step(inputs[0], model.build_state(1))
        second = model.step(inputs[1], first.state)
        third = model.step(inputs[2], second.state)
        pruned = model.drop_measurements(third.state, second.diagnostics["meas_ids"][0])
        rebuilt, receipt = model.rebuild(inputs[2], pruned)
        clean_first = model.step(inputs[0], model.build_state(1))
        clean = model.step(inputs[2], clean_first.state)
    assert receipt["from_zero"] is True
    assert len(rebuilt.geometry[0].entries) == 2
    torch.testing.assert_close(rebuilt.slot_hidden, clean.state.slot_hidden, rtol=0, atol=0)
    torch.testing.assert_close(rebuilt.geometry[0].tokens, clean.tokens[0], rtol=0, atol=0)


def test_training_permutation_covers_all_anchors_and_resume_cursor():
    from owac.frameworks.n02_full.sequence_v12 import training_anchor_order

    order = [training_anchor_order(97, 1103, cursor)[0] for cursor in range(194)]
    assert sorted(order[:97]) == list(range(97))
    assert sorted(order[97:]) == list(range(97))
    assert order[:97] != list(range(97)) and order[:97] != order[97:]
    assert [training_anchor_order(97, 1103, cursor)[0] for cursor in range(40, 45)] == order[40:45]


def test_training_preview_uses_real_attention_without_channel_broadcasting(tmp_path):
    from types import SimpleNamespace

    from owac.training.n02_v12 import N02V12Trainer

    model = tiny_model()
    batch = FakeBatch()
    batch.history = ROWS
    output = v12_eval.replay_window(model, batch, device="cpu", check_label=0)["rows"][-1]
    trainer = N02V12Trainer.__new__(N02V12Trainer)
    trainer.model, trainer.run_dir = model, tmp_path
    trainer.plan = SimpleNamespace(previews=1)
    trainer.current_step = 2
    receipt = trainer.write_previews(batch, output)
    assert receipt["status"] == "written", receipt
    assert len(receipt["files"]) == 3
    for filename in receipt["files"]:
        text = (tmp_path / filename).read_text()
        assert text.startswith("<svg") and "model step=2" in text
    attention = (tmp_path / receipt["files"][1]).read_text()
    assert "cross-attention" in attention and "<image" in attention


def test_eval_rgb_source_uses_the_declared_isp_and_official_prompt_reader(tmp_path):
    from dataclasses import make_dataclass

    from owac.frameworks.n02_full.data_v12 import (
        RepresentedWindowSource,
        apply_declared_isp,
        read_task_prompts,
    )

    batch = make_dataclass("ServedBatch", [("observation", object)])(FakeBatch().observation)

    class Source:
        def window(self, start, stop):
            return batch

    spec = {"kind": "raw3_fixed_isp", "isp": {"input_scale": 1, "gamma": 2.2}}
    represented = RepresentedWindowSource(Source(), spec).window(0, 1)
    expected = apply_declared_isp(batch.observation, spec)
    for camera in CAMERAS:
        torch.testing.assert_close(
            represented.observation.frames[camera].values,
            expected.frames[camera].values,
            rtol=0,
            atol=0,
        )
        assert not torch.equal(
            represented.observation.frames[camera].values, batch.observation.frames[camera].values
        )
    np.savez(
        tmp_path / "episode.npz", metadata_json=json.dumps({"task_description": "open the drawer"})
    )
    assert read_task_prompts(["episode.npz"], str(tmp_path)) == ["open the drawer"]


def test_language_tokenizer_failure_cannot_be_replaced_by_placeholder_tokens():
    from types import SimpleNamespace

    class BrokenTokenizer:
        def tokenize(self, *args):
            raise RuntimeError("tokenizer unavailable")

    policy = SimpleNamespace(
        base=SimpleNamespace(tokenizer=BrokenTokenizer()), protocol=SimpleNamespace(name="libero")
    )
    with pytest.raises(RuntimeError, match="tokenizer unavailable"):
        v12_eval._language_batch(policy, ["open the drawer"], np.zeros((1, 8)), None)


def test_multiple_windows_do_not_mix_episodes_or_discard_anchor_rows():
    from owac.backends.n02_v12_tokens import N02V12TokenSource
    from owac.frameworks.n02_full.sequence_v12 import replay_inputs, window_inputs

    batch = FakeBatch(rows=6)
    batch.history = 3
    model = tiny_model(n_queries=256, out_dim=2048)
    source = N02V12TokenSource(model, camera_ids=CAMERAS, history=3)
    with torch.no_grad():
        tokens = source(batch).values
        expected = torch.cat(
            [
                replay_inputs(model, window_inputs(batch, anchor))["rows"][-1].tokens
                for anchor in range(2)
            ]
        )
    assert tokens.shape == (2, 512, 2048)
    torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
    assert len(source.last_outputs) == 2


def test_zero_prior_mass_stays_zero_even_with_a_large_energy_advantage():
    from owac.frameworks.n02_full.inference_v12 import V12InferenceInput

    model = tiny_model()
    prior = torch.tensor([[0.0, 0.5, 0.5]])
    inp = V12InferenceInput(
        v12_eval.row_observation(FakeBatch().observation, 0), declared_prior=prior
    )
    log_prior, _ = model.inference._log_prior(inp, batch=1, device=torch.device("cpu"))
    support = model.inference._support(log_prior, torch.tensor([[-1e9, 0.0, 0.0]]))
    assert support[0, 0] == 0
    torch.testing.assert_close(support[0, 1:], torch.tensor([0.5, 0.5]))


def test_svg_embeds_a_decodable_png_rather_than_raw_pixels():
    import base64
    import struct
    import xml.etree.ElementTree as ET
    import zlib

    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    fragment = v12_viz._image(pixels, 0, 0, 3, 2, "real pixels")
    svg = f"<svg xmlns:xlink='http://www.w3.org/1999/xlink'>{fragment}</svg>"
    href = ET.fromstring(svg)[0].attrib["{http://www.w3.org/1999/xlink}href"]
    encoded = base64.b64decode(href.split(",", 1)[1])
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", encoded[16:24]) == (3, 2)
    offset, compressed = 8, b""
    while offset < len(encoded):
        length = struct.unpack(">I", encoded[offset : offset + 4])[0]
        kind = encoded[offset + 4 : offset + 8]
        payload = encoded[offset + 8 : offset + 8 + length]
        if kind == b"IDAT":
            compressed += payload
        offset += length + 12
    rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(2, 10)
    assert np.array_equal(rows[:, 1:].reshape(2, 3, 3), pixels)


def test_resize_preserves_subunit_attention_values_until_colour_mapping():
    values = np.array([[0.0, 0.0001], [0.0002, 0.0003]], dtype=np.float32)
    resized = v12_viz._resize(values, (8, 8), mode="nearest")
    assert resized.dtype == np.float32 and resized.max() == values.max()
    assert v12_viz.heat_rgb(resized).max() > 0


def test_dense_queries_match_every_scalar_on_a_rectangular_cropped_sensor():
    import dataclasses

    from owac.frameworks.n02_full.inference_v12 import V12InferenceInput

    model = tiny_model(camera_ids=("cam_a",), query_stride=1, max_queries_per_camera=0)
    frame = MeasurementFrame(
        values=torch.arange(3 * 32 * 64, dtype=torch.float32).reshape(1, 3, 32, 64),
        valid_mask=torch.ones(1, 32, 64, dtype=torch.bool),
        camera_id="cam_a",
        frame_ids=("4",),
        episode_ids=("ep",),
        metadata={"crop_offset": (10, 20)},
    )
    observation = ObservationBundle(
        frames={"cam_a": frame}, context=EmbodimentContext(), episode_ids=("ep",)
    )
    inp = V12InferenceInput(observation, None, [4])
    spatial = model.inference.encoder(observation)
    queries = model.inference.build_queries(inp, batch=1)
    grid, inside = model.inference.decoder._query_grid(spatial, 0, queries.coords)
    assert inside.all()
    torch.testing.assert_close(queries.coords[0, 0], torch.tensor([10.5, 20.5]))
    torch.testing.assert_close(queries.coords[0, 63], torch.tensor([10.5, 83.5]))
    # Sampling a sensor ramp must recover all matching labels, including rightmost columns.
    raster = frame.values.flatten(0, 0)
    sampled = torch.nn.functional.grid_sample(
        raster, grid.unsqueeze(2), align_corners=False
    ).squeeze(-1)
    channels = queries.query_channel.unsqueeze(1)
    actual = sampled.gather(1, channels).squeeze(1)
    torch.testing.assert_close(actual, queries.target, rtol=0, atol=0.001)
    slots = model.build_state().slot_hidden
    _, _, valid, _ = model.inference._decode(spatial, slots, torch.tensor([4]), queries)
    assert int(valid[0, 0].sum()) == 3 * 32 * 64
    # Same geometry without crop metadata still covers the same count.
    observation.frames["cam_a"] = dataclasses.replace(frame, metadata={})
    plain = model.step(V12InferenceInput(observation, None, [4]), model.build_state())
    assert plain.diagnostics["scored_scalars_per_candidate"][0][0] == 3 * 32 * 64


def test_checkpointed_decoder_preserves_causal_loss_and_gradient():
    from owac.frameworks.n02_full.sequence_v12 import replay_inputs, window_inputs

    models = [tiny_model(checkpoint_decoder=enabled) for enabled in (False, True)]
    values, gradients = [], []
    batch = FakeBatch()
    batch.history = ROWS
    for model in models:
        model.train()
        output = replay_inputs(model, window_inputs(batch), grad_anchor=True)["rows"][-1]
        loss = output.tokens.square().mean() + output.causal["mean"].square().mean()
        loss.backward()
        values.append(loss.detach())
        gradients.append(
            [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
        )
    torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)
    for a, b in zip(gradients[0], gradients[1], strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
