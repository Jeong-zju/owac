"""Regression tests for independently found training, data and state defects."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from owac.baselines.n04 import BASELINE_NAMES
from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig, N04TrainConfig
from owac.frameworks.n04.model import N04Encoder
from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from owac.training.n04 import build_context, make_synthetic_batches, split_npz_paths, train


def cfg():
    return N04Config(
        measurement_size=(16, 16),
        geometry=N04GeometryConfig(feature_size=(16, 16), pixel_stride=1),
        model=N04ModelConfig(
            width=8, hidden=16, blocks=1, scales=1, readout_tokens=4, readout_dim=16
        ),
    )


def recipe():
    return N04TrainConfig(
        batch_size=1, sequence_length=5, steps=3, burn_in=1, tbptt=3, threads=2, log_every=0
    )


@pytest.mark.parametrize("baseline", [None, *BASELINE_NAMES])
def test_every_baseline_trains_and_saves_probe(baseline, tmp_path):
    result = train(cfg(), recipe(), output_dir=tmp_path, baseline=baseline)
    assert result.steps == 3
    payload = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert payload["schema_version"] == 2
    assert payload["location_head"]["weight"].abs().sum() > 0
    assert all(np.isfinite(list(row.values())).all() for row in result.history)
    if baseline is None:
        # Head starts at zero; this requires a real optimizer update, not just a gradient.
        assert payload["encoder"]["decomposer.head.weight"].abs().sum() > 0
        assert payload["encoder"]["propagator.head.weight"].abs().sum() > 0


def test_checkpoint_resume_is_numerically_equal_to_uninterrupted_training(tmp_path):
    batches = make_synthetic_batches(cfg=cfg(), train_cfg=recipe(), steps=3, seed=0)
    train(cfg(), recipe(), batches=batches, output_dir=tmp_path / "full")
    train(cfg(), recipe(), batches=batches[:1], output_dir=tmp_path / "resume")
    train(
        cfg(),
        recipe(),
        batches=batches,
        output_dir=tmp_path / "resume",
        resume=tmp_path / "resume/checkpoint.pt",
    )
    full = torch.load(tmp_path / "full/checkpoint.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resume/checkpoint.pt", weights_only=False)
    for group in ("encoder", "predictor_head", "location_head"):
        for key, value in full[group].items():
            assert torch.equal(value, resumed[group][key]), (group, key)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_cfa_forward_and_gradient_for_all_phases(pattern):
    encoder = N04Encoder(replace(cfg(), input_mode="bayer_cfa", channels=1, cfa_pattern=pattern))
    frame = SyntheticSequenceGenerator(
        SceneConfig(height=16, width=16, input_mode="bayer_cfa", cfa_pattern=pattern)
    ).sequence(length=2)[0]
    out = encoder.step(frame.measured, build_context(frame, None), encoder.init_state(1)["head"])
    tokens = encoder.readout_tokens(out)
    tokens.values.square().mean().backward()
    assert encoder.stem.conv.weight.grad.abs().sum() > 0
    assert tokens.mask.any()


def test_partial_reset_preserves_other_samples_history():
    encoder = N04Encoder(cfg().with_model(predictor="identity"))
    frames = SyntheticSequenceGenerator(SceneConfig(height=16, width=16)).sequence(
        batch=2, length=2
    )
    state = encoder.step(
        frames[0].measured, build_context(frames[0], None), encoder.init_state(2)["head"]
    ).state
    state = encoder.reset({"head": state}, torch.tensor([True, False]))["head"]
    assert state.history_valid.tolist() == [False, True]
    _, valid, _, _ = encoder.predict_before_observe(state, build_context(frames[1], frames[0]))
    assert valid[0].max() == 0 and valid[1].max() > 0


def test_invalid_pixels_cannot_leak_through_global_stem_normalization():
    encoder = N04Encoder(cfg())
    frames = SyntheticSequenceGenerator(SceneConfig(height=16, width=16)).sequence(length=2)
    valid = torch.ones(1, 1, 16, 16)
    valid[:, :, :, 8:] = 0
    corrupted = frames[0].measured.clone()
    corrupted[:, :, :, 8:] += 100
    state = encoder.init_state(1)["head"]
    context = build_context(frames[0], None)
    a = encoder.step(frames[0].measured, context, state, measurement_valid=valid)
    b = encoder.step(corrupted, context, state, measurement_valid=valid)
    assert torch.equal(a.representation, b.representation)
    assert a.validity[:, :, :, 8:].sum() == 0


def test_variants_split_by_source_identity_and_never_force_break_a_group(tmp_path):
    paths = []
    for episode in range(4):
        directory = tmp_path / f"episode_{episode:06}"
        directory.mkdir()
        for variant in range(3):
            path = directory / f"variant_{variant:02}.npz"
            np.savez(path, data=np.zeros(1))
            paths.append(path)
    left, right = split_npz_paths(paths, val_fraction=0.5)
    assert {p.parent for p in left}.isdisjoint(p.parent for p in right)
    one = paths[:3]
    left, right = split_npz_paths(one, val_fraction=0.999)
    assert len(left) in (0, 3) and len(right) in (0, 3)


def test_object_stops_without_teleport_and_delayed_context_really_differs():
    generator = SyntheticSequenceGenerator(
        SceneConfig(height=16, width=16, stop_after=4, action_delay=2)
    )
    sequence = generator.sequence(length=8, condition="object_only")
    assert torch.equal(sequence[4].true_object_translation, sequence[7].true_object_translation)
    assert not any(sequence[5].metadata["object_moving"])
    base = generator.sequence(length=8, condition="camera_only")
    delayed = generator.sequence(length=8, condition="action_delay")
    assert not torch.equal(
        build_context(base[1], base[0]).proprio_delta,
        build_context(delayed[1], delayed[0]).proprio_delta,
    )
    assert all(torch.equal(a.measured, b.measured) for a, b in zip(base, delayed, strict=True))


def test_coarse_resolution_does_not_change_flow_units():
    encoder = N04Encoder(
        replace(
            cfg(),
            measurement_size=(64, 64),
            geometry=N04GeometryConfig(feature_size=(64, 64), pixel_stride=1),
        )
    )
    frames = SyntheticSequenceGenerator(SceneConfig(height=64, width=64)).sequence(length=2)
    context = build_context(frames[1], frames[0])
    encoder.step(frames[0].measured, context, encoder.init_state(1)["head"])
    with torch.no_grad():
        encoder.propagator.head.bias[:2].fill_(4)
    _, _, _, propagation = encoder.predict_before_observe(encoder.init_state(1)["head"], context)
    assert propagation.forward_flow.abs().mean() > 3  # feature px, not coarse-normalized .5px


def test_missing_motion_fields_mask_each_batch_sample():
    from owac.frameworks.n04.ego import EgoMotionContext

    context = EgoMotionContext(
        proprio_delta=torch.ones(2, 4),
        proprio_delta_available=torch.tensor([True, False]),
        executed_action=torch.ones(2, 3),
        executed_action_available=torch.tensor([False, True]),
    )
    vector, _ = context.motion_vector()
    assert torch.equal(vector[0], torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(vector[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))


def test_predict_correct_reused_for_one_dimensional_signal():
    from owac.frameworks.n04.ego import EgoMotionContext

    config = replace(
        cfg().with_model(predictor="identity"),
        measurement_size=(2, 32),
        geometry=N04GeometryConfig(feature_size=(2, 32), pixel_stride=1),
    )
    encoder = N04Encoder(config)
    x = torch.arange(32).float()
    signal = (0.1 + 0.5 * torch.exp(-((x - 12) / 3).square()))[None, None, None]
    signal = signal.expand(1, 3, 2, 32)
    context = EgoMotionContext()
    initial = encoder.step(signal, context, encoder.init_state(1)["head"])
    predicted, valid, uncertainty, _ = encoder.predict_before_observe(initial.state, context)
    assert torch.equal(predicted, initial.state.stable)
    changed = encoder.encode_measurement(signal.roll(3, -1))
    corrected, innovation = encoder.correct_after_observe(
        initial.state, predicted, changed, valid, uncertainty
    )
    assert corrected.stable.shape == (1, 8, 2, 32)
    assert innovation.residual.abs().sum() > 0
    assert not torch.equal(corrected.stable, predicted)


def test_fixed_raw_projection_seed_and_frozen_weights():
    from owac.baselines.n04 import DirectBaseline

    a, b = DirectBaseline(cfg(), "fixed_raw_patch"), DirectBaseline(cfg(), "fixed_raw_patch")
    assert torch.equal(a.stem.weights, b.stem.weights)
    assert not list(a.stem.parameters())
    assert torch.allclose(a.stem.weights.flatten(1).norm(dim=1), torch.ones(8))


def test_lower_scale_parameters_receive_real_training_updates(tmp_path):
    config = cfg().with_model(scales=2)
    train(config, recipe(), output_dir=tmp_path)
    payload = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    torch.manual_seed(0)
    initial = N04Encoder(config)
    assert not torch.equal(payload["encoder"]["downsample.0.weight"], initial.downsample[0].weight)


def test_invisible_target_has_no_fabricated_centroid_score():
    import math

    from owac.evaluation.n04 import probe_metrics

    head = torch.nn.Conv2d(8, 1, 1)
    error, iou, _ = probe_metrics(head, torch.ones(2, 8, 8, 8), torch.zeros(2, 1, 8, 8))
    assert math.isnan(error) and math.isnan(iou)


def test_discrete_state_tokenization_preserves_frozen_padding():
    from types import SimpleNamespace

    from owac.backends.rawvla_features_n04 import tokenize_backend_task

    class Recorder:
        def tokenize(self, prompt, state):
            self.state = state
            return [1], [True]

    tokenizer = Recorder()
    backend = SimpleNamespace(
        max_state_dim=32, state_pad_value=-2.0, discrete_state_input=True, tokenizer=tokenizer
    )
    tokenize_backend_task(backend, "task", torch.arange(14).float())
    assert tokenizer.state.shape == (32,)
    assert np.array_equal(tokenizer.state[:14], np.arange(14))
    assert np.all(tokenizer.state[14:] == -2)
    backend.discrete_state_input = False
    tokenize_backend_task(backend, "task", torch.arange(8).float())
    assert tokenizer.state is None


def test_http_policy_transport_infer_reset_and_bad_request():
    import json
    import threading
    import urllib.request
    from urllib.error import HTTPError

    from owac.backends.rawvla_feature_server import make_server

    class FixturePolicy:
        calls = 0

        def predict_action(self, frame_id):
            self.calls += 1
            return {"frame_id": frame_id}

        def reset(self):
            self.calls = 0

    policy = FixturePolicy()
    server = make_server(policy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        request = urllib.request.Request(url + "/infer", data=b'{"frame_id": 3}')
        with urllib.request.urlopen(request) as response:
            assert json.load(response) == {"frame_id": 3}
        assert policy.calls == 1
        with urllib.request.urlopen(urllib.request.Request(url + "/reset", data=b"{}")):
            pass
        assert policy.calls == 0
        with pytest.raises(HTTPError):
            urllib.request.urlopen(urllib.request.Request(url + "/bad", data=b"{}"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
