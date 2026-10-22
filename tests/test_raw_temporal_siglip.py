"""CPU tests for the persistent causal RAW temporal wrapper (no optimizer step)."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from owac.baselines.raw_stem_siglip import (
    LocalConvStem,
    RawStemInput,
    RawStemSiglipStudent,
)
from owac.baselines.raw_temporal_siglip import (
    TEMPORAL_HISTORY_LENGTH,
    TemporalPacket,
    TemporalRawStemStudent,
    TemporalState,
    causal_frame_ids,
    causal_padding_mask,
    export_temporal_student_artifact,
    load_temporal_student_artifact,
)


class TinyConfig:
    def __init__(self, *, image_size=8, patch_size=4, hidden_size=8, layers=2):
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.num_channels = 3
        self.layer_norm_eps = 1e-5
        self.attention_dropout = 0.0
        self._attn_implementation = "eager"
        self.num_attention_heads = 2
        self.intermediate_size = 16
        self.projection_dim = 6


class TinyEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            3, config.hidden_size, config.patch_size, config.patch_size
        )
        count = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(count, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(count).unsqueeze(0), persistent=False)


class TinyLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width * 2, width)

    def forward(self, hidden):
        return hidden + self.fc2(self.act(self.fc1(self.norm(hidden))))


class TinyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(config.hidden_size) for _ in range(config.layers)])

    def forward(self, inputs_embeds, **kwargs):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = TinyEmbeddings(config)
        self.encoder = TinyEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        hidden = self.embeddings.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        hidden = hidden + self.embeddings.position_embedding(self.embeddings.position_ids)
        hidden = self.encoder(inputs_embeds=hidden).last_hidden_state
        return SimpleNamespace(last_hidden_state=self.post_layernorm(hidden))


class TinyVisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = TinyVisionModel(config)


class TinyProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.projection_dim)

    def forward(self, features):
        return self.linear(features)


class TinyCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_tower = TinyVisionTower(config)
        self.multi_modal_projector = TinyProjector(config)


class TinyContainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TinyCore(config)


class TinyVLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TinyContainer(config)
        self.burst_frames = 1
        self.temporal_teacher = False

    def embed_image(self, image):
        features = self.model.model.vision_tower(image).last_hidden_state
        return self.model.model.multi_modal_projector(features)


class TinyPolicy(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.vlm = TinyVLM(TinyConfig())


def fixture_student(mode="rawvla_rgb3", *, stem_seed=0):
    policy = TinyPolicy(seed=0)
    return RawStemSiglipStudent.from_fixture(
        policy, input_mode=mode, fixture_id="tiny-temporal", stem_seed=stem_seed
    )


def make_frame(
    value, *, cameras=("base",), present=None, valid_mask=None, input_mode="rawvla_rgb3"
):
    kwargs = {}
    if present is not None:
        kwargs["camera_present"] = present
    if valid_mask is not None:
        kwargs["valid_mask"] = valid_mask
    channels = 3
    return RawStemInput(
        values=torch.full((1, len(cameras), channels, 8, 8), float(value)),
        input_mode=input_mode,
        camera_ids=tuple(cameras),
        units="teacher_rgb_minus1_1" if input_mode == "rgb3_control" else "normalized_0_1",
        **kwargs,
    )


def make_packet(
    endpoint,
    *,
    values=None,
    cameras=("base",),
    present=None,
    valid_masks=None,
    source="source-a",
    reset=False,
    input_mode="rawvla_rgb3",
):
    if input_mode == "rgb3_control":
        frame_ids = (endpoint,)
        padding = (False,)
        value = 0.1 if values is None else values[-1]
        mask = None if valid_masks is None else valid_masks[-1]
        frames = [
            make_frame(
                value,
                cameras=cameras,
                present=present,
                valid_mask=mask,
                input_mode=input_mode,
            )
        ]
    else:
        frame_ids = causal_frame_ids(endpoint)
        padding = causal_padding_mask(endpoint)
        frames = []
        for index, frame_id in enumerate(frame_ids):
            value = (frame_id + 1) / 10.0 if values is None else values[index]
            mask = None if valid_masks is None else valid_masks[index]
            frames.append(
                make_frame(
                    value,
                    cameras=cameras,
                    present=present,
                    valid_mask=mask,
                    input_mode=input_mode,
                )
            )
    return TemporalPacket(
        frames=tuple(frames),
        source_demo_id=source,
        endpoint=endpoint,
        frame_ids=frame_ids,
        padding_mask=padding,
        reset=reset,
    )


def test_first_endpoint_equals_per_frame_student_and_state_shape():
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    packet = make_packet(0, reset=True)
    output, state = student(packet, None)
    reference = base(packet.frames[-1])
    torch.testing.assert_close(output.projected_tokens, reference.projected_tokens, rtol=0, atol=0)
    assert state.hidden.shape == (1, 1, 4, 8)
    assert state.last_endpoint == 0
    assert state.source_demo_id == "source-a"
    assert state.model_identity == student.model_identity
    assert student.gate_logit.numel() == student.geometry.hidden_size
    torch.testing.assert_close(student.gate_alpha, torch.full((8,), 0.5))
    assert isinstance(student.stem, LocalConvStem)
    assert output.measurement_metadata["temporal"]["endpoint"] == 0
    assert output.measurement_metadata["temporal"]["frame_ids"] == [0, 0, 0, 0, 0, 0]


def test_mask_aware_mean_skips_left_padding_and_recurrence_is_channelwise():
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    first = make_packet(0, values=[0.1] * 6, reset=True)
    _, state0 = student(first, None)
    later = make_packet(5, values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    mean, has, _ = student._patch_burst(later)
    expected = 0.5 * mean + 0.5 * state0.hidden
    _, state1 = student(later, state0)
    torch.testing.assert_close(state1.hidden, expected, rtol=1e-6, atol=1e-6)
    assert bool(has.all())
    # The first endpoint's duplicated frame-0 positions are padding, not six
    # independent measurements; a zero-frame input still equals one frame.
    padded = make_packet(0, values=[0.0] * 6, reset=True)
    out_padded, _ = student(padded, None)
    out_reference = base(padded.frames[-1])
    torch.testing.assert_close(out_padded.projected_tokens, out_reference.projected_tokens)


def test_reset_replaces_source_and_state_exactly():
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    _, state0 = student(make_packet(0, reset=True), None)
    reset_packet = make_packet(0, values=[0.7] * 6, source="source-b", reset=True)
    out_reset, state_reset = student(reset_packet, state0)
    out_fresh, state_fresh = student(reset_packet, None)
    torch.testing.assert_close(out_reset.projected_tokens, out_fresh.projected_tokens)
    torch.testing.assert_close(state_reset.hidden, state_fresh.hidden)
    assert state_reset.source_demo_id == "source-b"


def test_state_snapshot_continuation_and_disk_roundtrip(tmp_path):
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    _, state0 = student(make_packet(0, reset=True), None)
    later = make_packet(5)
    expected, expected_state = student(later, state0)
    snapshot = student.snapshot_state(state0)
    loaded = student.state_from_snapshot(snapshot)
    actual, actual_state = student(later, loaded)
    torch.testing.assert_close(actual.projected_tokens, expected.projected_tokens)
    torch.testing.assert_close(actual_state.hidden, expected_state.hidden)
    path = state0.save(tmp_path / "state.pt")
    disk = TemporalState.load(path)
    _, disk_state = student(later, disk)
    torch.testing.assert_close(disk_state.hidden, expected_state.hidden)
    assert snapshot["hidden"].dtype == torch.float32
    assert snapshot["hidden"].device.type == "cpu"


def test_state_identity_and_carry_rejections():
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    _, state0 = student(make_packet(0, reset=True), None)
    with pytest.raises(ValueError, match="first temporal packet"):
        student(make_packet(5), None)
    with pytest.raises(ValueError, match="source changed"):
        student(make_packet(5, source="other"), state0)
    with pytest.raises(ValueError, match="requires endpoint"):
        student(make_packet(6), state0)
    tampered = TemporalState(
        hidden=state0.hidden,
        source_demo_id=state0.source_demo_id,
        camera_order=state0.camera_order,
        last_endpoint=state0.last_endpoint,
        model_identity="0" * 64,
    )
    with pytest.raises(ValueError, match="model identity"):
        student(make_packet(5), tampered)
    with pytest.raises(ValueError, match="camera order changed"):
        two_camera_state = student(make_packet(0, cameras=("base", "wrist"), reset=True), None)[1]
        student(
            make_packet(5, cameras=("wrist", "base"), source="source-a"),
            two_camera_state,
        )


def test_future_stride_padding_and_raw_range_denials():
    with pytest.raises(ValueError, match="frame_ids"):
        TemporalPacket(
            frames=tuple(make_frame(0.2) for _ in range(6)),
            source_demo_id="s",
            endpoint=5,
            frame_ids=(1, 2, 3, 4, 5, 6),
            padding_mask=(False,) * 6,
        )
    with pytest.raises(ValueError, match="padding_mask"):
        TemporalPacket(
            frames=tuple(make_frame(0.2) for _ in range(6)),
            source_demo_id="s",
            endpoint=0,
            frame_ids=causal_frame_ids(0),
            padding_mask=(False,) * 6,
        )
    with pytest.raises(ValueError, match="batch size 1"):
        TemporalPacket(
            frames=tuple(
                RawStemInput(
                    values=torch.zeros(2, 1, 3, 8, 8),
                    input_mode="rawvla_rgb3",
                    camera_ids=("base",),
                    units="normalized_0_1",
                )
                for _ in range(6)
            ),
            source_demo_id="s",
            endpoint=0,
            frame_ids=causal_frame_ids(0),
            padding_mask=causal_padding_mask(0),
        )


def test_missing_current_camera_does_not_update_its_state():
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    both = torch.tensor([[True, True]])
    missing_second = torch.tensor([[True, False]])
    first = make_packet(0, cameras=("base", "wrist"), present=both, reset=True)
    _, state0 = student(first, None)
    assert state0.hidden[0, 1].abs().sum() > 0
    later = make_packet(5, cameras=("base", "wrist"), present=missing_second)
    output, state1 = student(later, state0)
    mask = output.tokens.mask[0].reshape(2, 4)
    assert mask[0].all()
    assert not mask[1].any()
    torch.testing.assert_close(state1.hidden[0, 1], state0.hidden[0, 1])
    assert state1.hidden[0, 0].abs().sum() > 0


def test_gradient_through_carry_reaches_stem_and_gate_only():
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    student.configure_trainable_scope("stem_only")
    state0 = student.update_state(make_packet(0, reset=True), None)
    output, _ = student(make_packet(5), state0)
    output.projected_tokens.float().square().mean().backward()
    assert student.gate_logit.grad is not None
    assert student.gate_logit.grad.abs().sum() > 0
    assert student.stem.conv64.weight.grad is not None
    assert student.stem.conv64.weight.grad.abs().sum() > 0
    assert student.stem.patch_embedding.weight.grad is not None
    assert student.stem.patch_embedding.weight.grad.abs().sum() > 0
    assert student.vision_tower.vision_model.encoder.layers[0].fc1.weight.grad is None
    assert student.projector.linear.weight.grad is None
    report = student.gradient_report()
    assert report["all_trainable_have_gradients"]
    assert any("gate_logit" in entry["name"] for entry in report["entries"])
    assert any("base.vision_tower" in entry["name"] for entry in report["entries"])


def test_detach_burn_in_and_scope_reports():
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    burn = student.detach_state(student.update_state(make_packet(0, reset=True), None))
    assert burn.hidden.requires_grad is False
    assert burn.hidden.grad_fn is None
    output, _ = student(make_packet(5), burn)
    output.projected_tokens.float().square().mean().backward()
    assert burn.hidden.grad is None
    assert all(name == "gate_logit" or name.startswith("base.") for name in student.state_dict())
    report = student.trainable_report()
    assert report["scope"] == "stem_only"
    assert "gate_logit" in report["trainable_names"]
    assert any(name.startswith("base.") for name in report["trainable_names"])
    assert report["gate_parameters"] == student.geometry.hidden_size
    student.configure_trainable_scope("stem_plus_input_blocks", input_blocks=1, explicit=True)
    assert student.gate_logit.requires_grad
    assert student.vision_tower.vision_model.encoder.layers[0].fc1.weight.requires_grad
    assert not student.vision_tower.vision_model.encoder.layers[1].fc1.weight.requires_grad
    student.configure_trainable_scope("all_frozen")
    assert student.gate_logit.requires_grad is False
    assert set(student.dependency_audit()) == {
        "teacher",
        "isp",
        "cache",
        "rgb_generation",
        "privileged_rgb_control",
    }
    assert not any(student.dependency_audit()[key] for key in ("teacher", "isp", "cache"))


def test_malformed_snapshot_denied():
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    _, state = student(make_packet(0, reset=True), None)
    snapshot = state.snapshot()
    with pytest.raises(ValueError, match="hidden|dtype|CPU float32"):
        TemporalState.from_snapshot({**snapshot, "hidden": snapshot["hidden"].double()})
    missing = dict(snapshot)
    del missing["camera_order"]
    with pytest.raises(ValueError, match="unexpected/missing"):
        TemporalState.from_snapshot(missing)


def test_artifact_roundtrip_preserves_gate_and_output(tmp_path):
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    with torch.no_grad():
        student.gate_logit.fill_(0.25)
    packet = make_packet(0, reset=True)
    expected, expected_state = student(packet, None)
    export_temporal_student_artifact(
        student,
        tmp_path / "artifact",
        backend_identity={"kind": "fixture-backend", "sha256": "b" * 64},
        tokenizer_identity={"sha256": "c" * 64},
        normalizer_identity={"sha256": "d" * 64},
    )
    assert (tmp_path / "artifact" / "temporal.json").is_file()
    assert (tmp_path / "artifact" / "temporal.pt").is_file()
    calls = {"factory": 0}

    def factory(geometry, input_mode):
        calls["factory"] += 1
        policy = TinyPolicy(seed=999)
        tower = policy.vlm.model.model.vision_tower
        projector = policy.vlm.model.model.multi_modal_projector
        return tower, projector

    loaded = load_temporal_student_artifact(
        tmp_path / "artifact",
        frontend_factory=factory,
    )
    assert calls["factory"] == 1
    assert not hasattr(loaded, "teacher")
    torch.testing.assert_close(loaded.gate_logit, student.gate_logit)
    assert loaded.model_identity != student.model_identity  # process-local generations
    actual, actual_state = loaded(packet, None)
    torch.testing.assert_close(actual.projected_tokens, expected.projected_tokens, rtol=0, atol=0)
    torch.testing.assert_close(actual_state.hidden, expected_state.hidden, rtol=0, atol=0)
    with pytest.raises(ValueError, match="model identity"):
        loaded.state_from_snapshot(expected_state.snapshot())


def test_rgb_control_temporal_packet_is_declared_diagnostic_privilege():
    base = fixture_student("rgb3_control")
    student = TemporalRawStemStudent(base)
    packet = make_packet(0, values=[0.1] * 6, input_mode="rgb3_control", reset=True)
    assert len(packet.frames) == 1
    output, state = student(packet, None)
    assert student.privileged_rgb_control is True
    assert student.input_domain == "teacher_rgb_control_temporal_diagnostic"
    assert output.input_mode == "rgb3_control"
    assert output.tokens.values.shape == (1, 4, 6)
    assert state.hidden.shape == (1, 1, 4, 8)
    assert output.measurement_metadata["temporal"]["history_length"] == 1
    assert output.measurement_metadata["temporal"]["privileged_rgb_control"] is True


def test_parameter_and_stem_mutations_invalidate_state_generation():
    first = TemporalRawStemStudent(fixture_student(stem_seed=0))
    _, state0 = first(make_packet(0, reset=True), None)
    old_identity = first.model_identity
    with torch.no_grad():
        first.stem.conv64.weight.add_(0.01)
    assert first.model_identity != old_identity
    with pytest.raises(ValueError, match="model identity"):
        first(make_packet(5), state0)

    # Frontend-only weights do not affect recurrent state identity.
    inherited_identity = first.model_identity
    with torch.no_grad():
        first.projector.linear.weight.add_(0.01)
    assert first.model_identity == inherited_identity


def test_timestamps_must_match_causal_ids_and_be_monotonic():
    packet = make_packet(5)
    with pytest.raises(ValueError, match="nondecreasing"):
        TemporalPacket(
            frames=packet.frames,
            source_demo_id=packet.source_demo_id,
            endpoint=packet.endpoint,
            frame_ids=packet.frame_ids,
            padding_mask=packet.padding_mask,
            timestamps=(0.0, 0.2, 0.1, 0.3, 0.4, 0.5),
        )
    with pytest.raises(ValueError, match="repeat exactly"):
        TemporalPacket(
            frames=packet.frames,
            source_demo_id=packet.source_demo_id,
            endpoint=packet.endpoint,
            frame_ids=packet.frame_ids,
            padding_mask=packet.padding_mask,
            timestamps=(0.0, 0.0, 0.2, 0.3, 0.4, 0.5),
        )
    valid = TemporalPacket(
        frames=packet.frames,
        source_demo_id=packet.source_demo_id,
        endpoint=packet.endpoint,
        frame_ids=packet.frame_ids,
        padding_mask=packet.padding_mask,
        timestamps=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    )
    assert valid.timestamps == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    padded0 = make_packet(0, reset=True)
    with pytest.raises(ValueError, match="repeat exactly"):
        TemporalPacket(
            frames=padded0.frames,
            source_demo_id=padded0.source_demo_id,
            endpoint=padded0.endpoint,
            frame_ids=padded0.frame_ids,
            padding_mask=padded0.padding_mask,
            timestamps=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        )


def test_two_camera_burst_keeps_explicit_camera_patch_axis():
    base = fixture_student()
    student = TemporalRawStemStudent(base)
    both = torch.tensor([[True, True]])
    packet = make_packet(0, cameras=("base", "wrist"), present=both, reset=True)
    output, state = student(packet, None)
    assert state.hidden.shape == (1, 2, 4, 8)
    assert output.tokens.camera_ids == ("base", "wrist")
    assert output.tokens.values.shape == (1, 8, 6)
    assert output.tokens.mask.shape == (1, 8)
    assert output.tokens.mask.all()


def test_no_teacher_imports_in_temporal_module():
    source = Path(inspect.getsourcefile(TemporalRawStemStudent)).read_text(encoding="utf-8")
    assert "rawvla_teacher" not in source
    assert "TeacherOfflineEvaluator" not in source
    assert "owac.evaluation" not in source
    assert "import owac.backends" not in source
    assert TEMPORAL_HISTORY_LENGTH == 6


def test_model_identity_is_version_generation_without_tensor_hashing(monkeypatch):
    import hashlib

    from owac.baselines.raw_stem_siglip import RawStemContractError

    student = TemporalRawStemStudent(fixture_student())
    packet0 = make_packet(0, reset=True)
    packet5 = make_packet(5)
    first_identity = student.model_identity
    state = None

    def forbidden_sha(*args, **kwargs):
        raise AssertionError("active temporal forward must not hash tensors")

    monkeypatch.setattr(hashlib, "sha256", forbidden_sha)
    output, state = student(packet0, state)
    assert output.projected_tokens is not None
    assert state.model_identity == first_identity
    assert student.model_identity == first_identity

    with torch.no_grad():
        student.gate_logit.add_(1.0)
    changed_identity = student.model_identity
    assert changed_identity != first_identity
    with pytest.raises(RawStemContractError, match="identity"):
        student(packet5, state)

    # Replacement and temporal-config changes also advance the generation.
    previous = changed_identity
    student.gate_logit = nn.Parameter(torch.zeros_like(student.gate_logit))
    assert student.model_identity != previous
    previous = student.model_identity
    student.privileged_rgb_control = not student.privileged_rgb_control
    assert student.model_identity != previous


def test_two_different_model_instances_cannot_reuse_generation_state():
    first = TemporalRawStemStudent(fixture_student())
    second = TemporalRawStemStudent(fixture_student())
    _, state = first(make_packet(0, reset=True), None)
    assert first.model_identity != second.model_identity
    with pytest.raises(ValueError, match="model identity"):
        second(make_packet(5), state)
