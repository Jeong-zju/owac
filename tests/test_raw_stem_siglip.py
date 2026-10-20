"""CPU contract tests for the inherited SigLIP RAW stem (no optimizer updates)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from owac.baselines.raw_stem_siglip import (
    CheckpointIdentity,
    HistoryContract,
    LocalConvStem,
    PolicyLoadEvidence,
    RawStemInput,
    RawStemSiglipStudent,
    SiglipTeacher,
    VisionGeometry,
    assert_backend_frozen,
    assert_bridge_has_no_projection,
    build_raw_stem_student,
    export_student_artifact,
    load_student_artifact,
    pack_bayer4,
)


class TinyConfig:
    def __init__(self, *, image_size=8, patch_size=4, hidden_size=8, layers=4):
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.num_channels = 3
        self.layer_norm_eps = 1e-5
        self.attention_dropout = 0.5
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

    def forward(self, pixel_values):
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return patches + self.position_embedding(self.position_ids)


class TinyLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(0.5)
        self.layer_norm2 = nn.LayerNorm(width)

    def forward(self, hidden):
        hidden = hidden + self.dropout(self.fc2(self.act(self.fc1(self.layer_norm1(hidden)))))
        return hidden + self.dropout(self.fc2(self.act(self.fc1(self.layer_norm2(hidden)))))


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
        hidden = self.embeddings(pixel_values)
        hidden = self.encoder(inputs_embeds=hidden).last_hidden_state
        return SimpleNamespace(last_hidden_state=self.post_layernorm(hidden))


class TinyVisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = TinyVisionModel(config)

    def forward(self, pixel_values):
        return self.vision_model(pixel_values)


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
    def __init__(self, config=None, *, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.vlm = TinyVLM(config or TinyConfig())


def make_policy(seed=0, **config_kwargs):
    return TinyPolicy(TinyConfig(**config_kwargs), seed=seed)


def raw_input(mode="rawvla_rgb3", *, batch=1, cameras=1, value=0.25, **kwargs):
    channels = 4 if mode == "bayer4_packed" else 3
    image_size = 4 if mode == "bayer4_packed" else 8
    values = torch.full((batch, cameras, channels, image_size, image_size), value)
    camera_ids = tuple(f"camera_{index}" for index in range(cameras))
    base = {
        "values": values,
        "input_mode": mode,
        "camera_ids": camera_ids,
        "units": "teacher_rgb_minus1_1" if mode == "rgb3_control" else "normalized_0_1",
    }
    base.update(kwargs)
    if mode == "bayer4_packed":
        base.setdefault("cfa_pattern", "RGGB")
    return RawStemInput(**base)


def fixture_student(mode="rawvla_rgb3", *, seed=0, stem_seed=0):
    policy = make_policy(seed)
    student = RawStemSiglipStudent.from_fixture(
        policy,
        input_mode=mode,
        fixture_id=f"tiny-{mode}",
        stem_seed=stem_seed,
    )
    return policy, student


def test_inherited_weights_are_exact_deep_copies_and_stem_is_local():
    policy, student = fixture_student()
    teacher = SiglipTeacher.from_fixture(policy, fixture_id="tiny")
    original = policy.vlm.model.model.vision_tower.vision_model
    copied = student.vision_tower.vision_model
    torch.testing.assert_close(
        original.embeddings.position_embedding.weight,
        copied.embeddings.position_embedding.weight,
        rtol=0,
        atol=0,
    )
    for before, after in zip(original.encoder.layers, copied.encoder.layers, strict=True):
        for (name, a), (other, b) in zip(
            before.named_parameters(), after.named_parameters(), strict=True
        ):
            assert name == other
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(
        original.post_layernorm.weight, copied.post_layernorm.weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        policy.vlm.model.model.multi_modal_projector.linear.weight,
        student.projector.linear.weight,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        policy.vlm.model.model.vision_tower.vision_model.embeddings.patch_embedding.weight[:, :1],
        student.stem.patch_embedding.weight[:, :1],
    )
    assert student.stem is copied.embeddings.patch_embedding
    assert student.stem.conv_channels == 64
    assert isinstance(student.stem, LocalConvStem)
    assert student.bridge_projection is None
    assert teacher.geometry == student.geometry
    # Deep copies do not share storage.
    assert (
        student.position_embedding.weight.data_ptr()
        != original.embeddings.position_embedding.weight.data_ptr()
    )


def test_stage1_freeze_scope_and_teacher_stability():
    policy, student = fixture_student()
    report = student.trainable_report()
    assert report["scope"] == "stem_only"
    assert report["trainable_parameters"] > 0
    assert all("patch_embedding" in name for name in report["trainable_names"])
    assert report["backend_trainable"] is False
    assert report["bridge_projection"] is None
    teacher = SiglipTeacher.from_fixture(policy, fixture_id="tiny")
    rgb = torch.zeros(1, 3, 8, 8)
    rgb[:, :, 2:5, 3:6] = 0.5
    before = teacher(rgb)
    hashes = teacher.current_visual_hashes().copy()
    with torch.no_grad():
        student.stem.patch_embedding.weight.add_(3.0)
    after = teacher(rgb)
    torch.testing.assert_close(before.projected, after.projected, rtol=0, atol=0)
    assert before.sha256 == after.sha256
    assert hashes == teacher.current_visual_hashes()

    student.configure_trainable_scope("all_frozen")
    assert student.trainable_report()["trainable_parameters"] == 0
    student.configure_trainable_scope("stem_only")
    with pytest.raises(ValueError, match="explicit"):
        student.configure_trainable_scope("stem_plus_input_blocks", input_blocks=2)
    scoped = student.configure_trainable_scope(
        "stem_plus_input_blocks", input_blocks=2, explicit=True
    )
    assert scoped["input_blocks_unfrozen"] == 2
    trainable = scoped["trainable_names"]
    assert any("encoder.layers.0" in name for name in trainable)
    assert any("encoder.layers.1" in name for name in trainable)
    assert not any("encoder.layers.2" in name for name in trainable)
    student.train(True)
    assert student.stem.training is True
    assert student.vision_tower.vision_model.encoder.layers[0].training is True
    assert student.vision_tower.vision_model.encoder.layers[2].training is False
    assert student.projector.training is False
    assert student.post_layernorm.training is False


def test_stem_gradients_flow_through_frozen_tower_and_projector():
    _, student = fixture_student()
    student.configure_trainable_scope("stem_only")
    inp = raw_input()
    output = student(inp)
    loss = output.projected_tokens.float().square().mean()
    loss.backward()
    assert student.stem.patch_embedding.weight.grad is not None
    assert student.stem.patch_embedding.weight.grad.abs().sum() > 0
    assert student.stem.conv64.weight.grad is not None
    assert student.stem.conv64.weight.grad.abs().sum() > 0
    assert student.position_embedding.weight.grad is None
    assert student.projector.linear.weight.grad is None
    assert student.vision_tower.vision_model.encoder.layers[-1].fc1.weight.grad is None
    grad = student.gradient_report()
    assert grad["all_trainable_have_gradients"]
    assert all(entry["grad_present"] for entry in grad["entries"])
    assert output.projected_tokens.requires_grad
    assert output.hidden_tokens.shape == (1, 4, 8)


def test_position_and_projector_semantics_match_explicit_reference():
    _, student = fixture_student()
    inp = raw_input(value=0.125)
    output = student(inp)
    vision = student.vision_tower.vision_model
    stem = student.stem
    flat = inp.values.flatten(0, 1).to(stem.conv64.weight.dtype)
    patches = stem(flat).flatten(2).transpose(1, 2)
    hidden = patches + vision.embeddings.position_embedding(vision.embeddings.position_ids)
    hidden = vision.encoder(inputs_embeds=hidden).last_hidden_state
    hidden = vision.post_layernorm(hidden)
    projected = student.projector(hidden).reshape(1, 4, 6)
    torch.testing.assert_close(output.projected_tokens, projected, rtol=0, atol=0)
    assert output.hidden_tokens.shape == (1, 4, 8)
    assert output.tokens.values.shape == (1, 4, 6)
    assert output.tokens.camera_ids == ("camera_0",)
    positions = output.tokens.positions
    assert positions.shape == (1, 4, 3)
    expected_y = torch.tensor([2.0, 2.0, 6.0, 6.0])
    expected_x = torch.tensor([2.0, 6.0, 2.0, 6.0])
    torch.testing.assert_close(positions[0, :, 0], expected_y)
    torch.testing.assert_close(positions[0, :, 1], expected_x)
    torch.testing.assert_close(positions[0, :, 2], torch.full((4,), 4.0))


def test_camera_major_permutation_and_strict_shape():
    _, student = fixture_student()
    values = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 2, 3, 8, 8) / 1000
    first = student(
        RawStemInput(
            values=values,
            input_mode="rawvla_rgb3",
            camera_ids=("base", "wrist"),
            units="cache_units",
        )
    )
    second = student(
        RawStemInput(
            values=values.flip(1),
            input_mode="rawvla_rgb3",
            camera_ids=("wrist", "base"),
            units="cache_units",
        )
    )
    assert first.tokens.camera_ids == ("base", "wrist")
    assert second.tokens.camera_ids == ("wrist", "base")
    torch.testing.assert_close(first.tokens.values[:, :4], second.tokens.values[:, 4:])
    torch.testing.assert_close(first.tokens.values[:, 4:], second.tokens.values[:, :4])
    torch.testing.assert_close(first.tokens.positions[:, :4], second.tokens.positions[:, 4:])
    with pytest.raises(ValueError, match="spatial shape"):
        student(
            RawStemInput(
                values=torch.zeros(1, 1, 3, 7, 8),
                input_mode="rawvla_rgb3",
                camera_ids=("base",),
                units="cache_units",
            )
        )
    with pytest.raises(ValueError, match="camera_ids"):
        RawStemInput(
            values=torch.zeros(1, 2, 3, 8, 8),
            input_mode="rawvla_rgb3",
            camera_ids=("only",),
            units="cache_units",
        )


def test_bayer_packing_physical_positions_and_odd_geometry_rejection():
    mosaic = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    packed = pack_bayer4(mosaic, "RGGB")
    assert packed.values.shape == (2, 4, 4, 4)
    torch.testing.assert_close(packed.values[:, 0], mosaic[:, 0, 0::2, 0::2])
    torch.testing.assert_close(packed.values[:, 1], mosaic[:, 0, 0::2, 1::2])
    torch.testing.assert_close(packed.values[:, 2], mosaic[:, 0, 1::2, 0::2])
    torch.testing.assert_close(packed.values[:, 3], mosaic[:, 0, 1::2, 1::2])
    assert packed.plane_order == ("R", "G", "G", "B")

    _, student = fixture_student("bayer4_packed")
    inp = packed.to_input(
        camera_ids=("base",),
        units="raw_dn",
        black_level=[0, 0, 0, 0],
        white_level=[1023, 1023, 1023, 1023],
        physical_origins=((0, 0),),
    )
    out = student(inp)
    assert out.tokens.values.shape == (2, 4, 6)
    torch.testing.assert_close(out.tokens.positions[0, :, 0], torch.tensor([2.0, 2.0, 6.0, 6.0]))
    torch.testing.assert_close(out.tokens.positions[0, :, 1], torch.tensor([2.0, 6.0, 2.0, 6.0]))
    # The stem consumes half-resolution packed strides, not an interpolated grid.
    assert student.stem.patch_stride == 2
    with pytest.raises(ValueError, match="even aligned"):
        student(
            packed.to_input(
                camera_ids=("base",),
                units="raw_dn",
                black_level=0,
                white_level=1023,
                physical_origins=((1, 0),),
            )
        )
    odd = VisionGeometry(
        hidden_size=8,
        grid_height=2,
        grid_width=2,
        patch_size=3,
        image_size=6,
        num_positions=4,
        encoder_layers=1,
        projector_in=8,
        projector_out=6,
        dtype="float32",
        post_layernorm_eps=1e-5,
    )
    with pytest.raises(ValueError, match="even physical patch"):
        odd.packed_patch_stride("bayer4_packed")


def test_bad_missing_saturation_and_calibration_availability():
    _, student = fixture_student()
    valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    valid[0, 0, 0, 0] = False
    saturated = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    saturated[0, 0, 0, 4:8] = True
    out = student(
        raw_input(
            value=0.5,
            valid_mask=valid,
            saturated_mask=saturated,
        )
    )
    mask = out.tokens.mask[0]
    assert not mask[0].item()
    assert not mask[1].item()
    assert mask[2:].all()
    assert out.tokens.validity is not None
    assert torch.equal(out.tokens.validity.valid_mask, mask.unsqueeze(0))
    assert out.availability[0]["valid_mask"] == "available"
    assert out.availability[0]["saturated_mask"] == "available"
    assert out.availability[0]["calibration"] == "unavailable"
    assert out.availability[0]["exposure"] == "unavailable"

    # Calibrated raw values and pre-normalised values produce identical tokens.
    calibrated = raw_input(mode="bayer4_packed", value=512.0)
    calibrated.black_level = [0, 0, 0, 0]
    calibrated.white_level = [1023, 1023, 1023, 1023]
    calibrated.units = "raw_dn"
    normalized = raw_input(mode="bayer4_packed", value=512.0 / 1023.0)
    _, bayer_student = fixture_student("bayer4_packed")
    a = bayer_student(calibrated)
    b = bayer_student(normalized)
    torch.testing.assert_close(a.tokens.values, b.tokens.values, rtol=0, atol=0)


def test_teacher_single_frame_detached_and_student_independent():
    policy, student = fixture_student()
    teacher = SiglipTeacher.from_fixture(policy, fixture_id="tiny")
    rgb = torch.zeros(2, 3, 8, 8)
    rgb[:, 0] = 0.5
    rgb[:, 1] = -0.5
    before = teacher(rgb)
    assert before.projected.shape == (2, 4, 6)
    assert before.projected.requires_grad is False
    assert before.last_hidden_state is None
    with torch.no_grad():
        student.stem.conv64.weight.mul_(2.0)
    after = teacher(rgb)
    torch.testing.assert_close(before.projected, after.projected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="explicit HistoryContract"):
        SiglipTeacher.from_fixture(policy, fixture_id="tiny", temporal_frames=2)
    with pytest.raises(ValueError, match="preprocessed"):
        teacher(torch.zeros(1, 3, 7, 8))
    with pytest.raises(ValueError, match="preprocessed"):
        teacher(torch.full((1, 3, 8, 8), 2.0))
    policy.vlm.temporal_teacher = True
    with pytest.raises(RuntimeError, match="temporal|History"):
        SiglipTeacher.from_fixture(policy, fixture_id="tiny")


def test_real_mode_rejects_fixture_weights_and_unverified_evidence(tmp_path):
    policy = make_policy()
    checkpoint = CheckpointIdentity(
        path=str(tmp_path / "missing.safetensors"),
        sha256="a" * 64,
        verified=True,
        tensor_sha256={},
    )
    audit = PolicyLoadEvidence(
        checkpoint_sha256="a" * 64,
        policy_module="starVLA.model.modules.vlm.OpenPIPaliGemma",
        loaded_keys=10,
        model_keys=10,
        verified=True,
        acceptance="backbone_source_audit",
        visual_weights_source="backbone-source-audit",
    )
    with pytest.raises(ValueError, match="pinned SigLIP|real-policy acceptance"):
        build_raw_stem_student(
            policy,
            input_mode="rawvla_rgb3",
            checkpoint=checkpoint,
            load_audit=audit,
        )
    with pytest.raises(ValueError, match="fixture"):
        build_raw_stem_student(
            policy,
            input_mode="rawvla_rgb3",
            checkpoint=checkpoint,
            load_audit=audit,
            evidence_level="fixture_contract",
        )


def test_export_and_deployment_reload_only_frontend(tmp_path):
    _, student = fixture_student()
    inp = raw_input(value=0.375)
    expected = student(inp)
    metadata = export_student_artifact(
        student,
        tmp_path / "artifact",
        backend_identity={"kind": "fixture-backend", "sha256": "b" * 64},
        tokenizer_identity={"sha256": "c" * 64},
        normalizer_identity={"sha256": "d" * 64},
    )
    assert metadata["bridge_projection"] is None
    assert metadata["vision_architecture"]["config"]["hidden_size"] == 8
    assert metadata["vision_architecture"]["projector_linear"]["out_features"] == 6
    assert "teacher" not in json.dumps(metadata["state_dict_keys"]).lower()
    calls = {"factory": 0}

    def factory(geometry, input_mode):
        calls["factory"] += 1
        # A blank architecture only: no checkpoint, teacher, ISP or cache.
        policy = make_policy(seed=999)
        tower = policy.vlm.model.model.vision_tower
        projector = policy.vlm.model.model.multi_modal_projector
        assert geometry.image_size == 8
        assert input_mode == "rawvla_rgb3"
        return tower, projector

    deployment = load_student_artifact(
        tmp_path / "artifact",
        frontend_factory=factory,
        expected_state_dict_sha256=metadata["state_dict_sha256"],
    )
    assert calls["factory"] == 1
    assert not hasattr(deployment, "vlm")
    assert not hasattr(deployment, "teacher")
    assert deployment.bridge_projection is None
    actual = deployment(inp)
    torch.testing.assert_close(actual.projected_tokens, expected.projected_tokens, rtol=0, atol=0)
    assert actual.tokens.camera_ids == expected.tokens.camera_ids
    # Strict identity and version checks.
    manifest = json.loads((tmp_path / "artifact" / "frontend.json").read_text())
    manifest["state_dict_sha256"] = "e" * 64
    (tmp_path / "artifact" / "frontend.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_student_artifact(tmp_path / "artifact", frontend_factory=factory)
    with pytest.raises(ValueError, match="projection"):
        assert_bridge_has_no_projection(SimpleNamespace(projection=nn.Linear(2, 2)))
    assert assert_bridge_has_no_projection(SimpleNamespace(projection=None))


def test_backend_freeze_guard():
    backend = nn.Linear(3, 3)
    backend.requires_grad_(False)
    report = assert_backend_frozen(backend)
    assert report["backend_trainable"] is False
    backend.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="not frozen"):
        assert_backend_frozen(backend)


def test_rgb3_control_consumes_only_teacher_preprocessed_values():
    _, student = fixture_student("rgb3_control")
    rgb = torch.linspace(-1.0, 1.0, 3 * 8 * 8).reshape(1, 1, 3, 8, 8)
    out = student(
        RawStemInput(
            values=rgb,
            input_mode="rgb3_control",
            camera_ids=("base",),
            units="teacher_rgb_minus1_1",
        )
    )
    assert out.tokens.values.shape == (1, 4, 6)
    assert out.input_mode == "rgb3_control"
    with pytest.raises(ValueError, match="preprocessed"):
        student(
            RawStemInput(
                values=rgb * 2.0,
                input_mode="rgb3_control",
                camera_ids=("base",),
                units="teacher_rgb_minus1_1",
            )
        )
    with pytest.raises(ValueError, match="teacher_rgb_minus1_1"):
        RawStemInput(
            values=rgb,
            input_mode="rgb3_control",
            camera_ids=("base",),
            units="normalized_0_1",
        )


def test_initialization_audit_hashes_every_inherited_and_stem_tensor():
    _, student = fixture_student()
    audit = student.initialization_audit
    assert audit.evidence_level == "fixture_contract"
    assert audit.source_tensors
    assert audit.student_tensors
    assert audit.stem_tensors
    assert all(len(item.sha256) == 64 for item in audit.source_tensors)
    source = {item.name: item.sha256 for item in audit.source_tensors}
    copied = {item.name: item.sha256 for item in audit.student_tensors}
    for name, digest in source.items():
        assert copied[name] == digest
    assert all("patch_embedding" in name for name in audit.stem_tensors)


def test_temporal_teacher_requires_and_uses_explicit_history_adapter():
    policy = make_policy()
    history = HistoryContract.from_frames((0, 1), (0.0, 0.1), stride=1)

    def sequence_embed(images):
        assert images.shape == (1, 2, 3, 8, 8)
        return torch.ones(1, 2, 4, 6)

    teacher = SiglipTeacher.from_fixture(
        policy,
        fixture_id="tiny-temporal",
        history_contract=history,
        sequence_embed=sequence_embed,
    )
    output = teacher(torch.zeros(1, 2, 3, 8, 8))
    assert output.temporal is True
    assert output.projected.shape == (1, 2, 4, 6)
    assert teacher.temporal_supported is True
    with pytest.raises(ValueError, match="stride"):
        HistoryContract.from_frames((0, 1), (0.0, 0.1), stride=2)


def test_mixed_dtype_preserves_fp32_positions_and_bf16_encoder_input():
    _, student = fixture_student()
    vision = student.vision_tower.vision_model
    recorder = {}

    class BF16Encoder(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.q_proj = nn.Linear(width, width, bias=False).to(torch.bfloat16)
            self.layers = [SimpleNamespace(self_attn=SimpleNamespace(q_proj=self.q_proj))]

        def forward(self, inputs_embeds, **kwargs):
            recorder["dtype"] = inputs_embeds.dtype
            recorder["shape"] = tuple(inputs_embeds.shape)
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    vision.encoder = BF16Encoder(vision.encoder.layers[0].fc1.in_features)
    vision.post_layernorm = nn.Identity()
    student.projector = student.projector.to(torch.bfloat16)
    out = student(raw_input(value=0.25))
    assert recorder["dtype"] == torch.bfloat16
    assert recorder["shape"] == (1, 4, 8)
    assert student.position_embedding.weight.dtype == torch.float32
    assert student.stem.patch_embedding.weight.dtype == torch.float32
    assert out.projected_tokens.dtype == torch.bfloat16


def test_history_contract_allows_declared_initial_padding_and_later_windows():
    history = HistoryContract(
        frame_ids=(2, 2, 3),
        timestamps=(0.0, 0.0, 0.1),
        stride=1,
        padding_mask=(False, True, False),
        new_measurement=(True, False, True),
        burst_frames=3,
        rnn_frames=6,
        max_history=3,
    )
    assert history.padding_mask == (False, True, False)
    assert history.temporal_frames == 3
    later = HistoryContract.from_frames(
        (4, 5), (0.2, 0.3), stride=1, camera_order=(), burst_frames=3, rnn_frames=6
    )
    assert history.compatible_with(later) is False
    same_shape = HistoryContract.from_frames(
        (4, 5, 6),
        (0.2, 0.3, 0.4),
        stride=1,
        burst_frames=3,
        rnn_frames=6,
        max_history=3,
    )
    assert history.compatible_with(same_shape) is True
    with pytest.raises(ValueError, match="padding"):
        HistoryContract(
            frame_ids=(1, 1),
            timestamps=(0.0, 0.0),
            padding_mask=(False, False),
            new_measurement=(True, True),
        )


def test_default_artifact_loader_branch_with_allowlisted_factory(tmp_path, monkeypatch):
    import owac.baselines.raw_stem_siglip as module

    _, student = fixture_student()
    expected = student(raw_input(value=0.25))
    metadata = export_student_artifact(
        student,
        tmp_path / "artifact-default",
        backend_identity={"sha256": "b" * 64},
        tokenizer_identity={"sha256": "c" * 64},
        normalizer_identity={"sha256": "d" * 64},
        checkpoint_step=500,
    )
    calls = {"factory": 0}

    def fixture_factory(artifact_metadata):
        calls["factory"] += 1
        policy = make_policy(seed=777)
        return (
            policy.vlm.model.model.vision_tower,
            policy.vlm.model.model.multi_modal_projector,
        )

    monkeypatch.setattr(module, "build_artifact_frontend_from_metadata", fixture_factory)
    deployment = load_student_artifact(tmp_path / "artifact-default")
    assert calls["factory"] == 1
    assert deployment.metadata["step"] == 500
    actual = deployment(raw_input(value=0.25))
    torch.testing.assert_close(actual.projected_tokens, expected.projected_tokens)
    assert metadata["step"] == 500


def test_initial_left_padding_preserves_one_real_acquisition():
    history = HistoryContract.from_frames(
        (0, 0, 0, 0, 0, 0),
        (0.0,) * 6,
        burst_frames=6,
        rnn_frames=6,
        padding_mask=(True, True, True, True, True, False),
        new_measurement=(False, False, False, False, False, True),
    )
    assert sum(history.new_measurement) == 1
    assert history.timestamps == (0.0,) * 6
    with pytest.raises(ValueError, match="declared twice"):
        HistoryContract.from_frames((0, 0, 0), (0.0,) * 3, padding_mask=(False, True, False))
