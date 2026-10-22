"""Fixture tests for temporal teacher/data/cache/training/pure-runtime integration."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from owac.baselines.raw_stem_siglip import RawStemInput, RawStemSiglipStudent
from owac.baselines.raw_temporal_siglip import (
    TemporalPacket,
    TemporalRawStemStudent,
    causal_frame_ids,
    causal_padding_mask,
)
from owac.core.encoder_contract_n02 import VisualTokens
from owac.evaluation.raw_temporal_distillation import (
    TemporalPureRuntime,
    shuffle_visual_tokens,
    zero_visual_tokens,
)
from owac.raw.temporal_distillation import (
    ACTIVE_IMAGE_KEYS,
    CAMERA_ALIASES,
    CAMERA_ORDER,
    INACTIVE_IMAGE_KEY,
    TemporalAnchorSample,
    TemporalAssetSet,
    TemporalCache,
    TemporalTargetRow,
    native_camera_ids,
    physical_positions_224_to_256,
    resize_raw_256_to_224,
)
from owac.runtime.raw_temporal_distillation import (
    _asset_set,
    _receipt_ready,
    _research_config_digest,
    _stage_gate_ready,
    build_parser,
    main,
)
from owac.training.raw_distillation import token_distillation_loss
from owac.training.raw_temporal_distillation import (
    TemporalDistillationTrainer,
    TemporalTrainingAuthority,
    TemporalTrainingConfig,
    assert_split_allowed,
)

_FIXTURE_PATH = Path(__file__).parent / "test_raw_temporal_siglip.py"
_SPEC = importlib.util.spec_from_file_location("temporal_fixture_helpers", _FIXTURE_PATH)
_FIXTURE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_FIXTURE)


def _sha(char: str) -> str:
    return char * 64


def make_row(*, endpoint=5, prefix_sha256=None, source="source-a") -> TemporalTargetRow:
    frame_ids = causal_frame_ids(endpoint)
    zeros = torch.zeros(2, 1, 128)
    return TemporalTargetRow(
        source_id=source,
        endpoint=endpoint,
        frame_ids=frame_ids,
        camera_order=CAMERA_ORDER,
        raw_frame_indices=torch.tensor([frame_ids for _ in range(2)], dtype=torch.int64),
        raw_burst_sha256=_sha("a"),
        prefix_sha256=_sha("b") if prefix_sha256 is None else prefix_sha256,
        teacher_tokens=torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6),
        teacher_token_mask=torch.tensor([True, True]),
        teacher_rgb=torch.zeros(2, 3, 224, 224),
        inactive_image_keys=(INACTIVE_IMAGE_KEY,),
        lang_tokens=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        lang_masks=torch.tensor([[True, True, True]]),
        state32=torch.zeros(1, 32),
        state8=torch.zeros(8),
        state_raw=torch.zeros(8),
        actions32=torch.zeros(10, 32),
        actions7=torch.zeros(10, 7),
        future_mask=torch.ones(10, dtype=torch.bool),
        noise=torch.zeros(1, 10, 32),
        flow_time=torch.tensor([0.5]),
        state_before={"agentview": zeros[0], "wrist": zeros[1]},
        state_after={"agentview": zeros[0], "wrist": zeros[1]},
        theta_before={"cameras": {"agentview": None, "wrist": None}},
        theta_after={"cameras": {"agentview": None, "wrist": None}},
        teacher_artifact_sha256=_sha("c"),
        backend_sha256=_sha("d"),
        tokenizer_sha256=_sha("e"),
        stats_sha256=_sha("f"),
        source_raw_sha256=_sha("1"),
        code_revision=_sha("2"),
        preprocess_id="fixture",
        metadata={},
    )


def test_native_singleton_state_axis_preserves_bridge_semantics():
    from owac.backends.rawvla_features_n02 import build_stub_bridge
    from owac.raw.temporal_distillation import validate_temporal_row

    native = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32)
    row = replace(make_row(), state32=native)
    validate_temporal_row(row)
    bridge = build_stub_bridge()["bridge"]
    torch.testing.assert_close(
        bridge._prepare_state(native), bridge._prepare_state(native.reshape(1, 32)), rtol=0, atol=0
    )
    with pytest.raises(RuntimeError, match="state shape"):
        validate_temporal_row(replace(row, state32=native.expand(1, 2, 32)))


def test_temporal_metrics_keep_physical7_and_normalized_padding_separate():
    from owac.evaluation.raw_distillation import evaluate_windows
    from owac.evaluation.raw_temporal_distillation import _aggregate_physical_and_padding

    normalized = torch.ones(1, 10, 32)
    normalized[..., 7:] = 2.0
    mask = torch.tensor([[True] * 5 + [False] * 5])
    records = [
        evaluate_windows(
            normalized,
            torch.zeros_like(normalized),
            mask,
            split="validation",
            step=0,
            normalized=True,
        ),
        evaluate_windows(
            torch.full((1, 10, 7), 3.0),
            torch.zeros(1, 10, 7),
            mask,
            split="validation",
            step=0,
            normalized=False,
        ),
    ]
    result = _aggregate_physical_and_padding(records, split="validation", step=0)
    assert result.first5_mae7 == 3.0
    assert result.padding_mae == 2.0
    assert result.valid_first5 == 5


def test_geometry_aliases_resize_and_position_backmap():
    assert native_camera_ids() == tuple(CAMERA_ALIASES[name] for name in CAMERA_ORDER)
    assert ACTIVE_IMAGE_KEYS == ("base_0_rgb", "left_wrist_0_rgb")
    raw = torch.full((6, 256, 256, 3), 128, dtype=torch.uint8)
    resized = resize_raw_256_to_224(raw)
    assert resized.shape == (6, 3, 224, 224)
    assert resized.dtype == torch.float32
    torch.testing.assert_close(resized, torch.full_like(resized, 128 / 255))
    positions = torch.tensor([[[112.0, 112.0, 14.0], [0.0, 0.0, 14.0]]])
    back = physical_positions_224_to_256(positions)
    torch.testing.assert_close(back, torch.tensor([[[128.0, 128.0, 16.0], [0.0, 0.0, 16.0]]]))


class FakeStudent(nn.Module):
    model_identity = _sha("3")

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)


def test_training_authorization_and_source_test_selection_blocked():
    student = FakeStudent()
    trainer = TemporalDistillationTrainer(
        student=student,
        bridge=None,
        config=TemporalTrainingConfig(),
        authority=None,
    )
    allowed, reason = trainer._authorized_for_update(include_action=False)
    assert not allowed and "authority" in reason
    trainer.authority = TemporalTrainingAuthority(authorized=True, max_updates=10, max_anchors=10)
    allowed, reason = trainer._authorized_for_update(include_action=False)
    assert not allowed and "optimizer_updates" in reason
    trainer.config = replace(TemporalTrainingConfig(), authorization={"optimizer_updates": True})
    allowed, reason = trainer._authorized_for_update(include_action=False)
    assert not allowed and "teacher_ready" in reason
    with pytest.raises(RuntimeError, match="test split"):
        assert_split_allowed("test", purpose="select")
    with pytest.raises(RuntimeError, match="train split"):
        assert_split_allowed("validation", purpose="train")
    assert_split_allowed("validation", purpose="evaluate")


def test_checkpoint_resume_fake_only(tmp_path):
    student = FakeStudent()
    trainer = TemporalDistillationTrainer(
        student=student,
        bridge=None,
        config=TemporalTrainingConfig(),
    )
    trainer.cumulative_updates = 7
    trainer.anchor_presentations = 11
    path = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    other_student = FakeStudent()
    with torch.no_grad():
        other_student.linear.weight.fill_(9.0)
    other = TemporalDistillationTrainer(
        student=other_student,
        bridge=None,
        config=TemporalTrainingConfig(),
    )
    payload = other.load_checkpoint(path)
    assert payload["cumulative_updates"] == 7
    assert other.cumulative_updates == 7
    assert other.anchor_presentations == 11
    torch.testing.assert_close(other_student.linear.weight, student.linear.weight)


def test_token_loss_has_gradient_and_pure_runtime_isolation():
    prediction = torch.nn.Parameter(torch.randn(1, 4, 6))
    target = torch.randn(1, 4, 6)
    loss = token_distillation_loss(prediction, target, torch.ones(1, 4, dtype=torch.bool))
    loss.total.backward()
    assert prediction.grad is not None
    assert prediction.grad.abs().sum() > 0

    class FakeBridge(nn.Module):
        projection = None

        def __init__(self):
            super().__init__()
            self.base = nn.Linear(1, 1)
            self.token_dim = 6
            self.tokens_per_camera = 4
            self.image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    student = FakeStudent()
    student.bridge_projection = None
    runtime = TemporalPureRuntime(student=student, bridge=FakeBridge(), stats=None)
    assert runtime.dependency_audit()["teacher"] is False
    assert runtime.dependency_audit()["cache"] is False
    tokens = VisualTokens(
        values=torch.arange(1 * 8 * 6, dtype=torch.float32).reshape(1, 8, 6),
        mask=torch.ones(1, 8, dtype=torch.bool),
        camera_ids=("base_0_rgb", "left_wrist_0_rgb"),
        positions=torch.zeros(1, 8, 3),
    )
    zeroed = zero_visual_tokens(tokens)
    assert torch.count_nonzero(zeroed.values) == 0
    shuffled = shuffle_visual_tokens(tokens, (1, 0))
    torch.testing.assert_close(shuffled.values[:, :4], tokens.values[:, 4:])
    torch.testing.assert_close(shuffled.values[:, 4:], tokens.values[:, :4])


def test_align_config_and_cli_surface_are_concrete():
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/experiments/raw_stem_siglip_align_v1.json"
    )
    config = json.loads(config_path.read_text())
    assets = TemporalAssetSet.from_config(config, repo_root=config_path.parents[2])
    assert assets.registry_path.name == "teacher-registry.json"
    assert assets.config_paths and assets.config_paths[-1].name == "pi05.yaml"
    assert config["authorization"]["optimizer_updates"] is False
    parser = build_parser()
    args = parser.parse_args(["assets", "--config", str(config_path)])
    assert args.command == "assets"
    args = parser.parse_args(["preflight", "--config", str(config_path), "--anchor-endpoint", "25"])
    assert args.func.__name__ == "cmd_preflight"
    args = parser.parse_args(["train", "--config", str(config_path), "--max-updates", "0"])
    assert args.func.__name__ == "cmd_train"


def test_pure_evaluation_import_does_not_import_teacher_module():
    code = (
        "import sys; "
        "import owac.evaluation.raw_temporal_distillation; "
        "assert 'owac.evaluation.rawvla_teacher' not in sys.modules; "
        "assert 'owac.training.rawvla_teacher' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_cli_train_blocks_on_authority_before_model_build(capsys):
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/experiments/raw_stem_siglip_align_v1.json"
    )
    code = main(["train", "--config", str(config_path)])
    captured = capsys.readouterr()
    assert code == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "unavailable"
    assert "authority" in payload["reason"]
    assert payload["optimizer_updates"] == 0


def test_trainer_token_gradient_uses_real_temporal_unroll():
    base = _FIXTURE.fixture_student()
    student = TemporalRawStemStudent(base)
    row = replace(
        make_row(endpoint=25),
        frame_ids=causal_frame_ids(25),
        raw_frame_indices=torch.tensor([causal_frame_ids(25) for _ in range(2)], dtype=torch.int64),
    )
    packets = tuple(
        _FIXTURE.make_packet(
            endpoint, values=[0.1] * 6, cameras=("base", "wrist"), reset=endpoint == 0
        )
        for endpoint in (0, 5, 10, 15, 20, 25)
    )
    sample = TemporalAnchorSample(
        source_id="source-a",
        endpoint=25,
        target=row,
        unroll_rows=(row,) * 6,
        unroll_packets=packets,
        prefix_packets=(),
        charged_context_frames=36,
        charged_prefix_endpoints=0,
    )
    trainer = TemporalDistillationTrainer(
        student=student, bridge=None, config=TemporalTrainingConfig()
    )
    result = trainer.zero_update_gradient_diagnostic(sample, include_action=False)
    assert result["optimizer_updates"] == 0
    assert result["valid_tokens"] == 8
    assert student.gate_logit.grad is not None
    assert student.gate_logit.grad.abs().sum() > 0
    assert student.stem.conv64.weight.grad is not None
    assert student.stem.conv64.weight.grad.abs().sum() > 0


def test_trainer_full_prefix_burnin_precedes_six_endpoint_unroll():
    base = _FIXTURE.fixture_student()
    student = TemporalRawStemStudent(base)
    row = replace(
        make_row(endpoint=30),
        frame_ids=causal_frame_ids(30),
        raw_frame_indices=torch.tensor([causal_frame_ids(30) for _ in range(2)], dtype=torch.int64),
    )
    prefix = _FIXTURE.make_packet(0, values=[0.1] * 6, cameras=("base", "wrist"), reset=True)
    unroll = tuple(
        _FIXTURE.make_packet(endpoint, values=[0.1] * 6, cameras=("base", "wrist"), reset=False)
        for endpoint in (5, 10, 15, 20, 25, 30)
    )
    sample = TemporalAnchorSample(
        source_id="source-a",
        endpoint=30,
        target=row,
        unroll_rows=(row,) * 6,
        unroll_packets=unroll,
        prefix_packets=(prefix,),
        charged_context_frames=36,
        charged_prefix_endpoints=1,
    )
    trainer = TemporalDistillationTrainer(
        student=student, bridge=None, config=TemporalTrainingConfig()
    )
    result = trainer.zero_update_gradient_diagnostic(sample, include_action=False)
    assert result["valid_tokens"] == 8
    assert student.gate_logit.grad is not None


def test_actual_temporal_224_output_uses_256_positions_and_scale16():
    config = _FIXTURE.TinyConfig(image_size=224, patch_size=14, hidden_size=8, layers=1)
    policy = _FIXTURE.TinyPolicy(seed=0)
    policy.vlm = _FIXTURE.TinyVLM(config)
    base = RawStemSiglipStudent.from_fixture(
        policy, input_mode="rawvla_rgb3", fixture_id="tiny-224", stem_seed=0
    )
    student = TemporalRawStemStudent(base)
    frame = RawStemInput(
        values=torch.full((1, 1, 3, 224, 224), 0.5),
        input_mode="rawvla_rgb3",
        camera_ids=("base_0_rgb",),
        units="normalized_0_1",
    )
    frame_ids = causal_frame_ids(0)
    packet = TemporalPacket(
        frames=(frame,) * 6,
        source_demo_id="source-a",
        endpoint=0,
        frame_ids=frame_ids,
        padding_mask=causal_padding_mask(0),
        reset=True,
        position_source_size=256,
        position_target_size=224,
        position_backmap=True,
        preprocess_id="fixture_256_fullview",
    )
    output, _ = student(packet, None)
    assert output.tokens.scale == 16.0
    assert output.tokens.positions.shape == (1, 256, 3)
    torch.testing.assert_close(output.tokens.positions[0, 0, :2], torch.tensor([8.0, 8.0]))
    torch.testing.assert_close(output.tokens.positions[0, 0, 2], torch.tensor(16.0))
    assert output.measurement_metadata["physical_position_contract"]["source_image_size"] == 256


def test_actual_40char_git_revision_and_slots_device_override():
    row = replace(make_row(), code_revision="b" * 40)
    from owac.raw.temporal_distillation import validate_temporal_row

    validate_temporal_row(row)
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/experiments/raw_stem_siglip_align_v1.json"
    )
    config = json.loads(config_path.read_text())
    assets = _asset_set(config, device="cpu")
    assert assets.device == "cpu"
    assert assets.registry_path.name == "teacher-registry.json"
    assert assets.config_paths[-1].name == "pi05.yaml"


def test_pure_cli_fresh_import_has_no_teacher_or_cache_module():
    code = (
        "import sys; "
        "import owac.runtime.raw_temporal_distillation; "
        "assert 'owac.evaluation.rawvla_teacher' not in sys.modules; "
        "assert 'owac.raw.temporal_distillation' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_first_endpoint_state_before_is_explicit_none():
    from owac.raw.temporal_distillation import validate_temporal_row

    row = replace(
        make_row(endpoint=0),
        frame_ids=causal_frame_ids(0),
        raw_frame_indices=torch.tensor([causal_frame_ids(0) for _ in range(2)], dtype=torch.int64),
        state_before={"agentview": None, "wrist": None},
    )
    validate_temporal_row(row)
    payload = row.to_payload()
    assert payload["state_before"] == {"agentview": None, "wrist": None}


def test_actual_raw_dataset_packet_path_outputs_256_positions():
    from owac.raw.distillation_pairs import NormalisationStats
    from owac.raw.teacher_data import TeacherData, load_source_manifest
    from owac.raw.temporal_packets import RawTemporalDataset

    config_path = (
        Path(__file__).resolve().parents[1] / "configs/experiments/raw_stem_siglip_align_v1.json"
    )
    config = json.loads(config_path.read_text())
    assets = config["assets"]
    manifest = load_source_manifest(
        Path(assets["source_manifest"]),
        repo_root=Path(__file__).resolve().parents[1],
        strict=True,
    )
    data = TeacherData(manifest, repo_root=Path(__file__).resolve().parents[1])
    tiny_config = _FIXTURE.TinyConfig(image_size=224, patch_size=14, hidden_size=8, layers=1)
    policy = _FIXTURE.TinyPolicy(seed=0)
    policy.vlm = _FIXTURE.TinyVLM(tiny_config)
    base = RawStemSiglipStudent.from_fixture(
        policy, input_mode="rawvla_rgb3", fixture_id="tiny-raw-224", stem_seed=0
    )
    student = TemporalRawStemStudent(base)
    stats = NormalisationStats.from_json(Path(assets["statistics"]), mode="quantile")
    dataset = RawTemporalDataset(data, student, stats=stats, arm="C")
    source = data.manifest.sources_for_split("train")[0]
    packet = dataset.packet_for_endpoint(source.source_demo_id, 0)
    assert packet.camera_order == ("base_0_rgb", "left_wrist_0_rgb")
    assert packet.frames[0].values.shape == (1, 2, 3, 224, 224)
    output, _ = student(packet, None)
    assert output.tokens.scale == 16.0
    assert output.tokens.positions.shape == (1, 512, 3)
    torch.testing.assert_close(output.tokens.positions[0, 0, :2], torch.tensor([8.0, 8.0]))


def test_resize_explicit_dtype_contract_and_real_source_scale():
    import numpy as np

    from owac.raw.temporal_packets import resize_raw_256_to_224

    for value in (0, 1, 128, 255):
        raw = np.full((2, 256, 256, 3), value, dtype=np.uint8)
        out = resize_raw_256_to_224(raw)
        expected = torch.full_like(out, value / 255.0)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    raw_torch = torch.full((2, 256, 256, 3), 1, dtype=torch.uint8)
    torch.testing.assert_close(
        resize_raw_256_to_224(raw_torch),
        torch.full((2, 3, 224, 224), 1 / 255.0),
        rtol=0,
        atol=0,
    )
    normalized = torch.full((2, 256, 256, 3), 0.25, dtype=torch.float32)
    torch.testing.assert_close(
        resize_raw_256_to_224(normalized),
        torch.full((2, 3, 224, 224), 0.25),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        resize_raw_256_to_224(torch.full((2, 256, 256, 3), 255.0))
    with pytest.raises(ValueError, match="uint16|ambiguous"):
        resize_raw_256_to_224(np.zeros((2, 256, 256, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="square"):
        resize_raw_256_to_224(np.zeros((2, 128, 256, 3), dtype=np.uint8))

    config_path = (
        Path(__file__).resolve().parents[1] / "configs/experiments/raw_stem_siglip_align_v1.json"
    )
    config = json.loads(config_path.read_text())
    manifest = json.loads(Path(config["assets"]["source_manifest"]).read_text())
    source = next(item for item in manifest["sources"] if item["split"] == "train")
    with np.load(Path(__file__).resolve().parents[1] / source["raw_path"]) as archive:
        real = np.asarray(archive["agentview_raw_uint8"][:2])
    actual = resize_raw_256_to_224(real)
    reference = torch.nn.functional.interpolate(
        torch.from_numpy(real).float().permute(0, 3, 1, 2) / 255.0,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert 0.0 <= float(actual.mean()) < 0.5
    assert not torch.allclose(actual, torch.full_like(actual, float(real.min()) / 255.0))


def test_unknown_native_224_and_bayer_do_not_get_implicit_backmap():
    config = _FIXTURE.TinyConfig(image_size=224, patch_size=14, hidden_size=8, layers=1)
    policy = _FIXTURE.TinyPolicy(seed=0)
    policy.vlm = _FIXTURE.TinyVLM(config)
    base = RawStemSiglipStudent.from_fixture(
        policy, input_mode="rawvla_rgb3", fixture_id="tiny-native-224", stem_seed=0
    )
    student = TemporalRawStemStudent(base)
    frame = RawStemInput(
        values=torch.full((1, 1, 3, 224, 224), 0.5),
        input_mode="rawvla_rgb3",
        camera_ids=("base_0_rgb",),
        units="normalized_0_1",
    )
    packet = TemporalPacket(
        frames=(frame,) * 6,
        source_demo_id="source-a",
        endpoint=0,
        frame_ids=causal_frame_ids(0),
        padding_mask=causal_padding_mask(0),
        reset=True,
        position_source_size=224,
        position_target_size=224,
        position_backmap=False,
    )
    output, _ = student(packet, None)
    assert output.tokens.scale == 14.0
    torch.testing.assert_close(output.tokens.positions[0, 0, :2], torch.tensor([7.0, 7.0]))
    cfa = RawStemInput(
        values=torch.full((1, 1, 4, 112, 112), 0.5),
        input_mode="bayer4_packed",
        camera_ids=("base_0_rgb",),
        units="normalized_0_1",
        cfa_pattern="RGGB",
    )
    with pytest.raises(ValueError, match="Bayer|backmap"):
        TemporalPacket(
            frames=(cfa,) * 6,
            source_demo_id="source-a",
            endpoint=0,
            frame_ids=causal_frame_ids(0),
            padding_mask=causal_padding_mask(0),
            reset=True,
            position_source_size=256,
            position_target_size=224,
            position_backmap=True,
        )


def _fake_cache_bundle(source_id):
    import json

    from owac.raw.teacher_data import (
        TeacherData,
        load_pi05_normalization_stats,
        load_source_manifest,
    )

    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/experiments/raw_stem_siglip_align_v1.json"
        ).read_text()
    )
    assets = config["assets"]
    manifest = load_source_manifest(
        Path(assets["source_manifest"]), repo_root=Path(__file__).resolve().parents[1], strict=True
    )
    data = TeacherData(manifest, repo_root=Path(__file__).resolve().parents[1])
    if not any(item.source_demo_id == source_id for item in manifest.sources):
        source_id = manifest.sources[0].source_demo_id
    return (
        type(
            "Bundle",
            (),
            {
                "data": data,
                "teacher": _CacheFakeTeacher(),
                "stats": load_pi05_normalization_stats(Path(assets["statistics"])),
                "evaluator": _CacheFakeEvaluator(),
                "identity": {
                    "artifact_sha256": _sha("a"),
                    "backend_sha256": _sha("b"),
                    "tokenizer_sha256": _sha("c"),
                    "stats_sha256": _sha("d"),
                    "source_revision": _sha("e"),
                },
            },
        )(),
        source_id,
    )


class _CacheFakeFrontendModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.01))
        self.calls = 0

    def forward(self, raw, *, state=None, theta_prev=None):
        from types import SimpleNamespace

        self.calls += 1
        batch = raw.shape[0]
        if state is None:
            state = torch.zeros(batch, 128, device=raw.device)
        delta = raw.mean(dim=(1, 2, 3, 4), keepdim=True).reshape(batch, 1)
        state_out = state + delta * torch.arange(128, device=raw.device).reshape(1, 128) * 1e-4
        theta = theta_prev if theta_prev is not None else torch.zeros((), device=raw.device)
        theta = theta + raw.mean() * self.bias
        rgb = torch.nn.functional.interpolate(
            raw[:, -1], size=(224, 224), mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        return SimpleNamespace(state=state_out, theta=theta, rgb=rgb)


class _CacheFakeFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _CacheFakeFrontendModel()

    def eval(self):
        super().eval()
        self.model.eval()
        return self


class _CacheFakeVLM(nn.Module):
    def embed_image(self, image):
        mean = image.mean(dim=(1, 2, 3), keepdim=True)
        return mean.reshape(1, 1, 1).expand(1, 4, 8).clone()


class _CacheFakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = _CacheFakeVLM()


class _CacheFakeEvaluator:
    def _policy_observation(self, camera_rgb, state8, language):
        from types import SimpleNamespace

        import numpy as np

        images = {
            CAMERA_ALIASES[key]: (camera_rgb[key].unsqueeze(0) * 2.0 - 1.0).clamp(-1.0, 1.0)
            for key in CAMERA_ORDER
        }
        masks = {key: torch.tensor([True]) for key in ACTIVE_IMAGE_KEYS}
        state = torch.zeros(1, 1, 32)
        state[0, 0, :8] = torch.from_numpy(np.asarray(state8, dtype=np.float32))
        return SimpleNamespace(
            images=images,
            image_masks=masks,
            tokenized_prompt=torch.tensor([[1, 2, 3]], dtype=torch.int64),
            tokenized_prompt_mask=torch.tensor([[True, True, True]]),
            state=state,
            language=language,
        )


class _CacheFakeTeacher:
    def __init__(self):
        self.frontend = _CacheFakeFrontend()
        self.policy = _CacheFakePolicy()


def _fake_cache_bundle(source_id):
    import json

    from owac.raw.teacher_data import (
        TeacherData,
        load_pi05_normalization_stats,
        load_source_manifest,
    )

    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/experiments/raw_stem_siglip_align_v1.json"
        ).read_text()
    )
    assets = config["assets"]
    manifest = load_source_manifest(
        Path(assets["source_manifest"]), repo_root=Path(__file__).resolve().parents[1], strict=True
    )
    data = TeacherData(manifest, repo_root=Path(__file__).resolve().parents[1])
    if not any(item.source_demo_id == source_id for item in manifest.sources):
        source_id = manifest.sources[0].source_demo_id
    return (
        type(
            "Bundle",
            (),
            {
                "data": data,
                "teacher": _CacheFakeTeacher(),
                "stats": load_pi05_normalization_stats(Path(assets["statistics"])),
                "evaluator": _CacheFakeEvaluator(),
                "identity": {
                    "artifact_sha256": _sha("a"),
                    "backend_sha256": _sha("b"),
                    "tokenizer_sha256": _sha("c"),
                    "stats_sha256": _sha("d"),
                    "source_revision": _sha("e"),
                },
            },
        )(),
        source_id,
    )


def test_prepare_cache_hit_miss_carry_and_mutation(tmp_path):
    import json

    import torch

    from owac.raw.teacher_data import load_source_manifest
    from owac.raw.temporal_distillation import (
        prepare_temporal_cache,
    )

    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/experiments/raw_stem_siglip_align_v1.json"
        ).read_text()
    )
    manifest = load_source_manifest(
        Path(config["assets"]["source_manifest"]),
        repo_root=Path(__file__).resolve().parents[1],
        strict=True,
    )
    source = manifest.sources_for_split("train")[0]
    source_id = source.source_demo_id
    bundle, source_id = _fake_cache_bundle(source_id)
    identity = {"fixture": "prepare-cache-v2", "source": source_id}

    cache = TemporalCache(tmp_path / "cache", identity)
    first = prepare_temporal_cache(bundle, cache, source_ids=[source_id], max_endpoint=5)
    calls_after_miss = bundle.teacher.frontend.model.calls
    assert calls_after_miss == 4
    assert first["rows"] == 2
    second = prepare_temporal_cache(bundle, cache, source_ids=[source_id], max_endpoint=5)
    assert second["rows"] == 2
    assert bundle.teacher.frontend.model.calls == calls_after_miss
    for endpoint in (0, 5):
        assert cache.read(source_id, endpoint).source_id == source_id
    assert bundle.teacher.frontend.model.calls == calls_after_miss

    partial_bundle, _ = _fake_cache_bundle(source_id)
    partial_cache = TemporalCache(tmp_path / "partial", identity)
    prepare_temporal_cache(partial_bundle, partial_cache, source_ids=[source_id], max_endpoint=0)
    assert partial_bundle.teacher.frontend.model.calls == 2
    prepare_temporal_cache(partial_bundle, partial_cache, source_ids=[source_id], max_endpoint=5)
    assert partial_bundle.teacher.frontend.model.calls == 4
    for endpoint in (0, 5):
        left = cache.read(source_id, endpoint)
        right = partial_cache.read(source_id, endpoint)
        assert left.prefix_sha256 == right.prefix_sha256
        torch.testing.assert_close(left.teacher_tokens, right.teacher_tokens)
        torch.testing.assert_close(left.teacher_rgb, right.teacher_rgb)

    # Reset source 0 must have explicit None before-state/theta and restore real after-state.
    reset_source = manifest.sources_for_split("validation")[0].source_demo_id
    reset_bundle, reset_source = _fake_cache_bundle(reset_source)
    reset_cache = TemporalCache(tmp_path / "reset", {"fixture": "reset", "source": reset_source})
    prepare_temporal_cache(reset_bundle, reset_cache, source_ids=[reset_source], max_endpoint=0)
    reset_row = reset_cache.read(reset_source, 0)
    assert reset_row.state_before == {"agentview": None, "wrist": None}
    assert reset_row.theta_before["cameras"] == {"agentview": None, "wrist": None}
    assert all(value is not None for value in reset_row.state_after.values())
    assert all(value is not None for value in reset_row.theta_after["cameras"].values())
    fresh_reset_bundle, _ = _fake_cache_bundle(reset_source)
    fresh_reset_cache = TemporalCache(
        tmp_path / "reset-fresh", {"fixture": "reset", "source": reset_source}
    )
    prepare_temporal_cache(
        fresh_reset_bundle, fresh_reset_cache, source_ids=[reset_source], max_endpoint=0
    )
    assert fresh_reset_cache.read(reset_source, 0).source_id == reset_source


def test_research_config_digest_and_receipt_content_binding(tmp_path):
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/experiments/raw_stem_siglip_align_v1.json"
        ).read_text()
    )
    digest = _research_config_digest(config)
    changed = json.loads(json.dumps(config))
    changed["authorization"]["optimizer_updates"] = True
    changed["assets"]["output_root"] = "/tmp/other-output"
    assert _research_config_digest(changed) == digest
    changed["training"]["learning_rate"] *= 2
    assert _research_config_digest(changed) != digest

    payload = {
        "passed": True,
        "optimizer_updates": 0,
        "teacher_artifact_sha256": "a" * 64,
        "backend_sha256": "b" * 64,
        "sources_sha256": "c" * 64,
        "research_config_sha256": digest,
        "cache_identity_key": "d" * 64,
        "protocol": "fixture",
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload))
    kwargs = dict(
        artifact_sha256="a" * 64,
        backend_sha256="b" * 64,
        sources_sha256="c" * 64,
        research_config_sha256=digest,
        cache_identity_key="d" * 64,
        protocol="fixture",
    )
    assert _receipt_ready(str(path), **kwargs)
    assert not _receipt_ready(str(path), **{**kwargs, "cache_identity_key": "e" * 64})
    assert not _receipt_ready(str(path), **{**kwargs, "protocol": "wrong"})
    payload["passed"] = False
    path.write_text(json.dumps(payload))
    assert not _receipt_ready(str(path), **kwargs)


class _AuditBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.register_buffer("buffer", torch.zeros(2))


class _AuditStudent(nn.Module):
    model_identity = _sha("a")
    _scope = "stem_gate"

    def __init__(self):
        super().__init__()
        self.base = _AuditBase()
        self.p0 = nn.Parameter(torch.zeros(1))
        self.p1 = nn.Parameter(torch.zeros(1))
        self.p2 = nn.Parameter(torch.zeros(1))
        self.p3 = nn.Parameter(torch.zeros(1))
        self.p4 = nn.Parameter(torch.zeros(1))


def test_feature_denominator_count_avoids_vision_and_tracks_inactive_camera():
    base = _FIXTURE.fixture_student()
    student = TemporalRawStemStudent(base)
    trainer = TemporalDistillationTrainer(
        student=student, bridge=None, config=TemporalTrainingConfig()
    )
    row = replace(
        make_row(endpoint=25),
        teacher_token_mask=torch.tensor([True, False]),
    )
    final_mask = torch.zeros(1, 2, 8, 8, dtype=torch.bool)
    final_mask[0, 0, :4, :4] = True
    packets = tuple(
        _FIXTURE.make_packet(
            endpoint,
            values=[0.1] * 6,
            cameras=("base", "wrist"),
            present=((True, False),),
            valid_masks=(None, None, None, None, None, final_mask),
            reset=endpoint == 0,
        )
        for endpoint in (0, 5, 10, 15, 20, 25)
    )
    sample = TemporalAnchorSample(
        source_id="source-a",
        endpoint=25,
        target=row,
        unroll_rows=(row,) * 6,
        unroll_packets=packets,
        prefix_packets=(),
        charged_context_frames=36,
        charged_prefix_endpoints=0,
    )
    calls = 0
    original = student.base.vision_tower.vision_model.encoder.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    student.base.vision_tower.vision_model.encoder.forward = counted
    count = trainer._count_feature_tokens(sample)
    assert calls == 0
    result = trainer.forward_anchor(sample, include_action=False)
    assert calls == 1
    assert count == int(result["valid_tokens"])
    assert count == 1


def test_feature_denominator_count_differs_for_full_mask():
    base = _FIXTURE.fixture_student()
    student = TemporalRawStemStudent(base)
    trainer = TemporalDistillationTrainer(
        student=student, bridge=None, config=TemporalTrainingConfig()
    )
    row = make_row(endpoint=25)
    packets = tuple(
        _FIXTURE.make_packet(
            endpoint,
            values=[0.1] * 6,
            cameras=("base", "wrist"),
            reset=endpoint == 0,
        )
        for endpoint in (0, 5, 10, 15, 20, 25)
    )
    sample = TemporalAnchorSample(
        source_id="source-a",
        endpoint=25,
        target=row,
        unroll_rows=(row,) * 6,
        unroll_packets=packets,
        prefix_packets=(),
        charged_context_frames=36,
        charged_prefix_endpoints=0,
    )
    count = trainer._count_feature_tokens(sample)
    result = trainer.forward_anchor(sample, include_action=False)
    assert count == int(result["valid_tokens"])
    assert count > 1


def test_stage_gate_b_c_do_not_require_future_phase1_but_d_does():
    assert _stage_gate_ready(
        "B", parity_ready=True, resource_ready=True, phase1_ready=False, matched_ready=False
    )
    assert _stage_gate_ready(
        "C", parity_ready=True, resource_ready=True, phase1_ready=False, matched_ready=False
    )
    assert not _stage_gate_ready(
        "C", parity_ready=False, resource_ready=True, phase1_ready=False, matched_ready=False
    )
    assert not _stage_gate_ready(
        "D", parity_ready=True, resource_ready=True, phase1_ready=False, matched_ready=True
    )
    assert _stage_gate_ready(
        "D", parity_ready=True, resource_ready=True, phase1_ready=True, matched_ready=True
    )


class _VisualDiagStudent:
    def __init__(self):
        self.training = True
        self.calls = []

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def __call__(self, packet, state=None):
        from types import SimpleNamespace

        from owac.core.encoder_contract_n02 import VisualTokens

        self.calls.append((packet.source_demo_id, packet.endpoint, state))
        source_code = {"source-a": 0.1, "source-b": 0.2}.get(packet.source_demo_id, 0.3)
        values = torch.full((1, 4, 8), source_code + packet.endpoint * 0.001)
        positions = torch.tensor(
            [[[8.0, 8.0, 16.0], [24.0, 8.0, 16.0], [8.0, 24.0, 16.0], [24.0, 24.0, 16.0]]]
        )
        tokens = VisualTokens(
            values=values,
            mask=torch.ones(1, 4, dtype=torch.bool),
            camera_ids=("base_0_rgb", "left_wrist_0_rgb"),
            positions=positions,
            scale=16.0,
        )
        return SimpleNamespace(tokens=tokens), SimpleNamespace(
            source=packet.source_demo_id, endpoint=packet.endpoint
        )


class _VisualDiagRuntime:
    def __init__(self):
        from owac.evaluation.raw_temporal_distillation import TemporalInferenceResult

        self.student = _VisualDiagStudent()
        self.calls = []
        self._result_type = TemporalInferenceResult

    def infer_visual_tokens(self, tokens, *, state8, language, noise=None, flow_steps=10):
        self.calls.append(
            {
                "value": float(tokens.values.mean()),
                "state8": torch.as_tensor(state8).clone(),
                "language": language,
                "noise": torch.as_tensor(noise).clone(),
                "flow_steps": int(flow_steps),
                "mask": tokens.mask.clone() if tokens.mask is not None else None,
            }
        )
        actions = torch.zeros(1, 10, 32)
        value = float(tokens.values.mean())
        actions[..., :7] = value
        physical = actions[..., :7].clone()
        return self._result_type(
            normalized_actions=actions,
            physical_actions=physical,
            first5_physical=physical[:, :5],
            state=None,
            state8_raw=torch.as_tensor(state8).clone(),
            flow_steps=int(flow_steps),
            run_stats={},
        )


def _visual_diag_samples(source_ids=("source-a", "source-b")):
    from owac.raw.temporal_packets import RawTemporalSample

    out = []
    for source in source_ids:
        for endpoint in (0, 5):
            packet = _FIXTURE.make_packet(
                endpoint,
                values=[0.1] * 6,
                cameras=("base_0_rgb", "left_wrist_0_rgb"),
                source=source,
                reset=endpoint == 0,
            )
            out.append(
                RawTemporalSample(
                    source_id=source,
                    split="validation",
                    endpoint=endpoint,
                    packet=packet,
                    state8_raw=torch.tensor([endpoint / 100.0] * 8),
                    actions32=torch.zeros(10, 32),
                    actions7=torch.zeros(10, 7),
                    future_mask=torch.ones(10, dtype=torch.bool),
                    language="put the cream cheese in the bowl",
                    frame_ids=tuple(packet.frame_ids),
                )
            )
    return out


def test_cross_demo_visual_controls_use_target_state_language_and_noise():
    from owac.evaluation.raw_temporal_distillation import evaluate_temporal_visual_controls

    runtime = _VisualDiagRuntime()
    result = evaluate_temporal_visual_controls(
        runtime, _visual_diag_samples(), split="validation", flow_steps=10
    )
    assert result["supplied_total"] == 4
    assert result["matched_count"] == 4
    assert result["excluded_count"] == 0
    assert result["denominators"]["valid_actions"] == 40
    assert [item["donor_source_id"] for item in result["donor_mapping"]] == [
        "source-b",
        "source-b",
        "source-a",
        "source-a",
    ]
    assert all(item["donor_mask_used"] is False for item in result["matched"])
    assert all(item["mask_strategy"] == "target_mask_preserved" for item in result["matched"])
    assert runtime.calls[0]["value"] != runtime.calls[2]["value"]
    assert runtime.calls[1]["value"] == 0.0
    for index in range(0, 12, 3):
        assert torch.equal(runtime.calls[index]["noise"], runtime.calls[index + 1]["noise"])
        assert torch.equal(runtime.calls[index]["noise"], runtime.calls[index + 2]["noise"])
        assert torch.equal(runtime.calls[index]["state8"], runtime.calls[index + 2]["state8"])
        assert runtime.calls[index]["language"] == runtime.calls[index + 2]["language"]
        assert torch.equal(runtime.calls[index]["mask"], runtime.calls[index + 2]["mask"])
        assert torch.equal(
            runtime.calls[index + 1]["mask"], torch.zeros_like(runtime.calls[index + 1]["mask"])
        )
    assert result["provenance"]["teacher_or_cache_used"] is False
    assert result["provenance"]["target_position_intervention"] == "not_run"


def test_cross_demo_visual_controls_reject_invalid_source_contracts():
    from owac.evaluation.raw_temporal_distillation import evaluate_temporal_visual_controls

    runtime = _VisualDiagRuntime()
    with pytest.raises(RuntimeError, match="at least two sources"):
        evaluate_temporal_visual_controls(
            runtime, _visual_diag_samples(("source-a",)), split="validation"
        )
    samples = _visual_diag_samples()
    duplicate = list(samples) + [replace(samples[0])]
    with pytest.raises(RuntimeError, match="duplicate|order"):
        evaluate_temporal_visual_controls(runtime, duplicate, split="validation")
    gap_packet = _FIXTURE.make_packet(
        10,
        values=[0.1] * 6,
        cameras=("base_0_rgb", "left_wrist_0_rgb"),
        source="source-a",
        reset=False,
    )
    out_of_order = [
        samples[0],
        replace(samples[1], endpoint=10, packet=gap_packet, frame_ids=tuple(gap_packet.frame_ids)),
        samples[2],
    ]
    with pytest.raises(RuntimeError, match="contiguous|order|endpoint"):
        evaluate_temporal_visual_controls(runtime, out_of_order, split="validation")
    bad_split = [replace(samples[0], split="test")] + samples[1:]
    with pytest.raises(RuntimeError, match="split"):
        evaluate_temporal_visual_controls(runtime, bad_split, split="validation")
    bad_camera_packet = replace(
        samples[2].packet,
        frames=tuple(
            replace(
                frame,
                values=frame.values[:, :1],
                camera_ids=("base_0_rgb",),
                physical_origins=((0, 0),),
            )
            for frame in samples[2].packet.frames
        ),
    )
    bad_camera = samples[:2] + [replace(samples[2], packet=bad_camera_packet)] + samples[3:]
    with pytest.raises(RuntimeError, match="camera"):
        evaluate_temporal_visual_controls(runtime, bad_camera, split="validation")


def test_visual_controls_have_no_teacher_or_cache_import():
    code = (
        "import sys; "
        "import owac.evaluation.raw_temporal_distillation as m; "
        "assert hasattr(m, 'evaluate_temporal_visual_controls'); "
        "assert 'owac.evaluation.rawvla_teacher' not in sys.modules; "
        "assert 'owac.raw.temporal_distillation' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_cross_demo_visual_controls_excludes_missing_donor_endpoint():
    from owac.evaluation.raw_temporal_distillation import evaluate_temporal_visual_controls

    runtime = _VisualDiagRuntime()
    samples = _visual_diag_samples()
    # Source B supplies only endpoint 0; source A endpoint 5 therefore has no donor.
    samples = [samples[0], samples[1], samples[2]]
    result = evaluate_temporal_visual_controls(runtime, samples, split="validation")
    assert result["supplied_total"] == 3
    assert result["selected_total"] == 3
    assert result["matched_count"] == 2
    assert result["excluded"] == [
        {
            "source_id": "source-a",
            "endpoint": 5,
            "donor_source_id": "source-b",
            "reason": "donor_missing_endpoint",
        }
    ]
    assert result["denominators"]["valid_actions"] == 20
    assert result["donor_mapping"][1]["endpoint"] == 5
    assert result["donor_mapping"][1]["donor_source_id"] == "source-b"


def test_anchor_sampler_full_permutation_resume_no_tail_loss():
    from owac.training.raw_temporal_distillation import TemporalAnchorSampler

    candidates = [(f"source-{index // 10}", (index % 10) * 5) for index in range(37)]
    sampler = TemporalAnchorSampler(candidates, seed=101, batch_size=16)
    initial = tuple(sampler.state()["permutation"])
    first = sampler.next_batch()
    state = sampler.state()
    second = sampler.next_batch()
    third = sampler.next_batch()
    assert len(first) == len(second) == len(third) == 16
    seen = set(first) | set(second) | set(third)
    assert {candidates[index] for index in initial} <= seen
    resumed = TemporalAnchorSampler(candidates, seed=101, batch_size=16)
    resumed.load_state(state)
    assert resumed.next_batch() == second
    other = TemporalAnchorSampler(candidates, seed=101, batch_size=16)
    assert other.next_batch() == first
    bad = {**state, "candidates": list(state["candidates"])}
    bad["candidates"][0] = ("different-source", 0)
    with pytest.raises(RuntimeError, match="candidates"):
        other.load_state(bad)


def test_anchor_sampler_b_c_share_seed_order():
    from owac.training.raw_temporal_distillation import TemporalAnchorSampler

    candidates = [(f"source-{index}", 25) for index in range(23)]
    b = TemporalAnchorSampler(candidates, seed=101, batch_size=16)
    c = TemporalAnchorSampler(candidates, seed=101, batch_size=16)
    assert [b.next_batch() for _ in range(4)] == [c.next_batch() for _ in range(4)]


def test_selector_state_survives_fake_checkpoint(tmp_path):
    student = _AuditStudent()
    trainer = TemporalDistillationTrainer(
        student=student, bridge=None, config=TemporalTrainingConfig()
    )
    trainer.set_selector_state(
        best_value=0.25,
        best_step=7,
        best_checkpoint_path=str(tmp_path / "checkpoint_0007.pt"),
        best_checkpoint_sha256="a" * 64,
    )
    path = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    other = TemporalDistillationTrainer(
        student=_AuditStudent(), bridge=None, config=TemporalTrainingConfig()
    )
    other.load_checkpoint(path)
    assert other.selector_state["best_value"] == 0.25
    assert other.selector_state["best_step"] == 7


def test_cache_coverage_requires_all_train_sources_and_b_validation_rows():
    from types import SimpleNamespace

    from owac.runtime.raw_temporal_distillation import _cache_coverage

    def source(index, split):
        return SimpleNamespace(source_demo_id=f"{split}-{index}", num_frames=30, split=split)

    train = [source(index, "train") for index in range(40)]
    validation = [source(index, "validation") for index in range(5)]
    manifest = SimpleNamespace(
        sources_for_split=lambda split: train if split == "train" else validation
    )
    data = SimpleNamespace(manifest=manifest)

    class Cache:
        def __init__(self, skip):
            self.skip = skip

        def exists(self, source_id, endpoint):
            return (source_id, endpoint) not in self.skip

    ok = _cache_coverage(Cache(set()), data, arm="C")
    assert ok[0] == [(source.source_demo_id, 25) for source in train]
    assert ok[1] == []
    assert ok[2]["candidate_anchors"] == 40
    one_missing = _cache_coverage(Cache({("train-3", 10)}), data, arm="C")
    assert one_missing[1] == [("train-3", 10)]
    b_missing = _cache_coverage(Cache({("validation-4", 0)}), data, arm="B")
    assert ("validation-4", 0) in b_missing[1]
    c_missing = _cache_coverage(Cache({("validation-4", 0)}), data, arm="C")
    assert c_missing[1] == []


def test_authority_resource_fields_and_export_args_are_explicit(tmp_path, capsys):
    from owac.training.raw_temporal_distillation import TemporalTrainingAuthority

    authority = TemporalTrainingAuthority.from_mapping(
        {
            "authorized": True,
            "max_updates": 100,
            "max_anchors": 1600,
            "max_wall_seconds": 0,
            "max_output_bytes": 0,
        }
    )
    assert authority.max_wall_seconds == 0
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/experiments/raw_stem_siglip_align_v1.json"
        ).read_text()
    )
    temporary = tmp_path / "config.json"
    temporary.write_text(json.dumps(config))
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(
        json.dumps(
            {
                "authorized": True,
                "max_updates": 100,
                "max_anchors": 1600,
                "scope": "stem_gate",
            }
        )
    )
    config["authorization"]["optimizer_updates"] = True
    temporary.write_text(json.dumps(config))
    code = main(["train", "--config", str(temporary), "--authority", str(authority_path)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert "max_wall_seconds" in payload["reason"]

    parser = build_parser()
    args = parser.parse_args(
        [
            "export-student",
            "--config",
            "config.json",
            "--checkpoint",
            "checkpoint.pt",
            "--arm",
            "B",
            "--step",
            "0",
        ]
    )
    assert args.checkpoint == "checkpoint.pt"
    assert args.arm == "B"
    assert args.step == 0


def test_checkpoint_full_config_run_identity_and_external_selection(tmp_path):
    from owac.runtime.raw_temporal_distillation import _path_sha256, _write_selection_sidecar

    student = _AuditStudent()
    trainer = TemporalDistillationTrainer(
        student=student,
        bridge=None,
        config=TemporalTrainingConfig(activation_memory_mode="save_on_cpu"),
        run_identity={"scope": "stem_plus_gate", "research_config_sha256": "a" * 64},
    )
    trainer.set_selector_state(
        best_value=0.3,
        best_step=0,
        best_checkpoint_path=str(tmp_path / "checkpoint_0000.pt"),
        best_checkpoint_sha256=None,
        best_metrics={"first5_mae7": 0.3, "step": 0},
    )
    checkpoint = trainer.save_checkpoint(tmp_path / "checkpoint_0000.pt")
    sidecar = _write_selection_sidecar(
        tmp_path,
        trainer,
        best_value=0.3,
        best_step=0,
        best_path=str(checkpoint),
        best_metrics={"first5_mae7": 0.3, "step": 0},
    )
    assert sidecar["checkpoint_sha256"] == _path_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["training_config"]["activation_memory_mode"] == "save_on_cpu"
    assert payload["run_identity"]["scope"] == "stem_plus_gate"
    assert payload["selector"]["best_step"] == 0
    assert payload["selector"]["best_checkpoint_sha256"] is None
    other = TemporalDistillationTrainer(
        student=_AuditStudent(),
        bridge=None,
        config=TemporalTrainingConfig(activation_memory_mode="save_on_cpu"),
        run_identity={"scope": "stem_plus_gate", "research_config_sha256": "a" * 64},
    )
    other.load_checkpoint(checkpoint)
    assert other.selector_state["best_metrics"]["first5_mae7"] == 0.3
    mismatched = TemporalDistillationTrainer(
        student=_AuditStudent(),
        bridge=None,
        config=TemporalTrainingConfig(activation_memory_mode="save_on_cpu"),
        run_identity={"scope": "stem_plus_gate", "research_config_sha256": "b" * 64},
    )
    with pytest.raises(RuntimeError, match="run identity"):
        mismatched.load_checkpoint(checkpoint)


def test_validation_noise_is_stable_per_anchor_and_does_not_touch_global_rng():
    from owac.evaluation.raw_temporal_distillation import _validation_noise

    torch.manual_seed(77)
    before = torch.random.get_rng_state().clone()
    first = _validation_noise("source-a", 5, 101, torch.device("cpu"))
    second = _validation_noise("source-a", 5, 101, torch.device("cpu"))
    other = _validation_noise("source-b", 5, 101, torch.device("cpu"))
    after = torch.random.get_rng_state().clone()
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.equal(before, after)
