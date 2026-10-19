"""S1 测量网络的关键行为与数值参照测试。

重点不是形状，而是提示词列出的合同：单帧无历史、批量置换、CFA 相位坐标、NaN 清理、
无 GT 泄漏、位置协方差与独立数值参照一致、缓存可复现。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.frameworks.n04.contracts import (
    ANCHOR_CANDIDATES_MAX,
    DENSE_LATENT_CHANNELS,
    INPUT_SCALES,
    MeasurementFrame,
    build_measurement_id,
    build_sample_grid,
    input_mode_channels,
    sensor_channels,
)
from owac.frameworks.n04.measurement import (
    MemorylessCarry,
    build_measurement,
    extract_cfa_planes,
    position_with_covariance,
    preprocess_frames,
)
from owac.frameworks.n04.measurement_geometry import (
    PixelPoints,
    cfa_plane_names,
    cfa_plane_offsets,
    finite_difference_jacobian,
)

VENDOR_ROOT = Path(__file__).resolve().parents[3] / "src/owac/frameworks/n04/_vendor/convnext"
K_TEST = torch.tensor([[50.0, 0.0, 23.5], [0.0, 50.0, 31.5], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def model(s1_config):
    return build_measurement(s1_config)


def _frame(
    *,
    shape_hw: tuple[int, int] = (64, 48),
    mode: str = "rawvla_rgb3",
    seed: int = 0,
    measurement_id: str | None = None,
    provenance: dict | None = None,
    K: torch.Tensor | None | str = "default",
    raw: torch.Tensor | None = None,
    cfa_pattern: str | None = None,
    invalid_mask: torch.Tensor | None = None,
    scale_name: str | None = None,
) -> MeasurementFrame:
    generator = torch.Generator().manual_seed(seed)
    if raw is None:
        channels = sensor_channels(mode)
        raw = torch.rand(channels, *shape_hw, generator=generator)
        if mode == "bayer_cfa":
            raw = raw * 4095.0
    return MeasurementFrame(
        raw=raw,
        camera_id="cam",
        measurement_id=measurement_id or build_measurement_id("src/seq_0", 0, "cam", mode),
        input_mode=mode,
        scale=INPUT_SCALES[
            scale_name or ("libero_uint8" if mode == "rawvla_rgb3" else "rendered_factor_linear")
        ],
        size_hw=shape_hw,
        K=K_TEST if K == "default" else K,
        cfa_pattern=cfa_pattern,
        provenance=provenance or {},
        invalid_mask=invalid_mask,
    )


# --------------------------------------------------------------------------------------
# vendor 与配置
# --------------------------------------------------------------------------------------


def test_vendored_convnext_matches_provenance():
    provenance = json.loads((VENDOR_ROOT / "provenance.json").read_text())
    assert provenance["commit"] == "048efcea897d999aed302f2639b6270aedf8d4c8"
    for entry in provenance["files"]:
        path = VENDOR_ROOT / entry["local_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["local_sha256"], entry["local_path"]
    source = (VENDOR_ROOT / "convnext.py").read_text()
    assert "from timm" not in source
    assert "timm.models" not in source
    assert "@register_model" not in source
    assert "depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]" in source


def test_backbone_is_full_base_without_classification_head(model):
    stages = model.backbone.stages
    assert [len(stage) for stage in stages] == [3, 3, 27, 3]
    assert [stage[0].dwconv.weight.shape[0] for stage in stages] == [128, 256, 512, 1024]
    assert model.backbone.head is None and model.backbone.norm is None
    names = [name for name, _ in model.named_parameters()]
    assert not any(name.startswith(("backbone.head", "backbone.norm")) for name in names)
    assert sum(parameter.numel() for parameter in model.parameters()) == 88_942_210


def test_four_feature_levels_and_fusion_grid(model):
    out = model.measure([_frame()])
    # 48 列 pad 到 64 列后融合网格为 16x16，padding 区域由 support 判为无效
    assert [tuple(level.shape[1:]) for level in out.feature_pyramid] == [
        (128, 16, 16),
        (256, 8, 8),
        (512, 4, 4),
        (1024, 2, 2),
    ]
    assert out.latent_mean_map.shape == (1, DENSE_LATENT_CHANNELS, 16, 16)
    assert out.sample_grid.feature_size_hw == (16, 16)


def test_replicate_padding_preserves_measured_values_and_excludes_boundary(s1_config):
    from dataclasses import replace

    config = replace(
        s1_config, measurement=replace(s1_config.measurement, padding_mode="replicate")
    )
    frame = _frame(shape_hw=(48, 64))
    batch = preprocess_frames([frame], config)
    expected = frame.raw * float(frame.scale.factor)
    torch.testing.assert_close(batch.values[0, :, :48], expected, rtol=0, atol=0)
    torch.testing.assert_close(batch.values[0, :, 48:], expected[:, -1:].expand(-1, 16, -1))
    assert not bool(batch.support[..., 48:, :].any())
    assert not bool(batch.saturated[..., 48:, :].any())


def test_replicate_boundary_avoids_zero_padding_gradient_amplification(s1_config):
    from owac.frameworks.n04.measurement import _pad_to_multiple

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        model = build_measurement(s1_config)
        raw = torch.rand(1, 3, 48, 64)
        gradients = {}
        for mode in ("constant", "replicate"):
            values, _ = _pad_to_multiple(raw, 32, mode)
            latent = model.fuse(model.backbone_levels(values))
            # Only real pixels contribute, yet the zero boundary reaches them
            # through the coarser pyramid and bilinear fusion.
            objective = (latent[..., :12, :] - 1).square().mean()
            grad = torch.autograd.grad(objective, model.backbone.downsample_layers[0][0].bias)[0]
            gradients[mode] = float(grad.norm())
    assert gradients["constant"] > 1000 * gradients["replicate"]


# --------------------------------------------------------------------------------------
# 输入合同
# --------------------------------------------------------------------------------------


def test_frame_contract_rejects_wrong_channels_and_sizes():
    with pytest.raises(ValueError):
        _frame(mode="bayer_cfa", raw=torch.rand(3, 64, 48))
    with pytest.raises(ValueError):
        _frame(shape_hw=(64, 48), raw=torch.rand(3, 32, 48))
    with pytest.raises(ValueError):
        _frame(cfa_pattern="RGGB")  # rgb3 不允许携带 CFA 排列
    with pytest.raises(TypeError):
        _frame(raw=torch.zeros(3, 64, 48, dtype=torch.int32))
    assert input_mode_channels("bayer_cfa") == 4
    assert sensor_channels("bayer_cfa") == 1


def test_measurement_id_is_identity_only():
    first = build_measurement_id("libero:task:demo_1", 3, "agentview", "rawvla_rgb3")
    second = build_measurement_id("libero:task:demo_1", 3, "agentview", "rawvla_rgb3")
    other = build_measurement_id("libero:task:demo_1", 4, "agentview", "rawvla_rgb3")
    assert first == second and first != other
    assert first.endswith("/000003/agentview@rawvla_rgb3")
    with pytest.raises(ValueError):
        build_measurement_id("bad#id", 0, "cam", "rawvla_rgb3")


def test_sample_grid_affine_roundtrip_and_manual_padding():
    grid = build_sample_grid(
        input_mode="rawvla_rgb3", sensor_size_hw=(48, 64), pad_xy=(0.0, 0.0), pad_hw=(64, 64)
    )
    # 48 行 pad 到 64 行：第 4 个融合行块（12..15）覆盖 48..63，其中 48..63 中 48 起是 padding
    assert grid.feature_size_hw == (16, 16)
    centres = grid.feature_centres()
    # 融合格 (0,0) 覆盖原图 0..3，中心为 1.5（align_corners=False）
    assert torch.allclose(centres.xy[0, 0], torch.tensor([1.5, 1.5]))
    assert torch.allclose(centres.xy[1, 3], torch.tensor([13.5, 5.5]))
    feature_points = PixelPoints(xy=torch.tensor([[0.0, 0.0], [3.0, 5.0]]), unit="feature_px")
    sensor_points = grid.feature_to_sensor.apply(feature_points)
    roundtrip = grid.sensor_to_feature.apply(sensor_points)
    assert torch.allclose(roundtrip.xy, feature_points.xy, atol=1e-5)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_cfa_planes_match_manual_slices(pattern):
    raw = torch.arange(64 * 48, dtype=torch.float32).reshape(1, 64, 48)
    planes = extract_cfa_planes(raw, pattern)
    expected = torch.stack(
        [
            raw[0, 0::2, 0::2],
            raw[0, 0::2, 1::2],
            raw[0, 1::2, 0::2],
            raw[0, 1::2, 1::2],
        ]
    )
    assert torch.equal(planes, expected)
    assert cfa_plane_names(pattern) == tuple(name for name, _, _ in cfa_plane_offsets(pattern))
    offsets = cfa_plane_offsets(pattern)
    assert [(dy, dx) for _, dy, dx in offsets] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    grid = build_sample_grid(
        input_mode="bayer_cfa",
        sensor_size_hw=(64, 48),
        pad_xy=(0.0, 0.0),
        pad_hw=(32, 32),
        cfa_pattern=pattern,
    )
    assert grid.cfa_plane_names == cfa_plane_names(pattern)
    # 坐标帧约定：仿射表达在相位 0（δ=(0,0)）的子网格帧里，p_plane0 = p_sensor / 2
    sensor_points = PixelPoints(
        xy=torch.tensor([[0.0, 0.0], [7.0, 9.0], [63.0, 47.0]]), unit="original_px"
    )
    plane0 = grid.sensor_to_backbone_input.apply(sensor_points)
    assert torch.allclose(plane0.xy, sensor_points.xy / 2.0)
    # 每个相位的真实采样位置：传感器 (2i+dx, 2j+dy)（与表一致，不得共享一个中心）
    for _, dy, dx in offsets:
        plane_index = torch.tensor([[0.0, 0.0], [3.0, 5.0]])
        sensor_of_phase = torch.stack((2 * plane_index[:, 0] + dx, 2 * plane_index[:, 1] + dy), -1)
        assert torch.allclose(
            grid.sensor_to_backbone_input.apply(
                PixelPoints(xy=sensor_of_phase, unit="original_px")
            ).xy,
            plane_index + torch.tensor([dx, dy], dtype=torch.float32) / 2.0,
        )


def test_preprocess_pads_to_backbone_multiple_and_marks_support(s1_config):
    batch = preprocess_frames([_frame(shape_hw=(48, 64))], s1_config)
    assert batch.values.shape[-2:] == (64, 64)
    assert batch.support.shape[-2:] == (64, 64)
    assert bool(batch.support[..., :48, :].all())
    assert not bool(batch.support[..., 48:, :].any())
    out = build_measurement(s1_config).measure([_frame(shape_hw=(48, 64))])
    # 16 行 padding -> 4 个融合行块无效，共 16 行中的 4 行
    assert float(out.support.float().mean()) == pytest.approx(12 / 16)


def test_uint8_and_float_inputs_share_the_declared_scale(s1_config):
    raw_u8 = torch.randint(
        0, 256, (3, 64, 48), dtype=torch.uint8, generator=torch.Generator().manual_seed(5)
    )
    # LIBERO 按 uint8/255 显式缩放；已归一化的 float 输入用单位尺度，二者数值必须一致
    float_frame = _frame(
        raw=raw_u8.float() / 255.0,
        seed=6,
        scale_name="robotwin_float_linear",
    )
    uint8_frame = MeasurementFrame(
        raw=raw_u8,
        camera_id="cam",
        measurement_id=build_measurement_id("libero:task:demo_1", 0, "cam", "rawvla_rgb3"),
        input_mode="rawvla_rgb3",
        scale=INPUT_SCALES["libero_uint8"],
        size_hw=(64, 48),
        K=K_TEST,
    )
    batch_a = preprocess_frames([float_frame], s1_config)
    batch_b = preprocess_frames([uint8_frame], s1_config)
    assert torch.allclose(batch_a.values, batch_b.values, atol=1e-6)
    model = build_measurement(s1_config)
    out_a = model.measure([float_frame])
    out_b = model.measure([uint8_frame])
    assert torch.allclose(out_a.depth_mean, out_b.depth_mean, atol=1e-6)


def test_no_per_frame_max_normalisation():
    """逐帧 max 归一化会把同一内容的不同缩放映射到同一输出，这里必须保持可区分。"""
    grid = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8) / 100.0
    frame_a = _frame(raw=grid, shape_hw=(8, 8), K=torch.eye(3))
    frame_b = _frame(raw=grid * 2.0, shape_hw=(8, 8), K=torch.eye(3))
    batch_a = preprocess_frames([frame_a], _scale_config())
    batch_b = preprocess_frames([frame_b], _scale_config())
    assert torch.allclose(batch_b.values, 2 * batch_a.values)


def _scale_config():
    from owac.frameworks.n04.full_config import (
        BackboneConfig,
        DataConfig,
        MeasurementConfig,
        N04FullConfig,
        TrainingConfig,
    )

    return N04FullConfig(
        schema="n04-full-s1-v1",
        stage="S1",
        model_kind="n04_full_v1",
        checkpoint_schema="n04-full-v1",
        backbone=BackboneConfig(),
        measurement=MeasurementConfig(),
        data=DataConfig(paired_root="/tmp", factor_root="/tmp"),
        training=TrainingConfig(),
    )


# --------------------------------------------------------------------------------------
# 因果 / 独立性
# --------------------------------------------------------------------------------------


def test_measure_is_invariant_to_provenance_and_identity(model):
    base = _frame(
        seed=11,
        measurement_id=build_measurement_id("a/one", 0, "cam", "rawvla_rgb3"),
        provenance={"episode": "A", "time": 1.0},
    )
    other = MeasurementFrame(
        raw=base.raw,
        camera_id="cam",
        measurement_id=build_measurement_id("b/other-episode", 99, "cam", "rawvla_rgb3"),
        input_mode="rawvla_rgb3",
        scale=base.scale,
        size_hw=base.size_hw,
        K=base.K,
        time_s=42.0,
        provenance={"episode": "B", "time": 4242.0, "history": "replaced"},
    )
    with torch.no_grad():
        out_base = model.measure([base])
        out_other = model.measure([other])
    assert torch.equal(out_base.latent_mean_map, out_other.latent_mean_map)
    assert torch.equal(out_base.depth_mean, out_other.depth_mean)
    assert out_base.measurement_ids != out_other.measurement_ids


def test_batch_permutation_permutes_outputs(model):
    frames = [_frame(seed=21), _frame(seed=22)]
    with torch.no_grad():
        out = model.measure(frames)
        swapped = model.measure(list(reversed(frames)))
    # CPU/GPU 卷积归约顺序在换序后可能差最后几个 ulp，因此按紧公差断言“逐样本独立”
    torch.testing.assert_close(
        swapped.latent_mean_map[0], out.latent_mean_map[1], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(swapped.depth_mean[0], out.depth_mean[1], rtol=1e-5, atol=1e-6)
    assert swapped.measurement_ids[0] == out.measurement_ids[1]


def test_measure_rejects_other_input_mode(model, s1_config):
    cfa_model = build_measurement(_cfa_config(s1_config))
    with pytest.raises(ValueError):
        cfa_model.measure([_frame()])
    with pytest.raises(ValueError):
        model.measure([_frame(mode="bayer_cfa", cfa_pattern="GRBG")])


def _cfa_config(base):
    from dataclasses import replace

    return replace(base, data=replace(base.data, input_mode="bayer_cfa"))


def test_nan_and_invalid_mask_are_cleaned_before_network(s1_config):
    model = build_measurement(s1_config)
    raw = torch.full((3, 64, 48), float("nan"))
    poisoned = _frame(raw=raw, seed=31, K=K_TEST)
    out = model.measure([poisoned])
    assert bool(torch.isfinite(out.depth_mean).all())
    assert float(out.support.float().mean()) == 0.0
    assert float(out.valid.float().mean()) == 0.0
    mask = torch.zeros(64, 48, dtype=torch.bool)
    mask[:8, :] = True
    partially = _frame(seed=32, K=K_TEST, invalid_mask=mask)
    out_partial = model.measure([partially])
    assert bool(torch.isfinite(out_partial.depth_mean).all())
    assert float(out_partial.support.float().mean()) < 1.0


def test_saturation_is_flagged_without_clearing_geometry(s1_config):
    model = build_measurement(s1_config)
    raw = torch.full((3, 64, 48), 255, dtype=torch.uint8)
    out = model.measure([_frame(raw=raw, seed=41)])
    assert bool(out.saturated[..., :12].all())
    assert not bool(out.saturated[..., 12:].any())  # padding 不是饱和测量
    assert float(out.support.float().mean()) == 0.75


# --------------------------------------------------------------------------------------
# 几何头、候选与数值参照
# --------------------------------------------------------------------------------------


def test_depth_positive_and_variance_above_floor(model, s1_config):
    out = model.measure([_frame(seed=51)])
    floor = s1_config.measurement.variance_floor
    assert bool((out.depth_mean > 0).all())
    assert bool((out.depth_variance >= floor).all())
    assert torch.equal(
        out.latent_variance_map,
        torch.full_like(out.latent_variance_map, s1_config.measurement.latent_variance_init),
    )
    assert out.metadata["latent_variance_calibration"] == "fixed_init_uncalibrated_s1"


def test_candidates_partition_and_gt_free(model, s1_config):
    out = model.measure([_frame(seed=61)])
    candidates = out.candidates
    assert (
        candidates.count
        == s1_config.measurement.candidate_rows * s1_config.measurement.candidate_cols
    )
    assert candidates.count <= ANCHOR_CANDIDATES_MAX
    assert candidates.partition_row.min() == 0
    assert candidates.partition_row.max() == s1_config.measurement.candidate_rows - 1
    assert candidates.partition_col.max() == s1_config.measurement.candidate_cols - 1
    # 48 列 pad 到 64 列：融合网格最后 4 列是 padding，落在其中的候选必须无效
    padded_columns = (candidates.xy_feature[..., 0] >= 12.0) & (
        candidates.xy_feature[..., 0] >= 14.0
    )
    assert not bool(candidates.valid[padded_columns].any())
    feature_width = out.sample_grid.feature_size_hw[1]
    interior = candidates.xy_feature[..., 0] < (feature_width - 4)
    assert bool(candidates.valid[interior].all())
    assert bool(candidates.position_valid[interior].all())
    # 置信度只来自预测方差：同一帧反复测量必须给出同一组候选
    again = model.measure([_frame(seed=61)])
    assert torch.equal(candidates.xy_feature, again.candidates.xy_feature)
    assert torch.allclose(candidates.confidence, again.candidates.confidence)


def test_position_covariance_matches_finite_difference_reference():
    torch.manual_seed(0)
    sensor = torch.tensor([[12.0, 20.0], [30.0, 5.0]], dtype=torch.float64)
    depth = torch.tensor([0.7, 1.4], dtype=torch.float64)
    variance = torch.tensor([1e-4, 4e-4], dtype=torch.float64)
    K = K_TEST.to(torch.float64)
    sigma_pixel = 0.5
    position, covariance = position_with_covariance(
        sensor, depth, variance, K, sigma_pixel=sigma_pixel, valid=torch.ones(2, dtype=torch.bool)
    )
    for index in range(2):
        point = torch.cat((sensor[index], depth[index].view(1)))

        def fn(values):
            u, v, d = values[..., 0], values[..., 1], values[..., 2]
            ray = torch.linalg.solve(K, torch.stack((u, v, torch.ones_like(u)), dim=-1))
            return ray * d.unsqueeze(-1)

        jacobian = finite_difference_jacobian(fn, point, eps=1e-6)  # [3,3] wrt (u,v,d)
        scales = torch.diag(
            torch.tensor(
                [sigma_pixel**2, sigma_pixel**2, float(variance[index])], dtype=torch.float64
            )
        )
        expected = jacobian @ scales @ jacobian.transpose(-1, -2)
        assert torch.allclose(covariance[index], expected, rtol=1e-6, atol=1e-12)
        expected_position = (
            torch.linalg.solve(
                K, torch.tensor([sensor[index, 0], sensor[index, 1], 1.0], dtype=torch.float64)
            )
            * depth[index]
        )
        assert torch.allclose(position[index], expected_position, atol=1e-9)


def test_candidates_without_K_have_no_position(s1_config):
    model = build_measurement(s1_config)
    out = model.measure([_frame(seed=71, K=None)])
    assert not bool(out.candidates.position_valid.any())
    assert bool(torch.isnan(out.candidates.position_camera).all())


def test_cfa_mode_end_to_end(s1_config):
    model = build_measurement(_cfa_config(s1_config))
    frame = MeasurementFrame(
        raw=torch.randint(0, 4096, (1, 64, 48), generator=torch.Generator().manual_seed(3)).float(),
        camera_id="cam0",
        measurement_id=build_measurement_id("d0/seq_0", 0, "cam0", "bayer_cfa"),
        input_mode="bayer_cfa",
        scale=INPUT_SCALES["rendered_factor_linear"],
        size_hw=(64, 48),
        K=K_TEST,
        cfa_pattern="GRBG",
    )
    out = model.measure([frame])
    # 64x48 -> 四相位 32x24 -> pad 32x32 -> 融合 8x8；分区数随之收缩到 8x8
    assert out.latent_mean_map.shape == (1, DENSE_LATENT_CHANNELS, 8, 8)
    assert out.candidates.count == 64
    assert out.sample_grid.cfa_plane_names == ("Gr", "R", "B", "Gb")
    assert bool(torch.isfinite(out.depth_mean).all())


# --------------------------------------------------------------------------------------
# 缓存与无记忆基线
# --------------------------------------------------------------------------------------


def test_cache_roundtrip_keeps_identity(model):
    out = model.measure([_frame(seed=81)])
    payload = out.to_cache()
    assert payload["latent_mean_map"].shape == (1, 256, 16, 16)
    assert json.loads(payload["measurement_ids_json"]) == list(out.measurement_ids)
    grid_payload = json.loads(payload["sample_grid_json"])
    assert grid_payload["feature_size_hw"] == [16, 16]
    assert np.allclose(payload["depth_mean"], out.depth_mean.detach().numpy())
    assert payload["valid"].dtype == np.uint8


def test_memoryless_carry_keeps_last_valid_measurement():
    latent = torch.zeros(2, 256, 2, 2)
    carry = MemorylessCarry(latent, ["a/000000/cam@rawvla_rgb3", "b/000000/cam@rawvla_rgb3"])
    updated = torch.ones(2, 256, 2, 2)
    carry.update(
        updated,
        ["a/000001/cam@rawvla_rgb3", "b/000001/cam@rawvla_rgb3"],
        torch.tensor([True, False]),
    )
    assert torch.equal(carry.latent[0], updated[0])
    assert torch.equal(carry.latent[1], latent[1])
    assert carry.measurement_ids[0].endswith("000001/cam@rawvla_rgb3")
    assert carry.measurement_ids[1].endswith("000000/cam@rawvla_rgb3")
    assert carry.age_frames.tolist() == [0, 1]
    carry.update(updated, ["a/000002", "b/000002"], torch.tensor([False, False]))
    assert carry.age_frames.tolist() == [1, 2]
    assert carry.fresh.tolist() == [False, False]
