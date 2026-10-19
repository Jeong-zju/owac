"""WS-C 验收：读出结构、总调度因果行为、以及与既有 π0.5 特征桥的形状契约。

本地没有官方数据、也没有 π0.5 权重：这里用合成帧验证形状、dtype、重复观测、reset、梯度贯通
与 debug 同帧性。WS-A `RaftCorrespondence` 与 WS-B `QualityHead` / `StateEstimator` /
`evidence_update.update` 都由本模块直接调用（不是替身）；唯一的测试替身是零／固定 flow 的
`StandinCorrespondence`，用于注入已知位移并省掉 RAFT 前向。真实 π0.5 后端连通性由 WS-ENV 在
远端提供解释器与权重后另测，本文件不把它冒充成已完成的集成。
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from owac.backends.rawvla_features_n04 import FeatureBridgeConfig, RawVLAFeaturePolicy
from owac.frameworks.n04.contracts import (
    MeasurementFrame,
    build_measurement_id,
    build_sample_grid,
    input_scale_for,
)
from owac.frameworks.n04.correspondence import RaftCorrespondence, RawScale
from owac.frameworks.n04.encoder_v11 import (
    ARM_SPECS,
    EncoderV11Config,
    N04V11Encoder,
    RepeatedFrameError,
    StandinCorrespondence,
    build_encoder,
    flow_at_points,
    grid_to_px,
    platform_defaults,
    px_to_grid,
)
from owac.frameworks.n04.full_config import (
    BackboneConfig,
    DataConfig,
    MeasurementConfig,
    N04FullConfig,
    TrainingConfig,
)
from owac.frameworks.n04.measurement import build_measurement, preprocess_frames
from owac.frameworks.n04.measurement_geometry import PixelPoints, in_bounds_xy
from owac.frameworks.n04.readout import N04V11Readout, PerceiverAttention, ReadoutConfig
from owac.frameworks.n04.state_estimator import covering_grid

SHAPE_HW = (32, 32)
CAMERAS = ("cam_a", "cam_b")


def test_fused_readout_matches_explicit_attention_and_gradients():
    torch.manual_seed(101)
    model = PerceiverAttention(dim=32, dim_head=8, heads=4)
    latent = torch.randn(2, 7, 32, requires_grad=True)
    media = torch.randn(2, 11, 32, requires_grad=True)
    mask = torch.ones(2, 11, dtype=torch.bool)
    mask[:, -3:] = False
    reference, heatmap = model(latent, media, mask, return_attention=True)
    expected_grad = torch.autograd.grad(reference.square().sum(), (latent, media))
    actual, absent = model(latent, media, mask, return_attention=False)
    actual_grad = torch.autograd.grad(actual.square().sum(), (latent, media))
    assert heatmap is not None and absent is None
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
    for got, expected in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize(
    "scale_name,factor", [("libero_uint8", 1 / 255), ("robotwin_float_linear", 1)]
)
def test_rgb_arms_transform_source_once_at_shared_boundary(measurement, arm, scale_name, factor):
    from owac.raw.n04_v11_data import fixed_isp_rgb

    encoder = N04V11Encoder(
        measurement,
        EncoderV11Config(
            arm=arm,
            cameras=("cam_a",),
            scale_name=scale_name,
            saturation_level=3.5,
            readout=small_readout_config(),
        ),
        correspondence=StandinCorrespondence(),
    )
    raw = torch.linspace(0, 2 / factor, 3 * 32 * 32).reshape(3, 32, 32)
    frames, _ = encoder._as_frames([raw], ["cam_a"], None, None)
    expected = fixed_isp_rgb(raw, scale_factor=factor)
    torch.testing.assert_close(frames[0].raw, expected)
    assert float(frames[0].raw.max()) <= 1 / factor + 1e-4
    assert frames[0].saturation_level == 1.0
    assert encoder.raw_scale.saturation_level == 1.0
    # RAW 的饱和 mask 不得旁路影响 RGB 质量头/骨干。
    encoder.eval()
    with torch.no_grad():
        first = encoder.step([raw], camera_id=["cam_a"], saturation=[torch.ones(1, 32, 32)])
        encoder.reset()
        second = encoder.step([raw], camera_id=["cam_a"], saturation=[torch.zeros(1, 32, 32)])
    torch.testing.assert_close(first.prefix_tokens, second.prefix_tokens)


def small_readout_config() -> ReadoutConfig:
    """结构完整的读出（层数/宽度缩小以便 CPU 测试），输出仍是每相机 256×2048。"""
    return ReadoutConfig(dim=64, dim_head=16, heads=4, depth=1, ff_mult=2)


def test_normalized_live_encoder_and_stopped_target_share_scale(measurement):
    encoder = N04V11Encoder(
        config=EncoderV11Config(
            cameras=("cam_a",),
            readout=small_readout_config(),
            observation_normalization="rms",
            transition_init="identity",
            transition_norm_limit=1.0,
            correspondence_boundary_weight=1.0,
        ),
        measurement=measurement,
        correspondence=StandinCorrespondence(),
    )
    encoder.eval()
    with torch.no_grad():
        output = encoder.step(
            [torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)], camera_id=["cam_a"]
        )
    latent = output.target_features["cam_a"]["latent"]
    assert latent.dtype == torch.float32 and not latent.requires_grad
    torch.testing.assert_close(
        latent.square().mean(1), torch.ones_like(latent[:, 0]), rtol=2e-5, atol=2e-5
    )


def measurement_config() -> N04FullConfig:
    return N04FullConfig(
        schema="n04-v1.1-test",
        stage="s2",
        model_kind="n04_full_v1",
        checkpoint_schema="n04-full-v1",
        backbone=BackboneConfig(),
        measurement=MeasurementConfig(),
        data=DataConfig(paired_root=".", factor_root="."),
        training=TrainingConfig(),
        name="s2_test",
    )


@pytest.fixture(scope="module")
def measurement():
    return build_measurement(measurement_config())


def make_encoder(
    measurement,
    *,
    arm: str = "D",
    cameras: tuple[str, ...] = ("cam_a",),
    correspondence=None,
    allow_duplicate_step: bool = False,
) -> N04V11Encoder:
    config = EncoderV11Config(
        arm=arm,
        cameras=cameras,
        readout=small_readout_config(),
        allow_duplicate_step=allow_duplicate_step,
    )
    encoder = N04V11Encoder(
        measurement, config, correspondence=correspondence or StandinCorrespondence()
    )
    encoder.zero_grad(set_to_none=True)
    return encoder


def frame(seed: int, shape_hw: tuple[int, int] = SHAPE_HW) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, *shape_hw, generator=generator)


# ---------------------------------------------------------------------------------------
# 读出结构
# ---------------------------------------------------------------------------------------


def test_readout_default_structure_is_six_block_perceiver():
    """默认配置必须是 OpenFlamingo 完整 Perceiver：6 blocks / 256 queries / 2048 投影。"""
    config = ReadoutConfig(num_cameras=1)
    assert (config.depth, config.num_latents, config.prefix_dim) == (6, 256, 2048)
    readout = N04V11Readout(config)
    assert len(readout.resampler.blocks) == 6
    assert tuple(readout.resampler.latents.shape) == (256, 1024)
    assert readout.to_prefix.out_features == 2048
    parameters = sum(p.numel() for p in readout.parameters())
    print(f"[readout] default 6-block perceiver parameters = {parameters:,}")

    latent = torch.randn(1, 256, 8, 8)
    num_support = 128
    tokens, attention, counts = readout(
        latent,
        support_state=torch.randn(1, num_support, 2, 512),
        mode_weights=torch.full((1, num_support, 2), 0.5),
        xy_px=torch.rand(1, num_support, 2) * 30.0,
        age=torch.zeros(1, num_support),
        size_hw=SHAPE_HW,
    )
    assert tokens.shape == (1, 256, 2048)
    assert tokens.dtype == torch.float32
    assert counts == {
        "dense_tokens": 64,
        "support_tokens": num_support,
        "media_tokens": 192,
        "latent_tokens": 256,
        "attention_keys": 448,
    }
    assert attention.shape == (1, 256, 448)  # 与当前帧媒体 token 对齐，供可视化
    assert torch.allclose(attention.sum(-1), torch.ones(1, 256), atol=1e-5)


def test_readout_attention_is_bound_to_current_observation():
    """attention 必须来自当前帧：换一张当前稠密特征，attention 与 token 都要变。"""
    readout = N04V11Readout(small_readout_config())
    kwargs = dict(
        support_state=torch.randn(1, 8, 2, 512),
        mode_weights=torch.full((1, 8, 2), 0.5),
        xy_px=torch.rand(1, 8, 2) * 30.0,
        age=torch.arange(8.0).reshape(1, 8),
        size_hw=SHAPE_HW,
    )
    first, attention_a, counts = readout(torch.randn(1, 256, 8, 8), **kwargs)
    second, attention_b, _ = readout(torch.randn(1, 256, 8, 8), **kwargs)
    assert counts["media_tokens"] == 64 + 8
    assert attention_a.shape == (1, 256, 328)
    assert not torch.allclose(first, second)
    assert not torch.allclose(attention_a, attention_b)


# ---------------------------------------------------------------------------------------
# 与既有特征桥的形状契约
# ---------------------------------------------------------------------------------------


class _StubVLM:
    """只是为了跑通 ``embed_prefix_from_visual_tokens`` 的语言嵌入；没有预训练权重。"""

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, tokens.shape[1], 2048, dtype=torch.float32)


class _StubBackend:
    """满足 ``RawVLAFeaturePolicy`` 构造时的两个静态方法检查；不做真实前向。"""

    def __init__(self) -> None:
        self.vlm = _StubVLM()
        self.max_state_dim = 32
        self.state_pad_value = 0.0
        self.tokenizer = None
        self.discrete_state_input = False

    @staticmethod
    def forward_from_observation(view, observation, actions, **kwargs):  # pragma: no cover
        raise NotImplementedError("本地没有 π0.5 权重")

    @staticmethod
    def sample_actions(view, observation, **kwargs):  # pragma: no cover
        raise NotImplementedError("本地没有 π0.5 权重")


def test_prefix_tokens_shape_matches_feature_bridge(measurement):
    """前缀必须能被真实 ``embed_prefix_from_visual_tokens`` 直接吃掉（形状断言）。"""
    encoder = make_encoder(measurement)
    output = encoder.step(frame(0), {"proprio": torch.randn(16)}, None, frame_key=("ep", 0))
    policy = RawVLAFeaturePolicy(encoder, _StubBackend(), FeatureBridgeConfig())
    visual = encoder.readout_tokens(output, policy.config.tokens_per_camera)
    assert visual.values.shape[1] % policy.config.tokens_per_camera == 0
    assert visual.values.shape[-1] == policy.config.token_dim

    task_tokens = torch.zeros(1, 5, dtype=torch.long)
    task_mask = torch.ones(1, 5, dtype=torch.bool)
    prefix = policy.embed_prefix_from_visual_tokens(visual, task_tokens, task_mask)
    assert prefix["prefix"].shape == (1, 256 + 5, 2048)
    assert prefix["prefix_mask"].shape == (1, 261)
    assert prefix["prefix_mask"].all()
    assert prefix["attention_blocks"].shape == (1, 261)
    assert prefix["num_visual_tokens"] == 256
    assert prefix["num_language_tokens"] == 5
    assert prefix["positions"][0, -1].item() == 260
    assert abs(prefix["embedding_scale"] - math.sqrt(2048)) < 1e-6


def test_fp32_prefix_preserves_gradients_before_backend_cast():
    config = ReadoutConfig(
        dim=32,
        dim_head=8,
        heads=4,
        depth=1,
        num_latents=4,
        prefix_dim=16,
        latent_channels=8,
        prefix_fp32=True,
    )
    readout = N04V11Readout(config)
    latent = torch.randn(1, 8, 4, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prefix, _, _ = readout(latent, return_attention=False)
    assert prefix.dtype == torch.float32
    prefix.square().mean().backward()
    assert latent.grad is not None and bool(torch.isfinite(latent.grad).all())


# ---------------------------------------------------------------------------------------
# step：形状、dtype、debug
# ---------------------------------------------------------------------------------------


def test_step_shapes_dtype_and_debug_fields(measurement):
    encoder = make_encoder(measurement)
    first = encoder.step(
        frame(1), {"proprio": torch.randn(16)}, None, camera_id="cam_a", frame_key=("ep", 7)
    )
    second = encoder.step(
        frame(2), {"proprio": torch.randn(16)}, first.state, camera_id="cam_a", frame_key=("ep", 8)
    )
    prefix = second.prefix_tokens
    assert prefix.shape == (1, 256, 2048)
    assert prefix.dtype == torch.float32
    assert second.state is not None
    state = second.state.per_camera["cam_a"]
    debug = second.debug
    assert set(
        ["flow", "quality", "evidence", "mode_weights", "support", "readout_attention"]
    ) <= set(debug)

    flow = debug["flow"]["cam_a"]
    assert flow.shape == (1, 2, *SHAPE_HW)
    quality = debug["quality"]["cam_a"]["R_diag"]
    assert quality.shape == (1, 128, 256)
    assert float(quality.min()) >= 1e-6
    assert debug["quality"]["cam_a"]["min"] >= 1e-6
    weights = debug["mode_weights"]["cam_a"]["posterior"]
    assert weights.shape == (1, 128, 2)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 128), atol=1e-5)
    assert debug["mode_weights"]["cam_a"]["prior"].shape == (1, 128, 2)
    evidence = debug["evidence"]["cam_a"]
    assert evidence["residual"].shape == (1, 128, 2, 256)
    assert evidence["innovation_variance"].shape == (1, 128, 2, 256)
    assert evidence["log_weight"].shape == (1, 128, 2)
    assert evidence["obs_valid"].shape == (1, 128)
    support = debug["support"]["cam_a"]
    assert support["xy_px"].shape == (1, 128, 2)
    assert torch.allclose(support["xy_px"], state.xy)
    assert support["ids"].shape == (1, 128)
    assert support["age"].shape == (1, 128)
    assert float(support["age"].max()) >= 1.0  # 第二帧：出生帧 age=0
    assert len(support["trajectory"]) == 2
    attention = debug["readout_attention"]["cam_a"]
    dense = support["media_tokens"]["dense_tokens"]
    assert attention.shape == (1, 256, dense + 128 + 256)
    # 出生点必须在帧内，且覆盖像幅。
    assert support["xy_px"][0, :, 0].max() < SHAPE_HW[1]
    assert support["xy_px"][0, :, 1].max() < SHAPE_HW[0]


def test_two_cameras_concatenate_in_order(measurement):
    encoder = make_encoder(measurement, cameras=CAMERAS)
    output = encoder.step(
        [frame(2), frame(3)],
        None,
        None,
        camera_id=list(CAMERAS),
        frame_key=("ep", 0),
    )
    assert output.prefix_tokens.shape == (1, 2 * 256, 2048)
    assert list(output.debug["camera_ids"]) == list(CAMERAS)
    assert output.state is not None
    assert tuple(output.state.cameras()) == CAMERAS
    assert set(output.state.per_camera) == set(CAMERAS)
    # 每个相机有自己的状态张量（不同帧内容 -> 不同后验）。
    assert not torch.allclose(
        output.state.per_camera["cam_a"].mode_mean,
        output.state.per_camera["cam_b"].mode_mean,
    )
    assert encoder.readout_tokens(output, 256).values.shape == (1, 512, 2048)
    assert encoder.readout_tokens(output, 128).values.shape == (1, 128, 2048)


def test_flow_is_sampled_at_source_points(measurement):
    """flow 采样在**源点**（上一帧坐标）：``xy_t = xy_{t-1} + flow_at_points(flow, xy_{t-1})``。

    方向性必须可测：把同一 flow 用目标点／相反符号解释会得到相反位移。
    """
    height, width = SHAPE_HW
    # ① 常量场：源点采样给 +c；相反解释（目标点/反向符号）给 -c，必须能被测出。
    constant = torch.zeros(1, 2, height, width)
    constant[:, 0], constant[:, 1] = -1.0, 1.0
    encoder = make_encoder(measurement, correspondence=StandinCorrespondence(constant))
    first = encoder.step(frame(4), None, None, camera_id="cam_a", frame_key=("ep", 0))
    second = encoder.step(frame(5), None, first.state, camera_id="cam_a", frame_key=("ep", 1))
    assert first.state is not None and second.state is not None
    before = first.state.per_camera["cam_a"]
    after = second.state.per_camera["cam_a"]
    assert bool(after.valid.all())  # 位移留在画幅内，没有空槽补种
    moved = after.xy - before.xy
    expected = torch.zeros_like(moved)
    expected[..., 0], expected[..., 1] = -1.0, 1.0
    assert torch.allclose(moved, expected, atol=1e-3)
    opposite = (moved - (-expected)).abs().max()
    assert float(opposite) > 1.0  # 相反符号解释的位移差 = 2px，可测出
    source_gap = float((moved - expected).abs().max())
    print(
        f"[flow] src-point gap={source_gap:.2e} px, opposite-direction gap={float(opposite):.3f} px"
    )

    # ② 空间变化场：源点采样与目标点采样给出不同位移，也必须能被测出。
    varying = torch.zeros(1, 2, height, width)
    xs = torch.arange(width, dtype=torch.float32)
    varying[:, 0] = (-4.0 * xs / float(width)).reshape(1, width).expand(height, width)
    shifting = make_encoder(measurement, correspondence=StandinCorrespondence(varying))
    start = shifting.step(frame(6), None, None, camera_id="cam_a", frame_key=("ep", 0))
    end = shifting.step(frame(7), None, start.state, camera_id="cam_a", frame_key=("ep", 1))
    assert start.state is not None and end.state is not None
    xy_before = start.state.per_camera["cam_a"].xy
    delta = end.state.per_camera["cam_a"].xy - xy_before
    assert torch.allclose(delta, flow_at_points(varying, xy_before), atol=1e-3)  # 源点
    destination_gap = (
        (delta - flow_at_points(varying, xy_before + delta)).abs().max()
    )  # 目标点采样的结果不同
    assert float(destination_gap) > 0.2
    assert bool(end.state.per_camera["cam_a"].valid.all())
    print(f"[flow] destination-point gap={float(destination_gap):.3f} px")

    # 越界空槽：位置按 flow 推进后被标记 invalid 并由新区域补种（不重排已有支持）。
    outward = torch.full((1, 2, height, width), 4.0)
    outbound = make_encoder(measurement, correspondence=StandinCorrespondence(outward))
    start = outbound.step(frame(4), None, None, camera_id="cam_a", frame_key=("ep", 0))
    end = outbound.step(frame(5), None, start.state, camera_id="cam_a", frame_key=("ep", 1))
    assert start.state is not None and end.state is not None
    reborn = (end.state.per_camera["cam_a"].ids != start.state.per_camera["cam_a"].ids) | (
        end.state.per_camera["cam_a"].fresh
    )
    assert bool(reborn.any())  # 至少右/下边缘的支持越界并被补种
    # 补种只接受"下一帧仍在画幅内"的候选：出生点必须能活过当前 motion（否则就是 ID 抖动）。
    born_xy = end.state.per_camera["cam_a"].xy[reborn]
    next_xy = born_xy + flow_at_points(outward, born_xy.unsqueeze(0)).squeeze(0)
    assert bool(in_bounds_xy(PixelPoints(xy=next_xy, unit="original_px"), SHAPE_HW).all())


def test_real_raft_correspondence_is_wired_by_default(measurement):
    """默认对应网络是 WS-A 的完整 RAFT（这里 1 次迭代、小帧，只验证接通与形状）。"""
    encoder = N04V11Encoder(
        measurement,
        EncoderV11Config(cameras=("cam_a",), readout=small_readout_config(), raft_iters=1),
    )
    assert isinstance(encoder.correspondence, RaftCorrespondence)
    assert isinstance(encoder.correspondence.scale, RawScale)
    first = encoder.step(frame(20), None, None, camera_id="cam_a", frame_key=("e", 0))
    second = encoder.step(frame(21), None, first.state, camera_id="cam_a", frame_key=("e", 1))
    flow = second.debug["flow"]["cam_a"]
    assert flow.shape == (1, 2, *SHAPE_HW)
    assert bool(torch.isfinite(flow).all())


def test_correspondence_loss_trains_raft(measurement):
    """`losses["correspondence"]` 是真实对应自监督：调用 WS-A 损失并回传到 RAFT 三部件。"""
    encoder = N04V11Encoder(
        measurement,
        EncoderV11Config(cameras=("cam_a",), readout=small_readout_config(), raft_iters=1),
    )
    encoder.zero_grad(set_to_none=True)
    first = encoder.step(frame(30), None, None, camera_id="cam_a", frame_key=("e", 0))
    assert "correspondence" not in first.losses  # 首帧没有帧对：未产生，不用 0 冒充
    second = encoder.step(frame(31), None, first.state, camera_id="cam_a", frame_key=("e", 1))
    loss = second.losses["correspondence"]
    assert loss.dim() == 0
    assert bool(torch.isfinite(loss)) and float(loss) >= 0.0
    assert loss.requires_grad  # 来自本步的 RAFT flow，不是常数
    print(f"[correspondence] two-step loss = {float(loss):.6f}")
    loss.backward()
    raft = encoder.correspondence.raft
    for name, module in (
        ("fnet", raft.fnet),
        ("cnet", raft.cnet),
        ("update_block", raft.update_block),
    ):
        grad_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        assert grad_sum > 0.0, f"RAFT {name} 没拿到对应损失的梯度"
    # 对应损失只训练 RAFT；测量骨干的梯度来自 prefix 的其它损失项。
    assert measurement.fusion.weight.grad is None


def test_forecast_inputs_expose_prior_state(measurement):
    """`forecast_inputs` 给 WS-D：t 时刻的支持先验（predict 后、本帧校正前）、xy、ids。"""
    encoder = make_encoder(measurement)
    first = encoder.step(frame(32), None, None, camera_id="cam_a", frame_key=("e", 0))
    assert set(first.forecast_inputs) == {"cam_a"}
    spawn = first.forecast_inputs["cam_a"]
    assert torch.allclose(
        spawn["mode_weights"], encoder.state_stack.birth_weights().expand(1, 128, 2)
    )
    second = encoder.step(frame(33), None, first.state, camera_id="cam_a", frame_key=("e", 1))
    forecast = second.forecast_inputs["cam_a"]
    assert forecast["mode_weights"].shape == (1, 128, 2)
    assert forecast["mode_mean"].shape == (1, 128, 2, 512)  # 两个模式各自一份均值
    assert [tuple(block.shape) for block in forecast["mode_cov"]] == [
        (1, 128, 2, 256),
        (1, 128, 2, 256),
        (1, 128, 2, 256),
    ]
    assert forecast["xy"].shape == (1, 128, 2)
    assert forecast["ids"].shape == (1, 128)
    assert forecast["fresh"].shape == (1, 128)
    assert torch.allclose(forecast["mode_weights"].sum(-1), torch.ones(1, 128), atol=1e-5)
    assert torch.allclose(forecast["xy"], second.debug["support"]["cam_a"]["xy_px"])
    assert not torch.allclose(
        forecast["mode_weights"], second.state.per_camera["cam_a"].mode_weights
    )
    # 只有状态臂才有先验可给。
    stateless = make_encoder(measurement, arm="C")
    assert stateless.step(frame(34), None, None, camera_id="cam_a").forecast_inputs == {}


SMALL_READOUT_DICT = {"dim": 64, "dim_head": 16, "heads": 4, "depth": 1, "ff_mult": 2}


def test_build_encoder_accepts_training_entrypoint_convention(measurement):
    """WS-D 约定：``build_encoder(model_cfg_dict, platform=..., arm=...)``（本批漏掉的路径）。"""
    model_cfg = {
        "scale_name": "libero_uint8",
        "proprio_dim": 8,
        "readout": dict(SMALL_READOUT_DICT),
    }
    encoder = build_encoder(model_cfg, platform="libero", arm="D", measurement=measurement)
    assert isinstance(encoder, N04V11Encoder)
    assert encoder.config.arm == "D"
    assert encoder.config.cameras == ("agentview", "wrist")
    assert encoder.config.scale_name == "libero_uint8"
    assert encoder.config.num_cameras == 2
    assert encoder.config_ignored_keys == []
    assert isinstance(encoder.config.readout, ReadoutConfig)

    stepped = encoder.step(
        [frame(40), frame(41)],
        None,
        None,
        camera_id=["agentview", "wrist"],
        frame_key=("e", 0),
    )
    assert stepped.prefix_tokens.shape == (1, 2 * 256, 2048)  # 两个相机各 256 个 token

    robotwin = build_encoder(
        {"readout": dict(SMALL_READOUT_DICT)},
        platform="robotwin",
        arm="C",
        measurement=measurement,
        correspondence=StandinCorrespondence(),
    )
    assert robotwin.config.cameras == ("head", "left", "right")
    assert robotwin.config.scale_name == "robotwin_float_linear"
    assert robotwin.config.num_cameras == 3
    assert robotwin.config.arm == "C"
    assert platform_defaults("libero") == (("agentview", "wrist"), "libero_uint8")
    assert platform_defaults("robotwin") == (
        ("head", "left", "right"),
        "robotwin_float_linear",
    )
    # 输入域 ↔ 契约尺度：LIBERO uint8_0_255 期望 1/255，RoboTwin float32_linear 期望 1.0
    assert float(input_scale_for("libero_uint8").factor) == pytest.approx(1.0 / 255.0)
    assert float(input_scale_for("robotwin_float_linear").factor) == pytest.approx(1.0)
    for platform in ("libero", "robotwin"):
        _, scale_name = platform_defaults(platform)
        expected = 1.0 / 255.0 if platform == "libero" else 1.0
        assert float(input_scale_for(scale_name).factor) == pytest.approx(expected)


def test_platform_domain_and_contract_scale_must_match(monkeypatch):
    """``value_scale`` 记录的是输入原始域：与契约尺度不一致或域未登记都必须报错。"""
    from dataclasses import replace

    from owac.raw.n04_v11_data import PLATFORM_SPECS

    spec = PLATFORM_SPECS["libero"]
    monkeypatch.setitem(PLATFORM_SPECS, "libero", replace(spec, value_scale="float32_linear"))
    with pytest.raises(ValueError, match="不一致"):
        platform_defaults("libero")
    monkeypatch.setitem(PLATFORM_SPECS, "libero", replace(spec, value_scale="uint12_magic"))
    with pytest.raises(ValueError, match="未登记"):
        platform_defaults("libero")
    monkeypatch.setitem(PLATFORM_SPECS, "libero", replace(spec, value_scale="uint8_0_255"))
    assert platform_defaults("libero") == (("agentview", "wrist"), "libero_uint8")


def test_build_encoder_does_not_silently_drop_platform_or_arm(measurement):
    """平台／臂与显式配置冲突时报错；未识别键记录在 config_ignored_keys，不静默丢弃。"""
    stub = {"correspondence": StandinCorrespondence(), "measurement": measurement}
    with pytest.raises(ValueError, match="冲突"):
        build_encoder({"arm": "B"}, platform="libero", arm="D", **stub)
    with pytest.raises(ValueError, match="冲突"):
        build_encoder({"scale_name": "robotwin_float_linear"}, platform="libero", **stub)
    with pytest.raises(ValueError, match="未知平台"):
        build_encoder({}, platform="droid", **stub)
    with pytest.raises(ValueError, match="未知消融臂"):
        build_encoder({}, platform="libero", arm="E", **stub)
    with pytest.raises(TypeError):
        build_encoder(42, platform="libero", **stub)

    kept = build_encoder(
        {
            "scale_name": "libero_uint8",
            "readout": dict(SMALL_READOUT_DICT),
            "backbone": {"depths": [3, 3, 27, 3]},
            "token_budget": 256,
        },
        platform="libero",
        arm="D",
        **stub,
    )
    assert kept.config_ignored_keys == ["backbone", "token_budget"]
    assert isinstance(kept.config.readout, ReadoutConfig)
    # WS-D 本地 wrapper 走的是 dataclass 路径，仍需可用（不传 platform／arm）。
    direct = build_encoder(
        EncoderV11Config(cameras=("cam_a",), readout=small_readout_config()),
        correspondence=StandinCorrespondence(),
        measurement=measurement,
    )
    assert isinstance(direct, N04V11Encoder)
    assert direct.config.cameras == ("cam_a",)


def test_build_encoder_maps_training_json_keys(measurement):
    """训练 JSON 的键名（input_mode 别名、raft_iterations、readout_*）必须真的生效。"""
    payload = {
        "name": "owac-n04-v1.1-convnext-base",
        "builder": "owac.frameworks.n04.encoder_v11:build_encoder",
        "input_mode": "rawvla_raw3",
        "backbone": "convnext_base",
        "observation_dim": 256,
        "state_dim": 512,
        "fusion_channels": 256,
        "fusion_stride": 4,
        "raft_iterations": 2,
        "raft_small": False,
        "readout_blocks": 1,
        "tokens_per_camera": 256,
        "token_dim": 2048,
        "scale_name": "libero_uint8",
    }
    encoder = build_encoder(
        payload,
        platform="libero",
        arm="D",
        measurement=measurement,
        correspondence=StandinCorrespondence(),
    )
    assert encoder.config.input_mode == "rawvla_rgb3"  # S1 三通道容器
    assert ARM_SPECS["D"].input_domain == "raw3"
    assert encoder.config.raft_iters == 2
    assert (
        encoder.config.readout.depth,
        encoder.config.readout.num_latents,
        encoder.config.readout.prefix_dim,
    ) == (1, 256, 2048)
    assert encoder.config_ignored_keys == ["backbone", "builder", "name"]
    assert ARM_SPECS["A"].input_domain == "rgb3"


def test_build_encoder_rejects_contract_conflicts(measurement):
    """与契约常量冲突的 JSON 键必须报错，不能"配置写了却不生效"。"""
    for key, value in (
        ("observation_dim", 128),
        ("state_dim", 256),
        ("fusion_channels", 128),
        ("fusion_stride", 8),
        ("raft_small", True),
    ):
        with pytest.raises(ValueError, match="契约"):
            build_encoder(
                {key: value},
                measurement=measurement,
                correspondence=StandinCorrespondence(),
            )
    with pytest.raises(ValueError, match="未知输入容器"):
        EncoderV11Config(input_mode="raw3rgb")
    assert EncoderV11Config(input_mode="rawvla_raw3").input_mode == "rawvla_rgb3"
    assert EncoderV11Config(input_mode="bayer_cfa").input_mode == "bayer_cfa"


def _support_births(measurement, *, shape_hw, flow_px, steps, visibility=None, camera="cam_a"):
    """跑若干步，返回每步的 (出生掩码, SupportState 快照)，用于核对出生语义。"""
    import torch.nn.functional as F  # noqa: F401  (保持局部导入最小)

    flow = torch.zeros(1, 2, *shape_hw)
    if flow_px is not None:
        flow[:, 0] = float(flow_px[0])
        flow[:, 1] = float(flow_px[1])
    generator = torch.Generator().manual_seed(0)
    frames = [torch.rand(3, *shape_hw, generator=generator) for _ in range(steps + 1)]
    encoder = N04V11Encoder(
        measurement,
        EncoderV11Config(cameras=(camera,), readout=small_readout_config()),
        correspondence=StandinCorrespondence(flow, valid=visibility),
    )
    container, previous_ids = None, None
    history = []
    for index in range(steps):
        output = encoder.step(
            frames[index], None, container, camera_id=camera, frame_key=("e", index)
        )
        container = output.state
        state = container.per_camera[camera]
        born = (
            torch.zeros_like(state.ids[0], dtype=torch.bool)
            if previous_ids is None
            else state.ids[0] != previous_ids
        )
        history.append((born, state))
        previous_ids = state.ids[0].clone()
    return encoder, history


def test_occlusion_does_not_trigger_births_or_change_ids(measurement):
    """契约 §4：遮挡只传播预测，不产生新支持（WS-B `birth` 只填越界空槽）。"""
    shape_hw = (32, 32)
    visibility = torch.ones(*shape_hw, dtype=torch.bool)
    visibility[:16, :] = False  # 上半画幅不可见 -> 32 个支持被遮挡
    _, history = _support_births(
        measurement, shape_hw=shape_hw, flow_px=None, steps=3, visibility=visibility
    )
    for born, state in history:
        assert int(born.sum()) == 0  # 遮挡不触发出生
        assert torch.equal(state.ids[0], torch.arange(128))
    grid = covering_grid(128, shape_hw)
    occluded = int((grid[:, 1] < 16.0).sum())  # 落在上半画幅的支持数
    assert occluded == 64
    assert int(history[-1][1].valid.sum()) == 128 - occluded
    assert float(history[-1][1].age.max()) == 2.0  # 仍然按记录帧推进
    print(
        f"[birth] occlusion: births=0 ids=[0,127] "
        f"valid={int(history[-1][1].valid.sum())}/{128 - occluded}"
    )


def test_no_births_without_out_of_bounds_slots(measurement):
    """没有越界空槽时（flow=0）不得出生。"""
    _, history = _support_births(measurement, shape_hw=(32, 32), flow_px=None, steps=3)
    assert all(int(born.sum()) == 0 for born, _ in history)
    assert int(history[-1][1].valid.sum()) == 128
    print("[birth] flow=0: births=0 ids=[0,127] valid=128")


def test_new_area_births_survive_the_current_motion(measurement):
    """出生候选必须下一帧仍在画幅内：否则会在同一像点每帧重生（本地实测的 ID 抖动）。

    这里用 flow=6px/帧、64×64（网格间距 x=4）：旧规则会把最外圈 30 个「出生即越界」的候选
    一并递上去；修复后这些候选被拒，所有新生支持下一帧仍在画幅内。
    """
    shape_hw = (64, 64)
    flow_px = (6.0, 6.0)
    encoder, history = _support_births(measurement, shape_hw=shape_hw, flow_px=flow_px, steps=5)
    flow = torch.zeros(1, 2, *shape_hw)
    flow[:, 0] = float(flow_px[0])
    flow[:, 1] = float(flow_px[1])
    candidates = covering_grid(128, shape_hw).unsqueeze(0)
    next_xy = candidates + flow_at_points(flow, candidates)
    rejected = int((~in_bounds_xy(PixelPoints(xy=next_xy, unit="original_px"), shape_hw)).sum())
    assert rejected > 0  # 该几何下确实有"出生即越界"的候选

    born_total = 0
    for born, state in history:
        born_total += int(born.sum())
        if born.any():
            born_xy = state.xy[0][born]
            survivors = in_bounds_xy(
                PixelPoints(
                    xy=born_xy + flow_at_points(flow, born_xy.unsqueeze(0)).squeeze(0),
                    unit="original_px",
                ),
                shape_hw,
            )
            assert bool(survivors.all()), "有支持出生在下一帧即越界的位置（ID 抖动来源）"
        assert state.ids.shape == (1, 128)  # 容量固定
        assert int(state.valid.sum()) == 128
    assert born_total > 0  # 新区域确实会出生
    ids_max = int(history[-1][1].ids.max())
    assert ids_max > 127  # ID 是累计计数器，不是槽位索引
    print(
        f"[birth] flow={flow_px}/帧 {shape_hw}: births_total={born_total} "
        f"ids=[0,{ids_max}] valid=128 每帧被拒候选={rejected}"
    )

    # 真实比例（128×128、flow 3px ≪ 网格间距 x=8/y=16px）：防抖动守卫不拒绝任何候选（=0）。
    realistic = covering_grid(128, (128, 128)).unsqueeze(0)
    realistic_flow = torch.zeros(1, 2, 128, 128)
    realistic_flow[:] = 3.0
    realistic_next = realistic + flow_at_points(realistic_flow, realistic)
    rejected_realistic = int(
        (~in_bounds_xy(PixelPoints(xy=realistic_next, unit="original_px"), (128, 128))).sum()
    )
    assert rejected_realistic == 0


def test_debug_reports_window_diagnostics(measurement):
    """契约 §4 的覆盖判据可观测指标：births_per_frame / ids_max / median_age（只加字段）。"""
    # _support_births 只返回状态快照，这里显式跑两步取 output.debug。
    encoder = N04V11Encoder(
        measurement,
        EncoderV11Config(cameras=("cam_a",), readout=small_readout_config()),
        correspondence=StandinCorrespondence(torch.full((1, 2, 64, 64), 6.0)),
    )
    first = encoder.step(frame(60, (64, 64)), None, None, camera_id="cam_a", frame_key=("e", 0))
    second = encoder.step(
        frame(61, (64, 64)), None, first.state, camera_id="cam_a", frame_key=("e", 1)
    )
    debug = second.debug
    for key in ("births_per_frame", "ids_max", "median_age"):
        assert key in debug, key
    assert isinstance(debug["births_per_frame"], int) and debug["births_per_frame"] >= 0
    assert isinstance(debug["ids_max"], int) and debug["ids_max"] >= 127
    assert math.isfinite(debug["median_age"]) and debug["median_age"] >= 0.0
    # 字段来自真实计数器：出生数与该步 ID 变化数一致
    previous_ids = first.state.per_camera["cam_a"].ids
    changed = int((second.state.per_camera["cam_a"].ids != previous_ids).sum())
    assert debug["births_per_frame"] == changed
    assert debug["ids_max"] == int(second.state.per_camera["cam_a"].ids.max())
    assert debug["median_age"] == float(
        torch.median(
            second.state.per_camera["cam_a"]
            .age[second.state.per_camera["cam_a"].valid]
            .to(torch.float32)
        )
    )
    # 无状态臂（C）没有这些量：None，不用 0 冒充
    stateless = make_encoder(measurement, arm="C")
    empty = stateless.step(frame(62), None, None, camera_id="cam_a", frame_key=("e", 0))
    assert empty.debug["births_per_frame"] is None
    assert empty.debug["ids_max"] is None
    assert empty.debug["median_age"] is None
    print(
        f"[coverage] births_per_frame={debug['births_per_frame']} ids_max={debug['ids_max']} "
        f"median_age={debug['median_age']:.1f}"
    )


def test_long_lived_containers_stay_bounded(measurement):
    """age／trajectory／ids 等长生命周期容器有界：轨迹是定长 deque，状态张量形状固定。"""
    encoder, history = _support_births(measurement, shape_hw=(32, 32), flow_px=None, steps=5)
    assert encoder.prev_frame_cameras == ("cam_a",)  # 每相机一份
    assert len(encoder.support_trajectories["cam_a"]) == min(5, encoder.config.trajectory_len)
    for _, state in history:
        assert state.xy.shape == (1, 128, 2)
        assert state.age.shape == (1, 128) and state.age.dtype == torch.int64
        assert state.ids.shape == (1, 128) and state.ids.dtype == torch.int64
        assert state.mode_mean.shape == (1, 128, 2, 512)
    assert encoder.support_trajectories["cam_a"][-1].shape == (1, 128, 2)
    # 轨迹不会超过配置上限（有界）。
    assert encoder.config.trajectory_len <= 64


# ---------------------------------------------------------------------------------------
# 因果：重复观测、计数、reset
# ---------------------------------------------------------------------------------------


def test_duplicate_step_is_counted_and_does_not_re_fuse(measurement):
    encoder = make_encoder(measurement)
    raw = frame(6)
    first = encoder.step(raw, None, None, camera_id="cam_a", frame_key=("ep", 3))
    assert encoder.state_update_count == 1
    assert encoder.measurement_count == 1
    with pytest.raises(RepeatedFrameError):
        encoder.step(raw, None, first.state, camera_id="cam_a", frame_key=("ep", 3))
    assert encoder.duplicate_frame_count == 1
    assert encoder.state_update_count == 1
    assert encoder.measurement_count == 1

    tolerant = make_encoder(measurement, allow_duplicate_step=True)
    cached = tolerant.step(raw, None, None, camera_id="cam_a", frame_key=("ep", 3))
    again = tolerant.step(raw, None, cached.state, camera_id="cam_a", frame_key=("ep", 3))
    assert again is cached
    assert again.cache_hit
    assert tolerant.duplicate_frame_count == 1
    assert tolerant.state_update_count == 1
    assert tolerant.measurement_count == 1


def test_one_state_update_per_new_observation(measurement):
    encoder = make_encoder(measurement)
    first = encoder.step(frame(7), None, None, camera_id="cam_a", frame_key=("ep", 0))
    assert encoder.state_update_count == 1  # 出生帧被 update 吸收一次（fresh 支持由 y 初始化）
    second = encoder.step(frame(8), None, first.state, camera_id="cam_a", frame_key=("ep", 1))
    assert encoder.state_update_count == 2
    assert encoder.measurement_count == 2
    assert second.debug["mode_weights"]["cam_a"]["updated"]
    assert second.state is not None
    assert float(second.state.per_camera["cam_a"].age.max()) == 1.0
    # 没有新测量时不吸收额外观测：后验保持 prior。
    third = encoder.step(
        frame(9),
        None,
        second.state,
        camera_id="cam_a",
        frame_key=("ep", 2),
        new_measurement=False,
    )
    assert encoder.state_update_count == 2
    assert encoder.measurement_count == 3
    assert not third.debug["mode_weights"]["cam_a"]["updated"]
    assert third.state is not None
    assert torch.allclose(
        third.state.per_camera["cam_a"].mode_weights,
        third.debug["mode_weights"]["cam_a"]["prior"],
    )


def test_reset_clears_state_and_history(measurement):
    encoder = make_encoder(measurement)
    first = encoder.step(frame(10), None, None, camera_id="cam_a", frame_key=("ep", 0))
    encoder.step(frame(11), None, first.state, camera_id="cam_a", frame_key=("ep", 1))
    assert encoder.step_index == 2
    assert encoder.prev_frame_cameras == ("cam_a",)
    assert len(encoder.support_trajectories["cam_a"]) == 2

    encoder.reset("episode-2")
    assert encoder.step_index == 0
    assert encoder.prev_frame_cameras == ()
    assert encoder.support_trajectories == {}
    # 同一 frame_key 在 reset 后不再算重复；状态从覆盖网格重新出生（fresh、age 0）。
    again = encoder.step(frame(10), None, None, camera_id="cam_a", frame_key=("ep", 0))
    assert again.state is not None
    assert float(again.state.per_camera["cam_a"].age.max()) == 0.0
    assert torch.allclose(  # 出生先验，而不是上一 episode 的后验
        again.state.per_camera["cam_a"].mode_weights,
        encoder.state_stack.birth_weights().expand(1, 128, 2),
        atol=1e-5,
    )
    assert encoder.duplicate_frame_count == 0
    with pytest.raises(ValueError):
        encoder.reset("episode-3", batch_mask=torch.tensor([True, False]))


# ---------------------------------------------------------------------------------------
# 梯度：prefix -> 测量骨干 / 质量头 / 读出 / 状态栈
# ---------------------------------------------------------------------------------------


def test_gradient_reaches_measurement_quality_readout_and_state(measurement):
    encoder = make_encoder(measurement)
    encoder.zero_grad(set_to_none=True)
    proprio = torch.randn(16)
    first = encoder.step(
        frame(12), {"proprio": proprio}, None, camera_id="cam_a", frame_key=("e", 0)
    )
    second = encoder.step(
        frame(13), {"proprio": proprio}, first.state, camera_id="cam_a", frame_key=("e", 1)
    )
    second.prefix_tokens.square().mean().backward()

    def grad(module, name):
        parameter = dict(module.named_parameters())[name]
        assert parameter.grad is not None, name
        return float(parameter.grad.abs().sum())

    assert grad(measurement, "fusion.weight") > 0
    assert grad(encoder.quality_head, "latent_encoder.0.weight") > 0
    assert grad(encoder.state_stack, "control.4.weight") > 0  # 本体条件的加性控制
    assert grad(encoder.state_stack, "revision_q_raw") > 0  # 修订模式额外 Q
    assert grad(encoder.state_stack, "mode_transition_logits") > 0
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in encoder.state_stack.cells.parameters()
    ), "作者 RKN cell 没拿到梯度"
    assert grad(encoder.readout, "to_prefix.weight") > 0
    assert grad(encoder.readout.resampler.blocks[0].attn, "to_q.weight") > 0
    # 没有 no_grad 包住训练路径：测量骨干参数本身还是可训练的。
    assert all(p.requires_grad for p in measurement.parameters())


# ---------------------------------------------------------------------------------------
# 四臂开关与调用方标记
# ---------------------------------------------------------------------------------------


def test_arm_switches(measurement):
    assert {arm: (spec.use_state, spec.use_correspondence) for arm, spec in ARM_SPECS.items()} == {
        "D": (True, True),
        "C": (False, False),
        "B": (True, True),
        "A": (False, False),
    }
    single_c = make_encoder(measurement, arm="C")
    output = single_c.step(frame(14), None, None, camera_id="cam_a", frame_key=("e", 0))
    assert output.state is None
    assert output.prefix_tokens.shape == (1, 256, 2048)
    assert output.debug["support"]["cam_a"]["media_tokens"]["support_tokens"] == 0
    assert output.debug["mode_weights"] == {}
    assert single_c.state_update_count == 0

    for arm in ("A", "B"):
        encoder = make_encoder(measurement, arm=arm)
        result = encoder.step(frame(15), None, None, camera_id="cam_a", frame_key=("e", 0))
        assert result.prefix_tokens.shape == (1, 256, 2048)
        assert (result.state is not None) == ARM_SPECS[arm].use_state

    with pytest.raises(ValueError):
        EncoderV11Config(arm="E")


def test_caller_saturation_and_validity_marks_enter_the_quality_path(measurement):
    plain = make_encoder(measurement)
    marked = make_encoder(measurement)
    saturation = torch.zeros(*SHAPE_HW, dtype=torch.bool)
    saturation[:4, :4] = True
    valid = torch.ones(*SHAPE_HW, dtype=torch.bool)
    valid[:2, :] = False
    first = plain.step(frame(16), None, None, camera_id="cam_a", frame_key=("e", 0))
    second = marked.step(
        frame(16),
        None,
        None,
        camera_id="cam_a",
        frame_key=("e", 0),
        saturation=saturation,
        measurement_valid=valid,
    )
    assert not torch.allclose(
        first.debug["quality"]["cam_a"]["R_diag"], second.debug["quality"]["cam_a"]["R_diag"]
    )


def test_coordinate_helpers_round_trip(measurement):
    xy = torch.tensor([[[0.0, 0.0], [1.5, 3.5], [30.0, 31.0], [15.0, 7.0]]], dtype=torch.float32)
    grid = px_to_grid(xy, SHAPE_HW)
    back = grid_to_px(grid, SHAPE_HW)
    assert torch.allclose(back, xy, atol=1e-4)

    frame_value = MeasurementFrame(
        raw=frame(17),
        camera_id="cam_a",
        measurement_id=build_measurement_id("ep", 0, "cam_a", "rawvla_rgb3"),
        input_mode="rawvla_rgb3",
        scale=input_scale_for("libero_uint8"),
        size_hw=SHAPE_HW,
    )
    batch = preprocess_frames([frame_value], measurement.config)
    sample_grid = build_sample_grid(
        input_mode="rawvla_rgb3",
        sensor_size_hw=SHAPE_HW,
        pad_xy=batch.pad_xy,
        pad_hw=batch.pad_hw,
        stride=measurement.config.measurement.fusion_stride,
    )
    points = PixelPoints(xy=xy, unit="original_px")
    feature_points = sample_grid.sensor_to_feature.apply(points)
    round_trip = sample_grid.feature_to_sensor.apply(feature_points)
    error = (round_trip.xy - xy).abs().max()
    assert float(error) < 0.5  # 契约 §2：往返误差 < 0.5 feature px
    print(
        f"[coords] feature grid {sample_grid.feature_size_hw}, "
        f"round-trip max error {float(error):.3e} px"
    )


def test_cpu_step_cost_and_parameter_counts(measurement):
    """CPU 上的参数计数与一次 step 耗时（记录用，不做性能门槛）。"""
    encoder = make_encoder(measurement)
    total = sum(p.numel() for p in encoder.parameters())
    readout = sum(p.numel() for p in encoder.readout.parameters())
    state = sum(p.numel() for p in encoder.state_stack.parameters())
    quality = sum(p.numel() for p in encoder.quality_head.parameters())
    measurement_params = sum(p.numel() for p in measurement.parameters())
    encoder.step(frame(18), None, None, camera_id="cam_a", frame_key=("e", 0))
    start = time.perf_counter()
    encoder.step(frame(19), None, None, camera_id="cam_a", frame_key=("e", 1))
    seconds = time.perf_counter() - start
    print(
        f"[cost] measurement={measurement_params:,} readout={readout:,} quality={quality:,} "
        f"state={state:,} encoder_total={total:,} one_cpu_step={seconds * 1000:.1f} ms"
    )
    assert readout > 0 and state > 0 and quality > 0


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 CUDA 才能复现 BF16 autocast 下的几何 dtype 边界（远端 LIBERO smoke 现场）",
)
def test_cuda_bfloat16_autocast_keeps_geometry_fp32():
    """BF16 autocast 下 PixelAffine 的 compose（matmul）会变 BF16，与 FP32 的 ``points.xy``
    相遇即 `_require_same_dtype_device` 报错。几何／统计必须是显式 FP32 区。"""
    device = torch.device("cuda")
    network = build_measurement(measurement_config()).to(device).eval()
    encoder = (
        N04V11Encoder(
            network,
            EncoderV11Config(cameras=CAMERAS, raft_iters=1, readout=small_readout_config()),
        )
        .to(device)
        .eval()
    )
    frames = [frame(50).to(device), frame(51).to(device)]
    proprio = torch.randn(16, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first = encoder.step(
            frames, {"proprio": proprio}, None, camera_id=list(CAMERAS), frame_key=("e", 0)
        )
        second = encoder.step(
            [frame(52).to(device), frame(53).to(device)],
            {"proprio": proprio},
            first.state,
            camera_id=list(CAMERAS),
            frame_key=("e", 1),
            compute_auxiliary=True,
        )
        assert second.prefix_tokens.shape == (1, 2 * 256, 2048)
        assert bool(torch.isfinite(second.prefix_tokens).all())
        assert second.prefix_tokens.dtype == torch.bfloat16  # 网络仍走 BF16
        loss = second.losses["correspondence"]
        assert loss.dtype == torch.float32 and bool(torch.isfinite(loss))
        for cam in CAMERAS:
            assert second.debug["flow"][cam].dtype == torch.float32
            assert second.debug["quality"][cam]["R_diag"].dtype == torch.float32
            assert second.debug["support"][cam]["xy_px"].dtype == torch.float32
            assert second.state.per_camera[cam].xy.dtype == torch.float32
            forecast = second.forecast_inputs[cam]
            assert forecast["mode_weights"].dtype == torch.float32
            assert forecast["mode_mean"].dtype == torch.float32
            assert all(block.dtype == torch.float32 for block in forecast["mode_cov"])
        (second.prefix_tokens.float().square().mean() + loss).backward()
    raft = encoder.correspondence.raft
    for name, module in (
        ("fnet", raft.fnet),
        ("cnet", raft.cnet),
        ("update_block", raft.update_block),
    ):
        assert any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
            for parameter in module.parameters()
        ), f"{name} 在 BF16 autocast 下没拿到梯度"


def test_perceiver_matches_upstream_media_plus_latent_reference():
    from owac.frameworks.n04.readout import PerceiverAttention

    torch.manual_seed(19)
    layer = PerceiverAttention(12, dim_head=4, heads=2)
    media, latent = torch.randn(2, 5, 12), torch.randn(2, 3, 12)
    x, z = layer.norm_media(media), layer.norm_latents(latent)
    q = layer.to_q(z).reshape(2, 3, 2, 4).transpose(1, 2)
    k, v = layer.to_kv(torch.cat((x, z), 1)).chunk(2, -1)
    k, v = [t.reshape(2, 8, 2, 4).transpose(1, 2) for t in (k, v)]
    weights = (q @ k.transpose(-1, -2) / 2).softmax(-1)
    expected = layer.to_out((weights @ v).transpose(1, 2).reshape(2, 3, 8))
    actual, attention = layer(latent, media, return_attention=True)
    fused, _ = layer(latent, media, return_attention=False)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(fused, expected, atol=1e-6)
    assert attention.shape[-1] == 8 and float(attention[..., 5:].sum()) > 0


def test_invalid_measurement_cannot_seed_state_and_proprio_cannot_truncate(measurement):
    encoder = make_encoder(measurement)
    raw = frame(21)
    with pytest.raises(ValueError, match="proprio must be finite"):
        encoder.step(raw, torch.zeros(15), camera_id="cam_a")
    encoder.reset()
    output = encoder.step(
        raw, None, camera_id="cam_a", measurement_valid=torch.zeros(SHAPE_HW, dtype=torch.bool)
    )
    state = output.state.per_camera["cam_a"]
    assert bool(state.fresh.all())
    assert not bool(output.debug["evidence"]["cam_a"]["obs_valid"].any())


def test_invalid_dense_media_cannot_affect_readout():
    readout = N04V11Readout(small_readout_config()).eval()
    x = torch.randn(1, 256, 4, 4)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    mask[:, 2:, :] = False
    changed = x.clone()
    changed[:, :, 2:, :] += 100
    a, attention, _ = readout(x, dense_valid=mask)
    b, _, _ = readout(changed, dense_valid=mask)
    assert torch.allclose(a, b, atol=1e-6)
    assert not bool(attention[..., 8:16].any())


def test_missing_proprio_disables_control_bias(measurement):
    encoder = make_encoder(measurement)
    first = encoder.step(frame(31), None, camera_id="cam_a")
    encoder.state_stack.control[-1].bias.data.fill_(100)
    missing = encoder.step(frame(32), None, first.state, camera_id="cam_a")
    prior = missing.forecast_inputs["cam_a"]["mode_mean"]
    encoder.reset()
    first = encoder.step(frame(31), None, camera_id="cam_a")
    encoder.state_stack.control[-1].bias.data.zero_()
    zero = encoder.step(frame(32), None, first.state, camera_id="cam_a")
    assert torch.allclose(prior, zero.forecast_inputs["cam_a"]["mode_mean"], atol=1e-6)


def test_invalid_frame_pair_is_not_a_zero_auxiliary_sample(measurement):
    encoder = make_encoder(measurement)
    invalid = torch.zeros(SHAPE_HW, dtype=torch.bool)
    a = encoder.step(
        frame(41), None, camera_id="cam_a", measurement_valid=invalid, compute_auxiliary=True
    )
    b = encoder.step(
        frame(42),
        None,
        a.state,
        camera_id="cam_a",
        measurement_valid=invalid,
        compute_auxiliary=True,
    )
    assert "correspondence" not in b.losses
    assert encoder._prev_frames["cam_a"].invalid_mask.all()
