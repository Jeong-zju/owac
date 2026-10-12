"""N04 合成测量场的信息边界与条件生效性测试。

覆盖三个由实际机制评测暴露的真实缺陷：

1. ``frame.proprio`` 曾把**物体位移**放进自身运动向量（捷径泄漏）；
2. ``occlusion`` 条件只写 ``valid_mask``，而编码器当时根本没有接收测量有效性；
3. 评测场景没有注入 ``calibration_scale`` / ``action_delay``，两个条件退化成 camera_only，
   机制矩阵里出现整行逐位相同的数字。
"""

from __future__ import annotations

import torch

from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig
from owac.frameworks.n04.model import N04Encoder
from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from owac.training.n04 import build_context

HEIGHT = WIDTH = 32


def _scene(**overrides: object) -> SceneConfig:
    base = dict(
        height=HEIGHT,
        width=WIDTH,
        input_mode="rawvla_rgb3",
        camera_translation=(1.2, 1.2),
        camera_rotation=(0.0, 0.4),
        object_velocity=(0.8, 1.2),
        illumination=(0.7, 1.3),
        noise_sigma=0.01,
        seed=0,
    )
    base.update(overrides)
    return SceneConfig(**base)  # type: ignore[arg-type]


def _sequence(condition: str, **overrides: object):
    return SyntheticSequenceGenerator(_scene(**overrides)).sequence(
        batch=1, length=8, condition=condition, seed=0
    )


def _encoder() -> N04Encoder:
    cfg = N04Config(
        input_mode="rawvla_rgb3",
        measurement_size=(HEIGHT, WIDTH),
        channels=3,
        camera_ids=("head",),
        geometry=N04GeometryConfig(feature_size=(HEIGHT, WIDTH), pixel_stride=1.0, max_flow=4.0),
        model=N04ModelConfig(
            name="n04-synthetic-test",
            width=8,
            blocks=1,
            scales=1,
            scale_strides=(1,),
            coarse_stride=4,
            hidden=16,
            camera_embedding=4,
            readout_tokens=8,
            readout_dim=16,
        ),
    )
    torch.manual_seed(0)
    return N04Encoder(cfg).eval()


# -- 1. 自身运动向量不含外部运动 -------------------------------------------------------


def test_proprio_contains_no_object_motion() -> None:
    """物体运动不能出现在本体/已执行动作里：那是 ego 预测的捷径。"""
    object_only = _sequence("object_only")
    for frame in object_only:
        assert float(frame.proprio.abs().max()) == 0.0
        assert float(frame.action.abs().max()) == 0.0
    both = _sequence("both")
    camera_only = _sequence("camera_only")
    for left, right in zip(both, camera_only, strict=True):
        assert torch.allclose(left.proprio, right.proprio, atol=1e-6)
    #: 相反方向必须成立：相机运动确实进入 proprio，否则上面的断言会假通过。
    assert float(camera_only[-1].proprio.abs().max()) > 0.1


def test_proprio_dimension_is_self_motion_only() -> None:
    frame = _sequence("camera_only")[0]
    #: 相机平移(2) + 相机旋转(1)。加入物体位移会让它变成 4 维。
    assert frame.proprio.shape[-1] == 3
    assert frame.action.shape[-1] == 3


# -- 2. 遮挡：测量掩码必须真的能进入编码器 ---------------------------------------------


def test_occlusion_condition_masks_validity_without_changing_measurement() -> None:
    occlusion = _sequence("occlusion")
    camera_only = _sequence("camera_only")
    masked = [float(f.valid_mask.min()) for f in occlusion]
    assert min(masked) < 0.5, "遮挡条件必须真的产生无测量支持的区域"
    assert all(float(f.valid_mask.min()) == 1.0 for f in camera_only)
    #: 设计如此：遮挡只表达“没有测量支持”，不伪造像素内容；因此必须靠掩码传入编码器。
    for left, right in zip(occlusion, camera_only, strict=True):
        assert torch.allclose(left.measured, right.measured, atol=1e-6)


def test_measurement_validity_blocks_update_in_occluded_region() -> None:
    """遮挡区域的**内容**不能进入状态更新。

    在 ``correct_after_observe`` 这一层检验（直接给特征场），避免 stem 的
    ``GroupNorm(1, C)`` 全图统计把“输入像素被改”放大成“所有像素的特征都变”。
    判据是**泄漏比**：block 内仍有全局归一化，所以掩码把污染影响压低而不是归零。
    """
    encoder = _encoder()
    width = encoder.config.model.width
    region = (slice(None), slice(None), slice(0, HEIGHT // 2), slice(0, WIDTH // 2))
    torch.manual_seed(0)
    prediction = torch.randn(1, width, HEIGHT, WIDTH)
    clean = torch.randn(1, width, HEIGHT, WIDTH)
    corrupted = clean.clone()
    corrupted[:, :, : HEIGHT // 2, : WIDTH // 2] += 5.0
    valid = torch.ones(1, 1, HEIGHT, WIDTH)
    uncertainty = torch.zeros(1, 1, HEIGHT, WIDTH)
    mask = torch.ones(1, 1, HEIGHT, WIDTH)
    mask[:, :, : HEIGHT // 2, : WIDTH // 2] = 0.0

    def _stable(value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        return value[0] if isinstance(value, list) else value

    with torch.no_grad():
        state = encoder.init_state(1)["head"]
        masked_clean, _ = encoder.correct_after_observe(
            state, prediction, clean, valid, uncertainty, measurement_valid=mask
        )
        masked_corrupt, _ = encoder.correct_after_observe(
            state, prediction, corrupted, valid, uncertainty, measurement_valid=mask
        )
        plain_clean, _ = encoder.correct_after_observe(state, prediction, clean, valid, uncertainty)
        plain_corrupt, _ = encoder.correct_after_observe(
            state, prediction, corrupted, valid, uncertainty
        )
    masked_leak = float(
        (_stable(masked_clean.stable) - _stable(masked_corrupt.stable))[region].abs().max()
    )
    plain_leak = float(
        (_stable(plain_clean.stable) - _stable(plain_corrupt.stable))[region].abs().max()
    )
    assert plain_leak > 0.1, "反例必须真的泄漏，否则本测试没有分辨力"
    assert masked_leak < 0.05 * plain_leak, f"掩码未断开通路：{masked_leak} vs {plain_leak}"
    #: change cell 只由掩码后的残差写入，必须严格不变。
    assert torch.allclose(masked_clean.change[region], masked_corrupt.change[region], atol=1e-6)


def test_step_accepts_occlusion_mask_end_to_end() -> None:
    """整条 ``step`` 必须真的接收测量掩码：遮挡帧内容被替换后，状态改变量显著变小。"""
    encoder = _encoder()
    frames = _sequence("occlusion")
    #: 遮挡按 ``(t // 3) % 2`` 变化，必须挑一个真的被遮挡、且有前置状态的帧。
    index = next(
        i for i, frame in enumerate(frames) if i > 0 and float(frame.valid_mask.min()) < 0.5
    )
    #: 污染**确实被遮挡**的行（遮挡图案为 rows [H/4, H/2)，即这里 H=32 的第 16—31 行）。
    occluded = frames[index].valid_mask[0, 0] < 0.5
    rows = torch.nonzero(occluded.any(dim=1)).flatten()
    top, bottom = int(rows.min()), int(rows.max()) + 1
    corrupt_rows = slice(top, bottom)
    #: 比较区避开掩码边界的 2 像素（3x3 卷积的邻域影响）。
    compare_rows = slice(top + 2, bottom)
    region = (slice(None), slice(None), compare_rows, slice(None))
    with torch.no_grad():
        state = encoder.init_state(1)["head"]
        previous = None
        for frame in frames[:index]:
            state = encoder.step(
                frame.measured, build_context(frame, previous), state, camera_id="head"
            ).state
            previous = frame
        context = build_context(frames[index], previous)
        corrupted = frames[index].measured.clone()
        corrupted[:, :, corrupt_rows, :] += 5.0
        good = encoder.step(
            frames[index].measured,
            context,
            state,
            camera_id="head",
            measurement_valid=frames[index].valid_mask,
        )
        bad = encoder.step(
            corrupted,
            context,
            state,
            camera_id="head",
            measurement_valid=frames[index].valid_mask,
        )
        plain_good = encoder.step(frames[index].measured, context, state, camera_id="head")
        plain_bad = encoder.step(corrupted, context, state, camera_id="head")

    def _stable(value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        return value[0] if isinstance(value, list) else value

    masked_leak = float(
        (_stable(good.state.stable) - _stable(bad.state.stable))[region].abs().max()
    )
    plain_leak = float(
        (_stable(plain_good.state.stable) - _stable(plain_bad.state.stable))[region].abs().max()
    )
    assert plain_leak > 0.1
    #: 掩码必须确实降低了遮挡内容的影响，并确实改变了更新结果。
    assert masked_leak < plain_leak, f"step 未使用掩码：{masked_leak} vs {plain_leak}"
    assert not torch.allclose(
        _stable(good.state.stable)[region], _stable(plain_good.state.stable)[region], atol=1e-6
    )
    #: 说明：这里做不到严格不变，因为 stem 的 ``GroupNorm(1, C)`` 是全图统计——污染整行
    #: 会改变全局均值/方差，从而影响所有像素。严格的门控语义由上面的
    #: ``correct_after_observe`` 单元测试固定（泄漏 <5%）。


# -- 3. 条件参数必须真的生效 -----------------------------------------------------------


def test_calibration_error_and_action_delay_change_reported_motion() -> None:
    scene_args = dict(calibration_scale=1.08, action_delay=2)
    camera_only = _sequence("camera_only", **scene_args)
    calibration = _sequence("calibration_error", **scene_args)
    delayed = _sequence("action_delay", **scene_args)
    base_delta = (camera_only[-1].proprio - camera_only[0].proprio).abs().max()
    assert float((calibration[-1].proprio - camera_only[-1].proprio).abs().max()) > 1e-3
    assert float((delayed[-1].proprio - camera_only[-1].proprio).abs().max()) > 1e-3
    #: 标定误差只改“上报的运动”，不改真实观测：这正是 H3 要检验的失配。
    assert torch.allclose(calibration[-1].measured, camera_only[-1].measured, atol=1e-6)
    assert float(base_delta) > 0.1


def test_scene_without_calibration_scale_makes_condition_a_noop() -> None:
    """反例：不注入参数时 `calibration_error` 与 `camera_only` 完全相同。

    这条测试固定住“评测必须在场景里显式传入 scale/delay”的要求：默认值 1.0/0 会让
    整个条件悄悄退化成相机运动条件。
    """
    camera_only = _sequence("camera_only")
    calibration = _sequence("calibration_error")
    for left, right in zip(calibration, camera_only, strict=True):
        assert torch.allclose(left.proprio, right.proprio, atol=1e-6)
        assert torch.allclose(left.measured, right.measured, atol=1e-6)
