"""N04 本体相对运动测试：语义正确的通道组差分、旋转不可盲目相减、可用性标记。

全部为可手算的小例子，不依赖数据或训练。
"""

from __future__ import annotations

import math

import pytest
import torch

from owac.frameworks.n04.proprio import (
    LIBERO_STATE_LAYOUT,
    ROBOTWIN_JOINT_LAYOUT,
    layout_for_suite,
    proprio_delta,
    quaternion_to_rotvec,
    relative_rotation_vector,
    rotvec_to_quaternion,
)


def _libero_state(
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gripper: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    return torch.tensor([[*position, *rotation, *gripper]], dtype=torch.float32)


def test_identity_state_gives_zero_delta() -> None:
    state = _libero_state(position=(0.3, -0.2, 0.5), rotation=(0.1, 0.2, -0.3), gripper=(0.4, 0.4))
    result = proprio_delta(state, state.clone(), layout=LIBERO_STATE_LAYOUT)
    assert torch.allclose(result.delta, torch.zeros_like(state), atol=1e-6)
    assert bool(result.availability.min() > 0.5)


def test_pure_translation_leaves_rotation_and_gripper_channels_untouched() -> None:
    previous = _libero_state(position=(0.0, 0.0, 0.0), rotation=(0.2, 0.0, 0.0))
    current = _libero_state(position=(0.1, 0.05, -0.02), rotation=(0.2, 0.0, 0.0))
    result = proprio_delta(current, previous, layout=LIBERO_STATE_LAYOUT)
    assert torch.allclose(result.delta[0, :3], torch.tensor([0.1, 0.05, -0.02]), atol=1e-6)
    assert torch.allclose(result.delta[0, 3:], torch.zeros(5), atol=1e-6)


def test_rotation_delta_is_relative_rotation_not_subtraction() -> None:
    """绕 z 轴 90 度的相对旋转应为 (0, 0, pi/2)，而不是 45 度或逐元素差。"""
    previous = _libero_state(rotation=(0.0, 0.0, math.radians(30.0)))
    current = _libero_state(rotation=(0.0, 0.0, math.radians(120.0)))
    result = proprio_delta(current, previous, layout=LIBERO_STATE_LAYOUT)
    expected = torch.tensor([0.0, 0.0, math.pi / 2], dtype=torch.float32)
    assert torch.allclose(result.delta[0, 3:6], expected, atol=1e-5)


def test_rotation_delta_takes_short_arc_across_wrap_around() -> None:
    """+170 度到 -170 度是 20 度短弧，不是 340 度。"""
    previous = _libero_state(rotation=(0.0, 0.0, math.radians(170.0)))
    current = _libero_state(rotation=(0.0, 0.0, math.radians(-170.0)))
    result = proprio_delta(current, previous, layout=LIBERO_STATE_LAYOUT)
    angle = float(torch.linalg.vector_norm(result.delta[0, 3:6]))
    assert angle == pytest.approx(math.radians(20.0), abs=1e-4)
    #: 逐元素相减会给出 -340 度，这里必须完全不同。
    naive = float((current - previous)[0, 5].abs())
    assert naive > math.pi


def test_relative_rotation_is_composable_and_invertible() -> None:
    a = torch.tensor([[0.1, -0.2, 0.3]])
    b = torch.tensor([[0.4, 0.1, -0.2]])
    forward = relative_rotation_vector(b, a)
    backward = relative_rotation_vector(a, b)
    #: 相对旋转的逆应当抵消（模 2pi 的短弧下两者等长反向）。
    assert torch.allclose(forward, -backward, atol=1e-5)
    identity = relative_rotation_vector(a, a)
    assert torch.allclose(identity, torch.zeros_like(a), atol=1e-6)


def test_quaternion_roundtrip_for_small_and_large_angles() -> None:
    rotvec = torch.tensor([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [0.0, math.pi * 0.9, 0.0]])
    quaternion = rotvec_to_quaternion(rotvec)
    assert torch.allclose(torch.linalg.vector_norm(quaternion, dim=-1), torch.ones(3), atol=1e-6)
    restored = quaternion_to_rotvec(quaternion)
    assert torch.allclose(restored, rotvec, atol=1e-6)


def test_missing_previous_state_is_unavailable_not_zero_motion() -> None:
    state = _libero_state(position=(0.2, 0.1, 0.0))
    result = proprio_delta(state, None, layout=LIBERO_STATE_LAYOUT)
    assert torch.allclose(result.delta, torch.zeros_like(state))
    assert float(result.availability.max()) == 0.0
    assert result.info is not None and result.info["reason"] == "no_previous_state"


def test_unavailable_proprio_zeroes_delta_and_marks_all_groups() -> None:
    previous = torch.zeros(2, 8)
    current = torch.ones(2, 8)
    result = proprio_delta(
        current, previous, layout=LIBERO_STATE_LAYOUT, available=torch.tensor([True, False])
    )
    assert torch.allclose(result.delta[0], torch.ones(8))
    assert torch.allclose(result.delta[1], torch.zeros(8))
    assert bool(result.availability[0].min() > 0.5)
    assert float(result.availability[1].max()) == 0.0


def test_robotwin_layout_keeps_arms_and_grippers_separate() -> None:
    previous = torch.zeros(1, 14)
    current = torch.zeros(1, 14)
    current[0, 0] = 0.3  # 左臂关节
    current[0, 6] = 0.9  # 左夹爪
    current[0, 13] = -0.5  # 右夹爪
    result = proprio_delta(current, previous, layout=ROBOTWIN_JOINT_LAYOUT)
    assert torch.allclose(result.delta[0, :6], torch.tensor([0.3, 0, 0, 0, 0, 0]))
    assert float(result.delta[0, 6]) == pytest.approx(0.9)
    assert torch.allclose(result.delta[0, 7:13], torch.zeros(6))
    assert float(result.delta[0, 13]) == pytest.approx(-0.5)
    assert result.group_available(ROBOTWIN_JOINT_LAYOUT, "left_gripper")
    assert result.group_available(ROBOTWIN_JOINT_LAYOUT, "right_arm")


def test_layout_width_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="通道组"):
        proprio_delta(torch.zeros(1, 5), torch.zeros(1, 5), layout=ROBOTWIN_JOINT_LAYOUT)


def test_layout_for_unknown_suite_raises() -> None:
    with pytest.raises(ValueError, match="未知套件"):
        layout_for_suite("unknown-suite")


def test_frame_time_unit_is_not_fabricated_as_seconds() -> None:
    state = _libero_state()
    result = proprio_delta(state, state.clone(), layout=LIBERO_STATE_LAYOUT)
    assert result.time_unit == "frame"
    assert result.dt is None
