"""N04 相对旋转参考系测试：世界系增量 ``R_t R_prev^T`` 与上一体坐标系增量 ``R_prev^T R_t``。

手算反例（全部输入为 ``R_W_from_B`` 轴角）：

* ``R_prev = Rz(90°)``，轴角 ``(0, 0, pi/2)``；
* ``R_t = Ry(90°) Rz(90°)``，等价于绕 ``(1,1,1)/√3`` 转 120°，轴角 ``(c, c, c)``，
  ``c = 2 pi / (3 sqrt(3))``。

于是 世界系增量 ``R_t R_prev^T = Ry(90°)`` -> ``(0, pi/2, 0)``；
上一体坐标系增量 ``R_prev^T R_t = Rz(-90°) Ry(90°) Rz(90°) = Rx(90°)`` -> ``(pi/2, 0, 0)``。
测试自带的 Rodrigues 实现用于独立复核旋转矩阵，不复用被测代码。
"""

from __future__ import annotations

import math

import pytest
import torch

from owac.frameworks.n04.proprio import (
    LIBERO_STATE_LAYOUT,
    relative_rotation_increment,
    relative_rotation_vector,
)

FP64 = torch.float64
_C120 = 2.0 * math.pi / (3.0 * math.sqrt(3.0))


def _rodrigues(rotvec: torch.Tensor) -> torch.Tensor:
    """测试内独立 Rodrigues 实现：``[...,3]`` 轴角 -> ``[...,3,3]`` 旋转矩阵。"""
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    safe = torch.where(theta < 1e-12, torch.ones_like(theta), theta)
    axis = rotvec / safe
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(
        *rotvec.shape[:-1], 3, 3
    )
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand_as(skew)
    return (
        identity
        + torch.sin(theta).unsqueeze(-1) * skew
        + (1.0 - torch.cos(theta)).unsqueeze(-1) * (skew @ skew)
    )


def _rot_x(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=FP64)


def _rot_y(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=FP64)


def _rot_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=FP64)


def _hand_case() -> tuple[torch.Tensor, torch.Tensor]:
    previous = torch.tensor([[0.0, 0.0, math.pi / 2]], dtype=FP64)
    current = torch.tensor([[_C120, _C120, _C120]], dtype=FP64)
    return current, previous


def test_hand_case_premise_matches_matrix_product() -> None:
    """前提：Rz(90°) 与 Ry(90°)Rz(90°) 的轴角表示与手算矩阵一致。"""
    current, previous = _hand_case()
    assert torch.allclose(_rodrigues(previous), _rot_z(math.pi / 2), atol=1e-9)
    expected = _rot_y(math.pi / 2) @ _rot_z(math.pi / 2)
    assert torch.allclose(_rodrigues(current), expected, atol=1e-9)


def test_world_increment_is_current_times_previous_transpose() -> None:
    current, previous = _hand_case()
    world = relative_rotation_increment(current, previous, frame="world")
    assert torch.allclose(world, torch.tensor([[0.0, math.pi / 2, 0.0]], dtype=FP64), atol=1e-9)
    #: 独立矩阵复核：world = R_t R_prev^T = Ry(90°)。
    assert torch.allclose(
        _rodrigues(world), _rodrigues(current) @ _rodrigues(previous).transpose(-1, -2), atol=1e-9
    )
    #: 复合关系 R_t = ΔR_W R_prev。
    assert torch.allclose(_rodrigues(world) @ _rodrigues(previous), _rodrigues(current), atol=1e-9)


def test_body_increment_is_previous_transpose_times_current() -> None:
    current, previous = _hand_case()
    body = relative_rotation_increment(current, previous, frame="body")
    assert torch.allclose(body, torch.tensor([[math.pi / 2, 0.0, 0.0]], dtype=FP64), atol=1e-9)
    #: 独立矩阵复核：body = R_prev^T R_t = Rz(-90°) Ry(90°) Rz(90°) = Rx(90°)。
    assert torch.allclose(
        _rodrigues(body), _rodrigues(previous).transpose(-1, -2) @ _rodrigues(current), atol=1e-9
    )
    assert torch.allclose(_rodrigues(body), _rot_x(math.pi / 2), atol=1e-9)
    #: 复合关系 R_t = R_prev ΔR_B。
    assert torch.allclose(_rodrigues(previous) @ _rodrigues(body), _rodrigues(current), atol=1e-9)


def test_world_and_body_increments_differ_when_rotations_do_not_commute() -> None:
    current, previous = _hand_case()
    world = relative_rotation_increment(current, previous, frame="world")
    body = relative_rotation_increment(current, previous, frame="body")
    assert not torch.allclose(world, body, atol=1e-3)
    assert float(torch.linalg.vector_norm(world)) == pytest.approx(math.pi / 2, abs=1e-9)
    assert float(torch.linalg.vector_norm(body)) == pytest.approx(math.pi / 2, abs=1e-9)


def test_legacy_function_keeps_world_increment_numbers() -> None:
    """旧接口数值行为保留：等于世界系增量，不等于体坐标系增量。"""
    current, previous = _hand_case()
    legacy = relative_rotation_vector(current, previous)
    world = relative_rotation_increment(current, previous, frame="world")
    body = relative_rotation_increment(current, previous, frame="body")
    assert torch.allclose(legacy, world, atol=0.0)
    assert not torch.allclose(legacy, body, atol=1e-3)
    assert torch.allclose(legacy, torch.tensor([[0.0, math.pi / 2, 0.0]], dtype=FP64), atol=1e-9)


def test_increments_are_inverses_of_the_reverse_direction() -> None:
    current, previous = _hand_case()
    for frame in ("world", "body"):
        forward = relative_rotation_increment(current, previous, frame=frame)
        backward = relative_rotation_increment(previous, current, frame=frame)
        product = _rodrigues(forward) @ _rodrigues(backward)
        assert torch.allclose(product, torch.eye(3, dtype=FP64), atol=1e-9)


def test_seeded_random_pairs_satisfy_both_composition_identities() -> None:
    generator = torch.Generator().manual_seed(20261014)
    previous = (torch.rand(4, 3, generator=generator, dtype=FP64) - 0.5) * 1.5
    current = (torch.rand(4, 3, generator=generator, dtype=FP64) - 0.5) * 1.5
    world = relative_rotation_increment(current, previous, frame="world")
    body = relative_rotation_increment(current, previous, frame="body")
    R_previous = _rodrigues(previous)
    R_current = _rodrigues(current)
    assert torch.allclose(_rodrigues(world) @ R_previous, R_current, atol=1e-9)
    assert torch.allclose(R_previous @ _rodrigues(body), R_current, atol=1e-9)
    assert not torch.allclose(world, body, atol=1e-3)


def test_unknown_frame_or_convention_raises_instead_of_guessing() -> None:
    current, previous = _hand_case()
    with pytest.raises(ValueError, match="参考系"):
        relative_rotation_increment(current, previous, frame="camera")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="W_from_B"):
        relative_rotation_increment(
            current,
            previous,
            frame="world",
            rotation_convention="B_from_W",  # type: ignore[arg-type]
        )


def _libero_state(rotation: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, 0.0, *rotation, 0.0, 0.0]], dtype=torch.float32)


def test_proprio_delta_default_keeps_legacy_world_numbers() -> None:
    previous = _libero_state((0.0, 0.0, math.pi / 2))
    current = _libero_state((_C120, _C120, _C120))
    from owac.frameworks.n04.proprio import proprio_delta

    default = proprio_delta(current, previous, layout=LIBERO_STATE_LAYOUT)
    explicit_world = proprio_delta(
        current, previous, layout=LIBERO_STATE_LAYOUT, rotation_frame="world"
    )
    body = proprio_delta(current, previous, layout=LIBERO_STATE_LAYOUT, rotation_frame="body")
    legacy = relative_rotation_vector(current[:, 3:6], previous[:, 3:6])
    assert torch.allclose(default.delta[:, 3:6], legacy, atol=1e-6)
    assert torch.equal(default.delta, explicit_world.delta)
    assert torch.allclose(default.delta[0, 3:6], torch.tensor([0.0, math.pi / 2, 0.0]), atol=1e-6)
    assert torch.allclose(body.delta[0, 3:6], torch.tensor([math.pi / 2, 0.0, 0.0]), atol=1e-6)
    assert not torch.allclose(default.delta[:, 3:6], body.delta[:, 3:6], atol=1e-3)
    assert default.info is not None and default.info["rotation_frame"] == "world"
    assert body.info is not None and body.info["rotation_frame"] == "body"
    #: 线性通道（位置/夹爪）不受旋转参考系选择影响。
    assert torch.equal(default.delta[:, :3], body.delta[:, :3])
    assert torch.equal(default.delta[:, 6:], body.delta[:, 6:])


def test_missing_previous_state_reports_frame_and_stays_unavailable() -> None:
    from owac.frameworks.n04.proprio import proprio_delta

    state = _libero_state((0.1, 0.2, 0.3))
    result = proprio_delta(state, None, layout=LIBERO_STATE_LAYOUT, rotation_frame="body")
    assert float(result.availability.max()) == 0.0
    assert torch.allclose(result.delta, torch.zeros_like(state))
    assert result.info is not None
    assert result.info["rotation_frame"] == "body"
    assert result.info["reason"] == "no_previous_state"
