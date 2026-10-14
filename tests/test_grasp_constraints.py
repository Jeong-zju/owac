from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from owac.simulation.grasp_constraints import (
    SourceGraspConstraints,
    relative_pose_xyzw,
    validate_schedule,
)


def test_attachment_anchor_uses_actual_relative_pose_without_snapping():
    parent = [1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    child = [1, 2.04, 3.02, 0, 0, 0, 1]
    local, quat = relative_pose_xyzw(parent, child)
    np.testing.assert_allclose(local, [0.04, 0, 0.02], atol=1e-12)
    np.testing.assert_allclose(quat, [0, 0, -np.sqrt(0.5), np.sqrt(0.5)], atol=1e-12)
    assert child == [1, 2.04, 3.02, 0, 0, 0, 1]


def test_attachment_schedule_cannot_skip_a_source_release():
    actions = np.zeros((8, 8), np.float32)
    actions[1:3, 7] = 1
    actions[4:6, 7] = 1
    validate_schedule(actions, [{"close": 1, "open": 3, "object": "cube"}])
    with pytest.raises(ValueError, match="original gripper"):
        validate_schedule(actions, [{"close": 1, "open": 6, "object": "cube"}])
    with pytest.raises(ValueError, match="original gripper"):
        validate_schedule(actions, [{"close": 2, "open": 3, "object": "cube"}])
    with pytest.raises(ValueError, match="readiness"):
        validate_schedule(actions, [{"close": 1, "open": 3, "object": "cube", "not_before": 3}])


def test_duplicate_schedule_and_invalid_poses_are_rejected():
    actions = np.zeros((4, 8))
    actions[1:3, 7] = 1
    cycle = {"close": 1, "open": 3, "object": "cube"}
    with pytest.raises(ValueError, match="duplicate"):
        validate_schedule(actions, [cycle, cycle])
    with pytest.raises(ValueError, match="finite"):
        relative_pose_xyzw([0, 0, float("nan"), 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1])


def test_distant_object_is_not_attached_and_original_open_releases(monkeypatch):
    class Joint:
        def __init__(self):
            self.enabled = False

        def GetJointEnabledAttr(self):
            return SimpleNamespace(Set=lambda value: setattr(self, "enabled", value))

        def __getattr__(self, name):
            return lambda value: None

    monkeypatch.setitem(
        __import__("sys").modules,
        "pxr",
        SimpleNamespace(
            Gf=SimpleNamespace(Vec3f=lambda *x: x, Quatf=lambda *x: x),
            Sdf=SimpleNamespace(ChangeBlock=nullcontext),
        ),
    )
    controller = SourceGraspConstraints.__new__(SourceGraspConstraints)
    controller.specification = {"wait_frames": 8, "maximum_tcp_distance_m": 0.075}
    controller.schedule = [{"close": 1, "open": 4, "object": "cube", "not_before": 2}]
    joint = Joint()
    controller.joints = {"cube": joint}
    controller.collision_filters = {"cube": SimpleNamespace(SetTargets=lambda values: None)}
    controller.gripper_collision_bodies = ["/robot/finger"]
    controller.active = controller.joint_anchor = None
    controller.events, controller.relative_errors, controller.missed = [], [], set()
    parent = [0, 0, 0, 0, 0, 0, 1]
    closed = [0] * 7 + [1]
    controller.update(1, closed, parent, [0, 0, 0], {"cube": [0.04, 0, 0, 0, 0, 0, 1]})
    assert not joint.enabled
    controller.update(2, closed, parent, [0, 0, 0], {"cube": [1, 0, 0, 0, 0, 0, 1]})
    assert not joint.enabled
    pose = [0.04, 0, 0, 0, 0, 0, 1]
    controller.update(3, closed, parent, [0, 0, 0], {"cube": pose})
    assert joint.enabled
    # Both bodies move together; the relative anchor remains unchanged.
    controller.update(
        4, [0] * 8, [0, 0, 1, 0, 0, 0, 1], [0, 0, 1], {"cube": [0.04, 0, 1, 0, 0, 0, 1]}
    )
    summary = controller.summary()
    assert not joint.enabled
    assert summary["attachments"] == summary["releases"] == 1
    assert summary["active_at_end"] is None
    assert summary["maximum_relative_position_error_m"] == 0
    assert summary["object_state_overrides_after_initialization"] == 0
    assert summary["events"][1]["object_pose_before_xyzw"] == pose
