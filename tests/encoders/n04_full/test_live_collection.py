"""Corruption checks for the new saved-data contract; no simulator required."""

from __future__ import annotations

import json

import numpy as np
import pytest

from owac.evaluation.n04_live_contracts import temporal_audit, validate_sequence
from owac.raw.n04_live_collection import collect, save_json


class InstrumentedSimulator:
    meta = dict(
        platform="libero",
        task="fixture",
        camera_names=["a", "b"],
        pixel_coordinates="integer_pixel_centres_top_left",
    )

    def __init__(self):
        self.position = 0

    def state(self):
        return np.asarray([self.position], dtype=float)

    def close(self):
        pass

    def capture(self, index):
        return dict(
            online_rawvla_rgb3=np.zeros((2, 2, 2, 3)),
            gt_depth_m=np.ones((2, 2, 2)),
            gt_visible_support=np.ones((2, 2, 2), dtype=bool),
            online_K=np.broadcast_to(np.eye(3), (2, 3, 3)),
            gt_T_W_from_C=np.broadcast_to(np.eye(4), (2, 4, 4)),
            gt_contact_count=np.asarray(1),
            online_sim_time_s=np.asarray(self.position),
            online_qpos=np.zeros(1),
            online_qvel=np.zeros(1),
            gt_sim_qpos=np.zeros(2),
            gt_sim_qvel=np.zeros(2),
            audit_robot_qpos_indices=np.array([0]),
            audit_robot_qvel_indices=np.array([0]),
            audit_ray_matching_geom=np.ones(4, dtype=bool),
            audit_ray_error_m=np.zeros(4),
            audit_ray_bound_m=np.ones(4),
        )

    def step(self, index):
        import time

        start = time.monotonic()
        self.position += 1
        return dict(
            value=[0.1],
            start=start,
            end=time.monotonic(),
            sim_start=self.position - 1,
            sim_end=self.position,
        )


@pytest.fixture
def saved(tmp_path):
    out = tmp_path / "sequence"
    collect(InstrumentedSimulator(), out, frames=12, source_id="fixture", split="train")
    return out / "record.json"


def test_saved_context_and_continuity(saved):
    result = validate_sequence(saved)
    assert result["passed"]
    assert result["temporal"]["out_of_order_pairs"] == 4
    assert result["temporal"]["future_poison_checks"] == 24


def test_trace_corruption_detected(saved):
    record = json.loads(saved.read_text())
    record["controls"][2]["value"] = [10]
    save_json(saved, record)
    assert not validate_sequence(saved)["checks"]["trace"]


def test_early_arrival_detected(saved):
    record = json.loads(saved.read_text())
    record["frames"][1]["host_arrival_s"] = record["frames"][1]["host_sample_s"]
    assert not temporal_audit(record)["passed"]


def test_unverified_control_log_rejected(saved):
    record = json.loads(saved.read_text())
    for control in record["controls"]:
        control["evidence_level"] = "producer_code_order"
    assert not temporal_audit(record)["passed"]


def test_file_corruption_detected(saved):
    with (saved.parent / "frame-0000.npz").open("ab") as stream:
        stream.write(b"changed")
    assert not validate_sequence(saved)["checks"]["hashes"]


def test_extension_loader_never_passes_post_state_or_gt(saved):
    from owac.raw.n04_extension_loader import LiveExtensionSequence

    sequence = LiveExtensionSequence(saved)
    frame = sequence.training_frame(0)
    assert all(k.startswith("online_") for k in frame["inputs"])
    assert not any(k.startswith(("gt_", "trace_")) for k in frame["inputs"])
    first = sequence.record["frames"][0]
    assert 0 not in sequence.arrived_frames(first["host_render_done_s"])
    assert 0 in sequence.arrived_frames(first["host_arrival_s"])


def test_rendered_loader_applies_time_and_calibration_intervention(tmp_path):
    from owac.raw.n04_extension_loader import rendered_training_frame

    file = tmp_path / "factor.npz"
    np.savez(
        file,
        online_rawvla_rgb3=np.arange(3)[:, None],
        input_frame_index=[0, 0, 0],
        input_K=np.full((3, 3, 3), 42.0),
        input_association=[1, 0],
        gt_time_s=[0.0, 0.05, 0.1],
        gt_depth_m=[10.0, 20.0, 30.0],
    )
    sample = rendered_training_frame(file, 2)
    assert sample["inputs"]["raw"].item() == 0
    assert sample["inputs"]["K"][0, 0] == 42.0
    assert sample["targets"]["gt_depth_m"] == 30.0
    assert sample["inputs"]["sample_time_s"] == 0.0


def test_scene_qpos_is_never_online_proprio():
    from types import SimpleNamespace

    from owac.raw.n04_d1_capture import robot_proprio

    robot = SimpleNamespace(
        _ref_joint_pos_indexes=[1],
        _ref_joint_vel_indexes=[1],
        _ref_gripper_joint_pos_indexes=[2],
        _ref_gripper_joint_vel_indexes=[2],
    )
    state = SimpleNamespace(
        qpos=np.array([1e20, 0.1, 0.2, -1e20]), qvel=np.array([1e20, 0.3, 0.4, -1e20])
    )
    env = SimpleNamespace(robots=[robot], sim=SimpleNamespace(data=state))
    result = robot_proprio(env)
    np.testing.assert_array_equal(result["online_qpos"], [0.1, 0.2])
    np.testing.assert_array_equal(result["online_qvel"], [0.3, 0.4])
    assert len(result["gt_sim_qpos"]) == 4


def test_independent_numpy_ray_reference_known_depth():
    from owac.evaluation.n04_factor_contracts import reference_depth

    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 3.0
    k = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]])
    positions = np.array([[0, 0, 0], [0, 0, 0.4], [0.5, 0.4, 0.6], [3, 0, 1.6], [0, 0, -4]])
    depth, geom = reference_depth(pose, k, positions, np.array([[64, 64], [10, 10]]))
    np.testing.assert_allclose(depth, [2.42, 3.0], atol=1e-12)
    np.testing.assert_array_equal(geom, [1, 0])


def test_transparent_rendered_surface_is_not_unique_depth_supervision():
    from types import SimpleNamespace

    from owac.raw.n04_d1_capture import opaque_geometry_ids

    geoms = [
        SimpleNamespace(objid=1, objtype=5, rgba=[1, 1, 1, 1]),
        SimpleNamespace(objid=2, objtype=5, rgba=[0, 0.345, 0.545, 0.1]),
        SimpleNamespace(objid=3, objtype=1, rgba=[1, 1, 1, 1]),
    ]
    context = SimpleNamespace(scn=SimpleNamespace(geoms=geoms, ngeom=3))
    assert opaque_geometry_ids(context) == {1}


def test_drive_layout_distinguishes_angle_and_translation():
    from types import SimpleNamespace

    from owac.raw.n04_live_collection import joint_control_layout

    joints = [
        SimpleNamespace(name="arm", type="revolute"),
        SimpleNamespace(name="finger", type="prismatic"),
    ]
    entity = SimpleNamespace(get_active_joints=lambda: joints)
    layout = joint_control_layout([entity])
    assert layout[0]["position_unit"] == "rad"
    assert layout[1]["position_unit"] == "m"
    assert layout[1]["velocity_unit"] == "m/s"
    assert layout[1]["value_indices"] == [2, 3]


def test_drive_layout_rejects_unknown_joint_coordinates():
    from types import SimpleNamespace

    from owac.raw.n04_live_collection import joint_control_layout

    entity = SimpleNamespace(
        get_active_joints=lambda: [SimpleNamespace(name="unknown", type="ball")]
    )
    with pytest.raises(ValueError, match="unsupported"):
        joint_control_layout([entity])
