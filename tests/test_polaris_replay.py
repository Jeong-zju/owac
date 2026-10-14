import copy
import json
import math

import numpy as np
import pytest

from owac.runtime.polaris_replay import joint_transform, transform
from owac.runtime.polaris_replay_archive import verify_frames
from owac.runtime.polaris_replay_data import fields, varint
from owac.simulation.polaris_replay import evaluate


def test_joint_anchor_rotation_matches_known_child_origin():
    parent = transform((1, 0, 0), (1, 0, 0, 0))
    child = transform((0, 1, 0), (1, 0, 0, 0))
    result = joint_transform(parent, child, "Z", math.pi / 2)
    np.testing.assert_allclose(result @ [0, 0, 0, 1], [2, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(result @ [0, 1, 0, 1], [1, 0, 0, 1], atol=1e-12)


def test_truncated_source_wire_data_is_rejected():
    with pytest.raises(ValueError, match="Truncated protobuf varint"):
        varint(b"\x80", 0)
    with pytest.raises(ValueError, match="Truncated protobuf field"):
        list(fields(b"\x0a\x04xy"))


def fixture_rows():
    config = {
        "cube_size": 0.04,
        "target_center": [0.5, 0.25, 0.1],
        "bin_inner_half_extent": 0.11,
        "success": {
            "stable_last_frames": 15,
            "target_margin_m": 0.02,
            "lift_minimum_m": 0.05,
            "transport_minimum_m": 0.1,
            "released_tcp_separation_m": 0.05,
            "stable_maximum_speed_m_s": 0.05,
        },
    }
    rows = []
    for index in range(40):
        position = [0.5, -0.25, 0.12] if index == 0 else [0.5, 0.25, 0.13]
        if 5 <= index < 15:
            position = [0.5, 0, 0.3]
        rows.append(
            {
                "source_frame": index,
                "cube_position": position,
                "cube_speed": 0,
                "tcp_position": position if index < 20 else [0.5, 0.25, 0.4],
                "original_action": [0] * 7 + [int(index < 20)],
                "actual_gripper_joint": 0.7 if index < 20 else 0,
                "source_joint_mae_rad": 0.01,
            }
        )
    return rows, config


def test_task_success_requires_a_sustained_physical_grasp():
    rows, config = fixture_rows()
    assert evaluate(rows, config)["success"]
    for row in rows:
        row["tcp_position"] = [3, 3, 3]
    result = evaluate(rows, config)
    assert result["checks"]["inside_target_bin"]
    assert result["checks"]["lifted"]
    assert not result["success"]
    assert not result["checks"]["sustained_grasp"]


def test_bin_occupancy_with_closed_gripper_does_not_pass_release():
    rows, config = fixture_rows()
    rows[-1]["actual_gripper_joint"] = 0.7
    result = evaluate(rows, config)
    assert not result["checks"]["released"]
    assert not result["success"]


def test_one_successful_object_does_not_complete_a_multi_object_source_task():
    rows, config = fixture_rows()
    assert evaluate(rows, config)["success"]
    config["source_multi_object_task_incomplete"] = True
    result = evaluate(rows, config)
    assert result["checks"]["inside_target_bin"]
    assert not result["checks"]["all_source_task_objects_reconstructed"]
    assert not result["success"]


def test_multi_object_reconstruction_requires_every_object_to_complete():
    rows, config = fixture_rows()
    for row in rows:
        row["objects"] = {
            name: {
                "position": row["cube_position"].copy(),
                "quaternion_xyzw": [0, 0, 0, 1],
                "speed": 0,
            }
            for name in ("cube", "other")
        }
    config.update(
        reconstructed_objects=[{"name": name, "size": [0.04] * 3} for name in ("cube", "other")],
        source_object_count_visually_reviewed=True,
    )
    assert evaluate(rows, config)["success"]
    for row in rows:
        row["objects"]["other"]["position"] = [0.5, -0.25, 0.12]
    result = evaluate(rows, config)
    assert result["checks"]["object_cube_task_complete"]
    assert not result["checks"]["object_other_task_complete"]
    assert not result["success"]


def test_rotation_result_serializes_and_rejects_wrong_angle():
    rows, config = fixture_rows()
    config.update(
        task_kind="rotate", success_rotation_target_deg=90, success_rotation_tolerance_deg=25
    )
    for row in rows:
        theta = 0 if row is rows[0] else math.pi / 2
        row["cube_quaternion_xyzw"] = [0, 0, math.sin(theta / 2), math.cos(theta / 2)]
    result = evaluate(rows, config)
    assert json.loads(json.dumps(result))["success"]
    for row in rows[-15:]:
        row["cube_quaternion_xyzw"] = [0, 0, 1, 0]
    assert not evaluate(rows, config)["checks"]["rotated_90_degrees"]


def capture_fixture():
    config = {"frames": 2, "decimation": 8, "camera_resolution": [320, 180]}
    rows = []
    for frame in range(2):
        view = {
            "native_frame_id": frame + 1,
            "cfa_quantization_maximum_error_dn": 0,
            "dtype": "uint32",
            "shape": [180, 320],
            "paired_rgb": {"shape": [180, 320, 3]},
            "position_world": [frame * 0.1 + 0.011, -0.031, -0.074],
            "quaternion_opengl_xyzw": [0.570, 0.576, -0.409, -0.420],
        }
        rows.append(
            {
                "source_frame": frame,
                "original_action": [0.0] * 8,
                "gripper_base_quaternion_xyzw": [0, 0, 0, 1],
                "gripper_base_position": [frame * 0.1, 0, 0],
                "capture": {
                    "step": frame,
                    "physical_step_counter": frame,
                    "physics_substep_counter": frame * 8,
                    "views": {
                        name: copy.deepcopy(view) for name in ("external_camera", "wrist_camera")
                    },
                },
            }
        )
    result = {
        "source_frames": 2,
        "native_capture": {"control_frames": 2},
        "verified_file_count": 16,
        "checksum_mismatches": [],
        "object_state_overrides_after_initialization": 0,
        "original_action_changed": False,
    }
    return rows, result, config


def test_native_camera_must_follow_actual_wrist_motion():
    rows, result, config = capture_fixture()
    assert verify_frames(rows, result, config)["valid"]
    rows[1]["capture"]["views"]["wrist_camera"]["position_world"][0] = 0.011
    with pytest.raises(ValueError, match="stale or inconsistent"):
        verify_frames(rows, result, config)


def test_extra_physics_steps_during_capture_invalidate_expert_timing():
    rows, result, config = capture_fixture()
    rows[1]["capture"]["physics_substep_counter"] = 16
    with pytest.raises(ValueError, match="physical control timing"):
        verify_frames(rows, result, config)
