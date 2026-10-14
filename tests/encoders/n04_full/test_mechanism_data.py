"""D0 mechanism data: per-sample labels, CFA sampling, and the independent label audit.

These tests exercise the S0 D0 contract:
  * image, depth, labels and events come from one per-sample world state;
  * the audit recomputes labels through an independent projection path;
  * the historical ``[B,1,2] + [B,2] -> [B,B,2]`` label defect is detected;
  * the CFA branch samples the declared channel and phase from linear radiance;
  * D1 refuses a sidecar when frame correspondence is not verified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.raw.n04_mechanism_data import (
    CFA_PHASE_CHANNELS,
    D1_TRACE_CONTROLLER_BUFFER_POLICIES,
    D1_TRACE_STEP_FIELDS,
    D1_TRACE_WARMSTART_MODES,
    DEFAULT_COMBINATIONS,
    DEFAULT_FACTORS,
    BodyTrack,
    D0Config,
    PinholeCamera,
    PlaneBody,
    WorldBatch,
    _audit_world,
    _buggy_merged_labels,
    _render_factor,
    apply_cfa,
    audit_d0,
    build_factor_worlds,
    build_feasibility,
    cfa_phase_channel_map,
    compare_label_maps,
    d1_npz_field_probe,
    d1_plan,
    d1_recomputability_report,
    d1_trace_schema,
    d1_write_trace,
    default_cameras,
    default_factor_names,
    form_measurement,
    image_label_consistency,
    motion_phase,
    project_points,
    reference_visibility,
    render_world,
    resolve_plan,
    run_cli,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
S0_CONFIG = REPO_ROOT / "configs" / "n04" / "full" / "s0_data.json"


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinhole_camera(width: int = 64, height: int = 48) -> PinholeCamera:
    return PinholeCamera("test", 100.0, 100.0, (width - 1) / 2.0, (height - 1) / 2.0, width, height)


def plane_world(
    camera: PinholeCamera,
    *,
    near_depth: float = 0.5,
    far_depth: float = 1.0,
    near_size: float = 0.10,
    far_size: float = 0.60,
    translate_near: tuple[float, float] = (0.0, 0.0),
) -> WorldBatch:
    rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float64
    ).reshape(1, 3, 3)
    far = torch.zeros(1, 4, 4, dtype=torch.float64)
    far[:, :3, :3] = rotation
    far[:, :3, 3] = torch.tensor([[0.0, 0.0, far_depth]], dtype=torch.float64)
    far[:, 3, 3] = 1.0
    near = torch.zeros(1, 4, 4, dtype=torch.float64)
    near[:, :3, :3] = rotation
    near[:, :3, 3] = torch.tensor(
        [[translate_near[0], translate_near[1], near_depth]], dtype=torch.float64
    )
    near[:, 3, 3] = 1.0
    return WorldBatch(
        cameras=(camera,),
        T_world_from_cam={
            camera.name: torch.eye(4, dtype=torch.float64).reshape(1, 4, 4),
        },
        bodies=(
            BodyTrack(
                body=PlaneBody("near", (near_size, near_size), (1.0, 0.2, 0.2), "flat"),
                T_world_from_body=near,
                present=torch.ones(1, dtype=torch.bool),
            ),
            BodyTrack(
                body=PlaneBody("far", (far_size, far_size), (0.2, 0.2, 1.0), "flat"),
                T_world_from_body=far,
                present=torch.ones(1, dtype=torch.bool),
            ),
        ),
        factor="test",
    )


def test_pinhole_projection_matches_hand_calculation():
    camera = pinhole_camera(64, 48)
    assert camera.matrix().tolist() == [
        [100.0, 0.0, 31.5],
        [0.0, 100.0, 23.5],
        [0.0, 0.0, 1.0],
    ]
    rays = camera.ray_directions()
    assert rays.shape == (48, 64, 3)
    torch.testing.assert_close(
        rays[23, 31], torch.tensor([-0.005, -0.005, 1.0], dtype=torch.float64)
    )
    points = torch.tensor([[0.0, 0.0, 1.0], [0.1, -0.2, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    u, v = project_points(camera, points)
    torch.testing.assert_close(u[:2], torch.tensor([31.5, 36.5], dtype=torch.float64))
    torch.testing.assert_close(v[:2], torch.tensor([23.5, 13.5], dtype=torch.float64))
    assert not torch.isfinite(u[2])


def test_zbuffer_keeps_nearest_plane_and_depth_is_metric():
    camera = pinhole_camera()
    render = render_world(plane_world(camera))
    centre = (camera.height // 2, camera.width // 2)
    assert int(render.body_index[0, 0, *centre]) == 0
    torch.testing.assert_close(
        render.depth_m[0, 0, *centre].to(torch.float64), torch.tensor(0.5, dtype=torch.float64)
    )
    # Outside the small near plane only the far plane is visible.
    corner = (2, 2)
    assert int(render.body_index[0, 0, *corner]) == 1
    torch.testing.assert_close(
        render.depth_m[0, 0, *corner].to(torch.float64), torch.tensor(1.0, dtype=torch.float64)
    )
    # Moving the near plane on top of the far plane flips nothing until it moves off-centre.
    offset = render_world(plane_world(camera, translate_near=(0.30, 0.0)))
    assert int(offset.body_index[0, 0, *centre]) == 1
    assert int(offset.body_index[0, 0, *centre]) != int(render.body_index[0, 0, *centre])


def test_reference_projection_agrees_with_renderer_and_labels_match_image():
    config = D0Config(height=48, width=64, frames=2, batch=2)
    cameras = default_cameras(config.width, config.height, config.cameras)
    world = _audit_world(2, 3, config, seed=config.seed, cameras=cameras)
    render = render_world(world)
    for camera in cameras:
        view = render.camera_names.index(camera.name)
        reference, _, _, coverage, info = reference_visibility(world, camera, grid=None)
        assert set(info["grid_steps_per_body"]) == set(world.body_names)
        for body_id in range(len(world.bodies)):
            rendered = render.body_index[:, view] == body_id
            explained = rendered & ~(
                torch.nn.functional.max_pool2d(
                    coverage[:, body_id].float().unsqueeze(1), 3, 1, 1
                ).squeeze(1)
                > 0
            )
            assert int(explained.sum()) == 0
        consistency = image_label_consistency(world, camera, render)
        assert consistency["pixels_checked"] > 0
        assert consistency["max_radiance_error"] < 1e-5
        assert consistency["max_depth_error"] < 1e-5
        assert consistency["max_outside_face_margin_m"] < 1e-6
        assert reference.shape == render.body_index[:, view].shape


def test_audit_passes_batch_1_2_4_with_two_cameras():
    config = D0Config(height=48, width=64, frames=2, batch=2, cameras=("cam0", "cam1"))
    audit = audit_d0(config, batch_sizes=(1, 2, 4))
    assert audit["passed"] is True, audit["failures"]
    for batches in (1, 2, 4):
        entry = audit["checks"][f"B={batches}"]
        assert entry["max_unexplained"] == 0
        assert entry["max_missed"] == 0
        assert entry["max_symmetric_difference_excess"] == 0.0
        for camera in ("cam0", "cam1"):
            contamination = entry["per_camera"][camera]["contamination"]
            assert max(contamination.values(), default=0.0) == 0.0
    assert audit["checks"]["batch_permutation"]["exact"] is True
    assert audit["checks"]["single_sample_perturbation"] == {
        "others_unchanged": True,
        "perturbed_sample_changed": True,
    }
    assert audit["checks"]["static_zero_residual"]["max_abs_residual"] == 0.0
    assert audit["checks"]["cfa_channel_phase"]["exact"] is True
    assert audit["checks"]["unidentifiable_full_view_plane"]["identical"] is True


def test_audit_detects_historical_broadcast_label_defect():
    """Reproducible counterexample for ``[B,1,2] + [B,2] -> [B,B,2]`` in synthetic.py."""
    config = D0Config(height=48, width=64, frames=2, batch=4)
    cameras = default_cameras(config.width, config.height, config.cameras)
    world = _audit_world(4, 3, config, seed=config.seed + 11, cameras=cameras)
    render = render_world(world)
    camera = cameras[0]
    view = render.camera_names.index(camera.name)
    buggy = _buggy_merged_labels(world, camera, grid=129)
    reference, _, _, coverage, _ = reference_visibility(world, camera, grid=None)
    stats = compare_label_maps(buggy, reference, coverage=coverage)
    image = render.body_index[:, view]
    assert int(stats["unexplained"].max()) > 0
    assert int(((buggy >= 0) & (buggy != image)).sum()) > 0
    # Sample b's merged label claims the other samples' object pixels.
    claimed = []
    for target in range(4):
        for source in range(4):
            if target == source:
                continue
            source_objects = reference[source] >= 1
            if bool(source_objects.any()):
                claimed.append(float((buggy[target][source_objects] >= 1).float().mean()))
    assert max(claimed) > 0.5


@pytest.mark.parametrize("pattern", sorted(CFA_PHASE_CHANNELS))
def test_cfa_samples_declared_channel_and_phase(pattern):
    channels = cfa_phase_channel_map(pattern, 4, 6)
    for row in range(2):
        for col in range(2):
            expected = CFA_PHASE_CHANNELS[pattern][row][col]
            assert int(channels[row, col]) == expected
    # Encode the channel index in the radiance so the selection is directly observable.
    radiance = torch.zeros(1, 1, 3, 4, 6, dtype=torch.float64)
    radiance[0, 0, 0] = 0.25
    radiance[0, 0, 1] = 0.50
    radiance[0, 0, 2] = 0.75
    dn = apply_cfa(radiance, pattern=pattern, bit_depth=12, black_level_dn=64, white_level_dn=4095)
    assert dn.dtype == torch.uint16
    for row in range(2):
        for col in range(2):
            value = (int(channels[row, col]) + 1) * 0.25
            expected = 64 + value * (4095 - 64)
            assert abs(int(dn[0, 0, row, col]) - round(expected)) <= 1
    # The four phases must not be all identical for a pattern with distinct channels.
    assert len({int(channels[r, c]) for r in range(2) for c in range(2)}) > 1


def test_cfa_rejects_levels_outside_declared_bit_depth():
    radiance = torch.zeros(1, 1, 3, 4, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="CFA levels"):
        apply_cfa(radiance, pattern="GRBG", bit_depth=12, black_level_dn=4095, white_level_dn=4095)
    with pytest.raises(ValueError, match="unknown CFA pattern"):
        apply_cfa(radiance, pattern="XXXX", bit_depth=12, black_level_dn=0, white_level_dn=4095)


def test_factor_scenes_cover_time_shift_and_wrong_association():
    config = D0Config(height=32, width=48, frames=6, batch=1)
    shift_worlds, meta = build_factor_worlds(resolve_plan("time_shift"), config, frames=6, batch=1)
    assert meta["time_shift_frames"] == 3
    # Reported context lags the true camera pose by the declared shift.
    for frame in range(4, 6):
        reported = shift_worlds[frame].reported_translation
        true_shifted = shift_worlds[frame - 3].T_world_from_cam["cam0"][:, :3, 3]
        torch.testing.assert_close(reported, true_shifted, atol=1e-9, rtol=0.0)

    association_worlds, association_meta = build_factor_worlds(
        resolve_plan("wrong_association"), config, frames=4, batch=1
    )
    record = _render_factor(
        resolve_plan("wrong_association"),
        association_meta,
        config,
        association_worlds,
        out_dir=None,
    )
    assert record["metadata"]["wrong_association"] is True
    assert record["coverage"]["visible_pixels_total"] > 0


def test_gain_and_noise_change_measurement_without_moving_geometry():
    config = D0Config(height=32, width=48, frames=8, batch=1)
    worlds, _ = build_factor_worlds(resolve_plan("gain"), config, frames=8, batch=1)
    generator = torch.Generator().manual_seed(3)
    first = form_measurement(
        render_world(worlds[0]).radiance.to(torch.float64),
        gain=worlds[0].gain.to(torch.float64),
        noise_sigma=0.0,
        generator=generator,
    )
    second_radiance = render_world(worlds[1]).radiance.to(torch.float64)
    last = form_measurement(
        second_radiance,
        gain=worlds[-1].gain.to(torch.float64),
        noise_sigma=0.0,
        generator=generator,
    )
    assert float(worlds[-1].gain[0]) > float(worlds[0].gain[0])
    assert float(last.rgb3.mean()) > float(first.rgb3.mean())
    # Geometry is identical: the gain factor never moves the camera or the bodies.
    torch.testing.assert_close(
        worlds[0].T_world_from_cam["cam0"], worlds[-1].T_world_from_cam["cam0"]
    )
    torch.testing.assert_close(
        worlds[0].bodies[1].T_world_from_body, worlds[-1].bodies[1].T_world_from_body
    )


def test_factor_table_is_single_factors_then_combinations():
    assert list(DEFAULT_FACTORS) == [
        "static",
        "camera_only",
        "object_only",
        "both",
        "stop_reverse",
        "occlusion_recovery",
        "robot_self_occlusion",
        "gain",
        "noise",
        "time_shift",
        "wrong_association",
        "unidentifiable_full_view_plane",
    ]
    names = default_factor_names()
    assert list(names[: len(DEFAULT_FACTORS)]) == list(DEFAULT_FACTORS)
    assert list(names[len(DEFAULT_FACTORS) :]) == [
        "+".join(parts) for parts in DEFAULT_COMBINATIONS
    ]
    config = json.loads(S0_CONFIG.read_text(encoding="utf-8"))
    d0 = D0Config.from_config(config)
    assert set(config["mechanism_data"]["d0"]["factors"]).issubset(set(DEFAULT_FACTORS))
    assert d0.cfa_pattern == "GRBG"
    assert d0.cfa_white_level_dn == 4095
    assert d0.cameras == ("cam0", "cam1")
    assert d0.depths_m == (0.35, 0.6, 1.2)


def test_object_birth_and_exit_events_are_labeled():
    config = D0Config(height=48, width=64, frames=12, batch=1)
    worlds, meta = build_factor_worlds(resolve_plan("object_only"), config, frames=12, batch=1)
    names = list(worlds[0].body_names)
    object_1 = names.index("object_1")
    object_2 = names.index("object_2")
    column_1 = torch.stack([world.bodies[object_1].present[0] for world in worlds])
    column_2 = torch.stack([world.bodies[object_2].present[0] for world in worlds])
    assert bool(column_1[0]) is False and bool(column_1[-1]) is True
    assert bool(column_2[0]) is True and bool(column_2[-1]) is False
    assert meta["birth_death"] is True


def test_occlusion_recovery_hides_then_restores_a_body():
    config = D0Config(height=64, width=96, frames=24, batch=1)
    worlds, _ = build_factor_worlds(resolve_plan("occlusion_recovery"), config, frames=24, batch=1)
    names = list(worlds[0].body_names)
    occluded = names.index("object_2")
    visible = []
    for world in worlds:
        render = render_world(world)
        visible.append(int((render.body_index[0, 0] == occluded).sum()))
    assert max(visible) > 0
    hidden = [index for index, count in enumerate(visible) if count == 0]
    assert hidden, "the far object must be fully occluded during the sweep"
    assert hidden[0] > 0, "occlusion must start after the object was visible"
    assert hidden[-1] < len(visible) - 1, "occlusion must recover before the sequence ends"
    assert max(visible[hidden[-1] + 1 :]) > 0, "the object must become visible again"


def test_robot_self_occlusion_renders_a_visible_robot_body():
    config = D0Config(height=64, width=96, frames=4, batch=1)
    worlds, _ = build_factor_worlds(resolve_plan("robot_self_occlusion"), config, frames=4, batch=1)
    render = render_world(worlds[0])
    robot = list(render.body_names).index("robot")
    assert int((render.body_index == robot).sum()) > 0
    # The robot is nearer than the background, so it occludes it there.
    robot_mask = render.body_index == robot
    assert int((render.depth_m[robot_mask] < 0.5).all())  # robot participates at ~0.22 m


def test_stop_reverse_motion_phase_is_labeled():
    config = D0Config(height=32, width=48, frames=24, batch=1)
    worlds, meta = build_factor_worlds(resolve_plan("stop_reverse"), config, frames=24, batch=1)
    assert meta["object_motion"] == "stop_reverse"
    tracked = [
        next(track for track in world.bodies if track.body.kind == "object") for world in worlds
    ]
    positions = torch.stack([track.T_world_from_body[:, :3, 3] for track in tracked])
    deltas = (positions[1:] - positions[:-1]).norm(dim=-1)
    assert float(deltas[:8].min()) > 0.0
    assert float(deltas[8:15].max()) < 1e-9
    assert float(deltas[-4:].min()) > 0.0
    phases = motion_phase(positions)
    assert int((phases == -1).sum()) > 0
    assert int((phases == 0).sum()) > 0


def test_online_bundle_keeps_ground_truth_separate(tmp_path):
    config = {
        "split_seed": 7,
        "mechanism_data": {
            "d0": {
                "resolution": [32, 48],
                "frames": 2,
                "objects": 2,
                "cameras": ["cam0"],
                "factors": ["camera_only", "time_shift"],
                "cfa": {
                    "pattern": "GRBG",
                    "bit_depth": 12,
                    "black_level_dn": 64,
                    "white_level_dn": 4095,
                },
            }
        },
    }
    config_path = tmp_path / "s0.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "mechanism-data"
    status = run_cli(
        [
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--audit-batches",
            "1",
        ]
    )
    assert status == 0
    for name in (
        "d0/manifest.json",
        "d0/audit.json",
        "factors.json",
        "d1/feasibility.json",
        "run.json",
    ):
        assert (output / name).is_file(), name
    assert (output / "d1" / "collect_mechanism_data.py").is_file()
    audit = json.loads((output / "d0" / "audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True, audit["failures"]
    manifest = json.loads((output / "d0" / "manifest.json").read_text(encoding="utf-8"))
    online = set(manifest["online_vs_ground_truth"]["online_keys"])
    truth = set(manifest["online_vs_ground_truth"]["gt_keys"])
    assert online.isdisjoint(truth)
    assert manifest["frames_and_conventions"]["projection"] == "u = fx*x/z + cx, v = fy*y/z + cy"
    assert manifest["units"]["geometry"] == "metre"
    assert "camera-space z" in manifest["units"]["depth_m"]
    for factor in manifest["factors"]:
        keys = set(factor["arrays"])
        assert keys == online | truth
        assert not any(key.startswith("gt_") for key in online)
    # The npz arrays are the real products, not fixtures.
    with np.load(output / "d0" / "camera_only.npz") as data:
        assert data["online_measurement_cfa"].dtype == np.uint16
        assert data["online_measurement_rgb3"].shape[0] == 2
        assert int(data["gt_visible_pixels"].sum()) > 0
        finite_depth = data["gt_depth_m"][np.isfinite(data["gt_depth_m"])]
        assert finite_depth.size > 0 and float(finite_depth.min()) > 0.0
        assert not np.isnan(data["gt_depth_m"]).any()


def test_d1_refuses_sidecar_without_verified_correspondence(tmp_path):
    npz_path = tmp_path / "episode_000000.npz"
    np.savez_compressed(
        npz_path,
        actions=np.zeros((3, 7), dtype=np.float32),
        states=np.zeros((3, 79), dtype=np.float64),
        robot_states=np.zeros((3, 8), dtype=np.float32),
        agentview_raw_uint8=np.zeros((3, 4, 4, 3), dtype=np.uint8),
    )
    probe = d1_npz_field_probe(npz_path)
    assert probe["exists"] is True
    assert probe["state_dim"] == 79
    assert probe["actions_per_frame"] == 7
    assert probe["three_channel_image_keys"] == ["agentview_raw_uint8"]
    assert probe["has_depth"] is False and probe["has_intrinsics"] is False
    unverified = d1_plan(
        platform="libero",
        npz_path=npz_path,
        output_dir=tmp_path,
        correspondence_verified=False,
    )
    assert unverified["target"]["mode"] == "extension_set"
    assert unverified["target"]["name"] == "n04ext_libero_episode_000000.npz"
    assert npz_path.stem != unverified["target"]["name"]
    verified = d1_plan(
        platform="libero",
        npz_path=npz_path,
        output_dir=tmp_path,
        correspondence_verified=True,
    )
    assert verified["target"]["mode"] == "sidecar"
    assert verified["target"]["name"] == f"{npz_path.stem}.n04-sidecar.npz"


def test_d1_feasibility_marks_unprobed_items_and_uses_evidence():
    empty = build_feasibility({})
    assert empty["verdict_counts"]["not_probed"] == len(empty["items"])
    evidence = {
        "libero_sim": {"import_ok": True, "env_created": True, "bddl_files_present": True},
        "robotwin_sim": {"import_ok": True, "assets_present": True, "task_config_present": False},
        "npz_fields": {
            "libero_actions_per_frame": True,
            "robotwin_actions_per_frame": True,
            "libero_three_channel_only": True,
            "robotwin_three_channel_only": True,
            "d0_synthetic_cfa": True,
        },
    }
    report = build_feasibility(evidence, checked_at="2026-10-14T00:00:00Z")
    verdicts = {item["id"]: item["verdict"] for item in report["items"]}
    assert verdicts["libero_sim_assets"] == "available"
    assert verdicts["robotwin_sim_assets"] == "partial"
    assert verdicts["executed_action"] == "available"
    assert verdicts["cfa_representation"] == "available"
    assert verdicts["depth"] == "not_probed"
    robotwin = next(item for item in report["items"] if item["id"] == "robotwin_sim_assets")
    assert robotwin["missing"] == ["task_config_present"]
    assert robotwin["verification_level"] == "api_and_assets_only"
    assert robotwin["plan"]


# -- D1 executed-trace schema (bit-exact recomputability contract) -------------------------


def _trace_summary(**overrides):
    summary = {
        "fields_present": list(D1_TRACE_STEP_FIELDS),
        "warmstart_mode": "saved_before_step",
        "warmstart_array_present": True,
        "controller_buffer_policy": "recorded_initial_conditions",
        "recipe_recorded": True,
        "verification": {"bit_exact": True, "max_abs_diff": 0.0, "steps": 3},
    }
    summary.update(overrides)
    return summary


def _verdict(summary):
    report = d1_recomputability_report(summary)
    return report["bit_exact_recomputable"], report["evidence_level"], report["missing"]


def test_d1_trace_schema_declares_step_fields_and_semantics():
    schema = d1_trace_schema()
    assert schema["schema"] == "n04-d1-executed-trace-v1"
    assert set(schema["step_fields"]) == set(D1_TRACE_STEP_FIELDS)
    assert D1_TRACE_STEP_FIELDS == ("step_index", "pre_state", "issued_command", "post_state")
    for field_name, semantics in schema["step_fields"].items():
        assert semantics.strip(), field_name
    assert set(schema["warmstart_modes"]) == set(D1_TRACE_WARMSTART_MODES)
    assert set(schema["controller_buffer_policies"]) == set(D1_TRACE_CONTROLLER_BUFFER_POLICIES)
    assert set(schema["evidence_levels"]) == {
        "bit_exact_recomputable",
        "phase_and_state_recorded",
        "insufficient",
    }
    rule = schema["recomputability_rule"]
    assert "pre_state" in rule and "issued_command" in rule and "post_state" in rule
    assert "warm-start" in rule and "controller" in rule and "phase-level" in rule


@pytest.mark.parametrize("dropped", ("pre_state", "issued_command", "post_state"))
def test_recomputability_requires_pre_post_state_and_issued_command(dropped):
    fields = [name for name in D1_TRACE_STEP_FIELDS if name != dropped]
    bit_exact, level, missing = _verdict(_trace_summary(fields_present=fields))
    assert bit_exact is False
    assert level == "insufficient"
    assert dropped in missing


def test_recomputability_requires_a_controlled_warmstart():
    # saved mode without the recorded array cannot be recomputed
    bit_exact, level, missing = _verdict(_trace_summary(warmstart_array_present=False))
    assert (bit_exact, level, missing) == (False, "insufficient", ["qacc_warmstart_before"])
    # disabling the solver warm-start or zeroing it needs no array
    for mode in ("disabled_by_flag", "zeroed_each_step"):
        bit_exact, level, missing = _verdict(
            _trace_summary(warmstart_mode=mode, warmstart_array_present=False)
        )
        assert (bit_exact, level, missing) == (True, "bit_exact_recomputable", [])
    # an undeclared mode is refused rather than assumed
    assert _verdict(_trace_summary(warmstart_mode=None))[2] == ["warmstart_mode"]


def test_recomputability_needs_explicit_controller_buffer_policy_and_recipe():
    _, level, missing = _verdict(_trace_summary(controller_buffer_policy=None))
    assert level == "insufficient" and missing == ["controller_buffer_policy"]
    _, level, missing = _verdict(_trace_summary(recipe_recorded=False))
    assert level == "insufficient" and missing == ["replay_recipe"]
    # a declared non-recomputable buffer state must not be silently upgraded
    bit_exact, level, missing = _verdict(
        _trace_summary(controller_buffer_policy="declared_not_recomputable")
    )
    assert (bit_exact, level, missing) == (False, "phase_and_state_recorded", [])


def test_recomputability_is_not_claimed_without_a_passing_verification():
    bit_exact, level, _ = _verdict(_trace_summary(verification=None))
    assert (bit_exact, level) == (False, "phase_and_state_recorded")
    bit_exact, level, _ = _verdict(_trace_summary(verification={"bit_exact": False}))
    assert (bit_exact, level) == (False, "phase_and_state_recorded")
    report = d1_recomputability_report(_trace_summary(verification=None))
    assert any("verification" in note for note in report["notes"])


def test_d1_write_trace_roundtrip(tmp_path):
    steps = [
        {
            "step_index": index,
            "pre_state": np.array([float(index), 1.0, 2.0, -3.0], dtype=np.float64),
            "issued_command": np.array([0.1 * index, -0.2, 0.3], dtype=np.float32),
            "post_state": np.array([float(index) + 1, 1.1, 1.9, -3.1], dtype=np.float64),
            "qacc_warmstart_before": np.array([0.5, -0.25], dtype=np.float64),
            "qacc_warmstart_after": np.array([0.4, -0.2], dtype=np.float64),
        }
        for index in range(2)
    ]
    recipe = {"env": {"env_seed": 0}, "ordered_operations": ["reset", "step"]}
    record = d1_write_trace(
        tmp_path,
        "demo-sidecar",
        platform="libero",
        source_npz="/remote/episode_000000.npz",
        steps=steps,
        recipe=recipe,
        warmstart_mode="saved_before_step",
        controller_buffer_policy="recorded_initial_conditions",
        source_fidelity={"level": "phase_only", "basis": "external recording"},
        verification={"bit_exact": True, "max_abs_diff": 0.0, "steps": 2, "tolerance": 1e-12},
    )
    assert record["schema"] == "n04-d1-executed-trace-v1"
    assert record["recomputability"]["bit_exact_recomputable"] is True
    assert record["recomputability"]["evidence_level"] == "bit_exact_recomputable"
    assert record["source_fidelity"]["level"] == "phase_only"
    assert record["issued_command_source"] == "collector_step_call_argument"
    assert record["recipe"] == recipe
    npz_path = tmp_path / "demo-sidecar.npz"
    json_path = tmp_path / "demo-sidecar.trace.json"
    assert npz_path.is_file() and json_path.is_file()
    assert record["npz_sha256"] == _sha256_file(npz_path)
    with np.load(npz_path) as data:
        assert set(data.files) >= set(D1_TRACE_STEP_FIELDS) | {"qacc_warmstart_before"}
        assert data["pre_state"].dtype == np.float64
        assert data["issued_command"].dtype == np.float32
        assert data["issued_command"][1, 0] == pytest.approx(0.1, abs=1e-7)
        assert data["qacc_warmstart_before"].shape == (2, 2)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["warmstart_mode"] == "saved_before_step"
    assert saved["controller_buffer_policy"] == "recorded_initial_conditions"
    assert saved["step_field_semantics"] == d1_trace_schema()["step_fields"]
    assert saved["recomputability"]["bit_exact_recomputable"] is True


def test_d1_write_trace_refuses_incomplete_contract(tmp_path):
    step = {
        "step_index": 0,
        "pre_state": np.zeros(3),
        "issued_command": np.zeros(2, dtype=np.float32),
        "post_state": np.ones(3),
    }
    with pytest.raises(ValueError, match="warmstart mode"):
        d1_write_trace(
            tmp_path,
            "bad",
            platform="libero",
            source_npz="x.npz",
            steps=[step],
            recipe={"a": 1},
            warmstart_mode="guessed",
            controller_buffer_policy="recorded_initial_conditions",
        )
    with pytest.raises(ValueError, match="controller buffer policy"):
        d1_write_trace(
            tmp_path,
            "bad2",
            platform="libero",
            source_npz="x.npz",
            steps=[step],
            recipe={"a": 1},
            warmstart_mode="zeroed_each_step",
            controller_buffer_policy="unknown",
        )
    with pytest.raises(ValueError, match="at least one step"):
        d1_write_trace(
            tmp_path,
            "bad3",
            platform="libero",
            source_npz="x.npz",
            steps=[],
            recipe={"a": 1},
            warmstart_mode="zeroed_each_step",
            controller_buffer_policy="recorded_initial_conditions",
        )
    with pytest.raises(ValueError, match="qacc_warmstart_before"):
        d1_write_trace(
            tmp_path,
            "bad4",
            platform="libero",
            source_npz="x.npz",
            steps=[step],
            recipe={"a": 1},
            warmstart_mode="saved_before_step",
            controller_buffer_policy="recorded_initial_conditions",
        )


def test_d1_plan_carries_the_trace_schema():
    plan = d1_plan(
        platform="libero",
        npz_path="/remote/episode_000000.npz",
        output_dir="/tmp/out",
        correspondence_verified=True,
    )
    assert plan["target"]["mode"] == "sidecar"
    assert plan["trace_schema"] == "n04-d1-executed-trace-v1"
    assert plan["trace_step_fields"] == list(D1_TRACE_STEP_FIELDS)
    assert "phase-level" in plan["recomputability_rule"]
