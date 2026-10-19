"""Bounded LIBERO extension capture; never attaches labels to an external NPZ.

The optional frozen RawVLA producer is used only to form benchmark pseudo-RAW.
Rendered RGB remains a reference artifact, never a native CFA claim. Imports of the
simulator/producer are deferred until collection. All cameras share the current
simulation state, and image/depth/segmentation use the same top-left pixel grid.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def configure_renderer(env: Any) -> None:
    """Depth must describe the declared pixel centre, not an MSAA subpixel sample."""
    env.sim.model.vis.quality.offsamples = 0
    context = env.sim._render_context_offscreen
    context.gl_ctx.make_current()
    context.con.free()
    context._set_mujoco_context_and_buffers()


def opaque_geometry_ids(context: Any) -> set[int]:
    """Rendered alpha, not physics collision flags, defines unique depth support."""
    return {
        int(g.objid)
        for g in context.scn.geoms[: context.scn.ngeom]
        if g.objtype == 5 and g.rgba[3] >= 0.999
    }


def front_mesh_depth(model, data, geom: int, origin, direction) -> float:
    """Nearest front-facing rendered triangle; mj_ray also hits culled backfaces."""
    mesh = int(model.geom_dataid[geom])
    vertices = model.mesh_vert[
        model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]
    ]
    faces = model.mesh_face[
        model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]
    ]
    vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    triangles = vertices[faces]
    edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, edge2)
    determinant = (edge1 * h).sum(-1)
    safe = np.where(determinant > 1e-12, determinant, 1.0)
    relative = origin - triangles[:, 0]
    u = (relative * h).sum(-1) / safe
    q = np.cross(relative, edge1)
    v = (direction * q).sum(-1) / safe
    distance = (edge2 * q).sum(-1) / safe
    valid = (determinant > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > 0)
    return float(np.min(distance[valid], initial=np.inf))


def ray_depth_audit(env: Any, frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """CPU geometry vs saved GL depth, respecting culling and raster precision.

    Numerical bounds are derived from depth-buffer ULPs and GL_SUBPIXEL_BITS,
    independent of model-performance thresholds. Nonmatching silhouettes are
    explicitly unsupported, with a minimum 80% support gate.
    """
    import mujoco
    from OpenGL.GL import GL_SUBPIXEL_BITS, glGetIntegerv

    model, data = env.sim.model._model, env.sim.data._data
    context = env.sim._render_context_offscreen
    context.gl_ctx.make_current()
    bits = int(glGetIntegerv(GL_SUBPIXEL_BITS))
    if bits <= 0 or not context.scn.flags[int(mujoco.mjtRndFlag.mjRND_CULL_FACE)]:
        raise ValueError("unverified renderer precision or back-face convention")
    opaque_ids = opaque_geometry_ids(context)
    errors, bounds, matches, references = [], [], [], []
    for view in range(2):
        k, pose = frame["online_K"][view], frame["gt_T_W_from_C"][view]
        origin = pose[:3, 3].copy()
        for v in (32, 64, 128, 192, 224):
            for u in (32, 64, 128, 192, 224):
                geom = np.zeros(1, dtype=np.int32)
                ray = pose[:3, :3] @ np.linalg.solve(k, np.array([u, v, 1.0]))
                norm = np.linalg.norm(ray)
                distance = mujoco.mj_ray(
                    model,
                    data,
                    origin,
                    ray / norm,
                    np.asarray(context.vopt.geomgroup, dtype=np.uint8),
                    True,
                    -1,
                    geom,
                )
                expected_geom = int(frame["gt_segmentation"][view, v, u, 1])
                match = distance >= 0 and geom[0] == expected_geom and expected_geom in opaque_ids
                reference = distance / norm
                mesh = match and model.geom_type[geom[0]] == int(mujoco.mjtGeom.mjGEOM_MESH)
                if mesh:
                    reference = front_mesh_depth(model, data, int(geom[0]), origin, ray)
                    match = np.isfinite(reference)
                subpixel_error = 0.0
                for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    delta = 2.0 ** (-bits)
                    near_ray = pose[:3, :3] @ np.linalg.solve(
                        k, np.array([u + du * delta, v + dv * delta, 1.0])
                    )
                    if mesh:
                        nearby = front_mesh_depth(model, data, int(geom[0]), origin, near_ray)
                    else:
                        near_geom = np.zeros(1, dtype=np.int32)
                        near_norm = np.linalg.norm(near_ray)
                        near_dist = mujoco.mj_ray(
                            model,
                            data,
                            origin,
                            near_ray / near_norm,
                            np.asarray(context.vopt.geomgroup, dtype=np.uint8),
                            True,
                            -1,
                            near_geom,
                        )
                        nearby = near_dist / near_norm
                        match = match and near_geom[0] == geom[0]
                    if not np.isfinite(nearby):
                        match = False
                    elif np.isfinite(reference):
                        subpixel_error = max(subpixel_error, abs(nearby - reference))
                depth = float(frame["gt_depth_m"][view, v, u])
                near = model.stat.extent * model.vis.map.znear
                errors.append(abs(reference - depth) if match else 0.0)
                references.append(reference if match else 0.0)
                bounds.append(
                    8 * np.finfo(np.float32).eps * depth**2 / near + 1e-6 + subpixel_error
                )
                matches.append(match)
    return dict(
        audit_ray_error_m=np.asarray(errors),
        audit_ray_reference_depth_m=np.asarray(references),
        audit_ray_bound_m=np.asarray(bounds),
        audit_ray_matching_geom=np.asarray(matches),
        audit_subpixel_bits=np.asarray(bits),
    )


def robot_proprio(env: Any) -> dict[str, np.ndarray]:
    """Only robot and gripper joint sensors; free object joints are hidden GT."""
    position, velocity = [], []
    for robot in env.robots:
        position.extend(robot._ref_joint_pos_indexes)
        velocity.extend(robot._ref_joint_vel_indexes)
        position.extend(robot._ref_gripper_joint_pos_indexes)
        velocity.extend(robot._ref_gripper_joint_vel_indexes)
    if not position or len(set(position)) != len(position):
        raise ValueError("missing or duplicate robot sensor indices")
    return dict(
        online_qpos=np.asarray(env.sim.data.qpos)[position].copy(),
        online_qvel=np.asarray(env.sim.data.qvel)[velocity].copy(),
        gt_sim_qpos=np.asarray(env.sim.data.qpos).copy(),
        gt_sim_qvel=np.asarray(env.sim.data.qvel).copy(),
        audit_robot_qpos_indices=np.asarray(position, dtype=np.int64),
        audit_robot_qvel_indices=np.asarray(velocity, dtype=np.int64),
    )


def capture_frame(env: Any, *, frame_index: int, noise_seed: int) -> dict[str, np.ndarray]:
    from rawvla_bench.raw import make_rawvla_observation
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    values: dict[str, list[np.ndarray]] = {}
    for view, camera in enumerate(("agentview", "robot0_eye_in_hand")):
        rgb, depth = env.sim.render(256, 256, camera_name=camera, depth=True)
        segmentation = env.sim.render(256, 256, camera_name=camera, segmentation=True)
        # MuJoCo read_pixels is bottom-up; flip every pixel-aligned product together.
        rgb = np.ascontiguousarray(rgb[::-1])
        depth = np.ascontiguousarray(get_real_depth_map(env.sim, depth)[::-1])
        segmentation = np.ascontiguousarray(segmentation[::-1])
        raw = make_rawvla_observation(
            rgb,
            noise_seed=noise_seed,
            frame_index=frame_index,
            view_index=view,
            representation="raw",
            sensor_saturation_ev=0.0,
        )
        intrinsic = get_camera_intrinsic_matrix(env.sim, camera, 256, 256)
        intrinsic[:2, 2] -= 0.5  # viewport edge centre (128) -> pixel index centre (127.5)
        context = env.sim._render_context_offscreen
        opaque_ids = sorted(opaque_geometry_ids(context))
        # Transparent pixels mix multiple surfaces. Keep their RAW and visibility,
        # but do not pretend the depth buffer is a unique correspondence label.
        frame = dict(
            online_rawvla_rgb3=np.asarray(raw),
            reference_rgb=rgb,
            gt_depth_m=depth,
            gt_segmentation=segmentation,
            gt_visible_support=(segmentation[..., 1] >= 0),
            gt_geometry_support=np.isin(segmentation[..., 1], opaque_ids),
            online_K=intrinsic,
            gt_T_W_from_C=get_camera_extrinsic_matrix(env.sim, camera),
        )
        for key, value in frame.items():
            values.setdefault(key, []).append(np.array(value, copy=True))
    result = {key: np.stack(value) for key, value in values.items()}
    result["online_sim_time_s"] = np.asarray(env.sim.data.time, dtype=np.float64)
    result.update(robot_proprio(env))
    result["gt_body_position_W"] = np.array(env.sim.data.body_xpos, copy=True)
    result["gt_body_rotation_W"] = np.array(env.sim.data.body_xmat, copy=True)
    # Contact entries are variable-length; keep exact per-frame arrays in the archive.
    result["gt_contact_geom_pairs"] = np.asarray(
        [[c.geom1, c.geom2] for c in env.sim.data.contact[: env.sim.data.ncon]], dtype=np.int32
    ).reshape(-1, 2)
    result["gt_contact_distance_m"] = np.asarray(
        [c.dist for c in env.sim.data.contact[: env.sim.data.ncon]], dtype=np.float64
    )
    for key, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite collected field: {key}")
    if not (result["gt_depth_m"] > 0).all():
        raise ValueError("invalid metric depth")
    result.update(ray_depth_audit(env, result))
    return result


def write_capture(
    output: Path,
    name: str,
    frames: list[dict[str, np.ndarray]],
    *,
    replay_max_errors: dict[str, float],
    trace: dict[str, Any],
    noise_seed: int,
) -> dict[str, Any]:
    """Write aligned measurements with explicit provenance and verification scope."""
    arrays = {}
    for key in frames[0]:
        if "contact" in key:
            for index, frame in enumerate(frames):
                arrays[f"{key}_{index:04d}"] = frame[key]
        else:
            arrays[key] = np.stack([frame[key] for frame in frames])
    path = output / f"{name}.measurements.npz"
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)
    ray_passed = all(
        np.mean(frame["audit_ray_matching_geom"]) >= 0.8
        and np.all(frame["audit_ray_error_m"] <= frame["audit_ray_bound_m"])
        for frame in frames
    )
    record = dict(
        schema="n04-d1-libero-extension-v1",
        mode="extension_set",
        frames=len(frames),
        cameras=["agentview", "robot0_eye_in_hand"],
        npz=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_trace=trace["trace_record"],
        trace_sha256=trace["trace_record_sha256"],
        source_npz=trace["source_npz"],
        source_correspondence_verified=False,
        representation="rawvla_rgb3",
        native_cfa=False,
        noise_seed=noise_seed,
        geometry_audit=dict(
            passed=bool(ray_passed),
            samples=50 * len(frames),
            matching_geom_samples=int(sum(f["audit_ray_matching_geom"].sum() for f in frames)),
            max_depth_error_m=float(max(f["audit_ray_error_m"].max() for f in frames)),
            max_error_over_bound=float(
                max((f["audit_ray_error_m"] / f["audit_ray_bound_m"]).max() for f in frames)
            ),
            bound="depth ULPs + GL subpixel ray interval; front faces; >=80% supported rays",
        ),
        renderer=dict(offsamples=0, principal_point_pixel_index=[127.5, 127.5]),
        processing="MuJoCo RGB -> frozen RawVLA make_rawvla_observation; benchmark pseudo-RAW",
        image_coordinates="top-left; x right, y down; raw/depth/segmentation same grid; no resize",
        camera_coordinates="T_W_from_C; C has x right, y down, z forward; K in original pixels",
        time_semantics=(
            "ideal instantaneous simulator sampling in sim seconds; "
            "no real exposure or arrival latency"
        ),
        supervision_policy=(
            "gt_* only supervision; no external NPZ labels or source expert success inherited"
        ),
        action_semantics=(
            "trace issued_command is clipped normalized env.step command, not motor torque"
        ),
        label_coverage=["depth_m", "camera_pose", "body_pose", "segmentation", "contact"],
        factor_coverage="bounded replay of source commands; D1 factor coverage remains partial",
        arrays={
            key: dict(shape=list(value.shape), dtype=str(value.dtype))
            for key, value in arrays.items()
        },
        replay_max_errors=replay_max_errors,
        replay_verified=bool(ray_passed)
        and bool(replay_max_errors)
        and all(np.isfinite(v) and v <= 1e-12 for v in replay_max_errors.values()),
    )
    record_path = output / f"{name}.measurements.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
