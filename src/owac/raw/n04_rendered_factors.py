"""MuJoCo rasterized 3-D factor suite, separate from benchmark trajectories.

Scripted kinematic interventions are supervision, never robot executed controls.
The RGB-derived radiance proxy is explicitly synthetic pseudo-RAW, not native CFA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from owac.raw.n04_live_collection import save_json, sha256

FACTORS = (
    "static",
    "camera",
    "object",
    "both",
    "stop_reverse",
    "occlusion_recovery",
    "birth_disappearance",
    "gain",
    "noise",
    "time_shift",
    "wrong_association",
    "calibration_offset",
    "shadow_highlight",
    "moving_plane",
    "both_gain_noise",
    "occlusion_reverse",
)
XML = """<mujoco><option gravity="0 0 -9.81" timestep="0.01"/>
<visual><quality offsamples="0" shadowsize="1024"/><map znear="0.01" zfar="20"/></visual>
<asset><texture name="tex" type="2d" builtin="checker" rgb1=".3 .35 .4" rgb2=".6 .55 .5"
 width="256" height="256"/>
<material name="floor" texture="tex" texrepeat="4 4" reflectance="0"/>
<material name="red" rgba=".8 .15 .12 1" specular=".6" shininess=".6"/></asset>
<worldbody><light name="key" pos="-1 -1 4" dir="0 0 -1" directional="false"/>

<geom name="floor" type="plane" size="10 10 .01" material="floor"/>
<camera name="fixed" pos="0 0 3" xyaxes="1 0 0 0 1 0" fovy="50"/>
<camera name="offset" pos=".15 .1 3" xyaxes="1 0 0 0 1 0" fovy="50"/>
<body name="target" mocap="true" pos="0 0 .4">
<geom name="target" type="box" size=".18 .18 .18" material="red"/></body>
<body name="other" mocap="true" pos=".5 .4 .6">
<geom name="other" type="box" size=".16 .16 .16" rgba=".15 .3 .8 1"/></body>
<body name="occluder" mocap="true" pos="3 0 1.6">
<geom name="occluder" type="box" size=".65 .65 .02" rgba=".15 .6 .2 1"/></body>
<body name="fullplane" mocap="true" pos="0 0 -4">
<geom name="fullplane" type="box" size="10 10 .01" rgba=".4 .4 .4 1"/></body>
</worldbody></mujoco>"""


def trajectory(factor: str, step: int) -> tuple[float, float, float]:
    motion = 0.055 * step - 0.3
    reverse = [-0.30, -0.20, -0.10, 0, 0, 0, 0, -0.10, -0.20, -0.30, -0.40, -0.50][step]
    camera = motion if factor in ("camera", "both", "both_gain_noise") else 0.0
    obj = (
        motion
        if factor in ("object", "both", "both_gain_noise", "wrong_association", "time_shift")
        else 0.0
    )
    if factor in ("stop_reverse", "occlusion_reverse"):
        obj = reverse
    occluder = (
        0.0 if factor in ("occlusion_recovery", "occlusion_reverse") and 4 <= step < 8 else 3.0
    )
    if factor == "birth_disappearance":
        obj = 3.0 if 4 <= step < 8 else 0.0
    return camera, obj, occluder


def generate(output: Path, sources: int = 10):
    import mujoco
    from OpenGL.GL import GL_SUBPIXEL_BITS, glGetIntegerv

    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema="n04-rendered-factors-v1",
        scope=__doc__,
        renderer="MuJoCo GL raster, offsamples=0",
        frames_per_sequence=12,
        factors=list(FACTORS),
        sequences=[],
    )
    (output / "scene.xml").write_text(XML)
    for seed in range(sources):
        source_id = f"n04ext/rendered-rig/seed-{seed}"
        split = "train" if seed < 8 else ("dev" if seed == 8 else "final")
        for factor in FACTORS:
            if factor == "both_gain_noise" and seed < 8:
                continue  # preregistered held-out combination, no training variants
            model = mujoco.MjModel.from_xml_string(XML)
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=128, width=128)
            subpixel_bits = int(glGetIntegerv(GL_SUBPIXEL_BITS))
            if subpixel_bits <= 0:
                raise ValueError("cannot verify raster subpixel precision")
            basecam = model.cam_pos.copy()
            rng = np.random.default_rng(20261014 + seed)
            rows = []
            try:
                for step in range(12):
                    camera, obj, occ = trajectory(factor, step)
                    data.mocap_pos[:] = [
                        [obj, seed * 0.003, 0.4],
                        [0.5, 0.4, 0.6],
                        [occ, 0, 1.6],
                        [0, 0, -4],
                    ]
                    model.cam_pos[:] = basecam
                    model.cam_pos[:, 0] += camera
                    if factor == "moving_plane":
                        data.mocap_pos[3] = [0.05 * step, 0, 2.0]
                    if factor == "shadow_highlight":
                        model.light_pos[0, 0] = -1 + 0.2 * step
                    data.time = step * 0.05
                    mujoco.mj_forward(model, data)
                    rawviews = []
                    depths = []
                    segs = []
                    poses = []
                    ks = []
                    rayerrors = []
                    raybounds = []
                    raymatches = []
                    for view in range(2):
                        renderer.update_scene(data, camera=view)
                        rgb = renderer.render().copy()
                        renderer.enable_depth_rendering()
                        depth = renderer.render().copy()
                        renderer.disable_depth_rendering()
                        renderer.enable_segmentation_rendering()
                        seg = renderer.render().copy()
                        renderer.disable_segmentation_rendering()
                        focal = 64 / np.tan(np.deg2rad(model.cam_fovy[view]) / 2)
                        k = np.array([[focal, 0, 63.5], [0, focal, 63.5], [0, 0, 1.0]])
                        pose = np.eye(4)
                        pose[:3, :3] = data.cam_xmat[view].reshape(3, 3) @ np.diag([1, -1, -1])
                        pose[:3, 3] = data.cam_xpos[view]
                        raw = (rgb.astype(np.float32) / 255) ** 2.2
                        gain = (1.0 + step * 0.04) if factor in ("gain", "both_gain_noise") else 1.0
                        raw *= gain
                        if factor in ("noise", "both_gain_noise"):
                            raw = np.maximum(0, raw + rng.normal(0, 0.005, raw.shape)).astype(
                                np.float32
                            )
                        for v in (16, 32, 64, 96, 112):
                            for u in (16, 32, 64, 96, 112):
                                ray = np.linalg.solve(k, np.array([u, v, 1.0]))
                                direction = pose[:3, :3] @ ray
                                norm = np.linalg.norm(direction)
                                geom = np.zeros(1, dtype=np.int32)
                                dist = mujoco.mj_ray(
                                    model,
                                    data,
                                    pose[:3, 3].copy(),
                                    direction / norm,
                                    None,
                                    True,
                                    -1,
                                    geom,
                                )
                                match = dist >= 0 and geom[0] == seg[v, u, 0]
                                # GL snaps polygon coverage/vertices to a finite subpixel
                                # grid. At grazing faces this dominates depth-buffer ULPs.
                                # Bound it using independent rays at that measured grid
                                # precision; never enlarge a model performance threshold.
                                reference_depth = dist / norm
                                subpixel_error = 0.0
                                for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                    delta = 2.0 ** (-subpixel_bits)
                                    near_ray = np.linalg.solve(
                                        k, np.array([u + du * delta, v + dv * delta, 1.0])
                                    )
                                    near_direction = pose[:3, :3] @ near_ray
                                    near_norm = np.linalg.norm(near_direction)
                                    near_geom = np.zeros(1, dtype=np.int32)
                                    near_distance = mujoco.mj_ray(
                                        model,
                                        data,
                                        pose[:3, 3].copy(),
                                        near_direction / near_norm,
                                        None,
                                        True,
                                        -1,
                                        near_geom,
                                    )
                                    if near_geom[0] != geom[0]:
                                        match = False
                                    subpixel_error = max(
                                        subpixel_error,
                                        abs(near_distance / near_norm - reference_depth),
                                    )
                                raymatches.append(match)
                                rayerrors.append(abs(dist / norm - depth[v, u]) if match else 0.0)
                                raybounds.append(
                                    8
                                    * np.finfo(np.float32).eps
                                    * depth[v, u] ** 2
                                    / (model.stat.extent * model.vis.map.znear)
                                    + 1e-6
                                    + subpixel_error
                                )
                        rawviews.append(raw)
                        depths.append(depth)
                        segs.append(seg)
                        poses.append(pose)
                        ks.append(k)
                    target = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target"))
                    row = dict(
                        online_rawvla_rgb3=np.stack(rawviews),
                        gt_depth_m=np.stack(depths),
                        gt_segmentation=np.stack(segs),
                        gt_T_W_from_C=np.stack(poses),
                        online_K=np.stack(ks),
                        gt_body_position_W=data.xpos.copy(),
                        gt_target_geom_id=np.asarray(target),
                        gt_time_s=np.asarray(data.time),
                        gt_gain=np.asarray(gain),
                        audit_subpixel_bits=np.asarray(subpixel_bits),
                        audit_ray_error_m=np.array(rayerrors),
                        audit_ray_bound_m=np.array(raybounds),
                        audit_ray_matching_geom=np.array(raymatches),
                    )
                    rows.append(row)
            finally:
                renderer.close()
            arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
            # Perturbed inputs remain separately named; true correspondence never overwritten.
            arrays["input_frame_index"] = np.maximum(
                0, np.arange(12) - (2 if factor == "time_shift" else 0)
            )
            arrays["input_association"] = np.array(
                [1, 0] if factor == "wrong_association" else [0, 1]
            )
            arrays["input_K"] = arrays["online_K"].copy()
            if factor == "calibration_offset":
                arrays["input_K"][..., 0, 2] += 2
            file = output / f"seed-{seed:02d}-{factor}.npz"
            np.savez_compressed(file, **arrays)
            manifest["sequences"].append(
                dict(
                    file=file.name,
                    sha256=sha256(file),
                    source_trajectory_id=source_id,
                    split=split,
                    factor=factor,
                    seed=seed,
                )
            )
        save_json(output / "manifest.json", manifest)
        print("factor source", seed, flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=10)
    args = parser.parse_args()
    generate(args.output, args.sources)


if __name__ == "__main__":
    main()
