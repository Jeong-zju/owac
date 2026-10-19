"""Capture same-buffer pre/post sensor noise for diagnosis; no learned policy or success."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def get_model(arguments):
    import sapien

    original = sapien.render.set_ray_tracing_denoiser
    sapien.render.set_ray_tracing_denoiser = lambda name: original(
        "optix" if name == "oidn" else name
    )
    return {"captured": False}


def reset_model(model):
    model["captured"] = False


def eval(task, model, observation):
    if model["captured"]:
        raise RuntimeError("sensor capture must stop after one holding command")
    import envs.camera.raw_rgb10 as sensor
    from envs.camera.camera import _stable_uint32

    manager = task.cameras
    if not manager.raw_sensor_noise_enabled or manager.hdr_raw_white_level <= 0:
        raise ValueError("capture requires the declared noisy HDR RAW path")
    cameras = (
        next(
            c
            for c, n in zip(manager.static_camera_list, manager.static_camera_name, strict=True)
            if n == "head_camera"
        ),
        manager.left_camera,
        manager.right_camera,
    )
    names = ("head_camera", "left_camera", "right_camera")
    frame = manager.raw_sensor_noise_frame_index - 1
    payload = {"state": observation["joint_action"]["vector"]}
    rows = []
    for i, (camera, name) in enumerate(zip(cameras, names, strict=True)):
        hdr = camera.get_picture("HdrColor")[:, :, :3].astype(np.float32)
        clean = sensor.hdr_linear_to_raw_float(hdr, manager.hdr_raw_white_level)
        clean = sensor.apply_full_well_saturation_float(clean, manager.sensor_saturation_ev)
        seed = _stable_uint32("lighting_sensor_noise", manager.raw_sensor_noise_seed, frame, i)
        rebuilt = sensor.add_sensor_noise(clean, seed=seed, noise=manager.raw_sensor_noise)
        actual = observation["observation"][name]["raw_linear"]
        delta = float(np.max(np.abs(rebuilt - actual)))
        if delta != 0:
            raise ValueError(f"same HDR/noise buffer identity failed for {name}: {delta}")
        payload.update(
            {
                f"clean_{i}": clean,
                f"raw_{i}": actual,
                f"rgb_{i}": observation["observation"][name]["render_rgb"],
            }
        )
        for j in range(3):
            payload[f"redraw{j}_{i}"] = sensor.add_sensor_noise(
                clean, seed=seed + 1009 * (j + 1), noise=manager.raw_sensor_noise
            )
        rows.append(
            dict(
                camera=name,
                seed=seed,
                max_rebuild_difference=delta,
                clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
                pixel_noise_rms=float(np.mean((actual - clean) ** 2) ** 0.5),
            )
        )
    root = Path(os.environ["N02_SENSOR_CAPTURE_DIR"])
    np.savez_compressed(root / "sensor_capture.npz", **payload)
    (root / "sensor_capture.json").write_text(
        json.dumps(
            dict(
                diagnostic_only=True,
                policy_success_claim=False,
                same_render_buffer=True,
                sensor_source_sha256=hashlib.sha256(Path(sensor.__file__).read_bytes()).hexdigest(),
                frame=frame,
                white_level=manager.hdr_raw_white_level,
                shot_noise=manager.raw_sensor_noise.shot_noise,
                read_noise=manager.raw_sensor_noise.read_noise,
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    model["captured"] = True
    task.step_lim = 1
    # One hold at the observed current state terminates the standard evaluator.
    task.take_action(np.asarray(observation["joint_action"]["vector"]))
