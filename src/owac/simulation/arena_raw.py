"""Native CFA cameras mirroring configured Arena DROID RGB views."""

from __future__ import annotations

import re
import time
from pathlib import Path

from owac.runtime.arena_droid import sha256, write_json
from owac.simulation.raw_camera import NativeRawCamera, RawCameraProfile


def opengl_pose_matrix(position, quaternion_xyzw):
    """The pinned Fabric camera view uses xyzw; Gf accepts w then xyz."""
    from pxr import Gf

    x, y, z, w = (float(value) for value in quaternion_xyzw)
    matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    matrix.SetTranslateOnly(Gf.Vec3d(*(float(value) for value in position)))
    return matrix


class ArenaRawCapture:
    def __init__(
        self,
        env,
        output: Path,
        *,
        profile: RawCameraProfile | None = None,
        camera_names=("external_camera", "wrist_camera"),
    ):
        import omni.usd
        from pxr import UsdGeom

        self.env = env.unwrapped
        self.output = output
        self.cameras = {}
        self.selected = {}
        self.frames = 0
        self.maximum_quantization_error_dn = 0
        stage = omni.usd.get_context().get_stage()
        profile = profile or RawCameraProfile(width=1280, height=720)
        if not camera_names or len(set(camera_names)) != len(camera_names):
            raise ValueError("Native camera names must be nonempty and unique")
        for name in camera_names:
            source = self.env.scene.sensors[name]
            # The compiled path may use env_[^/]+; use the camera's resolved USD handle.
            source_path = source._sensor_prims[0].GetPrim().GetPath().pathString
            source_prim = stage.GetPrimAtPath(source_path)
            if not source_prim.IsValid():
                raise RuntimeError(f"Missing official RGB camera prim: {source_path}")
            path = f"/World/OWACRaw_{name}"
            camera = NativeRawCamera(
                output / "raw" / name, profile, (0, -1, 1), (0, 0, 0), camera_path=path
            )
            target = stage.GetPrimAtPath(path)
            optics = {}
            for attribute in (
                "focalLength",
                "horizontalAperture",
                "verticalAperture",
                "horizontalApertureOffset",
                "verticalApertureOffset",
                "clippingRange",
                "focusDistance",
                "fStop",
            ):
                src = source_prim.GetAttribute(attribute)
                if src.IsValid() and src.HasAuthoredValueOpinion():
                    value = src.Get()
                    target.GetAttribute(attribute).Set(value)
                    optics[attribute] = (
                        [float(value[0]), float(value[1])]
                        if attribute == "clippingRange"
                        else float(value)
                    )
            xform = UsdGeom.Xformable(target)
            xform.ClearXformOpOrder()
            operation = xform.AddTransformOp()
            camera.metadata.update(
                source_rgb_prim=source_path,
                camera_eye_xyz=None,
                camera_target_xyz=None,
                usd_focal_length=optics.get("focalLength"),
                optical_attributes=optics,
                pose_source="Camera Fabric view world poses (OpenGL xyzw), queried per step",
                saturation_mask="derive from stored CFA == white_level; no separate file",
                sampling_phase="pre-action; render without physics advance; last reset excluded",
                noise_range="native post-CFA noise may exceed CFA white; stored without clipping",
            )
            write_json(output / f"sensor-{name}.json", camera.metadata)
            self.cameras[name] = (source, camera, operation)
            self.selected[name] = set()
        self.warm = False

    def capture(self, step):
        from isaacsim.core.rendering_manager import RenderingManager

        from owac.simulation.arena_droid_rollout import array

        before = int(self.env.common_step_counter)
        physics_before = int(self.env.sim.get_physics_step_count())
        poses = {}
        for name, (source, _, operation) in self.cameras.items():
            # CameraCfg caches poses by default. Query the moving wrist's Fabric view.
            positions, orientations = source._view.get_world_poses()
            position = array(positions)[0]
            quaternion = array(orientations)[0]
            operation.Set(opengl_pose_matrix(position, quaternion))
            poses[name] = {
                "position_world": position.tolist(),
                "quaternion_opengl_xyzw": quaternion.tolist(),
            }
        renders = 0
        if not self.warm:
            for _ in range(5):
                RenderingManager.render()
                renders += 1
            self.warm = True
        RenderingManager.render()
        renders += 1
        record = {}
        for name, (_, camera, _) in self.cameras.items():
            for attempt in range(8):
                try:
                    _, noisy, audit = camera.capture()
                    break
                except RuntimeError:
                    if attempt == 7:
                        raise
                    RenderingManager.render()
                    renders += 1
            # Stage 3 adds noise after the CFA quantizer. Preserve its native output,
            # including overshoot, instead of imposing the stage-2 clipping range.
            audit["noise"] = {
                "dtype": str(noisy.dtype),
                "minimum_dn": int(noisy.min()),
                "maximum_dn": int(noisy.max()),
                "above_cfa_white_fraction": float((noisy > camera.profile.white_level).mean()),
                "processing": "native stage 3; no host clipping or normalization",
            }
            frame = audit["native_frame_id"]
            self.selected[name].add(frame)
            files = {}
            for kind in ("0-texread", "2-cfa", "3-noise"):
                path = camera.directory / f"{kind}{frame}.bin"
                files[kind] = {
                    "path": str(path.relative_to(self.output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            record[name] = {**audit, **poses[name], "files": files}
            self.maximum_quantization_error_dn = max(
                self.maximum_quantization_error_dn, audit["cfa_quantization_maximum_error_dn"]
            )
            # Retain HDR, CFA and native noise; auxiliary ISP buffers are unused.
            for path in camera.directory.glob("*.bin"):
                match = re.fullmatch(r"(0-texread|2-cfa|3-noise)(\d+)\.bin", path.name)
                if match is None or int(match[2]) not in self.selected[name]:
                    path.unlink()
        if int(self.env.common_step_counter) != before:
            raise RuntimeError("RAW render advanced physics")
        if int(self.env.sim.get_physics_step_count()) != physics_before:
            raise RuntimeError("RAW render advanced simulator physics substeps")
        self.frames += 1
        return {
            "step": step,
            "phase": "pre_action",
            "physical_step_counter": before,
            "physics_substep_counter": physics_before,
            "host_timestamp_ns": time.time_ns(),
            "extra_render_calls": renders,
            "views": record,
        }

    def summary(self):
        return {
            "control_frames": self.frames,
            "views": {
                name: {
                    "frames": len(ids),
                    "first_native_frame_id": min(ids) if ids else None,
                    "last_native_frame_id": max(ids) if ids else None,
                }
                for name, ids in self.selected.items()
            },
            "maximum_quantization_error_dn": self.maximum_quantization_error_dn,
            "scope": "synthetic native scene HDR to CFA; physical noise calibration is not claimed",
        }
