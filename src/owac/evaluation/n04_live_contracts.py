"""Recompute S0 contracts from saved live arrays and timestamps, never PASS fields."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from owac.raw.n04_live_collection import save_json, sha256
from owac.raw.telemetry import ExecutedControl, FrameTiming, TelemetrySidecar


def temporal_audit(record: dict) -> dict:
    timings = tuple(
        FrameTiming(
            frame=f["index"],
            clock_id="host",
            time_unit="s",
            exposure_start=f["host_sample_s"],
            exposure_mid=f["host_sample_s"],
            exposure_end=f["host_sample_s"],
            arrival_time=f["host_arrival_s"],
            source=f["file"],
        )
        for f in record["frames"]
    )
    controls = tuple(
        ExecutedControl(
            command_id=c["command_id"],
            value=np.asarray(c["value"]),
            clock_id="host",
            time_unit="s",
            command_time=c["start"],
            start_time=c["start"],
            end_time=c["end"],
            frame=c["frame"],
            status="executed",
            evidence_level=c["evidence_level"],
            source=c["source"],
        )
        for c in record["controls"]
    )
    sidecar = TelemetrySidecar(
        record["source_trajectory_id"], time_unit="s", timing=timings, controls=controls
    )
    sidecar.check_time_domain()
    counts = dict(
        decisions=0,
        delayed_frames=0,
        withheld_checks=0,
        replayed_controls=0,
        future_poison_checks=0,
        out_of_order_pairs=0,
    )
    failures = []
    for frame in record["frames"]:
        if frame["host_arrival_s"] < frame["host_render_done_s"]:
            failures.append("arrival precedes rendering completion")
        if frame["host_arrival_s"] <= frame["host_sample_s"]:
            failures.append("nonpositive observed delivery latency")
        counts["delayed_frames"] += int(frame["queue_delayed"])
        for decision in (frame["host_render_done_s"], frame["host_arrival_s"]):
            ctx = sidecar.context_for(exposure_time=decision, decision_time=decision)
            expected = tuple(
                f["index"]
                for f in record["frames"]
                if f["host_sample_s"] <= decision and f["host_arrival_s"] <= decision
            )
            if ctx.visible_frames != expected:
                failures.append("arrival selection differs from independent filter")
            if decision < frame["host_arrival_s"]:
                counts["withheld_checks"] += 1
                if frame["index"] in ctx.visible_frames:
                    failures.append("withheld image leaked")
            if ctx.latest_exposure_time is not None:
                ids = [
                    c.command_id
                    for c in controls
                    if c.start_time < decision
                    and c.end_time > ctx.latest_exposure_time
                    and c.command_time <= decision
                ]
                if [c.command_id for c in ctx.replay_controls] != ids:
                    failures.append("executed interval selection mismatch")
                for c in ctx.replay_controls:
                    if c.end_time > decision or c.start_time < ctx.latest_exposure_time:
                        failures.append("unclipped execution interval")
            # Poison all controls not yet issued. Available context must be unchanged.
            altered = tuple(
                replace(c, value=np.full(np.shape(c.value), 1e20, dtype=np.float64))
                if c.command_time > decision
                else c
                for c in controls
            )
            poison = replace(sidecar, controls=altered).context_for(
                exposure_time=decision, decision_time=decision
            )
            if len(ctx.replay_controls) != len(poison.replay_controls) or any(
                not np.array_equal(a.value, b.value)
                for a, b in zip(ctx.replay_controls, poison.replay_controls, strict=True)
            ):
                failures.append("future control values leaked")
            counts["future_poison_checks"] += 1
            counts["replayed_controls"] += len(ctx.replay_controls)
            counts["decisions"] += 1
    counts["out_of_order_pairs"] = sum(
        a.arrival_time > b.arrival_time for a, b in zip(timings[:-1], timings[1:], strict=True)
    )
    if not counts["out_of_order_pairs"] or not counts["replayed_controls"]:
        failures.append("no actual out-of-order/control replay coverage")
    return dict(
        passed=not failures,
        failures=failures,
        **counts,
        latency_s=[
            min(t.arrival_delay() for t in timings),
            max(t.arrival_delay() for t in timings),
        ],
    )


def validate_sequence(path: Path) -> dict:
    record = json.loads(path.read_text())
    checks = dict(
        input_boundary=record["schema"] == "n04-live-extension-v2"
        and record["metadata"].get("pixel_coordinates") == "integer_pixel_centres_top_left",
        extension=record["mode"] == "independent_extension"
        and not record["source_correspondence_verified"],
        frame_count=len(record["frames"]) >= 12
        and len(record["frames"]) == len(record["controls"]),
        hashes=True,
        finite=True,
        geometry=True,
        trace=True,
        timing_alignment=True,
    )
    previous = None
    camera_frames = contacts = 0
    metric = dict(continuity_max=0.0, depth_error_m=0.0, projection_error_px=0.0, ray_ratio=0.0)
    calibration = []
    for frame, control in zip(record["frames"], record["controls"], strict=True):
        filename = path.parent / frame["file"]
        checks["hashes"] &= sha256(filename) == frame["sha256"]
        with np.load(filename, allow_pickle=False) as data:
            checks["finite"] &= all(np.isfinite(data[k]).all() for k in data.files)
            raw, depth = data["online_rawvla_rgb3"], data["gt_depth_m"]
            if record["metadata"]["platform"] == "libero":
                for component in ("qpos", "qvel"):
                    if f"gt_sim_{component}" not in data:
                        checks["input_boundary"] = False
                    else:
                        indices = data[f"audit_robot_{component}_indices"]
                        checks["input_boundary"] &= np.array_equal(
                            data[f"online_{component}"], data[f"gt_sim_{component}"][indices]
                        )
                        checks["input_boundary"] &= len(indices) < len(data[f"gt_sim_{component}"])
            support = (
                data["gt_geometry_support"]
                if "gt_geometry_support" in data
                else data["gt_visible_support"]
            )
            checks["geometry"] &= raw.shape[:-1] == depth.shape == support.shape and support.any()
            pose, k = data["gt_T_W_from_C"], data["online_K"]
            rot = pose[:, :3, :3]
            checks["geometry"] &= bool(np.max(abs(rot.swapaxes(-1, -2) @ rot - np.eye(3))) < 1e-5)
            checks["geometry"] &= bool(np.all(np.linalg.det(rot) > 0))
            if previous is not None:
                metric["continuity_max"] = max(
                    metric["continuity_max"], float(np.max(abs(data["trace_pre_state"] - previous)))
                )
            previous = data["trace_post_state"].copy()
            checks["trace"] &= np.array_equal(data["trace_issued_command"], control["value"])
            if record["metadata"]["platform"] == "robotwin2":
                layout = record["metadata"].get("command_layout", [])
                checks["trace"] &= len(layout) * 2 == len(control["value"])
                for index, joint in enumerate(layout):
                    checks["trace"] &= joint["joint_type"] in (
                        "prismatic",
                        "revolute",
                        "revolute_unwrapped",
                    )
                    units = ("m", "m/s") if joint["joint_type"] == "prismatic" else ("rad", "rad/s")
                    checks["trace"] &= (joint["position_unit"], joint["velocity_unit"]) == units
                    checks["trace"] &= joint["value_indices"] == [2 * index, 2 * index + 1]
            checks["timing_alignment"] &= (
                frame["index"] == control["frame"]
                and frame["host_render_done_s"] <= control["start"] < control["end"]
                and frame["sim_time_s"] == control["sim_start"] < control["sim_end"]
            )
            camera_frames += raw.shape[0]
            contacts += int(data["gt_contact_count"])
            if "audit_ray_matching_geom" in data:
                checks["geometry"] &= bool(data["audit_ray_matching_geom"].mean() >= 0.8)
                if "audit_ray_reference_depth_m" in data:
                    sampled = depth[:, (32, 64, 128, 192, 224), :][
                        :, :, (32, 64, 128, 192, 224)
                    ].reshape(-1)
                    matched = data["audit_ray_matching_geom"]
                    independent_error = abs(sampled - data["audit_ray_reference_depth_m"])
                    checks["geometry"] &= bool(
                        np.all(independent_error[matched] <= data["audit_ray_bound_m"][matched])
                    )
                ratio = float(np.max(data["audit_ray_error_m"] / data["audit_ray_bound_m"]))
                metric["ray_ratio"] = max(metric["ray_ratio"], ratio)
                checks["geometry"] &= ratio <= 1
            else:
                # Recompute projection from saved camera-space position, independent
                # of the collector's aggregate error. Grid uses canonical integer pixel centres.
                xyz = data["audit_xyz_C"]
                projected = np.einsum("vij,vhwj->vhwi", k, xyz)
                z = xyz[..., 2]
                valid = z > 1e-5
                vv, uu = np.indices(z.shape[1:])
                expected = np.stack((uu * 32, vv * 32), -1)
                xy = projected[..., :2] / np.where(valid, z, 1)[..., None]
                error = float(
                    np.max(abs(xy[valid] - np.broadcast_to(expected, xy.shape)[valid]), initial=0)
                )
                metric["projection_error_px"] = max(metric["projection_error_px"], error)
                checks["geometry"] &= (
                    error < 0.01 and float(data["audit_extrinsic_error"].max()) < 1e-5
                )
                err = float(data["audit_depth_error_m"].max())
                metric["depth_error_m"] = max(metric["depth_error_m"], err)
                checks["geometry"] &= err < 1e-5
            for view, name in enumerate(record["metadata"]["camera_names"]):
                good = support[view] & (depth[view] > 1e-5)
                # Model-independent calibration references use train sources only.
                if record["split"] == "train" and good.any():
                    calibration.append(
                        dict(
                            camera=name,
                            focal_px=float(k[view, 0, 0]),
                            median_depth_m=float(np.median(depth[view][good])),
                            m_per_px=float(np.median(depth[view][good]) / k[view, 0, 0]),
                        )
                    )
    checks["trace"] &= metric["continuity_max"] <= 1e-12
    temporal = temporal_audit(record)
    checks["temporal"] = temporal["passed"]
    return dict(
        path=str(path),
        record_sha256=sha256(path),
        source_trajectory_id=record["source_trajectory_id"],
        split=record["split"],
        platform=record["metadata"]["platform"],
        task=record["metadata"]["task"],
        frames=len(record["frames"]),
        camera_frames=camera_frames,
        contacts=contacts,
        checks={k: bool(v) for k, v in checks.items()},
        metrics=metric,
        temporal=temporal,
        train_calibration=calibration,
        passed=bool(all(checks.values())),
    )


def audit(root: Path) -> dict:
    records = [validate_sequence(p) for p in sorted(root.glob("episode-*/record.json"))]
    groups = {
        s: {r["source_trajectory_id"] for r in records if r["split"] == s}
        for s in ("train", "dev", "final")
    }
    disjoint = all(
        not groups[a] & groups[b]
        for a, b in [("train", "dev"), ("train", "final"), ("dev", "final")]
    )
    platform_coverage = {
        p: {s: sum(r["platform"] == p and r["split"] == s for r in records) for s in groups}
        for p in ("libero", "robotwin2")
    }
    matrix = all(all(n > 0 for n in counts.values()) for counts in platform_coverage.values())
    run_path = root / "run.json"
    expected_complete = False
    if run_path.exists():
        run = json.loads(run_path.read_text())
        plan = run.get("resolved_plan", run["plan"])
        actual = {(r["source_trajectory_id"], r["split"], r["platform"]) for r in records}
        expected = {
            (e["source_trajectory_id"], e["split"], e["platform"]) for e in plan["episodes"]
        }
        expected_complete = (
            actual == expected
            and len(records) == len(plan["episodes"])
            and all(r["frames"] == plan["frames_per_episode"] for r in records)
        )
    return dict(
        schema="n04-live-audit-v1",
        records=records,
        split_counts={s: len(g) for s, g in groups.items()},
        source_disjoint=disjoint,
        planned_collection_complete=expected_complete,
        platform_split_coverage=platform_coverage,
        frames=sum(r["frames"] for r in records),
        camera_frames=sum(r["camera_frames"] for r in records),
        passed=bool(records)
        and disjoint
        and matrix
        and expected_complete
        and all(r["passed"] for r in records),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    save_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
