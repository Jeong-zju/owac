"""Disclosed source-action grasp assistance for offline dataset reconstruction only."""

from __future__ import annotations

import numpy as np

from owac.runtime.polaris_replay import transform


def relative_pose_xyzw(parent, child):
    """Return the child's current pose in the parent's frame, without moving either."""
    parent, child = np.asarray(parent), np.asarray(child)
    if parent.shape != (7,) or child.shape != (7,) or not np.all(np.isfinite([parent, child])):
        raise ValueError("Grasp anchors require finite xyz/xyzw poses")
    if any(abs(np.linalg.norm(p[3:]) - 1) > 1e-3 for p in (parent, child)):
        raise ValueError("Grasp anchors require unit quaternions")
    px, py, pz, pw = parent[3:]
    cx, cy, cz, cw = child[3:]
    local = np.linalg.inv(transform(parent[:3], [pw, px, py, pz])) @ transform(
        child[:3], [cw, cx, cy, cz]
    )
    # Quaternion product conjugate(parent) * child, scalar last.
    a, b = -parent[3:6], child[3:6]
    quat = np.r_[pw * b + cw * a + np.cross(a, b), pw * cw - np.dot(a, b)]
    quat /= np.linalg.norm(quat)
    return local[:3, 3], quat


def validate_schedule(actions, schedule):
    """An explicit source association may only span an actual close/open command cycle."""
    actions = np.asarray(actions)
    starts = set()
    for cycle in schedule:
        start, end = cycle["close"], cycle["open"]
        if not (0 <= start < end < len(actions)) or start in starts:
            raise ValueError("Invalid or duplicate source grasp cycle")
        if (
            actions[start, 7] <= 0.5
            or (start > 0 and actions[start - 1, 7] > 0.5)
            or np.any(actions[start:end, 7] <= 0.5)
            or actions[end, 7] > 0.5
        ):
            raise ValueError("Constraint schedule does not match original gripper actions")
        if not isinstance(cycle["object"], str):
            raise ValueError("Source association must name the reconstructed object")
        if not start <= cycle.get("not_before", start) < end:
            raise ValueError("Attachment readiness must stay within the original closed cycle")
        starts.add(start)


class SourceGraspConstraints:
    """Enable a maximum-coordinate USD fixed joint at the actual, unsnapped grasp pose.

    The source schedule identifies objects; proximity is measured in the new simulation.
    No root-state writes, attachment teleport, or changes to source actions are performed.
    A failed proximity check remains a recorded failure rather than forcing attachment.
    """

    def __init__(self, stage, gripper_path, object_paths, specification, actions):
        from pxr import Sdf, UsdPhysics

        self.specification = specification
        self.schedule = specification["cycles"]
        validate_schedule(actions, self.schedule)
        self.joints = {}
        self.active = None
        self.events = []
        self.relative_errors = []
        self.joint_anchor = None
        self.missed = set()
        group = gripper_path.rsplit("/", 1)[0] + "/"
        self.gripper_collision_bodies = [
            Sdf.Path(str(p.GetPath()))
            for p in stage.Traverse()
            if str(p.GetPath()).startswith(group) and p.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        self.collision_filters = {}
        for name, path in object_paths.items():
            self.collision_filters[name] = UsdPhysics.FilteredPairsAPI.Apply(
                stage.GetPrimAtPath(path)
            ).CreateFilteredPairsRel()
            joint = UsdPhysics.FixedJoint.Define(stage, f"/World/source_grasp_{name}")
            joint.CreateBody0Rel().SetTargets([Sdf.Path(gripper_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
            joint.CreateExcludeFromArticulationAttr(True)
            joint.CreateCollisionEnabledAttr(False)
            joint.CreateJointEnabledAttr(False)
            self.joints[name] = joint
        if set(c["object"] for c in self.schedule) - set(object_paths):
            raise ValueError("Constraint schedule refers to an absent scene object")

    def update(self, frame, action, gripper_pose, tcp, object_poses):
        from pxr import Gf, Sdf

        emitted = []
        if self.active is not None:
            local, quat = relative_pose_xyzw(gripper_pose, object_poses[self.active])
            position_error = float(np.linalg.norm(local - self.joint_anchor[0]))
            rotation_error = float(
                2 * np.arccos(np.clip(abs(np.dot(quat, self.joint_anchor[1])), 0, 1))
            )
            self.relative_errors.append([position_error, rotation_error])
            if action[7] <= 0.5:
                with Sdf.ChangeBlock():
                    self.joints[self.active].GetJointEnabledAttr().Set(False)
                    self.collision_filters[self.active].SetTargets([])
                emitted.append({"event": "release", "object": self.active, "frame": frame})
                self.active = None
                self.joint_anchor = None
        if self.active is None and action[7] > 0.5 and frame is not None:
            cycle = next((c for c in self.schedule if c["close"] <= frame < c["open"]), None)
            if (
                cycle is not None
                and cycle.get("not_before", cycle["close"]) <= frame
                and frame <= cycle["close"] + self.specification["wait_frames"]
            ):
                name = cycle["object"]
                distance = float(np.linalg.norm(np.asarray(object_poses[name])[:3] - tcp))
                if distance <= self.specification["maximum_tcp_distance_m"]:
                    local, quat = relative_pose_xyzw(gripper_pose, object_poses[name])
                    joint = self.joints[name]
                    with Sdf.ChangeBlock():
                        joint.CreateLocalPos0Attr(Gf.Vec3f(*local))
                        joint.CreateLocalRot0Attr(Gf.Quatf(float(quat[3]), Gf.Vec3f(*quat[:3])))
                        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
                        joint.CreateLocalRot1Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
                        self.collision_filters[name].SetTargets(self.gripper_collision_bodies)
                        joint.GetJointEnabledAttr().Set(True)
                    self.active, self.joint_anchor = name, (local, quat)
                    emitted.append(
                        {
                            "event": "attach",
                            "object": name,
                            "frame": frame,
                            "source_close_frame": cycle["close"],
                            "tcp_distance_m": distance,
                            "gripper_pose_before_xyzw": list(gripper_pose),
                            "object_pose_before_xyzw": list(object_poses[name]),
                            "local_position_m": local.tolist(),
                            "local_quaternion_xyzw": quat.tolist(),
                            "object_pose_write_performed": False,
                        }
                    )
                elif cycle["close"] not in self.missed:
                    emitted.append(
                        {
                            "event": "attachment_waiting_for_proximity",
                            "object": name,
                            "frame": frame,
                            "source_close_frame": cycle["close"],
                            "tcp_distance_m": distance,
                        }
                    )
                    self.missed.add(cycle["close"])
        self.events.extend(emitted)
        return {"active_after_command": self.active, "events": emitted}

    def summary(self):
        errors = np.asarray(self.relative_errors)
        attachments = [e for e in self.events if e["event"] == "attach"]
        releases = [e for e in self.events if e["event"] == "release"]
        return {
            "method": "source_action_fixed_joint",
            "constraint_assisted": True,
            "source_schedule": self.schedule,
            "events": self.events,
            "attachments": len(attachments),
            "releases": len(releases),
            "active_at_end": self.active,
            "maximum_relative_position_error_m": float(errors[:, 0].max()) if len(errors) else None,
            "maximum_relative_rotation_error_rad": float(errors[:, 1].max())
            if len(errors)
            else None,
            "object_state_overrides_after_initialization": 0,
            "joint_collision_enabled": False,
            "gripper_object_collisions_filtered_only_while_attached": True,
            "physics_joint_excluded_from_articulation": True,
            "free_contact_expert_replay": False,
        }
