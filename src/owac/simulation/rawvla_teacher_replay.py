"""Thin, bounded adapter around the author LIBERO replay generator.

This module never imports the simulator at import time and never starts a replay
unless the explicit runtime command is deliberately invoked.  It forces the
Normal lighting domain, caps attempted source episodes before each attempt, and
records failures rather than silently replacing or retrying a source.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from owac.raw.teacher_data import (
    CAMERA_ORDER,
    INSTRUCTION,
    LIGHTING,
    SCHEMA,
    TASK_ID,
    TASK_NAME,
    TASK_SUITE,
    assign_source_splits,
    canonical_policy_state,
    quaternion_to_axisangle,
    sha256_file,
)

AUTHOR_REVISION = "ba9c56b04da2c4931b70d8844710f5928d43775b"
AUTHOR_ARCHIVE_SHA256 = "57bc54e8a695ae07c2ffee9ebb2549e9efbb48536a97df14c5c8f63c52f3e012"
STARVLA_REVISION = "57d508cb180d0510ac9060358009c0f6ffaac420"
NORMAL_DOMAIN = "Normal"
MAX_REPLAY_SOURCES = 50
AUTHOR_REPLAY_REL = "scripts/replay/replay_libero_train_rawvla_light_npz.py"


class TeacherReplayBlocked(RuntimeError):
    """The exact replay contract or bounded runtime is unavailable."""


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    author_root: Path
    hdf5_root: Path
    out_root: Path
    rgb_out_root: Path
    seed: int = 20260813
    replay_mode: str = "states"
    representation: str = "raw"
    compression: str = "stored"


def _author_replay_path(author_root: Path) -> Path:
    path = author_root / AUTHOR_REPLAY_REL
    if not path.is_file():
        raise TeacherReplayBlocked(f"author replay script not found: {path}")
    return path


def inspect_source_hdf5(
    hdf5_root: str | Path,
    *,
    suite: str = TASK_SUITE,
    task_name: str = TASK_NAME,
) -> dict[str, Any]:
    """Read metadata only; no simulator/reset/step is performed."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment gate
        raise TeacherReplayBlocked("h5py is required for source inspection") from exc
    root = Path(hdf5_root)
    path = root / suite / f"{task_name}_demo.hdf5"
    layout = "suite_subdir"
    if not path.is_file():
        flat = root / f"{task_name}_demo.hdf5"
        if flat.is_file():
            path = flat
            layout = "flattened_task_file"
        else:
            raise TeacherReplayBlocked(f"task HDF5 is absent: {path}")
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise TeacherReplayBlocked(f"HDF5 lacks data group: {path}")
        keys = sorted(handle["data"].keys(), key=lambda value: int(value.split("_", 1)[1]))
        demos = []
        for key in keys:
            group = handle["data"][key]
            actions = np.asarray(group["actions"]) if "actions" in group else None
            states = np.asarray(group["states"]) if "states" in group else None
            robot = np.asarray(group["robot_states"]) if "robot_states" in group else None
            demos.append(
                {
                    "demo_key": key,
                    "num_frames": 0 if actions is None else int(actions.shape[0]),
                    "action_shape": None if actions is None else list(actions.shape),
                    "state_shape": None if states is None else list(states.shape),
                    "robot_state_shape": None if robot is None else list(robot.shape),
                    "has_model_file": "model_file" in group.attrs,
                }
            )
    return {
        "schema": "owac.raw-teacher.source-inspection.v1",
        "task": {"suite": suite, "task_id": TASK_ID, "task_name": task_name, "lighting": LIGHTING},
        "hdf5_path": str(path),
        "hdf5_layout": layout,
        "hdf5_sha256": sha256_file(path),
        "demo_count": len(demos),
        "source_demo_ids": [f"{suite}/{TASK_NAME}/{item['demo_key']}" for item in demos],
        "demos": demos,
        "simulator_imported": False,
    }


def _load_author_module(config: ReplayConfig) -> Any:
    path = _author_replay_path(config.author_root)
    spec = importlib.util.spec_from_file_location("owac_author_rawvla_replay", path)
    if spec is None or spec.loader is None:
        raise TeacherReplayBlocked(f"cannot load author replay module: {path}")
    expected_blob = "63d5cb16068a5557595c2ad947ec254aebdb7fb1"
    data = path.read_bytes()
    actual_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual_blob != expected_blob:
        raise TeacherReplayBlocked("author replay source differs from pinned Git blob")
    for extra in (config.author_root, config.author_root / "experiments/simulation/libero"):
        value = str(extra.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TeacherReplayBlocked(
            "author replay runtime dependencies/import failed; use the verified "
            f"author environment, no Lightwheel substitution: {exc}"
        ) from exc
    return module


def _normal_lighting(author: Any, ref: Any, *, seed: int) -> dict[str, Any]:
    # Author helper samples EV inside the requested domain only.  Do not use
    # its default random five-domain selector.
    return author._sample_lighting(
        seed=int(seed),
        domains=(NORMAL_DOMAIN,),
        suite=ref.suite,
        task_id=int(ref.task_id),
        demo_index=int(ref.demo_index),
        trajectory_id=int(ref.trajectory_id),
        variant_index=0,
        variant_domain_mode="cycle_domains",
    )


def _native_kept_indices(actions: np.ndarray, dones: np.ndarray | None, author: Any) -> np.ndarray:
    kept = []
    previous = None
    for index, action in enumerate(np.asarray(actions, dtype=np.float32)):
        if author._is_noop(action, previous):
            continue
        kept.append(index)
        previous = action
        if dones is not None and bool(dones[index]):
            break
    return np.asarray(kept, dtype=np.int64)


def _read_source_demo(ref: Any) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(ref.hdf5_path, "r") as handle:
        demo = handle["data"][ref.demo_key]
        out = {
            "actions": np.asarray(demo["actions"][()], dtype=np.float32),
            "states": np.asarray(demo["states"][()]),
            "robot_states": np.asarray(demo["robot_states"][()], dtype=np.float32),
            "dones": np.asarray(demo["dones"][()], dtype=np.uint8),
            "rewards": np.asarray(demo["rewards"][()], dtype=np.float32)
            if "rewards" in demo
            else np.zeros(len(demo["actions"]), dtype=np.float32),
            "model_file": str(demo.attrs.get("model_file", "")),
            "obs": {},
        }
        for key in ("ee_pos", "ee_ori", "gripper_states"):
            if key in demo.get("obs", {}):
                out["obs"][key] = np.asarray(demo["obs"][key][()], dtype=np.float32)
    return out


def _canonical_state(demo: Mapping[str, Any]) -> np.ndarray:
    obs = demo.get("obs", {})
    if not {"ee_pos", "ee_ori", "gripper_states"} <= set(obs):
        raise TeacherReplayBlocked(
            "HDF5 lacks obs/ee_pos, obs/ee_ori, obs/gripper_states for canonical state8"
        )
    state = canonical_policy_state(obs["ee_pos"], obs["ee_ori"], obs["gripper_states"])
    robot = np.asarray(demo["robot_states"], dtype=np.float64)
    if robot.shape != (state.shape[0], 9):
        raise TeacherReplayBlocked("source robot_states is not [N,9]")
    expected = np.concatenate(
        [
            robot[:, 2:5],
            quaternion_to_axisangle(robot[:, 5:9]),
            robot[:, 0:2],
        ],
        axis=1,
    ).astype(np.float32)
    if not np.allclose(state, expected, atol=2e-5, rtol=2e-5):
        raise TeacherReplayBlocked("canonical state8 differs from source robot9 interpretation")
    return state


def _save_arrays(path: Path, arrays: dict[str, np.ndarray], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise TeacherReplayBlocked(f"refusing to overwrite existing replay output: {path}")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        (np.savez_compressed if compressed else np.savez)(handle, **arrays)
    tmp.replace(path)


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def prepare_task6_normal(
    config: ReplayConfig,
    *,
    authority: dict[str, Any],
    repo_root: str | Path = ".",
    max_attempts: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare up to 50 author-replayed task6 Normal paired episodes.

    Attempts are written to a ledger *before* each replay.  Failures are retained
    in the full denominator and are never retried or replaced.  This function is
    not called by import, tests, or the status command.
    """
    root = Path(repo_root).resolve()
    caps = authority.get("limits", {}) if isinstance(authority, dict) else {}
    allowed = int(caps.get("demonstrations_replayed_max", 0) or 0)
    if not bool(authority.get("authorization", {}).get("demonstration_replay", False)):
        raise TeacherReplayBlocked("authority does not authorize demonstration replay")
    if allowed < MAX_REPLAY_SOURCES:
        raise TeacherReplayBlocked(f"authority permits only {allowed} replays; need 50")
    cap = MAX_REPLAY_SOURCES if max_attempts is None else min(int(max_attempts), MAX_REPLAY_SOURCES)
    if cap < MAX_REPLAY_SOURCES:
        raise TeacherReplayBlocked("the authorized task6 protocol requires exactly 50 source demos")
    source_inspection = inspect_source_hdf5(config.hdf5_root)
    source_ids = list(source_inspection["source_demo_ids"])
    if len(source_ids) < MAX_REPLAY_SOURCES:
        raise TeacherReplayBlocked(
            f"only {len(source_ids)} task6 source demos are present; refusing silent shrink"
        )
    split_map = assign_source_splits(source_ids, seed=101)
    selected = []
    for split, count in (("train", 40), ("validation", 5), ("test", 5)):
        selected.extend(
            [source_id for source_id in source_ids if split_map[source_id] == split][:count]
        )
    if len(selected) != MAX_REPLAY_SOURCES:
        raise TeacherReplayBlocked("cannot select the required 40/5/5 source set")
    if dry_run:
        return {
            "schema": "owac.raw-teacher.prepare-dry-run.v1",
            "selected_source_demo_ids": selected,
            "split_counts": {
                split: sum(split_map[source_id] == split for source_id in selected)
                for split in ("train", "validation", "test")
            },
            "cap": cap,
            "executed": False,
            "normal_domain": NORMAL_DOMAIN,
        }
    author = _load_author_module(config)
    if source_inspection["hdf5_layout"] == "flattened_task_file":
        from pathlib import Path as _Path

        hdf5_path = _Path(source_inspection["hdf5_path"])
        refs = [
            author.EpisodeRef(
                suite=TASK_SUITE,
                lerobot_dataset_name="libero_goal_no_noops_1.0.0_lerobot",
                task_id=TASK_ID,
                task_name=TASK_NAME,
                task_description=INSTRUCTION,
                hdf5_path=hdf5_path,
                demo_key=item["demo_key"],
                demo_index=int(item["demo_key"].split("_", 1)[1]),
                trajectory_id=index,
            )
            for index, item in enumerate(source_inspection["demos"])
        ]
    else:
        refs = [
            ref
            for ref in author._discover_episodes(config.hdf5_root, (TASK_SUITE,))
            if int(ref.task_id) == TASK_ID and ref.task_name == TASK_NAME
        ]
    ref_by_id = {f"{ref.suite}/{TASK_NAME}/{ref.demo_key}": ref for ref in refs}
    if set(selected) - set(ref_by_id):
        raise TeacherReplayBlocked("source inspection/discovery ids differ")
    ledger_path = config.out_root / "_manifests" / "attempts.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    attempted_ids: set[str] = set()
    if ledger_path.is_file():
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            attempts += 1
            try:
                attempted_ids.add(str(json.loads(line)["source_demo_id"]))
            except (KeyError, ValueError, TypeError) as exc:
                raise TeacherReplayBlocked("replay attempt ledger is malformed") from exc
    if attempts > cap:
        raise TeacherReplayBlocked("replay attempt ledger already exceeds the authority cap")
    manifest_path = config.out_root / "sources.json"
    sources = []
    completed_ids: set[str] = set()
    completion_path = config.out_root / "_manifests" / "completed.jsonl"
    if manifest_path.is_file() or completion_path.is_file():
        prior = (
            json.loads(manifest_path.read_text())
            if manifest_path.is_file()
            else {
                "sources": [
                    json.loads(line)
                    for line in completion_path.read_text().splitlines()
                    if line.strip()
                ]
            }
        )
        for entry in prior.get("sources", []):
            source_id = str(entry.get("source_demo_id", ""))
            raw_path = (
                root / str(entry["raw_path"])
                if not Path(str(entry["raw_path"])).is_absolute()
                else Path(str(entry["raw_path"]))
            )
            rgb_path = (
                root / str(entry["rgb_path"])
                if not Path(str(entry["rgb_path"])).is_absolute()
                else Path(str(entry["rgb_path"]))
            )
            if not raw_path.is_file() or not rgb_path.is_file():
                raise TeacherReplayBlocked(f"completed source files are missing: {source_id}")
            if sha256_file(raw_path) != entry.get("raw_sha256") or sha256_file(
                rgb_path
            ) != entry.get("rgb_sha256"):
                raise TeacherReplayBlocked(f"completed source hash changed: {source_id}")
            sources.append(entry)
            completed_ids.add(source_id)
    failures = []
    for source_id in selected:
        if source_id in completed_ids:
            continue
        if source_id in attempted_ids:
            # Never repeat a failed or ambiguous attempt.
            continue
        if attempts >= cap:
            break
        ref = ref_by_id[source_id]
        split = split_map[source_id]
        budget_root = Path(authority.get("run_root", config.out_root)).resolve()
        disk_limit = int(caps.get("additional_disk_bytes", 0))
        current_disk = sum(p.stat().st_size for p in budget_root.rglob("*") if p.is_file())
        expected_pair_bytes = (
            next(
                d["num_frames"] for d in source_inspection["demos"] if d["demo_key"] == ref.demo_key
            )
            * 4
            * 256
            * 256
            * 3
            + 1024 * 1024
        )
        if disk_limit <= 0 or current_disk + expected_pair_bytes > disk_limit:
            raise TeacherReplayBlocked("phase-wide disk budget cannot fit next source pair")
        with ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write(
                json.dumps(
                    {
                        "source_demo_id": source_id,
                        "split": split,
                        "attempt": attempts + 1,
                        "lighting": NORMAL_DOMAIN,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        attempts += 1
        try:
            source_demo = _read_source_demo(ref)
            native_indices = _native_kept_indices(
                source_demo["actions"], source_demo["dones"], author
            )
            native_anchor_mask = np.zeros(len(source_demo["actions"]), dtype=np.bool_)
            native_anchor_mask[native_indices] = True
            canonical_states = _canonical_state(source_demo)
            original_is_noop = author._is_noop
            author._is_noop = lambda *args, **kwargs: False
            try:
                episode = author._replay_episode(
                    ref,
                    seed=int(config.seed),
                    lighting=_normal_lighting(author, ref, seed=config.seed),
                    representation=config.representation,
                    replay_mode=config.replay_mode,
                )
            finally:
                author._is_noop = original_is_noop
            if episode is None:
                raise RuntimeError("author replay returned None")
            source_frames = int(source_demo["actions"].shape[0])
            if int(episode["actions"].shape[0]) != source_frames:
                raise RuntimeError("dense override did not retain every source frame")
            if canonical_states.shape != (source_frames, 8):
                raise RuntimeError("canonical state8 length differs from source frames")
            frame_indices = np.arange(source_frames, dtype=np.int64)
            relative = Path(ref.lerobot_dataset_name) / ref.task_name / f"{ref.demo_key}.npz"
            raw_path = config.out_root / relative
            rgb_path = config.rgb_out_root / relative
            raw_arrays = {
                "agentview_raw_uint8": episode["agentview_raw_uint8"],
                "wrist_raw_uint8": episode["wrist_raw_uint8"],
                "actions": episode["actions"].astype(np.float32),
                "states": episode["states"],
                "robot_states": canonical_states.astype(np.float32),
                "source_robot_states9": source_demo["robot_states"],
                "rewards": source_demo["rewards"],
                "dones": source_demo["dones"],
                "source_frame_indices": frame_indices,
                "native_anchor_mask": native_anchor_mask,
                "metadata_json": np.asarray(
                    json.dumps(
                        {
                            "source_demo_id": source_id,
                            "suite": ref.suite,
                            "task_id": ref.task_id,
                            "task_name": ref.task_name,
                            "lighting_domain": NORMAL_DOMAIN,
                            "representation": config.representation,
                            "replay_mode": config.replay_mode,
                            "noops_filtered": int(episode["noops_filtered"]),
                            "source_frame_indices_exact": True,
                            "native_anchor_mask": "exact original no-op/done mask",
                            "source_robot_states_order": "gripper2,eef_xyz3,quat_xyzw4",
                            "control_freq_hz": 20,
                            "dense_observation_history": True,
                            "public_generator_deviation": (
                                "dense camera/action frames retained; "
                                "native no-op filtering only marks action anchors"
                            ),
                            "source_recorded_success": bool(source_demo["dones"][-1]),
                            "model_xml_sha256": hashlib.sha256(
                                source_demo["model_file"].encode()
                            ).hexdigest(),
                        },
                        sort_keys=True,
                    )
                ),
            }
            rgb_arrays = {
                "agentview_rgb_uint8": episode["agentview_rgb_uint8"],
                "wrist_rgb_uint8": episode["wrist_rgb_uint8"],
                "actions": episode["actions"].astype(np.float32),
                "states": episode["states"],
                "robot_states": canonical_states.astype(np.float32),
                "source_robot_states9": source_demo["robot_states"],
                "rewards": source_demo["rewards"],
                "dones": source_demo["dones"],
                "source_frame_indices": frame_indices,
                "native_anchor_mask": native_anchor_mask,
            }
            _save_arrays(raw_path, raw_arrays, compressed=config.compression == "deflated")
            _save_arrays(rgb_path, rgb_arrays, compressed=config.compression == "deflated")
            sources.append(
                {
                    "source_demo_id": source_id,
                    "split": split,
                    "raw_path": _relative_or_absolute(raw_path, root),
                    "rgb_path": _relative_or_absolute(rgb_path, root),
                    "raw_sha256": sha256_file(raw_path),
                    "rgb_sha256": sha256_file(rgb_path),
                    "num_frames": int(episode["actions"].shape[0]),
                    "task_name": TASK_NAME,
                    "lighting": NORMAL_DOMAIN,
                    "camera_order": list(CAMERA_ORDER),
                    "metadata": {
                        "source_hdf5": str(ref.hdf5_path),
                        "demo_key": ref.demo_key,
                        "noops_filtered": int(episode["noops_filtered"]),
                        "source_frame_indices_exact": True,
                        "representation": "pseudo_raw3_uint8_over_255",
                    },
                }
            )
            with completion_path.open("a", encoding="utf-8") as completed:
                completed.write(json.dumps(sources[-1], sort_keys=True) + "\n")
                completed.flush()
                os.fsync(completed.fileno())
            print(
                json.dumps(
                    {
                        "prepared_sources": len(sources),
                        "attempts": attempts,
                        "source_demo_id": source_id,
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # retain full attempt denominator
            failures.append({"source_demo_id": source_id, "split": split, "error": str(exc)})
            break
    if failures or len(sources) != MAX_REPLAY_SOURCES:
        failure_path = config.out_root / "_manifests" / "failures.json"
        failure_path.write_text(
            json.dumps(
                {"attempts": attempts, "completed": len(sources), "failures": failures},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise TeacherReplayBlocked(
            f"replay incomplete: {len(sources)}/{MAX_REPLAY_SOURCES} sources; failures retained"
        )
    manifest = {
        "schema": SCHEMA,
        "task": {
            "suite": TASK_SUITE,
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "instruction": INSTRUCTION,
            "lighting": LIGHTING,
        },
        "provenance": {
            "author_revision": AUTHOR_REVISION,
            "author_archive_sha256": AUTHOR_ARCHIVE_SHA256,
            "starvla_revision": STARVLA_REVISION,
            "replay_domain": NORMAL_DOMAIN,
            "replay_representation": "pseudo_raw3_uint8_over_255",
            "source_hdf5_sha256": source_inspection["hdf5_sha256"],
        },
        "source_hdf5_sha256": source_inspection["hdf5_sha256"],
        "split_seed": 101,
        "path_semantics": "repo_relative"
        if all(not Path(item["raw_path"]).is_absolute() for item in sources)
        else "absolute",
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "schema": "owac.raw-teacher.prepare.v1",
        "manifest": str(manifest_path),
        "attempts": attempts,
    }


__all__ = [
    "AUTHOR_ARCHIVE_SHA256",
    "AUTHOR_REVISION",
    "AUTHOR_REPLAY_REL",
    "MAX_REPLAY_SOURCES",
    "NORMAL_DOMAIN",
    "ReplayConfig",
    "TeacherReplayBlocked",
    "inspect_source_hdf5",
    "prepare_task6_normal",
]
