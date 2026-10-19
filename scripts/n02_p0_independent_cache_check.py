#!/usr/bin/env python3
"""Rebuild task-stratified P0 manifests and verify actual loader values against NPZ indexing.

Cache reads only; output must be an independent directory. No torch or policy is loaded.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
from pathlib import Path

import numpy as np

from owac.frameworks.n02_full.data import (
    ManifestWindowDataset,
    PartitionManifest,
    build_anchor_table,
    build_partition_manifest,
    load_formal_normalisation,
    verify_partitions,
)
from owac.raw.rawvla_sequence_n02 import (
    FORMAT_SPECS,
    EpisodeRef,
    _acquisition_source,
    _episode_identity,
)


def run(root: Path, metadata_path: Path, checkpoint_root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    hf = json.loads(metadata_path.read_text())
    pins = {entry["path"]: entry for entry in hf["files"] if entry["path"].endswith(".npz")}
    summary = {
        "source_revision": hf["revision"],
        "hash_provenance": "HF LFS; not recomputed over full cache",
        "platforms": {},
    }
    for fmt, directory in (("libero", "libero"), ("robotwin", "robotwin2")):
        episodes, metadata = [], {}
        for path in sorted((root / "data" / directory / "raw").rglob("*.npz")):
            relative = path.relative_to(root).as_posix()
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata_json"]))
                frames = z["actions"].shape[0]
            metadata[relative] = meta
            pin = pins[relative]
            assert path.stat().st_size == pin["size"], relative
            episodes.append(
                EpisodeRef(
                    str(path),
                    relative,
                    _acquisition_source(relative),
                    _episode_identity(relative),
                    frames,
                    pin["size"],
                    pin["lfs"]["sha256"],
                )
            )
        manifest = build_partition_manifest(
            episodes, dataset_format=fmt, cache_root=root, seed=1103
        )
        (output / f"manifest_{fmt}.json").write_text(manifest.to_json())
        verify = verify_partitions(manifest)
        assert not verify["violations"], verify
        assert (
            PartitionManifest.from_json(manifest.to_json()).content_hash()
            == manifest.content_hash()
        )
        tasks = collections.defaultdict(collections.Counter)
        for partition, paths in manifest.partitions.items():
            for path in paths:
                tasks[manifest.entry_of(path).task_id][partition] += 1
        stats_path = checkpoint_root / (
            "pi05_robotwin/policy_norm_stats.json"
            if fmt == "robotwin"
            else "pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json"
        )
        stats = load_formal_normalisation(fmt, stats_path)
        selected, used_tasks = {}, set()
        for path, meta in metadata.items():
            domain = meta["lighting_domain"]
            task = manifest.entry_of(path).task_id
            if domain not in selected and task not in used_tasks:
                selected[domain] = path
                used_tasks.add(task)
        assert len(selected) == 5
        comparisons = []
        for index, (domain, path) in enumerate(sorted(selected.items())):
            partition = manifest.partition_of(path)
            full_table = build_anchor_table(manifest, partition)
            length = manifest.entry_of(path).num_frames
            t = (0, length // 2, length - 1)[index % 3]
            anchors = tuple(
                a for a in full_table if a.relative_path == path and a.current_frame_index == t
            )
            table = dataclasses.replace(full_table, entries=anchors, positions=(0,))
            sample = ManifestWindowDataset(root, manifest, partition, table, normalisation=stats)[0]
            spec = FORMAT_SPECS[fmt]
            with np.load(root / path, allow_pickle=False) as z:
                raw_actions = z["actions"].astype(np.float32)
                start = t + spec.default_action_offset
                indices = np.arange(start, start + spec.default_action_horizon)
                expected_mask = indices < length
                expected = raw_actions[np.clip(indices, 0, length - 1)].copy()
                state = z[spec.state_source_key][t].astype(np.float32)
                if fmt == "robotwin":
                    expected = (expected - np.array(stats.action["mean"], dtype=np.float32)) / (
                        np.array(stats.action["std"], dtype=np.float32) + np.float32(1e-8)
                    )
                    state = (state - np.array(stats.state["mean"], dtype=np.float32)) / (
                        np.array(stats.state["std"], dtype=np.float32) + np.float32(1e-8)
                    )
                else:
                    low, high = [
                        np.array(stats.action[k], dtype=np.float32) for k in ("q01", "q99")
                    ]
                    expected[:, :6] = np.clip(
                        2 * (expected[:, :6] - low[:6]) / np.maximum(high[:6] - low[:6], 1e-6) - 1,
                        -1,
                        1,
                    )
                    expected[:, 6] = (expected[:, 6] + 1) * 0.5
                    low, high = [np.array(stats.state[k], dtype=np.float32) for k in ("q01", "q99")]
                    state = np.clip(2 * (state - low) / np.maximum(high - low, 1e-6) - 1, -1, 1)
                action_error = float(np.max(np.abs(sample.supervision_actions - expected)))
                state_error = float(np.max(np.abs(sample.proprio - state)))
                raw_errors = []
                for camera, key in enumerate(spec.measurement_keys):
                    raw = z[key][t].astype(np.float32)
                    if fmt == "libero":
                        raw /= np.float32(255)
                    got = sample.burst_values[camera][0, 0]
                    raw_errors.append(float(np.max(np.abs(got - raw.transpose(2, 0, 1)))))
            assert action_error == state_error == max(raw_errors) == 0
            assert np.array_equal(sample.supervision_mask, expected_mask)
            assert not sample.executed_action_available
            comparisons.append(
                dict(
                    path=path,
                    domain=domain,
                    task=manifest.entry_of(path).task_id,
                    t=t,
                    frames=length,
                    action_error=action_error,
                    state_error=state_error,
                    raw_errors=raw_errors,
                    valid_actions=int(expected_mask.sum()),
                )
            )
        summary["platforms"][fmt] = dict(
            verification=verify,
            manifest_hash=manifest.content_hash(),
            task_count=len(tasks),
            per_task=dict(tasks),
            stats_sha256=stats.sha256,
            stats_path=str(stats_path),
            stats_mode=stats.mode,
            epsilon=stats.epsilon,
            epsilon_mode=stats.epsilon_mode,
            comparisons=comparisons,
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hf-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.hf_metadata, args.checkpoint_root, args.output), indent=2))
