#!/usr/bin/env python3
"""Audit a deployed RawVLA-Bench NPZ cache and build the explicit partition manifests.

This is a thin driver around the P0 library code; it holds no measurement logic of its own.

    PYTHONPATH=src python scripts/n02_p0_real_cache_audit.py \
        --root /path/to/RawVLA-Bench --output <dir> \
        --hf-metadata /path/to/dataset-source-pinned.json

What it does, all read-only with respect to the cache:

1. ``audit_dataset_root`` on the real root for both platforms (``evidence_kind="real_cache"``),
   which reads whole NPZ files and reports measured fields, dtypes, ranges, resolutions, camera
   order, timestamps/noise/language/proprio availability and the protocol checks.
2. A curated symlink farm that preserves the ``data/<platform>/raw/...`` relative layout and
   points at real cache files spread across suites/tasks, audited with
   ``evidence_kind="real_cache_sample"`` so it can never be mistaken for the deployed root.
3. ``build_partition_manifest`` over every real ``raw`` episode using the pinned HuggingFace
   LFS ``sha256`` values as source-file hashes (no re-hashing of the multi-hundred-GB cache),
   plus ``verify_partitions``.

Nothing is written into the cache and no training or evaluation process is started.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def _raw_paths(paths: list[str]) -> list[str]:
    return sorted(p for p in paths if "/raw/" in pathlib.PurePath(p).as_posix())


def _curated_selection(raw_paths: list[str], target: int) -> list[str]:
    """One file from each of ``target`` evenly spaced episode directories."""
    groups: dict[str, list[str]] = {}
    for path in raw_paths:
        groups.setdefault(str(pathlib.Path(path).parent), []).append(path)
    keys = sorted(groups)
    if not keys:
        return []
    step = max(1, len(keys) // target)
    return [groups[key][0] for key in keys[::step][:target]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, help="RawVLA-Bench root (contains data/<platform>/raw)"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--hf-metadata", default="", help="dataset-source-pinned.json from the download"
    )
    parser.add_argument(
        "--max-files", type=int, default=8, help="files read by the real-root audit"
    )
    parser.add_argument(
        "--curated-files", type=int, default=10, help="files read per platform by the farm audit"
    )
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from owac.frameworks.n02_full.data import (
        PartitionManifest,
        audit_dataset_root,
        build_partition_manifest,
        verify_partitions,
    )
    from owac.raw.rawvla_sequence_n02 import (
        EpisodeRef,
        _acquisition_source,
        _episode_identity,
        _relative_episode_path,
        discover_npz_files,
    )

    root = pathlib.Path(args.root).resolve()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    sha256_by_path: dict[str, str] = {}
    size_by_path: dict[str, int] = {}
    metadata_revision = None
    if args.hf_metadata:
        meta_path = pathlib.Path(args.hf_metadata)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            metadata_revision = meta.get("revision")
            for entry in meta.get("files", ()):
                if entry["path"].endswith(".npz"):
                    sha256_by_path[entry["path"]] = (entry.get("lfs") or {}).get("sha256", "") or ""
                    size_by_path[entry["path"]] = int(entry.get("size") or 0)

    summary: dict = {
        "root": str(root),
        "hf_metadata": {"revision": metadata_revision, "npz_entries": len(sha256_by_path)},
        "platforms": {},
    }

    for fmt in ("libero", "robotwin"):
        entry: dict = {}
        report = audit_dataset_root(root, fmt, max_files=args.max_files, evidence_kind="real_cache")
        (out / f"audit_{fmt}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str)
        )
        entry["audit_real_root"] = {
            "status": report.get("status"),
            "files_total": report.get("files_total"),
            "files_checked": report.get("files_checked"),
            "checks_summary": report.get("checks_summary"),
            "problems": report.get("problems"),
        }

        discovered, _pattern = discover_npz_files(root, fmt)
        raw_paths = _raw_paths(list(discovered))
        entry["discovered_raw_npz"] = len(raw_paths)

        if not args.skip_manifest:
            episodes = []
            frames_by_path: dict[str, int] = {}
            for path in raw_paths:
                relative = _relative_episode_path(path, root)
                with np.load(path, allow_pickle=False) as archive:
                    frames = (
                        int(np.asarray(archive["actions"]).shape[0])
                        if "actions" in archive.files
                        else 0
                    )
                frames_by_path[relative] = frames
                episodes.append(
                    EpisodeRef(
                        path=path,
                        relative_path=relative,
                        acquisition_source=_acquisition_source(relative),
                        episode_group_id=_episode_identity(relative),
                        num_frames=frames,
                        size_bytes=int(
                            size_by_path.get(relative) or pathlib.Path(path).stat().st_size
                        ),
                        sha256=sha256_by_path.get(relative, ""),
                    )
                )
            manifest = build_partition_manifest(
                episodes, dataset_format=fmt, cache_root=root, seed=args.seed
            )
            verification = verify_partitions(manifest)
            payload_text = manifest.to_json()
            readback = PartitionManifest.from_json(payload_text)
            readback_same = (
                readback.content_hash() == manifest.content_hash()
                and readback.partitions == manifest.partitions
                and readback.groups == manifest.groups
            )
            (out / f"manifest_{fmt}.json").write_text(
                json.dumps(
                    {
                        "manifest": json.loads(payload_text),
                        "manifest_content_hash": manifest.content_hash(),
                        "readback_identical": readback_same,
                        "verify": verification,
                        "seed": args.seed,
                        "sha256_provenance": (
                            "source_file sha256 comes from the pinned HuggingFace LFS metadata "
                            f"(revision {metadata_revision}); files were not re-hashed"
                        ),
                        "frames_provenance": "num_frames read from each NPZ actions array shape",
                        "frames_total": int(sum(frames_by_path.values())),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            entry["manifest"] = {
                "verify": verification,
                "readback_identical": readback_same,
                "partitions": {name: len(paths) for name, paths in manifest.partitions.items()},
                "groups": len(manifest.groups),
                "entries": len(manifest.entries),
                "frames_total": int(sum(frames_by_path.values())),
            }

        farm_root = out / f"curated_farm_{fmt}"
        selection = _curated_selection(raw_paths, args.curated_files)
        for path in selection:
            relative = _relative_episode_path(path, root)
            link = farm_root / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(path)
        farm_report = audit_dataset_root(
            farm_root, fmt, max_files=len(selection) or 1, evidence_kind="real_cache_sample"
        )
        farm_report["farm_selection"] = [
            {
                "relative_path": _relative_episode_path(p, root),
                "target": p,
                "sha256": sha256_by_path.get(_relative_episode_path(p, root), ""),
            }
            for p in selection
        ]
        (out / f"audit_{fmt}_curated.json").write_text(
            json.dumps(farm_report, ensure_ascii=False, indent=2, default=str)
        )
        entry["audit_curated_sample"] = {
            "status": farm_report.get("status"),
            "files_checked": farm_report.get("files_checked"),
            "checks_summary": farm_report.get("checks_summary"),
            "problems": farm_report.get("problems"),
        }
        summary["platforms"][fmt] = entry

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
