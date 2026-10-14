"""Audit every published PolaRiS simulation episode before scene reconstruction or release."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import io
import json
import shutil
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.polaris_replay_data import trajectories

REPO = "owhan/PolaRiS-datasets"
REVISION = "dca86b8e186df63323eb375344a1977d92f508f7"
PREFIX = "polaris_droid_cotrain_dataset/1.0.0/"
VIEWS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")
ASSETS = {
    "owhan/DROID-sim-environments": "5b815672a9a2e3ae596ae845188ae4be5c5d2f34",
    "owhan/PolaRiS-Hub": "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b",
}


def utc():
    return datetime.now(UTC).isoformat()


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def jpeg_shape(payload):
    """Read JPEG SOF dimensions without decoding or resampling original image bytes."""
    if payload[:2] != b"\xff\xd8":
        raise ValueError("Source RGB is not a JPEG")
    stream = io.BytesIO(payload)
    stream.seek(2)
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while marker := stream.read(1):
        if marker != b"\xff":
            raise ValueError("Invalid JPEG marker boundary")
        code = stream.read(1)
        while code == b"\xff":
            code = stream.read(1)
        if not code:
            break
        if code[0] in {0xD9, 0xDA}:
            break
        if code[0] in {0x01, *range(0xD0, 0xD9)}:
            continue
        encoded_length = stream.read(2)
        if len(encoded_length) != 2:
            break
        length = int.from_bytes(encoded_length, "big")
        if length < 2:
            raise ValueError("Invalid JPEG segment length")
        data = stream.read(length - 2)
        if len(data) != length - 2:
            break
        if code[0] in sof:
            if len(data) < 6 or data[0] != 8:
                raise ValueError("Source RGB has invalid or unsupported JPEG precision")
            height = int.from_bytes(data[1:3], "big")
            width = int.from_bytes(data[3:5], "big")
            if not height or not width or data[5] != 3:
                raise ValueError("Source RGB must have three nonempty channels")
            return [height, width, 3]
    raise ValueError("JPEG has no complete RGB SOF marker")


def inspect_record(record, identity, record_digest, destination):
    instruction = [v.decode("utf-8") for v in record["steps/language_instruction"]]
    count = len(instruction)
    if not count or any(not value.strip() for value in instruction):
        raise ValueError(f"Missing language labels: {identity}")
    actions = record["steps/action"].reshape(count, 8)
    commands = np.column_stack(
        (
            record["steps/action_dict/joint_position"].reshape(count, 7),
            record["steps/action_dict/gripper_position"],
        )
    )
    states = np.column_stack(
        (
            record["steps/observation/joint_position"].reshape(count, 7),
            record["steps/observation/gripper_position"],
        )
    )
    if not np.array_equal(actions, commands) or not np.isfinite(actions).all():
        raise ValueError(f"Non-finite or inconsistent expert actions: {identity}")
    if not np.isfinite(states).all():
        raise ValueError(f"Non-finite expert robot states: {identity}")
    for key, value in record.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite source field {key}: {identity}")
    rgb = {}
    for name in VIEWS:
        images = record[f"steps/observation/{name}"]
        if len(images) != count:
            raise ValueError(f"Incomplete original view {name}: {identity}")
        shapes = sorted({tuple(jpeg_shape(image)) for image in images})
        rgb[name] = {"frames": count, "shapes_hwc": shapes, "bytes": sum(map(len, images))}
    folder = record["episode_metadata/recording_folderpath"][0].decode()
    path = record["episode_metadata/file_path"][0].decode()
    episode = destination / identity
    episode.mkdir(parents=True, exist_ok=False)
    numeric = {k: v for k, v in record.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(episode / "source_numeric.npz", **numeric, actions=actions, states=states)
    write_json(episode / "language.json", {"instructions": instruction})
    close = np.flatnonzero((actions[:, 7] > 0.5) & (np.r_[0, actions[:-1, 7]] <= 0.5))
    opening = np.flatnonzero((actions[:, 7] <= 0.5) & (np.r_[0, actions[:-1, 7]] > 0.5))
    keyframes = sorted({0, count - 1, *close.tolist(), *opening.tolist()})
    references = {}
    for name in VIEWS:
        references[name] = []
        for frame in keyframes:
            image = episode / "reference_rgb" / name / f"{frame:05d}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(record[f"steps/observation/{name}"][frame])
            references[name].append(
                {"frame": frame, "path": str(image.relative_to(episode)), "sha256": sha256(image)}
            )
    row = {
        "id": identity,
        "record_sha256": record_digest,
        "source_file_path": path,
        "source_recording_folderpath": folder,
        "source_scene_name": Path(path).parent.name,
        "source_episode_name": Path(path).name,
        "frames": count,
        "instructions": sorted(set(instruction)),
        "rgb": rgb,
        "close_transition_frames": close.tolist(),
        "open_transition_frames": opening.tolist(),
        "reference_rgb_keyframes": references,
        "last_reward": float(record["steps/reward"][-1]),
        "source_is_terminal": bool(record["steps/is_terminal"][-1]),
        "feature_keys": sorted(record),
        "numeric_sha256": sha256(episode / "source_numeric.npz"),
        "language_sha256": sha256(episode / "language.json"),
        "source_action_modified": False,
        "original_scene_mapping_verified": False,
        "original_camera_mapping_verified": False,
        "three_view_rgb_raw_capture_complete": False,
        "reconstruction_status": "unresolved_source_scene",
    }
    write_json(episode / "source.json", row)
    return row


def download_file(entry, root, cache):
    destination = root / "source" / entry["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = entry.get("lfs", {}).get("oid")
    if not digest or len(digest) != 64:
        raise ValueError(f"Missing upstream LFS SHA256: {entry['path']}")
    existing = cache / entry["path"] if cache else None
    if existing and existing.is_file() and sha256(existing) == digest:
        shutil.copyfile(existing, destination)
    else:
        temporary = destination.with_name(destination.name + ".partial")
        for attempt in range(3):
            try:
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{entry['path']}"
                with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as f:
                    shutil.copyfileobj(response, f, length=1024**2)
                if temporary.stat().st_size != entry["size"] or sha256(temporary) != digest:
                    raise ValueError("Downloaded shard size or SHA256 differs from upstream")
                temporary.replace(destination)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2)
    if destination.stat().st_size != entry["size"] or sha256(destination) != digest:
        raise ValueError("Cached shard failed upstream size/SHA256 validation")
    return {"path": entry["path"], "bytes": entry["size"], "sha256": digest}


def run(root, cache=None):
    root.mkdir(parents=True, exist_ok=False)
    started = utc()
    status = {"status": "running", "stage": "source_catalog", "started_at_utc": started}

    def save(**changes):
        status.update(changes, updated_at_utc=utc())
        write_json(root / "status.json", status)

    save()
    try:
        catalog = fetch_json(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}")
        tree = fetch_json(
            f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true"
        )
        entries = sorted(
            [e for e in tree if e["path"].startswith(PREFIX) and ".tfrecord-" in e["path"]],
            key=lambda e: e["path"],
        )
        info_url = (
            f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{PREFIX}dataset_info.json"
        )
        info = fetch_json(info_url)
        write_json(root / "source_catalog.json", {"repo": catalog, "tree": tree, "info": info})
        asset_catalogs = {
            repo: fetch_json(f"https://huggingface.co/api/datasets/{repo}/revision/{rev}")
            for repo, rev in ASSETS.items()
        }
        write_json(root / "asset_catalogs.json", asset_catalogs)
        split = next(s for s in info["splits"] if s["name"] == "train")
        lengths = [int(n) for n in split["shardLengths"]]
        if len(entries) != len(lengths) or len(entries) != 32:
            raise ValueError("Pinned corpus shard coverage disagrees with publisher metadata")
        if catalog.get("cardData", {}).get("license") != "mit":
            raise ValueError("Pinned source license changed or was not declared")
        save(stage="download_source", planned_shards=len(entries), expected_episodes=sum(lengths))
        downloaded = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(download_file, entry, root, cache) for entry in entries]
            for future in concurrent.futures.as_completed(futures):
                downloaded.append(future.result())
                save(downloaded_shards=len(downloaded))
                print("SOURCE_SHARD", len(downloaded), "/", len(entries), flush=True)
        write_json(root / "shards.json", {"files": sorted(downloaded, key=lambda e: e["path"])})
        rows = []
        save(stage="inspect_every_record", inspected_episodes=0)
        with (root / "episodes.jsonl").open("w") as stream:
            for shard_index, (entry, expected) in enumerate(zip(entries, lengths, strict=True)):
                count = 0
                for index, digest, record in trajectories(root / "source" / entry["path"]):
                    row = inspect_record(
                        record,
                        f"shard-{shard_index:05d}-record-{index:03d}",
                        digest,
                        root / "episodes",
                    )
                    row.update(source_shard=entry["path"], source_record_index=index)
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    count += 1
                if count != expected:
                    raise ValueError(f"Shard {shard_index} contained {count}, expected {expected}")
                save(inspected_episodes=len(rows), inspected_shards=shard_index + 1)
                print("SOURCE_EPISODES", len(rows), "/", sum(lengths), flush=True)
        records = [r["record_sha256"] for r in rows]
        source_ids = [(r["source_file_path"], r["source_recording_folderpath"]) for r in rows]
        scene_counts = collections.Counter(r["source_scene_name"] for r in rows)
        tasks = collections.Counter(task for r in rows for task in r["instructions"])
        summary = {
            "status": "source_inventory_complete_reconstruction_pending",
            "source_repo": REPO,
            "source_revision": REVISION,
            "license": catalog["cardData"]["license"],
            "episodes": len(rows),
            "expected_episodes": sum(lengths),
            "frames": sum(r["frames"] for r in rows),
            "source_rgb_view_frames": 3 * sum(r["frames"] for r in rows),
            "views": VIEWS,
            "scenes": dict(sorted(scene_counts.items())),
            "instructions": dict(sorted(tasks.items())),
            "duplicate_record_hashes": len(records) - len(set(records)),
            "duplicate_source_episode_ids": len(source_ids) - len(set(source_ids)),
            "feature_keys": sorted(set().union(*(r["feature_keys"] for r in rows))),
            "original_scene_mapping_verified_episodes": 0,
            "original_camera_mapping_verified_episodes": 0,
            "complete_three_view_rgb_raw_episodes": 0,
            "publish_ready": False,
            "release_blockers": [
                "Original training scene/asset and initial object-state mapping are absent.",
                "Original camera extrinsics/intrinsics and capture timestamps are absent.",
                "The existing pilot has only two newly rendered views in a reconstructed scene.",
                "No corpus-wide physical replay or pi05 training-loader validation is complete.",
            ],
            "started_at_utc": started,
            "finished_at_utc": utc(),
            "source_manifest_sha256": sha256(root / "episodes.jsonl"),
            "shards_manifest_sha256": sha256(root / "shards.json"),
        }
        write_json(root / "summary.json", summary)
        save(status="complete", stage="source_inventory_complete_reconstruction_pending")
        return summary
    except BaseException as error:
        save(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def audit_views(inventory, output):
    """Verify whether public image fields represent independent or duplicated camera streams."""
    expected = {
        r["id"]: r for r in map(json.loads, (inventory / "episodes.jsonl").read_text().splitlines())
    }
    rows = []
    shards = json.loads((inventory / "shards.json").read_text())["files"]
    for shard_index, shard in enumerate(shards):
        path = inventory / "source" / shard["path"]
        if sha256(path) != shard["sha256"]:
            raise ValueError("Source shard changed since complete inventory")
        for index, digest, record in trajectories(path):
            identity = f"shard-{shard_index:05d}-record-{index:03d}"
            source = expected[identity]
            if digest != source["record_sha256"]:
                raise ValueError("Source record changed since complete inventory")
            exterior1 = record["steps/observation/exterior_image_1_left"]
            exterior2 = record["steps/observation/exterior_image_2_left"]
            count = source["frames"]
            if len(exterior1) != count or len(exterior2) != count:
                raise ValueError("Missing original exterior image")
            equal = sum(a == b for a, b in zip(exterior1, exterior2, strict=True))
            rows.append({"id": identity, "frames": count, "identical_exterior_jpeg_frames": equal})
    if {r["id"] for r in rows} != set(expected):
        raise ValueError("View audit did not cover all original episodes")
    summary = {
        "checked_at_utc": utc(),
        "source_revision": REVISION,
        "episodes": len(rows),
        "frames": sum(r["frames"] for r in rows),
        "identical_exterior_jpeg_frames": sum(r["identical_exterior_jpeg_frames"] for r in rows),
        "episodes_with_identical_exteriors": sum(
            r["frames"] == r["identical_exterior_jpeg_frames"] for r in rows
        ),
        "comparison": "exact JPEG bytes at every source frame; no image conversion",
        "rows": rows,
    }
    if output.exists():
        raise FileExistsError(output)
    write_json(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--audit-views-from", type=Path)
    args = parser.parse_args()
    result = (
        audit_views(args.audit_views_from, args.root)
        if args.audit_views_from
        else run(args.root, args.source_cache)
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
