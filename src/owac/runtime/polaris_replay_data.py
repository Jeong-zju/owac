"""Extract public PolaRiS expert trajectories without importing TensorFlow or Isaac."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from owac.runtime.arena_droid import sha256, write_json


def varint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError("Truncated protobuf varint")
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, position
    raise ValueError("Invalid protobuf varint")


def fields(data):
    position = 0
    while position < len(data):
        tag, position = varint(data, position)
        wire = tag & 7
        if tag >> 3 == 0:
            raise ValueError("Invalid protobuf field")
        if wire == 0:
            value, position = varint(data, position)
        else:
            if wire == 2:
                length, position = varint(data, position)
            elif wire in (1, 5):
                length = 8 if wire == 1 else 4
            else:
                raise ValueError(f"Unsupported protobuf wire type: {wire}")
            end = position + length
            if end > len(data):
                raise ValueError("Truncated protobuf field")
            value, position = data[position:end], end
        yield tag >> 3, wire, value


def decode_example(data):
    """Decode bytes, packed float32 and packed int64 lists in tensorflow.Example."""
    features = next(value for number, _, value in fields(memoryview(data)) if number == 1)
    result = {}
    for number, _, entry in fields(features):
        if number != 1:
            continue
        pair = {key: value for key, _, value in fields(entry)}
        name = bytes(pair[1]).decode()
        kind, _, values = next(fields(pair[2]))
        encoded = list(fields(values))
        if kind == 1:
            result[name] = [bytes(value) for _, _, value in encoded]
        elif kind == 2:
            result[name] = np.frombuffer(encoded[0][2], dtype="<f4").copy()
        elif kind == 3:
            packed = encoded[0][2]
            integers = []
            position = 0
            while position < len(packed):
                value, position = varint(packed, position)
                integers.append(value if value < 2**63 else value - 2**64)
            result[name] = np.asarray(integers, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported Feature kind: {kind}")
    return result


def trajectories(path):
    with path.open("rb") as stream:
        index = 0
        while header := stream.read(12):
            if len(header) != 12:
                raise ValueError("Truncated TFRecord header")
            size = struct.unpack("<Q", header[:8])[0]
            if size > 256 * 1024**2:
                raise ValueError("TFRecord exceeds bounded episode size")
            data = stream.read(size)
            if len(data) != size or len(stream.read(4)) != 4:
                raise ValueError("Truncated TFRecord payload")
            yield index, hashlib.sha256(data).hexdigest(), decode_example(data)
            index += 1


def extract(shard: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, digest, record in trajectories(shard):
        instruction = record["steps/language_instruction"]
        count = len(instruction)
        actions = record["steps/action"].reshape(count, 8)
        commands = np.column_stack(
            (
                record["steps/action_dict/joint_position"].reshape(count, 7),
                record["steps/action_dict/gripper_position"],
            )
        )
        if not np.array_equal(actions, commands) or not np.isfinite(actions).all():
            raise ValueError("Inconsistent or non-finite original expert actions")
        state = np.column_stack(
            (
                record["steps/observation/joint_position"].reshape(count, 7),
                record["steps/observation/gripper_position"],
            )
        )
        if not np.isfinite(state).all():
            raise ValueError("Non-finite original robot states")
        episode = destination / f"episode-{index:03d}"
        episode.mkdir()
        np.savez_compressed(episode / "trajectory.npz", actions=actions, states=state)
        for view in ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"):
            images = record[f"steps/observation/{view}"]
            if len(images) != count:
                raise ValueError("Original RGB length differs from trajectory")
            folder = episode / "original_rgb" / view
            folder.mkdir(parents=True)
            for frame, payload in enumerate(images):
                (folder / f"{frame:05d}.jpg").write_bytes(payload)
        row = {
            "index": index,
            "record_sha256": digest,
            "frames": count,
            "instruction": instruction[0].decode(),
            "all_instructions": sorted({value.decode() for value in instruction}),
            "source_file_path": record["episode_metadata/file_path"][0].decode(),
            "source_recording_folderpath": record["episode_metadata/recording_folderpath"][
                0
            ].decode(),
            "trajectory_sha256": sha256(episode / "trajectory.npz"),
            "serialized_numeric_dtype": "float32",
            "action_changed": False,
            "last_reward": float(record["steps/reward"][-1]),
            "source_is_terminal": bool(record["steps/is_terminal"][-1]),
            "success_marker_is_source_claim_only": True,
        }
        write_json(episode / "source.json", row)
        rows.append(row)
    write_json(destination / "index.json", {"shard_sha256": sha256(shard), "episodes": rows})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.shard, args.destination), ensure_ascii=False))


if __name__ == "__main__":
    main()
