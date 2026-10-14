import concurrent.futures
import copy
import hashlib
import io
import json

import numpy as np
import pytest

from owac.runtime.polaris_dataset_inventory import VIEWS, download_file, inspect_record, jpeg_shape


def jpeg_header(height=180, width=320):
    dimensions = bytes([8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + bytes([3])
    return b"\xff\xd8\xff\xc0\x00\x08" + dimensions + b"\xff\xd9"


def record():
    commands = np.asarray([[0] * 7 + [0], [0] * 7 + [1]], dtype=np.float32)
    return {
        "steps/language_instruction": [b"put the cube in the bin"] * 2,
        "steps/action": commands.ravel(),
        "steps/action_dict/joint_position": commands[:, :7].ravel(),
        "steps/action_dict/gripper_position": commands[:, 7],
        "steps/observation/joint_position": commands[:, :7].ravel(),
        "steps/observation/gripper_position": commands[:, 7],
        "steps/reward": np.asarray([0, 1], dtype=np.float32),
        "steps/is_terminal": np.asarray([0, 1]),
        "episode_metadata/file_path": [b"success_/demos/scene_a/episode_1.npz"],
        "episode_metadata/recording_folderpath": [b"success_/demos/scene_a/episode_1.npz"],
        **{f"steps/observation/{view}": [jpeg_header()] * 2 for view in VIEWS},
    }


def test_source_resolution_is_read_without_image_conversion():
    assert jpeg_shape(jpeg_header(270, 480)) == [270, 480, 3]
    with pytest.raises(ValueError, match="complete RGB SOF"):
        jpeg_shape(jpeg_header()[:6])


def test_all_three_views_and_per_frame_language_survive_inventory(tmp_path):
    source = record()
    source["steps/language_instruction"][1] = b"release the cube"
    row = inspect_record(source, "episode-0", "source-digest", tmp_path)
    assert len(row["rgb"]) == 3
    assert row["instructions"] == ["put the cube in the bin", "release the cube"]
    assert not row["three_view_rgb_raw_capture_complete"]
    with np.load(tmp_path / "episode-0/source_numeric.npz") as saved:
        np.testing.assert_array_equal(saved["actions"].ravel(), source["steps/action"])
    labels = json.loads((tmp_path / "episode-0/language.json").read_text())
    assert labels["instructions"] == ["put the cube in the bin", "release the cube"]


def test_missing_original_view_frame_blocks_corpus_inventory(tmp_path):
    source = record()
    source["steps/observation/exterior_image_2_left"].pop()
    with pytest.raises(ValueError, match="Incomplete original view"):
        inspect_record(source, "episode-0", "digest", tmp_path)


def test_conflicting_action_fields_cannot_be_silently_normalized(tmp_path):
    source = copy.deepcopy(record())
    source["steps/action_dict/joint_position"][0] = 0.3
    with pytest.raises(ValueError, match="inconsistent expert actions"):
        inspect_record(source, "episode-0", "digest", tmp_path)


def test_concurrent_shards_preserve_independent_download_payloads(tmp_path, monkeypatch):
    payloads = {f"dataset-train.tfrecord-{i:05d}": bytes([i]) * 2048 for i in range(2)}
    entries = [
        {"path": name, "size": len(payload), "lfs": {"oid": hashlib.sha256(payload).hexdigest()}}
        for name, payload in payloads.items()
    ]

    def response(url, timeout):
        return io.BytesIO(payloads[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr("urllib.request.urlopen", response)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda e: download_file(e, tmp_path, None), entries))
    assert len(results) == 2
    for name, payload in payloads.items():
        assert (tmp_path / "source" / name).read_bytes() == payload
