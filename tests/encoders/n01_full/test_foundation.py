"""P0 cross-module integration checks of the N01 v1.0 evidence-integration foundation.

This is the integration wave (workflow F).  It does not repeat the per-module unit
tests of ``test_contracts_config.py`` (A), ``test_manifest.py`` (B) and
``test_preflight.py`` (C); it checks the seams between them, on the real artifacts in
the worktree:

1. A's ``SourceKey`` built from B's real manifest ``rows``;
2. the four-region partition rebuilt here with independent set algebra -- never through
   B's own ``verify_manifest``/``assert_manifest_partition``;
3. the real loader (``RawVLASequenceDataset``) asked for each region in turn;
4. ``protocol_v1.json`` against the mandatory items of A's config module;
5. the ``bundle`` artifact carrying the new P0 sources that HEAD does not have.

The multi-hundred-gigabyte RAW caches live on the remote server only.  This machine
holds B's ``loader-evidence-*.json`` (a recorded run of the real loader against the real
cache) plus the manifests, so nothing here downloads data or fabricates NPZ input.  Every
expectation below is a decidable absolute value -- a count, an empty intersection, an
exact path set, a refusal -- never "equal to B's own report", because two sides sharing
one mistake would still agree.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from owac.frameworks.n01_full import config as n01_config
from owac.frameworks.n01_full.contracts import EvidenceEstimate, SourceKey
from owac.raw.n01_full_manifest import loader_split_manifest
from owac.raw.rawvla_sequence import RawVLASequenceDataset, assign_side, source_group
from owac.runtime import n01_full_preflight as preflight

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests" / "n01-full-v1"
INDEX_PATH = MANIFEST_DIR / "n01-full-v1-index.json"
PROTOCOL_PATH = REPO_ROOT / "configs" / "encoders" / "n01_full" / "protocol_v1.json"

PLATFORMS = ("libero", "robotwin2")
REGIONS = ("train", "calibration", "validation", "test")
SPLIT_VERSION = "n01-full-v1"
#: The frozen pool sizes of the P0 split (acceptance numbers, not a re-read of B).
FROZEN_POOL_FILES = {"libero": 1771, "robotwin2": 632}
#: P0 sources created in this wave and absent from HEAD: a bundle built by
#: ``git archive HEAD`` would drop exactly these files.
P0_NEW_SOURCES = (
    "src/owac/frameworks/n01_full/contracts.py",
    "src/owac/frameworks/n01_full/config.py",
    "src/owac/raw/n01_full_manifest.py",
    "src/owac/runtime/n01_full_preflight.py",
    "configs/encoders/n01_full/protocol_v1.json",
    "configs/encoders/n01_full/upstream_lock.json",
    "tests/encoders/n01_full/test_foundation.py",
)


# ---------------------------------------------------------------------------------------
# artifact access
# ---------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(platform: str) -> dict[str, Any]:
    return _read_json(MANIFEST_DIR / f"n01-full-v1-{platform}-manifest.json")


def _loader_evidence(platform: str) -> dict[str, Any]:
    return _read_json(MANIFEST_DIR / f"n01-full-v1-loader-evidence-{platform}.json")


def _index() -> dict[str, Any]:
    return _read_json(INDEX_PATH)


def _protocol() -> dict[str, Any]:
    return _read_json(PROTOCOL_PATH)


def _row(platform: str, position: int = 0) -> dict[str, Any]:
    return _manifest(platform)["rows"][position]


def _sensor_grid(platform: str, camera: str) -> tuple[int, int]:
    """Real per-camera ``(height, width)`` from the recorded loader run."""
    shape = _loader_evidence(platform)["regions"]["train"]["sample"]["window_shapes"][camera]
    return int(shape[2]), int(shape[3])


def _channels(platform: str) -> int:
    """Measurement channel count of one platform, as the real loader emitted it."""
    sample = _loader_evidence(platform)["regions"]["train"]["sample"]["window_shapes"]
    counts = {int(shape[1]) for shape in sample.values()}
    assert len(counts) == 1, f"{platform}: cameras disagree on the channel count: {counts}"
    return counts.pop()


def _source_key(
    row: dict[str, Any],
    *,
    camera: str,
    frame: int,
    channel: int,
    sensor_y: int = 0,
    sensor_x: int = 0,
) -> SourceKey:
    """A's identity for one sensor sample of one real manifest row."""
    return SourceKey(
        dataset_id=str(row["dataset_id"]),
        episode_id=str(row["episode_id"]),
        camera_id=camera,
        frame_id=int(frame),
        channel_or_phase=int(channel),
        sensor_y=int(sensor_y),
        sensor_x=int(sensor_x),
    )


def _region_paths_from_rows(manifest: dict[str, Any]) -> dict[str, list[str]]:
    """Rebuild the four region sets from the ``rows`` table alone."""
    return {
        region: sorted(
            str(row["relative_path"]) for row in manifest["rows"] if row["split"] == region
        )
        for region in REGIONS
    }


def _region_cameras(platform: str, region: str) -> set[str]:
    """Protocol camera names of one region, read from the manifest rows."""
    return {
        str(camera)
        for row in _manifest(platform)["rows"]
        if row["split"] == region
        for camera in row["camera_ids"]
    }


def _side_from_recorded_rule(
    group: str,
    *,
    order: tuple[str, ...],
    fractions: dict[str, float],
    split_version: str,
) -> str:
    """The rule the index records: the first cumulative threshold whose side is train.

    ``order`` and ``fractions`` come from the manifest's own ``regions`` /
    ``region_fractions`` fields, so this is an independent application of the recorded
    rule through the legacy binary primitive.
    """
    cumulative = 0.0
    for name in order[:-1]:
        cumulative += float(fractions[name])
        if assign_side(group, split_version=split_version, train_fraction=cumulative) == "train":
            return name
    return order[-1]


def _loader_view(platform: str) -> tuple[dict[str, Any], dict[str, Any], Path, Any]:
    """The real loader's region selection, without the remote NPZ cache.

    The constructor needs the real cache on disk (it globs ``*.npz`` and loads the
    statistics file), which this machine does not have and must not fake.  The split
    wrapper and the episode list below are exactly what the constructor feeds the
    selection method with; ``test_instance_paths_come_from_the_region_selection`` pins
    that ``RawVLASequenceDataset.__init__`` assigns ``instances.paths`` from it.
    """
    manifest = _manifest(platform)
    root = Path(_loader_evidence(platform)["raw_cache_root"])
    split = loader_split_manifest(manifest, root)
    episodes = sorted(str(root / str(row["relative_path"])) for row in manifest["rows"])
    instance = object.__new__(RawVLASequenceDataset)
    instance.split = split
    instance.episodes = episodes
    return manifest, split, root, instance


# ---------------------------------------------------------------------------------------
# (1) identity: B's real rows -> A's SourceKey
# ---------------------------------------------------------------------------------------


def test_real_manifest_rows_project_to_per_camera_source_keys() -> None:
    """Every real row yields one independent key per protocol camera.

    A consumer that keys on the row-level ``camera_id`` token (``multi_camera``) instead
    of the per-camera protocol name collapses the two or three cameras of one file onto
    a single identity, so this test fails on the collision.
    """
    for platform in PLATFORMS:
        row = _row(platform)
        cameras = list(row["camera_ids"])
        assert row["camera_id"] == "multi_camera"
        assert len(cameras) == row["camera_count"] > 1
        keys = []
        for camera in cameras:
            height, width = _sensor_grid(platform, camera)
            keys.append(_source_key(row, camera=camera, frame=0, channel=0, sensor_y=0, sensor_x=0))
            keys.append(
                _source_key(
                    row,
                    camera=camera,
                    frame=row["num_frames"] - 1,
                    channel=0,
                    sensor_y=height - 1,
                    sensor_x=width - 1,
                )
            )
        assert {key.camera_id for key in keys} == set(cameras)
        assert len({key.identity() for key in keys}) == len(keys)


def test_two_exposures_of_one_real_row_stay_distinct_sources() -> None:
    """Two exposures of one trajectory are two samples even at one identical coordinate.

    An identity derived from the pixel value, from the batch slot or from a per-arrival
    counter merges the first and the last exposure of a real row; keeping the exposure
    identity (and no value) in the key keeps them apart while a repeated read of one
    exposure stays one key.
    """
    row = _row("libero")
    camera = row["camera_ids"][0]
    first = _source_key(row, camera=camera, frame=0, channel=0)
    last = _source_key(row, camera=camera, frame=row["num_frames"] - 1, channel=0)
    assert first.frame_id == 0
    assert last.frame_id == row["num_frames"] - 1 > 0
    # the two keys agree on dataset, episode, camera, channel and sensor position:
    # only the exposure identity separates two samples the pixel value cannot separate
    assert first.identity()[:3] == last.identity()[:3]
    assert first.identity()[4:] == last.identity()[4:]
    assert first.identity() != last.identity()
    assert _source_key(row, camera=camera, frame=0, channel=0) == first


def test_repeated_packet_folds_to_one_source_and_one_observation() -> None:
    """A packet read twice is one source; an estimate counts it once, not twice.

    Counting arrivals instead of distinct identities (or adding a prior to the
    independent count) inflates the observation count and fails here.
    """
    row = _row("libero")
    camera = row["camera_ids"][0]
    key = _source_key(row, camera=camera, frame=0, channel=0)
    repeated = [_source_key(row, camera=camera, frame=0, channel=0), key]
    assert len(set(repeated)) == 1

    def estimate(sources: list[SourceKey]) -> EvidenceEstimate:
        return EvidenceEstimate(
            theta=(0.0,),
            uncertainty_kind="none",
            identifiable_mask=(True,),
            evidence_source_ids=tuple(sources),
            support_version="support-v1",
        )

    assert estimate(repeated).independent_observation_count == 1
    later = _source_key(row, camera=camera, frame=row["num_frames"] - 1, channel=0)
    assert estimate([key, later]).independent_observation_count == 2
    # The pool never double-counts a trajectory through its paired RGB mirror.
    index = _index()
    assert index["rgb_mirror"]["libero_files"] == FROZEN_POOL_FILES["libero"]
    assert any(
        "excluded here so one trajectory is counted once" in note
        for note in _manifest("libero")["notes"]
    )


def test_camera_episode_dataset_and_channel_never_collide() -> None:
    """Camera, episode, dataset and channel are separate identity dimensions.

    Dropping any one of them (for example identity = episode + frame only) merges
    unrelated samples and fails on the corresponding pair.
    """
    libero_row = _row("libero", 0)
    other_episode_row = _row("libero", 1)
    robotwin_row = _row("robotwin2", 0)
    assert libero_row["episode_id"] != other_episode_row["episode_id"]

    camera = libero_row["camera_ids"][0]
    base = _source_key(libero_row, camera=camera, frame=0, channel=0)
    assert _source_key(libero_row, camera=libero_row["camera_ids"][1], frame=0, channel=0) != base
    assert _source_key(other_episode_row, camera=camera, frame=0, channel=0) != base

    channels = _channels("libero")
    assert channels == 3
    per_channel = [
        _source_key(libero_row, camera=camera, frame=0, channel=channel)
        for channel in range(channels)
    ]
    assert len({key.identity() for key in per_channel}) == channels

    cross_dataset = _source_key(
        {**libero_row, "dataset_id": robotwin_row["dataset_id"]}, camera=camera, frame=0, channel=0
    )
    assert cross_dataset.dataset_id == "robotwin2"
    assert cross_dataset.identity() != base.identity()


@pytest.mark.parametrize("field", ["dataset_id", "episode_id", "camera_id"])
def test_empty_identity_field_is_refused(field: str) -> None:
    """A partial identity is refused instead of silently merging unrelated samples."""
    row = _row("robotwin2")
    payload: dict[str, Any] = {
        "dataset_id": row["dataset_id"],
        "episode_id": row["episode_id"],
        "camera_id": row["camera_ids"][0],
        "frame_id": 0,
        "channel_or_phase": 0,
        "sensor_y": 0,
        "sensor_x": 0,
    }
    assert SourceKey(**payload)  # the real row projects to a valid key
    payload[field] = ""
    with pytest.raises(ValueError, match=field):
        SourceKey(**payload)


# ---------------------------------------------------------------------------------------
# (2) the four-region partition, rebuilt with independent set algebra
# ---------------------------------------------------------------------------------------


def test_four_regions_partition_the_frozen_pool_with_real_rows() -> None:
    """Rebuilt regions are disjoint, cover the frozen pool and match the ``sides`` column.

    The absolute expectations (1771 / 632, empty intersections, the per-region counts)
    are stated here rather than read back from B's checker, so a manifest that lost files
    or leaked one trajectory across two regions cannot agree with itself and pass.
    """
    index = _index()
    protocol = _protocol()
    for platform in PLATFORMS:
        manifest = _manifest(platform)
        rows = manifest["rows"]
        pool = FROZEN_POOL_FILES[platform]
        by_region = _region_paths_from_rows(manifest)

        assert {row["split"] for row in rows} == set(REGIONS)
        assert len(rows) == len({str(row["relative_path"]) for row in rows}) == pool

        for left_index, left in enumerate(REGIONS):
            for right in REGIONS[left_index + 1 :]:
                assert not (set(by_region[left]) & set(by_region[right]))
        union = set().union(*(set(paths) for paths in by_region.values()))
        assert len(union) == pool

        entry = index["datasets"][platform]
        assert entry["files"] == entry["groups"] == pool
        for region in REGIONS:
            assert manifest["sides"][region] == by_region[region]
            assert len(by_region[region]) == entry["regions"][region]["files"]
            assert len(by_region[region]) == entry["regions"][region]["groups"]
            assert len(by_region[region]) == protocol["data"]["splits"][region]["counts"][platform]
        assert protocol["data"]["platforms"][platform]["expected_npz"] == pool

        group_regions: dict[str, set[str]] = {}
        for row in rows:
            group_regions.setdefault(str(row["source_group"]), set()).add(str(row["split"]))
        assert all(len(regions) == 1 for regions in group_regions.values())
        assert len(group_regions) == pool


def test_region_rule_is_pool_level_and_recomputed_from_the_recorded_rule() -> None:
    """Each real group's side follows the recorded pool-level cumulative threshold rule.

    Recomputing the side through the legacy binary primitive at the cumulative
    boundaries recorded in the manifest catches a rule that first splits off a train
    pool and subdivides it, uses non-cumulative thresholds, or assigns per file instead
    of per source group.
    """
    for platform in PLATFORMS:
        manifest = _manifest(platform)
        order = tuple(manifest["regions"])
        fractions = manifest["region_fractions"]
        assert order == REGIONS
        assert fractions == {"train": 0.7, "calibration": 0.1, "validation": 0.1, "test": 0.1}
        cumulative = 0.0
        boundaries = {}
        for name in order[:-1]:
            cumulative += fractions[name]
            boundaries[name] = cumulative
        assert boundaries == pytest.approx({"train": 0.7, "calibration": 0.8, "validation": 0.9})

        for row in manifest["rows"]:
            group = str(row["source_group"])
            assert group == source_group(str(row["relative_path"]), dataset_format=platform)
            expected = _side_from_recorded_rule(
                group,
                order=order,
                fractions=fractions,
                split_version=manifest["split_version"],
            )
            assert row["split"] == expected
        covered = {
            _side_from_recorded_rule(
                str(group["group"]),
                order=order,
                fractions=fractions,
                split_version=manifest["split_version"],
            )
            for group in manifest["source_groups"]
        }
        assert covered == set(REGIONS)


def test_index_records_the_pool_level_rule_and_the_absent_official_split() -> None:
    """The index states, in writing, that the rule is pool-level and that no official split exists.

    A rule invented at read time (for example "keep the official test set, split the
    rest") would leave this record contradicted or empty and fail here.
    """
    index = _index()
    assert index["split_version"] == SPLIT_VERSION
    for platform in PLATFORMS:
        entry = index["datasets"][platform]
        rule = str(entry["region_rule"])
        assert "first cumulative threshold" in rule
        assert "assign_side" in rule
        literal = rule[rule.index("{") : rule.index("}") + 1]
        assert ast.literal_eval(literal) == {
            "train": 0.7,
            "calibration": 0.1,
            "validation": 0.1,
            "test": 0.1,
        }
        assert set(entry["regions"]) == set(REGIONS)

    official = index["official_split"]
    assert official["status"] == "absent"
    consequence = str(official["consequence"])
    assert "same demonstration pool" in consequence
    assert "13,250" in consequence
    assert any("same demonstration pool" in str(note) for note in _manifest("libero")["notes"])


# ---------------------------------------------------------------------------------------
# (3) the real consumer: RawVLASequenceDataset per region
# ---------------------------------------------------------------------------------------


def test_loader_requests_each_region_and_returns_exactly_that_region() -> None:
    """Four region requests return exactly that region's paths, with no overlap.

    A loader that ignores ``split_manifest`` and returns every episode (or that builds
    its own split) fails the equality against the region set, the pairwise disjointness
    and the union size at once.
    """
    for platform in PLATFORMS:
        manifest, split, root, instance = _loader_view(platform)
        evidence = _loader_evidence(platform)
        pool = FROZEN_POOL_FILES[platform]
        seen: dict[str, set[str]] = {}

        for region in REGIONS:
            paths = instance._paths_for_split(region)
            instance.paths = paths  # the assignment RawVLASequenceDataset.__init__ performs
            expected = sorted(str(root / str(rel)) for rel in manifest["sides"][region])
            assert instance.paths == paths == expected == split["sides"][region]
            assert instance.split_name == region
            assert {Path(path).relative_to(root).as_posix() for path in paths} == set(
                manifest["sides"][region]
            )
            recorded = evidence["regions"][region]
            assert len(paths) == recorded["loader_paths"] == recorded["manifest_files"]
            assert recorded["requested_split"] == recorded["loader_split_name"] == region
            assert paths[0] == recorded["first_path"]
            assert paths[-1] == recorded["last_path"]
            seen[region] = set(paths)

        for left_index, left in enumerate(REGIONS):
            for right in REGIONS[left_index + 1 :]:
                assert not (seen[left] & seen[right])
        all_paths = set().union(*seen.values())
        assert len(all_paths) == pool
        assert all_paths == set(instance.episodes)
        assert set(instance._paths_for_split("all")) == all_paths
        with pytest.raises(ValueError, match="unknown split"):
            instance._paths_for_split("heldout")


def test_instance_paths_come_from_the_region_selection() -> None:
    """``instances.paths`` is filled from the method this integration test exercises.

    The real constructor needs the remote NPZ cache (it globs ``*.npz`` and loads the
    statistics file), which this machine does not have and must not fabricate.  This
    pins the reviewed seam: if the loader stops deriving ``paths`` from the region
    selection, the path-equality above must be re-established through the new route.
    """
    assert "self.paths = self._paths_for_split(split)" in inspect.getsource(
        RawVLASequenceDataset.__init__
    )


def test_recorded_loader_run_confirms_the_same_four_regions() -> None:
    """The remote loader run recorded by B reports the same four regions and the same root.

    A manifest changed after that run keeps its counts here but loses the path anchors,
    so the recorded digests and endpoints are re-checked against the files on disk.
    """
    index = _index()
    for platform in PLATFORMS:
        manifest = _manifest(platform)
        evidence = _loader_evidence(platform)
        entry = index["datasets"][platform]
        assert evidence["mode"] == "loader"
        assert evidence["dataset_format"] == platform
        assert evidence["raw_cache_root"] == entry["server_raw_cache_root"]
        assert evidence["manifest_digest"] == entry["manifest_digest"]
        assert (
            evidence["union_files"] == evidence["sum_region_files"] == FROZEN_POOL_FILES[platform]
        )
        assert set(evidence["pairwise_overlaps"].values()) == {0}
        assert len(evidence["pairwise_overlaps"]) == 6
        for region in REGIONS:
            recorded = evidence["regions"][region]
            assert recorded["paths_equal_as_lists"] is True
            assert recorded["paths_equal_as_sets"] is True
            assert _region_cameras(platform, region) == set(recorded["sample"]["cameras"])
            assert manifest["counts"]["regions"][region]["files"] == recorded["manifest_files"]


# ---------------------------------------------------------------------------------------
# (4) protocol_v1.json against A's mandatory validation
# ---------------------------------------------------------------------------------------


def test_protocol_matches_the_config_module_mandatory_items() -> None:
    """Every mandatory P0 item agrees across the protocol and A's enforcing constructors.

    Each assertion is a decidable number or an enforced refusal; if the protocol drifts
    away from the config module (or a constructor stops enforcing), the projection below
    either disagrees or stops raising.
    """
    protocol = _protocol()
    backbone = protocol["models_and_upstream"]["content_backbone"]
    readout = protocol["models_and_upstream"]["readout"]

    content = n01_config.ContentConfig(
        backbone=backbone["backbone"],
        channels=backbone["channels"],
        num_blocks=backbone["num_blocks"],
        feature_width=backbone["feature_width"],
        readout_tokens=backbone["readout_tokens"],
        readout_width=backbone["readout_width"],
        bridge_in_dim=backbone["feature_width"],
        bridge_width=backbone["readout_width"],
    ).validate()
    assert (content.channels, content.num_blocks, content.feature_width) == (64, 9, 512)
    assert (content.readout_tokens, content.readout_width) == (256, 2048)
    assert content.bridge_in_dim == content.feature_width == backbone["feature_width"]
    assert content.bridge_width == content.readout_width == readout["readout_width"]
    assert (readout["readout_tokens"], readout["readout_width"]) == (
        content.readout_tokens,
        content.readout_width,
    )

    choices = tuple(protocol["exact_replay"]["unique_frames_choices"])
    assert choices == n01_config.EXACT_WINDOW_CHOICES == (8, 16, 32)
    replay = n01_config.ReplayConfig(
        exact_window_unique_frames=protocol["exact_replay"]["exact_window_unique_frames"]
    ).validate()
    assert replay.exact_window_unique_frames in choices

    modes = {
        platform: str(protocol["data"]["platforms"][platform]["model_input_mode"])
        for platform in PLATFORMS
    }
    assert set(modes.values()) <= set(n01_config.INPUT_MODES)
    for platform, mode in modes.items():
        declared = protocol["input_identity"]["input_modes"][mode]
        identity = n01_config.InputIdentityConfig(
            dataset_id=platform, input_mode=mode, channels=declared["channels"]
        ).validate()
        assert identity.channels == 3 == _channels(platform)
        assert identity.is_cfa is False
        assert identity.cfa_pattern is None
    assert protocol["input_identity"]["input_modes"]["bayer_cfa"]["channels"] == 4
    cfa = n01_config.InputIdentityConfig(
        input_mode="bayer_cfa", channels=4, cfa_pattern="RGGB"
    ).validate()
    assert cfa.is_cfa is True and cfa.channels == 4

    model_seeds = tuple(protocol["seeds"]["model"]["values"])
    data_seeds = tuple(protocol["seeds"]["data"]["values"])
    budget = n01_config.TrainingBudgetConfig(
        model_seeds=model_seeds, data_seeds=data_seeds
    ).validate()
    assert budget.model_seeds == model_seeds
    assert protocol["seeds"]["separated"] is True
    assert not set(model_seeds) & set(data_seeds)

    identity = n01_config.InputIdentityConfig(
        dataset_id=PLATFORMS[0], input_mode=modes[PLATFORMS[0]], channels=3
    )
    full = n01_config.N01FullConfig(
        input=identity,
        replay=replay,
        content=content,
        budget=budget,
        correspondence=n01_config.CorrespondenceConfig(input_channels=3),
    )
    assert full.validate() is full
    assert full.history.max_history <= replay.exact_window_unique_frames
    # the protocol's "CFA 4/8, rawvla_rgb3 3/6" multiple rule is enforced by the full config
    doubled = n01_config.N01FullConfig(
        input=identity,
        correspondence=n01_config.CorrespondenceConfig(input_channels=6),
    ).validate()
    assert doubled.correspondence.input_channels == 2 * identity.channels

    # The projection is not vacuous: the same constructors refuse drifted values.
    with pytest.raises(ValueError):
        n01_config.ContentConfig(channels=32).validate()
    with pytest.raises(ValueError):
        n01_config.ContentConfig(bridge_in_dim=backbone["readout_width"]).validate()
    with pytest.raises(ValueError):
        n01_config.ContentConfig(readout_tokens=128).validate()
    with pytest.raises(ValueError):
        n01_config.ReplayConfig(exact_window_unique_frames=12).validate()
    with pytest.raises(ValueError):
        n01_config.InputIdentityConfig(input_mode="rawvla_rgb3", channels=4).validate()
    with pytest.raises(ValueError):
        n01_config.InputIdentityConfig(input_mode="bayer_cfa", channels=3).validate()
    with pytest.raises(ValueError):
        n01_config.TrainingBudgetConfig(
            model_seeds=model_seeds, data_seeds=(model_seeds[0],)
        ).validate()
    with pytest.raises(ValueError):
        n01_config.N01FullConfig(
            input=identity, correspondence=n01_config.CorrespondenceConfig(input_channels=5)
        ).validate()
    with pytest.raises(ValueError):
        n01_config.N01FullConfig(
            input=identity, correspondence=n01_config.CorrespondenceConfig(input_channels=8)
        ).validate()


def test_protocol_pins_the_manifests_it_declares() -> None:
    """The protocol's per-region manifest hashes match the files on disk and the index.

    A manifest edited after the protocol (or an index reporting a different file) leaves
    the declared hash stale and fails here, which is exactly the binding the P0 receipt
    relies on.
    """
    protocol = _protocol()
    index = _index()
    for region in REGIONS:
        spec = protocol["data"]["splits"][region]
        counts = spec["counts"]
        assert set(counts) == set(spec["platform_manifests"]) == set(PLATFORMS)
        assert sum(int(counts[platform]) for platform in PLATFORMS) == int(
            spec["cross_platform_total"]
        )
        for platform, declared in spec["platform_manifests"].items():
            path = REPO_ROOT / str(declared["path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == declared["sha256"] == index["datasets"][platform]["manifest_sha256"]
            mine = len(_region_paths_from_rows(_manifest(platform))[region])
            assert mine == int(counts[platform])


# ---------------------------------------------------------------------------------------
# (5) packaging closure: the bundle carries the new P0 sources
# ---------------------------------------------------------------------------------------


def test_bundle_packages_new_sources_that_head_does_not_carry(tmp_path: Path) -> None:
    """The bundle is walked from the filesystem, so P0's new files survive packaging.

    A bundle built by ``git archive HEAD`` (or by any tracked-files-only listing) drops
    every source of this wave: the files below are absent from HEAD, so they can only be
    present here if the artifact really walks the worktree, and their absence is a
    failure rather than a warning.
    """
    output = tmp_path / "bundle"
    result = preflight.bundle(REPO_ROOT, output, include_upstream=False)
    manifest = _read_json(output / preflight.MANIFEST_NAME)
    bundled = set(manifest["files"])
    tracked_at_head = set(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-tree", "-r", "--name-only", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    new_sources = bundled - tracked_at_head
    assert new_sources, "the bundle equals HEAD's tree; a git-archive bundle would pass silently"
    assert set(P0_NEW_SOURCES) <= bundled
    assert set(P0_NEW_SOURCES) <= new_sources

    untracked = manifest["untracked_inclusion"]
    assert untracked["available"] is True
    assert set(P0_NEW_SOURCES) <= set(untracked["included"])
    assert not set(P0_NEW_SOURCES) & set(untracked["denied"])
    assert untracked["missing"] == []

    with tarfile.open(result["archive"], "r:gz") as archive:
        archived = set(archive.getnames())
    assert set(P0_NEW_SOURCES) <= archived
    assert preflight.MANIFEST_NAME in archived
    for relative, entry in manifest["files"].items():
        assert hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest() == entry["sha256"]
