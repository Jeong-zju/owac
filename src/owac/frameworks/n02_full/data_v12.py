"""Real RAW windows on the frozen P0 partitions (N02 v1.2 P2).

This module adds no loader.  It composes the two frozen entry points:

* :class:`owac.raw.rawvla_sequence.RawVLAWindowDataset` - the only code that reads an NPZ,
  converts units, normalises actions/proprio and builds the causal window with its masks;
* :mod:`owac.frameworks.n02_full.data` - the P0 partition manifest, anchor table and
  :class:`ManifestWindowDataset` that restrict loading to one declared partition.

What v1.2 adds on top of the frozen P0 anchors (the P0 receipt froze ``rnn_frames=1``):

* the window is regenerated with ``history`` causal observation frames (default 8) on the
  *same* partition identity.  The anchor set is rebuilt with ``build_anchor_table`` from the
  same manifest and the ``(relative_path, current_frame_index, action_start, action_stop,
  action_mask_hash)`` signature is compared with the frozen anchor receipt before any window
  is served; a rebuild that does not reproduce the frozen anchors raises instead of quietly
  serving a different subset of the cache.
* every batch row carries its own causal frame index.  A row never looks past its anchor
  frame, and the short prefix is padded by repeating the oldest real frame with
  ``new_measurement=False``, so a repeated frame is not new evidence.
* one deduplicated measurement identity set ``(episode, frame, camera)`` is exposed per
  batch, so inner-loop rounds, prefix replays and multi-scale features of one frame stay
  one piece of evidence.  Counting is a plain Python set, not a hash ledger.

Numeric boundary (matching P0, do not stack a second normalisation on top): after
``load_formal_normalisation`` the sample's ``proprio`` and ``supervision_actions`` are
already normalised by the loader, and ``prepare_control_targets()`` / ``control_actions()``
only pad to 32 dims / slice back to 7 or 14.  Nothing here normalises or de-normalises;
online de-normalisation belongs to the training/evaluation entry points.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.frameworks.n02_full.data import (
    AnchorTable,
    ManifestWindowDataset,
    PartitionManifest,
    build_anchor_table,
    build_partition_manifest,
    load_formal_normalisation,
    sample_frames,
    verify_partitions,
)
from owac.raw.rawvla_sequence import FORMAT_SPECS, RawVLAWindowDataset, WindowOptions

__all__ = [
    "PARTITIONS",
    "V12WindowBatch",
    "V12WindowSource",
    "anchor_signature",
    "load_anchor_table",
    "measurement_ids_of_sample",
]

#: The three partitions the P0 manifest allocates and that must stay isolated.
PARTITIONS = ("train", "dev", "calibration")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_anchor_table(path: str | Path) -> tuple[AnchorTable, dict[str, Any]]:
    """Read a frozen P0 anchor table (``.json`` or ``.json.gz``) plus its receipt."""
    raw = Path(path).read_bytes()
    text = gzip.decompress(raw).decode("utf-8") if str(path).endswith(".gz") else raw.decode()
    table = AnchorTable.from_json(text)
    receipt = {
        "path": str(path),
        "gzip_sha256": hashlib.sha256(raw).hexdigest(),
        "json_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "version": table.version,
        "partition": table.partition,
        "dataset_format": table.dataset_format,
        "entries": len(table),
        "window": dict(table.window),
    }
    return table, receipt


def anchor_signature(table: AnchorTable) -> tuple[tuple[Any, ...], ...]:
    """The part of an anchor table that a window change must not alter.

    ``history_indices`` is excluded on purpose: v1.2 regenerates the *window* on the same
    partition identity, so the anchor frame and its supervised action bounds stay frozen
    while the history length changes.
    """
    return tuple(
        (
            entry.relative_path,
            int(entry.current_frame_index),
            int(entry.action_start),
            int(entry.action_stop),
            str(entry.action_mask_hash),
        )
        for entry in table
    )


def _first_mismatch(
    left: Sequence[tuple[Any, ...]], right: Sequence[tuple[Any, ...]]
) -> dict[str, Any]:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return {"index": index, "expected": list(b), "got": list(a)}
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "expected_count": len(right),
            "got_count": len(left),
        }
    return {}


def measurement_ids_of_sample(sample: Any) -> tuple[tuple[str, str, str], ...]:
    """The deduplicated ``(episode_id, frame_id, camera_id)`` set of one loaded sample.

    The frozen loader puts one identity per ``(row, camera)`` in ``frame_ids``; a frame that
    the window repeated as padding keeps the same identity, so the set collapses it.
    """
    seen: set[tuple[str, str, str]] = set()
    episode_ids = tuple(getattr(sample, "episode_ids", ())) or tuple(
        sample.episode.episode_group_id for _ in range(len(sample.observation_indices))
    )
    for camera_slot, camera_id in enumerate(sample.camera_ids):
        for row, frame_id in enumerate(sample.frame_ids[camera_slot]):
            episode_id = (
                episode_ids[row] if row < len(episode_ids) else sample.episode.episode_group_id
            )
            seen.add((str(episode_id), str(frame_id[-1]), str(camera_id)))
    return tuple(sorted(seen))


@dataclass(slots=True)
class V12WindowBatch:
    """One decision moment's real window batch (partition identity already frozen).

    Rows follow the loader's causal order: ``batch_size`` anchors, each contributing
    ``history`` rows oldest first, so row ``i * history + r`` is the ``r``-th oldest
    observation of anchor ``i``.  ``frame_index`` holds each row's real source frame index
    (frame indices, not fabricated seconds - the cache has no timestamps).
    """

    observation: Any
    frame_index: Any
    measurement_ids: tuple
    partition: str
    declared_prior: Any = None
    history: int = 1
    batch_size: int = 1
    anchors: tuple = ()
    row_spans: tuple = ()
    new_measurement: Any = None
    provenance: dict = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        """Number of causal rows in the batch (history rows times batch elements)."""
        return int(np.asarray(self.frame_index).size)

    @property
    def unique_measurement_count(self) -> int:
        return len(self.measurement_ids)

    def measurement_id_set(self) -> set[tuple[str, str, str]]:
        """The unique measurements of this batch as a plain set (no ledger object)."""
        return set(self.measurement_ids)

    def current_frame_indices(self) -> tuple[int, ...]:
        """The anchor (newest) frame index of every batch element."""
        values = np.asarray(self.frame_index)
        return tuple(
            int(values[anchor * self.history + self.history - 1])
            for anchor in range(self.batch_size)
        )

    def as_record(self) -> dict[str, Any]:
        """Small JSON-safe record for run manifests and reports."""
        return {
            "partition": self.partition,
            "history": int(self.history),
            "batch_size": int(self.batch_size),
            "row_count": int(np.asarray(self.frame_index).size),
            "unique_measurement_count": self.unique_measurement_count,
            "measurement_ids": [list(item) for item in self.measurement_ids],
            "frame_index": [int(value) for value in np.asarray(self.frame_index).tolist()],
            "anchors": [dict(item) for item in self.anchors],
        }


class V12WindowSource:
    """Serve real RAW windows of one frozen P0 partition; train/dev/calibration stay apart.

    The source refuses to start when

    * the manifest does not verify structurally (overlapping partitions / group leakage), or
    * the frozen anchor receipt is not reproduced exactly by rebuilding the anchors from the
      manifest with the receipt's own stride / burst / horizon / offset, or
    * the regenerated ``history``-frame anchors do not start from exactly the frozen anchor
      frames.
    """

    def __init__(
        self,
        *,
        dataset_format: str,
        data_root: str | Path,
        partition: str,
        history: int = 8,
        stats_path: str | Path | None = None,
        anchor_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        seed: int = 1103,
        batch_size: int = 1,
        stride: int | None = None,
        max_windows: int | None = None,
        noise: Any = None,
        declared_prior: Any = None,
        task_names: Sequence[str] | None = None,
        exclude_episode_ids: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        allowed = {"format_spec", "cfa_pattern", "cfa_calibration"}
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(f"unknown V12WindowSource keyword arguments: {unknown}")
        if dataset_format not in FORMAT_SPECS:
            raise ValueError(f"unknown dataset_format: {dataset_format!r}")
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
        if int(history) < 1:
            raise ValueError("history must be >= 1")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        if max_windows is not None and int(max_windows) < 1:
            raise ValueError("max_windows must be >= 1 when set")

        self.dataset_format = str(dataset_format)
        self.data_root = str(data_root)
        self.partition = str(partition)
        self.history = int(history)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.max_windows = None if max_windows is None else int(max_windows)
        self.declared_prior = declared_prior

        # -- manifest: load a receipt or rebuild it with the P0 entry point at the same seed
        if manifest_path is not None:
            manifest = PartitionManifest.from_json(Path(manifest_path).read_text())
            self._manifest_source = {
                "kind": "receipt_file",
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
            }
        else:
            episodes = RawVLAWindowDataset(
                self.data_root, dataset_format=self.dataset_format
            ).episodes()
            manifest = build_partition_manifest(
                episodes,
                dataset_format=self.dataset_format,
                cache_root=self.data_root,
                seed=self.seed,
            )
            self._manifest_source = {
                "kind": "rebuilt_from_cache",
                "episodes": len(episodes),
                "seed": self.seed,
            }
        if partition not in manifest.partitions:
            have = sorted(manifest.partitions)
            raise KeyError(f"partition {partition!r} is not in the manifest; have {have}")
        verification = verify_partitions(manifest)
        if verification.get("violations"):
            raise ValueError(f"partition manifest does not verify: {verification['violations']}")
        self.manifest = manifest
        self._partition_verification = verification

        # -- frozen anchors: rebuild and compare before serving anything
        self._frozen_receipt: dict[str, Any] | None = None
        if anchor_path is not None:
            frozen, receipt = load_anchor_table(anchor_path)
            if frozen.partition != partition or frozen.dataset_format != self.dataset_format:
                raise ValueError(
                    f"anchor receipt is {frozen.dataset_format}/{frozen.partition}, "
                    f"source is {self.dataset_format}/{partition}"
                )
            reference = build_anchor_table(
                manifest,
                partition,
                stride=int(frozen.stride),
                rnn_frames=int(frozen.rnn_frames),
                burst_frames=int(frozen.burst_frames),
                action_horizon=int(frozen.action_horizon),
                action_offset=int(frozen.action_offset),
            )
            mismatch = _first_mismatch(anchor_signature(reference), anchor_signature(frozen))
            if mismatch:
                raise ValueError(
                    "the manifest does not reproduce the frozen anchors; refusing to serve "
                    f"windows: {mismatch}"
                )
            receipt = dict(receipt)
            receipt["reproduced_from_manifest"] = True
            receipt["reproduced_entries"] = len(reference)
            self._frozen_receipt = receipt
            grid_stride = int(frozen.stride)
            horizon = int(frozen.action_horizon)
            offset = int(frozen.action_offset)
            observation_stride = int(frozen.window.get("observation_stride", 1))
            frozen_pairs = tuple((e.relative_path, int(e.current_frame_index)) for e in frozen)
        else:
            spec = FORMAT_SPECS[self.dataset_format]
            grid_stride = 1 if stride is None else int(stride)
            horizon = int(spec.default_action_horizon)
            offset = int(spec.default_action_offset)
            observation_stride = int(spec.default_observation_stride)
            frozen_pairs = ()

        # -- the v1.2 window: same anchors, declared causal history
        window = WindowOptions(
            observation_stride=observation_stride,
            rnn_frames=self.history,
            burst_frames=1,
            action_horizon=horizon,
            action_offset=offset,
        ).resolved_for(FORMAT_SPECS[self.dataset_format])
        table = build_anchor_table(
            manifest,
            partition,
            window=window,
            stride=grid_stride,
            rnn_frames=self.history,
            burst_frames=1,
        )
        derived_pairs = tuple((e.relative_path, int(e.current_frame_index)) for e in table)
        if anchor_path is not None and derived_pairs != frozen_pairs:
            raise ValueError(
                "the regenerated history window does not start from the frozen anchor frames: "
                f"{_first_mismatch(derived_pairs, frozen_pairs)}"
            )
        self._selection = None
        if task_names or exclude_episode_ids:
            parent_hash = table.sha256()
            tasks, excluded = set(task_names or ()), set(exclude_episode_ids)
            selected = tuple(
                entry
                for entry in table
                if (not tasks or entry.relative_path.split("/")[-2] in tasks)
                and entry.episode_id not in excluded
            )
            if not selected:
                raise ValueError(f"empty declared task selection in {partition}")
            self._selection = {
                "tasks": sorted(tasks),
                "excluded_episodes": sorted(excluded),
                "parent_anchor_sha256": parent_hash,
                "parent_entries": len(table),
                "selected_entries": len(selected),
                "selected_episodes": sorted({entry.episode_id for entry in selected}),
            }
            table = dataclasses.replace(
                table,
                entries=selected,
                positions=(),
                notes=table.notes + "; explicit task subset of verified frozen parent",
            )
        self._anchor_pairs_preserved = bool(anchor_path is not None and self._selection is None)
        self.window_options = window
        self.anchor_table = table
        self.stride = grid_stride

        # -- the frozen loader (one normalisation / mask path for every consumer)
        stats = None
        stats_receipt: dict[str, Any] | None = None
        if stats_path is not None:
            stats = load_formal_normalisation(self.dataset_format, stats_path)
            stats_receipt = {
                "path": str(stats_path),
                "sha256": str(getattr(stats, "sha256", "")),
                "mode": str(getattr(stats, "mode", "")),
                "epsilon": float(getattr(stats, "epsilon", 0.0)),
                "epsilon_mode": str(getattr(stats, "epsilon_mode", "")),
            }
        self.stats = stats
        self._stats_receipt = stats_receipt
        self._dataset = ManifestWindowDataset(
            self.data_root,
            manifest,
            partition,
            table,
            window_options=window,
            normalisation=stats,
            noise=noise,
            format_spec=kwargs.get("format_spec"),
            cfa_pattern=str(kwargs.get("cfa_pattern", "")),
            cfa_calibration=kwargs.get("cfa_calibration"),
        )
        self._n_windows = (
            len(self.anchor_table)
            if self.max_windows is None
            else min(self.max_windows, len(self.anchor_table))
        )

    # -- introspection -----------------------------------------------------------------

    def __len__(self) -> int:
        return (self._n_windows + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[V12WindowBatch]:
        for start in range(0, self._n_windows, self.batch_size):
            yield self.window(start, min(start + self.batch_size, self._n_windows))

    @property
    def anchor_count(self) -> int:
        return len(self.anchor_table)

    def anchor(self, index: int) -> Any:
        return self.anchor_table[index]

    def identity(self) -> dict[str, Any]:
        """Run-manifest record: partition, manifest, anchors, window, stats."""
        spec = FORMAT_SPECS[self.dataset_format]
        return {
            "module": "owac.frameworks.n02_full.data_v12",
            "dataset_format": self.dataset_format,
            "partition": self.partition,
            "history": self.history,
            "batch_size": self.batch_size,
            "stride": self.stride,
            "observation_stride": int(self.window_options.observation_stride),
            "window": dataclasses.asdict(self.window_options),
            "cache_root": self.manifest.cache_root,
            "measurement_keys": list(spec.measurement_keys),
            "camera_ids": list(spec.camera_ids),
            "manifest": {
                "version": self.manifest.version,
                "content_hash": self.manifest.content_hash(),
                "seed": int(self.manifest.seed),
                "fractions": dict(self.manifest.fractions),
                "partition_sizes": {
                    name: len(paths) for name, paths in self.manifest.partitions.items()
                },
                "source": dict(self._manifest_source),
            },
            "partition_verification": self._partition_verification,
            "anchor_table": {
                "version": self.anchor_table.version,
                "sha256": self.anchor_table.sha256(),
                "entries": len(self.anchor_table),
                "notes": self.anchor_table.notes,
            },
            "frozen_anchor_receipt": self._frozen_receipt,
            "frozen_anchor_identity_preserved": self._anchor_pairs_preserved,
            "stats": self._stats_receipt,
            "selection": self._selection,
        }

    # -- sampling ----------------------------------------------------------------------

    def sample(self, index: int) -> Any:
        """The frozen loader's own ``RawVLASample`` for one anchor.

        Used for protocol checks and unit/range evidence: the sample is exactly what the P0
        loader produced, nothing is re-derived here.
        """
        return self._dataset[int(index)]

    def targets(self, start: int, stop: int | None = None) -> dict[str, Any]:
        """The supervised action block of anchors ``[start, stop)``.

        These are the loader's already-normalised targets (see the module docstring): the
        action dimension is the platform's 7 or 14 columns, the horizon and offset are the
        frozen ones, and ``action_mask`` marks the real (non-padding) entries.
        """
        stop = int(start) + 1 if stop is None else int(stop)
        samples = [self._dataset[index] for index in range(int(start), stop)]
        actions = np.concatenate(
            [np.asarray(sample.supervision_actions, dtype=np.float32) for sample in samples], axis=0
        )
        return {
            "actions": actions,
            "action_mask": np.concatenate(
                [np.asarray(sample.supervision_mask) for sample in samples], axis=0
            ),
            "supervision_indices": np.concatenate(
                [np.asarray(sample.supervision_indices) for sample in samples], axis=0
            ),
            "action_dim": int(actions.shape[-1]),
            "action_horizon": int(self.window_options.action_horizon),
            "action_offset": int(self.window_options.action_offset),
            "padding_contribution": [float(sample.padding_contribution) for sample in samples],
        }

    def window(self, start: int, stop: int | None = None) -> V12WindowBatch:
        """Load the windows of anchors ``[start, stop)`` as one batch."""
        stop = int(start) + 1 if stop is None else int(stop)
        if not 0 <= int(start) < stop <= len(self.anchor_table):
            raise IndexError(
                f"anchor range [{start}, {stop}) is outside [0, {len(self.anchor_table)})"
            )
        samples = [self._dataset[index] for index in range(int(start), stop)]
        return self._build_batch(samples, anchors=tuple(range(int(start), stop)))

    def _build_batch(self, samples: Sequence[Any], *, anchors: tuple[int, ...]) -> V12WindowBatch:
        camera_ids = tuple(samples[0].camera_ids)
        frames: dict[str, Any] = {}
        row_spans: list[dict[str, Any]] = []
        row = 0
        for anchor, sample in zip(anchors, samples, strict=True):
            sample_frames_by_camera = sample_frames(sample)
            span = {
                "anchor": int(anchor),
                "row_start": row,
                "row_stop": row + int(sample.observation_indices.size),
                "episode_group_id": str(sample.episode.episode_group_id),
                "relative_path": str(sample.episode.relative_path),
                "current_frame_index": int(sample.observation_indices[-1]),
            }
            row_spans.append(span)
            row += int(sample.observation_indices.size)
            for camera_id in camera_ids:
                frames.setdefault(camera_id, []).append(sample_frames_by_camera[camera_id])

        merged: dict[str, Any] = {}
        from owac.core.encoder_contract import MeasurementFrame

        for camera_id in camera_ids:
            payloads = frames[camera_id]
            first = payloads[0]
            values = np.concatenate([np.asarray(item.values) for item in payloads], axis=0)
            merged[camera_id] = MeasurementFrame(
                values=values,
                valid_mask=_concat_optional(payloads, "valid_mask"),
                bad_mask=_concat_optional(payloads, "bad_mask"),
                saturated_mask=_concat_optional(payloads, "saturated_mask"),
                sample_xy=first.sample_xy,
                episode_ids=tuple(
                    episode_id for item in payloads for episode_id in item.episode_ids
                ),
                frame_ids=tuple(frame_id for item in payloads for frame_id in item.frame_ids),
                camera_id=str(first.camera_id),
                timestamp=None,
                time_unit="frame",
                new_measurement=np.concatenate(
                    [np.asarray(item.new_measurement, dtype=bool) for item in payloads]
                ),
                metadata={
                    **dict(first.metadata),
                    "batched_row_spans": [dict(span) for span in row_spans],
                },
                metadata_available=dict(first.metadata_available),
            )

        from owac.frameworks.n02_full import state as state_module

        bundle = state_module.ObservationBundle(
            frames=merged,
            context=_batched_context(samples),
            episode_ids=tuple(
                episode_id
                for item in samples
                for episode_id in [item.episode.episode_group_id]
                * int(item.observation_indices.size)
            ),
            anchor_ids=tuple(
                f"{item.episode.episode_group_id}#anchor{int(item.observation_indices[-1]):06d}"
                for item in samples
                for _ in range(int(item.observation_indices.size))
            ),
            new_time_mask=np.concatenate(
                [np.asarray(item.new_measurement, dtype=bool) for item in samples]
            ),
        )
        frame_index = torch.tensor(
            np.concatenate(
                [np.asarray(item.observation_indices, dtype=np.int64) for item in samples]
            ),
            dtype=torch.int64,
        )
        measurement_ids: set[tuple[str, str, str]] = set()
        for item in samples:
            measurement_ids.update(measurement_ids_of_sample(item))
        return V12WindowBatch(
            observation=bundle,
            frame_index=frame_index,
            measurement_ids=tuple(sorted(measurement_ids)),
            partition=self.partition,
            declared_prior=self.declared_prior,
            history=self.history,
            batch_size=len(samples),
            anchors=tuple(self.anchor_table[anchor].as_dict() for anchor in anchors),
            row_spans=tuple(row_spans),
            new_measurement=bundle.new_time_mask,
            provenance={
                "source": "V12WindowSource",
                "dataset_format": self.dataset_format,
                "partition": self.partition,
                "manifest_hash": self.manifest.content_hash(),
                "anchor_table_sha256": self.anchor_table.sha256(),
                "history": self.history,
                "frame_index_unit": "frame",
                "timestamps_available": False,
                "stats_sha256": (self._stats_receipt or {}).get("sha256", ""),
            },
        )


def _concat_optional(payloads: Sequence[Any], name: str) -> Any:
    values = [getattr(item, name) for item in payloads]
    if any(value is None for value in values):
        return None
    return np.concatenate([np.asarray(value) for value in values], axis=0)


def _batched_context(samples: Sequence[Any]) -> Any:
    """The declared context of the batch: proprio rows, language, no fabricated action.

    ``proprio`` is the loader's already-normalised proprio of each anchor's current frame;
    it is expanded to one row per window row so that a model reading row-wise context sees
    the anchor's own state on every history row of that anchor.
    """
    from owac.core.encoder_contract import EmbodimentContext

    proprio_rows = []
    available = True
    for sample in samples:
        if sample.proprio is None or not sample.proprio_available:
            available = False
        rows = int(sample.observation_indices.size)
        row = (
            np.asarray(sample.proprio, dtype=np.float32)
            if sample.proprio is not None
            else np.zeros((1, 0), dtype=np.float32)
        )
        proprio_rows.append(np.broadcast_to(row, (rows,) + row.shape).copy())
    proprio = np.concatenate(proprio_rows, axis=0) if proprio_rows else None
    descriptions = [str(sample.provenance.get("task_description", "")) for sample in samples]
    task = descriptions[0] if descriptions else ""
    return EmbodimentContext(
        proprio=proprio,
        proprio_available=bool(available and proprio is not None),
        executed_action=None,
        executed_action_available=False,
        camera_transform=None,
        camera_transform_available=False,
        task_tokens=task or None,
        task_mask=bool(task),
        hypothesis_prior=None,
        hypothesis_prior_available=False,
    )


def read_task_prompts(
    relative_paths: list[str], data_root: str, *, instructions: dict | None = None
) -> list[str]:
    """Per-anchor task instruction, read the way the frozen loader reads it.

    Only the small ``metadata_json`` member of each NPZ is opened, so this does not duplicate the
    window load.  The fallback rule is the loader's own: ``task_description`` first and, for
    RoboTwin, ``task_name`` with underscores turned into spaces.
    """
    prompts: list[str] = []
    for relative in relative_paths:
        metadata: dict = {}
        try:
            with np.load(Path(data_root) / str(relative), allow_pickle=False) as archive:
                metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            metadata = {}
        text = ""
        if isinstance(metadata, dict):
            text = str(metadata.get("task_description") or "")
            if not text and "robotwin" in str(relative):
                text = str(metadata.get("task_name") or "").replace("_", " ")
        if instructions:
            from owac.backends.rawvla_language import canonical_task_prompt

            text = canonical_task_prompt(metadata.get("task_name"), text, instructions)
        prompts.append(text)
    return prompts


def apply_declared_isp(bundle: Any, spec: dict) -> Any:
    """Apply the arm's declared, per-frame-independent ISP to the loader's RAW values.

    The RAW arm declares ``kind="raw3"`` and nothing is touched.  The controlled RGB arm declares
    ``kind="raw3_fixed_isp"``: the *same* RAW measurement of the *same* acquisition passes through
    one written-down transform with no per-frame statistic in it (clip / fixed scale, per-channel
    gain, gamma encoding, 8-bit quantisation).  No official normal-light RGB pair is substituted.
    """
    import dataclasses

    kind = str(spec.get("kind", "raw3"))
    if kind in ("", "raw3"):
        return bundle
    if kind != "raw3_fixed_isp":
        raise ValueError(f"unknown representation kind {kind!r}")
    isp = dict(spec.get("isp") or {})
    scale = float(isp.get("input_scale", 1.0))
    gain = torch.as_tensor(isp.get("gain", (1.0, 1.0, 1.0)), dtype=torch.float32).view(1, -1, 1, 1)
    gamma = float(isp.get("gamma", 2.2))
    levels = float(2 ** int(isp.get("quantise_bits", 8)) - 1)
    frames = {}
    for camera_id, frame in bundle.frames.items():
        values = torch.as_tensor(frame.values).to(torch.float32)
        if values.ndim < 4 or int(values.shape[1]) < 3:
            frames[camera_id] = frame
            continue
        transformed = (values / scale).clamp(0.0, 1.0)
        transformed = (transformed * gain.to(transformed.device)).clamp(0.0, 1.0)
        if gamma != 1.0:
            transformed = transformed.clamp_min(0.0).pow(1.0 / gamma)
        transformed = torch.round(transformed * levels) / levels
        frames[camera_id] = dataclasses.replace(
            frame,
            values=transformed,
            metadata={
                **dict(getattr(frame, "metadata", {}) or {}),
                "declared_isp": "raw3_fixed_isp",
                "isp_scale": scale,
                "isp_gamma": gamma,
                "isp_quantise_bits": int(isp.get("quantise_bits", 8)),
            },
        )
    return dataclasses.replace(bundle, frames=frames)


class RepresentedWindowSource:
    """Same frozen sample identities/targets, with the arm's explicitly declared ISP."""

    def __init__(self, source: Any, representation: dict) -> None:
        self.source = source
        self.representation = representation

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def raw_window(self, start: int, stop: int) -> Any:
        return self.source.window(start, stop)

    def represent(self, batch: Any) -> Any:
        return dataclasses.replace(
            batch, observation=apply_declared_isp(batch.observation, self.representation)
        )

    def window(self, start: int, stop: int) -> Any:
        return self.represent(self.raw_window(start, stop))

    def identity(self) -> dict:
        return {**self.source.identity(), "representation": self.representation}
