#!/usr/bin/env python3
"""Survey the declared per-episode metadata of a deployed RawVLA-Bench cache.

Read-only. Every NPZ carries a ``metadata_json`` string written by the benchmark author; this
driver extracts the declared lighting / sensor / provenance fields for every raw episode and
measures value ranges on a lighting-stratified sample, so the P1 measurement model can use
*declared* conditions instead of assuming a fixed white level.

    PYTHONPATH=src python scripts/n02_p0_cache_metadata_scan.py --root <RawVLA-Bench> --output <dir>

Nothing is written into the cache; no training or evaluation process is started.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

#: Metadata fields worth recording; anything absent is reported as absent, never defaulted.
INTERESTING_KEYS = (
    "benchmark",
    "dataset_name",
    "lighting_domain",
    "lighting_ev",
    "lighting_scale",
    "lighting_sample_unit",
    "lighting_rig",
    "lighting_strategy",
    "raw_white_level",
    "raw_sensor_read_noise",
    "raw_sensor_shot_noise",
    "sensor_saturation_ev",
    "noise_seed",
    "dataset_seed",
    "demo_index",
    "episode",
    "entry_index",
    "clean_protocol",
)

MEASUREMENT_KEYS = {
    "libero": ("agentview_raw_uint8", "wrist_raw_uint8"),
    "robotwin": ("head_camera_raw_float32", "left_camera_raw_float32", "right_camera_raw_float32"),
}


def _nested(meta: dict, key: str):
    """Return ``meta[key]`` or the same key one level down (RoboTwin nests lighting_applied)."""
    if key in meta:
        return meta[key]
    for value in meta.values():
        if isinstance(value, dict) and key in value:
            return value[key]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--survey-per-domain", type=int, default=4, help="range survey files per lighting domain"
    )
    args = parser.parse_args()

    import numpy as np

    from owac.raw.rawvla_sequence_n02 import discover_npz_files

    root = pathlib.Path(args.root).resolve()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    result: dict = {"root": str(root), "platforms": {}}

    for fmt in ("libero", "robotwin"):
        # Use the loader's own marker discovery: the deployed RoboTwin directory is
        # ``data/robotwin2/raw``, so a hard-coded ``data/<format>/raw`` would silently find nothing.
        discovered, _pattern = discover_npz_files(root, fmt)
        paths = sorted(
            pathlib.Path(p) for p in discovered if "/raw/" in pathlib.PurePath(p).as_posix()
        )
        entries = []
        for path in paths:
            record = {"relative_path": str(path.relative_to(root))}
            with np.load(path, allow_pickle=False) as archive:
                raw = archive["metadata_json"] if "metadata_json" in archive.files else None
                if raw is None:
                    record["metadata_present"] = False
                else:
                    text = str(np.asarray(raw).ravel()[0])
                    try:
                        meta = json.loads(text)
                        record["metadata_present"] = True
                        record["metadata"] = {key: _nested(meta, key) for key in INTERESTING_KEYS}
                        record["metadata_keys"] = sorted(meta)
                    except json.JSONDecodeError as exc:  # recorded, not swallowed
                        record["metadata_present"] = True
                        record["metadata_parse_error"] = str(exc)
            entries.append(record)

        domains = collections.Counter(
            str(entry.get("metadata", {}).get("lighting_domain")) for entry in entries
        )
        evs = [
            entry["metadata"]["lighting_ev"]
            for entry in entries
            if isinstance(entry.get("metadata"), dict)
            and isinstance(entry["metadata"].get("lighting_ev"), (int, float))
        ]
        key_availability = collections.Counter()
        for entry in entries:
            for key in entry.get("metadata") or {}:
                if entry["metadata"].get(key) is not None:
                    key_availability[key] += 1

        # Lighting-stratified range survey: quantifies what the stored representation keeps.
        by_domain: dict[str, list] = collections.defaultdict(list)
        for path, entry in zip(paths, entries, strict=True):
            domain = str((entry.get("metadata") or {}).get("lighting_domain"))
            by_domain[domain].append(path)
        survey = []
        for domain in sorted(by_domain):
            for path in by_domain[domain][: args.survey_per_domain]:
                with np.load(path, allow_pickle=False) as archive:
                    for key in MEASUREMENT_KEYS[fmt]:
                        if key not in archive.files:
                            continue
                        values = np.asarray(archive[key])
                        flat = values.reshape(-1)
                        sample = flat[:: max(1, flat.size // 2_000_000)]
                        as_float = sample.astype(np.float64)
                        percentiles = np.percentile(as_float, [0, 50, 90, 99, 100])
                        survey.append(
                            {
                                "relative_path": str(path.relative_to(root)),
                                "lighting_domain": domain,
                                "key": key,
                                "dtype": str(values.dtype),
                                "shape": list(values.shape),
                                "sampled_elements": int(sample.size),
                                "percentiles_0_50_90_99_100": [float(v) for v in percentiles],
                                "zero_fraction": float((sample == 0).mean()),
                            }
                        )
        result["platforms"][fmt] = {
            "episodes": len(entries),
            "metadata_present": sum(1 for e in entries if e.get("metadata_present")),
            "metadata_parse_errors": [
                e["relative_path"] for e in entries if e.get("metadata_parse_error")
            ],
            "lighting_domains": dict(domains),
            "lighting_ev": {
                "count": len(evs),
                "min": min(evs) if evs else None,
                "median": float(np.median(evs)) if evs else None,
                "max": max(evs) if evs else None,
            },
            "metadata_key_availability": dict(key_availability),
            "metadata_keys_seen": sorted(
                {k for e in entries for k in (e.get("metadata_keys") or [])}
            ),
            "range_survey": survey,
            "entries": entries,
        }

    (out / "metadata_scan.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str)
    )
    compact = {
        fmt: {k: v for k, v in payload.items() if k != "entries"}
        | {"entries_recorded": len(payload["entries"])}
        for fmt, payload in result["platforms"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
