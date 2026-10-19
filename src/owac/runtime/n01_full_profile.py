"""Executable P0 cost probe of existing DBSR modules, not the P4 encoder.

The factory consumes oldest-first four-channel synthetic CFA windows. Histories
above eight bypass only the old state buffer; no N01 support/replay is claimed.
"""

from __future__ import annotations


def probe_loaders(protocol, repo_root, *, samples=20):
    """Read each frozen region; time warm sequence output bytes, not disk bandwidth."""
    import json
    import time
    from pathlib import Path

    import torch

    from owac.raw.n01_full_manifest import REGIONS, dataset_from_protocol
    from owac.runtime.n01_full_preflight import percentile

    result = {}
    for platform, spec in protocol["data"]["platform_manifests"].items():
        manifest = json.loads((Path(repo_root) / spec["manifest_path"]).read_text())
        regions = {}
        for region in REGIONS:
            dataset = dataset_from_protocol(protocol, platform, region, repo_root=repo_root)
            expected = [str(dataset.raw_cache_root / p) for p in manifest["sides"][region]]
            if list(dataset.paths) != expected:
                raise ValueError(f"{platform}/{region}: loader paths differ from manifest")
            first = dataset[0]
            windows = {}
            for camera, window in first.windows.items():
                if window.channels != 3 or not bool(torch.isfinite(window.values).all()):
                    raise ValueError(f"{platform}/{region}: invalid real RAW window")
                windows[camera] = {
                    "shape": list(window.values.shape),
                    "min": float(window.values.min()),
                    "max": float(window.values.max()),
                    "frame_ids": window.frame_ids.tolist(),
                    "new_measurement": window.new_measurement.tolist(),
                }
            durations = []
            total_bytes = 0
            # Warm the decoded episode cache above. Each timed call materializes a
            # different causal window; bytes count returned RAW values only.
            for index in range(samples):
                started = time.perf_counter()
                sample = dataset[index]
                durations.append(time.perf_counter() - started)
                total_bytes += sum(
                    w.values.numel() * w.values.element_size() for w in sample.windows.values()
                )
            regions[region] = {
                "paths_equal": True,
                "files": len(expected),
                "first_path": expected[0],
                "windows": windows,
                "action_shape": list(first.action.shape),
                "action_finite": bool(torch.isfinite(first.action).all()),
                "state_kind": first.state_kind,
                "stats_key": dataset.stats_key,
                "action_stats_key": dataset.action_stats_key,
                "samples": samples,
                "raw_output_bytes": total_bytes,
                "seconds": sum(durations),
                "p50_seconds": percentile(durations, 0.5),
                "p95_seconds": percentile(durations, 0.95),
                "decoded_raw_mib_per_second": total_bytes / 2**20 / sum(durations),
                "io_scope": (
                    "single-process warm decoded-episode cache; RAW output tensor "
                    "throughput, not disk bandwidth"
                ),
            }
            del first, sample, dataset
        result[platform] = regions
    return result


def dbsr_factory(*, protocol, point, device, backbone="dbsr"):
    import torch

    from owac.baselines.burst_features import (
        BurstFeatureConfig,
        BurstFeatureEncoder,
        MaskedDbsrFusion,
        RawBurstBackbone,
        align_features,
    )

    channels = point.get("channels", protocol.get("profiling", {}).get("measurement_channels", 4))
    if backbone != "dbsr" or channels != 4:
        raise ValueError("P0 DBSR probe supports only the existing four-channel CFA backbone")
    config = BurstFeatureConfig(max_history=int(point["history"]))

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.entry = "encoder" if config.max_history <= 8 else "direct"
            if self.entry == "encoder":
                self.encoder = BurstFeatureEncoder(config)
            else:
                self.backbone = RawBurstBackbone(config)
                self.fusion = MaskedDbsrFusion(config)

        def forward(self, oldest_first):
            values = oldest_first.flip(1)
            valid = torch.isfinite(values)
            b, t, c, h, w = values.shape
            offsets = values.new_zeros(b, t, 2, h, w)
            if self.entry == "encoder":
                return self.encoder(values, valid, offsets)[0]
            centers = valid.all(2, keepdim=True)
            features = self.backbone(values.masked_fill(~valid, 0).reshape(b * t, c, h, w))
            features = features.reshape(b, t, -1, h, w).masked_fill(~centers, 0)
            aligned, usable, offsets = align_features(features, centers, offsets)
            result, _ = self.fusion(aligned, usable, offsets)
            return result.masked_fill(~usable.any(1), 0)

    return Probe().to(device)
