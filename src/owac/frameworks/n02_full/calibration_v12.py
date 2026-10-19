"""Serializable fixed-weight calibration; stale weights cannot reuse thresholds."""

import hashlib

import torch

from owac.frameworks.n02_full.checks import ViewingRule, WindowedCalibrator
from owac.frameworks.n02_full.checks_v12 import WindowedCalibration

FORMAT = "n02-fixed-weight-calibration-r11-v1"


def parameter_fingerprint(model, adapters=(), bridge_modules=()):
    digest = hashlib.sha256()
    modules = [("frontend", model)]
    modules.extend((f"adapter/{i}", module) for i, module in enumerate(adapters))
    modules.extend((f"bridge/{i}", module) for i, module in enumerate(bridge_modules))
    for prefix, module in modules:
        for name, value in sorted(module.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update(f"{prefix}/{name}/{value.dtype}/{tuple(value.shape)}".encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def pack_calibration(calibration, *, fingerprint, source_identity):
    return {
        "format": FORMAT,
        "weights_sha256": fingerprint,
        "source_identity": source_identity,
        "calibration": calibration.as_dict(),
        "rule": calibration.rule.as_dict(),
    }


def restore_calibration(payload, *, fingerprint, source_identity):
    if payload.get("format") != FORMAT or payload.get("weights_sha256") != fingerprint:
        raise ValueError("calibration belongs to different weights")
    if payload.get("source_identity") != source_identity:
        raise ValueError("calibration partition identity changed")
    raw_rule = {k: v for k, v in payload["rule"].items() if k != "version"}
    rule = ViewingRule(**raw_rule)
    saved = payload["calibration"]
    if saved["partition"] != "calibration" or saved["viewing_rule_hash"] != rule.rule_hash():
        raise ValueError("invalid calibration partition/viewing rule")
    core = WindowedCalibrator(
        threshold=saved["threshold"],
        quantile=saved["quantile"],
        rule_hash=rule.rule_hash(),
        n_samples=saved["finite_sequences"],
    )
    return WindowedCalibration(
        core,
        saved["version"],
        "calibration",
        rule,
        saved["quantile"],
        saved["n_sequences"],
        saved["finite_sequences"],
    )
