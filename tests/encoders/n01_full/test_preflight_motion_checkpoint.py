"""Stem-pairing diagnostics for ``--motion-checkpoint`` on the raw_adapter path.

A checkpoint trained on the wrong stem used to fail inside torch as a raw
``load_state_dict`` size mismatch.  The preflight now infers the checkpoint stem
from ``fnet.conv1.weight`` / ``cnet.conv1.weight`` and refuses a mismatch with
both sides' channel counts, the protocol's pairing rule and the recommendation,
before any estimator is built and without switching the mode.

The checkpoints here are tiny state dicts (226 MB files are not needed): the
motion core's guarded loader accepts a plain tensor state dict and its versioned
schema, both written with ``torch.save``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from owac.backends.raw_correspondence import MOTION_CHECKPOINT_SCHEMA
from owac.frameworks.n01_full.adapters import DbsrAdapterConfig
from owac.runtime.n01_full_policy_preflight import (
    MOTION_STEM_CHANNELS,
    _motion_checkpoint_stem,
    _require_matching_motion_stem,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "configs/encoders/n01_full/protocol_v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _write_checkpoint(
    path: Path,
    fnet: int,
    cnet: int,
    *,
    versioned: bool = False,
    declared_mode: str | None = None,
) -> Path:
    state = {
        "fnet.conv1.weight": torch.zeros(64, fnet, 7, 7),
        "cnet.conv1.weight": torch.zeros(64, cnet, 7, 7),
    }
    if not versioned:
        torch.save(state, path)
        return path
    payload: dict = {"schema": MOTION_CHECKPOINT_SCHEMA, "model": state, "core_fingerprint": "fx"}
    if declared_mode is not None:
        payload["mode"] = declared_mode
    torch.save(payload, path)
    return path


def test_cfa4_checkpoint_with_rgb3_windows_is_refused_with_both_sides(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path / "cfa4.pth", 4, 8)
    protocol = _protocol()
    with pytest.raises(ValueError) as excinfo:
        _require_matching_motion_stem(
            checkpoint, DbsrAdapterConfig(input_mode="rawvla_rgb3"), protocol
        )
    message = str(excinfo.value)
    assert "stem mismatch" in message
    assert "cfa4" in message and "rgb3" in message
    # Both sides' channel counts are in the message.
    assert "fnet 4 input channels" in message
    assert "cnet 8 channels" in message
    assert "3 measured channels" in message
    assert "needs 6" in message
    # The protocol pairing rule and the actionable recommendation are quoted.
    rule = protocol["coordinate_units"]["input_channels_multiple_rule"]
    assert rule in message
    assert "rawvla_rgb3 (3/6)" in message
    assert "never switched or adapted implicitly" in message


def test_rgb3_checkpoint_with_rgb3_windows_is_accepted_unchanged(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path / "rgb3.pth", 3, 6)
    report = _require_matching_motion_stem(
        checkpoint, DbsrAdapterConfig(input_mode="rawvla_rgb3"), _protocol()
    )
    assert report["inferred_mode"] == "rgb3"
    assert (report["fnet_channels"], report["cnet_channels"]) == (3, 6)
    assert report["schema"] == "plain_state_dict"
    assert report["expected_pairs"] == {"cfa4": [4, 8], "rgb3": [3, 6]}


def test_rgb3_checkpoint_with_cfa_windows_is_refused_the_other_way(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path / "rgb3.pth", 3, 6)
    with pytest.raises(ValueError, match="cfa4 windows with 4 measured channels"):
        _require_matching_motion_stem(
            checkpoint, DbsrAdapterConfig(input_mode="bayer_cfa"), _protocol()
        )


def test_versioned_checkpoint_with_matching_stem_and_declaration_is_accepted(tmp_path: Path):
    checkpoint = _write_checkpoint(
        tmp_path / "cfa4-versioned.pth", 4, 8, versioned=True, declared_mode="cfa4"
    )
    report = _require_matching_motion_stem(
        checkpoint, DbsrAdapterConfig(input_mode="bayer_cfa"), _protocol()
    )
    assert report["schema"] == MOTION_CHECKPOINT_SCHEMA
    assert report["declared_mode"] == "cfa4"
    assert report["inferred_mode"] == "cfa4"


def test_checkpoint_whose_declaration_contradicts_its_tensors_is_refused(tmp_path: Path):
    checkpoint = _write_checkpoint(
        tmp_path / "liar.pth", 3, 6, versioned=True, declared_mode="cfa4"
    )
    with pytest.raises(ValueError, match="declaration and the weights disagree"):
        _motion_checkpoint_stem(checkpoint)


def test_unknown_stem_pair_is_refused_with_the_two_accepted_pairs(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path / "odd.pth", 8, 4)
    with pytest.raises(ValueError, match="the two accepted stems"):
        _motion_checkpoint_stem(checkpoint)


def test_checkpoint_without_the_first_convolutions_is_refused(tmp_path: Path):
    path = tmp_path / "empty.pth"
    torch.save({"dummy.weight": torch.zeros(2, 2)}, path)
    with pytest.raises(ValueError, match="no usable 'fnet.conv1.weight'"):
        _motion_checkpoint_stem(path)


def test_missing_checkpoint_file_is_reported_before_any_loading(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="motion checkpoint not found"):
        _motion_checkpoint_stem(tmp_path / "absent.pth")


def test_stem_pairs_come_from_the_declared_protocol_rule():
    """The two accepted pairs are exactly the rule's CFA 4/8 and rgb3 3/6."""
    rule = _protocol()["coordinate_units"]["input_channels_multiple_rule"]
    assert MOTION_STEM_CHANNELS == {"cfa4": (4, 8), "rgb3": (3, 6)}
    assert "CFA 4/8" in rule and "rawvla_rgb3 3/6" in rule
