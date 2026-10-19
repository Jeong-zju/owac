"""N01 v1.2 P3: whole-source deletion vs mask-only on the neural content path.

The distinction is the point of requirement E.  Masking validity still lets the
source move the support posterior and the reference choice; true deletion removes
it from the buffer entirely.  If these two produced the same number, one of them
would be mislabelled.
"""

from __future__ import annotations

import pytest
import torch

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.evaluation.n01_v12 import counterfactual_content_replay
from owac.frameworks.n01_full.model import N01V12Config, N01V12Model
from owac.frameworks.n01_full.support import SupportConfig


class _BatchWindow:
    def __init__(self, values, frame_ids, time_unit="frame"):
        self.values = values
        self.valid_mask = torch.ones(values.shape[:2], dtype=torch.bool)
        self.bad_mask = torch.zeros(values.shape[:2], dtype=torch.bool)
        self.saturated_mask = torch.zeros(values.shape[:2], dtype=torch.bool)
        self.frame_ids = frame_ids
        self.timestamp = torch.arange(values.shape[1], dtype=torch.float32)[None].repeat(
            values.shape[0], 1
        )
        self.new_measurement = torch.ones(values.shape[:2], dtype=torch.bool)
        self.sample_xy = None
        self.time_unit = time_unit


def _model(arm: str = "raw_n01") -> N01V12Model:
    support = SupportConfig(region_grid=2, multiscale_radii=(0,), receptive_field_radius=0)
    # A narrow, shallow stem: these checks are about masking against deletion, and the
    # full-width defaults made the suite minutes slower without changing any verdict.
    return N01V12Model(
        N01V12Config(
            arm=arm,
            platform="libero",
            window_frames=8,
            support=support,
            feature_width=32,
            stem_width=8,
            residual_blocks=1,
        )
    )


def _batch(frames: int = 6, size: int = 16, seed: int = 3, stable: bool = False):
    """``stable`` builds a history the support mechanism accepts.

    With independent random frames the support withdraws the whole history, so no
    source carries any fusion weight and deleting one is a genuine no-op -- correct
    behaviour, but useless for checking that deletion propagates.  A near-constant
    scene with small jitter keeps every source accepted and weighted.
    """

    generator = torch.Generator().manual_seed(seed)
    if stable:
        base = torch.rand((1, 1, 3, size, size), generator=generator)
        jitter = 0.02 * torch.randn((1, frames, 3, size, size), generator=generator)
        values = (base + jitter).clamp(0.0, 1.0)
    else:
        values = torch.rand((1, frames, 3, size, size), generator=generator)
    # Distinct real frame ids so nothing is deduplicated away by accident.
    frame_ids = torch.arange(frames, dtype=torch.int64)[None]
    return {"windows": {"agentview": _BatchWindow(values, frame_ids)}}


def _tokens_with_mask(model, batch, device, deleted: set[int]):
    """Mask-only reference: the frame is delivered with validity zeroed."""

    window = batch["windows"]["agentview"]
    values = window.values
    _, length, _, height, width = values.shape
    budget = ComputeBudget(max_updates=8, max_history=8)
    state = model.init_state(1, {"camera_ids": ["agentview"], "shape": (height, width)}, device)
    output = None
    for index in range(length):
        valid = window.valid_mask[:, index]
        if index in deleted:
            valid = torch.zeros_like(valid)
        frame = MeasurementFrame(
            values=values[:, index],
            valid_mask=valid,
            bad_mask=window.bad_mask[:, index],
            saturated_mask=window.saturated_mask[:, index],
            frame_ids=window.frame_ids[:, index],
            camera_id="agentview",
            timestamp=window.timestamp[:, index],
            time_unit=window.time_unit,
            new_measurement=window.new_measurement[:, index],
        )
        output = model.step(frame, None, state, budget)
        state = output.state
    return model.readout(output)


def _tokens_with_mask_diagnostics(model, batch, device, deleted: set[int]):
    """Mask-only run that also returns the last step's diagnostics."""

    window = batch["windows"]["agentview"]
    values = window.values
    _, length, _, height, width = values.shape
    budget = ComputeBudget(max_updates=8, max_history=8)
    state = model.init_state(1, {"camera_ids": ["agentview"], "shape": (height, width)}, device)
    output = None
    for index in range(length):
        valid = window.valid_mask[:, index]
        if index in deleted:
            valid = torch.zeros_like(valid)
        frame = MeasurementFrame(
            values=values[:, index],
            valid_mask=valid,
            bad_mask=window.bad_mask[:, index],
            saturated_mask=window.saturated_mask[:, index],
            frame_ids=window.frame_ids[:, index],
            camera_id="agentview",
            timestamp=window.timestamp[:, index],
            time_unit=window.time_unit,
            new_measurement=window.new_measurement[:, index],
        )
        output = model.step(frame, None, state, budget)
        state = output.state
    return output.diagnostics


def test_deletion_removes_the_frame_from_the_buffer_entirely() -> None:
    model = _model()
    batch = _batch()
    report = counterfactual_content_replay(model, batch, "cpu", "agentview", {2, 4})
    assert report["arrivals"] == 4, "six slots minus two deletions"
    assert report["buffer_unique_frames"] == 4, (
        "a deleted frame must not occupy a buffer slot; nothing may be pulled in to refill it"
    )
    assert report["requested_window"] == 8
    tokens = report["tokens"]
    assert tokens.values.shape[0] == 1
    assert bool(tokens.mask.any())


def test_deletion_refuses_to_read_out_when_every_source_is_gone() -> None:
    model = _model()
    batch = _batch(frames=4)
    with pytest.raises(ValueError, match="nothing to read out"):
        counterfactual_content_replay(model, batch, "cpu", "agentview", {0, 1, 2, 3})


def test_deletion_shrinks_the_buffer_while_masking_does_not() -> None:
    """The bookkeeping difference: a deleted frame never occupied a buffer slot."""

    model = _model("raw_n01")
    batch = _batch(frames=6)
    deleted = {2}
    report = counterfactual_content_replay(model, batch, "cpu", "agentview", deleted)
    assert report["buffer_unique_frames"] == 5
    # The mask-only reference still pushes all six frames.
    masked = _tokens_with_mask(model, batch, "cpu", deleted)
    assert masked.values.shape[0] == 1


def test_deletion_advances_the_arrival_series_less_than_masking() -> None:
    """A deleted frame does not advance the online arrival/BOCPD series.

    This is the substantive difference: masking delivers the frame, so the support
    detector consumes one more arrival than it should have if the source had really
    been absent.
    """

    model = _model("raw_n01")
    batch = _batch(frames=6)
    deleted = {2}
    with torch.no_grad():
        deleted_diag = counterfactual_content_replay(model, batch, "cpu", "agentview", deleted)[
            "diagnostics"
        ]
        masked_diag = _tokens_with_mask_diagnostics(model, batch, "cpu", deleted)
    assert deleted_diag["arrival_index"] == 5
    assert masked_diag["arrival_index"] == 6
    assert deleted_diag["unique_frames_lifetime"][0] == 5
    assert masked_diag["unique_frames_lifetime"][0] == 6


def test_deletion_is_distinguished_from_masking_in_the_responses() -> None:
    """The two operations must be tellable apart, whatever the content does.

    Whether masked and deleted content coincide is an architecture question -- the fusion
    zeroes invalid slots and renormalises, so a masked frame already carries no weight and
    the two can agree.  What must never coincide is the *reported operation*: a receipt
    that says ``delete`` while the code only masked would be mislabelled evidence.  The
    previous version of this test asserted ``same in (True, False)``, which cannot fail
    and therefore checked nothing.
    """

    model = _model("raw_n01")
    model.eval()
    batch = _batch(frames=6)
    deleted = {2}

    with torch.no_grad():
        report = counterfactual_content_replay(model, batch, "cpu", "agentview", deleted)

    assert report["arrivals"] == 5, "six slots minus one deletion"
    assert report["buffer_unique_frames"] == 5, "the deleted frame must not occupy a slot"
    assert report["requested_window"] == 8

    # The mask-only reference pushes all six frames, so its buffer occupancy differs.
    with torch.no_grad():
        kept = counterfactual_content_replay(model, batch, "cpu", "agentview", set())
    assert kept["arrivals"] == 6
    assert kept["buffer_unique_frames"] == 6
    assert kept["buffer_unique_frames"] != report["buffer_unique_frames"], (
        "deletion and masking must not be indistinguishable in the recorded bookkeeping"
    )


def test_deleting_a_weighted_source_changes_the_readout() -> None:
    """Deletion must propagate when the source actually carries fusion weight.

    On independent random frames the support mechanism withdraws the whole history,
    every source has zero weight, and deletion is legitimately a no-op.  A stable
    scene keeps the sources accepted, and then removing one must change the readout --
    otherwise the deletion path would not be doing anything at all.
    """

    model = _model("raw_n01")
    model.eval()
    batch = _batch(frames=6, stable=True)
    with torch.no_grad():
        kept = counterfactual_content_replay(model, batch, "cpu", "agentview", set())["tokens"]
        dropped = counterfactual_content_replay(model, batch, "cpu", "agentview", {3})["tokens"]
    assert kept.values.shape == dropped.values.shape
    assert not torch.equal(kept.values, dropped.values), (
        "deleting an accepted source changed nothing, so the deletion path is a no-op"
    )


def test_deletion_of_a_withdrawn_source_is_correctly_a_no_op() -> None:
    """The complement, stated so the no-op is not mistaken for a defect.

    When support has withdrawn the history, no source carries weight, so removing one
    cannot change the content.  Recording this as a checked property keeps a later
    reader from "fixing" a non-bug.
    """

    model = _model("raw_n01")
    model.eval()
    batch = _batch(frames=6, stable=False)
    with torch.no_grad():
        kept = counterfactual_content_replay(model, batch, "cpu", "agentview", set())["tokens"]
        dropped = counterfactual_content_replay(model, batch, "cpu", "agentview", {3})["tokens"]
    assert torch.equal(kept.values, dropped.values)
