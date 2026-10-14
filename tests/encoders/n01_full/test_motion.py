"""P2 RAW motion adapter: geometry, sampling coefficients and vendor evidence.

Every expectation here is decided by an independent oracle or by a property the
old implementation would fail: a constant-flow test double gives known packed
displacements, an affine source field is reproduced exactly by bilinear
sampling, the pull sign is checked in both directions, out-of-frame taps must be
absent rather than wrapped, and the vendored files are checked against the
hashes recorded in ``provenance.json``.  The real vendored network runs on CPU
only through the shipped 128-cell minimum window.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from owac.backends.raw_correspondence import (
    CFA_PHASE_OFFSETS,
    CorrespondenceEstimator,
    RawNormalization,
    RawSensorFrame,
)
from owac.baselines._vendor.sea_raft.extractor import ResNetFPN
from owac.baselines._vendor.sea_raft.raft import RAFT
from owac.baselines._vendor.sea_raft.utils.utils import InputPadder
from owac.frameworks.n01_full.config import CorrespondenceConfig

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).parents[3]
VENDOR = REPO_ROOT / "src/owac/baselines/_vendor/sea_raft"
UPSTREAM_CACHE = REPO_ROOT / "outputs/n01-v1-prompt-preparation/upstream/princeton-vl/SEA-RAFT"
MIN_SIDE = 128


class ConstantFlow(nn.Module):
    """Test double with the vendored RAFT call signature and a known flow."""

    def __init__(self, dy: float, dx: float, log_scale=(0.0, 0.0), weight=(0.0, 0.0)):
        super().__init__()
        self.dy = dy
        self.dx = dx
        self.log_scale = log_scale
        self.weight = weight

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
        n, _, h, w = image1.shape
        flow = torch.zeros(n, 2, h, w)
        flow[:, 0] = self.dy
        flow[:, 1] = self.dx
        info = torch.zeros(n, 4, h, w)
        info[:, 0] = self.weight[0]
        info[:, 1] = self.weight[1]
        info[:, 2] = self.log_scale[0]
        info[:, 3] = self.log_scale[1]
        return {"final": flow, "info": info}


def cfa_frame(
    frame_id=0, values=None, validity=None, origin=(0, 0), channel_phase=(0, 1, 2, 3), **kwargs
):
    if values is None:
        values = torch.zeros(4, 4, 4)
    return RawSensorFrame(
        dataset_id="ds",
        episode_id="ep",
        camera_id="cam",
        frame_id=frame_id,
        values=values,
        channel_phase=channel_phase,
        cfa_pattern="RGGB",
        sensor_origin_xy=origin,
        validity=validity,
        **kwargs,
    )


def rgb3_frame(frame_id=0, values=None, validity=None, origin=(0, 0), **kwargs):
    if values is None:
        values = torch.zeros(3, 3, 3)
    return RawSensorFrame(
        dataset_id="ds",
        episode_id="ep",
        camera_id="cam",
        frame_id=frame_id,
        values=values,
        channel_phase=(),
        cfa_pattern="rgb3",
        sensor_origin_xy=origin,
        validity=validity,
        **kwargs,
    )


def affine_cfa(origin=(0, 0), a=3.0, b=5.0, d=1.0, shape=(4, 4)):
    """``values[k, i, j] = a * sensor_y + b * sensor_x + d`` on each phase lattice."""
    hp, wp = shape
    values = torch.zeros(4, hp, wp)
    for k in range(4):
        off_y, off_x = CFA_PHASE_OFFSETS[k]
        for i in range(hp):
            for j in range(wp):
                values[k, i, j] = (
                    a * (origin[0] + 2 * i + off_y) + b * (origin[1] + 2 * j + off_x) + d
                )
    return values


def resample_affine(taps, a=3.0, b=5.0, d=1.0):
    """Bilinear reconstruction of the affine field from the emitted taps."""
    return sum(weight * (a * key.sensor_y + b * key.sensor_x + d) for key, weight in taps)


def estimator(dy=0.0, dx=0.0, mode="cfa4", **kwargs):
    kwargs.setdefault("log_scale", (0.0, 0.0))
    kwargs.setdefault("weight", (0.0, 0.0))
    return CorrespondenceEstimator(mode=mode, model=ConstantFlow(dy, dx, **kwargs))


# --- geometry: identity, subpixel pull, units ----------------------------------------------


def test_identity_mapping_is_exact_and_valid_everywhere():
    est = estimator()
    mapping = est.estimate(cfa_frame(1), cfa_frame(0, values=affine_cfa()))
    assert tuple(mapping.valid.shape) == (4, 4, 4)
    assert bool(mapping.valid.all())
    for k in range(4):
        for i in range(4):
            for j in range(4):
                expected = (2 * i + CFA_PHASE_OFFSETS[k][0], 2 * j + CFA_PHASE_OFFSETS[k][1])
                assert mapping.source_xy[k, i, j].tolist() == list(map(float, expected))
                assert mapping.reference_sensor_xy(k, i, j) == expected
    assert resample_affine(mapping.taps_for_query(3, 1, 2)) == pytest.approx(3 * 3 + 5 * 5 + 1)
    # one unit weight on its own sample: identity is one tap per query
    assert set(mapping.tap_weights) == {1.0}
    assert len(mapping.tap_source_ids) == 4 * 4 * 4


# packed cells; cfa4 maps one packed cell to two sensor pixels, so 0.125/0.25/0.375
# packed are the required 0.25/0.5/0.75 sensor-pixel displacements
@pytest.mark.parametrize(
    "dy,dx",
    [
        (0.125, 0.0),
        (0.25, 0.0),
        (0.0, 0.375),
        (0.25, 0.0),
        (0.0, 0.5),
        (0.75, 0.75),
        (-0.5, 0.25),
        (1.0, -1.0),
        (-0.25, -0.75),
    ],
)
def test_subpixel_pull_reproduces_an_affine_source(dy, dx):
    shape = (6, 6)
    values = affine_cfa(shape=shape)
    est = estimator(dy, dx)
    mapping = est.estimate(cfa_frame(1, values=torch.zeros(4, *shape)), cfa_frame(0, values=values))
    checked = 0
    for k in (0, 3):
        for i in range(1, shape[0] - 1):
            for j in range(1, shape[1] - 1):
                pull_y = 2 * (i + dy) + CFA_PHASE_OFFSETS[k][0]
                pull_x = 2 * (j + dx) + CFA_PHASE_OFFSETS[k][1]
                assert mapping.source_xy[k, i, j].tolist() == pytest.approx([pull_y, pull_x])
                taps = mapping.taps_for_query(k, i, j)
                assert sum(weight for _, weight in taps) == pytest.approx(1.0)
                assert resample_affine(taps) == pytest.approx(3 * pull_y + 5 * pull_x + 1, abs=1e-9)
                assert all(key.channel_or_phase == k for key, _ in taps)
                checked += 1
    assert checked > 0


def test_packed_to_sensor_unit_conversion_differs_per_mode():
    mapping = estimator(1.5, -0.5).estimate(cfa_frame(1), cfa_frame(0))
    assert tuple(mapping.packed_flow.shape) == (2, 4, 4)
    assert float(mapping.packed_flow[0, 0, 0]) == pytest.approx(1.5)
    # one packed cell is two sensor pixels: a 1.5-cell pull moves three sensor pixels
    base = mapping.reference_sensor_xy(0, 2, 2)
    assert mapping.source_xy[0, 2, 2].tolist() == pytest.approx([base[0] + 3.0, base[1] - 1.0])

    rgb = estimator(1.5, -0.5, mode="rgb3").estimate(rgb3_frame(1), rgb3_frame(0))
    assert rgb.sensor_per_packed == 1
    base = rgb.reference_sensor_xy(0, 1, 1)
    assert rgb.source_xy[0, 1, 1].tolist() == pytest.approx([base[0] + 1.5, base[1] - 0.5])


def test_pull_direction_is_reference_to_source_for_both_signs():
    for dy, dx in ((2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0)):
        mapping = estimator(dy, dx).estimate(cfa_frame(7), cfa_frame(3))
        reference = mapping.reference_sensor_xy(0, 2, 2)
        pull = mapping.source_xy[0, 2, 2].tolist()
        assert pull[0] - reference[0] == pytest.approx(2 * dy)
        assert pull[1] - reference[1] == pytest.approx(2 * dx)
        assert mapping.source_frame_id == 3
        assert mapping.reference_frame_id == 7
        assert all(key.frame_id == 3 for key in mapping.tap_source_ids)


def test_each_phase_has_its_own_sensor_offset():
    mapping = estimator().estimate(cfa_frame(1), cfa_frame(0))
    assert mapping.channel_phase == (0, 1, 2, 3)
    assert [mapping.phase_offset(k) for k in range(4)] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [mapping.reference_sensor_xy(k, 1, 1) for k in range(4)] == [
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
    ]

    # a non-canonical packing with an odd origin still lands on its own lattice
    permuted = cfa_frame(1, channel_phase=(2, 3, 0, 1), origin=(1, 0))
    taps = estimator().estimate(permuted, permuted).taps_for_query(1, 0, 0)
    assert [t[0].channel_or_phase for t in taps] == [3]
    assert (taps[0][0].sensor_y, taps[0][0].sensor_x) == (1, 1)
    with pytest.raises(ValueError, match="declared origin and packing disagree"):
        cfa_frame(0, channel_phase=(2, 3, 0, 1), origin=(0, 0))


def test_rgb3_has_no_phase_and_three_co_located_channels():
    mapping = estimator(0.0, 0.0, mode="rgb3").estimate(rgb3_frame(1), rgb3_frame(0))
    assert mapping.channel_phase == ()
    assert mapping.phase_offset(2) == (0, 0)
    assert mapping.reference_sensor_xy(2, 1, 2) == (1, 2)
    subpixel = estimator(0.5, 0.5, mode="rgb3").estimate(rgb3_frame(1), rgb3_frame(0))
    taps = subpixel.taps_for_query(0, 1, 1)
    assert len(taps) == 4  # one co-located channel, four spatial corners
    assert {key.channel_or_phase for key, _ in taps} == {0}
    assert sorted({(key.sensor_y, key.sensor_x) for key, _ in taps}) == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]
    assert sorted({key.channel_or_phase for key in subpixel.tap_source_ids}) == [0, 1, 2]
    assert sum(weight for _, weight in taps) == pytest.approx(1.0)


# --- boundary, validity and tap bookkeeping ------------------------------------------------


def test_boundary_taps_are_invalid_and_never_wrap():
    mapping = estimator(0.0, -1.0).estimate(cfa_frame(1), cfa_frame(0))
    assert not bool(mapping.valid[:, :, 0].any())
    assert bool(mapping.valid[:, :, 1:].all())
    for key in mapping.tap_source_ids:
        assert 0 <= key.sensor_x < 8 and 0 <= key.sensor_y < 8
        phase = 2 * (key.sensor_y % 2) + (key.sensor_x % 2)
        assert key.channel_or_phase == phase

    # a displacement that pushes every query out of frame rejects the whole map
    rejected = estimator(-100.0, 0.0).estimate(cfa_frame(1), cfa_frame(0))
    assert rejected.valid is False
    assert rejected.tap_source_ids == ()


def test_invalid_or_missing_source_samples_never_enter_taps():
    validity = torch.ones(4, 4, 4, dtype=torch.bool)
    validity[2, 1, 1] = False
    mapping = estimator().estimate(cfa_frame(1, validity=validity), cfa_frame(0, validity=validity))
    assert not bool(mapping.valid[2, 1, 1])
    assert int(mapping.valid.sum()) == 4 * 4 * 4 - 1
    assert all(
        not (key.channel_or_phase == 2 and key.sensor_y == 3 and key.sensor_x == 3)
        for key in mapping.tap_source_ids
    )
    # the untouched neighbours of the invalid sample stay valid and sampled
    sampled = {(key.channel_or_phase, key.sensor_y, key.sensor_x) for key in mapping.tap_source_ids}
    assert (2, 3, 2) not in sampled  # the rejected sample itself
    assert (2, 3, 4) in sampled


def test_emitted_taps_match_the_documented_flat_layout():
    mapping = estimator(0.25, -0.5).estimate(cfa_frame(1), cfa_frame(0))
    hp, wp = mapping.packed_shape
    expected_ids = []
    expected_weights = []
    for channel in range(4):
        for row in range(hp):
            for col in range(wp):
                if not bool(mapping.valid[channel, row, col]):
                    continue
                for key, weight in mapping.taps_for_query(channel, row, col):
                    expected_ids.append(key)
                    expected_weights.append(pytest.approx(weight))
    assert list(mapping.tap_source_ids) == expected_ids
    assert list(mapping.tap_weights) == expected_weights


# --- causality, frames and normalization ---------------------------------------------------


def test_estimate_is_memoryless_and_never_reaches_a_third_frame():
    parameters = inspect.signature(CorrespondenceEstimator.estimate).parameters
    assert list(parameters) == ["self", "reference", "source", "iters", "taps"]
    assert parameters["reference"].default is inspect.Parameter.empty
    assert parameters["source"].default is inspect.Parameter.empty
    assert CorrespondenceEstimator.estimate_bidirectional is not None

    est = estimator(0.5, 0.5)
    newer, older = cfa_frame(9), cfa_frame(4)
    forward = est.estimate(newer, older)
    before = [parameter.detach().clone() for parameter in est.model_parameters()]
    backward = est.estimate(older, newer)
    assert forward.source_frame_id == 4 and backward.source_frame_id == 9
    assert torch.equal(forward.packed_flow, backward.packed_flow)
    again = est.estimate(newer, older)
    assert torch.equal(forward.source_xy, again.source_xy)
    assert forward.tap_source_ids == again.tap_source_ids
    assert torch.equal(forward.valid, again.valid)
    assert all(
        torch.equal(after, reference)
        for after, reference in zip(est.model_parameters(), before, strict=True)
    )


def test_bidirectional_consistency_runs_the_same_pair_twice():
    consistent = estimator(0.0, 0.0).estimate_bidirectional(cfa_frame(1), cfa_frame(0))
    assert float(consistent.cycle_error.max()) == pytest.approx(0.0)
    assert bool(consistent.consistent.all())
    assert consistent.forward.source_frame_id == 0
    assert consistent.backward.source_frame_id == 1

    small = estimator(0.25, 0.0).estimate_bidirectional(cfa_frame(1), cfa_frame(0), tolerance=1.0)
    assert float(small.cycle_error.max()) == pytest.approx(0.5)
    assert bool(small.consistent[:-1].all())  # interior cells round-trip inside the frame
    assert not bool(small.consistent[-1].any())  # the last row maps past the sampled frame

    inconsistent = estimator(1.0, 0.0).estimate_bidirectional(cfa_frame(1), cfa_frame(0))
    assert float(inconsistent.cycle_error.max()) == pytest.approx(2.0)
    assert not bool(inconsistent.consistent.any())
    with pytest.raises(ValueError, match="tolerance"):
        estimator().estimate_bidirectional(cfa_frame(1), cfa_frame(0), tolerance=0.0)


def test_normalization_is_a_declared_pure_transform():
    interval = RawNormalization.fixed_interval(0.0, 16.0, source="p2-fixture-interval")
    values = torch.tensor([[[0.0, 4.0], [8.0, 16.0]]])
    assert torch.equal(interval.apply(values), (values - 8.0) / 8.0)
    assert interval.apply(interval.apply(values)).shape == values.shape
    calibration = RawNormalization.affine_calibration(64.0, 960.0, source="bayer-black-white-v1")
    assert calibration.calibration_version == "unversioned"
    assert calibration.apply(torch.tensor([512.0])).tolist() == pytest.approx([0.0])
    with pytest.raises(ValueError, match="half_range"):
        RawNormalization(method="fixed_interval", center=0.0, half_range=0.0, source="x")
    with pytest.raises(ValueError, match="Unknown RawNormalization.method"):
        RawNormalization(method="per_image_minmax", center=0.0, half_range=1.0, source="x")


def test_normalization_reaches_the_network_unchanged_and_is_per_calibration():
    seen = []

    class Recorder(ConstantFlow):
        def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
            seen.append(image1.detach().clone())
            return super().forward(
                image1, image2, iters=iters, flow_gt=flow_gt, test_mode=test_mode
            )

    spec = RawNormalization.fixed_interval(0.0, 8.0, source="cal-a", calibration_version="cal-a")
    est = CorrespondenceEstimator(
        mode="cfa4", model=Recorder(0.0, 0.0), normalization={"cal-a": spec}
    )
    frame = cfa_frame(1, values=torch.full((4, 4, 4), 8.0), time=3.0, calibration_version="cal-a")
    mapping = est.estimate(frame, frame)
    assert torch.equal(seen[0][0], (torch.full((4, 4, 4), 8.0) - 4.0) / 4.0)
    assert mapping.normalization_source == "cal-a"

    with pytest.raises(KeyError, match="unknown calibration"):
        est.estimate(frame, cfa_frame(0, values=torch.zeros(4, 4, 4), calibration_version="cal-b"))

    raw = estimator().estimate(frame, frame)
    assert raw.normalization_source == "caller_pre_normalized"


def test_frame_identity_and_packing_validation():
    with pytest.raises(ValueError, match="four packed phases"):
        RawSensorFrame("d", "e", "c", 0, torch.zeros(3, 4, 4), (0, 1, 2, 3), "RGGB", (0, 0))
    with pytest.raises(ValueError, match="permutation"):
        cfa_frame(0, channel_phase=(0, 1, 2, 2))
    with pytest.raises(ValueError, match="no CFA phase"):
        RawSensorFrame("d", "e", "c", 0, torch.zeros(3, 3, 3), (0, 1, 2), "rgb3", (0, 0))
    with pytest.raises(ValueError, match="finite"):
        cfa_frame(0, values=torch.full((4, 4, 4), float("nan")))
    with pytest.raises(ValueError, match="same"):
        cfa_frame(0, validity=torch.ones(3, 4, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="Unknown cfa_pattern"):
        RawSensorFrame("d", "e", "c", 0, torch.zeros(4, 4, 4), (0, 1, 2, 3), "XYZ", (0, 0))


def test_pairs_with_mismatched_frames_are_refused():
    est = estimator()
    with pytest.raises(ValueError, match="cross-camera"):
        est.estimate(
            cfa_frame(1),
            RawSensorFrame(
                "ds", "ep", "other", 0, torch.zeros(4, 4, 4), (0, 1, 2, 3), "RGGB", (0, 0)
            ),
        )
    with pytest.raises(ValueError, match="CFA patterns differ"):
        est.estimate(
            cfa_frame(1),
            RawSensorFrame(
                "ds", "ep", "cam", 0, torch.zeros(4, 4, 4), (0, 1, 2, 3), "BGGR", (0, 0)
            ),
        )
    with pytest.raises(ValueError, match="share one packed grid"):
        est.estimate(cfa_frame(1), cfa_frame(0, values=torch.zeros(4, 4, 6)))
    with pytest.raises(ValueError, match="stem"):
        est.estimate(cfa_frame(1), rgb3_frame(0))
    with pytest.raises(ValueError, match="channel orders"):
        est.estimate(cfa_frame(1), cfa_frame(0, channel_phase=(1, 0, 2, 3), origin=(0, 1)))


# --- proxy scores and the calibration protocol ---------------------------------------------


def test_risk_fields_stay_labelled_proxies_until_a_table_calibrates_them():
    mapping = estimator(log_scale=(0.0, 0.0), weight=(0.0, 0.0)).estimate(
        cfa_frame(1), cfa_frame(0)
    )
    assert mapping.risk_kind == "proxy"
    assert mapping.calibrated_risk is None and mapping.coverage is None
    assert tuple(mapping.visibility.shape) == (4, 4)
    assert mapping.visibility.tolist() == [[0.5] * 4] * 4
    assert mapping.flow_risk.tolist() == [[2.0] * 4] * 4  # cfa4: two sensor px

    class Table:
        version = "motion-cal-v1"

        def map_proxy(self, *, flow_risk, visibility):
            return flow_risk * 0.0 + 0.25, visibility * 0.0 + 0.75

    calibrated = CorrespondenceEstimator(
        mode="cfa4", model=ConstantFlow(0.0, 0.0), risk_calibration=Table()
    ).estimate(cfa_frame(1), cfa_frame(0))
    assert calibrated.risk_kind == "calibrated:motion-cal-v1"
    assert bool(torch.all(calibrated.calibrated_risk == 0.25))
    assert bool(torch.all(calibrated.coverage == 0.75))

    class BadTable(Table):
        def map_proxy(self, *, flow_risk, visibility):
            return flow_risk[:1], visibility

    with pytest.raises(ValueError, match="packed grid"):
        CorrespondenceEstimator(
            mode="cfa4", model=ConstantFlow(0.0, 0.0), risk_calibration=BadTable()
        ).estimate(cfa_frame(1), cfa_frame(0))

    class NoInfo(nn.Module):
        def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
            return {"final": torch.zeros(image1.shape[0], 2, *image1.shape[2:])}

    with pytest.raises(ValueError, match="no proxy scores"):
        CorrespondenceEstimator(mode="cfa4", model=NoInfo(), risk_calibration=Table()).estimate(
            cfa_frame(1), cfa_frame(0)
        )
    uncalibrated = CorrespondenceEstimator(mode="cfa4", model=NoInfo()).estimate(
        cfa_frame(1), cfa_frame(0)
    )
    assert uncalibrated.risk_kind == "unavailable"
    assert uncalibrated.visibility is None and uncalibrated.flow_risk is None


def test_proxy_formula_matches_the_author_mixture_head():
    weight = (2.0, -2.0)
    log_scale = (2.0, -3.0)  # second component clamps to var_min=0 in the author loss
    mapping = estimator(log_scale=log_scale, weight=weight).estimate(cfa_frame(1), cfa_frame(0))
    mixture = torch.softmax(torch.tensor(weight), dim=0)
    expected_packed = float(
        mixture[0] * torch.exp(torch.tensor(log_scale[0]))
        + mixture[1] * torch.exp(torch.tensor(0.0))
    )
    assert float(mapping.visibility[0, 0]) == pytest.approx(float(mixture[1]))
    assert float(mapping.flow_risk[0, 0]) == pytest.approx(2.0 * expected_packed)


# --- vendored source: declared changes and executable evidence ------------------------------

#: AST fields that exist on only some CPython versions.  ``type_params`` (PEP 695)
#: is present on FunctionDef/AsyncFunctionDef/ClassDef from 3.12 on and absent on
#: 3.11, so ``ast.dump`` alone is version-dependent; the canonical serializer below
#: never reads it.  Everything else is emitted, including empty lists, so CPython
#: 3.13's ``ast.dump(show_empty=False)`` default cannot move a digest either.
_VOLATILE_AST_FIELDS = frozenset({"type_params"})


class _RemoveDocstrings(ast.NodeTransformer):
    """Drop docstring statements so prose is never mistaken for executable evidence."""

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return node


def _canonical_ast(value):
    """Version-independent AST serialization: every field of every node, in order."""
    if isinstance(value, ast.AST):
        fields = [
            (name, _canonical_ast(field))
            for name, field in ast.iter_fields(value)
            if name not in _VOLATILE_AST_FIELDS
        ]
        return (type(value).__name__, tuple(fields))
    if isinstance(value, list):
        return tuple(_canonical_ast(item) for item in value)
    return value


def executable_ast_dump(source: str) -> str:
    """Canonical serialization of a module without its docstrings."""
    return repr(_canonical_ast(_RemoveDocstrings().visit(ast.parse(source))))


def executable_ast_digest(source: str) -> str:
    """Digest of the top-level executable definitions of one module source."""
    tree = _RemoveDocstrings().visit(ast.parse(source))
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    payload = "\n".join(repr(_canonical_ast(node)) for node in definitions)
    return hashlib.sha256(payload.encode()).hexdigest()


def executable_ast_sha256(path: Path) -> str:
    return executable_ast_digest(path.read_text())


def test_executable_ast_digest_detects_real_changes_and_ignores_prose():
    source = (VENDOR / "layer.py").read_text()
    baseline = executable_ast_digest(source)

    # a comment or a rewritten docstring is prose, not executable evidence
    assert executable_ast_digest(source + "\n# a new trailing comment\n") == baseline
    old_doc = '"""1x1 convolution without padding"""'
    assert old_doc in source
    rewritten = source.replace(old_doc, '"""1x1 convolution, reworded prose"""')
    assert rewritten != source and executable_ast_digest(rewritten) == baseline

    # a changed function body must move the digest
    old_body = "        return grad_x.clamp(min=-0.01, max=0.01)"
    assert old_body in source
    mutated_body = source.replace(old_body, "        return grad_x.clamp(min=-0.02, max=0.01)")
    assert mutated_body != source
    assert executable_ast_digest(mutated_body) != baseline

    # so must adding a method to an existing class (round-trip checked first)
    tree = ast.parse(source)
    assert executable_ast_digest(ast.unparse(tree)) == baseline
    target = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LayerNorm"
    )
    target.body.append(ast.parse("def extra(self):\n    return 1\n").body[0])
    assert executable_ast_digest(ast.unparse(tree)) != baseline


def test_executable_ast_digest_does_not_read_version_dependent_type_params():
    # 3.11 trees have no type_params field at all; 3.12 trees carry it.  Injecting
    # the field on any interpreter proves the serializer ignores it either way.
    tree = ast.parse("def f():\n    return 1\n")
    node = tree.body[0]
    node._fields = tuple(node._fields) + ("type_params",)
    node.type_params = []
    without_params = repr(_canonical_ast(node))
    node.type_params = [ast.Name(id="T", ctx=ast.Load())]
    with_params = repr(_canonical_ast(node))
    assert without_params == with_params
    assert "type_params" not in with_params

    # and a whole real module digests identically with the field deleted everywhere
    class DropTypeParams(ast.NodeTransformer):
        def generic_visit(self, node):
            if "type_params" in getattr(node, "_fields", ()):
                node._fields = tuple(f for f in node._fields if f != "type_params")
            return super().generic_visit(node)

    source = (VENDOR / "extractor.py").read_text()
    stripped = ast.unparse(DropTypeParams().visit(ast.parse(source)))
    assert executable_ast_digest(stripped) == executable_ast_digest(source)


def test_vendored_hashes_match_the_recorded_provenance():
    provenance = json.loads((VENDOR / "provenance.json").read_text())
    assert provenance["commit"] == "9137517ba24e628442aec097d3afe71d03503b75"
    assert set(provenance["files"]) == {
        "raft.py",
        "extractor.py",
        "corr.py",
        "update.py",
        "layer.py",
        "utils/utils.py",
        "LICENSE",
    }
    for name, entry in provenance["files"].items():
        path = VENDOR / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["local_sha256"], name
        if path.suffix == ".py":
            assert executable_ast_sha256(path) == entry["executable_ast_sha256"], name
    assert provenance["files"]["LICENSE"]["verbatim_bytes"] is True

    cache = {
        "raft.py": UPSTREAM_CACHE / "core/raft.py",
        "extractor.py": UPSTREAM_CACHE / "core/extractor.py",
        "corr.py": UPSTREAM_CACHE / "core/corr.py",
        "update.py": UPSTREAM_CACHE / "core/update.py",
        "LICENSE": UPSTREAM_CACHE / "LICENSE",
    }
    if not all(path.is_file() for path in cache.values()):
        pytest.skip("pinned upstream cache is absent in this checkout")
    for name, path in cache.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == provenance["files"][name]["upstream_sha256"]
        ), name
    # files whose only declared change is formatting keep the author's executable AST
    for name in ("corr.py", "update.py", "layer.py"):
        assert (
            provenance["files"][name]["executable_ast_sha256"]
            == provenance["files"][name]["upstream_executable_ast_sha256"]
        ), name


def test_imagenet_download_and_huggingface_are_gone_and_cannot_be_reenabled(monkeypatch):
    forbidden = (
        "torchvision",
        "huggingface_hub",
        "PyTorchModelHubMixin",
        "IMAGENET1K",
        "ResNet18_Weights",
        "ResNet34_Weights",
    )
    for name in ("extractor.py", "raft.py"):
        dumped = executable_ast_dump((VENDOR / name).read_text())
        for token in forbidden:
            assert token not in dumped, (name, token)

    resnet_parameters = inspect.signature(ResNetFPN.__init__).parameters
    assert "init_weight" not in resnet_parameters and "pretrain" not in resnet_parameters
    assert "depth" in resnet_parameters and "input_dim" in resnet_parameters
    raft_parameters = inspect.signature(RAFT.__init__).parameters
    assert "depth" in raft_parameters and "input_dim" in raft_parameters

    def forbidden(*args, **kwargs):
        raise AssertionError("the vendored model tried to download weights")

    monkeypatch.setattr("torch.hub.download_url_to_file", forbidden)
    monkeypatch.setattr("torch.hub.load_state_dict_from_url", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    est = CorrespondenceEstimator(mode="cfa4", depth="resnet18", iters=2)
    assert est.depth == "resnet18"
    assert est.pretrained_report is None
    with pytest.raises(TypeError):
        ResNetFPN(est.model.args, init_weight=True)


def test_vendored_forward_keeps_the_supplied_values_and_removes_255():
    est = CorrespondenceEstimator(mode="cfa4", depth="resnet18", iters=2)
    model = est.model
    code = "\n".join(
        line
        for line in inspect.getsource(RAFT.forward).splitlines()
        if not line.strip().startswith("#")
    )
    assert "255" not in code
    reference = cfa_frame(1, values=torch.rand(4, 128, 128))
    source = cfa_frame(0, values=torch.rand(4, 128, 128))
    image1 = est.preprocess(reference)
    image2 = est.preprocess(source)
    padder = InputPadder(image1.shape)
    padded1, padded2 = padder.pad(image1, image2)

    seen: list[torch.Tensor] = []

    def capture(module, inputs):
        seen.append(inputs[0].detach().clone())

    handles = [model.cnet.conv1.register_forward_pre_hook(capture)]
    handles.append(model.fnet.conv1.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            model(image1, image2, iters=2, test_mode=True)
    finally:
        for handle in handles:
            handle.remove()
    assert len(seen) == 3
    assert torch.equal(seen[0], torch.cat([padded1, padded2], dim=1))
    assert torch.equal(seen[1], padded1)
    assert torch.equal(seen[2], padded2)


def test_pretrained_files_are_an_independent_explicit_input(tmp_path):
    est = CorrespondenceEstimator(mode="cfa4", depth="resnet18", iters=2)
    assert est.pretrained_report is None
    checkpoint = tmp_path / "motion.pth"
    torch.save(est.model.state_dict(), checkpoint)
    loaded = CorrespondenceEstimator(
        mode="cfa4",
        depth="resnet18",
        iters=2,
        pretrained_checkpoint=checkpoint,
    )
    assert loaded.depth == "resnet18"
    report = loaded.pretrained_report
    assert report is not None and report["loaded_tensors"] > 0
    assert loaded.pretrained_report["path"] == str(checkpoint)

    with pytest.raises(FileNotFoundError):
        CorrespondenceEstimator(
            mode="cfa4", depth="resnet18", pretrained_checkpoint=tmp_path / "missing.pth"
        )
    garbage = tmp_path / "garbage.pth"
    garbage.write_bytes(b"not a checkpoint")
    with pytest.raises(torch.serialization.pickle.UnpicklingError):
        CorrespondenceEstimator(mode="cfa4", depth="resnet18", pretrained_checkpoint=garbage)


def test_declared_config_and_stem_choices_are_enforced():
    cfa_config = CorrespondenceConfig(
        mapping_kind="sea_raft", mapping_version="n01-p2-v1", input_channels=8
    )
    est = CorrespondenceEstimator(mode="cfa4", config=cfa_config, model=ConstantFlow(0.0, 0.0))
    assert est.mapping_version == "n01-p2-v1"
    assert est.estimate(cfa_frame(1), cfa_frame(0)).mapping_version == "n01-p2-v1"
    CorrespondenceEstimator(
        mode="rgb3",
        config=CorrespondenceConfig(mapping_kind="sea_raft", input_channels=3),
        model=ConstantFlow(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="input_channels"):
        CorrespondenceEstimator(
            mode="cfa4",
            config=CorrespondenceConfig(mapping_kind="sea_raft", input_channels=5),
            model=ConstantFlow(0.0, 0.0),
        )
    with pytest.raises(ValueError, match="mapping_kind"):
        CorrespondenceEstimator(
            mode="cfa4",
            config=CorrespondenceConfig(mapping_kind="identity"),
            model=ConstantFlow(0.0, 0.0),
        )
    with pytest.raises(ValueError, match="Unknown mode"):
        CorrespondenceEstimator(mode="cfa2", model=ConstantFlow(0.0, 0.0))
    with pytest.raises(ValueError, match="non-negative int"):
        CorrespondenceEstimator(mode="cfa4", model=ConstantFlow(0.0, 0.0), iters=-1)


# --- the real vendored network on CPU -------------------------------------------------------


@pytest.fixture(scope="module")
def real_estimator():
    return CorrespondenceEstimator(mode="cfa4", depth="resnet18", iters=2)


def test_real_vendored_model_runs_end_to_end_on_cpu(real_estimator):
    values = torch.rand(4, MIN_SIDE, MIN_SIDE)
    reference = cfa_frame(1, values=values)
    source = cfa_frame(0, values=values.flip(-1))
    mapping = real_estimator.estimate(reference, source)
    assert tuple(mapping.packed_flow.shape) == (2, MIN_SIDE, MIN_SIDE)
    assert tuple(mapping.source_xy.shape) == (4, MIN_SIDE, MIN_SIDE, 2)
    assert torch.isfinite(mapping.source_xy).all()
    assert mapping.risk_kind == "proxy"
    assert torch.isfinite(mapping.flow_risk).all() and torch.isfinite(mapping.visibility).all()
    assert bool((mapping.visibility >= 0).all() and (mapping.visibility <= 1).all())
    assert 0.5 < float(mapping.valid.float().mean()) <= 1.0
    assert len(mapping.tap_source_ids) == len(mapping.tap_weights) > 0
    assert real_estimator.model.training is False

    rgb = CorrespondenceEstimator(mode="rgb3", depth="resnet18", iters=2)
    rgb_frame = rgb3_frame(1, values=torch.rand(3, MIN_SIDE, MIN_SIDE))
    rgb_map = rgb.estimate(rgb_frame, rgb_frame)
    assert tuple(rgb_map.source_xy.shape) == (3, MIN_SIDE, MIN_SIDE, 2)
    assert rgb_map.sensor_per_packed == 1


def test_owned_model_refuses_a_window_below_the_correlation_pyramid(real_estimator):
    small = cfa_frame(1, values=torch.rand(4, MIN_SIDE // 2, MIN_SIDE))
    with pytest.raises(ValueError, match="correlation pyramid"):
        real_estimator.estimate(small, small)
