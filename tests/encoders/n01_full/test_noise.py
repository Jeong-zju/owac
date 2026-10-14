"""Numeric and behavioural properties of the P2 first-version sensor noise.

Every numeric assertion has an independent oracle: the variance law is checked
against a NumPy evaluation of ``shot * max(mu - black, 0) + read_var``, the
interpolation variance against an explicit ``a @ Sigma @ a`` quadratic form, the
block draw against its declared standard deviation over many seeds, and the
version binding against ``fit_structure``'s committed-domain check.  Two
counterexamples must keep failing: interpolating a variance map instead of the
joint covariance, and treating the first-order flow propagation as exact across
a step edge.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01_full.contracts import SourceKey
from owac.frameworks.n01_full.domains import make_domain
from owac.frameworks.n01_full.ledger import MeasurementLedger
from owac.frameworks.n01_full.statistics import fit_structure
from owac.raw.n01_noise import (
    CROSS_PHASE_UNKNOWN_REFUSAL,
    FITTING_SPLITS,
    MAX_DENSE_SOURCES,
    NOISE_FORMULA,
    NOISE_VERSION_PREFIX,
    NoiseCalibration,
    NoiseProvenance,
    SharedBlackLevel,
    ShotReadParameters,
    UnsupportedNoiseStructureError,
    add_sensor_noise,
    apply_exposure_gain,
    build_noise_model,
    draw_block_error,
    flow_first_order_variance,
    interpolated_variance,
    load_noise_calibration,
    shot_read_variance,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/encoders/n01_full/p2_motion_noise.json"

SHOT = 0.14
READ_VAR = 4.0
BLACK = 64.0
WHITE = 4095.0
BLOCK_VAR = 2.25


def provenance(**overrides) -> NoiseProvenance:
    payload = {
        "kind": "explicit_synthesis",
        "split": "synthesis",
        "source": "tests/encoders/n01_full/test_noise.py",
        "method": "declared for this test",
    }
    payload.update(overrides)
    return NoiseProvenance(**payload)


def calibration(*, block_variance=BLOCK_VAR, block_upper=None, units="raw_dn_12bit"):
    return NoiseCalibration(
        calibration_version="unit-test-v1",
        units=units,
        shot_read=ShotReadParameters(
            shot=SHOT, read_var=READ_VAR, units=units, provenance=provenance()
        ),
        black_level=SharedBlackLevel(
            kind="per_frame",
            id_template="{dataset_id}/{camera_id}/frame_{frame_id}",
            variance=block_variance,
            upper_bound=block_upper,
            provenance=provenance(),
        ),
        black_level_dn=BLACK,
        white_level_dn=WHITE,
    )


def key(frame_id: int, x: int = 0, y: int = 0, phase: int = 0) -> SourceKey:
    return SourceKey("n01-test", "ep0", "cam0", frame_id, phase, y, x)


# ---------------------------------------------------------------------------
# The variance law and the exposure transform
# ---------------------------------------------------------------------------


def test_variance_law_matches_independent_numpy_formula():
    means = np.array([-500.0, 0.0, BLACK - 1.0, BLACK, BLACK + 1.0, 900.0, WHITE])
    expected = np.asarray(SHOT * np.maximum(means - BLACK, 0.0) + READ_VAR)
    actual = shot_read_variance(
        torch.tensor(means, dtype=torch.float64), shot=SHOT, read_var=READ_VAR, black_level=BLACK
    )
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=0.0)
    assert float(actual[2]) == pytest.approx(READ_VAR)
    assert float(actual[4]) == pytest.approx(SHOT + READ_VAR)


def test_variance_law_holds_on_drawn_samples():
    """Empirical variance of many draws matches the declared law within a stated band."""
    calibration_ = calibration()
    mean = 900.0
    draws = 40_000
    generator = torch.Generator().manual_seed(20261014)
    samples = torch.stack(
        [
            add_sensor_noise(
                torch.full((1,), mean, dtype=torch.float64),
                calibration=calibration_,
                generator=generator,
                block_error=0.0,
            )
            for _ in range(draws)
        ]
    )
    empirical = float(samples.var(unbiased=True))
    expected = SHOT * (mean - BLACK) + READ_VAR
    assert expected == pytest.approx(0.14 * 836.0 + 4.0)
    # 40k draws: the 99.9% band of a chi-square with 39,999 dof is about +-2.1% of the mean.
    assert abs(empirical - expected) / expected < 0.05


def test_block_error_matches_its_declared_sigma_and_is_reproducible():
    calibration_ = calibration()
    draws = [
        draw_block_error("blk0", calibration_, torch.Generator().manual_seed(seed))
        for seed in range(4000)
    ]
    assert float(np.mean(draws)) == pytest.approx(0.0, abs=3.0 * np.sqrt(BLOCK_VAR / 4000))
    assert float(np.var(draws)) == pytest.approx(BLOCK_VAR, rel=0.1)
    again = draw_block_error("blk0", calibration_, torch.Generator().manual_seed(1))
    assert again == draws[1]
    unaudited = calibration(block_variance=None, block_upper=9.0)
    with pytest.raises(UnsupportedNoiseStructureError):
        draw_block_error("blk0", unaudited, torch.Generator().manual_seed(1))


def test_exposure_gain_scales_value_and_variance_together():
    values = torch.tensor([BLACK, BLACK + 100.0, BLACK + 900.0], dtype=torch.float64)
    variance = shot_read_variance(values, shot=SHOT, read_var=READ_VAR, black_level=BLACK)
    gain = 1.35
    scaled_values, scaled_variance = apply_exposure_gain(
        values, variance, gain=gain, black_level=BLACK
    )
    oracle_values = BLACK + gain * (np.array([BLACK, BLACK + 100.0, BLACK + 900.0]) - BLACK)
    oracle_variance = gain**2 * np.array(
        [READ_VAR, SHOT * 100.0 + READ_VAR, SHOT * 900.0 + READ_VAR]
    )
    np.testing.assert_allclose(scaled_values.numpy(), oracle_values, rtol=1e-12)
    np.testing.assert_allclose(scaled_variance.numpy(), oracle_variance, rtol=1e-12)
    # The rescaled parameter set reproduces the same law in the new unit.
    rescaled = calibration().shot_read.scaled(gain)
    np.testing.assert_allclose(
        rescaled.variance(scaled_values, black_level=BLACK).numpy(),
        scaled_variance.numpy(),
        rtol=1e-12,
    )
    assert rescaled.shot == pytest.approx(SHOT * gain)
    assert rescaled.read_var == pytest.approx(READ_VAR * gain**2)
    assert "gain" in rescaled.provenance.method
    # Counterexample: variance scaled by the gain itself, not its square, is wrong.
    wrong = gain * variance
    assert not np.allclose(wrong.numpy(), scaled_variance.numpy())
    # Counterexample: scaling the value without moving the black level is wrong.
    naive_value = gain * values
    assert not np.allclose(naive_value.numpy()[0], scaled_values.numpy()[0])


# ---------------------------------------------------------------------------
# Interpolation variance
# ---------------------------------------------------------------------------


def real_four_taps(fx: float, fy: float) -> torch.Tensor:
    """The four bilinear tap weights of one fractional packed displacement."""
    return torch.tensor(
        [(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx], dtype=torch.float64
    )


def test_interpolation_variance_is_the_tap_quadratic_form():
    keys = [key(0, x) for x in range(4)]
    means = [BLACK + 100.0, BLACK + 200.0, BLACK + 300.0, BLACK + 400.0]
    model = build_noise_model(keys, means, calibration())
    assert model.kind == "dense"
    sigma = model.covariance_for(keys, dtype=torch.float64)
    tap_weights = real_four_taps(0.25, 0.75)
    actual = interpolated_variance(tap_weights, sigma)
    weights = np.array([w for w in tap_weights.tolist()])
    oracle = float(weights @ sigma.numpy() @ weights)
    assert float(actual) == pytest.approx(oracle, rel=1e-12)
    assert float(tap_weights.sum()) == pytest.approx(1.0)


def test_interpolated_variance_is_not_interpolating_a_variance_map():
    """Counterexamples: both naive readings of the same weights must disagree.

    With one shared block and equal per-sample variances ``c`` the exact value is
    ``c * sum(w**2) + block * (sum(w))**2``.  Interpolating the variance map
    (``sum(w * sigma_ii)``) and assuming independent taps (``sum(w**2 * sigma_ii)``)
    both get a different number, and the independent reading always understates
    the variance because the shared block is positively correlated.
    """
    keys = [key(0, x) for x in range(4)]
    means = [BLACK + 50.0] * 4
    c = SHOT * 50.0 + READ_VAR
    model = build_noise_model(keys, means, calibration())
    sigma = model.covariance_for(keys, dtype=torch.float64)
    tap_weights = real_four_taps(0.5, 0.25)
    weights = np.array(tap_weights.tolist())
    actual = float(interpolated_variance(tap_weights, sigma))
    oracle = float(c * np.sum(weights**2) + BLOCK_VAR * np.sum(weights) ** 2)
    assert actual == pytest.approx(oracle, rel=1e-12)
    naive_map = float((tap_weights * torch.diagonal(sigma)).sum())
    independent = float(c * np.sum(weights**2))
    assert naive_map != pytest.approx(actual)
    assert independent != pytest.approx(actual)
    assert independent < actual
    assert actual - independent == pytest.approx(BLOCK_VAR * float(np.sum(weights)) ** 2)
    # Without a shared block the independent reading is the exact one again.
    independent_model = build_noise_model(
        keys, means, calibration(), block_keys=(None, None, None, None)
    )
    diagonal_sigma = independent_model.covariance_for(keys, dtype=torch.float64)
    assert float(interpolated_variance(tap_weights, diagonal_sigma)) == pytest.approx(independent)


def test_interpolation_variance_rejects_mismatched_taps():
    keys = [key(0, x) for x in range(4)]
    model = build_noise_model(keys, [BLACK + 10.0] * 4, calibration())
    sigma = model.covariance_for(keys, dtype=torch.float64)
    with pytest.raises(ValueError, match="real taps"):
        interpolated_variance(torch.tensor([0.5, 0.5]), sigma)
    with pytest.raises(ValueError, match="square"):
        interpolated_variance(torch.tensor([0.5, 0.5]), torch.ones(2, 3))


# ---------------------------------------------------------------------------
# NoiseModel kinds, cross-phase handling and version binding
# ---------------------------------------------------------------------------


def test_shared_block_builds_dense_cross_phase_covariance():
    keys = [key(0, 0, 0, 0), key(0, 1, 0, 1), key(1, 0, 0, 2), key(1, 1, 0, 3)]
    means = [BLACK + 10.0, BLACK + 20.0, BLACK + 30.0, BLACK + 40.0]
    model = build_noise_model(keys, means, calibration())
    assert model.kind == "dense"
    assert model.version == f"{NOISE_VERSION_PREFIX}unit-test-v1"
    sigma = model.covariance_for(keys, dtype=torch.float64)
    for row in range(4):
        for column in range(4):
            expected = (SHOT * (means[row] - BLACK) + READ_VAR) * (row == column)
            expected += BLOCK_VAR if keys[row].frame_id == keys[column].frame_id else 0.0
            assert float(sigma[row, column]) == pytest.approx(expected, rel=1e-12)
    # The cross-phase block term is present, not dropped.
    assert float(sigma[0, 1]) == pytest.approx(BLOCK_VAR)
    assert float(sigma[0, 2]) == 0.0
    # Declaring the samples independent removes the block and yields the diagonal kind.
    diagonal = build_noise_model(keys, means, calibration(), block_keys=(None, None, None, None))
    assert diagonal.kind == "diagonal"
    assert diagonal.variances[0] == pytest.approx(SHOT * 10.0 + READ_VAR)


def test_unaudited_shared_block_is_refused_not_dropped():
    unaudited = calibration(block_variance=None, block_upper=9.0)
    keys = [key(0, 0), key(0, 1)]
    means = [BLACK + 10.0, BLACK + 20.0]
    with pytest.raises(UnsupportedNoiseStructureError) as error:
        build_noise_model(keys, means, unaudited)
    assert CROSS_PHASE_UNKNOWN_REFUSAL in str(error.value)
    assert "silently" in str(error.value) or "dropping" in str(error.value)
    proxy = build_noise_model(keys, means, unaudited, correlation="unknown")
    assert proxy.kind == "unknown_correlation"
    assert proxy.variances[0] == pytest.approx(SHOT * 10.0 + READ_VAR + 9.0)
    with pytest.raises(ValueError, match="unknown correlation"):
        build_noise_model(keys, means, unaudited, correlation="maybe")


def repro_calibration(*, block_variance=0.25, upper=None):
    """The D2 reproduction calibration: shot 0.5, read 2.0, block 0.25, black 100."""
    return NoiseCalibration(
        calibration_version="d2-repro-v1",
        units="raw_dn_12bit",
        shot_read=ShotReadParameters(
            shot=0.5, read_var=2.0, units="raw_dn_12bit", provenance=provenance()
        ),
        black_level=SharedBlackLevel(
            kind="per_frame",
            id_template="{dataset_id}/{camera_id}/frame_{frame_id}",
            variance=block_variance,
            upper_bound=upper,
            provenance=provenance(),
        ),
        black_level_dn=100.0,
        white_level_dn=4095.0,
    )


def test_single_sample_block_keeps_the_audited_black_level_variance():
    """D2 regression: a one-sample block must not lose ``sigma_b**2``.

    A sample's declared noise cannot depend on which siblings were passed into the
    same fit.  The counterexample is the pre-fix value (shot/read only): every
    branch must report ``NoiseCalibration.variance`` instead.
    """
    calibration_ = repro_calibration()
    alone = build_noise_model([key(0, 0)], [110.0], calibration_)
    assert alone.kind == "diagonal"
    assert float(alone.variance_of(key(0, 0))) == pytest.approx(7.25)
    assert float(calibration_.variance([110.0])[0]) == pytest.approx(7.25)
    with_sibling = build_noise_model([key(0, 0), key(0, 1)], [110.0, 120.0], calibration_)
    assert with_sibling.kind == "dense"
    assert float(with_sibling.variance_of(key(0, 0))) == pytest.approx(7.25)
    assert float(with_sibling.variance_of(key(0, 1))) == pytest.approx(12.25)
    without_block = 0.5 * np.clip(110.0 - 100.0, 0.0, None) + 2.0
    assert without_block == pytest.approx(7.0)
    assert float(alone.variance_of(key(0, 0))) != pytest.approx(without_block)
    assert float(alone.variance_of(key(0, 0))) - without_block == pytest.approx(0.25)


def test_every_branch_reports_the_same_marginal_per_sample():
    """Diagonal and dense agree per sample; explicit independence removes the term."""
    calibration_ = repro_calibration()
    keys = [key(0, 0), key(0, 1), key(1, 0), key(2, 0)]
    means = [110.0, 120.0, 130.0, 140.0]
    expected = np.asarray([0.5 * max(mean - 100.0, 0.0) + 2.0 + 0.25 for mean in means])
    declared = build_noise_model(keys, means, calibration_)
    assert declared.kind == "dense"
    np.testing.assert_allclose(np.diagonal(declared.matrix.numpy()), expected, rtol=0.0, atol=1e-12)
    for position, source in enumerate(keys):
        assert float(declared.variance_of(source)) == pytest.approx(expected[position])
    matrix = declared.matrix.numpy()
    assert matrix[0, 1] == pytest.approx(0.25)
    assert matrix[0, 2] == 0.0 and matrix[0, 3] == 0.0 and matrix[1, 2] == 0.0
    # The singleton blocks taken on their own give the very same marginals.
    singletons = build_noise_model([key(1, 0), key(2, 0)], [130.0, 140.0], calibration_)
    assert singletons.kind == "diagonal"
    np.testing.assert_allclose(np.array(singletons.variances), expected[2:], rtol=0.0, atol=1e-12)
    # An explicit "shares no block" declaration is still the one way to drop the term.
    independent = build_noise_model(keys, means, calibration_, block_keys=(None,) * 4)
    assert independent.kind == "diagonal"
    np.testing.assert_allclose(
        np.array(independent.variances), expected - 0.25, rtol=0.0, atol=1e-12
    )


def test_one_sample_per_block_still_carries_every_block_term():
    """Worst case: no block has a sibling, so nothing correlates but every marginal keeps it."""
    calibration_ = repro_calibration()
    keys = [key(frame, 0) for frame in range(4)]
    means = [110.0, 120.0, 130.0, 140.0]
    model = build_noise_model(keys, means, calibration_)
    assert model.kind == "diagonal"
    expected = np.asarray([0.5 * (mean - 100.0) + 2.0 + 0.25 for mean in means])
    np.testing.assert_allclose(np.array(model.variances), expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        model.covariance_for(keys, dtype=torch.float64).numpy(),
        np.diag(expected),
        rtol=0.0,
        atol=1e-12,
    )
    assert float(model.variance_of(keys[3])) == pytest.approx(22.25)


def test_unknown_branch_keeps_the_declared_upper_bound_and_the_proxy_label():
    """The proxy route keeps its own meaning: a bound, never an exact covariance."""
    keys = [key(0, 0), key(0, 1)]
    means = [110.0, 120.0]
    unaudited = repro_calibration(block_variance=None, upper=0.5)
    proxy = build_noise_model(keys, means, unaudited, correlation="unknown")
    assert proxy.kind == "unknown_correlation"
    np.testing.assert_allclose(np.array(proxy.variances), [7.5, 12.5], rtol=0.0, atol=1e-12)
    # An audited block carries its known marginal here too; the label stays the proxy.
    audited = repro_calibration()
    audited_proxy = build_noise_model(keys, means, audited, correlation="unknown")
    assert audited_proxy.kind == "unknown_correlation"
    np.testing.assert_allclose(
        np.array(audited_proxy.variances), [7.25, 12.25], rtol=0.0, atol=1e-12
    )
    assert audited_proxy.kind != "dense"
    assert float(audited_proxy.variances[0]) == pytest.approx(float(audited.variance([110.0])[0]))


def test_dense_model_is_bounded_and_positive_definite():
    keys = [key(0, index) for index in range(MAX_DENSE_SOURCES + 1)]
    means = [BLACK + 10.0] * len(keys)
    with pytest.raises(UnsupportedNoiseStructureError, match="declared limit"):
        build_noise_model(keys, means, calibration())
    dense_keys = [key(0, index) for index in range(8)]
    dense = build_noise_model(dense_keys, [BLACK + 1.0] * 8, calibration())
    eigenvalues = np.linalg.eigvalsh(dense.matrix.numpy())
    per_sample = SHOT * 1.0 + READ_VAR
    assert eigenvalues.min() == pytest.approx(per_sample, rel=1e-9)
    assert eigenvalues.max() == pytest.approx(per_sample + 8.0 * BLOCK_VAR, rel=1e-9)


def test_noise_version_binds_to_the_committed_domain():
    """``fit_structure`` must accept the matching version and reject any other."""
    keys = [key(0, 0), key(0, 1), key(0, 2), key(0, 3)]
    ledger = MeasurementLedger("n01-test", value_units="raw_dn_12bit")
    ledger.observe(
        MeasurementFrame(
            values=torch.tensor([[[[100.0, 120.0, 140.0, 160.0]]]]),
            valid_mask=None,
            bad_mask=None,
            saturated_mask=None,
            episode_ids=["ep0"],
            frame_ids=[0],
            camera_id="cam0",
            timestamp=torch.tensor([0.0]),
            new_measurement=torch.tensor([True]),
            sample_xy=None,
        )
    )
    model = build_noise_model(keys, [100.0, 120.0, 140.0, 160.0], calibration())
    domain = make_domain(
        "noise-binding",
        reference_chart="sensor_xy",
        basis="phase_local_affine",
        phase=0,
        evidence_source_ids=keys,
        decision_version="dec-v1",
        mapping_version="map-v1",
        noise_version=model.version,
    )
    fit = fit_structure(domain, ledger, noise_model=model, dimension=1)
    assert fit.estimate.uncertainty_kind == "covariance"
    mismatched = make_domain(
        "noise-binding-other",
        reference_chart="sensor_xy",
        basis="phase_local_affine",
        phase=0,
        evidence_source_ids=keys,
        decision_version="dec-v1",
        mapping_version="map-v1",
        noise_version="n01-noise-some-other-version",
    )
    with pytest.raises(ValueError, match="noise version differs"):
        fit_structure(mismatched, ledger, noise_model=model, dimension=1)


# ---------------------------------------------------------------------------
# Provenance, JSON and the shipped calibration
# ---------------------------------------------------------------------------


def test_fitted_parameters_may_only_come_from_train_or_calibration():
    assert FITTING_SPLITS == ("train", "calibration")
    for split in ("train", "calibration"):
        fitted = provenance(kind="fitted", split=split)
        assert fitted.to_dict()["split"] == split
    for split in ("test", "validation"):
        with pytest.raises(ValueError, match="train or calibration"):
            provenance(kind="fitted", split=split)
    with pytest.raises(ValueError, match="synthesis"):
        provenance(kind="explicit_synthesis", split="train")


def test_calibration_json_roundtrip_keeps_provenance_and_unknowns():
    original = calibration(block_variance=None, block_upper=4.0)
    payload = original.to_dict()
    assert payload["formula"] == NOISE_FORMULA
    assert payload["shot_read"]["provenance"]["kind"] == "explicit_synthesis"
    assert payload["black_level"]["variance"] is None
    assert payload["black_level"]["upper_bound"] == 4.0
    restored = NoiseCalibration.from_dict(json.loads(json.dumps(payload)))
    assert restored == original
    assert restored.to_dict() == payload
    tampered = json.loads(json.dumps(payload))
    tampered["noise_version"] = "n01-noise-somewhere-else"
    with pytest.raises(ValueError, match="does not match"):
        NoiseCalibration.from_dict(tampered)
    unknown = json.loads(json.dumps(payload))
    unknown["shot_read"]["extra"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        NoiseCalibration.from_dict(unknown)
    missing = json.loads(json.dumps(payload))
    del missing["units"]
    with pytest.raises(ValueError, match="missing the required field"):
        NoiseCalibration.from_dict(missing)
    # A fitted calibration is representable too, and keeps its split on the way back.
    fitted = NoiseCalibration.from_dict(
        json.loads(
            json.dumps(
                calibration().to_dict()
                | {
                    "shot_read": calibration().to_dict()["shot_read"]
                    | {"provenance": provenance(kind="fitted", split="calibration").to_dict()}
                }
            )
        )
    )
    assert fitted.shot_read.provenance.split == "calibration"


def test_shipped_config_calibration_is_loadable_and_declares_the_refusal(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration_ = NoiseCalibration.from_dict(config["noise"])
    assert calibration_.noise_version == "n01-noise-p2-synth-cfa-v1"
    assert calibration_.units == "raw_dn_12bit"
    assert calibration_.black_level.audited
    assert tuple(calibration_.refused_structures) == (CROSS_PHASE_UNKNOWN_REFUSAL,)
    assert config["noise"]["formula"] == NOISE_FORMULA
    # The artifact survives a disk round trip through the module's own writer.
    written = calibration_.to_path(tmp_path / "noise.json")
    assert load_noise_calibration(written) == calibration_


# ---------------------------------------------------------------------------
# First-order flow propagation: a diagnostic, not a variance law
# ---------------------------------------------------------------------------


def test_first_order_flow_variance_is_exactly_the_quadratic_form():
    gradient = torch.tensor([12.0, -3.0], dtype=torch.float64).reshape(2)
    sigma = torch.tensor([[0.25, 0.05], [0.05, 0.5]], dtype=torch.float64)
    actual = float(flow_first_order_variance(gradient, sigma))
    oracle = float(np.array([12.0, -3.0]) @ sigma.numpy() @ np.array([12.0, -3.0]))
    assert actual == pytest.approx(oracle, rel=1e-12)
    with pytest.raises(ValueError, match=r"\[dx, dy\]"):
        flow_first_order_variance(torch.zeros(3), sigma)
    with pytest.raises(ValueError, match=r"\[2, 2\]"):
        flow_first_order_variance(gradient, torch.ones(2, 2, 2))


def test_first_order_flow_variance_fails_across_a_step_edge():
    """Counterexample: the linearisation misses a jump it cannot see."""
    step = np.array([0.0, 0.0, 1.0, 1.0])
    gradient = float(step[2] - step[1])  # the only non-zero difference, at the jump
    sigma_flow = 0.25  # a 0.5 px horizontal flow uncertainty
    estimate = float(
        flow_first_order_variance(
            torch.tensor([gradient, 0.0], dtype=torch.float64),
            torch.tensor([[sigma_flow, 0.0], [0.0, 0.0]], dtype=torch.float64),
        )
    )
    # A 0.5 px displacement can move the edge by one full intensity step, ~1.0:
    # the linear estimate (0.25) misses most of the true propagated change (1.0).
    assert estimate == pytest.approx(0.25)
    true_change = float(np.abs(step[1:].mean() - step[:-1].mean()) * 4.0)
    assert true_change > 4.0 * estimate
