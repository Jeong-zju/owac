"""Original per-plane integration, immutable source sums and real withdrawal."""

import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01_dbsr_raw import RawDomainHistory, config_for


def frame(values, now, variance=0.01):
    return MeasurementFrame(
        values=values,
        valid_mask=torch.ones_like(values, dtype=torch.bool),
        camera_id="burst",
        frame_ids=torch.full((values.shape[0],), now),
        timestamp=torch.full((values.shape[0],), float(now)),
        metadata={"variance": variance},
    )


def test_two_measurement_reference_has_original_inverse_variance_mean_age_and_variance():
    domain = RawDomainHistory(config_for("stock_dynamic", {}), 1, (12, 12))
    domain.step(frame(torch.full((1, 4, 12, 12), 0.1), 0, 0.04), "stock_dynamic")
    stats = domain.step(frame(torch.full((1, 4, 12, 12), 0.3), 1, 0.01), "stock_dynamic")
    torch.testing.assert_close(stats["mu"], torch.full_like(stats["mu"], 0.26), atol=5e-6, rtol=0)
    torch.testing.assert_close(
        stats["variance"], torch.full_like(stats["variance"], 0.008), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(stats["age"], torch.full_like(stats["age"], 0.2), atol=2e-5, rtol=0)
    assert len(set(stats["unique_source_ids"][0]) - {-1}) == 2


def test_constant_support_cached_sufficient_sums_match_raw_recomputation():
    cfg = config_for("calibrated_recompute", {}).replace(
        edge_alpha=1.0, edge_temperature=1e12, edge_beta=1e12
    )
    a, b = RawDomainHistory(cfg, 1, (12, 12)), RawDomainHistory(cfg, 1, (12, 12))
    rng = torch.Generator().manual_seed(66)
    for now in range(3):
        measurement = frame(torch.rand(1, 4, 12, 12, generator=rng), now, 0.01 * (now + 1))
        fresh = a.step(measurement, "calibrated_recompute")
        cached = b.step(measurement, "calibrated_stale")
        for key in ["mu", "variance", "age"]:
            torch.testing.assert_close(fresh[key], cached[key], atol=1e-6, rtol=1e-5)


def test_split_raw_recomputation_retracts_old_neighbour_sums_that_cache_keeps():
    cfg = config_for(
        "calibrated_recompute", {"edge_temperature": 3.9740435495127926, "edge_beta": 8.0}
    )
    domains = [RawDomainHistory(cfg, 1, (12, 12)) for _ in range(4)]
    outputs = []
    for now in range(7):
        plain = torch.full((1, 4, 12, 12), 0.3)
        changed = plain.clone()
        if now < 6:
            changed[..., 6:] += 0.01
        else:
            plain[..., 6:] = 0.9
            changed[..., 6:] = 0.9
        outputs = [
            domains[0].step(frame(plain, now), "calibrated_recompute"),
            domains[1].step(frame(changed, now), "calibrated_recompute"),
            domains[2].step(frame(plain, now), "calibrated_stale"),
            domains[3].step(frame(changed, now), "calibrated_stale"),
        ]
    assert torch.equal(outputs[0]["mu"][..., :6], outputs[1]["mu"][..., :6])
    assert not torch.equal(outputs[2]["mu"][..., :6], outputs[3]["mu"][..., :6])
    assert sum(outputs[0]["splits"]) > 0
