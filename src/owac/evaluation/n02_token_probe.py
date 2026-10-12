"""Consume actual readout tokens with a fitted localization probe.

This is an explicit two-candidate fixture, not a learned N02 or PI05 result.
The frozen random TokenReadout uses 256 x 2048 tokens. A ridge decoder is fitted
only on unambiguous calibration fields. Geometry diagnostics are poisoned with
NaN so neither fitting nor inference can use the geometry shortcut.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from owac.frameworks.n02.readout import StructuredRepresentation, TokenReadout


def run(seed=0):
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        readout = TokenReadout(8, 2, token_dim=2048).eval()
        direction = torch.randn(8)
        nuisance = torch.randn(8)
        nuisance -= direction * (nuisance @ direction) / direction.square().sum()
        direction /= direction.norm()
        nuisance /= nuisance.norm()

        def representation(labels, disturbance, weights):
            fields = (
                labels[..., None] * direction[None, None, :]
                + disturbance[:, None, None] * nuisance[None, None, :]
            )
            h = fields[:, None, :, :, None, None].expand(-1, 1, -1, -1, 16, 16)
            zeros = torch.zeros(h.shape[0], 1, 2)
            return StructuredRepresentation(
                h=h,
                weights=weights[:, None, :],
                measurement_energy=zeros,
                prior_energy=zeros,
                relation_energy=zeros,
                geometry=torch.full((h.shape[0], 1, 2, 1), float("nan")),
                support=torch.ones(h.shape[0], 1, 2, 16, 16),
            )

        def consume(rep, mode):
            out = readout(rep, token_budget=256, mode=mode, camera_ids=("probe",))
            values = out.tokens.values
            if values.shape[1:] != (256, 2048) or not bool(torch.isfinite(values).all()):
                raise RuntimeError("invalid token contract in localization probe")
            groups = 2 if mode == "tokens" else 1
            return values.reshape(values.shape[0], groups, 256 // groups, 2048).mean(2)

        with torch.no_grad():
            # Both candidate identities carry each sign during calibration.
            # The sign cannot be recovered from hypothesis ID or weight alone.
            labels = torch.tensor([-1.0, 1.0]).repeat(16)
            labels = labels[:, None].expand(-1, 2)
            disturbance = torch.linspace(-0.6, 0.6, 32)
            weights = torch.tensor([[0.6, 0.4], [0.4, 0.6]]).repeat(16, 1)
            calibration = representation(labels, disturbance, weights)
            grouped = consume(calibration, "tokens").reshape(-1, 2048)
            mapped = consume(calibration, "map").reshape(-1, 2048)
            features = torch.cat([grouped, mapped])
            targets = torch.cat([labels.reshape(-1), labels[:, 0]])
            center = features.mean(0)
            centered = features - center
            target_mean = targets.mean()
            ridge = 1e-3
            alpha = torch.linalg.solve(
                centered @ centered.T + ridge * torch.eye(centered.shape[0]),
                targets - target_mean,
            )
            coefficients = centered.T @ alpha

            def decode(values):
                return (values - center) @ coefficients + target_mean

            calibration_error = (decode(features) - targets).square().mean()
            # Hold out disturbances; these fields explicitly encode opposite
            # localization alternatives. No PI05/action labels are fabricated.
            disturbance = torch.linspace(-0.5, 0.5, 16).repeat_interleave(4)
            weights = torch.tensor([[0.5, 0.5], [0.65, 0.35], [0.85, 0.15], [0.95, 0.05]])
            weights = weights.repeat(16, 1)
            fixture = representation(
                torch.tensor([[-1.0, 1.0]]).repeat(64, 1), disturbance, weights
            )
            groups = decode(consume(fixture, "tokens"))
            mapped = decode(consume(fixture, "map"))
            # Direct mean is formed in the state field before the same readout.
            mean_h = (fixture.h * weights[:, None, :, None, None, None]).sum(2, keepdim=True)
            mean_field = replace(fixture, h=mean_h.expand_as(fixture.h))
            averaged = decode(consume(mean_field, "map"))
            wrong_fusion = groups.mean(-1, keepdim=True)
            outputs = {
                "map": mapped,
                "grouped": groups,
                "direct_mean": averaged,
                "grouped_then_mean_pool": wrong_fusion,
            }
            records = []
            for index in range(64):
                records.append(
                    {
                        "disturbance": float(disturbance[index]),
                        "weights": weights[index].tolist(),
                        "predicted_localizations": {
                            k: v[index].tolist() for k, v in outputs.items()
                        },
                    }
                )
            return {
                "seed": seed,
                "scope": (
                    "explicit incompatible-field fixture; fitted token localization decoder; "
                    "not learned N02/PI05"
                ),
                "token_budget": 256,
                "token_dim": 2048,
                "readout_initialization": "seeded random, frozen",
                "decoder": "ridge regression on unambiguous calibration fields only",
                "calibration_mse": float(calibration_error),
                "geometry_diagnostic": "NaN; all consumed tokens finite",
                "invalid_middle_threshold": "abs(localization) < 0.5 for endpoints -1,+1",
                "invalid_middle_fraction": {
                    k: float((v.abs() < 0.5).float().mean()) for k, v in outputs.items()
                },
                "nearest_endpoint_mae": {
                    k: float((v.abs() - 1).abs().mean()) for k, v in outputs.items()
                },
                "records": records,
                "conclusion": (
                    "Grouping offers separate alternatives; backend pooling can still collapse "
                    "them. This probe cannot validate PI05 ambiguity handling."
                ),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["invalid_middle_fraction"]))


if __name__ == "__main__":
    main()
