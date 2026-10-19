"""Frozen diagnostic of measurement rejection, not a revival of N02 scheduling.

Nuisance changes preserve the generated world label. Structural violations are
outside the guard's single-positive-PSF family, not proof of a wrong world label.
No training, threshold tuning, candidate proposals or dependency learning.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from owac.baselines.n02_measurement_adapters import AdaptedMeasurementScorer
from owac.evaluation.n02_measurement_isolation import file_digest, tensor_digest, write_json
from owac.evaluation.n02_revision_comparison import absolute_patch_residual
from owac.frameworks.n02.synthetic import SyntheticGenerator, SyntheticOptions


def protocol():
    return {
        "scope": "absolute-fit boundary diagnostic; archived scheduler unchanged",
        "scene_seeds": list(range(28000, 28016)),
        "scorer_seeds": [0, 1, 2],
        "source": "outputs/n02-retrained-20261013/controls",
        "threshold": 2.5,
        "cases": {
            "normal": "unmodified measured final frame",
            "gain": "final measured frame multiplied by1.25; same world",
            "translation": "integer circular translation2pixels; same world",
            "noise": "additional independent Gaussian sigma.03; same world",
            "two_markers": "reference plus two separated positive PSFs; outside single-PSF family",
            "bright_field": "reference plus constant.15; outside single-PSF family",
        },
        "gate": "each nuisance false-rejection<=10%; each structural rejection>=90%",
        "interpretation": "rejection is model inadequacy, not semantic contradiction; "
        "ratio-confidence and world correctness reported separately",
        "budget": "zero training; frozen score checkpoints and threshold; CPU2threads",
    }


def boundary_cases(raw, seed):
    reference, final = raw[:, 0], raw[:, -1]
    rng = torch.Generator().manual_seed(seed + 100000)
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    markers = sum(
        torch.exp(-((y - cy).square() + (x - cx).square()) / 8) for cy, cx in ((16, 16), (48, 48))
    )
    return {
        "normal": final.clone(),
        "gain": final * 1.25,
        "translation": torch.roll(final, (2, 2), dims=(-2, -1)),
        "noise": final + 0.03 * torch.randn(final.shape, generator=rng),
        "two_markers": reference + 0.5 * markers[None, None],
        "bright_field": reference + 0.15,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = protocol()
    write_json(args.output / "protocol.json", declaration)
    source = Path(declaration["source"])
    stats = json.loads((source / "manifest.json").read_text())["modes"]["innovation"]
    sequences, labels = [], []
    for seed in declaration["scene_seeds"]:
        world = seed % 2
        episode = SyntheticGenerator(SyntheticOptions(height=64, width=64), seed=seed).generate(
            seed=seed,
            episode_index=seed,
            prior_condition="missing",
            identifiable=True,
            value=("left", "right")[world],
        )
        sequences.append(torch.from_numpy(episode.frames.copy()))
        labels.append(world)
    raw = torch.stack(sequences)
    labels = torch.tensor(labels)
    cases = boundary_cases(raw, declaration["scene_seeds"][0])
    torch.save({"raw": raw, "labels": labels, "cases": cases}, args.output / "data.pt")
    rows, summaries, checkpoints = [], {}, {}
    for model_seed in declaration["scorer_seeds"]:
        model = AdaptedMeasurementScorer(stats["mean"], stats["std"], "innovation").eval()
        checkpoint = source / "innovation" / f"seed{model_seed}" / "checkpoint200.pt"
        model.load_state_dict(
            torch.load(checkpoint, weights_only=False, map_location="cpu")["model"]
        )
        checkpoints[str(model_seed)] = {"path": str(checkpoint), "sha256": file_digest(checkpoint)}
        for name, final in cases.items():
            sequence = raw.clone()
            sequence[:, -1] = final
            scores = model.scores(sequence)[:, -1]
            probability = scores.softmax(-1)
            residual = absolute_patch_residual(final, raw[:, 0])
            best = scores.argmax(-1)
            both_bad = (residual > declaration["threshold"]).all(-1)
            # Production veto follows the ratio-selected candidate, not min residual.
            veto = residual.gather(1, best[:, None])[:, 0] > declaration["threshold"]
            summaries[f"seed{model_seed}/{name}"] = {
                "both_bad": int(both_bad.sum()),
                "selected_veto": int(veto.sum()),
                "relative_confident": int((probability.max(-1).values >= 0.95).sum()),
                "world_correct": int((best == labels).sum())
                if name not in ("two_markers", "bright_field")
                else None,
                "denominator": len(raw),
            }
            rows.append(
                {
                    "seed": model_seed,
                    "case": name,
                    "scores": scores.tolist(),
                    "residual": residual.tolist(),
                    "veto": veto.tolist(),
                }
            )
    nuisance_ok = all(
        summaries[f"seed{s}/{c}"]["selected_veto"] / len(raw) <= 0.1
        for s in declaration["scorer_seeds"]
        for c in ("normal", "gain", "translation", "noise")
    )
    structural_ok = all(
        summaries[f"seed{s}/{c}"]["both_bad"] / len(raw) >= 0.9
        for s in declaration["scorer_seeds"]
        for c in ("two_markers", "bright_field")
    )
    write_json(args.output / "metrics.json", rows)
    write_json(
        args.output / "summary.json",
        {
            "cases": summaries,
            "nuisance_gate": nuisance_ok,
            "structural_gate": structural_ok,
            "ready_for_semantic_revision": False,
            "reason": "Passing a rejection guard does not identify a semantic claim to retract.",
        },
    )
    write_json(
        args.output / "manifest.json",
        {
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "checkpoints": checkpoints,
            "raw_sha256": tensor_digest(raw),
            "source_sha256": {
                str(p): file_digest(p)
                for p in (Path(__file__), Path("src/owac/evaluation/n02_revision_comparison.py"))
            },
            "protocol_sha256": file_digest(args.output / "protocol.json"),
            "data_sha256": file_digest(args.output / "data.pt"),
        },
    )


if __name__ == "__main__":
    main()
