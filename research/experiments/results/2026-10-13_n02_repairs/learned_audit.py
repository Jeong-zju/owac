"""Post-training diagnostics and physical-template score calibration reference."""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from owac.core.encoder_contract import ComputeBudget, EmbodimentContext, MeasurementFrame
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.synthetic import SyntheticGenerator, SyntheticOptions, values_for_task

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "outputs/n02-repair-20261013"


def main():
    torch.set_num_threads(1)
    checkpoint = OUTPUT / "repaired/checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = N02Config.from_dict(payload["config"])
    model = N02Encoder(config)
    model.load_state_dict(payload["model"])
    model.eval()
    energies, penalties, traces, slots = [], [], [], []
    original_step = model.reviser.step

    def capture_trace(*args, **kwargs):
        result = original_step(*args, **kwargs)
        traces.append(result.trace)
        return result

    model.reviser.step = capture_trace
    with torch.no_grad():
        for seed in range(100, 116):
            gen = SyntheticGenerator(
                SyntheticOptions(height=64, width=64), seed=seed, split="validation"
            )
            episode = gen.generate(seed=seed, prior_condition="uninformative", identifiable=False)
            state = None
            for index, values in enumerate(episode.frames):
                output = model.step(
                    MeasurementFrame(
                        values=torch.from_numpy(values)[None],
                        frame_ids=(index,),
                        episode_ids=(episode.episode_id,),
                        metadata={"variance": 0.01**2},
                        metadata_available={"variance": True},
                    ),
                    EmbodimentContext(),
                    state,
                    ComputeBudget(4, 4),
                )
                state = output.state
                energies.append(output.representation.measurement_energy.flatten().tolist())
                penalties.append(output.representation.revision_energy.flatten().tolist())
        source = torch.zeros(1, 1, 4, 5, 5, dtype=torch.bool)
        for k in range(4):
            source.zero_()
            source[0, 0, k, 2, 2] = True
            edge = model.edge_parameter[None, None, None, None].expand(1, 1, 5, 5, 4, 4, 4)
            selected, _ = model.reviser.propagate(source, edge, torch.ones(1, 1, 5, 5, 4))
            slots.append(int(selected.sum()) - 1)
    # Information check at the very same pooled grid, robust loss and noise scale.
    # Both physical templates are given to this oracle only; no model is trained
    # or supplied those parameters. This is an upper-reference, not a fair baseline.
    oracle = []
    for seed in range(100, 116):
        gen = SyntheticGenerator(
            SyntheticOptions(height=64, width=64), seed=seed, split="validation"
        )
        episode = gen.generate(seed=seed, prior_condition="correct", identifiable=True)
        logodds = -math.log(19)
        for step in range(8):
            templates = gen.measurement_templates(episode, step)
            y = F.avg_pool2d(torch.from_numpy(episode.frames[step])[None], 4)
            means = F.avg_pool2d(torch.from_numpy(templates), 4)
            normalized = (y - means) / math.sqrt(0.01**2 / 16)
            absolute = normalized.abs()
            delta = config.measurement.huber_delta
            cost = torch.where(
                absolute <= delta, 0.5 * normalized.square(), delta * (absolute - 0.5 * delta)
            ).mean((1, 2, 3))
            truth = values_for_task(episode.task).index(episode.value_over_time[step])
            withdrawal = torch.zeros(4)
            if step > 0:
                residual_map = normalized.abs().mean(1)[torch.arange(4) % 2][None, None]
                withdrawal = model.reviser.withdrawal_energy(
                    residual_map, torch.ones_like(residual_map), temperature=config.temperature
                ).flatten()
            logodds += (
                float(cost[1 - truth] - cost[truth] + withdrawal[1 - truth] - withdrawal[truth])
                / config.temperature
            )
        mass = 1 / (1 + math.exp(float(np.clip(-logodds, -700, 700))))
        oracle.append({"seed": seed, "final_truth_mass": mass})
    report = {
        "scope": "learned diagnostics and hidden-template input oracle; no new acceptance",
        "no_cue_measurement_energy_mean_by_slot": np.mean(energies, axis=0).tolist(),
        "no_cue_nonzero_withdrawal_frames": sum(any(x > 0 for x in row) for row in penalties),
        "no_cue_frames_audited": len(penalties),
        "no_cue_mean_direct_contradicted_fraction": float(
            np.mean([r["contradicted_fraction"] for r in traces])
        ),
        "no_cue_mean_propagated_fraction": float(
            np.mean([r["propagated_fraction"] for r in traces])
        ),
        "trained_edge_isolated_refutation_added_cells_by_source_slot": slots,
        "same_grid_huber_physical_template_oracle": oracle,
        "same_grid_oracle_final_mass_ge_point7": sum(r["final_truth_mass"] >= 0.7 for r in oracle),
        "oracle_note": "physical templates at 16x16; withdrawal once per new frame; evaluator only",
    }
    (Path(__file__).resolve().parent / "learned_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
