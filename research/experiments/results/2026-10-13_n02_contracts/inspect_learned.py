"""Read-only, post-training contract diagnosis on the development seeds."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.evaluation.n02_bottlenecks import paired_episodes, tensor_digest
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.synthetic import values_for_task
from owac.training.n02 import episode_prior_context

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "outputs/n02-contracts-20261013"


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    summaries, all_records = {}, []
    for seed in (0, 1, 2):
        payload = torch.load(OUTPUT / f"seed{seed}/checkpoint200.pt", weights_only=False)
        cfg = N02Config.from_dict(payload["config"])
        model = N02Encoder(cfg).eval()
        model.load_state_dict(payload["model"])
        before = tensor_digest(model)
        original = model.reviser.semantic_map
        traces = []

        def captured(energy, original=original, cfg=cfg, traces=traces):
            result = original(energy)
            support = energy.support_mask
            geo = energy.geometry_gap_from_best >= cfg.revision.geometry_separation
            bf = energy.local_log_evidence_against >= cfg.revision.semantic_log_bayes_threshold
            good = energy.best_normalized_residual <= cfg.revision.contradiction_threshold
            traces.append(
                {
                    "geometry_eligible": int((support & geo).sum()),
                    "local_bf_eligible": int((support & bf).sum()),
                    "good_rival_eligible": int((support & good).sum()),
                    "semantic_refutation": int(result.sum()),
                    "supported": int(support.sum()),
                }
            )
            return result

        model.reviser.semantic_map = captured
        rows = []
        for data_seed in range(400, 416):
            generator, episodes = paired_episodes(data_seed)
            for condition in ("misleading", "unidentifiable", "missing"):
                episode = episodes[condition]
                quality = generator.measurement_quality(episode)
                context = episode_prior_context([episode], cfg.K)
                state = None
                for index, raw in enumerate(episode.frames):
                    frame = MeasurementFrame(
                        values=torch.from_numpy(raw).unsqueeze(0),
                        frame_ids=(index,),
                        episode_ids=(episode.episode_id,),
                        camera_id="synthetic_measurement",
                        metadata={"variance": 0.0001},
                        metadata_available={"variance": True},
                    )
                    start_trace = len(traces)
                    output = model.step(
                        frame, context, state, ComputeBudget(cfg.cycles, cfg.history)
                    )
                    state = model.detach_state(output.state)
                    rep = output.representation
                    mean, logvar, geo = model.measurement.predict(rep.h)
                    mean = mean[0, 0]
                    target = F.adaptive_avg_pool2d(
                        torch.from_numpy(generator.measurement_templates(episode, index)),
                        mean.shape[-2:],
                    )
                    physical_contrast = target[1] - target[0]
                    learned_contrast = mean[1::2].mean(0) - mean[::2].mean(0)
                    norm = physical_contrast.square().sum()
                    energy = rep.measurement_energy.flatten()
                    logbf = torch.logsumexp(-energy[1::2], 0) - torch.logsumexp(-energy[::2], 0)
                    truth = values_for_task(episode.task).index(episode.value_over_time[index])
                    row = {
                        "seed": seed,
                        "data_seed": data_seed,
                        "condition": condition,
                        "frame": index,
                        "cue_visible": bool(quality["identifiable_now"][index]),
                        "geometry_span": float(geo[0, 0, :, 0].max() - geo[0, 0, :, 0].min()),
                        "physical_contrast_rms": float(physical_contrast.square().mean().sqrt()),
                        "learned_contrast_rms": float(learned_contrast.square().mean().sqrt()),
                        "contrast_gain": float((learned_contrast * physical_contrast).sum() / norm)
                        if norm > 0
                        else None,
                        "measurement_logbf_for_truth": float(logbf * (2 * truth - 1)),
                        "absolute_measurement_logbf": float(logbf.abs()),
                        "model_std": float(logvar.exp().mean().sqrt()),
                        "sensor_std": float(
                            state.diagnostics["_buffer"].frames[-1].variance.mean().sqrt()
                        ),
                        "cycles": traces[start_trace:],
                    }
                    rows.append(row)
        assert before == tensor_digest(model)
        summaries[str(seed)] = {
            "parameters_unchanged": True,
            "frames": len(rows),
            "cycles": len(traces),
        }
        for visible in (False, True):
            selected = [row for row in rows if row["cue_visible"] == visible]
            cycles = [cycle for row in selected for cycle in row["cycles"]]
            group = {
                key: float(np.mean([row[key] for row in selected]))
                for key in (
                    "geometry_span",
                    "physical_contrast_rms",
                    "learned_contrast_rms",
                    "absolute_measurement_logbf",
                    "model_std",
                    "sensor_std",
                )
            }
            group["contrast_gain"] = (
                float(np.mean([row["contrast_gain"] for row in selected])) if visible else None
            )
            group["correct_measurement_direction_fraction"] = (
                float(np.mean([row["measurement_logbf_for_truth"] > 0 for row in selected]))
                if visible
                else None
            )
            for key in (
                "geometry_eligible",
                "local_bf_eligible",
                "good_rival_eligible",
                "semantic_refutation",
            ):
                group[key + "_fraction"] = sum(cycle[key] for cycle in cycles) / sum(
                    cycle["supported"] for cycle in cycles
                )
            summaries[str(seed)]["cue" if visible else "no_cue"] = group
        all_records.extend(rows)
    (OUTPUT / "learned_diagnostic_records.json").write_text(
        json.dumps(all_records, indent=2) + "\n"
    )
    (Path(__file__).parent / "learned_diagnostics.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
