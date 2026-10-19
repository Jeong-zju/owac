"""Small deterministic probes against the delivered model and checkpoint; no training."""

import dataclasses
import hashlib
import json

import numpy as np
import torch

from owac.evaluation.n02_full import p1 as evaluation
from owac.frameworks.n02_full import config, factors, synthetic
from owac.training import n02_full as training


def raw_hash(sequence):
    digest = hashlib.sha256()
    for inference in sequence.inference:
        for camera, frame in sorted(inference.bundle.frames.items()):
            digest.update(camera.encode())
            digest.update(np.asarray(frame.values).tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(1103)
    cfg = config.load_config("configs/encoders/n02_full/p1_measurement.json")
    measurement = cfg.training["measurement"]
    sensors = tuple(synthetic.SensorSpec(**entry) for entry in measurement["synthetic_sensors"])
    generator = synthetic.MeasurementModel(**measurement["measurement_model"])
    common = dict(
        sensors=sensors,
        measurement_model=generator,
        partitions=("train",),
        frames=8,
        views_per_scene=1,
        seed=1103,
        canvas=tuple(measurement["canvas"]),
        action_horizon=measurement.get("action_horizon", 4),
    )
    train_sequences = iter(
        synthetic.generate_dataset(scenes=measurement["synthetic_scenes"], **common)
    )
    eval_sequences = iter(synthetic.generate_dataset(scenes=3000, **common))
    overlaps = []
    shortcuts = []
    first = None
    for _ in range(16):
        left, right = next(train_sequences), next(eval_sequences)
        if first is None:
            first = left
        overlaps.append(
            {
                "train": left.scene_id,
                "eval": right.scene_id,
                "raw_sha256_equal": raw_hash(left) == raw_hash(right),
            }
        )
        for correct_index in (0, 1):
            task = factors.build_p1_task(right, frame_index=1, correct_index=correct_index)
            # This rule uses only the candidate's numeric nuisance input; no y/truth.
            choices = [
                i
                for i, item in enumerate(task.dictionary.specs)
                if item.nuisance.get("log_exposure", 0.0) == 0.0
            ]
            shortcuts.append(len(choices) == 1 and choices[0] == task.candidate_index)
    task = factors.build_p1_task(first, frame_index=1)
    model = training.build_observation_model(
        cfg, "cuda", units=task.units, world_size=task.dictionary.world_size()
    )
    path = (
        "outputs/n02-full/p1-formal/"
        "p1-measurement-seed1103-formal-cfa4-meas-5k-true/checkpoints/final.pt"
    )
    payload = training.load_checkpoint(path)
    loaded = training.load_base_subset(model, payload["model"])
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "encoder"):
        model.encoder.eval()
    results = []
    for peak in (0, 1):
        prior = np.full(len(task.dictionary.specs), 0.05 / (len(task.dictionary.specs) - 1))
        prior[peak] = 0.95
        changed = dataclasses.replace(
            task, inference=dataclasses.replace(task.inference, declared_prior=prior)
        )
        output = evaluation._model_step(model, changed, first)
        scores, posterior = evaluation._posterior(output)
        results.append(
            {
                "prior": prior.tolist(),
                "scores": scores.tolist(),
                "posterior": posterior.tolist(),
                "locked_check_types": [type(item).__name__ for item in output.locked_checks],
                "verdicts": evaluation._verdicts(output),
                "score_kinds": [item.kind for item in output.locked_checks],
            }
        )
    print(
        json.dumps(
            {
                "train_eval_overlap_probe": overlaps,
                "numeric_nuisance_only_classifier": {
                    "correct": sum(shortcuts),
                    "total": len(shortcuts),
                },
                "checkpoint_load": loaded,
                "same_input_opposite_priors": results,
                "opposite_priors_scores_bitwise_equal": results[0]["scores"]
                == results[1]["scores"],
                "opposite_priors_posterior_bitwise_equal": results[0]["posterior"]
                == results[1]["posterior"],
            }
        )
    )


if __name__ == "__main__":
    main()
