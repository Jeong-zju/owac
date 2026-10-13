"""Before/after numeric receipts; no learned capability or robot-score claims."""

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.synthetic import SyntheticGenerator, values_for_task
from owac.training.n02 import step_terms

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/n02-repair-20261013"


def legacy(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def loss_probe(function):
    result = {}
    for key, mass in (("biased", [0.01, 0.49, 0.01, 0.49]), ("uniform", [0.25] * 4)):
        energy = torch.zeros(1, 1, 4, requires_grad=True)
        weights = torch.tensor([[mass]], requires_grad=True)
        rep = SimpleNamespace(
            weights=weights, measurement_energy=energy, geometry=torch.zeros(1, 1, 4, 8)
        )
        terms = function(
            rep,
            anchor=torch.tensor([0]),
            wrong=torch.tensor([1]),
            post_contradiction=torch.tensor([False]),
            observations=1,
            diversity_active=False,
        )
        terms["mixture"].backward(retain_graph=True)
        grad = None if weights.grad is None else weights.grad.detach().flatten().tolist()
        energy.grad = None
        terms["measurement"].backward()
        result[key] = {
            "mixture": float(terms["mixture"].detach()),
            "posterior_gradient": grad,
            "measurement_gradient": energy.grad.flatten().tolist(),
        }
    return result


def main():
    torch.set_num_threads(1)
    old = legacy("n02_original_sensor", OUT / "before/synthetic_original.py")
    old_loss = legacy("n02_old_training", OUT / "before/src/owac/training/n02.py")
    old_revision = legacy("n02_old_revision", OUT / "before/src/owac/frameworks/n02/revision.py")
    sensor = []
    for seed in range(100, 104):
        row = {"seed": seed}
        for label, module in (
            ("before", old),
            ("after", sys.modules[SyntheticGenerator.__module__]),
        ):
            gen = module.SyntheticGenerator(
                module.SyntheticOptions(height=64, width=64), seed=seed, split="validation"
            )
            ep = gen.generate(seed=seed, identifiable=True)
            clean = gen.clean_measurements(ep)
            logbf = 0.0
            for step in range(8):
                truth = values_for_task(ep.task).index(ep.value_over_time[step])
                rival = replace(
                    ep,
                    geometry_before=ep.geometry_before.with_value(
                        values_for_task(ep.task)[1 - truth]
                    ),
                    geometry_after=ep.geometry_after.with_value(
                        values_for_task(ep.task)[1 - truth]
                    ),
                )
                competing = gen.clean_measurements(rival)[step]
                attenuation = (
                    ep.occluder_mask * ep.transmission[step]
                    if label == "before"
                    else 1 - ep.occluder_mask * (1 - ep.transmission[step])
                )
                errors = [
                    np.square(ep.frames[step] - attenuation[None] * template).sum()
                    for template in (clean[step], competing)
                ]
                logbf += (errors[1] - errors[0]) / (2 * 0.01**2)
            row[label] = {
                "gaussian_log_bayes_factor": float(logbf),
                "correct_mass_from_95_percent_wrong_prior": 1
                / (1 + math.exp(-float(np.clip(logbf - math.log(19), -700, 700)))),
                "uncovered_noise_variance": float(ep.noise[:, :, ep.occluder_mask == 0].var()),
            }
        sensor.append(row)
    cfg = N02Config(K=4, D=8)
    new_model = N02Encoder(cfg)
    controllers = {"before": old_revision.RevisionController(cfg), "after": new_model.reviser}
    source = torch.zeros(1, 1, 4, 5, 5, dtype=torch.bool)
    source[0, 0, 2, 2, 2] = True
    edge = torch.eye(4).expand(1, 1, 5, 5, 4, 4, 4).clone()
    edge[..., 2, 2] = 0.985
    edge[..., 2, 0] = 0.015
    valid = torch.ones(1, 1, 5, 5, 4)
    propagation = {}
    for label, controller in controllers.items():
        selected, priority = controller.propagate(source, edge, valid)
        propagation[label] = {
            "additional_cells": int(selected.sum()) - 1,
            "single_neighbour_priority": float(priority[0, 0, 2, 3, 2]),
        }
    score = torch.tensor([0.0, 4.0, 0.0, 4.0]).reshape(1, 1, 4, 1, 1)
    withdrawal = {}
    for factor in (0, 0.5, 1):
        config = replace(cfg, revision=replace(cfg.revision, downweight_contradicted=factor))
        withdrawal[str(factor)] = (
            N02Encoder(config)
            .reviser.withdrawal_energy(score, torch.ones_like(score), temperature=1)
            .flatten()
            .tolist()
        )
    old_checkpoint = ROOT / "outputs/n02-final-candidate/checkpoint.pt"
    payload = torch.load(old_checkpoint, map_location="cpu", weights_only=False)
    model = N02Encoder(N02Config.from_dict(payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    report = {
        "scope": "independent source regressions, not learned-model acceptance",
        "sensor": sensor,
        "loss": {"before": loss_probe(old_loss.step_terms), "after": loss_probe(step_terms)},
        "weak_dependency": propagation,
        "withdrawal_energy_by_retention": withdrawal,
        "old_checkpoint_strict_load": True,
        "old_checkpoint_sha256": hashlib.sha256(old_checkpoint.read_bytes()).hexdigest(),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    (Path(__file__).resolve().parent / "audit_probes.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
