"""Real official-window v4 auxiliary checks; no optimizer updates or checkpoint writes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from owac.raw.n04_v11_data import load_window_batch, sample_windows
from owac.training.n04_v11 import V11Trainer, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer = V11Trainer(cfg, seed=101, run_dir=args.output, device="cuda:0")

    def forbidden(*args, **kwargs):
        raise AssertionError("verification cannot update or save model weights")

    trainer.optimizer.step = forbidden
    trainer.scheduler.step = forbidden
    trainer.save_checkpoint = forbidden
    declared = dict(trainer.loss_weights)
    versions = {name: p._version for name, p in trainer._trainable_parameters()}
    results = {"platform": trainer.platform, "optimizer_updates": 0}
    args.output.mkdir(parents=True, exist_ok=True)

    def write():
        (args.output / "verification.json").write_text(json.dumps(results, indent=2) + "\n")

    def window(spec):
        entry, start = spec
        return load_window_batch(
            entry, start=start, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
        )

    fixed = window(trainer.dev_windows[0])
    trainer.encoder.eval()
    trainer.backend.eval()
    results["initialization_counterfactual"] = {}
    initializations = (
        (("configured_identity", None),)
        if cfg["model"].get("transition_init") == "identity"
        else (("author_rotation", -0.2), ("constant_velocity", 0.0))
    )
    for label, lower_left in initializations:
        with torch.no_grad():
            if lower_left is not None:
                for cell in trainer.encoder.state_stack.cells:
                    cell._tm_21_basis.zero_()
                    cell._tm_21_basis[:, cell._diag_idx] = lower_left
            trainer._seed_window(0, 0)
            _, info = trainer._forward_window(fixed, train=False)
        row = {"terms": info["terms"], "forecast": info["forecast"]}
        results["initialization_counterfactual"][label] = row
        print(json.dumps({"initialization": label, **row}), flush=True)
        write()
    # Changes above were a deliberate no-update initialization counterfactual.
    versions = {name: p._version for name, p in trainer._trainable_parameters()}
    trainer.encoder.train()
    trainer.backend.train()
    plan = sample_windows(trainer.train_candidates, seed=101, step=0, batch=8)
    cache = trainer._representation_batch(plan, 0)
    results["representation_batch"] = cache.diagnostics
    active = window(plan[0])
    gradients = []
    for objective in ("action", "correspondence", "forecast", "representation", "joint_exact"):
        trainer.loss_weights = (
            declared
            if objective == "joint_exact"
            else {key: weight if key == objective else 0.0 for key, weight in declared.items()}
        )
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer._seed_window(0, 0)
        _, info = trainer._forward_window(
            active,
            train=True,
            representation_cache=cache,
            representation_index=0,
            _fast_counts=False if objective == "joint_exact" else None,
        )
        norms = {}
        for name, parameter in trainer._trainable_parameters():
            assert parameter._version == versions[name], name
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            assert bool(torch.isfinite(grad).all()), name
            group = ".".join(name.split(".")[:3])
            norms[group] = norms.get(group, 0) + float(grad.square().sum())
        row = {
            "objective": objective,
            "terms": info["terms"],
            "valid_counts": info["valid_loss_counts"],
            "gradient_norms": {key: math.sqrt(value) for key, value in norms.items()},
            "parameters_unchanged": True,
            "gradient_eager_fallback": info.get("gradient_eager_fallback", False),
        }
        gradients.append(row)
        results["objective_gradients"] = gradients
        print(json.dumps(row), flush=True)
        write()
    trainer.loss_weights = declared
    trainer.optimizer.zero_grad(set_to_none=True)
    results["complete"] = True
    write()


if __name__ == "__main__":
    main()
