"""Read-only multi-window objective/gradient/Adam diagnostic on actual r5 weights.

No optimizer.step call, weight mutation, or training continuation. Old language
layout is explicit so this inspects the failed training objective, not r6 input.
"""

import argparse
import gc
import json
import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, resolve_plan
from owac.training.n02_v12_profile import forbid_optimizer_steps, parameter_digest

WINDOWS = (
    (100, "previous_grab_roller"),
    (58058, "fixed_middle"),
    (100088, "rgb675_logged_window"),
    (38342, "rgb360_logged_window"),
)


def legacy_language(self, prompts, proprio):
    state = torch.zeros((len(prompts), 32), device=self.device)
    tokens, masks = [], []
    for index, prompt in enumerate(prompts):
        row = np.asarray(proprio[index], dtype=np.float32)
        state[index, : row.size] = torch.as_tensor(row, device=self.device)
        token, mask = self.bridge.base.tokenizer.tokenize(prompt, row)
        tokens.append(torch.as_tensor(np.asarray(token)).reshape(-1))
        masks.append(torch.as_tensor(np.asarray(mask)).reshape(-1).bool())
    return torch.stack(tokens).to(self.device), torch.stack(masks).to(self.device), state


def geometry(grads, indices):
    # Dot products only on the requested role; no parameter-count dilution by other modules.
    totals = torch.zeros(6, device="cuda", dtype=torch.float64)
    for index in indices:
        a, m, c = (grads[name][index] for name in ("action", "masked_raw", "causal_raw"))
        values = (a, m, c)
        for k, value in enumerate(values):
            if value is not None:
                totals[k] += value.double().square().sum()
        for k, (left, right) in enumerate(((a, m), (a, c), (m, c)), 3):
            if left is not None and right is not None:
                totals[k] += (left.double() * right.double()).sum()
    a2, m2, c2, am, ac, mc = totals.cpu().tolist()

    def cosine(dot, left, right):
        return dot / math.sqrt(left * right) if left * right else None

    total2 = max(0.0, a2 + m2 + c2 + 2 * (am + ac + mc))
    return {
        "action_l2": math.sqrt(a2),
        "masked_l2": math.sqrt(m2),
        "causal_l2": math.sqrt(c2),
        "total_l2": math.sqrt(total2),
        "cosine_action_masked": cosine(am, a2, m2),
        "cosine_action_causal": cosine(ac, a2, c2),
        "action_dot_total": a2 + am + ac,
        "action_dot_action": a2,
        "combined_aux_to_action_l2": math.sqrt(max(0.0, m2 + c2 + 2 * mc) / a2) if a2 else None,
    }


def virtual_adam(parameters, grads, saved_optimizer):
    # Counterfactual next-vector at the SAME checkpoint and SAME historical Adam moments.
    # First-order action change only; no vector is applied to model or optimizer.
    combinations = {
        "action_only": (1, 0, 0),
        "action_masked": (1, 1, 0),
        "action_causal": (1, 0, 1),
        "full": (1, 1, 1),
    }
    results = {}
    for label, weights in combinations.items():
        norm2 = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
        for index in range(len(parameters)):
            active = [
                g[index] * w
                for g, w in zip(grads.values(), weights, strict=True)
                if w and g[index] is not None
            ]
            if active:
                norm2 += sum(active).double().square().sum()
        norm = math.sqrt(float(norm2))
        clip = min(1.0, 1.0 / (norm + 1e-6))
        dots = torch.zeros(3, device=parameters[0].device, dtype=torch.float64)
        offset = 0
        for group in saved_optimizer["param_groups"]:
            b1, b2 = group["betas"]
            for saved_id in group["params"]:
                index = offset
                offset += 1
                active = [
                    g[index] * w
                    for g, w in zip(grads.values(), weights, strict=True)
                    if w and g[index] is not None
                ]
                # AdamW skips parameters whose gradient is absent (not zero).
                if not active:
                    continue
                g = sum(active) * clip
                old = saved_optimizer["state"].get(saved_id, {})
                first = old["exp_avg"].to(g.device) if "exp_avg" in old else torch.zeros_like(g)
                second = (
                    old["exp_avg_sq"].to(g.device) if "exp_avg_sq" in old else torch.zeros_like(g)
                )
                count = float(old.get("step", 0)) + 1
                first = first * b1 + g * (1 - b1)
                second = second * b2 + g.square() * (1 - b2)
                delta = (
                    -(group["lr"] / (1 - b1**count))
                    * first
                    / ((second / (1 - b2**count)).sqrt() + group["eps"])
                )
                delta = delta - group["lr"] * group["weight_decay"] * parameters[index].detach()
                action = grads["action"][index]
                dots[1] += delta.double().square().sum()
                if action is not None:
                    dots[0] += (action.double() * delta.double()).sum()
                    dots[2] += action.double().square().sum()
        dot, delta2, action2 = dots.cpu().tolist()
        results[label] = {
            "gradient_l2_before_clip": norm,
            "clip_coefficient": clip,
            "delta_l2": math.sqrt(delta2),
            "first_order_action_loss_change": dot,
            "action_gradient_cosine_with_delta": dot / math.sqrt(delta2 * action2)
            if delta2 * action2
            else None,
        }
    return results


def probe(arm, step):
    output = Path(f"outputs/n02-v12/aux-audit-20261016/{arm}{step}")
    output.mkdir(parents=True, exist_ok=False)
    plan = resolve_plan(
        build_parser().parse_args(
            [
                "--config",
                f"configs/encoders/n02_v12/{arm}_n02_robotwin.json",
                "--output",
                str(output / "loader"),
                "--run-id",
                f"readonly-{arm}{step}",
            ]
        )
    )
    trainer = N02V12Trainer(plan, LogTail())
    trainer.build()
    payload = trainer.load_weights_for_probe(
        Path(f"outputs/n02-v12/P2/{arm}-n02-robotwin-r5-s1103/checkpoint_{step:07d}.pt")
    )
    trainer.language = MethodType(legacy_language, trainer)
    trainer.model.train()
    trainer.masked.train()
    parameters = [p for g in trainer.optimizer.param_groups for p in g["params"]]
    names = {id(p): name for name, p in trainer.model.named_parameters()}
    names.update(
        {
            id(p): f"masked.{name}"
            for name, p in trainer.masked.named_parameters()
            if id(p) not in names
        }
    )
    indices = {
        role: [i for i, p in enumerate(parameters) if id(p) in {id(v) for v in values}]
        for role, values in trainer.parameter_groups.items()
    }
    saved = payload["optimizer"]
    assert len(parameters) == sum(len(g["params"]) for g in saved["param_groups"])
    assert [g["name"] for g in trainer.optimizer.param_groups] == [
        g["name"] for g in saved["param_groups"]
    ]
    before = parameter_digest(parameters)
    reports = []
    with forbid_optimizer_steps(trainer.optimizer):
        for number, (start, label) in enumerate(WINDOWS):
            torch.manual_seed(20261016 + number)
            trainer.optimizer.zero_grad(set_to_none=True)
            result = trainer.microbatch(
                start,
                start + 2,
                generator=torch.Generator(device=trainer.device).manual_seed(1104 + number),
            )
            terms = result["loss_tensors"]
            grads = {
                name: torch.autograd.grad(
                    terms[name], parameters, retain_graph=True, allow_unused=True
                )
                for name in ("action", "masked_raw", "causal_raw")
            }
            roles = {role: geometry(grads, positions) for role, positions in indices.items()}
            roles["all"] = geometry(grads, range(len(parameters)))
            qs = [o.q for o in trainer.token_source.last_outputs]
            derivatives = torch.autograd.grad(
                terms["action"], qs, retain_graph=True, allow_unused=True
            )
            pairs = [(q, g) for q, g in zip(qs, derivatives, strict=True) if g is not None]
            chain = torch.autograd.grad(
                [q for q, g in pairs],
                parameters,
                grad_outputs=[g.detach() for q, g in pairs],
                retain_graph=True,
                allow_unused=True,
            )
            q_norm = torch.zeros(3, device="cuda", dtype=torch.float64)
            for a, qg in zip(grads["action"], chain, strict=True):
                if a is not None:
                    no_q = a if qg is None else a - qg
                    q_norm[0] += a.double().square().sum()
                    q_norm[1] += no_q.double().square().sum()
                if qg is not None:
                    q_norm[2] += qg.double().square().sum()
            norms = q_norm.sqrt().cpu().tolist()
            top = sorted(
                (
                    (float(g.norm()), names[id(p)])
                    for p, g in zip(parameters, grads["action"], strict=True)
                    if g is not None
                ),
                reverse=True,
            )[:8]
            record = {
                "label": label,
                "window": result["info"]["window"],
                "terms": result["terms"],
                "roles": roles,
                "virtual_adam": virtual_adam(parameters, grads, saved),
                "action_path_l2": {
                    "full": norms[0],
                    "without_current_q_derivative": norms[1],
                    "via_current_q": norms[2],
                },
                "top_action_gradients": top,
                "current_q": [q.detach().cpu().tolist() for q in qs],
                "current_energy_range": [
                    [float(o.energies.min()), float(o.energies.max())]
                    for o in trainer.token_source.last_outputs
                ],
            }
            reports.append(record)
            write_json(output / f"window{number}.json", record)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "step": step,
                        "window": label,
                        "roles": roles,
                        "action_path": record["action_path_l2"],
                    }
                ),
                flush=True,
            )
            del result, terms, grads, qs, derivatives, pairs, chain
            gc.collect()
            torch.cuda.empty_cache()
    after = parameter_digest(parameters)
    assert before == after and not trainer.optimizer.state
    write_json(
        output / "probe.json",
        {
            "arm": arm,
            "step": step,
            "optimizer_updates": 0,
            "optimizer_state_mutated": False,
            "weights_unchanged": before == after,
            "trainable_sha256": before,
            "language_contract": "r5-legacy14-explicit",
            "limitation": (
                "new fixed flow draws, same checkpoint moments; "
                "not replay of historical spike or training ablation"
            ),
            "windows": reports,
        },
    )
    del trainer, payload, parameters, saved
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("raw", "rgb"), required=True)
    args = parser.parse_args()
    for checkpoint_step in (900,) if args.arm == "raw" else (500, 700):
        probe(args.arm, checkpoint_step)
