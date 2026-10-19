"""Zero-update fit diagnostic: cached/full velocity parity and Euler resolution.

Teacher-forced velocity checks use labels only inside this diagnostic. They are
never policy inputs, optimizer updates, or evidence of closed-loop success.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from fewshot import EVIDENCE, JOINTS, ROOT, SEEDS, FewShotTrainer

from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import LogTail, anchor_action_targets, build_parser, resolve_plan


@torch.no_grad()
def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = EVIDENCE / (args.label + "_config.json")
    plan = resolve_plan(
        build_parser().parse_args(["--config", str(config), "--output", str(output)])
    )
    trainer = FewShotTrainer(plan, LogTail())
    trainer.build()
    trainer.configure_subset()
    payload = trainer.load_weights_for_probe(Path(args.checkpoint))
    trainer.restore_fixed_calibration(payload["fixed_weight_calibration"])
    trainer.model.eval()
    stats = json.loads(Path(plan.config["data"]["stats_path"]).read_text())["norm_stats"]
    std, mean = np.asarray(stats["actions"]["std"]), np.asarray(stats["actions"]["mean"])
    bridge = trainer.bridge
    solver_rows, parity_rows, arrays = [], [], {}

    def metrics(normalized, target, valid):
        physical = np.asarray(normalized) * std + mean
        error = physical[valid] - target[valid]
        if not np.isfinite(error).all():
            raise ValueError("nonfinite diagnostic action")
        return {
            "joint_rms_deg": float(np.square(error[:, JOINTS]).mean() ** 0.5 * 180 / np.pi),
            "first_joint_max_deg": float(np.abs(error[0, JOINTS]).max() * 180 / np.pi),
            "gripper_mae": float(np.abs(error[:, [6, 13]]).mean()),
            "gripper_max_error": float(np.abs(error[:, [6, 13]]).max()),
        }

    for window, identity in enumerate(trainer.fit_indices):
        index = identity["index"]
        batch = trainer.window(trainer.fit_source, index, index + 1)
        raw, valid = anchor_action_targets(trainer.fit_source.targets(index, index + 1), 1)
        target = raw[0] * std + mean
        prompts = trainer.task_prompts(
            [str(a["relative_path"]) for a in batch.anchors], plan.config["data"]["data_root"]
        )
        lang, mask, state = trainer.language(prompts, trainer._anchor_proprio(batch))
        bridge.reset_episodes()
        visual = bridge.visual_tokens(batch)
        arrays[f"target_{window}"] = target
        arrays[f"valid_{window}"] = valid[0]
        for seed in SEEDS:
            generator = torch.Generator(device=trainer.device).manual_seed(seed)
            noise = torch.randn((1, 50, 32), generator=generator, device=trainer.device)
            for steps in (10, 20, 50):
                prediction = bridge.predict_action_from_visual_tokens(
                    visual, lang, mask, state, noise=noise, num_steps=steps
                )["control_actions"][0]
                solver_rows.append(
                    {
                        "window": window,
                        "path": identity["path"],
                        "t": identity["t"],
                        "seed": seed,
                        "denoise_steps": steps,
                        **metrics(prediction, target, valid[0]),
                    }
                )
                arrays[f"sample_{window}_{seed}_{steps}"] = prediction * std + mean

        # Compare identical x_t and time through joint training and cached
        # inference. The labels construct x_t only for this offline diagnostic.
        generator = torch.Generator(device=trainer.device).manual_seed(SEEDS[0])
        noise = torch.randn((1, 50, 32), generator=generator, device=trainer.device)
        padded = bridge.prepare_control_targets(raw, device=trainer.device)
        for t in (1.0, 0.5, 0.1):
            time = torch.tensor([t], device=trainer.device)
            full = bridge.forward_from_visual_tokens(
                visual,
                lang,
                mask,
                state,
                padded,
                noise=noise,
                time=time,
                return_debug=True,
                action_mask=valid,
            )
            prefix = bridge.embed_prefix_from_visual_tokens(visual, lang, mask)
            bridge.base.vlm.language_model.config._attn_implementation = "eager"
            cache = bridge.base.vlm.forward_prefix(
                inputs_embeds=prefix.embeds,
                attention_mask=bridge.mask_4d(bridge.mask_2d(prefix.pad_masks, prefix.att_masks)),
                position_ids=torch.cumsum(prefix.pad_masks, dim=1) - 1,
                past_key_values=None,
                use_cache=True,
            ).past_key_values
            cached = bridge.denoise_step(state, prefix.pad_masks, cache, full["x_t"], time)
            delta = (cached.float() - full["v_t"].float()).cpu().numpy()
            endpoint = (full["x_t"] - t * full["v_t"]).float().cpu().numpy()[0, :, :14]
            parity_rows.append(
                {
                    "window": window,
                    "path": identity["path"],
                    "t": identity["t"],
                    "flow_time": t,
                    "cached_full_velocity_max_abs": float(np.abs(delta).max()),
                    "cached_full_velocity_rms": float(np.square(delta).mean() ** 0.5),
                    "teacher_endpoint": metrics(endpoint, target, valid[0]),
                }
            )
        print(json.dumps({"completed_fit_windows": window + 1}), flush=True)

    gate = plan.config["task_adaptation"]["fewshot"]["fit_gate"]
    maxima_by_solver = {}
    for steps in (10, 20, 50):
        rows = [r for r in solver_rows if r["denoise_steps"] == steps]
        maxima = {
            key: max(r[key] for r in rows)
            for key in ("joint_rms_deg", "first_joint_max_deg", "gripper_mae", "gripper_max_error")
        }
        maxima_by_solver[str(steps)] = {
            "maxima": maxima,
            "passed_same_absolute_fit_limits": all(
                maxima[key] <= gate[key + "_max"] for key in maxima
            ),
        }
    checkpoint = Path(args.checkpoint)
    report = {
        "optimizer_updates": 0,
        "new_closed_loops": 0,
        "scope": "training-sample diagnostic; no policy execution or teacher supervision",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": payload["step"],
        "solver_summaries": maxima_by_solver,
        "solver_rows": solver_rows,
        "parity_rows": parity_rows,
        "limits": gate,
    }
    np.savez_compressed(output / "actions.npz", **arrays)
    write_json(output / "report.json", report)
    print(json.dumps({"solver_summaries": maxima_by_solver}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("one", "three"), default="one")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
