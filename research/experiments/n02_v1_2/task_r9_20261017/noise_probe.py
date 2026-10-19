"""Fixed clean sensor buffers x independent sensor draws x independent flow draws."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from owac.runtime.n02_v12_robotwin import Policy, online_window
from owac.runtime.rawvla_full_eval import write_json
from owac.runtime.rawvla_robotwin import normalize_state, unnormalize_actions


@torch.inference_mode()
def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    policy = Policy(Path(args.config), Path(args.checkpoint), root / "loader")
    trainer = policy.trainer

    def forbidden(*a, **k):
        raise RuntimeError("optimizer updates forbidden in noise probe")

    trainer.optimizer.step = forbidden
    language = "Grab the light brown cylindrical roller on the table."
    rows = []
    actions = {}
    captures = Path(args.captures)
    for domain in ("Normal", "Low", "ExtremeLow"):
        source = captures / domain / "sensor_capture.npz"
        with np.load(source, allow_pickle=False) as data:
            state_original = data["state"].copy()
            normalized = normalize_state(state_original, policy.stats["state"])
            lang, lmask, state = trainer.language([language], normalized[None])
            visuals = {}
            pixels = {}
            for variant in ("clean", "raw", "redraw0", "redraw1", "redraw2", "flip"):
                images = [
                    data[f"{'clean' if variant == 'flip' else variant}_{i}"].copy()
                    for i in range(3)
                ]
                if variant == "flip":
                    images = [im[:, ::-1].copy() for im in images]
                pixels[variant] = np.stack(images)
                example = dict(
                    history=[dict(step=0, image=images)],
                    step=0,
                    case_id=f"sensor-{domain}",
                    lang=language,
                    normalized_state=normalized,
                )
                window = online_window(
                    example,
                    policy.cameras,
                    history=policy.history,
                    representation=policy.representation,
                )
                trainer.bridge.reset_episodes()
                visuals[variant] = trainer.token_source(window)
            predictions = {}
            for variant, visual in visuals.items():
                flow_seeds = (
                    (188390828, 188390878, 188390928, 188390978)
                    if variant == "clean"
                    else (188390828,)
                )
                for seed in flow_seeds:
                    generator = torch.Generator(device=trainer.device).manual_seed(seed)
                    noise = torch.randn((1, 50, 32), generator=generator, device=trainer.device)
                    value = trainer.bridge.predict_action_from_visual_tokens(
                        visual, lang, lmask, state, noise=noise, num_steps=10
                    )["control_actions"][0]
                    predictions[f"{variant}-{seed}"] = unnormalize_actions(
                        np.asarray(value), policy.stats["actions"]
                    )
            baseline = predictions["clean-188390828"]
            joints = [*range(6), *range(7, 13)]
            token_base = visuals["clean"].values.float()
            for key, value in predictions.items():
                variant, seed = key.rsplit("-", 1)
                delta = value[:, joints] - baseline[:, joints]
                rows.append(
                    dict(
                        domain=domain,
                        variant=variant,
                        flow_seed=int(seed),
                        joint_action_rms_difference_rad=float(np.mean(delta**2) ** 0.5),
                        first_joint_max_difference_rad=float(np.max(np.abs(delta[0]))),
                        pixel_rms_difference=float(
                            np.mean((pixels[variant] - pixels["clean"]) ** 2) ** 0.5
                        ),
                        token_relative_l2=float(
                            (visuals[variant].values.float() - token_base).norm()
                            / token_base.norm()
                        ),
                        first_target_vs_state_max_rad=float(
                            np.max(np.abs(value[0, joints] - state_original[joints]))
                        ),
                    )
                )
                actions[f"{domain}-{key}"] = value
    np.savez_compressed(root / "actions.npz", **actions)
    write_json(
        root / "noise_probe.json",
        dict(
            checkpoint=args.checkpoint,
            checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            language=language,
            rows=rows,
            optimizer_updates=0,
            new_policy_rollouts=0,
            scope="three initial states; same pre-noise buffer per domain; no closed-loop claim",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--captures", required=True)
    p.add_argument("--output", required=True)
    torch.set_num_threads(4)
    main(p.parse_args())
