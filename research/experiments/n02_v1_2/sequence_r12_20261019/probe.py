"""Zero-update factorization of flow noise, visual observation and proprioception."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from owac.runtime.rawvla_full_eval import write_json

ROOT = Path("/workspace/owac/worktrees/n02-measurement-completion")
EVIDENCE = ROOT / "research/experiments/n02_v1_2/sequence_r12_20261019"
SEEDS = (188390828, 188390878, 188390928, 188390978)
JOINTS = [*range(6), *range(7, 13)]


def differences(value, before):
    delta = value[:, JOINTS] - before[:, JOINTS]
    return dict(
        first_joint_max_deg=float(np.abs(delta[0]).max() * 180 / np.pi),
        first10_joint_rms_deg=float(np.square(delta[:10]).mean() ** 0.5 * 180 / np.pi),
        full_joint_rms_deg=float(np.square(delta).mean() ** 0.5 * 180 / np.pi),
        first_gripper_max=float(np.abs(value[0, [6, 13]] - before[0, [6, 13]]).max()),
    )


@torch.inference_mode()
def raw_probe():
    from owac.runtime.n02_v12_robotwin import Policy, online_window
    from owac.runtime.rawvla_robotwin import normalize_state, unnormalize_actions

    output = EVIDENCE / "raw_probe_a1"
    output.mkdir(exist_ok=False)
    config = EVIDENCE.parent / "training_r11_20261017/task_config.json"
    checkpoint = ROOT / "outputs/n02-v12/P2/raw-task-r11-task-s1103-a1/checkpoint_0000850.pt"
    policy = Policy(config, checkpoint, output / "loader")
    trainer = policy.trainer

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("no optimizer updates in stage 1")

    trainer.optimizer.step = forbidden
    captured = ROOT / "outputs/n02-v12/representation-r8-rollouts/r11-task850-normal-a1/policy"
    with np.load(captured / "r11-task850-normal-a1_first_input.npz") as data:
        original_state = data["state"].copy()
        images = [data["raw_" + str(i)].copy() for i in range(3)]
    normalized = normalize_state(original_state, policy.stats["state"])
    language = "Grab the light brown cylindrical roller on the table."
    base = dict(
        history=[dict(step=0, image=images)],
        step=0,
        case_id="r12-live-initial",
        lang=language,
        normalized_state=normalized,
    )
    visuals = {}
    for label in ("live", "flip", "zero"):
        example = copy.deepcopy(base)
        if label == "flip":
            example["history"][0]["image"] = [x[:, ::-1].copy() for x in images]
        elif label == "zero":
            example["history"][0]["image"] = [np.zeros_like(x) for x in images]
        window = online_window(
            example, policy.cameras, history=policy.history, representation=policy.representation
        )
        trainer.bridge.reset_episodes()
        visuals[label] = trainer.bridge.visual_tokens(window)
    lang, mask, state = trainer.language([language], normalized[None])
    arrays = {}
    rows = []
    noises = {}
    for seed in SEEDS:
        generator = torch.Generator(device=trainer.device).manual_seed(seed)
        noises[seed] = torch.randn((1, 50, 32), generator=generator, device=trainer.device)

    def sample(visual, tokens, masks, proprio, noise):
        value = trainer.bridge.predict_action_from_visual_tokens(
            visual, tokens, masks, proprio, noise=noise, num_steps=10
        )["control_actions"][0]
        return unnormalize_actions(value, policy.stats["actions"])

    for label, visual in visuals.items():
        for seed in SEEDS if label == "live" else SEEDS[:1]:
            value = sample(visual, lang, mask, state, noises[seed])
            arrays[f"{label}-{seed}"] = value
            baseline = arrays["live-188390828"]
            rows.append(dict(label=label, flow_seed=seed, **differences(value, baseline)))
    repeat = sample(visuals["live"], lang, mask, state, noises[SEEDS[0]])
    repeat_difference = float(np.max(np.abs(repeat - arrays["live-188390828"])))
    windows = trainer.dev_batch()
    candidates = trainer.task_dev_manifest["rows"]
    opening_index = next(
        i
        for i, row in enumerate(candidates)
        if row["light"] == "Normal" and row["phase"] == "opening"
    )
    episode = candidates[opening_index]["path"]
    approach_index = next(
        i
        for i, row in enumerate(candidates)
        if row["path"] == episode and row["phase"] == "approach"
    )
    dev = []
    for index in (opening_index, approach_index):
        batch, _, prompts = windows[index]
        proprio = trainer._anchor_proprio(batch)
        tokens, masks, st = trainer.language(prompts, proprio)
        trainer.bridge.reset_episodes()
        visual = trainer.bridge.visual_tokens(batch)
        dev.append((visual, tokens, masks, st))
    for vision_index, state_index in ((0, 0), (1, 0), (0, 1), (1, 1)):
        visual = dev[vision_index][0]
        tokens, masks, proprio = dev[state_index][1:]
        for seed in SEEDS[:2]:
            value = sample(visual, tokens, masks, proprio, noises[seed])
            label = f"vision{vision_index}-state{state_index}"
            arrays[f"{label}-{seed}"] = value
            before = arrays.get("vision0-state0-188390828", value)
            rows.append(dict(label=label, flow_seed=seed, **differences(value, before)))
    report = dict(
        optimizer_updates=0,
        new_closed_loops=0,
        repeat_max_difference=repeat_difference,
        rows=rows,
        observation_scope="one real live initial frame and one dev episode t0/t10",
        dev_identities=[candidates[i] for i in (opening_index, approach_index)],
        checkpoint=str(checkpoint),
    )
    np.savez_compressed(output / "actions.npz", **arrays)
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)


@torch.inference_mode()
def native_probe():
    from owac.runtime.rawvla_robotwin import Policy

    output = EVIDENCE / "native_probe_a1"
    output.mkdir(exist_ok=False)
    root = Path("/workspace/owac/rawvla")
    policy = Policy(
        root / "checkpoints/pi05_robotwin",
        root / "source/openpi_converted_protocol/paligemma_tokenizer.model",
        output / "policy",
    )
    source = ROOT / "outputs/n02-v12/repair-r11-20261017/native-original-a1/policy/input_00000.npz"
    with np.load(source) as data:
        initial = data["state"].copy()
        images = [data[f"camera_{i}"].copy() for i in range(3)]
    rows, arrays = [], {}
    for seed in SEEDS:
        example = dict(
            state=initial,
            image=images,
            flow_seed=seed,
            lang="Grab the light brown cylindrical roller on the table.",
            case_id="r12-native-initial",
        )
        value = policy.predict_action([example])["actions"]
        arrays[str(seed)] = value
        rows.append(dict(flow_seed=seed, **differences(value, arrays[str(SEEDS[0])])))
    report = dict(
        optimizer_updates=0,
        new_closed_loops=0,
        rows=rows,
        scope="native RGB reference only; never RAW capability or supervision",
    )
    np.savez_compressed(output / "actions.npz", **arrays)
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("raw", "native"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw_probe() if args.kind == "raw" else native_probe()
