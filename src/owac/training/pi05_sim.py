"""Fine-tune the pinned official PyTorch pi05 model and serve its RGB baseline."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from owac.evaluation.baseline_protocol import action_chunk, file_sha256, write_json


def normalization(values):
    import numpy as np

    low, high = np.percentile(values, [1, 99], axis=0)
    return {
        "center": ((low + high) / 2).tolist(),
        "scale": np.where(high - low > 0.002, (high - low) / 2, 1.0).tolist(),
    }


class Episodes:
    def __init__(self, directory, split, horizon):
        import numpy as np

        self.directory, self.horizon = directory, horizon
        manifest = json.loads((directory / "dataset.json").read_text())
        self.episodes = []
        self.index = []
        for item in manifest["episodes"]:
            if item["split"] != split:
                continue
            source = directory / item["path"]
            if file_sha256(source) != item["sha256"]:
                raise ValueError("Dataset checksum mismatch: " + str(source))
            packed = directory / "memmap" / source.stem
            packed.mkdir(parents=True, exist_ok=True)
            with np.load(source, allow_pickle=False) as arrays:
                for key in ("external", "wrist", "state", "actions"):
                    if not (packed / (key + ".npy")).exists():
                        np.save(packed / (key + ".npy"), arrays[key], allow_pickle=False)
            episode = {
                k: np.load(packed / (k + ".npy"), mmap_mode="r", allow_pickle=False)
                for k in ("external", "wrist", "state", "actions")
            }
            self.index.extend((len(self.episodes), j) for j in range(len(episode["actions"])))
            self.episodes.append(episode)
        if not self.index:
            raise ValueError("No episodes for split " + split)

    def __len__(self):
        return len(self.index)

    def get(self, index):
        episode_index, frame = self.index[index]
        episode = self.episodes[episode_index]
        return {k: episode[k][frame].copy() for k in ("external", "wrist", "state")} | {
            "actions": action_chunk(episode["actions"], frame, self.horizon).copy()
        }


def model_config(protocol, dtype="float32"):
    from openpi.models.pi0_config import Pi0Config

    return Pi0Config(
        pi05=True,
        action_horizon=protocol["action_horizon"],
        discrete_state_input=True,
        dtype=dtype,
        pytorch_compile_mode=None,
    )


def make_model(protocol, weights, dtype="float32"):
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    class EvaluationAwarePI05(PI0Pytorch):
        # The upstream forward unconditionally requests training augmentation.
        # Held-out losses must instead use clean images and deterministic flow noise.
        def _preprocess_observation(self, observation, *, train=True):
            return super()._preprocess_observation(observation, train=train and self.training)

    model = EvaluationAwarePI05(model_config(protocol, dtype=dtype))
    safetensors.torch.load_model(model, weights, strict=True)
    return model


def batch(items, stats, tokenizer, device):
    import numpy as np
    import torch
    from openpi.models.model import Observation

    state = np.stack([item["state"] for item in items])
    state = (state - stats["state"]["center"]) / stats["state"]["scale"]
    state = np.pad(state.astype(np.float32), ((0, 0), (0, 24)))
    tokens, masks = zip(
        *(tokenizer.tokenize("Put the red mug on the left plate.", row) for row in state),
        strict=True,
    )
    external = np.stack([item["external"] for item in items])
    wrist = np.stack([item["wrist"] for item in items])
    tensor = lambda value: torch.from_numpy(np.asarray(value).copy()).to(device)  # noqa: E731
    obs = Observation.from_dict(
        {
            "image": {
                "base_0_rgb": tensor(external),
                "left_wrist_0_rgb": tensor(wrist),
                "right_wrist_0_rgb": tensor(np.zeros_like(external)),
            },
            "image_mask": {
                "base_0_rgb": tensor(np.ones(len(items), bool)),
                "left_wrist_0_rgb": tensor(np.ones(len(items), bool)),
                "right_wrist_0_rgb": tensor(np.zeros(len(items), bool)),
            },
            "state": tensor(state),
            "tokenized_prompt": tensor(np.stack(tokens)),
            "tokenized_prompt_mask": tensor(np.stack(masks)),
        }
    )
    if "actions" not in items[0]:
        return obs, None
    actions = np.stack([item["actions"] for item in items])
    actions = (actions - stats["actions"]["center"]) / stats["actions"]["scale"]
    actions = np.pad(actions.astype(np.float32), ((0, 0), (0, 0), (0, 25)))
    return obs, tensor(actions)


def validation_loss(model, dataset, stats, tokenizer, recipe):
    import numpy as np
    import torch

    model.eval()
    rng = np.random.default_rng(recipe["validation_noise_seed"])
    generator = torch.Generator(device="cuda").manual_seed(recipe["validation_noise_seed"])
    indices = rng.choice(
        len(dataset), min(recipe["validation_samples"], len(dataset)), replace=False
    )
    losses, counts = 0.0, 0
    with torch.no_grad():
        for offset in range(0, len(indices), recipe["batch_size"]):
            items = [dataset.get(i) for i in indices[offset : offset + recipe["batch_size"]]]
            obs, actions = batch(items, stats, tokenizer, "cuda")
            noise = torch.randn(actions.shape, generator=generator, device="cuda")
            # Fixed time samples with the same beta(1.5, 1) law as upstream.
            flow_time = torch.rand((len(items),), generator=generator, device="cuda")
            flow_time = flow_time.pow(1 / 1.5) * 0.999 + 0.001
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(obs, actions, noise=noise, time=flow_time)[..., :7].mean()
            losses += float(loss) * len(items)
            counts += len(items)
    model.train()
    return losses / counts


def train(args, protocol):
    import numpy as np
    import safetensors.torch
    import torch
    from openpi.models.tokenizer import PaligemmaTokenizer

    recipe = protocol["training"]
    manifest = json.loads((args.data / "dataset.json").read_text())
    if args.steps is None and (
        manifest["status"] != "complete"
        or manifest["protocol_sha256"] != file_sha256(args.protocol)
    ):
        raise ValueError("Formal training requires a complete dataset from this exact protocol")
    seed = recipe["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(8)
    dataset = Episodes(args.data, "train", protocol["action_horizon"])
    held_out = Episodes(args.data, "validation", protocol["action_horizon"])
    if args.steps is None and (
        len(dataset.episodes) != protocol["train_episodes"]
        or len(held_out.episodes) != protocol["validation_episodes"]
    ):
        raise ValueError("Formal training requires all prescribed train/validation episodes")
    stats = {
        key: normalization(np.concatenate([ep[key] for ep in dataset.episodes]))
        for key in ("state", "actions")
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "normalization.json", stats)
    write_json(args.output / "protocol.json", protocol)
    model = make_model(protocol, args.weights / "model.safetensors").cuda()
    model.gradient_checkpointing_enable()
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=recipe["peak_lr"],
        weight_decay=recipe["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )
    tokenizer = PaligemmaTokenizer(200)
    report = {
        "status": "training",
        "smoke_test": args.steps is not None,
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_manifest_sha256": file_sha256(args.data / "dataset.json"),
        "initial_model_sha256": file_sha256(args.weights / "model.safetensors"),
        "train_frames": len(dataset),
        "validation_frames": len(held_out),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "training_precision": "FP32 parameters/optimizer/loss, BF16 autocast forward",
        "ema": False,
        "loss": "official action flow MSE, mean over seven actual action dimensions",
        "validation": "no image augmentation, fixed 256 frames/noise/time, episode-disjoint",
    }
    write_json(args.output / "training.json", report)
    started = time.perf_counter()
    probe_before = model.action_out_proj.weight.detach().cpu().clone()
    model.eval()
    obs, actions = batch(
        [held_out.get(i) for i in range(recipe["batch_size"])], stats, tokenizer, "cuda"
    )
    noise = torch.randn_like(actions)
    flow_time = torch.full((recipe["batch_size"],), 0.5, device="cuda")
    with torch.no_grad():
        reference = float(model(obs, actions, noise=noise, time=flow_time)[..., :7].mean())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mixed = float(model(obs, actions, noise=noise, time=flow_time)[..., :7].mean())
    report["precision_check"] = {
        "float32_loss": reference,
        "autocast_loss": mixed,
        "relative_loss_difference": abs(reference - mixed) / max(abs(reference), 1e-8),
    }
    if report["precision_check"]["relative_loss_difference"] > 0.05:
        raise FloatingPointError("BF16/FP32 precision check exceeded 5% loss tolerance")
    model.train()
    initial_loss = validation_loss(model, held_out, stats, tokenizer, recipe)
    report["initial_validation_loss"] = initial_loss
    best, significant_best, stale = math.inf, math.inf, 0
    maximum_steps = args.steps or recipe["maximum_steps"]
    with (args.output / "curve.jsonl").open("w") as stream:
        for step in range(1, maximum_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            if step <= recipe["warmup_steps"]:
                lr = recipe["peak_lr"] * step / recipe["warmup_steps"]
            else:
                progress = (step - recipe["warmup_steps"]) / (
                    recipe["maximum_steps"] - recipe["warmup_steps"]
                )
                lr = recipe["minimum_lr"] + 0.5 * (recipe["peak_lr"] - recipe["minimum_lr"]) * (
                    1 + math.cos(math.pi * progress)
                )
            for group in optimizer.param_groups:
                group["lr"] = lr
            training_loss = 0.0
            for _ in range(recipe["gradient_accumulation"]):
                indices = rng.integers(0, len(dataset), recipe["batch_size"])
                obs, actions = batch([dataset.get(i) for i in indices], stats, tokenizer, "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(obs, actions)[..., :7].mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training loss")
                (loss / recipe["gradient_accumulation"]).backward()
                training_loss += float(loss.detach()) / recipe["gradient_accumulation"]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, recipe["gradient_clip"]))
            if not math.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            if step == 1:
                report["first_step_action_projection_max_update"] = float(
                    (model.action_out_proj.weight.detach().cpu() - probe_before).abs().max()
                )
                if report["first_step_action_projection_max_update"] <= 0:
                    raise RuntimeError("Optimizer failed to update action projection")
            row = {
                "step": step,
                "train_flow_mse": training_loss,
                "gradient_norm": grad_norm,
                "lr": lr,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if step % recipe["validation_interval"] == 0 or step == maximum_steps:
                value = validation_loss(model, held_out, stats, tokenizer, recipe)
                row["validation_flow_mse"] = value
                if value < best:
                    best = value
                    report["best_step"] = step
                    temporary = args.output / "model.tmp.safetensors"
                    safetensors.torch.save_model(model, temporary)
                    temporary.replace(args.output / "model.safetensors")
                if value < significant_best * (1 - recipe["relative_improvement"]):
                    significant_best, stale = value, 0
                else:
                    stale += 1
                report.update(
                    completed_steps=step,
                    best_validation_loss=best,
                    latest_validation_loss=value,
                    plateau_checks=stale,
                    wall_seconds=time.perf_counter() - started,
                    torch_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                    torch_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                )
                write_json(args.output / "training.json", report)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if step % 10 == 0 or "validation_flow_mse" in row:
                print("TRAIN_STEP", json.dumps(row), flush=True)
            if step >= recipe["minimum_steps"] and stale >= recipe["plateau_patience"]:
                report["stop_reason"] = "held-out loss plateau"
                break
    report.update(
        status="complete",
        completed_steps=step,
        wall_seconds=time.perf_counter() - started,
        selected_model_sha256=file_sha256(args.output / "model.safetensors"),
        stop_reason=report.get("stop_reason", "maximum prescribed steps"),
        convergence_observed=stale >= recipe["plateau_patience"],
    )
    write_json(args.output / "training.json", report)


def serve(args, protocol):
    import numpy as np
    import torch
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    torch.set_num_threads(8)
    stats = json.loads((args.weights / "normalization.json").read_text())
    model = make_model(protocol, args.weights / "model.safetensors", dtype="bfloat16").cuda()
    model.eval()
    tokenizer = PaligemmaTokenizer(200)
    metadata = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(args.protocol),
        "model_sha256": file_sha256(args.weights / "model.safetensors"),
        "training": json.loads((args.weights / "training.json").read_text()),
        "input": "two RGB views + current proprioception + fixed task prompt",
        "privileged_state": False,
        "flow_steps": protocol["flow_steps"],
    }

    class RGBPolicy:
        def infer(self, obs):
            observation, _ = batch([obs], stats, tokenizer, "cuda")
            # Each episode/inference has its own noise, independent of prior runs.
            generator = torch.Generator(device="cuda").manual_seed(
                int(obs["seed"]) * 10000 + int(obs["inference_index"])
            )
            noise = torch.randn(
                (1, protocol["action_horizon"], 32), generator=generator, device="cuda"
            )
            with torch.inference_mode():
                prediction = (
                    model.sample_actions(
                        "cuda", observation, noise=noise, num_steps=protocol["flow_steps"]
                    )[0, :, :7]
                    .cpu()
                    .numpy()
                )
            actions = prediction * stats["actions"]["scale"] + stats["actions"]["center"]
            return {"actions": np.asarray(actions, dtype=np.float32)}

    WebsocketPolicyServer(
        RGBPolicy(), host="127.0.0.1", port=args.port, metadata=metadata
    ).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "serve"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.mode == "train":
        if args.data is None or args.output is None:
            parser.error("train requires --data and --output")
        train(args, protocol)
    else:
        serve(args, protocol)


if __name__ == "__main__":
    main()
