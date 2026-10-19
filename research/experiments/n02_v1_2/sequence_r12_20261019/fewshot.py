"""Bounded one/three-episode RAW fit test, using the production trainer and sources."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from owac.backends.low_rank import restore_adapters
from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import (
    LogTail,
    N02V12Trainer,
    anchor_action_targets,
    build_parser,
    build_source,
    resolve_plan,
    validate_run_directory,
)
from owac.training.n02_v12_task import BalancedBlocks, source_records

ROOT = Path("/workspace/owac/worktrees/n02-measurement-completion")
EVIDENCE = ROOT / "research/experiments/n02_v1_2/sequence_r12_20261019"
SOURCE = ROOT / "outputs/n02-v12/P2/raw-task-r11-task-s1103-a1/checkpoint_0000850.pt"
SOURCE_SHA = "a660f0a0c6dc3d7006f1cda391de27c2b35d31fb7b1d51163e7f480e16b0068c"
SEEDS = (188390828, 188390878, 188390928, 188390978)
JOINTS = [*range(6), *range(7, 13)]


def prepare():
    config = json.loads((EVIDENCE.parent / "training_r11_20261017/task_config.json").read_text())
    source = build_source(config, "train", history=8, batch_size=12)
    records = source_records(source)
    paths = sorted({r["path"] for r in records if r["light"] == "Normal"})[:3]
    if len(paths) != 3 or any("episode_000000." in p for p in paths):
        raise ValueError("three training Normal episodes, excluding seed0, are required")
    for label, count in (("one", 1), ("three", 3)):
        c = json.loads(json.dumps(config))
        c["run_id"] = f"raw-fewshot-r12-{label}-s1103-a1"
        c["initialisation"] = {
            "kind": "fewshot_r12",
            "checkpoint": str(SOURCE),
            "checkpoint_sha256": SOURCE_SHA,
            "optimizer": "fresh",
        }
        c["optim"].update(lr=1e-5, lr_head=5e-5, lr_backend_adapter=1e-4, warmup_updates=8)
        c["loop"].update(max_updates=128, validate_every=64, save_every=64)
        c["loop"]["profile_note"] = (
            "r12 fewshot fit, source RAW850; fresh optimizer/cursor; "
            "128 initial updates, at most256 per arm"
        )
        c["runtime"]["gpu_note"] = f"GPU{0 if label == 'one' else 1} dedicated to this arm"
        c["task_adaptation"]["fewshot"] = {
            "episode_paths": paths[:count],
            "selection": "lexicographically first Normal train episodes, no outcome selection",
            "fit_gate": {
                "joint_rms_deg_max": 3.0,
                "gripper_mae_max": 0.05,
                "gripper_max_error_max": 0.15,
                "first_joint_max_deg_max": 5.0,
                "flow_seeds": list(SEEDS),
            },
            "scope": "training-sample fit only; all5 dev episodes remain independent",
        }
        destination = EVIDENCE / (label + "_config.json")
        if destination.exists() and json.loads(destination.read_text()) != c:
            raise FileExistsError(destination)
        write_json(destination, c)
    write_json(
        EVIDENCE / "fewshot_selection.json",
        {
            "partition": "train",
            "normal_episode_paths": paths,
            "anchor_counts": {
                p: sum(r["path"] == p and r["valid"] > 0 for r in records) for p in paths
            },
            "optimizer_updates": 0,
            "dev_and_calibration_used_for_updates": False,
        },
    )
    print(json.dumps({"selected_training_paths": paths}), flush=True)


class FewShotTrainer(N02V12Trainer):
    def configure_subset(self):
        records = source_records(self.sources["train"])
        selected = set(self.plan.config["task_adaptation"]["fewshot"]["episode_paths"])
        sampler = BalancedBlocks(records, self.plan.batch_size, self.plan.seed)
        sampler.groups = {
            key: [(a, b) for a, b in group if records[a]["path"] in selected]
            for key, group in sampler.groups.items()
        }
        sampler.groups = {k: v for k, v in sampler.groups.items() if v}
        sampler.keys = sorted(sampler.groups)
        sampler.covered = [
            i for group in sampler.groups.values() for a, b in group for i in range(a, b)
        ]
        expected = {r["index"] for r in records if r["path"] in selected and r["valid"] > 0}
        if set(sampler.covered) != expected or len(sampler.covered) != len(expected):
            raise ValueError("fewshot selected-anchor coverage mismatch")
        self.task_sampler = sampler
        receipt = sampler.record()
        receipt.update(
            fewshot_episode_paths=sorted(selected),
            complete_valid_coverage_scope="declared selected episodes only",
        )
        self._write_json("task_sampler.json", receipt)
        self.fit_indices = []
        for path in sorted(selected):
            episode = [r for r in records if r["path"] == path and r["valid"] >= 10]
            n = episode[0]["frames"]
            for t in (0, 10, n // 2, n - 11):
                self.fit_indices.append(min(episode, key=lambda r: (abs(r["t"] - t), r["index"])))
        self._write_json("fit_windows.json", self.fit_indices)
        # Training prefetch owns a mutable episode cache. Fit evaluation must use
        # a separate source rather than racing that worker on the same cache.
        self.fit_source = build_source(
            self.plan.config, "train", history=self.plan.history, batch_size=self.plan.batch_size
        )

    def transfer(self):
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError("source checkpoint changed")
        payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
        for key in ("model", "representation", "data", "policy"):
            if payload["config"][key] != self.plan.config[key]:
                raise ValueError("undeclared transfer change: " + key)
        if payload["identity"] != self.identity:
            raise ValueError("frozen data identity changed")
        self.model.load_state_dict(payload["model"], strict=True)
        masked = self.masked.state_dict()
        masked.update(payload["masked_branch"])
        self.masked.load_state_dict(masked, strict=True)
        for module, state in zip(
            self.bridge.bridge_modules(), payload["bridge_projection"], strict=True
        ):
            module.load_state_dict(state, strict=True)
        restore_adapters(self.backend_adapters, payload["backend_adapters"])
        if self.optimizer.state:
            raise ValueError("r12 must start with a fresh optimizer")
        self.restore_fixed_calibration(payload["fixed_weight_calibration"])
        self._write_json(
            "initialisation.json",
            {
                "kind": "fewshot_r12",
                "source": str(SOURCE),
                "source_sha256": SOURCE_SHA,
                "source_step": 850,
                "optimizer": "fresh",
                "data_cursor": 0,
                "model_and_backend_adapters": "strict complete state transfer",
            },
        )

    @torch.no_grad()
    def fit_test(self):
        was_training = self.model.training
        self.model.eval()
        std = np.asarray(self.plan_stats["actions"]["std"])
        mean = np.asarray(self.plan_stats["actions"]["mean"])
        rows = []
        for identity in self.fit_indices:
            index = identity["index"]
            batch = self.window(self.fit_source, index, index + 1)
            targets = self.fit_source.targets(index, index + 1)
            prompts = self.task_prompts(
                [str(anchor["relative_path"]) for anchor in batch.anchors],
                str(self.plan.config["data"]["data_root"]),
            )
            raw, valid = anchor_action_targets(targets, 1)
            target = raw[0] * std + mean
            proprio = self._anchor_proprio(batch)
            lang, mask, state = self.language(prompts, proprio)
            self.bridge.reset_episodes()
            visual = self.bridge.visual_tokens(batch)
            for seed in SEEDS:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                noise = torch.randn((1, 50, 32), generator=generator, device=self.device)
                pred = self.bridge.predict_action_from_visual_tokens(
                    visual, lang, mask, state, noise=noise, num_steps=10
                )["control_actions"][0]
                physical = np.asarray(pred) * std + mean
                error = physical[valid[0]] - target[valid[0]]
                rows.append(
                    {
                        "index": index,
                        "path": identity["path"],
                        "t": identity["t"],
                        "seed": seed,
                        "joint_rms_deg": float(
                            np.square(error[:, JOINTS]).mean() ** 0.5 * 180 / np.pi
                        ),
                        "first_joint_max_deg": float(np.abs(error[0, JOINTS]).max() * 180 / np.pi),
                        "gripper_mae": float(np.abs(error[:, [6, 13]]).mean()),
                        "gripper_max_error": float(np.abs(error[:, [6, 13]]).max()),
                    }
                )
        gate = self.plan.config["task_adaptation"]["fewshot"]["fit_gate"]
        maxima = {
            key: max(r[key] for r in rows)
            for key in ("joint_rms_deg", "first_joint_max_deg", "gripper_mae", "gripper_max_error")
        }
        passed = all(maxima[key] <= gate[key + "_max"] for key in maxima)
        step = int(getattr(self, "current_step", 0))
        record = {
            "step": step,
            "passed": passed,
            "scope": "training-sample fit, not closed loop",
            "limits": gate,
            "maxima": maxima,
            "rows": rows,
        }
        self._write_json(f"fit_{step:07d}.json", record)
        self._write_json("fit_latest.json", record)
        self.model.train(was_training)
        self.log("training sample fit: " + json.dumps({"step": step, "passed": passed, **maxima}))
        return record

    def evaluate_dev(self, *args, **kwargs):
        result = super().evaluate_dev(*args, **kwargs)
        result["fewshot_fit"] = self.fit_test()
        return result


def train(args):
    config = EVIDENCE / (args.label + "_config.json")
    output = ROOT / f"outputs/n02-v12/P2/raw-fewshot-r12-{args.label}-s1103-a1"
    argv = ["--config", str(config), "--output", str(output), "--max-updates", str(args.updates)]
    if args.resume:
        argv += ["--resume", args.resume]
    plan = resolve_plan(build_parser().parse_args(argv))
    validate_run_directory(plan, profile_only=False)
    if not args.resume:
        plan.output.mkdir(parents=True, exist_ok=False)
    trainer = FewShotTrainer(plan, LogTail())
    trainer.build()
    trainer.configure_subset()
    trainer.plan_stats = json.loads(Path(plan.config["data"]["stats_path"]).read_text())[
        "norm_stats"
    ]
    if not args.resume:
        trainer.transfer()
    trainer._write_json("run_config.json", plan.config)
    trainer.checks = trainer.correctness_checks()
    trainer.train()
    trainer.checks["final"] = trainer.final
    trainer._write_json("checks.json", trainer.checks)
    trainer._write_json(
        "report.json",
        {
            "run_id": plan.run_id,
            "config": plan.config,
            "overview": trainer.overview,
            "final": trainer.final,
            "trained_parameters": {
                name: sum(p.numel() for p in parameters)
                for name, parameters in trainer.parameter_groups.items()
            },
            "training_sample_fit": json.loads((trainer.run_dir / "fit_latest.json").read_text()),
            "closed_loop": "not run by this training process",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--label", choices=("one", "three"), default="one")
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if not 1 <= args.updates <= 256:
        raise ValueError("per-arm budget must be 1..256 updates")
    prepare() if args.command == "prepare" else train(args)
