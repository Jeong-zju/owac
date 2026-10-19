"""Resume only the two declared r12 fits, without changing their training objective."""

import argparse
import hashlib
import json
from pathlib import Path

from fewshot import EVIDENCE, ROOT, FewShotTrainer

from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import LogTail, build_parser, resolve_plan, validate_run_directory


class ResumedFitTrainer(FewShotTrainer):
    def load_for_resume(self, path):
        result = super().load_for_resume(path)
        # The production trainer verifies its saved prediction before its first
        # update. Attribute that fit to the restored step, preserving fit_0.
        self.current_step = result[0]
        return result


def main(args):
    output = ROOT / f"outputs/n02-v12/P2/raw-fewshot-r12-{args.label}-s1103-a1"
    checkpoint = output / "checkpoint_0000128.pt"
    archive = output / "segments/initial128"
    if not (archive / "report.json").is_file():
        raise ValueError("preserve initial128 evidence before continuing")
    before_rows = [
        json.loads(line) for line in (output / "optimization.jsonl").read_text().splitlines()
    ]
    if [r["step"] for r in before_rows] != list(range(1, 129)):
        raise ValueError("resume accepts exactly128 prior updates, only once")
    initial_fit_hash = hashlib.sha256((output / "fit_0000000.json").read_bytes()).hexdigest()
    plan = resolve_plan(
        build_parser().parse_args(
            [
                "--config",
                str(EVIDENCE / (args.label + "_config.json")),
                "--output",
                str(output),
                "--max-updates",
                "256",
                "--resume",
                str(checkpoint),
            ]
        )
    )
    validate_run_directory(plan, profile_only=False)
    with (output / "resume128.lock").open("x") as lock:
        lock.write("One bounded128-update continuation; preserve this receipt on failure.\n")
    write_json(
        output / "resume128_plan.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "new_updates_max": 128,
            "per_arm_total_max": 256,
            "two_arm_total_max": 512,
            "optimizer_and_cursor": "restore exactly; same objective, LR and data subset",
            "fit_step_metadata_fix": "record restored-step verification as128, preserve fit_0",
            "initial_fit_sha256": initial_fit_hash,
        },
    )
    trainer = ResumedFitTrainer(plan, LogTail())
    trainer.build()
    trainer.configure_subset()
    trainer.plan_stats = json.loads(Path(plan.config["data"]["stats_path"]).read_text())[
        "norm_stats"
    ]
    # Shape/wiring probes must inspect restored weights. train() then restores
    # the complete optimizer/RNG/cursor through the normal resume path.
    payload = trainer.load_weights_for_probe(checkpoint)
    trainer.restore_fixed_calibration(payload["fixed_weight_calibration"])
    trainer._write_json("run_config.json", plan.config)
    trainer.checks = trainer.correctness_checks()
    trainer.train()
    after_rows = [
        json.loads(line) for line in (output / "optimization.jsonl").read_text().splitlines()
    ]
    if [r["step"] for r in after_rows] != list(range(1, 257)):
        raise ValueError("actual optimizer-update receipts differ from the bounded plan")
    if hashlib.sha256((output / "fit_0000000.json").read_bytes()).hexdigest() != initial_fit_hash:
        raise ValueError("resume overwrote the original zero-update fit")
    trainer.checks["final"] = trainer.final
    if "save_resume" in trainer.checks:
        trainer.checks["save_resume"]["measurement"] = (
            "same fixed dev windows/noise/time/precision; teacher-forced velocity tensor, "
            "not a closed-loop action trajectory"
        )
    trainer._write_json("checks.json", trainer.checks)
    trainer._write_json(
        "report.json",
        {
            "run_id": plan.run_id,
            "config": plan.config,
            "overview": trainer.overview,
            "final": trainer.final,
            "initial128_evidence": str(archive),
            "optimizer_updates_total": 256,
            "trained_parameters": {
                name: sum(p.numel() for p in parameters)
                for name, parameters in trainer.parameter_groups.items()
            },
            "training_sample_fit": json.loads((output / "fit_latest.json").read_text()),
            "closed_loop": "not run by this training process",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("one", "three"), required=True)
    main(parser.parse_args())
