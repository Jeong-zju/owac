"""Thin, import-safe CLI for the RAW-stem distillation engineering paths.

Heavy imports happen in handlers.  Without valid assets commands return a
structured unavailable/blocked result; with a registered fixture context the
same callable engine paths run under explicit fixture evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

CONFIG_DEFAULT = "configs/experiments/raw_stem_siglip_distillation_v1.json"


def _emit(payload: dict[str, Any], *, exit_code: int = 0) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return int(exit_code)


def _config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"config not found: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    return data


def _asset_report(config: dict[str, Any]) -> dict[str, Any]:
    teacher = config.get("teacher_requirement", {})
    assets = config.get("assets", {})
    missing = {
        "learned_teacher": assets.get("teacher_frontend_checkpoint") is None,
        "teacher_distillation_ready": not bool(teacher.get("distillation_ready")),
        "policy_checkpoint": assets.get("checkpoint") is None,
        "paired_manifest": assets.get("paired_manifest") is None,
        "stage0_parity": assets.get("stage0_parity") is None,
    }
    return {
        "schema": "owac.raw-distillation.assets.v1",
        "status": "unavailable" if any(missing.values()) else "ready",
        "missing": missing,
        "teacher_requirement": {
            "requested_model": teacher.get("requested_model"),
            "input": teacher.get("input"),
            "temporal_teacher_allowed": teacher.get("temporal_teacher_allowed"),
            "distillation_ready": teacher.get("distillation_ready"),
        },
        "note": (
            "No trained author RawVLA RAW frontend checkpoint is present. Default ISP "
            "and official pi05 alone are historical diagnostics, not this teacher."
        ),
    }


def _context(args: argparse.Namespace):
    from owac.backends.raw_distillation_context import load_experiment_context

    loader = getattr(args, "_context_loader", None)
    if loader is not None:
        return loader(args)
    evidence = "fixture_contract" if args.fixture_context else "real_policy"
    return load_experiment_context(
        args.config,
        evidence_level=evidence,
        fixture_factory=args.fixture_context,
    )


def _unavailable(config: dict[str, Any], reason: str, schema: str) -> int:
    return _emit(
        {
            "schema": schema,
            "status": "unavailable",
            "updates": 0,
            "reason": reason,
            "assets": _asset_report(config),
        },
        exit_code=2,
    )


def cmd_assets(args: argparse.Namespace) -> int:
    report = _asset_report(_config(args.config))
    return _emit(report, exit_code=0 if report["status"] == "ready" else 2)


def cmd_preflight(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        if getattr(args, "student_artifact", None):
            from owac.backends.raw_distillation_context import build_pure_context

            context = build_pure_context(
                Path(args.config),
                config,
                args.student_artifact,
                checkpoint_step=getattr(args, "step", None),
                allow_fixture=bool(args.fixture_context),
            )
        else:
            context = _context(args)
    except Exception as exc:
        return _unavailable(config, str(exc), "owac.raw-distillation.preflight.v1")
    report = context.preflight()
    report.update(
        {
            "schema": "owac.raw-distillation.preflight.v1",
            "status": "preflight_ready" if context.student_ready else "unavailable",
            "updates": 0,
            "actual_updates": 0,
            "authorization": config.get("authorization", {}),
        }
    )
    profile_runner = getattr(context, "profile_runner", None)
    if profile_runner is not None:
        report["profile"] = profile_runner.run()
    return _emit(report, exit_code=0 if context.student_ready else 2)


def cmd_prepare_pairs(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        context = _context(args)
    except Exception as exc:
        return _unavailable(config, str(exc), "owac.raw-distillation.prepare-pairs.v1")
    if context.teacher_adapter is None:
        return _unavailable(
            config,
            "learned RAW teacher adapter/checkpoint is absent; no Default-ISP substitution",
            "owac.raw-distillation.prepare-pairs.v1",
        )
    import numpy as np
    import torch

    if not args.raw:
        return _unavailable(
            config,
            "prepare-pairs requires --raw and a valid learned teacher adapter",
            "owac.raw-distillation.prepare-pairs.v1",
        )
    raw_path = Path(args.raw)
    if not raw_path.is_file():
        return _unavailable(
            config,
            f"RAW burst array not found: {raw_path}",
            "owac.raw-distillation.prepare-pairs.v1",
        )
    raw = torch.from_numpy(np.load(raw_path, allow_pickle=False))
    prepared = context.teacher_adapter.prepare(raw, history=context.history)
    from owac.training.raw_distillation import write_prepared_teacher_bundle

    hashes = write_prepared_teacher_bundle(args.output, prepared)
    return _emit(
        {
            "schema": "owac.raw-distillation.prepare-pairs.v1",
            "status": "prepared",
            "evidence_level": context.evidence_level,
            "real_ready": prepared.receipt.real_ready,
            "output": str(Path(args.output)),
            "hashes": hashes,
            "receipt": prepared.receipt.as_dict(),
        },
        exit_code=0,
    )


def cmd_parity(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        context = _context(args)
    except Exception as exc:
        return _unavailable(config, str(exc), "owac.raw-distillation.parity.v1")
    runner = getattr(context, "parity_runner", None)
    if runner is None:
        return _unavailable(
            config,
            "native policy observation/token preparation or parity runner is absent; "
            "no tolerance was frozen",
            "owac.raw-distillation.parity.v1",
        )
    report = runner.run()
    payload = report.as_dict() if hasattr(report, "as_dict") else dict(report)
    source_ids = (
        sorted(source.source_id for source in context.dataset.manifest.sources)
        if context.dataset is not None
        else []
    )
    if hasattr(report, "tolerance"):
        payload["tolerance_receipt"] = {
            "digest": report.tolerance.digest,
            "repeat_max_error": dict(report.tolerance.repeat_max_error),
            "tolerances": dict(report.tolerance.tolerances),
            "floor": report.tolerance.floor,
            "multiplier": report.tolerance.multiplier,
        }
    payload.update(
        {
            "schema": "owac.raw-distillation.parity.v1",
            "actual_updates": 0,
            "config_path": str(context.config_path),
            "source_ids": source_ids,
            "real_expert_input": False,
        }
    )
    return _emit(payload, exit_code=0 if payload.get("passed") else 3)


def _stage0_report(
    path: Any, *, checkpoint_sha: str, source_sha: str, config_sha: str
) -> dict[str, Any]:
    if not path:
        raise ValueError("stage0 parity report is absent")
    report = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("stage0 parity report did not pass")
    if report.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("stage0 report checkpoint identity differs")
    if report.get("source_sha256") != source_sha:
        raise ValueError("stage0 report source identity differs")
    if report.get("config_sha256") != config_sha:
        raise ValueError("stage0 report config identity differs")
    if not report.get("tolerance_digest") or not report.get("native_digest"):
        raise ValueError("stage0 report lacks frozen tolerance/native digest")
    return report


def cmd_train(args: argparse.Namespace) -> int:
    config = _config(args.config)
    if not args.authority:
        return _emit(
            {
                "schema": "owac.raw-distillation.train.v1",
                "status": "blocked",
                "updates": 0,
                "actual_updates": 0,
                "reason": (
                    "explicit actual training authority receipt is required; default is preflight"
                ),
            },
            exit_code=2,
        )
    if config.get("authorization", {}).get("optimizer_updates") is not True:
        return _emit(
            {
                "schema": "owac.raw-distillation.train.v1",
                "status": "blocked",
                "actual_updates": 0,
                "reason": "config.authorization.optimizer_updates is false",
            },
            exit_code=2,
        )
    try:
        context = _context(args)
    except Exception as exc:
        return _unavailable(config, str(exc), "owac.raw-distillation.train.v1")
    fixture_train = bool(getattr(context, "fixture_contract", False) and args.fixture_context)
    if not context.teacher_ready or not (context.real_ready or fixture_train):
        return _unavailable(
            config,
            "accepted learned RAW teacher or real policy/load-audit is absent",
            "owac.raw-distillation.train.v1",
        )

    stage0 = {}
    split = context.preflight()["source_split_sizes"]
    if split != {"train": 40, "validation": 5, "test": 5}:
        if not (context.fixture_contract and args.fixture_context):
            return _unavailable(
                config,
                f"source split is not 40/5/5: {split}",
                "owac.raw-distillation.train.v1",
            )
    if any(
        config.get("resource_limits", {}).get(key) is None
        for key in ("gpu_hours", "training_disk_bytes")
    ):
        return _unavailable(
            config,
            "measured resource caps/profile are not frozen",
            "owac.raw-distillation.train.v1",
        )
    from owac.training.raw_distillation import (
        AnchorBatchIterator,
        RawDistillationTrainer,
        TrainingAuthority,
        load_training_checkpoint,
        run_distillation_loop,
    )

    trainer = None
    try:
        authority = TrainingAuthority.from_file(args.authority)
        trainer = RawDistillationTrainer(
            student=context.student,
            bridge=context.bridge,
            config=config,
            history=context.history,
            authority=authority,
            teacher_ready=context.teacher_ready,
            stage0_parity_ready=True,
            source_split_ready=True,
            resource_budget_ready=True,
            phase1_diagnosis=getattr(context, "phase1_diagnosis", None),
            matched_cd_declaration=getattr(context, "matched_cd_declaration", None),
            teacher=context.teacher_adapter,
        )
        iterator = AnchorBatchIterator(context, split="train", arm=args.arm)
        optimizer_factory = getattr(context, "optimizer_factory", None)
        scheduler_factory = getattr(context, "scheduler_factory", None)
        if optimizer_factory is None or scheduler_factory is None:
            optimizer, scheduler = trainer.optimizer_and_scheduler()
        else:
            optimizer = optimizer_factory(trainer)
            scheduler = scheduler_factory(trainer)
        if context.fixture_contract:
            validation = getattr(context, "validation_callback", None)
            if validation is None:
                from owac.training.raw_distillation import TrainingUnavailable

                raise TrainingUnavailable(
                    "fixture command training requires an explicit validation callback"
                )
        else:
            from owac.training.raw_distillation import make_validation_callback

            validation = make_validation_callback(context, trainer, args.arm)
        if args.resume:
            load_training_checkpoint(
                args.resume,
                trainer=trainer,
                optimizer=optimizer,
                scheduler=scheduler,
                iterator=iterator,
                arm=args.arm,
                config=config,
            )
        report = run_distillation_loop(
            trainer=trainer,
            iterator=iterator,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=args.checkpoint_dir,
            arm=args.arm,
            max_updates=args.max_updates,
            validation=validation,
            checkpoint_updates=tuple(
                int(v)
                for v in config.get("training", {}).get("checkpoint_updates", (0, 500, 2000, 5000))
            ),
        )
    except Exception as exc:
        return _emit(
            {
                "schema": "owac.raw-distillation.train.v1",
                "status": "blocked",
                "actual_updates": 0 if trainer is None else int(trainer.cumulative_updates),
                "cumulative_updates": 0 if trainer is None else int(trainer.cumulative_updates),
                "anchor_presentations": 0 if trainer is None else int(trainer.anchor_presentations),
                "reason": str(exc),
            },
            exit_code=2,
        )
    report.update(
        {
            "schema": "owac.raw-distillation.train.v1",
            "arm": args.arm,
            "stage0_digest": stage0.get("native_digest"),
            "actual_updates": report.get("actual_updates", 0),
        }
    )
    return _emit(report, exit_code=0)


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _config(args.config)
    artifact = getattr(args, "student_artifact", None)
    if artifact:
        try:
            from owac.backends.raw_distillation_context import build_pure_context

            context = build_pure_context(
                Path(args.config),
                config,
                artifact,
                checkpoint_step=getattr(args, "step", None),
                allow_fixture=bool(args.fixture_context),
            )
        except Exception as exc:
            return _unavailable(config, str(exc), "owac.raw-distillation.evaluate.v1")
    else:
        if not args.fixture_context and getattr(args, "_context_loader", None) is None:
            return _unavailable(
                config,
                "real evaluation requires --student-artifact; no fresh/random stem is loaded",
                "owac.raw-distillation.evaluate.v1",
            )
        try:
            context = _context(args)
        except Exception as exc:
            return _unavailable(config, str(exc), "owac.raw-distillation.evaluate.v1")
        if not context.fixture_contract:
            return _unavailable(
                config,
                "real evaluation requires --student-artifact; "
                "injected non-fixture context rejected",
                "owac.raw-distillation.evaluate.v1",
            )
    if not context.student_ready or context.dataset is None:
        return _unavailable(
            config,
            "pure student/export/frozen backend or student dataset is absent",
            "owac.raw-distillation.evaluate.v1",
        )
    from dataclasses import asdict

    import torch

    from owac.evaluation.raw_distillation import (
        FrozenBackend,
        PureStudentRuntime,
        aggregate_metrics,
        denormalize_actions,
        evaluate_windows,
        load_labels_without_teacher,
        select_validation,
    )

    backend = FrozenBackend(
        bridge=context.bridge,
        tokenizer=context.tokenizer,
        stats=context.stats,
        identity=context.backend_identity,
        tokenizer_sha256=str(context.tokenizer_identity.get("sha256", "")),
        stats_sha256=str(context.stats_identity.get("sha256", "")),
    )
    runtime = PureStudentRuntime(student=context.student, backend=backend, history=context.history)
    step = int(getattr(context, "checkpoint_step", 0))
    normalized_records = []
    physical_records = []
    anchor_count = 0
    for anchor in context.camera_anchors():
        split = context.dataset.manifest.source_split(anchor.source_id)
        if split not in {"validation", "test"} or anchor.variant != "rawvla_rgb3":
            continue
        views = [context.dataset.student_view(frame_key) for frame_key in anchor.frame_keys]
        runtime.reset()
        generator = torch.Generator(device=runtime._device()).manual_seed(101)
        noise = torch.randn(
            (1, backend.action_horizon, backend.padded_dim),
            generator=generator,
            device=runtime._device(),
            dtype=torch.float32,
        )
        prediction = runtime.infer_anchor(views, noise=noise)
        labels = load_labels_without_teacher(context.dataset, anchor.frame_keys[0])
        actions = labels["actions"]
        future_mask = labels["future_mask"].unsqueeze(0)
        target_device = prediction.normalized_actions.device
        target_normalized = torch.zeros(
            1,
            backend.action_horizon,
            backend.padded_dim,
            dtype=torch.float32,
            device=target_device,
        )
        target_normalized[:, :, : backend.control_dim] = actions[
            : backend.action_horizon
        ].unsqueeze(0)
        target_physical = denormalize_actions(
            actions[: backend.action_horizon, : backend.control_dim].unsqueeze(0).to(target_device),
            backend.stats,
        )
        normalized_records.append(
            evaluate_windows(
                prediction.normalized_actions,
                target_normalized,
                future_mask,
                split=split,
                step=step,
                control_dim=backend.control_dim,
                normalized=True,
            )
        )
        physical_records.append(
            evaluate_windows(
                prediction.physical_actions,
                target_physical,
                future_mask,
                split=split,
                step=step,
                control_dim=backend.control_dim,
                normalized=False,
            )
        )
        anchor_count += 1
    normalized_aggregates = []
    physical_aggregates = []
    for split in ("validation", "test"):
        try:
            normalized_aggregates.append(
                aggregate_metrics(normalized_records, split=split, step=step)
            )
            physical_aggregates.append(aggregate_metrics(physical_records, split=split, step=step))
        except ValueError:
            continue
    validation = [record for record in physical_aggregates if record.split == "validation"]
    selection = select_validation(validation) if validation else None
    return _emit(
        {
            "schema": "owac.raw-distillation.evaluate.v1",
            "status": "evaluated",
            "evidence_level": context.evidence_level,
            "student_artifact": artifact,
            "checkpoint_step": step,
            "anchors": anchor_count,
            "normalized": [asdict(record) for record in normalized_aggregates],
            "physical": [asdict(record) for record in physical_aggregates],
            "records": [asdict(record) for record in physical_aggregates],
            "normalized_windows": len(normalized_records),
            "physical_windows": len(physical_records),
            "selection": None if selection is None else asdict(selection),
            "actual_rollouts": 0,
        },
        exit_code=0,
    )


def cmd_export_student(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        context = _context(args)
    except Exception as exc:
        return _unavailable(config, str(exc), "owac.raw-distillation.export-student.v1")
    if context.student is None:
        return _unavailable(
            config,
            "actual accepted policy/load audit and student frontend are absent",
            "owac.raw-distillation.export-student.v1",
        )
    from owac.baselines.raw_stem_siglip import export_student_artifact

    training_checkpoint = getattr(args, "training_checkpoint", None)
    requested_step = getattr(args, "step", None)
    if training_checkpoint:
        import torch

        checkpoint_path = Path(training_checkpoint)
        if not checkpoint_path.is_file():
            return _unavailable(
                config,
                f"training checkpoint not found: {checkpoint_path}",
                "owac.raw-distillation.export-student.v1",
            )
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            return _unavailable(
                config,
                f"training checkpoint cannot be safely loaded: {exc}",
                "owac.raw-distillation.export-student.v1",
            )
        if payload.get("schema") != "owac.raw-distillation.checkpoint.v1":
            return _unavailable(
                config,
                "training checkpoint schema is not guarded",
                "owac.raw-distillation.export-student.v1",
            )
        if payload.get("arm") != getattr(args, "arm", "C"):
            return _unavailable(
                config,
                "training checkpoint arm differs from requested export arm",
                "owac.raw-distillation.export-student.v1",
            )
        actual_step = int(payload.get("cumulative_updates", -1))
        if actual_step < 0:
            return _unavailable(
                config,
                "training checkpoint lacks cumulative step",
                "owac.raw-distillation.export-student.v1",
            )
        if requested_step is not None and int(requested_step) != actual_step:
            return _unavailable(
                config,
                f"--step {requested_step} does not match checkpoint step {actual_step}",
                "owac.raw-distillation.export-student.v1",
            )
        try:
            context.student.load_state_dict(payload["student"], strict=True)
        except Exception as exc:
            return _unavailable(
                config,
                f"training student state does not match artifact architecture: {exc}",
                "owac.raw-distillation.export-student.v1",
            )
        checkpoint_step = actual_step
        export_extra = {
            "evidence_level": context.evidence_level,
            "training_checkpoint": str(checkpoint_path),
            "training_arm": payload.get("arm"),
        }
    else:
        if requested_step not in (None, 0):
            return _unavailable(
                config,
                "fresh initialized export is step 0 only; --step cannot relabel random weights",
                "owac.raw-distillation.export-student.v1",
            )
        checkpoint_step = int(getattr(context, "checkpoint_step", 0))
        export_extra = {"evidence_level": context.evidence_level}
    metadata = export_student_artifact(
        context.student,
        args.output,
        backend_identity=context.backend_identity,
        tokenizer_identity=context.tokenizer_identity,
        normalizer_identity=context.stats_identity,
        extra_metadata=export_extra,
        checkpoint_step=checkpoint_step,
    )
    return _emit(
        {
            "schema": "owac.raw-distillation.export-student.v1",
            "status": "exported",
            "output": str(Path(args.output)),
            "checkpoint_step": metadata["step"],
            "evidence_level": context.evidence_level,
        },
        exit_code=0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="owac.runtime.raw_distillation",
        description=(
            "RAW-stem distillation engineering CLI. Default invocations are zero-update; "
            "real training/rollout require separate explicit authority."
        ),
    )
    parser.add_argument("--config", default=CONFIG_DEFAULT, help="experiment config path")
    parser.add_argument("--fixture-context", default=None, help="registered fixture context name")
    subs = parser.add_subparsers(dest="command", required=True)
    assets = subs.add_parser("assets", help="report required learned teacher/policy/data assets")
    assets.set_defaults(func=cmd_assets)
    prepare = subs.add_parser("prepare-pairs", help="prepare RAW-teacher paired arrays")
    prepare.add_argument("--raw", default=None, help="RAW burst .npy [B,T,C,H,W]")
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(func=cmd_prepare_pairs)
    preflight = subs.add_parser("preflight", help="zero-update readiness/budget preflight")
    preflight.set_defaults(func=cmd_preflight)
    parity = subs.add_parser("parity", help="stage-0 native-vs-bridge parity")
    parity.set_defaults(func=cmd_parity)
    train = subs.add_parser("train", help="gated training path; never steps without authority")
    train.add_argument("--authority", default=None)
    train.add_argument("--arm", choices=("B", "C", "D"), default="C")
    train.add_argument(
        "--checkpoint-dir", default="outputs/raw-distillation-20261020-sol/checkpoints"
    )
    train.add_argument("--resume", default=None)
    train.add_argument("--max-updates", type=int, default=None)
    train.set_defaults(func=cmd_train)
    evaluate = subs.add_parser("evaluate", help="pure-student offline evaluation")
    evaluate.add_argument("--student-artifact", default=None)
    evaluate.add_argument("--step", type=int, default=None)
    evaluate.set_defaults(func=cmd_evaluate)
    export = subs.add_parser("export-student", help="export the pure student front end")
    export.add_argument("--output", required=True)
    export.add_argument("--step", type=int, default=None)
    export.add_argument("--training-checkpoint", default=None)
    export.add_argument("--arm", choices=("B", "C", "D"), default="C")
    export.set_defaults(func=cmd_export_student)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    context_loader: Callable[[argparse.Namespace], Any] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args._context_loader = context_loader
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        return _emit({"status": "unavailable", "reason": str(exc)}, exit_code=2)
    except ValueError as exc:
        return _emit({"status": "blocked", "reason": str(exc)}, exit_code=2)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
