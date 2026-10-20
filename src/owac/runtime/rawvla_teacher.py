"""Import-safe CLI for the authorized task6 Normal teacher data pipeline.

Only ``inspect-source`` and ``status`` are read-only. ``prepare-data`` is the
explicit replay entry point; this round does not invoke it. Importing this module
does not import the simulator or start any process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from owac.evaluation.rawvla_teacher import TeacherEvaluationError, save_evaluation_result
from owac.raw.teacher_data import (
    LIGHTING,
    TASK_ID,
    TASK_NAME,
    TASK_SUITE,
    TeacherData,
    load_source_manifest,
)
from owac.runtime.rawvla_teacher_budget import PhaseBudget
from owac.simulation.rawvla_teacher_replay import (
    MAX_REPLAY_SOURCES,
    NORMAL_DOMAIN,
    ReplayConfig,
    TeacherReplayBlocked,
    inspect_source_hdf5,
    prepare_task6_normal,
)
from owac.training.rawvla_teacher import (
    TeacherAssets,
    TeacherAuthority,
    TeacherTrainer,
    TeacherTrainingBlocked,
    build_author_teacher,
)

SCHEMA = "owac.raw-teacher.cli.v1"


def _authority(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise TeacherReplayBlocked(f"authority receipt not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "owac.raw-teacher.authorization.v1":
        raise TeacherReplayBlocked("authority schema is not owac.raw-teacher.authorization.v1")
    auth = payload.get("authorization", {})
    limits = payload.get("limits", {})
    if not auth.get("demonstration_replay"):
        raise TeacherReplayBlocked("authority does not authorize demonstration_replay")
    if int(limits.get("demonstrations_replayed_max", 0)) > 50:
        raise TeacherReplayBlocked("authority demonstration cap exceeds 50")
    if int(payload.get("training", {}).get("max_cumulative_updates", 2001)) > 2000:
        raise TeacherReplayBlocked("authority teacher update cap exceeds 2000")
    if float(limits.get("local_gpu_hours", 0)) > 12:
        raise TeacherReplayBlocked("authority GPU-hour cap exceeds 12")
    if int(limits.get("additional_disk_bytes", 0)) > 50 * 1024**3:
        raise TeacherReplayBlocked("authority disk cap exceeds 50 GiB")
    return payload


def cmd_inspect_source(args: argparse.Namespace) -> int:
    report = inspect_source_hdf5(
        args.hdf5_root,
        suite=str(args.suite),
        task_name=str(args.task_name),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    authority = _authority(args.authority)
    if args.max_attempts is not None and int(args.max_attempts) > MAX_REPLAY_SOURCES:
        raise TeacherReplayBlocked("--max-attempts may not exceed 50")
    config = ReplayConfig(
        author_root=Path(args.author_source),
        hdf5_root=Path(args.hdf5_root),
        out_root=Path(args.out_root),
        rgb_out_root=Path(args.rgb_out_root),
        seed=int(args.seed),
        replay_mode=str(args.replay_mode),
        representation=str(args.representation),
        compression=str(args.compression),
    )
    if config.out_root == config.rgb_out_root:
        raise TeacherReplayBlocked("RAW and RGB output roots must be distinct")
    result = prepare_task6_normal(
        config,
        authority=authority,
        repo_root=Path(args.repo_root),
        max_attempts=args.max_attempts,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _device(args: argparse.Namespace) -> str:
    if getattr(args, "device", None):
        return str(args.device)
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _budget(args: argparse.Namespace) -> PhaseBudget:
    authority = json.loads(Path(args.authority).read_text())
    run_root = Path(str(authority.get("run_root", "")))
    if not run_root.is_absolute():
        run_root = Path(args.repo_root) / run_root
    return PhaseBudget(run_root)


def _teacher_assets(args: argparse.Namespace) -> TeacherAssets:
    return TeacherAssets(
        author_root=Path(args.author_root),
        starvla_root=Path(args.starvla_root),
        checkpoint=Path(args.checkpoint),
        tokenizer=Path(args.tokenizer),
        statistics=Path(args.statistics),
        config_paths=tuple(Path(item) for item in args.config),
    )


def _teacher_stats(args: argparse.Namespace):
    from owac.raw.teacher_data import load_pi05_normalization_stats

    return load_pi05_normalization_stats(Path(args.statistics))


def _teacher_data(args: argparse.Namespace) -> TeacherData:
    manifest = load_source_manifest(
        args.source_manifest, repo_root=Path(args.repo_root), strict=True
    )
    return TeacherData(manifest, repo_root=Path(args.repo_root))


def cmd_preflight(args: argparse.Namespace) -> int:
    authority = TeacherAuthority.from_file(args.authority)
    budget = _budget(args)
    reserve = min(600.0, float(budget.remaining["gpu_seconds"]))
    with budget.lease("preflight", gpu=True, reserved_gpu_seconds=reserve):
        teacher = build_author_teacher(_teacher_assets(args), device=_device(args))
        trainer = TeacherTrainer(
            teacher=teacher,
            data=_teacher_data(args),
            authority=authority,
            run_root=Path(args.run_root),
            stats=_teacher_stats(args),
            activation_memory_mode="none",
        )
        report = trainer.preflight()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _phase_path(value: str | Path, phase_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = phase_root / path
    path = path.resolve()
    if not path.is_relative_to(phase_root.resolve()):
        raise TeacherTrainingBlocked(f"path escapes phase root: {path}")
    return path


def cmd_train(args: argparse.Namespace) -> int:
    from owac.evaluation.rawvla_teacher import TeacherOfflineEvaluator

    authority = TeacherAuthority.from_file(args.authority)
    phase_root = Path(args.run_root).resolve()
    authority_payload = json.loads(Path(args.authority).read_text())
    if Path(authority_payload.get("run_root", "")).resolve() != phase_root:
        raise TeacherTrainingBlocked("run root differs from the explicit phase authority")
    if Path(args.authority).resolve() != phase_root / "authority.json":
        raise TeacherTrainingBlocked("training requires the canonical phase authority file")
    phase_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = _phase_path(args.checkpoint_dir or (phase_root / "training"), phase_root)
    validation_root = _phase_path(
        args.validation_dir or (checkpoint_dir / "validation"), phase_root
    )
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise TeacherTrainingBlocked(f"resume checkpoint absent: {resume_path}")
        if not resume_path.resolve().is_relative_to(phase_root):
            raise TeacherTrainingBlocked("resume checkpoint must be in the phase root")
    budget = _budget(args)
    remaining = float(budget.remaining["gpu_seconds"])
    reserve = float(args.eval_reserve_hours) * 3600.0
    if float(args.eval_reserve_hours) != 3.0:
        raise TeacherTrainingBlocked("this run preserves the authorized 3h evaluation reserve")
    train_reserve = min(8 * 3600.0, remaining - reserve)
    if train_reserve < 600.0:
        raise TeacherTrainingBlocked(
            f"remaining GPU allowance {remaining:.1f}s does not preserve "
            f"{reserve:.1f}s evaluation reserve and a bounded training run"
        )
    if not args.catalog or not Path(args.catalog).is_file():
        raise TeacherTrainingBlocked("required frozen validation catalog is absent")
    execution_path = phase_root / "execution-config.json"
    if execution_path.is_file():
        import hashlib

        identities = json.loads(execution_path.read_text())["identities"]
        for filename, supplied in (
            ("validation-catalog.json", args.catalog),
            ("raw/sources-audited.json", args.source_manifest),
        ):
            if hashlib.sha256(Path(supplied).read_bytes()).hexdigest() != identities[filename]:
                raise TeacherTrainingBlocked(f"frozen identity differs for {filename}")
    # Lease covers model construction, training, validation, and saves.
    with budget.lease("train", gpu=True, reserved_gpu_seconds=train_reserve):
        teacher = build_author_teacher(_teacher_assets(args), device=_device(args))
        data = _teacher_data(args)
        stats = _teacher_stats(args)
        trainer = TeacherTrainer(
            teacher=teacher,
            data=data,
            authority=authority,
            run_root=checkpoint_dir,
            stats=stats,
            budget=budget,
            activation_memory_mode=str(args.activation_memory_mode),
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        global_updates = int(budget._state.get("updates", -1))
        global_anchors = int(budget._state.get("anchors", -1))
        if global_updates != int(trainer.cumulative_updates):
            raise TeacherTrainingBlocked(
                "phase ledger update count differs from trainer checkpoint; "
                "refusing to reset or double-charge"
            )
        if global_anchors != int(trainer.anchor_presentations):
            raise TeacherTrainingBlocked(
                "phase ledger anchor count differs from trainer checkpoint; "
                "refusing to reset or double-charge"
            )
        target = (
            authority.max_updates
            if args.max_updates is None
            else min(int(args.max_updates), authority.max_updates)
        )
        if target <= trainer.cumulative_updates or target > 2000:
            raise TeacherTrainingBlocked("cumulative target must advance within 2000")
        run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        evaluator = TeacherOfflineEvaluator(
            teacher=teacher,
            data=data,
            stats=stats,
            catalog_path=args.catalog,
        )

        def bound_callback() -> dict[str, Any]:
            step = int(trainer.cumulative_updates)
            output = validation_root / f"step-{step:04d}-{run_id}"
            if output.exists():
                raise TeacherTrainingBlocked(f"validation output already exists: {output}")
            output.mkdir(parents=True, exist_ok=False)
            result = evaluator.evaluate(split="validation", step=step)
            saved = save_evaluation_result(result, output)
            return {
                "first5_mae7": float(saved["first5_mae7"]),
                "full10_mae7": float(saved["full10_mae7"]),
                "step": step,
                "validation_dir": str(output),
            }

        report = trainer.train(max_updates=target, validation=bound_callback)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_offline_evaluate(args: argparse.Namespace) -> int:

    from owac.evaluation.rawvla_teacher import (
        TeacherOfflineEvaluator,
        load_teacher_frontend_artifact,
        save_evaluation_result,
    )

    data = _teacher_data(args)
    stats = _teacher_stats(args)
    budget = _budget(args)
    with budget.lease(
        "offline-validation",
        gpu=True,
        reserved_gpu_seconds=min(600.0, float(budget.remaining["gpu_seconds"])),
    ):
        teacher = build_author_teacher(_teacher_assets(args), device=_device(args))
        identity = {
            "author_revision": "ba9c56b04da2c4931b70d8844710f5928d43775b",
            "author_archive_sha256": (
                "57bc54e8a695ae07c2ffee9ebb2549e9efbb48536a97df14c5c8f63c52f3e012"
            ),
            "checkpoint_sha256": "68710f4e7906d1fbf65fc73f57a6d9de3369831aba0e4fed433e09d6407ba6e8",
            "tokenizer_sha256": "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6",
            "statistics_sha256": "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
            "source_manifest_sha256": __import__("hashlib")
            .sha256(Path(args.source_manifest).read_bytes())
            .hexdigest(),
            "backend": "PI05",
            "burst_frames": 6,
            "recurrent_endpoints": 6,
            "cameras": ["agentview", "wrist"],
            "input_provenance": "raw_pseudo_raw3_uint8_normal",
            "cumulative_updates": 0,
        }
        if not getattr(args, "frontend_artifact", None):
            raise TeacherEvaluationError(
                "offline-evaluate requires --frontend-artifact; random frontend is forbidden"
            )
        state = load_teacher_frontend_artifact(args.frontend_artifact, expected_identity=identity)
        teacher.frontend.load_state_dict(state, strict=True)
        evaluator = TeacherOfflineEvaluator(
            teacher=teacher, data=data, stats=stats, catalog_path=args.catalog
        )
        result = evaluator.evaluate(split="validation", step=int(args.step))
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result = save_evaluation_result(result, output)
        result["identity"] = identity
        result["prediction_npz"] = str(output / "predictions.npz")
        (output / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        )
        result["frontend_artifact"] = str(args.frontend_artifact)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import torch

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise TeacherTrainingBlocked(f"teacher checkpoint absent: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != "owac.rawvla-teacher.checkpoint.v1":
        raise TeacherTrainingBlocked("teacher checkpoint schema differs")
    output = Path(args.output)
    if output.exists():
        raise TeacherTrainingBlocked(f"refusing to overwrite export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "owac.rawvla-teacher.frontend-export.v1",
            "frontend": payload["frontend"],
            "frontend_state_keys": sorted(payload["frontend"]),
            "cumulative_updates": payload["cumulative_updates"],
            "config_sha256": payload["config_sha256"],
            "data_sha256": payload["data_sha256"],
            "source": payload.get("source"),
            "trainable_names": payload.get("trainable_names"),
            "origin_checkpoint": str(checkpoint),
        },
        output,
    )
    print(json.dumps({"schema": "owac.rawvla-teacher.export.v1", "output": str(output)}))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        report = {
            "schema": SCHEMA,
            "status": "unavailable",
            "reason": f"source manifest not found: {manifest_path}",
            "authorized_task": {
                "suite": TASK_SUITE,
                "task_id": TASK_ID,
                "task_name": TASK_NAME,
                "lighting": LIGHTING,
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    try:
        manifest = load_source_manifest(
            manifest_path,
            repo_root=Path(args.repo_root),
            strict=True,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"schema": SCHEMA, "status": "invalid", "reason": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    counts = {
        split: len(manifest.sources_for_split(split)) for split in ("train", "validation", "test")
    }
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "ready",
                "manifest": str(manifest_path),
                "split_counts": counts,
                "source_count": len(manifest.sources),
                "lighting": LIGHTING,
                "normal_domain": NORMAL_DOMAIN,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="owac.runtime.rawvla_teacher",
        description=(
            "Task6 Normal source inspection and bounded author replay preparation. "
            "Import and status/inspection do not start a simulator."
        ),
    )
    parser.add_argument("--repo-root", default=".")
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect-source", help="inspect HDF5 demo metadata only")
    inspect.add_argument("--hdf5-root", required=True)
    inspect.add_argument("--suite", default=TASK_SUITE)
    inspect.add_argument("--task-name", default=TASK_NAME)
    inspect.set_defaults(func=cmd_inspect_source)

    prepare = subs.add_parser("prepare-data", help="bounded task6 Normal replay preparation")
    prepare.add_argument("--authority", required=True)
    prepare.add_argument("--author-source", required=True)
    prepare.add_argument("--hdf5-root", required=True)
    prepare.add_argument("--out-root", required=True)
    prepare.add_argument("--rgb-out-root", required=True)
    prepare.add_argument("--max-attempts", type=int, default=None)
    prepare.add_argument("--seed", type=int, default=20260813)
    prepare.add_argument("--replay-mode", choices=("states", "actions"), default="states")
    prepare.add_argument("--representation", choices=("raw",), default="raw")
    prepare.add_argument("--compression", choices=("stored", "deflated"), default="stored")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(func=cmd_prepare_data)

    def teacher_common(parser):
        parser.add_argument("--authority", required=True)
        parser.add_argument("--source-manifest", required=True)
        parser.add_argument("--author-root", required=True)
        parser.add_argument("--starvla-root", required=True)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--tokenizer", required=True)
        parser.add_argument("--statistics", required=True)
        parser.add_argument("--config", action="append", required=True)
        parser.add_argument("--run-root", required=True)
        parser.add_argument("--device", default=None)

    preflight = subs.add_parser("preflight", help="zero-update native teacher preflight")
    teacher_common(preflight)
    preflight.set_defaults(func=cmd_preflight)
    train = subs.add_parser(
        "train",
        help="authorized native teacher training with bound offline validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: outputs/rawvla-teacher-20261020/runtime/bin/python "
            "-m owac.runtime.rawvla_teacher train "
            "--authority /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/authority.json "  # noqa: E501
            "--source-manifest /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/raw/sources-audited.json "  # noqa: E501
            "--catalog /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/validation-catalog.json "  # noqa: E501
            "--author-root /home/jeong/zeno/owac/repo/outputs/raw-distillation-20261020-sol/teacher-search-v2/author-main "  # noqa: E501
            "--starvla-root /home/jeong/zeno/owac/repo/outputs/raw-distillation-20261020-sol/starvla-pinned-source "  # noqa: E501
            "--checkpoint /home/jeong/zeno/owac/repo/outputs/checkpoints/raw-distillation/pi05_libero_starvla/bfloat16/model.safetensors "  # noqa: E501
            "--tokenizer /home/jeong/zeno/owac/repo/outputs/checkpoints/raw-distillation/paligemma_tokenizer.model "  # noqa: E501
            "--statistics /home/jeong/zeno/owac/repo/outputs/checkpoints/raw-distillation/pi05_libero_jax/assets/physical-intelligence/libero/norm_stats.json "  # noqa: E501
            "--config /home/jeong/zeno/owac/repo/outputs/raw-distillation-20261020-sol/teacher-search-v2/author-main/configs/libero/pi05.yaml "  # noqa: E501
            "--run-root /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020 "  # noqa: E501
            "--checkpoint-dir /home/jeong/zeno/owac/repo/outputs/rawvla-teacher-20261020/training "  # noqa: E501
            "--activation-memory-mode save_on_cpu"
        ),
    )
    teacher_common(train)
    train.add_argument("--catalog", required=True)
    train.add_argument("--checkpoint-dir", default=None)
    train.add_argument("--validation-dir", default=None)
    train.add_argument("--run-id", default=None)
    train.add_argument(
        "--activation-memory-mode",
        choices=("none", "save_on_cpu"),
        required=True,
    )
    train.add_argument("--eval-reserve-hours", type=float, default=3.0)
    train.add_argument("--max-updates", type=int, default=None)
    train.add_argument("--resume", default=None)
    train.set_defaults(func=cmd_train)
    offline = subs.add_parser(
        "offline-evaluate", help="validation-only native offline teacher evaluation"
    )
    teacher_common(offline)
    offline.add_argument("--step", type=int, default=0)
    offline.add_argument("--output-dir", required=True)
    offline.add_argument("--catalog", required=True)
    offline.add_argument("--frontend-artifact", required=True)
    offline.set_defaults(func=cmd_offline_evaluate)

    export = subs.add_parser("export", help="export learned frontend-only checkpoint")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(func=cmd_export)

    status = subs.add_parser("status", help="validate an existing source manifest")
    status.add_argument("--manifest", required=True)
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (TeacherReplayBlocked, TeacherTrainingBlocked) as exc:
        print(
            json.dumps(
                {"schema": SCHEMA, "status": "blocked", "reason": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    except FileNotFoundError as exc:
        print(
            json.dumps(
                {"schema": SCHEMA, "status": "unavailable", "reason": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
