"""N04 v1.1 S2 评估入口：开发集动作 loss + 每模型 100 次闭环驱动。

CLI（只保留常用参数）::

    # 固定 dev 窗口的动作 loss
    python -m owac.evaluation.n04_v11 --config <cfg> --output <run> --checkpoint <ckpt>

    # 闭环：逐 episode 写 eval_results.jsonl 与 artifacts/<platform>/<identity>/
    python -m owac.evaluation.n04_v11 --config <cfg> --output <run> \
        --checkpoint <ckpt> --eval-list configs/n04/v1_1/eval100.json

计数口径（契约 §7 与 v1.1 方案 §10）：

* ``success`` / ``failure`` 进分母；任务超时算失败（``timeout=true``），不删分母；
* ``infrastructure_error`` 单独计数、可原身份补跑，不计入失败分母；
* 同一个 ``identity`` 只按最后一行结算：补跑成功后不再重复计入。

执行段长度、重规划节奏和成功判据都在仿真器执行器里沿用既有协议；本驱动只接线——
``closed_loop.command`` 把单 episode 交给执行器，并从 ``artifacts/<identity>/result.json``
读回状态，不按 action horizon 猜节奏。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..raw.n04_v11_data import PLATFORM_SPECS
from ..training.n04_v11 import (
    V11Trainer,
    load_config,
    read_jsonl,
    summarize_eval_results,
    write_json_atomic,
    write_progress,
)

__all__ = [
    "DEFAULT_LIGHTING_DOMAINS",
    "EpisodeRunner",
    "SUBPROCESS_RESULT",
    "SubprocessEpisodeRunner",
    "load_eval_list",
    "main",
    "run_closed_loop",
    "run_dev_action_loss",
]

#: 官方清单里的五个光照域（见 RawVLA-Bench 全量清单的 ``lighting_domains``）。
DEFAULT_LIGHTING_DOMAINS = ("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver")

#: 执行器需要在 episode 产物目录里写下的结果文件名（只含 status/success/…）。
SUBPROCESS_RESULT = "result.json"


# --------------------------------------------------------------------------------------
# 清单
# --------------------------------------------------------------------------------------


def load_eval_list(
    path: str | Path, *, domains: Sequence[str] = DEFAULT_LIGHTING_DOMAINS, per_domain: int = 10
) -> dict[str, Any]:
    """读并检查 ``eval100.json``：每平台 5 域 × ``per_domain`` 个 episode、身份唯一。"""
    payload = json.loads(Path(path).read_text())
    entries = payload["entries"] if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"{path} 不是 eval 清单：期望 entries 列表")
    domains = tuple(domains)
    seen: set[tuple[str, str]] = set()
    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        platform = str(entry["platform"])
        if platform not in PLATFORM_SPECS:
            raise ValueError(f"未知平台 {platform!r}（记录 {entry}）")
        domain = str(entry["lighting_domain"])
        if domain not in domains:
            raise ValueError(f"未知光照域 {domain!r}（记录 {entry}）")
        identity = str(entry.get("identity") or _identity_of(entry))
        key = (platform, identity)
        if key in seen:
            raise ValueError(f"重复 episode 身份 {key}")
        seen.add(key)
        counts.setdefault(platform, {}).setdefault(domain, 0)
        counts[platform][domain] += 1
    for platform in platform_counts(entries):
        for domain in domains:
            count = counts[platform].get(domain, 0)
            if count != per_domain:
                raise ValueError(f"{platform}/{domain} 有 {count} 个 episode，期望 {per_domain}")
    return {
        "path": str(path),
        "lighting_domains": list(domains),
        "entries": entries,
        "counts": counts,
    }


def platform_counts(entries: Sequence[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for entry in entries:
        platform = str(entry["platform"])
        if platform not in seen:
            seen.append(platform)
    return seen


def _identity_of(entry: dict[str, Any]) -> str:
    platform = str(entry["platform"])
    task = str(entry.get("task", entry.get("task_name", "")))
    state = entry.get("initial_state", entry.get("episode", entry.get("dataset_seed", "")))
    domain = str(entry.get("lighting_domain", ""))
    return f"{platform}:{task}:{state}:{domain}"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:120]


# --------------------------------------------------------------------------------------
# dev 动作 loss
# --------------------------------------------------------------------------------------


def run_dev_action_loss(
    config: dict[str, Any],
    *,
    checkpoint: str,
    output: str | Path,
    seed: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """在固定 dev 窗口上评动作 loss，并刷新 ``progress.json``（phase=dev_eval）。"""
    run_dir = Path(output)
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = V11Trainer(
        config,
        seed=int(seed if seed is not None else config.get("seed", 101)),
        run_dir=run_dir,
        device=device,
    )
    payload = trainer.load_checkpoint(checkpoint)
    report = trainer.evaluate_dev_action_loss()
    record = {
        "stage": config.get("stage", "S2"),
        "platform": trainer.platform,
        "arm": config.get("arm", "D"),
        "seed": trainer.seed,
        "checkpoint": str(checkpoint),
        "step": int(payload.get("step", trainer.step)),
        "dev_action_loss": report["dev_loss"],
        "windows": report["windows"],
        "points": report["points"],
        "seconds": report.get("seconds"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(run_dir / "dev-action-loss.json", record)
    write_progress(
        run_dir / "progress.json",
        config=trainer.config,
        phase="dev_eval",
        step=record["step"],
        budget_steps=trainer.budget_steps,
        train_loss=trainer.loss_ema,
        dev_loss=report["dev_loss"],
        step_seconds=report.get("seconds"),
        eval_summary=summarize_eval_results(run_dir / "eval_results.jsonl"),
        artifacts=sorted(
            str(path.relative_to(run_dir))
            for path in (run_dir / "artifacts").rglob("*")
            if path.is_file()
        )[:20]
        if (run_dir / "artifacts").exists()
        else [],
        log_tail=list(trainer.log_tail),
    )
    return record


# --------------------------------------------------------------------------------------
# 闭环驱动
# --------------------------------------------------------------------------------------


class EpisodeRunner:
    """单 episode 执行器接口：返回 ``status/success``（失败与超时都算 failure）。"""

    def __call__(
        self, entry: dict[str, Any], *, checkpoint: str, artifacts_dir: Path, config: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover - 接口定义
        raise NotImplementedError


class SubprocessEpisodeRunner(EpisodeRunner):
    """把单 episode 交给既有仿真器执行器：``closed_loop.command`` 模板 + 结果文件。

    模板占位符：``{entry}``（写到产物目录的 episode JSON）、``{artifacts}``、``{checkpoint}``、
    ``{platform}``、``{output}``。执行器需要写 ``artifacts/result.json``：
    ``{"status": "success"|"failure", "success": bool, ...}``。返回码 124（超时）按失败计。
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        command = settings.get("command")
        if not command:
            raise SystemExit(
                "闭环需要配置 closed_loop.command（复用现有 LIBERO／RoboTwin 执行器）；"
                "本机没有仿真器时不要伪造成功率，请在远端执行器可用后接线"
            )
        self.command = list(command) if isinstance(command, (list, tuple)) else [str(command)]
        self.cwd = settings.get("cwd")
        self.timeout = int(settings.get("timeout_seconds", 900))
        self.env_passthrough = tuple(settings.get("env", ()))

    def __call__(
        self, entry: dict[str, Any], *, checkpoint: str, artifacts_dir: Path, config: dict[str, Any]
    ) -> dict[str, Any]:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        entry_path = artifacts_dir / "entry.json"
        write_json_atomic(entry_path, entry)
        replacements = {
            "entry": str(entry_path),
            "artifacts": str(artifacts_dir),
            "checkpoint": str(checkpoint),
            "platform": str(entry.get("platform", config.get("platform", ""))),
            "output": str(artifacts_dir.parent.parent),
        }
        command = [part.format(**replacements) for part in self.command]
        environment = os.environ.copy()
        for name in self.env_passthrough:
            if name in os.environ:
                environment[name] = os.environ[name]
        log_path = artifacts_dir / "executor.log"
        started = time.perf_counter()
        with log_path.open("a") as stream:
            stream.write(json.dumps({"command": command, "cwd": self.cwd}) + "\n")
            stream.flush()
            try:
                process = subprocess.run(
                    command,
                    cwd=self.cwd,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                )
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 124
        seconds = time.perf_counter() - started
        result_path = artifacts_dir / SUBPROCESS_RESULT
        payload = json.loads(result_path.read_text()) if result_path.exists() else {}
        if code == 124:
            return {
                "status": "failure",
                "success": False,
                "timeout": True,
                "returncode": code,
                "seconds": seconds,
                "detail": f"执行器超过 {self.timeout}s 未完成，按失败计",
            }
        if code != 0 or not payload:
            return {
                "status": "infrastructure_error",
                "success": None,
                "timeout": False,
                "returncode": code,
                "seconds": seconds,
                "detail": payload.get("error") or f"执行器返回码 {code} 且没有结果文件",
            }
        status = str(payload.get("status", "failure"))
        if status not in ("success", "failure"):
            raise ValueError(f"执行器结果状态非法：{status!r}")
        payload.setdefault("success", status == "success")
        payload.update(
            {"timeout": bool(payload.get("timeout", False)), "returncode": code, "seconds": seconds}
        )
        return payload


def run_closed_loop(
    config: dict[str, Any],
    *,
    eval_list: str | Path,
    checkpoint: str,
    output: str | Path,
    runner: Callable[..., dict[str, Any]] | None = None,
    platform: str | None = None,
) -> dict[str, Any]:
    """逐 episode 驱动闭环；已结算身份直接复用，``infrastructure_error`` 允许原身份补跑。"""
    run_dir = Path(output)
    run_dir.mkdir(parents=True, exist_ok=True)
    listing = load_eval_list(
        eval_list, domains=config.get("eval", {}).get("lighting_domains", DEFAULT_LIGHTING_DOMAINS)
    )
    entries = listing["entries"]
    if platform:
        entries = [entry for entry in entries if str(entry["platform"]) == platform]
    runner = runner or SubprocessEpisodeRunner(config.get("closed_loop", {}))
    results_path = run_dir / "eval_results.jsonl"
    settled = _settled_identities(results_path)
    planned = len(entries)
    completed = errors = 0
    for entry in entries:
        identity = str(entry.get("identity") or _identity_of(entry))
        if identity in settled:
            continue
        artifacts_dir = run_dir / "artifacts" / str(entry["platform"]) / _slug(identity)
        started = time.perf_counter()
        try:
            outcome = dict(
                runner(entry, checkpoint=checkpoint, artifacts_dir=artifacts_dir, config=config)
            )
        except Exception as error:  # noqa: BLE001 - 执行器异常按基础设施错误保留，不吞成成功
            outcome = {
                "status": "infrastructure_error",
                "success": None,
                "detail": f"{type(error).__name__}: {error}",
            }
        status = str(outcome.get("status", "infrastructure_error"))
        if status not in ("success", "failure", "infrastructure_error"):
            raise ValueError(f"执行器返回了非法状态 {status!r}")
        success = outcome.get("success")
        if status == "success":
            success = True if success is None else bool(success)
        elif status == "failure":
            success = False if success is None else bool(success)
        else:
            success = None  # 基础设施错误不进分母，也不冒充失败
        row = {
            "identity": identity,
            "platform": str(entry["platform"]),
            "task": entry.get("task", entry.get("task_name")),
            "initial_state": entry.get("initial_state", entry.get("episode")),
            "env_seed": entry.get("env_seed", entry.get("dataset_seed")),
            "lighting_domain": entry.get("lighting_domain"),
            "checkpoint": str(checkpoint),
            "status": status,
            "success": success,
            "timeout": bool(outcome.get("timeout", False)),
            "seconds": outcome.get("seconds", time.perf_counter() - started),
            "artifacts": sorted(
                str(path.relative_to(run_dir))
                for path in artifacts_dir.rglob("*")
                if path.is_file()
            ),
            "detail": outcome.get("detail"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with results_path.open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            stream.flush()
        if row["status"] in ("success", "failure"):
            completed += 1
        elif row["status"] == "infrastructure_error":
            errors += 1
        summary = summarize_eval_results(results_path, planned=planned)
        write_progress(
            run_dir / "progress.json",
            config=config,
            phase="eval",
            step=int(summary["completed"]),
            budget_steps=planned,
            eval_summary=summary,
            artifacts=sorted(
                str(path.relative_to(run_dir))
                for path in (run_dir / "artifacts").rglob("*")
                if path.is_file()
            )[:20],
            log_tail=[f"{identity}: {row['status']} ({row['seconds']:.1f}s)"],
        )
    summary = summarize_eval_results(results_path, planned=planned)
    summary["new_completed"] = completed
    summary["new_errors"] = errors
    write_json_atomic(run_dir / "eval-summary.json", summary)
    return summary


def _settled_identities(results_path: Path) -> set[str]:
    settled: set[str] = set()
    for row in read_jsonl(results_path):
        if row.get("status") in ("success", "failure"):
            settled.add(str(row.get("identity")))
    return settled


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", "--resume", dest="checkpoint", default=None)
    parser.add_argument("--eval-list", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.checkpoint:
        raise SystemExit("评估需要 --checkpoint（训练产物）；不要用随机权重冒充模型")
    config = load_config(args.config, data_root=args.data_root)
    if args.eval_list:
        report = run_closed_loop(
            config,
            eval_list=args.eval_list,
            checkpoint=args.checkpoint,
            output=args.output,
        )
    else:
        report = run_dev_action_loss(
            config,
            checkpoint=args.checkpoint,
            output=args.output,
            seed=args.seed,
            device=args.device,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
