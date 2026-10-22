"""Non-blocking CPU SwanLab sidecar for the full-visual run.

Only small metrics from the run's real JSONL/status files are uploaded.  This
module is independent of DDP/training, never uploads weights/RAW/cache/configs,
and reads SWANLAB_API_KEY only at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT = "OWAC"
SCHEMA = "owac.raw-full-visual.swanlab-sidecar.v1"
DEFAULT_ID = "owac-fullvisual-shared-stage"


class SidecarError(RuntimeError):
    pass


def _num(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _update_metrics(record: dict[str, Any]) -> dict[str, int | float]:
    components = record.get("loss_components") or {}
    metrics = {
        "train/loss_total": _num(components.get("total")),
        "train/mse": _num(components.get("mse")),
        "train/cosine": _num(components.get("cosine")),
        "train/lr": _num(record.get("lr_used")),
        "train/anchors": _num(record.get("new_anchor_presentations")),
        "train/global_updates": _num(record.get("new_global_updates")),
        "train/tokens": _num(record.get("global_token_denominator")),
        "gpu/allocated_bytes": _num(record.get("gpu_allocated_bytes")),
        "gpu/reserved_bytes": _num(record.get("gpu_reserved_bytes")),
        "budget/remote_gpu_wall_seconds": _num(record.get("remote_gpu_wall_used_seconds")),
    }
    wall = _num((record.get("host_wall_phases") or {}).get("total_seconds"))
    if wall is not None and wall > 0:
        metrics["train/step_seconds"] = wall
        batch = _num(record.get("effective_batch"))
        if batch is not None:
            metrics["train/anchors_per_second"] = float(batch) / wall
        tokens = metrics["train/tokens"]
        if tokens is not None:
            metrics["train/tokens_per_second"] = float(tokens) / wall
    return {key: value for key, value in metrics.items() if value is not None}


def _validation_metrics(record: dict[str, Any]) -> dict[str, int | float]:
    metrics = {
        "val/loss_total": _num(record.get("feature_total")),
        "val/mse": _num(record.get("feature_mse")),
        "val/cosine": _num(record.get("feature_cosine")),
        "val/valid_tokens": _num(record.get("valid_tokens")),
        "val/windows": _num(record.get("windows")),
    }
    startup = record.get("startup")
    if isinstance(startup, dict):
        metrics["val/startup_mse"] = _num(startup.get("feature_mse"))
    eligible = record.get("eligible_ge_25")
    if isinstance(eligible, dict):
        metrics["val/eligible_mse"] = _num(eligible.get("feature_mse"))
    per_domain = record.get("per_domain")
    if isinstance(per_domain, dict):
        for domain, item in per_domain.items():
            if isinstance(item, dict):
                metrics[f"val/domain_{domain}/mse"] = _num(item.get("feature_mse"))
    per_lighting = record.get("per_lighting")
    if isinstance(per_lighting, dict):
        for lighting, item in per_lighting.items():
            if isinstance(item, dict):
                metrics[f"val/light_{lighting}/mse"] = _num(item.get("feature_mse"))
    return {key: value for key, value in metrics.items() if value is not None}


def _status(record: dict[str, Any]) -> tuple[str, bool, bool, int | float | None]:
    status = str(record.get("status") or record.get("phase") or "")
    reason = str(record.get("stop_reason") or "")
    text = f"{status} {reason}".lower()
    finished = status.lower() in {"completed", "finished"} or any(
        key in text for key in ("maxcap", "max_cap", "max_capacity", "max_updates")
    )
    paused = "paused" in text or reason.lower() in {"pause", "paused"}
    budget = _num(record.get("remote_gpu_wall_used_seconds") or record.get("gpu_wall_sum_seconds"))
    return status, finished, paused, budget


@dataclass
class Cursor:
    train_steps: set[int] = field(default_factory=set)
    validation_steps: set[int] = field(default_factory=set)
    last_step: int | None = None
    url: str | None = None
    status: str = "starting"
    finished: bool = False
    finished_sdk: bool = False
    error: str | None = None

    @classmethod
    def load(cls, path: Path) -> Cursor:
        if not path.is_file():
            return cls()
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            train_steps={int(item) for item in value.get("train_steps", [])},
            validation_steps={int(item) for item in value.get("validation_steps", [])},
            last_step=value.get("last_step"),
            url=value.get("url"),
            status=str(value.get("status", "starting")),
            finished=bool(value.get("finished", False)),
            finished_sdk=bool(value.get("finished_sdk", False)),
            error=value.get("error"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "train_steps": sorted(self.train_steps),
            "validation_steps": sorted(self.validation_steps),
            "last_step": self.last_step,
            "url": self.url,
            "status": self.status,
            "finished": self.finished,
            "finished_sdk": self.finished_sdk,
            "error": self.error,
        }


class GpuSampler:
    def __init__(self, interval: float = 10.0) -> None:
        self.interval = max(1.0, float(interval))
        self.last = 0.0

    def sample(self) -> dict[str, int | float]:
        now = time.monotonic()
        if now - self.last < self.interval:
            return {}
        self.last = now
        try:
            run = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        result: dict[str, int | float] = {}
        if run.returncode != 0:
            return result
        for line in run.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                index, util, used, total = (
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                )
            except ValueError:
                continue
            result[f"gpu/{index}/utilization_percent"] = util
            result[f"gpu/{index}/memory_used_mb"] = used
            result[f"gpu/{index}/memory_total_mb"] = total
        return result


class SwanLabUploader:
    def __init__(
        self, *, sdk: Any, logdir: Path, project: str = PROJECT, swanlab_id: str = DEFAULT_ID
    ) -> None:
        self.sdk = sdk
        self.logdir = logdir
        self.project = project
        self.swanlab_id = swanlab_id
        self.ready = False

    def initialize(self) -> str | None:
        if self.ready:
            return None
        key = os.environ.get("SWANLAB_API_KEY")
        if not key:
            raise SidecarError("SWANLAB_API_KEY is not set")
        login = getattr(self.sdk, "login", None)
        if callable(login):
            login(api_key=key, save=False)
        self.logdir.mkdir(parents=True, exist_ok=True)
        settings_factory = getattr(self.sdk, "Settings", None)
        settings = None
        if callable(settings_factory):
            settings = settings_factory(
                probe={
                    "hardware": False,
                    "runtime": False,
                    "requirements": False,
                    "conda": False,
                    "git": False,
                    "swanlab": False,
                    "monitor": False,
                },
                terminal={"proxy_type": "none"},
            )
        run = self.sdk.init(
            project=self.project,
            mode="online",
            public=False,
            id=self.swanlab_id,
            resume="allow",
            logdir=str(self.logdir),
            settings=settings,
        )
        self.ready = True
        url = getattr(run, "url", None)
        get_url = getattr(run, "get_url", None)
        if not url and callable(get_url):
            try:
                url = get_url()
            except Exception:
                url = None
        return url if isinstance(url, str) and url else None

    def log(self, metrics: dict[str, int | float], step: int) -> None:
        self.initialize()
        self.sdk.log(metrics, step=step)

    def finish(self) -> None:
        if self.ready:
            self.sdk.finish()
            self.ready = False


class Sidecar:
    def __init__(
        self,
        *,
        run_root: Path,
        uploader: SwanLabUploader,
        gpu_sampler: GpuSampler | None = None,
        interval: float = 2.0,
    ) -> None:
        self.root = run_root
        self.uploader = uploader
        self.gpu = gpu_sampler or GpuSampler()
        self.interval = max(0.2, float(interval))
        self.state_path = run_root / "swanlab-sidecar-state.json"
        self.status_path = run_root / "swanlab-sidecar-status.json"
        self.cursor = Cursor.load(self.state_path)
        # A new uploader session needs its own finish, including after a restart.
        self.cursor.finished_sdk = False

    def _finish(self) -> None:
        if not self.cursor.finished_sdk:
            self.uploader.finish()
            self.cursor.finished_sdk = True

    def _save(self) -> None:
        _write_json(self.state_path, self.cursor.as_dict())
        _write_json(
            self.status_path,
            {
                "schema": SCHEMA,
                "status": self.cursor.status,
                "url": self.cursor.url,
                "last_step": self.cursor.last_step,
                "train_steps": len(self.cursor.train_steps),
                "validation_steps": len(self.cursor.validation_steps),
                "finished": self.cursor.finished,
                "error": self.cursor.error,
            },
        )

    def poll_once(self) -> bool:
        updates = _read_jsonl(self.root / "updates.jsonl")
        validations = _read_jsonl(self.root / "validation.jsonl")
        status_path = self.root / "run-status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        )
        status_text, finished, paused, budget = _status(status)
        train = {
            int(row["new_global_updates"]): row for row in updates if "new_global_updates" in row
        }
        val = {int(row["step"]): row for row in validations if "step" in row}
        try:
            session_url = self.uploader.initialize()
            self.cursor.url = self.cursor.url or session_url
            for step in sorted(set(train) | set(val)):
                new_train = step in train and step not in self.cursor.train_steps
                new_val = step in val and step not in self.cursor.validation_steps
                if not (new_train or new_val):
                    continue
                metrics = {}
                if step in train:
                    metrics.update(_update_metrics(train[step]))
                if step in val:
                    metrics.update(_validation_metrics(val[step]))
                latest_step = max(train) if train else None
                if budget is not None and step == latest_step:
                    metrics.setdefault("budget/remote_gpu_wall_seconds", budget)
                if metrics:
                    self.uploader.log(metrics, step=step)
                    self.cursor.train_steps.add(step) if new_train else None
                    self.cursor.validation_steps.add(step) if new_val else None
                    self.cursor.last_step = step
            gpu = self.gpu.sample()
            if gpu and self.cursor.last_step is not None:
                self.uploader.log(gpu, step=self.cursor.last_step)
            self.cursor.status = (
                "waiting_paused" if paused else "finished" if finished else "running"
            )
            self.cursor.finished = finished
            self.cursor.error = None
        except SidecarError as exc:
            self.cursor.status = "auth_error"
            self.cursor.error = str(exc)
            raise
        except Exception as exc:
            self.cursor.status = "upload_error"
            self.cursor.error = type(exc).__name__
        if self.cursor.finished:
            self._finish()
        self._save()
        return self.cursor.finished

    def run(self) -> int:
        stop = {"value": False}

        def handler(*_: Any) -> None:
            stop["value"] = True

        previous = {sig: signal.signal(sig, handler) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            while not stop["value"]:
                if self.poll_once():
                    break
                time.sleep(self.interval)
        finally:
            if self.uploader.ready:
                self._finish()
            self.cursor.status = "finished" if self.cursor.finished else "stopped"
            self._save()
            for sig, old in previous.items():
                signal.signal(sig, old)
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--swanlab-id", default=DEFAULT_ID)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--gpu-interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    import swanlab  # type: ignore[import-not-found]

    root = Path(args.run_root)
    sidecar = Sidecar(
        run_root=root,
        uploader=SwanLabUploader(
            sdk=swanlab,
            logdir=root / "swanlab",
            project=args.project,
            swanlab_id=args.swanlab_id,
        ),
        gpu_sampler=GpuSampler(args.gpu_interval),
        interval=args.interval,
    )
    return sidecar.run() if not args.once else (0 if not sidecar.poll_once() else 0)


if __name__ == "__main__":
    raise SystemExit(main())
