"""N02 v1.2 run 状态写入端（纯标准库，训练进程可每步调用）。

契约（与监控端 n02_v12_monitor.py 共用）：
  <run_dir>/status.json   单对象，字段：
    run_id, stage, arm, platform, seed, state, activity, gpu, step, target_steps,
    updated_at, started_at, step_seconds, loss, dev_action_error, checkpoint,
    eval{planned, completed, success, task_failure, infrastructure_error},
    preview_files, log_tail
    不适用项写 null。
  <run_dir>/history.jsonl 每行一个对象：step, wall_time, train_loss, dev_action_error

写入一律「临时文件 + os.replace」原子替换，读者只会看到旧版本或新版本，不会看到半个文件。
heartbeat() 是给长步骤用的轻量心跳（约每 5 秒一次），不调用 fsync。
目录不存在时自动创建。

用法：
    from owac.runtime.n02_v12_status import update_status, append_history, heartbeat

    update_status(run_dir, run_id="raw-n02-robotwin-s1103", stage="P2", arm="raw_n02",
                  platform="robotwin", seed=1103, state="running", step=0,
                  target_steps=1000, loss=None, dev_action_error=None)
    append_history(run_dir, step=10, wall_time=12.5, train_loss=0.41, dev_action_error=1.72)
    heartbeat(run_dir)                 # 只刷新 updated_at
    heartbeat(run_dir, activity="保存 checkpoint 中")   # 也可顺带带字段
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

__all__ = [
    "update_status",
    "append_history",
    "heartbeat",
    "read_status",
    "status_path",
    "history_path",
]

_LOCK = threading.Lock()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def status_path(run_dir) -> Path:
    return Path(run_dir) / "status.json"


def history_path(run_dir) -> Path:
    return Path(run_dir) / "history.jsonl"


def read_status(run_dir) -> dict:
    """读回当前 status.json；文件缺失或损坏时返回 {}。"""
    try:
        with open(status_path(run_dir), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_status_unlocked(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write_json(path: Path, payload: dict, *, fsync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".status-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"), default=str)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_status(run_dir, **fields) -> dict:
    """读-改-写 status.json：只覆盖传入字段，其余保持原值。

    自动写入 updated_at；首次调用时补 started_at（= 本次时间）。
    返回合并后的完整 dict。
    """
    path = status_path(run_dir)
    with _LOCK:
        payload = _read_status_unlocked(path)
        payload.update(fields)
        now = _utc_now()
        payload["updated_at"] = now
        payload.setdefault("started_at", now)
        _atomic_write_json(path, payload, fsync=True)
        return payload


def heartbeat(run_dir, **fields) -> dict:
    """轻量心跳：刷新 updated_at（可选顺带若干字段），不做 fsync。

    适合训练长步骤里由后台线程每 5 秒调用一次。
    """
    path = status_path(run_dir)
    with _LOCK:
        payload = _read_status_unlocked(path)
        payload.update(fields)
        payload["updated_at"] = _utc_now()
        _atomic_write_json(path, payload, fsync=False)
        return payload


def append_history(run_dir, step, wall_time, train_loss, dev_action_error=None, **metrics) -> dict:
    """向 history.jsonl 追加一行 {step, wall_time, train_loss, dev_action_error}。

    只 append、不重写整个文件，训练步内开销是一次 write()。
    """
    record = {
        "step": step,
        "wall_time": wall_time,
        "train_loss": train_loss,
        "dev_action_error": dev_action_error,
        **metrics,
    }
    path = history_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str))
        fh.write("\n")
    return record
