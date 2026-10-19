#!/usr/bin/env python3
"""写入 / 刷新 N02 v1.2 进度页的「界面测试」样例 run（不碰真实作业目录）。

样例放在 <run-root>/P2/ui-sample-raw 与 <run-root>/P2/ui-sample-rgb，status.json 里带
ui_test=true，页面会显示「界面测试」徽标；它们不是训练结果。

用法：
  python3 n02_v12_seed_sample.py --run-root <path>              # 只写一次静态样例
  python3 n02_v12_seed_sample.py --run-root <path> --animate --ticks 400 --interval 3
  python3 n02_v12_seed_sample.py --run-root <path> --clean      # 删除样例目录
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from pathlib import Path

from owac.runtime.n02_v12_status import append_history, update_status

SAMPLE_LABEL = "界面测试"
# rgb 样例故意只给 raw/rgb 两个槽位：用来验证「尚未生成」分支。
ARM_SPECS = (
    ("ui-sample-raw", "raw_n02", 0, ("raw", "rgb", "attention", "residual")),
    ("ui-sample-rgb", "rgb_n02", 1, ("raw", "rgb")),
)
CAMERAS = (("camera_high", (0, 7), (0, 1)), ("camera_left", (0,), (0,)))


def _tint(text: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return digest[0], digest[1], digest[2]


def make_svg(kind: str, camera: str, frame: int, loop: int, seed: int) -> str:
    """生成一张带「界面测试」字样的占位图（确定性伪随机，仅用于验证渲染）。"""
    cols, rows, cell = 8, 6, 40
    width, height = cols * cell, rows * cell
    parts = []
    for row in range(rows):
        for col in range(cols):
            key = f"{kind}:{camera}:{frame}:{loop}:{seed}:{row}:{col}"
            r, g, b = _tint(key)
            if kind == "raw":
                r = g = b = 40 + (r + g + b) % 180
            elif kind == "rgb":
                r, g, b = 30 + r % 190, 30 + g % 190, 30 + b % 190
            elif kind == "attention":
                base = 60 + (r + g) % 70
                heat = (row * cols + col) % max(1, cols * rows)
                r, g, b = min(255, base + heat * 3), base // 2, base // 3
            else:  # residual
                v = 255 if (r + g + b) % 3 == 0 else 20
                r = g = b = v
            parts.append(
                f'<rect x="{col * cell}" y="{row * cell}" width="{cell}" height="{cell}" '
                f'fill="rgb({r},{g},{b})"/>'
            )
    if kind == "attention":
        parts.append(
            f'<circle cx="{width // 2}" cy="{height // 2}" r="{cell}" '
            'fill="#ff2d2d" fill-opacity="0.45"/>'
        )
        parts.append(
            f'<circle cx="{width // 3}" cy="{height // 2}" r="{cell // 2}" '
            'fill="#ffd400" fill-opacity="0.5"/>'
        )
    header = f"{SAMPLE_LABEL} · {camera} · frame {frame} · 内循环 {loop} · {kind}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 46}" '
        f'viewBox="0 0 {width} {height + 46}">'
        f'<rect width="{width}" height="{height + 46}" fill="#ffffff"/>'
        + "".join(parts)
        + f'<rect x="0" y="{height}" width="{width}" height="46" fill="#f2f5f2"/>'
        + f'<text x="8" y="{height + 19}" font-family="sans-serif" font-size="13" fill="#17352a">'
        + f"{header}</text>"
        + f'<text x="8" y="{height + 37}" font-family="sans-serif" font-size="12" fill="#8a5a00">'
        + "样例数据，非训练结果</text>"
        + "</svg>"
    )


def write_previews(run_dir: Path, run_id: str, seed: int, kinds) -> list[dict]:
    records = []
    for camera, frames, loops in CAMERAS:
        for frame in frames:
            for loop in loops:
                record = {"camera": camera, "frame": frame, "inner_loop": loop}
                for kind in kinds:
                    relative = f"preview/{camera}_f{frame:04d}_loop{loop}_{kind}.svg"
                    path = run_dir / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():  # 样例图是静态的，只在首次写入，避免每 tick 重复生成
                        path.write_text(make_svg(kind, camera, frame, loop, seed), encoding="utf-8")
                    record[kind] = relative
                records.append(record)
    return records


def write_run(
    run_root: Path, run_id: str, arm: str, gpu: int, tick: int, ticks: int, kinds
) -> None:
    run_dir = run_root / "P2" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if tick == 0:  # 样例是新的一轮演示：清空旧 history，避免曲线出现回跳
        (run_dir / "history.jsonl").unlink(missing_ok=True)
    target, total_loops = 1000, 100
    step = min(target, 4 + tick * 4)
    loss = round(0.92 * (0.985**tick) + 0.03, 4)
    dev = round(1.31 * (0.988**tick) + 0.08, 4)
    completed = min(total_loops, 3 + tick // 4)
    success = max(0, completed - 1)
    failed = 1 if completed >= 4 else 0
    infra = 1 if tick >= 40 else 0
    tail = [
        f"[{SAMPLE_LABEL}] 样例 run {run_id}：这不是训练结果",
        f"tick={tick} step={step}/{target} loss={loss} dev_action_error={dev}",
        "闭环：planned=100 只用于进度条；成功率分母是已完成回合 n",
    ] + [f"样例日志行 {i:02d} —— 用于验证「最近 30 行日志」区块" for i in range(1, 34)]

    update_status(
        run_dir,
        run_id=run_id,
        stage="P2",
        arm=arm,
        platform="robotwin",
        seed=1103,
        ui_test=True,
        state="running",
        activity=f"{SAMPLE_LABEL}：第 {tick + 1}/{ticks} 次样例刷新（非训练结果）",
        gpu=gpu,
        step=step,
        target_steps=target,
        step_seconds=0.42,
        loss=loss,
        dev_action_error=dev,
        checkpoint=None if tick < 1 else f"checkpoints/{run_id}/step-{step:06d}.pt",
        preview_files=write_previews(run_dir, run_id, 1103, kinds),
        log_tail=tail,
        eval={
            "planned": total_loops,
            "completed": completed,
            "success": success,
            "task_failure": failed,
            "infrastructure_error": infra,
        },
    )
    append_history(
        run_dir, step=step, wall_time=round(0.42 * step, 3), train_loss=loss, dev_action_error=dev
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=f"N02 v1.2 {SAMPLE_LABEL} 样例写入端")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--ticks", type=int, default=1)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    root = Path(args.run_root)
    if args.clean:
        for run_id, _, _, _kinds in ARM_SPECS:
            shutil.rmtree(root / "P2" / run_id, ignore_errors=True)
        print(f"removed samples under {root / 'P2'}")
        return 0

    rounds = args.ticks if args.animate else 1
    for tick in range(rounds):
        for run_id, arm, gpu, kinds in ARM_SPECS:
            write_run(root, run_id, arm, gpu, tick, max(rounds, 1), kinds)
        if tick + 1 < rounds:
            time.sleep(args.interval)
    names = ", ".join(spec[0] for spec in ARM_SPECS)
    print(f"{SAMPLE_LABEL} 样例已写入 {root / 'P2'}（run: {names}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
