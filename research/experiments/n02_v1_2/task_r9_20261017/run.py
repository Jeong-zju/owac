"""Bounded paired adaptation. Predeclared development gate precedes physical rollout."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from owac.runtime.n02_v12_status import update_status
from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import Heartbeat


def selection(output):
    baseline = json.loads((output / "dev_baseline.json").read_text())
    config = (
        json.loads((output / "run_config.json").read_text())
        if (output / "run_config.json").exists()
        else None
    )
    limit = 1.05 if config is None else config["task_adaptation"]["normal_opening_max_ratio"]
    opening = baseline["task_control"]["strata"]["Normal/opening"]["mse"]
    choices = []
    manifest = baseline["task_dev_manifest_sha256"]
    for path in output.glob("checkpoint_*.metrics.json"):
        m = json.loads(path.read_text())
        checkpoint = output / f"checkpoint_{m['step']:07d}.pt"
        if not checkpoint.is_file() or m.get("task_dev_manifest_sha256") != manifest:
            continue
        score = m["sampled_control_mse"]
        start = m["task_control"]["strata"]["Normal/opening"]["mse"]
        if math.isfinite(score) and math.isfinite(start):
            choices.append(
                dict(
                    step=m["step"],
                    score=score,
                    opening=start,
                    checkpoint=str(checkpoint),
                    eligible=score < baseline["sampled_control_mse"] and start <= opening * limit,
                )
            )
    eligible = [c for c in choices if c["eligible"]]
    selected = min(eligible, key=lambda c: (c["score"], c["step"])) if eligible else None
    return dict(
        baseline_score=baseline["sampled_control_mse"],
        baseline_opening=opening,
        normal_opening_max_ratio=limit,
        candidates=choices,
        selected=selected,
        rule="improve balanced sampled14D MSE and Normal opening <= baseline x1.05",
    )


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    begin = time.monotonic()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "owac.training.n02_v12",
            "--config",
            args.config,
            "--output",
            str(output),
        ],
        check=True,
    )
    write_json(output / "run_config.json", json.loads(Path(args.config).read_text()))
    decision = selection(output)
    write_json(output / "selection.json", decision)
    chosen = decision["selected"]
    if not chosen:
        update_status(
            output, state="completed", activity="本轮完成：未通过开场与整段动作验证，未晋级闭环"
        )
        write_json(
            output / "pipeline_result.json",
            dict(
                state="completed",
                promoted=False,
                decision=decision,
                physical_accepted=None,
                wall_seconds=time.monotonic() - begin,
            ),
        )
        return
    case = Path(args.case + f"-step{chosen['step']:04d}")
    update_status(
        output,
        state="evaluating",
        activity="动作验证通过，正在做真实抓持验收",
        checkpoint=chosen["checkpoint"],
    )
    rollout = Path(__file__).parents[1] / "representation_r8_20261016/rollout.py"
    with Heartbeat(output, "正在进行真实抓持验收"):
        subprocess.run(
            [
                sys.executable,
                str(rollout),
                "--config",
                args.config,
                "--checkpoint",
                chosen["checkpoint"],
                "--output",
                str(case),
                "--port",
                str(args.port),
                "--replan",
                "10",
                "--joint-delta",
                "0.05",
                "--max-steps",
                "400",
                "--timeout",
                "1800",
            ],
            check=True,
        )
    physical = json.loads((case / "result.json").read_text())
    accepted = physical.get("accepted_success")
    if not isinstance(accepted, bool):
        raise RuntimeError("physical acceptance unavailable")
    write_json(
        output / "pipeline_result.json",
        dict(
            state="completed",
            promoted=True,
            decision=decision,
            selected_step=chosen["step"],
            physical_accepted=accepted,
            case=str(case),
            wall_seconds=time.monotonic() - begin,
        ),
    )
    update_status(
        output,
        state="completed",
        activity="训练完成：真实抓持通过" if accepted else "训练完成：真实抓持未通过",
        eval=dict(
            planned=1,
            completed=1,
            success=int(accepted),
            task_failure=int(not accepted),
            infrastructure_error=0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if Path(args.output).is_dir():
            update_status(Path(args.output), state="error", activity="本轮运行异常，请查看详细记录")
            write_json(Path(args.output) / "pipeline_error.json", dict(error=repr(error)))
        raise
