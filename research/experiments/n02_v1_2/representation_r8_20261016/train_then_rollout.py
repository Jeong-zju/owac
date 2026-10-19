"""One bounded adaptation run followed by one live physical grasp assessment."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from owac.runtime.n02_v12_status import update_status
from owac.runtime.rawvla_full_eval import write_json


def select_saved_checkpoint(output):
    """Rank actual saved weights by sampled actions, never by closed-loop outcomes."""
    candidates = []
    for path in output.glob("checkpoint_*.metrics.json"):
        record = json.loads(path.read_text())
        step, score = record.get("step"), record.get("sampled_control_mse")
        if (
            not isinstance(step, int)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score < 0
        ):
            continue
        checkpoint = output / f"checkpoint_{step:07d}.pt"
        if checkpoint.is_file():
            candidates.append((float(score), step, checkpoint))
    if not candidates:
        raise ValueError("no saved checkpoint has a finite complete-sampling score")
    return min(candidates)


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    command = [
        sys.executable,
        "-m",
        "owac.training.n02_v12",
        "--config",
        args.config,
        "--output",
        str(output),
    ]
    start = time.monotonic()
    try:
        subprocess.run(command, check=True)
        config = json.loads(Path(args.config).read_text())
        step = int(config["loop"]["max_updates"])
        checkpoint = output / f"checkpoint_{step:07d}.pt"
        selected_score = None
        if args.select_dev_best:
            selected_score, step, checkpoint = select_saved_checkpoint(output)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        update_status(
            output,
            state="evaluating",
            checkpoint=str(checkpoint),
            activity=f"训练完成，使用开发集选中的第{step}步进行闭环验收",
        )
        rollout = [
            sys.executable,
            str(Path(__file__).with_name("rollout.py")),
            "--config",
            args.config,
            "--checkpoint",
            str(checkpoint),
            "--output",
            args.case + f"-step{step:04d}" if args.select_dev_best else args.case,
            "--port",
            str(args.port),
            "--max-steps",
            "200",
            "--replan",
            str(args.replan),
        ]
        if args.joint_delta is not None:
            rollout.extend(["--joint-delta", str(args.joint_delta)])
        from owac.training.n02_v12 import Heartbeat

        case = Path(rollout[rollout.index("--output") + 1])
        with Heartbeat(output, f"正在验收第{step}步权重"):
            subprocess.run(rollout, check=True)
        result = json.loads((case / "result.json").read_text())
        accepted = result.get("accepted_success")
        if not isinstance(accepted, bool):
            raise RuntimeError("closed-loop result has no physical assessment")
        update_status(
            output,
            state="completed",
            activity="训练与闭环完成：抓持验收通过" if accepted else "训练与闭环完成：抓持未通过",
            eval={
                "planned": 1,
                "completed": 1,
                "success": int(accepted),
                "task_failure": int(not accepted),
                "infrastructure_error": 0,
            },
        )
        write_json(
            output / "pipeline_result.json",
            {
                "state": "completed",
                "training_command": command,
                "rollout_command": rollout,
                "physical_accepted": accepted,
                "case": str(case),
                "selected_step": step,
                "selected_sampled_control_mse": selected_score,
                "wall_seconds": time.monotonic() - start,
            },
        )
    except Exception as error:
        update_status(output, state="error", activity="训练或闭环运行异常，需要检查日志")
        write_json(
            output / "pipeline_result.json",
            {
                "state": "error",
                "error": repr(error),
                "wall_seconds": time.monotonic() - start,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--select-dev-best", action="store_true")
    parser.add_argument("--replan", type=int, default=50)
    parser.add_argument("--joint-delta", type=float)
    main(parser.parse_args())
