"""CLI for paired teacher/student LIBERO task6 closed-loop rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--libero-config", default=None)
    args = parser.parse_args(argv)
    from owac.simulation.raw_student_closed_loop import run

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = run(
        config,
        args.catalog,
        args.checkpoint,
        args.output,
        device=args.device,
        max_episodes=args.max_episodes,
        libero_config=args.libero_config,
    )
    print(json.dumps({"status": "completed", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
