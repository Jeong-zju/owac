"""CLI: offline teacher/student action comparison on LIBERO validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-sources", type=int, default=20)
    args = parser.parse_args(argv)
    from owac.evaluation.raw_all_bench_actions import run

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config.setdefault("assets", {})
    config["assets"]["tokenizer"] = args.tokenizer
    config["assets"]["statistics"] = args.statistics
    report = run(
        config,
        args.inventory,
        args.checkpoint,
        args.output,
        device=args.device,
        max_sources=args.max_sources,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "windows": report["windows"],
                "sources": report["sources"],
                "output": args.output,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
