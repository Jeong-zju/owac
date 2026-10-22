"""CLI for the all-RawVLA-Bench tokens-only distillation path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_prepare(args: argparse.Namespace) -> int:
    from owac.raw.all_bench_distillation import (
        TargetStore,
        build_vision_teacher,
        load_inventory,
        prepare_source,
    )

    config = _load(args.config)
    sources, _ = load_inventory(args.inventory)
    sources = [s for s in sources if s.split in {"train", "validation"}]
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    mine = sources[rank::world]
    cap = int(args.disk_cap_gib * (1024**3))
    store = TargetStore(args.temp / "targets", max_bytes=cap)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    teacher = build_vision_teacher(config, device=f"cuda:{local_rank}")
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output / "prepare-progress.jsonl"
    with progress.open("a", encoding="utf-8") as stream:
        for index, source in enumerate(mine, 1):
            info = prepare_source(teacher, source, store)
            stream.write(
                json.dumps(
                    {
                        "source_id": source.group_id,
                        "split": source.split,
                        "rows": info["windows"],
                        "rank": rank,
                        "index": index,
                        "total": len(mine),
                    }
                )
                + "\n"
            )
            stream.flush()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from owac.raw.all_bench_distillation import train_all_bench

    config = _load(args.config)
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    result = train_all_bench(config, args, device=torch.device(f"cuda:{local}"))
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--temp", default="/root/autodl-tmp/owac-all-bench-20261022")
    parser.add_argument(
        "--output", default="/root/owac/repo/outputs/raw-distillation-all-bench-20261022"
    )
    parser.add_argument("--disk-cap-gib", type=float, default=380.0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.set_defaults(func=cmd_prepare)
    t = sub.add_parser("train")
    t.set_defaults(func=cmd_train)
    args = parser.parse_args(argv)
    args.temp = Path(args.temp)
    args.output = Path(args.output)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
