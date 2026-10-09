"""Render a self-contained local progress page from archived numerical evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def read_rows(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    # A live curve/episode file may be copied while its final line is being written.
    return [json.loads(line) for line in lines if line.endswith("\n")]


def first_file(root, *names):
    return next((root / name for name in names if (root / name).exists()), root / names[0])


def training_points(rows, width=50):
    points = []
    for start in range(0, len(rows), width):
        block = rows[start : start + width]
        values = [row["train_flow_mse"] for row in block]
        if values and all(math.isfinite(value) for value in values):
            points.append([block[-1]["step"], sum(values) / len(values)])
    return points


def load_payload(directory):
    protocol = read_json(directory / "protocol.json")
    status = read_json(directory / "status.json")
    monitor = read_json(directory / "monitor.json")
    check = read_json(directory / "progress/check.json")
    dataset = read_json(first_file(directory, "dataset.json", "progress/dataset.json"))
    training = read_json(first_file(directory, "training.json", "progress/training.json"))
    rows = read_rows(first_file(directory, "curve.jsonl", "progress/curve.jsonl"))
    evaluation = read_json(directory / "evaluation/summary.json")
    episodes = read_rows(directory / "evaluation/episodes.jsonl")
    checked = check.get("checked_at_utc") or monitor.get("last_checked_utc")
    step = rows[-1]["step"] if rows else training.get("completed_steps", 0)
    splits = {
        split: {
            "episodes": sum(item["split"] == split for item in dataset.get("episodes", [])),
            "frames": sum(
                item["frames"] for item in dataset.get("episodes", []) if item["split"] == split
            ),
        }
        for split in ("train", "validation")
    }
    validations = (
        [[0, training["initial_validation_loss"]]]
        if training.get("initial_validation_loss") is not None
        else []
    )
    validations.extend(
        [row["step"], row["validation_flow_mse"]] for row in rows if "validation_flow_mse" in row
    )
    return {
        "checked_at": checked,
        "check_error": check.get("check_error") or monitor.get("last_check_error"),
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "stage": status.get("stage", monitor.get("stage", "collect")),
        "step": step,
        "last_step": rows[-1] if rows else {},
        "protocol": protocol,
        "training": training,
        "dataset_status": dataset.get("status"),
        "splits": splits,
        "collection_attempts": len(dataset.get("attempts", [])),
        "collection_failures": sum(
            item["status"] == "failed" for item in dataset.get("attempts", [])
        ),
        "chart": {"train": training_points(rows), "validation": validations, "bin_steps": 50},
        "evaluation": evaluation,
        "episodes": [
            {key: row.get(key) for key in ("seed", "success", "status", "control_steps", "error")}
            for row in episodes
        ],
        # Zero attempted tests is an unmeasured score, not 0% success.
        "success_rate": sum(row["success"] for row in episodes) / len(episodes)
        if episodes
        else None,
        "gpu_sample": check.get("gpu_sample", {}),
        "history": read_rows(directory / "monitor-history.jsonl"),
    }


def render(directory, output):
    payload = load_payload(directory)
    history = payload["history"]
    if payload["checked_at"] and not any(
        row["checked_at"] == payload["checked_at"] for row in history
    ):
        row = {
            "checked_at": payload["checked_at"],
            "stage": payload["stage"],
            "status": payload["status"].get("status"),
            "step": payload["step"],
            "test_episodes": len(payload["episodes"]),
            "train_episodes": payload["splits"]["train"]["episodes"],
            "validation_episodes": payload["splits"]["validation"]["episodes"],
        }
        with (directory / "monitor-history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        history.append(row)
    payload["history"] = sorted(history, key=lambda row: row["checked_at"])[-12:]
    payload["links"] = {
        "experiment": os.path.relpath(
            directory.parents[1] / "2026-10-09_pi05_lw_mug_baseline.md", output.parent
        ),
        "data": os.path.relpath(directory / "dataset.json", output.parent),
        "training": os.path.relpath(
            first_file(directory, "training.json", "progress/training.json"), output.parent
        ),
    }
    # Source metadata/errors are untrusted; keep them inside JSON and use textContent in the page.
    packed = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("baseline_dashboard_template.html").read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(template.replace("__PAYLOAD__", packed))
    temporary.replace(output)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/cloud-pi05-baseline/progress.html")
    )
    args = parser.parse_args()
    payload = render(args.directory.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "html": str(args.output.resolve()),
                "stage": payload["stage"],
                "step": payload["step"],
                "checked_at": payload["checked_at"],
            }
        )
    )


if __name__ == "__main__":
    main()
