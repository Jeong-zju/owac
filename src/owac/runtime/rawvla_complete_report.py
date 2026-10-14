"""Export audited complete scores and compact paired-case references, never images or weights."""

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from owac.runtime.rawvla_eval_audit import audit, entry_hash, percentile
from owac.runtime.rawvla_full_eval import digest, paired_seed, rows, write_json

DOMAINS = ("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver")


def metrics(records):
    count = len(records)
    success = sum(int(row["success"]) for row in records)
    steps = [row["num_steps"] for row in records]
    wall = [row["wall_seconds"] for row in records]
    return {
        "completed": count,
        "success": success,
        "policy_failure": count - success,
        "success_rate_percent": 100 * success / count if count else None,
        "actual_policy_steps": sum(steps),
        "steps_mean": sum(steps) / count if count else None,
        "steps_p50": percentile(steps, 0.5),
        "steps_p95": percentile(steps, 0.95),
        "episode_wall_mean_seconds": sum(wall) / count if count else None,
        "episode_wall_p50_seconds": percentile(wall, 0.5),
        "episode_wall_p95_seconds": percentile(wall, 0.95),
        "summed_episode_wall_seconds": sum(wall),
    }


def grouped_metrics(records, platform):
    by_domain, by_task, by_suite = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in records:
        entry = row.get("rawvla_bench_entry", row.get("entry"))
        task = (
            f"{entry['suite']}/{entry['task_id']}" if platform == "libero" else entry["task_name"]
        )
        by_domain[entry["lighting_domain"]].append(row)
        by_task[task].append(row)
        if platform == "libero":
            by_suite[entry["suite"]].append(row)
    tasks = {}
    for task, values in sorted(by_task.items()):
        label = values[0].get("task_description", task)
        tasks[task] = {
            "description": label,
            "overall": metrics(values),
            "domains": {
                domain: metrics(
                    [
                        row
                        for row in values
                        if row.get("rawvla_bench_entry", row.get("entry"))["lighting_domain"]
                        == domain
                    ]
                )
                for domain in DOMAINS
            },
        }
    result = {
        "overall": metrics(records),
        "domains": {domain: metrics(by_domain[domain]) for domain in DOMAINS},
        "tasks": tasks,
    }
    if platform == "libero":
        result["suites"] = {suite: metrics(values) for suite, values in sorted(by_suite.items())}
        result["paper_layout"] = {}
        for domain in (*DOMAINS, "Overall"):
            chosen = records if domain == "Overall" else by_domain[domain]
            sog = [row for row in chosen if row["rawvla_bench_entry"]["suite"] != "libero_10"]
            long = [row for row in chosen if row["rawvla_bench_entry"]["suite"] == "libero_10"]
            result["paper_layout"][domain] = {
                "SOG": metrics(sog),
                "Long": metrics(long),
                "Avg": metrics(chosen),
            }
    return result


def collect_rpc(parent, repair, main_records, paper_records):
    evidence = parent.parent / "pi05-full-20261010-v2/robotwin-policy"
    accepted = {}
    for scope, records in (
        ("robotwin_main", main_records),
        ("robotwin_original_conditions", paper_records),
    ):
        for row in records:
            run = Path(row["directory"]).parents[2].name
            key = f"{run}/{row['case_id']}"
            seed = paired_seed("robotwin", row["entry"]["task_name"], row["entry"]["dataset_seed"])
            accepted[key] = (scope, seed, math.ceil(row["num_steps"] / 50))
    sessions = {}
    prefixes = {key.split("/")[0] for key in accepted}
    with (evidence / "inference.jsonl").open() as stream:
        for line in stream:
            if not any(prefix in line for prefix in prefixes):
                continue
            record = json.loads(line)
            key = record.get("case_id")
            if key not in accepted:
                continue
            _, seed, _ = accepted[key]
            if record["flow_seed"] == seed:
                sessions[key] = []
            if key not in sessions:
                continue
            calls = sessions[key]
            if record["flow_seed"] != (seed + 50 * len(calls)) % (2**32):
                raise ValueError(f"RPC chunk seed continuity failed: {key}")
            if record["shape"] != [50, 14] or not record["finite"]:
                raise ValueError(f"RPC action protocol failed: {key}")
            calls.append(
                {
                    "inference_seconds": record["inference_seconds"],
                    "input": record["input"],
                    "peak_gpu_bytes": record["peak_gpu_bytes"],
                }
            )
    result = {"source": str(evidence), "scopes": {}, "coverage_errors": []}
    for key, (_, _, expected) in accepted.items():
        if len(sessions.get(key, [])) != expected:
            result["coverage_errors"].append(
                {
                    "case_id": key,
                    "expected_calls": expected,
                    "observed_calls": len(sessions.get(key, [])),
                }
            )
    for scope in ("robotwin_main", "robotwin_original_conditions"):
        keys = [key for key, value in accepted.items() if value[0] == scope]
        calls = [call for key in keys for call in sessions.get(key, [])]
        seconds = [call["inference_seconds"] for call in calls]
        result["scopes"][scope] = {
            "episodes": len(keys),
            "calls": len(calls),
            "gpu_forward_mean_seconds": sum(seconds) / len(seconds) if seconds else None,
            "gpu_forward_p50_seconds": percentile(seconds, 0.5),
            "gpu_forward_p95_seconds": percentile(seconds, 0.95),
            "service_lifetime_peak_allocated_bytes": max(
                (call["peak_gpu_bytes"] for call in calls), default=0
            ),
        }
    result["timing_scope"] = (
        "Synchronized model forward only, excludes NPZ, RPC and simulation; shared GPU"
    )
    result["recovery_prefix"] = repair.name
    return result


def audit_robotwin_actions(records):
    """Check every counted episode against its stored, actually applied policy trajectory."""
    steps = 0
    for record in records:
        case = record["case_id"]
        if (
            record["eval_seed"] != record["entry"]["dataset_seed"]
            or record["action_source"] != "model_prediction"
            or record["image_input_key"] != "render_rgb"
        ):
            raise ValueError(f"Counted RoboTwin policy identity/source differs: {case}")
        path = Path(record["directory"]) / "actions.jsonl"
        if digest(path) != record["actions_sha256"]:
            raise ValueError(f"Stored actual trajectory hash differs: {case}")
        count = 0
        with path.open() as stream:
            for line in stream:
                action = json.loads(line)
                if (
                    action["step"] != count
                    or len(action["state"]) != 14
                    or len(action["action"]) != 14
                    or not all(math.isfinite(x) for x in action["state"] + action["action"])
                ):
                    raise ValueError(f"Invalid actual RoboTwin trajectory: {case}/{count}")
                count += 1
        if count != record["num_steps"] or count <= 0:
            raise ValueError(f"Incomplete actual trajectory: {case}")
        steps += count
    return {"episodes": len(records), "actual_policy_steps": steps, "errors": []}


def export(parent, repair, output):
    report = audit(parent)
    if report["state"] != "complete" or report["errors"] or report["completed"] != 13250:
        raise ValueError("Cannot export a complete report with an incomplete original manifest")
    main = rows(parent / "robotwin/episodes.jsonl")
    libero_files = sorted((parent / "libero").glob("*/*.episodes.jsonl"))
    libero = [row for path in libero_files for row in rows(path)]
    paper = rows(parent / "robotwin-paper/episodes.jsonl")
    if len(paper) != 632 or len({row["case_id"] for row in paper}) != 632:
        raise ValueError("Original-condition evaluation is incomplete")
    action_audit = audit_robotwin_actions(main + paper)
    rpc = collect_rpc(parent, repair, main, paper)
    output.mkdir(parents=True, exist_ok=False)
    compact = output / "per-episode-baseline.jsonl"
    with compact.open("w") as stream:
        for platform, records in (
            ("libero", libero),
            ("robotwin", main),
            ("robotwin-original", paper),
        ):
            for row in records:
                entry = row.get("rawvla_bench_entry", row.get("entry"))
                task = (
                    f"{entry['suite']}/{entry['task_id']}"
                    if platform == "libero"
                    else entry["task_name"]
                )
                base = entry["init_state_index"] if platform == "libero" else entry["dataset_seed"]
                language = row.get("language", row.get("task_description", ""))
                item = {
                    "platform": platform,
                    "task": task,
                    "base_id": base,
                    "domain": entry["lighting_domain"],
                    "entry_sha256": entry_hash(entry),
                    "language_sha256": hashlib.sha256(language.encode()).hexdigest(),
                    "success": row["success"],
                    "num_steps": row["num_steps"],
                    "wall_seconds": row["wall_seconds"],
                    "policy_flow_seed": row.get(
                        "policy_flow_seed", paired_seed("robotwin", task, base)
                    ),
                    "actions_sha256": row.get("actions_sha256"),
                }
                stream.write(json.dumps(item, sort_keys=True) + "\n")
    write_json(output / "rpc-summary.json", rpc)
    write_json(output / "robotwin-action-audit.json", action_audit)
    protocol = json.loads((parent / "protocol.json").read_text())
    result = {
        "version": "owac-rawvla-pi05-complete-20261014-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state": "complete",
        "main_completed": 13250,
        "original_conditions_completed": 632,
        "protocol_sha256": digest(parent / "protocol.json"),
        "manifest_sha256": {
            name: digest(parent / f"{name}-manifest.json") for name in ("libero", "robotwin")
        },
        "weights_sha256": protocol["weights_sha256"],
        "source_commit": protocol["source_commit"],
        "repair_adoption": json.loads((repair / "adoption.json").read_text()),
        "libero": grouped_metrics(libero, "libero"),
        "robotwin": grouped_metrics(main, "robotwin"),
        "original_conditions": grouped_metrics(paper, "robotwin"),
        "per_episode_numeric_sha256": digest(compact),
        "rpc_summary_sha256": digest(output / "rpc-summary.json"),
        "rpc_coverage_errors": rpc["coverage_errors"],
        "robotwin_action_audit": action_audit,
        "source_ledgers_sha256": {
            str(path): digest(path)
            for path in [
                *libero_files,
                parent / "robotwin/episodes.jsonl",
                parent / "robotwin-paper/episodes.jsonl",
            ]
        },
        "cost_scope": (
            "Episode wall includes initialization/expert precheck/simulation/RPC/logging; "
            "concurrent runs share GPU"
        ),
        "limitations": protocol["limitations"],
        "extra_training_steps": 0,
    }
    write_json(output / "complete-summary.json", result)
    write_json(output / "identity-audit.json", report)
    hashes = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "checksums.json", hashes)
    print(
        json.dumps(
            {
                "state": "complete",
                "libero": result["libero"]["overall"],
                "robotwin": result["robotwin"]["overall"],
                "original_conditions": result["original_conditions"]["overall"],
                "rpc_coverage_errors": len(rpc["coverage_errors"]),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--repair", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.parent, args.repair, args.output)


if __name__ == "__main__":
    main()
