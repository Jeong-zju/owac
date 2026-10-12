"""Independent identity, denominator and progress audit for frozen full runs."""

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from owac.runtime.rawvla_full_eval import digest, write_json


def completed_rows(path):
    """Ignore a concurrently appended tail, reject malformed complete records."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text()
    return [
        json.loads(line)
        for line in text.splitlines(keepends=True)
        if line.endswith("\n") and line.strip()
    ]


def entry_hash(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(quantile * len(values)) - 1)]


def build_paper_subset(root, output):
    import numpy as np

    manifest_path = output / "robotwin-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    lookup = {
        (entry["task_name"], entry["dataset_seed"], entry["lighting_domain"]): (index, entry)
        for index, entry in enumerate(manifest["entries"])
    }
    cache = root / "source/benchmark_data/RawVLA-Bench"
    records, errors, differences = [], [], []
    for path in sorted((cache / "data/robotwin2/raw").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(arrays["metadata_json"].item())
        key = (metadata["task_name"], metadata["dataset_seed"], metadata["lighting_domain"])
        if key not in lookup:
            errors.append({"file": str(path), "issue": "Entry absent from full manifest"})
            continue
        index, entry = lookup[key]
        for field in ("lighting_ev", "lighting_scale", "noise_seed"):
            if abs(float(metadata[field]) - float(entry[field])) > 1e-12:
                differences.append({"file": str(path.relative_to(cache)), "field": field})
        original = dict(entry)
        for field in ("lighting_ev", "lighting_scale", "lighting_sample_unit", "noise_seed"):
            original[field] = metadata[field]
        original["episode_idx"] = metadata["episode"]
        records.append(
            {
                "case_id": f"paper-{len(records):04d}",
                "parent_case_id": f"robotwin-{index:04d}",
                "dataset_file": str(path.relative_to(cache)),
                "metadata_sha256": entry_hash(metadata),
                "parent_entry_sha256": entry_hash(entry),
                "entry": original,
            }
        )
    if len(records) != 632 or len({row["parent_case_id"] for row in records}) != 632 or errors:
        raise ValueError(
            f"Paper condition metadata mismatch: count={len(records)}, errors={errors[:3]}"
        )
    subset = {
        "status": "Candidate paper denominator and original cache conditions; "
        "not proof of the unpublished paper eval ledger",
        "source_dataset_revision": "8586c594fff9ce38580e29745f074e82e96e74f7",
        "parent_manifest_sha256": digest(manifest_path),
        "selection": "All 632 metadata identities and assigned lighting; no policy outcomes used",
        "entries_count": len(records),
        "domain_counts": dict(Counter(row["entry"]["lighting_domain"] for row in records)),
        "mismatches": errors,
        "differences_from_full_manifest": dict(Counter(row["field"] for row in differences)),
        "entries": records,
    }
    write_json(output / "robotwin-paper-conditions.json", subset)
    print(json.dumps({key: value for key, value in subset.items() if key != "entries"}))


def audit(output):
    protocol = json.loads((output / "protocol.json").read_text())
    report = {"protocol_sha256": digest(output / "protocol.json"), "platforms": {}, "errors": []}
    for platform in ("libero", "robotwin"):
        manifest = json.loads((output / f"{platform}-manifest.json").read_text())
        entries = manifest["entries"]
        expected = {entry_hash(entry) for entry in entries}
        if platform == "libero":
            episodes = [
                row
                for path in (output / "libero").glob("*/*.episodes.jsonl")
                for row in completed_rows(path)
            ]
        else:
            episodes = completed_rows(output / "robotwin/episodes.jsonl")
        observed = set()
        tasks = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))
        durations = []
        languages = defaultdict(set)
        errors = []
        for episode in episodes:
            entry = episode.get("rawvla_bench_entry", episode.get("entry"))
            identity = entry_hash(entry)
            if identity not in expected:
                errors.append("Result entry absent from frozen manifest")
            if identity in observed:
                errors.append("Duplicate completed episode")
            observed.add(identity)
            task = (
                f"{entry['suite']}/{entry['task_id']}"
                if platform == "libero"
                else entry["task_name"]
            )
            base = (
                (task, entry["init_state_index"])
                if platform == "libero"
                else (task, entry["dataset_seed"])
            )
            languages[base].add(episode.get("language", episode.get("task_description")))
            counts = tasks[task][entry["lighting_domain"]]
            counts["total"] += 1
            counts["success"] += int(episode["success"])
            if platform == "robotwin" and episode["eval_seed"] != entry["dataset_seed"]:
                errors.append("Dataset seed changed")
            if episode.get("num_steps", 0) <= 0:
                errors.append("No measured policy steps")
            durations.append(episode["wall_seconds"])
        if any(len(values) != 1 for values in languages.values()):
            errors.append("Language differs between paired lighting domains")
        domain_counts = defaultdict(lambda: {"success": 0, "total": 0})
        for domains in tasks.values():
            for domain, counts in domains.items():
                for key in counts:
                    domain_counts[domain][key] += counts[key]
        states = len(observed & expected)
        report["platforms"][platform] = {
            "state": "complete" if states == len(entries) and not errors else "running",
            "expected": len(entries),
            "completed": states,
            "remaining": len(entries) - states,
            "success": sum(episode["success"] for episode in episodes),
            "domains": dict(domain_counts),
            "tasks": {task: dict(domains) for task, domains in tasks.items()},
            "wall_seconds_p50": percentile(durations, 0.5),
            "wall_seconds_p95": percentile(durations, 0.95),
            "mean_episode_seconds": sum(durations) / len(durations) if durations else None,
            "errors": errors,
        }
        if platform == "robotwin":
            attempts = completed_rows(output / "robotwin/attempts.jsonl")
            failed_attempts = [row for row in attempts if row["returncode"] != 0]
            report["platforms"][platform]["historical_infrastructure_attempts"] = len(
                failed_attempts
            )
            report["platforms"][platform]["pending_failed_case_ids"] = sorted(
                {row["case_id"] for row in failed_attempts} - {row["case_id"] for row in episodes}
            )
            subset_path = output / "robotwin-paper/summary.json"
            if subset_path.exists():
                subset = json.loads(subset_path.read_text())
                report["additional_paper_conditions"] = subset
        if states == len(entries) and not errors:
            report["platforms"][platform]["final_success_rate_percent"] = (
                100 * sum(episode["success"] for episode in episodes) / states
            )
            for counts in domain_counts.values():
                counts["success_rate_percent"] = 100 * counts["success"] / counts["total"]
        report["errors"].extend(f"{platform}: {error}" for error in errors)
    report["expected"] = protocol["total_episodes"]
    report["completed"] = sum(item["completed"] for item in report["platforms"].values())
    report["state"] = (
        "complete"
        if report["completed"] == report["expected"] and not report["errors"]
        else "running"
    )
    report["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--build-paper-subset", action="store_true")
    parser.add_argument("--root", default="/workspace/owac/rawvla", type=Path)
    args = parser.parse_args()
    if args.build_paper_subset:
        build_paper_subset(args.root, args.output)
        return
    while True:
        report = audit(args.output)
        if args.watch:
            status = subprocess.run(
                ["supervisorctl", "status", "owac-rawvla-full-libero", "owac-rawvla-full-robotwin"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            report["supervisor"] = status.splitlines()
            if not any(word in status for word in ("RUNNING", "STARTING")):
                if report["state"] != "complete":
                    report["state"] = "incomplete"
        write_json(args.output / "audit-status.json", report)
        if not args.watch or report["state"] in {"complete", "incomplete"} or report["errors"]:
            print(json.dumps(report, ensure_ascii=False))
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
