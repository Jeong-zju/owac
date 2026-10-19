"""Independently check the recorded experiment identities and key contrasts."""

import hashlib
import json
from pathlib import Path

import numpy as np

from owac.evaluation.n02_bottlenecks import paired_episodes

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "outputs/n02-bottlenecks-20261013"
REPLAY = ROOT / "outputs/n02-bottlenecks-replay-20261013"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = [json.loads(line) for line in (OUTPUT / "records.jsonl").read_text().splitlines()]
    summary = json.loads((OUTPUT / "summary.json").read_text())
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    ordinary = [r for r in records if "counterfactual_truth" not in r]
    assert len(ordinary) == 32 * 25
    assert len({(r["seed"], r["condition"], r["arm"]) for r in ordinary}) == len(ordinary)
    assert all(len(r["steps"]) == 8 for r in records)
    for seed in range(200, 232):
        for condition in ("correct", "misleading", "missing", "unidentifiable", "stale"):
            rows = [r for r in ordinary if r["seed"] == seed and r["condition"] == condition]
            assert len({r["measurement_sha256"] for r in rows}) == 1
        common = [
            r
            for r in ordinary
            if r["seed"] == seed and r["condition"] in {"correct", "misleading", "missing"}
        ]
        assert len({r["measurement_sha256"] for r in common}) == 1
    replay = [json.loads(line) for line in (REPLAY / "records.jsonl").read_text().splitlines()]
    assert replay == [r for r in records if r["seed"] == 200]
    assert all(digest(ROOT / p) == h for p, h in manifest["source_hashes"].items())
    for family in ("gaussian", "huber"):
        table = summary["B1"]["misleading"]
        mean = table[f"grid16_{family}_mean"]["mean_log_evidence"]
        total = table[f"grid16_{family}_sum"]["mean_log_evidence"]
        assert np.isclose(total, mean * 768, atol=1e-8)
    assert all(r["max_uniform_drift"] == 0 for r in summary["B1"]["unidentifiable"].values())
    old_plan = json.loads(
        (ROOT / "research/experiments/results/2026-10-13_n02_repairs/plan.json").read_text()
    )
    assert not set(manifest["seeds"]) & set(old_plan["validation_seeds"])
    prefix_information = []
    for seed in manifest["seeds"]:
        gen, episodes = paired_episodes(seed)
        quality = gen.measurement_quality(episodes["misleading"])
        prefix_information.extend(
            evidence
            for evidence, visible in zip(
                quality["expected_log_bayes_factor"], quality["identifiable_now"], strict=True
            )
            if not visible
        )
    # Check that the pre-cue gate really sees identical physical hypotheses,
    # rather than several weak individually subthreshold but cumulative cues.
    assert max(prefix_information) == 0.0
    no_cue = [r for r in ordinary if r["arm"] == "full" and r["condition"] == "unidentifiable"]
    misleading = [r for r in ordinary if r["arm"] == "full" and r["condition"] == "misleading"]
    pairs = summary["B2"]["counterfactual_records"]
    report = {
        "identity_checks_passed": True,
        "complete_episode_records": len(records),
        "paired_rule_records": len(ordinary),
        "same_state_revision_forks": sum(len(r["forks"]) for r in ordinary),
        "replayed_seed": 200,
        "replay_records_bitwise_equal_as_json_numbers": True,
        "runtime_source_hashes_match_delivery": True,
        "validation_seed_overlap_with_previous_round": 0,
        "misleading_prefix_max_expected_log_bayes_factor": max(prefix_information),
        "no_cue_nonzero_withdrawal_frames": sum(
            any(x > 0 for x in s["withdrawal"]) for r in no_cue for s in r["steps"]
        ),
        "no_cue_frames": len(no_cue) * 8,
        "misleading_final_ge_point7": sum(r["steps"][-1]["truth_mass"] >= 0.7 for r in misleading),
        "misleading_prefix_violations": sum(
            [s for s in r["steps"] if not s["cue_visible"]][-1]["truth_mass"] > 0.25
            for r in misleading
        ),
        "counterfactual_mean_p1_when0": float(np.mean([p["p1_when_0"] for p in pairs])),
        "counterfactual_mean_p1_when1": float(np.mean([p["p1_when_1"] for p in pairs])),
        "counterfactual_class1_choices": sum(
            p[key] > 0.5 for p in pairs for key in ("p1_when_0", "p1_when_1")
        ),
        "artifact_hashes": {p.name: digest(p) for p in sorted(OUTPUT.glob("*.json*"))},
        "replay_records_sha256": digest(REPLAY / "records.jsonl"),
    }
    (Path(__file__).resolve().parent / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
