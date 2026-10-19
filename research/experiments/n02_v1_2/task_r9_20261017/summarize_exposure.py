"""Audit actual optimization receipts against the declared deterministic coverage schedule."""

import json
from collections import Counter
from pathlib import Path

from owac.training.n02_v12_task import LegacyBalancedBlocks as BalancedBlocks

root = Path(__file__).resolve().parent
old = root.parent / "failure_attribution_20261017/existing_evidence.json"
metadata = json.loads(old.read_text())["cached_metadata"]
identity = json.loads(
    (root.parent / "representation_r8_20261016/receipts/fresh/data_identity.json").read_text()
)["train"]
selected = set(identity["selection"]["selected_episodes"])
records = []
for m in sorted(metadata, key=lambda m: m["relative_path"]):
    episode = "/".join(m["relative_path"].split("/")[-2:]).removesuffix(".npz")
    if episode not in selected:
        continue
    n = m["num_frames"]
    for t in range(n):
        records.append(
            dict(
                index=len(records),
                path=m["relative_path"],
                t=t,
                frames=n,
                light=m["lighting_domain"],
                valid=min(50, max(0, n - t - 1)),
                phase="opening" if t < 10 else "approach" if t < 0.6 * n else "late",
            )
        )
assert len(records) == 3668
sampler = BalancedBlocks(records, 8, 1103)
result = {}
for arm in ("task", "control"):
    folder = root / "receipts" / arm
    logged = json.loads((folder / "task_sampler.json").read_text())
    assert logged["groups"] == sampler.record()["groups"]
    rows = [json.loads(s) for s in (folder / "optimization.jsonl").read_text().splitlines() if s]
    actual = []
    for cursor, row in enumerate(rows):
        start, stop = sampler.bounds(cursor)
        window = row["window"]
        assert (window["start"], window["stop"]) == (start, stop)
        expected = records[start:stop]
        assert [r["path"] for r in expected] == window["anchors"]
        assert [r["t"] for r in expected] == window["current_frame_indices"]
        actual.extend(expected)

    def counts(samples):
        return dict(
            exposures=len(samples),
            unique_anchors=len({r["index"] for r in samples}),
            episodes=len({r["path"] for r in samples}),
            opening_t0=sum(r["t"] == 0 for r in samples),
            light_exposures=dict(Counter(r["light"] for r in samples)),
            early_t_lt10_by_light=dict(Counter(r["light"] for r in samples if r["t"] < 10)),
        )

    planned = [r for cursor in range(64) for r in records[slice(*sampler.bounds(cursor))]]
    result[arm] = dict(
        actual_optimizer_rows=len(rows),
        actual=counts(actual),
        planned_64=counts(planned),
        observed_schedule_verified=True,
    )
(root / "exposure_summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
