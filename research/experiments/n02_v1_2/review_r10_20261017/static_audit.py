"""Recompute sampler coverage and optimization facts from immutable r9 receipts."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from owac.training.n02_v12_task import LegacyBalancedBlocks as BalancedBlocks


def main():
    root = Path(__file__).resolve().parent
    previous = root.parent / "task_r9_20261017"
    metadata = json.loads(
        (root.parent / "failure_attribution_20261017/existing_evidence.json").read_text()
    )["cached_metadata"]
    identity = json.loads(
        (root.parent / "representation_r8_20261016/receipts/fresh/data_identity.json").read_text()
    )["train"]
    selected = set(identity["selection"]["selected_episodes"])
    records = []
    for m in sorted(metadata, key=lambda m: m["relative_path"]):
        episode = "/".join(m["relative_path"].split("/")[-2:]).removesuffix(".npz")
        if episode not in selected:
            continue
        for t in range(m["num_frames"]):
            records.append(
                dict(
                    index=len(records),
                    path=m["relative_path"],
                    t=t,
                    frames=m["num_frames"],
                    light=m["lighting_domain"],
                    valid=min(50, max(0, m["num_frames"] - t - 1)),
                    phase="opening"
                    if t < 10
                    else "approach"
                    if t < 0.6 * m["num_frames"]
                    else "late",
                )
            )
    assert len(records) == 3668
    sampler = BalancedBlocks(records, 8, 1103)
    reachable = {i for blocks in sampler.groups.values() for a, b in blocks for i in range(a, b)}
    unreachable = [r for r in records if r["index"] not in reachable and r["valid"]]
    wrong_phase = [
        r
        for (_, phase), blocks in sampler.groups.items()
        for a, b in blocks
        for r in records[a:b]
        if r["phase"] != phase
    ]
    result = dict(
        optimizer_updates=0,
        anchors=len(records),
        valid_anchors=sum(r["valid"] > 0 for r in records),
        unreachable_valid_anchors=len(unreachable),
        unreachable_opening_anchors=sum(r["t"] < 10 for r in unreachable),
        all_t0=sum(r["t"] == 0 for r in records),
        unreachable_t0=sum(r["t"] == 0 for r in unreachable),
        unreachable_t0_by_light=dict(Counter(r["light"] for r in unreachable if r["t"] == 0)),
        unreachable_normal_opening=[
            r for r in unreachable if r["light"] == "Normal" and r["t"] < 10
        ],
        mislabelled_phase_rows=len(wrong_phase),
        opening_blocks={
            light: [
                dict(
                    start=a,
                    stop=b,
                    t=[r["t"] for r in records[a:b]],
                    actual_opening_count=sum(r["t"] < 10 for r in records[a:b]),
                )
                for a, b in blocks
            ]
            for (light, phase), blocks in sampler.groups.items()
            if phase == "opening"
        },
        optimization={},
        selection={},
    )
    for arm in ("task", "control"):
        folder = previous / "receipts" / arm
        rows = [json.loads(s) for s in (folder / "optimization.jsonl").read_text().splitlines()]
        result["optimization"][arm] = {
            group: dict(
                min=float(min(values)),
                median=float(np.median(values)),
                max=float(max(values)),
                clipped_updates=sum(x > 1 for x in values),
            )
            for group in rows[0]["gradient_norms_by_clip_group"]
            for values in [[r["gradient_norms_by_clip_group"][group] for r in rows]]
        }
        baseline = json.loads((folder / "dev_baseline.json").read_text())
        result["selection"][arm] = []
        for p in sorted(folder.glob("checkpoint_*.metrics.json")):
            d = json.loads(p.read_text())
            normal = d["task_control"]["strata"]["Normal/opening"]
            bnormal = baseline["task_control"]["strata"]["Normal/opening"]
            result["selection"][arm].append(
                dict(
                    step=d["step"],
                    balanced=d["sampled_control_mse"],
                    opening=normal["mse"],
                    opening_first10=normal["first10_mse"],
                    balanced_change=d["sampled_control_mse"] / baseline["sampled_control_mse"] - 1,
                    opening_change=normal["mse"] / bnormal["mse"] - 1,
                    opening_first10_change=normal["first10_mse"] / bnormal["first10_mse"] - 1,
                )
            )
    (root / "static_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("opening_blocks", "unreachable_normal_opening")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
