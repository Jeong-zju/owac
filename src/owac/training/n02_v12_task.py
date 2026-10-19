"""Task adaptation protocol: fixed episode coverage and separate real control metrics.

Uses only the existing source's frozen partition. Timeline strata describe observation
positions, not hidden contact labels. No clean sensor signal is inferred from noisy RAW.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from owac.runtime.n02_task_selection import REPAIR_PROTOCOL as REPAIR_PROTOCOL
from owac.runtime.n02_task_selection import select_repaired_checkpoint as select_repaired_checkpoint

LIGHTS = ("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver")
PROTOCOL = "n02-task-adaptation-r9-v1"


def is_repair(config):
    return config.get("task_adaptation", {}).get("protocol") == REPAIR_PROTOCOL


def validate_repair_config(config):
    from owac.runtime.n02_control import ControlProtocol

    if not is_repair(config):
        return
    task = config["task_adaptation"]
    if task.get("objective") != "native_flow_32":
        raise ValueError("r11 must retain all native action dimensions")
    protocol = ControlProtocol.parse(config["execution"])
    if protocol.replan_interval not in (10, 50) or protocol.as_dict() != config["execution"]:
        raise ValueError("r11 validation requires a complete supported execution contract")
    if not config.get("online_checks", {}).get("enabled"):
        raise ValueError("r11 requires fixed-weight checks")
    if not config["policy"].get("task_instructions"):
        raise ValueError("r11 requires canonical task instructions")
    if config["model"].get("revision_policy") != "reset_explanations":
        raise ValueError("r11 requires online explanation rebuilding")
    if config["model"].get("fast_prefix") and config.get("aux", {}).get("lambda_pred", 0) != 0:
        raise ValueError("fast prefix cannot skip an enabled causal auxiliary objective")


def source_records(source):
    metadata = {}
    rows = []
    for i in range(source.anchor_count):
        a = source.anchor(i)
        if a.relative_path not in metadata:
            with np.load(Path(source.data_root) / a.relative_path, allow_pickle=False) as data:
                metadata[a.relative_path] = json.loads(str(data["metadata_json"].item()))
        m = metadata[a.relative_path]
        t, n = int(a.current_frame_index), int(m["num_frames"])
        phase = "opening" if t < 10 else "approach" if t < 0.6 * n else "late"
        rows.append(
            dict(
                index=i,
                path=a.relative_path,
                t=t,
                frames=n,
                light=m["lighting_domain"],
                phase=phase,
                valid=min(source.window_options.action_horizon, max(0, n - t - 1)),
            )
        )
    return rows


def dev_selection(records, *, protocol=PROTOCOL):
    episodes = defaultdict(list)
    for row in records:
        episodes[row["path"]].append(row)
    selected = []
    for path, rows in sorted(episodes.items()):
        n = rows[0]["frames"]
        for phase, t in (("opening", 0), ("approach", 10), ("middle", n // 2), ("tail", n - 11)):
            candidates = [r for r in rows if r["valid"] >= min(10, n - 1)]
            if not candidates:
                raise ValueError(f"no meaningful action horizon in dev episode {path}")
            row = min(candidates, key=lambda r: (abs(r["t"] - t), r["index"]))
            if row["index"] in [r["index"] for r in selected]:
                raise ValueError("dev episode too short for four distinct trajectory landmarks")
            selected.append({**row, "phase": phase})
    content = dict(
        protocol=protocol,
        partition="dev",
        rows=selected,
        missing_lighting=sorted(set(LIGHTS) - {r["light"] for r in selected}),
        selection="every frozen dev episode: t0, t10, midpoint, ten-valid-action tail",
    )
    content["sha256"] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return content


class LegacyBalancedBlocks:
    """Balanced light/timeline strata, shuffled without replacement inside each stratum.

    Contiguous blocks preserve the loader's one-episode cache efficiency. Blocks crossing
    episodes and all-padding targets are excluded explicitly, never silently resampled.
    """

    def __init__(self, records, batch_size, seed):
        self.seed = seed
        self.groups = defaultdict(list)
        self.excluded = []
        for start in range(0, len(records), batch_size):
            block = records[start : start + batch_size]
            if len({r["path"] for r in block}) != 1 or not any(r["valid"] for r in block):
                self.excluded.append(start)
                continue
            row = block[0]
            self.groups[(row["light"], row["phase"])].append((start, start + len(block)))
        self.keys = sorted(self.groups)
        if not self.keys:
            raise ValueError("empty task sampler")

    def bounds(self, cursor):
        cycle, slot = divmod(cursor, len(self.keys))
        order = np.random.default_rng(self.seed + cycle).permutation(len(self.keys))
        group_id = int(order[slot])
        group = self.groups[self.keys[group_id]]
        epoch, position = divmod(cycle, len(group))
        permutation = np.random.default_rng(
            self.seed + 100003 * (group_id + 1) + epoch
        ).permutation(len(group))
        return group[int(permutation[position])]

    def record(self):
        return dict(
            protocol=PROTOCOL,
            seed=self.seed,
            selection="balanced light x timeline blocks",
            groups={"/".join(k): len(v) for k, v in self.groups.items()},
            excluded_cross_episode_or_padding_block_starts=self.excluded,
        )


class BalancedBlocks(LegacyBalancedBlocks):
    """Partition each episode/phase into bounded blocks without dropping valid anchors.

    A short boundary block is a real smaller microbatch, never padding or a sample
    borrowed from another episode. The inherited schedule visits every block in
    each stratum before repeating it. Cursor and seed fully determine the order.
    """

    def __init__(self, records, batch_size, seed):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.seed = seed
        self.groups = defaultdict(list)
        self.excluded = []
        self.covered = []
        start = 0
        while start < len(records):
            row = records[start]
            if row["valid"] <= 0:
                self.excluded.append(start)
                start += 1
                continue
            key = (row["light"], row["phase"])
            stop = start + 1
            while stop < min(len(records), start + batch_size):
                nxt = records[stop]
                if (
                    nxt["path"] != row["path"]
                    or (nxt["light"], nxt["phase"]) != key
                    or nxt["valid"] <= 0
                ):
                    break
                stop += 1
            self.groups[key].append((start, stop))
            self.covered.extend(range(start, stop))
            start = stop
        self.keys = sorted(self.groups)
        expected = {i for i, row in enumerate(records) if row["valid"] > 0}
        if not self.keys or set(self.covered) != expected or len(self.covered) != len(expected):
            raise ValueError("task sampler failed exact valid-anchor coverage")

    def record(self):
        return dict(
            protocol=REPAIR_PROTOCOL,
            seed=self.seed,
            selection="balanced light x phase; episode/phase boundaries retain short blocks",
            groups={"/".join(k): len(v) for k, v in self.groups.items()},
            excluded_all_padding_anchors=self.excluded,
            reachable_valid_anchors=len(self.covered),
            complete_valid_coverage=True,
        )


def control_flow_loss(velocity, target, mask, dimensions):
    """Mask before subtraction: padding NaNs and unused action columns contribute nothing."""
    if not 0 < dimensions <= velocity.shape[-1] or velocity.shape != target.shape:
        raise ValueError("invalid control flow dimensions")
    valid = torch.as_tensor(mask, device=velocity.device, dtype=torch.bool)
    if valid.shape != velocity.shape[:2] or not bool(valid.any()):
        raise ValueError("empty or invalid control flow action mask")
    v, u = velocity[..., :dimensions][valid], target[..., :dimensions][valid]
    return (v.float() - u.float()).square().mean()


def control_metrics(sample, target, mask, identity):
    sample, target = np.asarray(sample), np.asarray(target)
    mask = np.asarray(mask, dtype=bool)
    if sample.shape != target.shape or mask.shape != sample.shape[:2] or not mask.any():
        raise ValueError("invalid sampled control metric shapes/mask")
    error = np.square(sample[mask].astype(np.float64) - target[mask])
    if not np.isfinite(error).all():
        raise ValueError("nonfinite sampled control error")
    joints = [i for i in range(sample.shape[-1]) if i not in (6, 13)]
    grips = [i for i in (6, 13) if i < sample.shape[-1]]
    short = mask.copy()
    short[:, 10:] = False
    early = np.square(sample[short].astype(np.float64) - target[short])
    return {
        **identity,
        "valid_actions": int(mask.sum()),
        "squared_error_sum": float(error.sum()),
        "control_mse": float(error.mean()),
        "first10_mse": float(early.mean()),
        "prefix_valid_actions": int(short.sum()),
        "prefix_squared_error_sum": float(early.sum()),
        "prefix_joint_mse": float(early[:, joints].mean()),
        "prefix_gripper_mse": float(early[:, grips].mean()) if grips else None,
        "joint_mse": float(error[:, joints].mean()),
        "gripper_mse": float(error[:, grips].mean()) if grips else None,
        "first_action_mse": float(np.square(sample[:, 0] - target[:, 0]).mean()),
        "dimensions": int(sample.shape[-1]),
    }


def aggregate_control(rows):
    if not rows:
        raise ValueError("no task validation rows")

    def aggregate(items):
        return dict(
            windows=len(items),
            valid_actions=sum(r["valid_actions"] for r in items),
            mse=sum(r["squared_error_sum"] for r in items)
            / sum(r["valid_actions"] * r["dimensions"] for r in items),
            first10_mse=float(np.mean([r["first10_mse"] for r in items])),
            prefix_joint_mse=float(
                np.mean([r.get("prefix_joint_mse", r["joint_mse"]) for r in items])
            ),
            prefix_gripper_mse=float(
                np.mean([r.get("prefix_gripper_mse", r["gripper_mse"]) for r in items])
            ),
            joint_mse=float(
                np.average(
                    [r["joint_mse"] for r in items], weights=[r["valid_actions"] for r in items]
                )
            ),
            gripper_mse=float(
                np.average(
                    [r["gripper_mse"] for r in items], weights=[r["valid_actions"] for r in items]
                )
            ),
        )

    strata = defaultdict(list)
    for row in rows:
        strata[row["light"] + "/" + row["phase"]].append(row)
    grouped = {key: aggregate(items) for key, items in sorted(strata.items())}
    return {
        **aggregate(rows),
        "strata": grouped,
        "balanced_mse": float(np.mean([r["mse"] for r in grouped.values()])),
        "balanced_prefix_mse": float(np.mean([r["first10_mse"] for r in grouped.values()])),
        "missing_lighting": sorted(set(LIGHTS) - {r["light"] for r in rows}),
    }


def execution_proxy_metrics(sample, target, mask, identity, proprio, stats, protocol):
    """Apply the live executor from expert state, explicitly an offline proxy.

    No predicted/hidden physical state is fabricated between replans. Only one
    prefix is transformed; full stateful verification still requires a rollout.
    """
    from owac.runtime.n02_control import ActionExecution

    action_mean = np.asarray(stats["actions"]["mean"], dtype=np.float32)
    action_std = np.asarray(stats["actions"]["std"], dtype=np.float32)
    state = np.asarray(proprio)[0] * (np.asarray(stats["state"]["std"]) + 1e-8) + np.asarray(
        stats["state"]["mean"]
    )
    physical = np.asarray(sample)[0] * action_std + action_mean
    executed = ActionExecution(protocol).prefix(physical, state)
    transformed = np.asarray(sample).copy()
    transformed[0, : len(executed)] = (executed - action_mean) / np.maximum(action_std, 1e-8)
    valid = np.asarray(mask).copy()
    valid[:, protocol.replan_interval :] = False
    result = control_metrics(transformed, target, valid, identity)
    result["scope"] = "teacher_started_prefix_proxy_not_closed_loop"
    result["execution_contract"] = protocol.as_dict()
    return result
