"""N01 v1.2 P3 development closed loop on LIBERO.

This is the P3 development loop, not the P4 formal matrix.  It runs a bounded,
identity-selected subset of LIBERO episodes against the complete v1.2 model served
by :mod:`owac.runtime.n01_v12_service`, and reports successes over the *real*
denominator.

Two rules matter more than the code:

**The denominator is the number of identities actually completed**, merged by
identity across workers.  Counting per worker and adding would double-count and
inflate the score.

**Selection is stratified and written down.**  Per suite the first five tasks
contribute three episodes and the last five contribute two, giving 25 per suite
and 100 in total.  Episodes are then split across the two workers by interleaving
the sorted identities, so both workers see the same mix of suites, tasks and
lighting; splitting as "first 50 / last 50" would give the cards different
conditions and make any per-card comparison meaningless.

The author client has no per-episode bound -- in ``rawvla_bench`` mode it walks
all 250 episodes of every task.  Rather than modify the frozen client in the eval
tree, this launcher copies it and applies one documented patch that honours an
identity allow-list from ``OWAC_EPISODE_ALLOWLIST``; every unselected episode is
skipped before any environment work, so the run costs only the selected episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

__all__ = [
    "SELECTION_RULE",
    "IDENTITY_KEY",
    "select_identities",
    "split_by_identity",
    "patch_client_source",
    "read_episode_ledger",
    "merge_ledgers",
    "main",
]

SELECTION_RULE = (
    "per suite: round-robin over lighting domains, and within a domain over tasks in order, "
    "taking the first unused episode of each (task, lighting) pair up to the requested "
    "per-suite budget (default 25, total 100); smaller probes spread across tasks; "
    "split across workers by interleaving the identities sorted by "
    "(suite, task_id, episode_idx). The earlier index-order rule was withdrawn: taking "
    "episode_idx 0-2 skews the subset dark because the manifest orders dark lighting first, "
    "which silently turns a mechanism comparison into a lighting comparison."
)
IDENTITY_KEY = ("suite", "task_id", "episode_idx")

ALLOWLIST_PREAMBLE = """

# --- OWAC P3 development-loop patch -----------------------------------------
# The author client selects episodes only by task, so this launcher supplies an
# explicit identity allow-list through the environment.  Unselected episodes are
# skipped before any environment work.
def _owac_load_allowlist():
    import json as _json
    import os as _os

    path = _os.environ.get("OWAC_EPISODE_ALLOWLIST")
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        payload = _json.load(handle)
    return {
        (
            str(entry.get("suite", entry.get("task_suite"))),
            int(entry["task_id"]),
            int(entry["episode_idx"]),
        )
        for entry in payload["identities"]
    }


_OWAC_ALLOWLIST = _owac_load_allowlist()


# --- OWAC P3 synchronized-timeline recorder ---------------------------------
# Records, per policy query, the RAW frame, its frozen-ISP RGB, the support/fusion
# diagnostics the server returned, and the actions that were executed.  Recording beats
# reconstructing because the support decision and fusion weights are computed on the way
# to the action; a replay would have to assume a checkpoint that may no longer be the one
# that produced the action.
def _owac_timeline_open():
    import os as _os

    return _os.environ.get("OWAC_TIMELINE_PATH") or None


def _owac_timeline_set_identity(record):
    _OWAC_TIMELINE["identity"] = dict(record)
    _OWAC_TIMELINE["query"] = 0


def _owac_timeline_downsample(frame, size=48):
    import numpy as _np

    array = _np.asarray(frame)
    if array.ndim != 3:
        return None
    step_y = max(array.shape[0] // size, 1)
    step_x = max(array.shape[1] // size, 1)
    small = array[::step_y, ::step_x][:size, :size]
    if small.dtype != _np.uint8:
        small = _np.clip(small * 255.0, 0, 255).astype(_np.uint8)
    return small.tolist()


def _owac_timeline_raw_dir(path):
    import os as _os_local

    return _os_local.path.dirname(path) + "/" + _os_local.path.basename(path).replace(
        ".jsonl", "-raw"
    )


def _owac_timeline_save_raw(frame, path):
    # Full-resolution RAW is written as a PNG rather than inline JSON: the ISP cannot be
    # applied to a downsampled thumbnail, and the client cannot compute it itself because
    # its PYTHONPATH resolves a different `owac` that has no `owac.raw`.
    import numpy as _np
    from PIL import Image

    array = _np.asarray(frame)
    if array.ndim != 3:
        return None
    if array.dtype != _np.uint8:
        array = _np.clip(array * 255.0, 0, 255).astype(_np.uint8)
    Image.fromarray(array).save(path)
    return path


def _owac_timeline_record(example, actions, response):
    path = _OWAC_TIMELINE["path"]
    identity = _OWAC_TIMELINE["identity"]
    if not path or identity is None:
        return
    try:
        import json as _json

        import numpy as _np

        raw = example.get("image")
        raw_frame = raw[0] if isinstance(raw, (list, tuple)) and raw else raw

        raw_name = None
        if _OWAC_TIMELINE["raw_dir"] and raw_frame is not None:
            # Isolated so a write failure cannot discard the diagnostics and actions too:
            # an earlier version wrapped everything in one bare except and lost the whole
            # record, leaving no trace of where it failed.
            try:
                import os as _os_local

                _os_local.makedirs(_OWAC_TIMELINE["raw_dir"], exist_ok=True)
                # The identity is part of the name because every identity restarts its
                # query counter at zero: naming by query alone made later episodes
                # overwrite earlier ones, and a figure then mixed frames from different
                # rollouts while looking perfectly plausible.
                raw_name = (
                    f"{identity['task_suite']}_t{identity['task_id']:02d}"
                    f"_e{identity['episode_idx']:02d}_q{_OWAC_TIMELINE['query']:04d}.png"
                )
                _owac_timeline_save_raw(
                    raw_frame, _os_local.path.join(_OWAC_TIMELINE["raw_dir"], raw_name)
                )
            except Exception as _raw_exc:
                raw_name = None
                _OWAC_TIMELINE["last_error"] = f"{type(_raw_exc).__name__}: {_raw_exc}"
        data = response.get("data") or {}
        record = {
            "task_suite": identity["task_suite"],
            "task_id": identity["task_id"],
            "episode_idx": identity["episode_idx"],
            "query": _OWAC_TIMELINE["query"],
            # Only the full-resolution PNG is read (by the figure builder); the inline
            # 48x48 thumbnail was written on every line and never read by anything.
            "raw_path": raw_name,
            "actions": _np.asarray(actions, dtype=_np.float32)[:8].tolist(),
            "diagnostics": data.get("diagnostics"),
            "unique_frames": data.get("unique_frames"),
            "latency_ms": data.get("latency_ms"),
        }
        _OWAC_TIMELINE["query"] += 1
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(_json.dumps(record) + chr(10))
    except Exception as _exc:
        # Recording must never break a rollout, but it must not fail silently either: the
        # count and the last error are written to a sidecar the operator can read.
        _OWAC_TIMELINE["failures"] = int(_OWAC_TIMELINE.get("failures", 0)) + 1
        _OWAC_TIMELINE["last_error"] = f"{type(_exc).__name__}: {_exc}"
        try:
            with open(path + ".errors", "a", encoding="utf-8") as _handle:
                _handle.write(f"{type(_exc).__name__}: {_exc}" + chr(10))
        except Exception:
            pass
        return


_OWAC_TIMELINE = {
    "path": _owac_timeline_open(),
    "raw_dir": None,
    "identity": None,
    "query": 0,
    "last_error": None,
}
_OWAC_TIMELINE["failures"] = 0
_OWAC_TIMELINE["raw_dir"] = (
    None if _OWAC_TIMELINE["path"] is None else _owac_timeline_raw_dir(_OWAC_TIMELINE["path"])
)

# --- end OWAC patch ---------------------------------------------------------


"""

ALLOWLIST_CHECK = """            if _OWAC_ALLOWLIST is not None and (
                str(args.task_suite),
                int(task_id),
                int(ep_idx),
            ) not in _OWAC_ALLOWLIST:
                continue
            _owac_timeline_set_identity(
                {
                    "task_suite": str(args.task_suite),
                    "task_id": int(task_id),
                    "episode_idx": int(ep_idx),
                }
            )
"""


def _publish_to_monitor(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """Publish the development-loop counts through the monitor's single-writer path.

    Failures are reported, not swallowed: a monitor that silently stops receiving data is
    worse than one that says it could not be written to.
    """

    try:
        from owac.runtime.n01_v12_monitor import RunStateWriter

        run_root = Path(args.devloop_root).parent / "v1-2-p3-main"
        if not run_root.is_dir():
            raise FileNotFoundError(f"monitor run root {run_root} does not exist")
        label = f"devloop{'' if not args.label else '-' + args.label}"
        writer = RunStateWriter(
            run_root,
            f"{label}-libero-{getattr(args, 'method', 'raw_n01')}",
            method=getattr(args, "method", "raw_n01"),
            input_domain="RAW",
            data_kind="libero_dequantized",
            platform="libero",
            seed=101,
            planned_steps=None,
            stage="P3",
            state="stage_end",
            phase="development_loop",
        )
        writer.rollouts(
            completed=report["completed_rollouts"],
            expected=report["selected_identities"],
            successes=report["successes"],
            failures=report["failures"],
            infra_pending=len(report["not_completed"]),
            note=(
                "P3 development loop, merged by identity across workers"
                + (" (lighting-balanced selection)" if args.label else "")
            ),
        )
        writer.finish(
            exit_reason=f"completed {report['completed_rollouts']}"
            f"/{report['selected_identities']} selected identities",
            state="stage_end",
        )
        print(
            json.dumps({"event": "monitor_published", "completed": report["completed_rollouts"]}),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never silent
        print(
            json.dumps(
                {"event": "monitor_publish_failed", "error": f"{type(exc).__name__}: {exc}"}
            ),
            flush=True,
        )


def episode_split_index(manifest_path: Path, raw_root: Path) -> dict[tuple[str, int, int], str]:
    """Map ``(suite, task_id, demo_index)`` to the training split it belongs to.

    The cache declares ``suite``, ``task_id`` and ``demo_index``. The benchmark's
    ``episode_idx`` is a LIGHTING TRIAL index, not a demonstration index: the real
    manifest maps e.g. trial 17 to ``init_state_index=3``. Even using that physical
    initial-state index still requires verifying demo-to-init-state identity from
    source contents before claiming a held-out result. This index alone does not
    establish that final mapping.
    """

    import numpy as np

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    index: dict[tuple[str, int, int], str] = {}
    for row in manifest["rows"]:
        path = Path(raw_root) / row["relative_path"]
        with np.load(path, allow_pickle=True) as data:
            metadata = json.loads(str(data["metadata_json"]))
        key = (str(metadata["suite"]), int(metadata["task_id"]), int(metadata["demo_index"]))
        if key in index and index[key] != str(row["split"]):
            raise ValueError(f"one demonstration occurs in conflicting splits: {key}")
        index[key] = str(row["split"])
    return index


def filter_to_held_out(
    identities: list[dict[str, Any]],
    index: dict[tuple[str, int, int], str],
    *,
    sides: tuple[str, ...] = ("validation", "test", "calibration"),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep only identities whose episode lies outside the training split.

    Identities the index does not know are dropped rather than kept: an unknown episode
    cannot be shown to be held out, and silently keeping it would reintroduce exactly the
    contamination this filter exists to remove.
    """

    kept: list[dict[str, Any]] = []
    tally: dict[str, int] = {}
    for item in identities:
        if item.get("init_state_index") is None:
            tally["missing_init_state_index"] = tally.get("missing_init_state_index", 0) + 1
            continue
        key = (str(item["suite"]), int(item["task_id"]), int(item["init_state_index"]))
        side = index.get(key, "not_in_manifest")
        tally[side] = tally.get(side, 0) + 1
        if side in sides:
            kept.append(item)
    return kept, tally


def select_identities(
    manifest_path: Path,
    per_suite: int = 25,
    *,
    lighting_balanced: bool = True,
    lighting_domains: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Stratified identity subset.

    ``lighting_balanced`` (the default) round-robins across lighting domains and takes
    the first unused episode of each ``(task, lighting)`` pair.  With three domains and
    ten tasks that covers 25 distinct pairs per suite, so the subset is balanced in
    lighting *and* covers every task.  Setting it to ``False`` reproduces the original
    index-order rule, which is kept only so an already-run plan can be described; it is
    dark-skewed and must not be used for a comparison.
    """

    if per_suite < 1:
        raise ValueError("per_suite must be positive")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = manifest["entries"]
    if lighting_domains:
        available = {item.get("lighting_domain") for item in entries}
        if set(lighting_domains) - available:
            raise ValueError("requested lighting domain is absent from the manifest")
        entries = [item for item in entries if item.get("lighting_domain") in lighting_domains]
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        suite = str(entry.get("suite", entry.get("task_suite")))
        by_suite.setdefault(suite, []).append(entry)

    selected: list[dict[str, Any]] = []
    for suite in sorted(by_suite):
        pool = by_suite[suite]
        tasks = sorted({int(item["task_id"]) for item in pool})
        if not lighting_balanced:
            selected.extend(_select_by_index_order(suite, pool, tasks, per_suite))
            continue
        if per_suite < len(tasks):
            tasks = [tasks[index * len(tasks) // per_suite] for index in range(per_suite)]
        # Index by (task, lighting domain) so the subset can be made balanced in both
        # at once.  Walking a single domain-major stream would exhaust the first task's
        # episodes before reaching the second task, which is how the earlier attempt
        # lost task coverage.
        grid: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for item in pool:
            key = (int(item["task_id"]), str(item.get("lighting_domain", "unlabelled")))
            grid.setdefault(key, []).append(item)
        for stream in grid.values():
            stream.sort(key=lambda item: int(item["episode_idx"]))
        domains = sorted({domain for _, domain in grid})
        if not domains:
            continue
        base = max(per_suite // len(tasks), 1)
        extra = max(per_suite - base * len(tasks), 0)
        chosen = []
        for position, task in enumerate(tasks):
            want = base + (1 if position < extra else 0)
            # Rotate the domain order per task so the per-suite totals stay balanced
            # instead of always favouring the alphabetically first domain.
            offset = position % len(domains)
            order = domains[offset:] + domains[:offset]
            taken = 0
            episode_round = 0
            while taken < want:
                progressed = False
                for domain in order:
                    episodes = grid.get((task, domain), [])
                    if episode_round < len(episodes):
                        chosen.append(episodes[episode_round])
                        taken += 1
                        progressed = True
                        if taken >= want:
                            break
                episode_round += 1
                if not progressed:
                    break
        selected.extend(_describe(suite, chosen))
    return selected


def _select_by_index_order(
    suite: str, pool: list[dict[str, Any]], tasks: list[int], per_suite: int
) -> list[dict[str, Any]]:
    """The withdrawn index-order rule, retained only for describing past plans."""

    by_key = {(int(item["task_id"]), int(item["episode_idx"])): item for item in pool}
    chosen: list[dict[str, Any]] = []
    half = len(tasks) // 2
    for group, take in ((tasks[:half], 3), (tasks[half:], 2)):
        for task in group:
            for episode in range(take):
                key = (task, episode)
                if key in by_key and len(chosen) < per_suite:
                    chosen.append(by_key[key])
    return _describe(suite, chosen)


def _describe(suite: str, chosen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    described = []
    for item in chosen:
        described.append(
            {
                "suite": suite,
                "task_id": int(item["task_id"]),
                "episode_idx": int(item["episode_idx"]),
                "base_seed": item.get("base_seed"),
                "init_state_index": item.get("init_state_index"),
                "lighting_domain": item.get("lighting_domain"),
                "lighting_ev": item.get("lighting_ev"),
            }
        )
    return described


def split_by_identity(
    identities: list[dict[str, Any]], workers: int = 2
) -> list[list[dict[str, Any]]]:
    """Interleave so every worker sees the same suite/task/lighting mix."""

    ordered = sorted(
        identities, key=lambda item: (item["suite"], item["task_id"], item["episode_idx"])
    )
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    for index, identity in enumerate(ordered):
        buckets[index % workers].append(identity)
    return buckets


def patch_client_source(source: str) -> str:
    """Add identity-allow-list support to the frozen LIBERO client.

    Two exact anchors are replaced; if either is missing the author file changed
    and the launcher must be re-read rather than silently running unpatched.
    """

    anchor_def = "def _append_episode(path: Path | None, record: dict[str, Any]) -> None:"
    if source.count(anchor_def) != 1:
        raise ValueError("client anchor for the allow-list preamble not found")
    source = source.replace(anchor_def, ALLOWLIST_PREAMBLE.lstrip("\n") + anchor_def, 1)

    anchor_loop = "            if ep_idx in task_completed:\n                continue\n"
    if source.count(anchor_loop) != 1:
        raise ValueError("client anchor for the episode loop not found")
    source = source.replace(anchor_loop, anchor_loop + ALLOWLIST_CHECK, 1)

    anchor_stats = "        self.last_action_stats = {"
    if source.count(anchor_stats) != 1:
        raise ValueError("client anchor for the policy query not found")
    source = source.replace(anchor_stats, TIMELINE_RECORD + anchor_stats, 1)
    return source


TIMELINE_RECORD = """        _owac_timeline_record(example, actions, response)
"""


def read_episode_ledger(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Episode records keyed by identity; unusable lines are skipped, not counted."""

    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            key = (
                str(record.get("task_suite", record.get("suite"))),
                int(record["task_id"]),
                int(record["episode_idx"]),
            )
            if "success" not in record:
                continue
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        records[key] = record
    return records


def merge_ledgers(paths: list[Path]) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Merge worker ledgers by identity; later duplicates overwrite, never add."""

    merged: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in paths:
        for key, record in read_episode_ledger(path).items():
            merged[key] = record
    return merged


def _report(
    merged: dict[tuple[str, int, int], dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    stage: str = "P3",
) -> dict[str, Any]:
    selected_keys = {(item["suite"], item["task_id"], item["episode_idx"]) for item in selected}
    completed = {key: value for key, value in merged.items() if key in selected_keys}
    successes = sum(1 for value in completed.values() if value.get("success"))
    by_suite: dict[str, dict[str, int]] = {}
    for key, value in completed.items():
        bucket = by_suite.setdefault(key[0], {"completed": 0, "successes": 0})
        bucket["completed"] += 1
        bucket["successes"] += int(bool(value.get("success")))
    wall = [float(value["wall_seconds"]) for value in completed.values() if "wall_seconds" in value]
    return {
        "stage": stage,
        "selected_identities": len(selected_keys),
        "completed_rollouts": len(completed),
        "successes": successes,
        "failures": len(completed) - successes,
        "success_rate": (successes / len(completed)) if completed else None,
        "denominator": len(completed),
        "denominator_note": (
            "the denominator is the number of identities actually completed, merged by "
            "identity across workers; worker counts must never be added together"
        ),
        "not_completed": sorted(
            f"{key[0]}/{key[1]}/{key[2]}" for key in selected_keys - set(completed)
        ),
        "by_suite": by_suite,
        "wall_seconds": {
            "mean": (sum(wall) / len(wall)) if wall else None,
            "max": max(wall) if wall else None,
        },
        "budget_note": (
            "P3 development loop by design: this is not the P4 formal list and its "
            "denominator is not 10,000 or 13,250"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devloop-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--deploy-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--identity-proof",
        type=Path,
        default=None,
        help="content-verified demo-to-initial-state receipt; required for held-out claims",
    )
    parser.add_argument("--ports", default="18022,18023", help="one service port per worker")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu-indices", default="0,1", help="physical GPU per client worker")
    parser.add_argument("--per-suite", type=int, default=25)
    parser.add_argument("--lighting-domains", default="", help="comma-separated planned domains")
    parser.add_argument(
        "--index-order-selection",
        action="store_true",
        help=(
            "reproduce the withdrawn index-order rule (episode_idx 0-2). It is dark-skewed "
            "and only exists to describe the plan already run; do not use it for a comparison."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="write the plan and exit")
    parser.add_argument("--report-only", action="store_true", help="merge ledgers and exit")
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per suite invocation")
    parser.add_argument(
        "--record-timeline",
        default=None,
        help=(
            "write the synchronized-timeline JSONL here. It used to depend on an inherited "
            "OWAC_TIMELINE_PATH, so recording could be off without anything saying so; "
            "passing the path here makes it a decision the run records."
        ),
    )
    parser.add_argument(
        "--held-out-only",
        action="store_true",
        help=(
            "restrict selection to episodes outside the training split. The benchmark "
            "draws initial states from the whole suite and ignores the partition, so "
            "without this a 'development closed loop' includes episodes the model trained "
            "on (measured: 69 of 100 on the P3 loop). Requires --raw-root and the training "
            "manifest; the retained/excluded tally is written into plan.json."
        ),
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="RawVLA-Bench raw cache root, needed by --held-out-only to read episode metadata",
    )
    parser.add_argument(
        "--training-manifest",
        default=None,
        help="the training manifest that indexed the cache, needed by --held-out-only",
    )
    parser.add_argument(
        "--suites",
        default="",
        help=(
            "comma-separated suite subset. Empty runs every suite in the manifest; the "
            "filter exists for short instrumented passes whose only purpose is recording, "
            "where running all four suites would hold the GPUs for no extra evidence."
        ),
    )
    parser.add_argument(
        "--label",
        default="",
        help=(
            "suffix for the plan directory (devloop-<label>). Use it to keep a corrected "
            "run beside an earlier one instead of overwriting evidence -- an already-run "
            "denominator must stay readable."
        ),
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=128,
        help=(
            "client-side square resize of the rendered RAW frame. 128 matches the "
            "scale this arm was trained at (the trainer used --crop 128); at 224 the "
            "32-frame window needs ~92 GiB and cannot fit on a 97 GiB card."
        ),
    )
    parser.add_argument(
        "--assets-checkpoint",
        type=Path,
        default=None,
        help=(
            "directory containing assets/physical-intelligence/libero/norm_stats.json; the "
            "client's own default resolver falls back to a path this deployment does not have"
        ),
    )
    args = parser.parse_args(argv)
    gpu_indices = [int(value) for value in args.gpu_indices.split(",")]
    if len(gpu_indices) < args.workers:
        raise ValueError("--gpu-indices needs one physical GPU per worker")

    root = args.devloop_root
    runtime = root / "runtime"
    manifest = root / "libero-manifest.json"
    source_client = runtime / "libero_client.py"
    if not manifest.is_file() or not source_client.is_file():
        raise FileNotFoundError(
            f"{manifest} or {source_client} missing; run `rawvla_full_eval prepare` first"
        )

    # The held-out filter applies to the candidate pool, before the 100-identity cap:
    # filtering afterwards only re-labels whatever the unfiltered draw happened to include
    # (measured: 21 of 100), whereas filtering first lets the loop actually reach 100
    # held-out identities.
    selection_manifest = manifest
    held_out_tally: dict[str, int] | None = None
    if args.held_out_only:
        if not args.raw_root or not args.training_manifest:
            raise SystemExit("--held-out-only needs --raw-root and --training-manifest")
        if not args.identity_proof:
            raise SystemExit(
                "--held-out-only requires --identity-proof; metadata order is not proof"
            )
        from owac.evaluation.n01_v12_identity import verified_split_index

        index = verified_split_index(
            Path(args.training_manifest), Path(args.raw_root), args.identity_proof
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        kept, excluded = filter_to_held_out(
            [
                {
                    "suite": str(entry.get("suite", entry.get("task_suite"))),
                    "task_id": entry["task_id"],
                    "episode_idx": entry["episode_idx"],
                    "init_state_index": entry.get("init_state_index"),
                    "_entry": entry,
                }
                for entry in payload["entries"]
            ],
            index,
        )
        held_out_tally = excluded
        payload["entries"] = [item["_entry"] for item in kept]
        selection_manifest = root / "held-out-entries.json"
        selection_manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "held_out_pool",
                    "candidates_before": sum(excluded.values()),
                    "candidates_after": len(payload["entries"]),
                    "tally": excluded,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if len(payload["entries"]) < args.per_suite * 4:
            raise SystemExit(
                f"only {len(payload['entries'])} held-out candidates for "
                f"{args.per_suite * 4} identities; the benchmark's coverage of the held-out "
                "sides is too thin for this loop size"
            )

    identities = select_identities(
        selection_manifest,
        per_suite=args.per_suite,
        lighting_balanced=not args.index_order_selection,
        lighting_domains=tuple(filter(None, args.lighting_domains.split(","))) or None,
    )
    wanted = [name for name in str(args.suites).split(",") if name]
    if wanted:
        identities = [item for item in identities if item["suite"] in set(wanted)]
        if not identities:
            raise SystemExit(f"--suites {args.suites!r} matched none of the manifest suites")
    buckets = split_by_identity(identities, workers=args.workers)

    plan_dir = root / ("devloop" if not args.label else f"devloop-{args.label}")
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "selection_rule": SELECTION_RULE,
        "lighting_balanced": not args.index_order_selection,
        "lighting_domains": list(filter(None, args.lighting_domains.split(","))),
        "suites_filter": [n for n in str(args.suites).split(",") if n],
        "record_timeline": None if not args.record_timeline else str(args.record_timeline),
        "held_out_only": bool(args.held_out_only),
        "held_out_tally": held_out_tally,
        "split_identity_note": (
            "filter joins actual init_state_index to a content-verified split mapping; "
            "episode_idx is a lighting trial index."
        ),
        "holdout_status": "content_verified_initial_state_split"
        if args.held_out_only
        else "development",
        "resize_size": int(args.resize_size),
        "input_path_note": (
            "Complete configurations use native RAW transport and the same model-owned "
            "observation transform as training; legacy crop/resize runs remain "
            "historical diagnostics."
        ),
        "identity_key": list(IDENTITY_KEY),
        "selected": len(identities),
        "workers": [],
        "config": str(args.config),
    }

    dev_client = plan_dir / "libero_client_devloop.py"
    patched_source = patch_client_source(source_client.read_text(encoding="utf-8"))
    from owac.frameworks.n01_full.model import N01V12Config

    effective_model = N01V12Config.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    if effective_model.observation is not None:
        from owac.runtime.n01_v12_client_patch import patch_native_libero_client

        patched_source = patch_native_libero_client(patched_source)
    # Compare content, not existence.  A run in a reused plan directory silently kept a
    # client patched by an older version of this launcher, so a fix to the patch had no
    # effect and the pass looked like it had run when it had not.
    if not dev_client.is_file() or dev_client.read_text(encoding="utf-8") != patched_source:
        dev_client.write_text(patched_source, encoding="utf-8")

    ledgers: list[Path] = []
    for index, bucket in enumerate(buckets):
        allowlist_path = plan_dir / f"worker{index}-identities.json"
        allowlist_path.write_text(
            json.dumps({"selection_rule": SELECTION_RULE, "identities": bucket}, indent=2),
            encoding="utf-8",
        )
        ledger = plan_dir / f"worker{index}-episodes.jsonl"
        ledgers.append(ledger)
        plan["workers"].append(
            {
                "worker": index,
                "port": args.ports.split(",")[index],
                "identities": len(bucket),
                "allowlist": str(allowlist_path),
                "ledger": str(ledger),
            }
        )
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    if args.report_only:
        report = _report(merge_ledgers(ledgers), identities)
        # Feed the monitor from the launcher that actually runs the rollouts.  The page
        # reads ``resources.dev_loop`` from worker files, and nothing in production wrote
        # one, so the development-loop panel was only ever populated by hand.
        _publish_to_monitor(args, report)
        report["plan"] = str(plan_dir / "plan.json")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
        return 0

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0

    source = args.deploy_root / "source"
    suites = sorted({item["suite"] for item in identities})
    env = os.environ.copy()
    if args.record_timeline:
        env["OWAC_TIMELINE_PATH"] = str(args.record_timeline)
    env.update(
        {
            # Mirrors rawvla_full_eval.run_libero exactly: the frozen owac modules,
            # this repo's src, the author tree, the LIBERO glue, and the openpi
            # eval_files directory (which is where pi_libero_common lives).
            "PYTHONPATH": (
                f"{root}/frozen/src:{args.repo_root}/src:{source}/starVLA:"
                f"{source}/experiments/simulation/libero:"
                f"{source}/starVLA/examples/simBenchmarks/LIBERO/eval_files/openpi"
            ),
            "LIBERO_CONFIG_PATH": f"{source}/.local/libero",
            "OPENPI_CONVERTED_ROOT": f"{source}/openpi_converted_protocol",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    python = str(args.deploy_root / "envs/starvla/bin/python")
    assets_checkpoint = args.assets_checkpoint
    if assets_checkpoint is None:
        candidate = args.deploy_root / "checkpoints/pi05_libero_jax"
        if (candidate / "assets/physical-intelligence/libero/norm_stats.json").is_file():
            assets_checkpoint = candidate
        else:
            raise FileNotFoundError(
                "no --assets-checkpoint given and the default candidate "
                f"{candidate} has no assets/physical-intelligence/libero/norm_stats.json; "
                "the client cannot normalize actions without it"
            )
    # Each worker gets its own GPU, its own service port and its own ledger; the
    # per-episode artifacts land in separate directories so they cannot collide.
    # One process per worker per suite, and only two run at a time: starting all
    # eight at once would have eight LIBERO environments competing for the same two
    # GPUs and would blur which worker produced which episode.
    for suite in suites:
        running: list[tuple[int, subprocess.Popen, Path]] = []
        for index, bucket in enumerate(buckets):
            if not bucket:
                continue
            allowlist_path = plan_dir / f"worker{index}-identities.json"
            ledger = plan_dir / f"worker{index}-episodes.jsonl"
            worker_env = dict(env)
            worker_env["OWAC_EPISODE_ALLOWLIST"] = str(allowlist_path)
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_indices[index])
            directory = plan_dir / f"worker{index}" / suite
            directory.mkdir(parents=True, exist_ok=True)
            result_json = directory / "result.json"
            if result_json.is_file():
                continue
            command = [
                python,
                str(dev_client),
                "--model",
                "PI05",
                "--model-source",
                "openpi",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.ports.split(",")[index]),
                "--task-suite",
                suite,
                "--task-start",
                "0",
                "--max-tasks",
                "10",
                "--image-mode",
                "rawvla_bench",
                "--rawvla-bench-representation",
                "raw",
                "--rawvla-bench-manifest",
                str(manifest),
                "--resume-path",
                str(ledger),
                "--result-json",
                str(result_json),
                "--resize-size",
                str(args.resize_size),
            ]
            if assets_checkpoint is not None:
                command.extend(["--assets-checkpoint", str(assets_checkpoint)])
            log_path = directory / "run.log"
            handle = log_path.open("ab")
            process = subprocess.Popen(
                command, cwd=str(source / "starVLA"), env=worker_env, stdout=handle, stderr=handle
            )
            running.append((index, process, log_path))
            print(
                json.dumps(
                    {"event": "launch", "worker": index, "suite": suite, "pid": process.pid},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        for index, process, log_path in running:
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                code = -1
            print(
                json.dumps(
                    {
                        "event": "finish",
                        "worker": index,
                        "suite": suite,
                        "returncode": code,
                        "log": str(log_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            # Progress is visible while the loop runs, not only at the end.
            partial = _report(merge_ledgers(ledgers), identities)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "completed_rollouts": partial["completed_rollouts"],
                        "successes": partial["successes"],
                        "selected": partial["selected_identities"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    time.sleep(1)
    report = _report(merge_ledgers(ledgers), identities)
    report["plan"] = str(plan_dir / "plan.json")
    (plan_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
