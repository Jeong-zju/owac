"""N01 v1.2 P3 development-loop selection, splitting and denominator rules.

These are the parts that decide what the reported success rate means, so they are
tested directly: an inflated denominator or a mis-split worker pair would produce a
number that looks fine and is wrong.
"""

from __future__ import annotations

import json

import pytest

from owac.runtime.n01_v12_libero_dev import (
    ALLOWLIST_CHECK,
    IDENTITY_KEY,
    SELECTION_RULE,
    TIMELINE_RECORD,
    _report,
    filter_to_held_out,
    merge_ledgers,
    patch_client_source,
    read_episode_ledger,
    select_identities,
    split_by_identity,
)

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _manifest(tmp_path, suites=SUITES, tasks=10, episodes=6):
    entries = []
    for suite in suites:
        for task in range(tasks):
            for episode in range(episodes):
                entries.append(
                    {
                        "suite": suite,
                        "task_id": task,
                        "episode_idx": episode,
                        "base_seed": task * 100 + episode,
                        "init_state_index": episode,
                        # Three domains, matching the real manifest's shape.
                        "lighting_domain": ("ExtremeLow", "Low", "Normal")[episode % 3],
                    }
                )
    path = tmp_path / "libero-manifest.json"
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path


def test_selection_totals_one_hundred_and_covers_every_task(tmp_path) -> None:
    identities = select_identities(_manifest(tmp_path), per_suite=25)
    assert len(identities) == 100
    assert "lighting" in SELECTION_RULE
    counts: dict[str, int] = {}
    for item in identities:
        counts[item["suite"]] = counts.get(item["suite"], 0) + 1
    assert counts == {suite: 25 for suite in SUITES}
    for suite in SUITES:
        tasks = {item["task_id"] for item in identities if item["suite"] == suite}
        assert tasks == set(range(10)), "every task must appear in the subset"
    assert all(item["base_seed"] is not None for item in identities)


def test_small_probe_obeys_budget_and_covers_lighting_in_each_suite(tmp_path):
    chosen = select_identities(_manifest(tmp_path), per_suite=3)
    assert len(chosen) == 12
    for suite in SUITES:
        group = [item for item in chosen if item["suite"] == suite]
        assert len(group) == 3
        assert len({item["task_id"] for item in group}) == 3
        assert {item["lighting_domain"] for item in group} == {"ExtremeLow", "Low", "Normal"}
    with pytest.raises(ValueError, match="positive"):
        select_identities(_manifest(tmp_path), per_suite=0)


def test_probe_can_predeclare_three_domains_without_changing_client_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    extra = [
        dict(row, lighting_domain="ExtremeOver", episode_idx=row["episode_idx"] + 10)
        for row in payload["entries"]
    ]
    payload["entries"].extend(extra)
    manifest.write_text(json.dumps(payload))
    before = manifest.read_bytes()
    selected = select_identities(
        manifest, per_suite=3, lighting_domains=("ExtremeLow", "Low", "Normal")
    )
    assert len(selected) == 12
    assert {item["lighting_domain"] for item in selected} == {"ExtremeLow", "Low", "Normal"}
    assert manifest.read_bytes() == before


def test_selection_is_balanced_across_lighting_domains(tmp_path) -> None:
    """The withdrawn index-order rule was dark-skewed; the default must not be."""

    balanced = select_identities(_manifest(tmp_path), per_suite=25)
    counts: dict[str, int] = {}
    for item in balanced:
        counts[item["lighting_domain"]] = counts.get(item["lighting_domain"], 0) + 1
    # Three domains, 25 per suite: no domain may dominate.
    assert set(counts) == {"ExtremeLow", "Normal", "Low"}
    assert max(counts.values()) - min(counts.values()) <= 25, counts
    per_suite_domain: dict[tuple[str, str], int] = {}
    for item in balanced:
        key = (item["suite"], item["lighting_domain"])
        per_suite_domain[key] = per_suite_domain.get(key, 0) + 1
    for suite in SUITES:
        values = [v for (s, _), v in per_suite_domain.items() if s == suite]
        assert max(values) - min(values) <= 1, (suite, values)


def test_index_order_rule_is_retained_but_marked_withdrawn(tmp_path) -> None:
    """It must still describe the plan already run, and must not be the default."""

    identities = select_identities(_manifest(tmp_path), per_suite=25, lighting_balanced=False)
    assert len(identities) == 100
    per_task: dict[tuple[str, int], int] = {}
    for item in identities:
        key = (item["suite"], item["task_id"])
        per_task[key] = per_task.get(key, 0) + 1
    assert set(per_task.values()) == {2, 3}
    assert "withdrawn" in SELECTION_RULE


def test_split_interleaves_so_both_workers_see_every_suite(tmp_path) -> None:
    identities = select_identities(_manifest(tmp_path))
    buckets = split_by_identity(identities, workers=2)
    assert sum(len(bucket) for bucket in buckets) == 100
    assert [len(bucket) for bucket in buckets] == [50, 50]
    for bucket in buckets:
        assert {item["suite"] for item in bucket} == set(SUITES), (
            "a worker missing a whole suite would make the two cards incomparable"
        )
    keys = [(item["suite"], item["task_id"], item["episode_idx"]) for item in identities]
    assert len(set(keys)) == 100, "selection must not contain duplicates"


def test_patch_client_source_inserts_all_three_anchors() -> None:
    # Indentation mirrors the author file: the episode loop sits inside the task
    # loop, so the anchor is at twelve spaces.
    source = (
        "def _append_episode(path: Path | None, record: dict[str, Any]) -> None:\n"
        "    pass\n"
        "\n"
        "def run():\n"
        "    for task_id in range(2):\n"
        "        for ep_idx in ep_iter:\n"
        "            if ep_idx in task_completed:\n"
        "                continue\n"
        "            work()\n"
        "        self = None\n"
        "        self.last_action_stats = {\n"
    )
    patched = patch_client_source(source)
    assert "_owac_load_allowlist" in patched
    assert ALLOWLIST_CHECK.strip() in patched
    # one definition in the preamble, and two reads in the episode guard
    # (the condition and the membership test)
    assert patched.count("_OWAC_ALLOWLIST") == 3
    # The third anchor records each policy query for the synchronized timeline.
    assert TIMELINE_RECORD.strip() in patched
    assert patched.index("_owac_timeline_record(example, actions, response)") < patched.index(
        "self.last_action_stats"
    )


def test_patch_client_source_refuses_a_changed_author_file() -> None:
    with pytest.raises(ValueError, match="preamble not found"):
        patch_client_source("def something_else():\n    pass\n")
    source = "def _append_episode(path: Path | None, record: dict[str, Any]) -> None:\n    pass\n"
    with pytest.raises(ValueError, match="episode loop not found"):
        patch_client_source(source)


def test_ledger_merge_deduplicates_by_identity(tmp_path) -> None:
    first = tmp_path / "w0.jsonl"
    second = tmp_path / "w1.jsonl"
    shared = {"task_suite": "libero_10", "task_id": 3, "episode_idx": 1, "success": True}
    first.write_text(
        json.dumps(shared)
        + "\n"
        + json.dumps({"task_suite": "libero_10", "task_id": 3, "episode_idx": 2, "success": False})
        + "\n",
        encoding="utf-8",
    )
    # The same identity appears again in the other worker's ledger, plus garbage.
    second.write_text(
        json.dumps(shared) + "\n" + "{not json}\n" + json.dumps({"task_suite": "x"}) + "\n",
        encoding="utf-8",
    )
    merged = merge_ledgers([first, second])
    assert set(merged) == {("libero_10", 3, 1), ("libero_10", 3, 2)}
    assert len(read_episode_ledger(first)) == 2
    assert list(IDENTITY_KEY) == ["suite", "task_id", "episode_idx"]


def test_report_denominator_is_completed_identities_not_selected(tmp_path) -> None:
    identities = [
        {"suite": "libero_10", "task_id": 0, "episode_idx": 0},
        {"suite": "libero_10", "task_id": 0, "episode_idx": 1},
        {"suite": "libero_10", "task_id": 0, "episode_idx": 2},
    ]
    ledger = tmp_path / "w0.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "task_suite": "libero_10",
                "task_id": 0,
                "episode_idx": 0,
                "success": True,
                "wall_seconds": 42.0,
            }
        )
        + "\n"
        + json.dumps({"task_suite": "libero_10", "task_id": 0, "episode_idx": 1, "success": False})
        + "\n",
        encoding="utf-8",
    )
    report = _report(merge_ledgers([ledger]), identities)
    assert report["selected_identities"] == 3
    assert report["completed_rollouts"] == 2
    assert report["denominator"] == 2, "the denominator must be what actually completed"
    assert report["successes"] == 1
    assert report["success_rate"] == pytest.approx(0.5)
    assert report["not_completed"] == ["libero_10/0/2"]
    assert "never" in report["denominator_note"]
    assert report["by_suite"]["libero_10"] == {"completed": 2, "successes": 1}
    # only the record that carries wall_seconds contributes to the mean
    assert report["wall_seconds"]["mean"] == pytest.approx(42.0)
    assert "not 10,000" in report["budget_note"]


def test_held_out_filter_drops_train_and_unknown_episodes() -> None:
    """A held-out loop must exclude training episodes *and* anything it cannot place.

    The second half is the one that is easy to get wrong: keeping an episode the index
    does not know would silently reintroduce the contamination the filter exists to remove.
    """

    identities = [
        {"suite": "libero_10", "task_id": 0, "episode_idx": 0, "init_state_index": 0},
        {"suite": "libero_10", "task_id": 0, "episode_idx": 1, "init_state_index": 1},
        {"suite": "libero_10", "task_id": 0, "episode_idx": 2, "init_state_index": 2},
        {"suite": "libero_goal", "task_id": 3, "episode_idx": 4, "init_state_index": 4},
    ]
    index = {
        ("libero_10", 0, 0): "train",
        ("libero_10", 0, 1): "test",
        # (libero_10, 0, 2) deliberately absent: unknown must not be kept.
        ("libero_goal", 3, 4): "calibration",
    }
    kept, tally = filter_to_held_out(identities, index)
    assert [(item["suite"], item["task_id"], item["episode_idx"]) for item in kept] == [
        ("libero_10", 0, 1),
        ("libero_goal", 3, 4),
    ]
    assert tally == {"train": 1, "test": 1, "not_in_manifest": 1, "calibration": 1}


def test_held_out_filter_can_be_restricted_to_named_sides() -> None:
    identities = [
        {"suite": "libero_10", "task_id": 0, "episode_idx": 0, "init_state_index": 0},
        {"suite": "libero_10", "task_id": 0, "episode_idx": 1, "init_state_index": 1},
    ]
    index = {("libero_10", 0, 0): "test", ("libero_10", 0, 1): "calibration"}
    kept, _ = filter_to_held_out(identities, index, sides=("test",))
    assert len(kept) == 1 and kept[0]["episode_idx"] == 0


def test_every_arm_needs_a_config_the_queue_can_load() -> None:
    """Every arm the matrix can run must have a config its own validation accepts.

    A threshold correction was applied by a hand-written list of three config names and
    silently omitted the fourth queued arm.  Nothing was wrong with the code -- the config
    file simply could not be loaded by the corrected validation, and it sat second in a
    multi-day queue, so the run would have died at launch long after the mistake.

    This pins the naming the queue builds (``{platform}-{arm}-seed101``) against the arm
    matrix, so adding or renaming an arm breaks here rather than in the queue.
    """

    from owac.frameworks.n01_full.model import arm_matrix

    stems = {f"{spec.platform}-{spec.arm}-seed101" for spec in arm_matrix()}
    assert stems == {
        "libero-raw_base-seed101",
        "libero-raw_n01-seed101",
        "libero-rgb_base-seed101",
        "libero-rgb_n01-seed101",
        "robotwin2-raw_base-seed101",
        "robotwin2-raw_n01-seed101",
        "robotwin2-rgb_base-seed101",
        "robotwin2-rgb_n01-seed101",
    }, "the queue resolves configs by this naming; a drift here arms a run that cannot start"


def test_held_out_filter_uses_physical_initial_state_not_lighting_episode_index():
    # The real author manifest expands each initial state into five lighting trials.
    # trial 17 is initial state 3; a split on demo 17 leaks when demo 3 trained.
    identities = [
        {"suite": "libero_10", "task_id": 8, "episode_idx": 17, "init_state_index": 3},
        {"suite": "libero_10", "task_id": 8, "episode_idx": 23, "init_state_index": 4},
    ]
    index = {
        ("libero_10", 8, 3): "train",
        ("libero_10", 8, 4): "test",
        ("libero_10", 8, 17): "test",
        ("libero_10", 8, 23): "train",
    }
    kept, tally = filter_to_held_out(identities, index)
    assert [entry["episode_idx"] for entry in kept] == [23]
    assert tally == {"train": 1, "test": 1}
