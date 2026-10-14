"""核对官方 10000+3250 仿真测试清单与 S0 训练源轨迹的身份交集（只读）。

背景：Lead 要求确认官方 test manifest 不参与训练拆分，并把两侧条目数、交集大小、
身份字段定义与命令写成独立文件。本脚本只读：

* 训练侧：`data/s0-manifest/run-*/inventory.json`（逐 NPZ 记录，含 metadata 身份字段）
  与 `manifests/s0-manifest/splits.json`（冻结的 2403 个 source_trajectory_id）。
* 测试侧：`/workspace/owac/rawvla/records/pi05-full-20261010-v3/{libero,robotwin}-manifest.json`。

它不写回 splits/coverage/inventory；输出单独的 `official-test-disjointness.json`。

身份层级（两侧字段名不同，必须显式对齐）：

``task``
    测试 LIBERO `(suite, task_id)`；训练 LIBERO `(metadata.suite, metadata.task_id)`。
    测试 RoboTwin `task_name`；训练 RoboTwin `metadata.task_name`。
``source_episode``
    测试 LIBERO 额外加上 `init_state_index`；训练 LIBERO 用 `metadata.demo_index`。
    测试 RoboTwin 用 `(task_name, dataset_episode_index)`；训练 RoboTwin 用
    `(metadata.task_name, metadata.episode)`。
    两侧索引语义是否严格同序**没有生产者代码确认**，只登记为 caveat。

用法（远端）：

    python3 official_test_disjointness.py --run-dir <run> --output <out.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RECORDS = Path("/workspace/owac/rawvla/records/pi05-full-20261010-v3")
TEST_MANIFESTS = {
    "libero": RECORDS / "libero-manifest.json",
    "robotwin": RECORDS / "robotwin-manifest.json",
}
SAMPLE_LIMIT = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded(values: set) -> dict[str, Any]:
    ordered = sorted(values, key=lambda item: json.dumps(item, ensure_ascii=False))
    return {"count": len(ordered), "sample": [list(item) for item in ordered[:SAMPLE_LIMIT]]}


def intersection(train: set, test: set) -> dict[str, Any]:
    shared = train & test
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_intersection": len(shared),
        "intersection": bounded(shared),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="官方测试清单与训练源轨迹身份交集（只读）")
    parser.add_argument(
        "--run-dir", required=True, type=Path, help="冻结 run 目录（含 inventory.json）"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--splits", type=Path, default=None, help="冻结 splits.json（默认 run 目录同级）"
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    # 伴随文件与 inventory.json 同目录；也允许显式指定
    splits_path = args.splits or (run_dir / "splits.json")
    inventory = json.loads((run_dir / "inventory.json").read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    records = inventory["records"]
    split_members = {
        item["source_trajectory_id"]: item["split"] for item in splits["main"]["assignments"]
    }

    train_libero = [r for r in records if r["platform"] == "libero"]
    train_robotwin = [r for r in records if r["platform"] == "robotwin"]
    train_task = {
        "libero": {(r["metadata"]["suite"], r["metadata"]["task_id"]) for r in train_libero},
        "robotwin": {r["metadata"]["task_name"] for r in train_robotwin},
    }
    train_source = {
        "libero": {
            (r["metadata"]["suite"], r["metadata"]["task_id"], r["metadata"]["demo_index"])
            for r in train_libero
        },
        "robotwin": {
            (r["metadata"]["task_name"], r["metadata"]["episode"]) for r in train_robotwin
        },
    }

    test_manifests: dict[str, dict[str, Any]] = {}
    for platform, path in TEST_MANIFESTS.items():
        test_manifests[platform] = json.loads(path.read_text(encoding="utf-8"))
    test_task = {
        "libero": {
            (entry["suite"], entry["task_id"]) for entry in test_manifests["libero"]["entries"]
        },
        "robotwin": {entry["task_name"] for entry in test_manifests["robotwin"]["entries"]},
    }
    test_source = {
        "libero": {
            (entry["suite"], entry["task_id"], entry["init_state_index"])
            for entry in test_manifests["libero"]["entries"]
        },
        "robotwin": {
            (entry["task_name"], entry["dataset_episode_index"])
            for entry in test_manifests["robotwin"]["entries"]
        },
    }

    # task_id ↔ task_name 对照（训练侧 metadata 同时有两者），供 Lead 核对编号同序假设
    task_id_map: dict[str, dict[str, str]] = defaultdict(dict)
    for record in train_libero:
        metadata = record["metadata"]
        task_id_map[metadata["suite"]][str(metadata["task_id"])] = metadata["task_name"]

    # 每个 (suite, task_id) / task 的测试条目数与训练文件数
    per_task_counts: dict[str, dict[str, Any]] = {}
    for platform in ("libero", "robotwin"):
        train_counter: Counter = Counter()
        test_counter: Counter = Counter()
        if platform == "libero":
            for record in train_libero:
                train_counter[(record["metadata"]["suite"], record["metadata"]["task_id"])] += 1
            for entry in test_manifests["libero"]["entries"]:
                test_counter[(entry["suite"], entry["task_id"])] += 1
            key_of = lambda pair: f"{pair[0]}/task_id={pair[1]}"  # noqa: E731
        else:
            for record in train_robotwin:
                train_counter[record["metadata"]["task_name"]] += 1
            for entry in test_manifests["robotwin"]["entries"]:
                test_counter[entry["task_name"]] += 1
            key_of = lambda pair: str(pair)  # noqa: E731
        per_task_counts[platform] = {
            key_of(key): {
                "train_files": train_counter.get(key, 0),
                "test_entries": test_counter.get(key, 0),
            }
            for key in sorted(set(train_counter) | set(test_counter), key=key_of)
        }

    source_intersection = {
        platform: intersection(train_source[platform], test_source[platform])
        for platform in ("libero", "robotwin")
    }
    # 条目级：测试清单里有多少条 entry 落在训练源 episode 身份上（每个身份 5 个照明域）
    entry_counts: dict[str, Any] = {}
    for platform in ("libero", "robotwin"):
        shared = train_source[platform] & test_source[platform]
        if platform == "libero":
            matching = sum(
                1
                for entry in test_manifests["libero"]["entries"]
                if (entry["suite"], entry["task_id"], entry["init_state_index"]) in shared
            )
        else:
            matching = sum(
                1
                for entry in test_manifests["robotwin"]["entries"]
                if (entry["task_name"], entry["dataset_episode_index"]) in shared
            )
        total = len(test_manifests[platform]["entries"])
        entry_counts[platform] = {
            "matching_entries": matching,
            "total_entries": total,
            "matching_fraction": round(matching / total, 6),
            "note": "每个源 episode 身份在测试清单中出现 5 次（5 个照明域）",
        }
    task_intersection = {
        platform: intersection(train_task[platform], test_task[platform])
        for platform in ("libero", "robotwin")
    }

    # RoboTwin：测试清单顶层 dataset 模板把 dataset_episode_index 直接绑定到 episode{N}.pkl/.hdf5，
    # 与训练 metadata 的 source_trajectory/source_hdf5 同名 → 身份映射为字段级声明，不是假设
    robotwin_templates = test_manifests["robotwin"].get("dataset", {})
    test_rw_seed = {
        (entry["task_name"], entry["dataset_episode_index"]): entry["dataset_seed"]
        for entry in test_manifests["robotwin"]["entries"]
    }
    seed_matches = sum(
        1
        for record in train_robotwin
        if test_rw_seed.get((record["metadata"]["task_name"], record["metadata"]["episode"]))
        == record["metadata"]["dataset_seed"]
    )
    cross_checks = {
        "robotwin_episode_template": {
            "test_manifest_dataset_block": {
                "episode_data_template": robotwin_templates.get("episode_data_template"),
                "trajectory_template": robotwin_templates.get("trajectory_template"),
                "relative_dir_template": robotwin_templates.get("relative_dir_template"),
            },
            "train_metadata_fields": {
                "source_hdf5": "episode{N}.hdf5",
                "source_trajectory": "episode{N}.pkl",
            },
            "verdict": (
                "测试 dataset_episode_index == 训练 source_trajectory 的 episode 编号："
                "由测试清单自身的模板字段声明，不是推断"
            ),
        },
        "robotwin_dataset_seed_agreement": {
            "n_compared": len(train_robotwin),
            "n_equal": seed_matches,
            "note": (
                "训练 metadata.dataset_seed 与测试同 (task, dataset_episode_index) "
                "条目的 dataset_seed 相等数"
            ),
        },
        "libero_index_mapping": {
            "test_side_fields": ["suite", "task_id", "init_state_index"],
            "train_side_fields": ["metadata.suite", "metadata.task_id", "metadata.demo_index"],
            "test_manifest_source_name": (
                "protocol.json manifests.libero.source 末尾为 "
                "libero_manifest_50_init_states_10000_rollouts.json"
            ),
            "evidence_for_same_indexing": [
                "两侧每个 task 的索引基数都是 50"
                "（测试 init_state_index 0..49；训练 demo_index 0..49）",
                "测试清单无 task_name，只有 suite/task_id；"
                "训练侧 task_id ↔ task_name 对照见 train_side",
                "LIBERO 标准评测使用数据集初始状态；但清单未声明 init_state_index 指向哪一份列表",
            ],
            "verdict": (
                "语义级『同一源 episode』为高可能性推断，未经生产者代码确认；"
                "结构级包含关系（训练身份全部落在测试网格内）与该推断无关，恒成立"
            ),
        },
    }

    # 交集在 train/dev/final 中的分布（若存在交集，必须能定位到具体条目）
    shared_members: dict[str, Any] = {}
    if source_intersection["libero"]["n_intersection"]:
        shared_members["libero"] = {
            f"{suite}/task_id={task_id}#demo={index}": (
                f"{suite} | {task_id_map.get(suite, {}).get(str(task_id))} | demo_index={index}"
            )
            for suite, task_id, index in sorted(train_source["libero"] & test_source["libero"])[
                :SAMPLE_LIMIT
            ]
        }
    if source_intersection["robotwin"]["n_intersection"]:
        shared_members["robotwin"] = {
            f"{task}#episode={episode}": f"episode{episode}.pkl"
            for task, episode in sorted(train_source["robotwin"] & test_source["robotwin"])[
                :SAMPLE_LIMIT
            ]
        }

    split_participation = Counter(split_members.values())
    intersects = (
        source_intersection["libero"]["n_intersection"]
        + source_intersection["robotwin"]["n_intersection"]
    )
    payload = {
        "schema": "n04-s0-official-test-disjointness-v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "官方 10000(LIBERO)+3250(RoboTwin) 仿真测试清单是否参与 S0 训练拆分",
        "command": (
            "PYTHONPATH=<snapshot>/src /usr/bin/python3.12 official_test_disjointness.py "
            f"--run-dir {run_dir} --output <this file>"
        ),
        "inputs": {
            "train_inventory": {
                "path": str(run_dir / "inventory.json"),
                "sha256": sha256_file(run_dir / "inventory.json"),
                "size_bytes": (run_dir / "inventory.json").stat().st_size,
            },
            "train_splits": {"path": str(splits_path), "sha256": sha256_file(splits_path)},
            "test_manifests": {
                platform: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "entries": len(test_manifests[platform]["entries"]),
                    "version": test_manifests[platform].get("version"),
                    "benchmark": test_manifests[platform].get("benchmark"),
                    "benchmark_seed": test_manifests[platform].get("benchmark_seed"),
                    "frozen_protocol_reference": (
                        "records_frozen_protocol: "
                        "/workspace/owac/rawvla/records/pi05-full-20261010-v3"
                    ),
                }
                for platform, path in TEST_MANIFESTS.items()
            },
        },
        "identity_definitions": {
            "train_source_trajectory_id": {
                "libero": "libero:<dataset_name>:<task_name>:<demo_key>（NPZ metadata）",
                "robotwin": "robotwin:<task_name>:<source_trajectory>（NPZ metadata）",
            },
            "train_task_level": {
                "libero": "(metadata.suite, metadata.task_id)",
                "robotwin": "metadata.task_name",
            },
            "train_source_episode_level": {
                "libero": "(metadata.suite, metadata.task_id, metadata.demo_index)",
                "robotwin": "(metadata.task_name, metadata.episode)",
            },
            "test_task_level": {
                "libero": "(entry.suite, entry.task_id)",
                "robotwin": "entry.task_name",
            },
            "test_source_episode_level": {
                "libero": "(entry.suite, entry.task_id, entry.init_state_index)",
                "robotwin": "(entry.task_name, entry.dataset_episode_index)",
            },
            "caveat": (
                "测试清单不携带训练侧 source_trajectory_id 字符串，也不携带 "
                "task_name（LIBERO）；source_episode 级比较依赖『训练 demo_index 与测试 "
                "init_state_index、训练 episode 与测试 dataset_episode_index 指向同一源 "
                "episode』这一未由生产者代码确认的假设"
            ),
        },
        "train_side": {
            "n_records": len(records),
            "n_source_trajectory_ids": len(split_members),
            "split_membership": dict(split_participation),
            "libero_task_id_to_train_task_name": {
                suite: dict(sorted(mapping.items(), key=lambda kv: int(kv[0])))
                for suite, mapping in sorted(task_id_map.items())
            },
        },
        "counts": {
            "train_sources": {
                "libero": len(train_source["libero"]),
                "robotwin": len(train_source["robotwin"]),
            },
            "test_entries": {
                "libero": len(test_manifests["libero"]["entries"]),
                "robotwin": len(test_manifests["robotwin"]["entries"]),
            },
        },
        "interpretation": {
            "structural_overlap": (
                "恒成立：测试清单覆盖 (suite, task_id, init_state_index) 的完整网格 "
                "4×10×50=2000 与 (task_name, dataset_episode_index) 的 13×50=650，"
                "训练侧身份必然落在网格内，因此该层交集等于训练侧全部条目；"
                "此结论不依赖两侧编号是否同序"
            ),
            "semantic_overlap": (
                "RoboTwin：字段级确认（测试模板 episode{dataset_episode_index}.pkl/.hdf5 "
                "与训练 source_trajectory/source_hdf5 同名）→ 同一源 episode。"
                "LIBERO：高可能性推断（50/任务基数、无 task_name、标准评测使用数据集初始状态），"
                "未经生产者代码确认"
            ),
            "what_is_not_claimed": (
                "测试 rollout 的观测/动作不在训练集中；训练拆分代码从未读取测试清单"
                "（见 code_path_isolation）"
            ),
        },
        "cross_checks": cross_checks,
        "task_level": task_intersection,
        "source_episode_level": source_intersection,
        "test_entries_matching_train_source_episode": entry_counts,
        "intersection_members": shared_members,
        "per_task_counts": per_task_counts,
        "code_path_isolation": {
            "claim": "官方测试清单从不被清单/拆分代码读取",
            "evidence": [
                "src/owac/raw/n04_manifest.py 只读取配置 remote.data.libero_raw / "
                "robotwin_raw 下的 **/*.npz",
                "grep records_frozen_protocol|libero-manifest|robotwin-manifest "
                "src/owac/raw/n04_manifest.py → 0 命中",
                "splits.json 的 2403 个 source_trajectory_id 全部来自训练数据根",
            ],
            "verified_by": "grep + 代码路径检查（非运行时插桩）",
        },
        "verdict": {
            "test_manifest_participates_in_training_split": False,
            "source_episode_identity_overlap": intersects,
            "note": (
                "测试清单不在训练拆分中；但若 source_episode 级交集非空，说明部分测试 "
                "episode 与训练 episode 共享同一源 episode（同样的初始状态/数据集 "
                "episode），这是数据协议事实，不是拆分代码泄漏；是否可接受需由 Lead "
                "记录到 S0 结论与后续 S1/S6 泛化表述中"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "task_level": {
                    key: {k: v for k, v in value.items() if k != "intersection"}
                    for key, value in task_intersection.items()
                },
                "source_episode_level": {
                    key: {k: v for k, v in value.items() if k != "intersection"}
                    for key, value in source_intersection.items()
                },
                "test_entries_matching_train_source_episode": entry_counts,
                "intersects": intersects,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
