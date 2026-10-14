"""Repair empty expert plans only, then restore methods before fixed-seed PI05 evaluation."""

import argparse
import ast
import hashlib
import inspect
import json
import os
import shutil
import signal
import textwrap
from pathlib import Path

from owac.runtime.rawvla_full_eval import digest, rows, write_json

PAPER_CASES = ("paper-0038", "paper-0222", "paper-0235", "paper-0454", "paper-0470")
TASKS = {"grab_roller", "open_laptop", "put_bottles_dustbin"}


class _ExpertGuards(ast.NodeTransformer):
    def __init__(self, kind):
        self.kind = kind
        self.count = 0

    def visit_Assign(self, node):
        if self.kind == "grasp":
            target = node.targets[0]
            names = (
                [item.id for item in target.elts if isinstance(item, ast.Name)]
                if isinstance(target, ast.Tuple)
                else []
            )
            if names == ["pre_grasp_pose", "grasp_pose"]:
                self.count += 1
                guard = ast.parse(
                    "if pre_grasp_pose is None or grasp_pose is None:\n"
                    "    self.plan_success = False\n"
                    "    _owac_repair_note('empty_grasp_plan')\n"
                    "    return arm_tag, []\n"
                ).body[0]
                return [node, ast.copy_location(guard, node)]
        else:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id in {"right_action", "left_action"}
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "grasp_actor"
            ):
                self.count += 1
                name = target.id
                guard = ast.parse(
                    f"if not {name}[1]:\n"
                    f"    _owac_repair_note('empty_bottle_actions:{name}')\n"
                    "    break\n"
                ).body[0]
                return [node, ast.copy_location(guard, node)]
        return node


def guarded_method(method, kind, events):
    """Retain the author method, adding only guards for its empty-plan branches."""
    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source)
    transformer = _ExpertGuards(kind)
    tree = transformer.visit(tree)
    expected = 1 if kind == "grasp" else 2
    if transformer.count != expected:
        raise ValueError(f"Expert guard anchor changed: {kind}: {transformer.count} != {expected}")
    ast.fix_missing_locations(tree)
    namespace = dict(method.__globals__)
    namespace["_owac_repair_note"] = events.append
    exec(compile(tree, inspect.getsourcefile(method) or "expert-guard", "exec"), namespace)
    proof = {
        "function": method.__qualname__,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "guarded_ast_sha256": hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        "guards": transformer.count,
    }
    return namespace[method.__name__], proof


def setup_identity(task):
    """Record initial physical context without rendering or advancing the simulator."""
    result = {"task": task.task_name, "actors": {}, "attributes": {}}
    for name in ("model_id", "model_name", "bottle_id", "bottle_num", "arm_tag"):
        if hasattr(task, name):
            value = getattr(task, name)
            if hasattr(value, "item"):
                value = value.item()
            result["attributes"][name] = str(value) if name == "arm_tag" else value
    for name in ("roller", "laptop", "dustbin", "bottles"):
        if not hasattr(task, name):
            continue
        actors = getattr(task, name)
        actors = actors if isinstance(actors, list) else [actors]
        for index, actor in enumerate(actors):
            pose = actor.get_pose()
            item = {"p": pose.p.tolist(), "q": pose.q.tolist()}
            if hasattr(actor, "get_qpos"):
                item["qpos"] = actor.get_qpos().tolist()
            result["actors"][f"{name}/{index}"] = item
    return result


def expert_check_with_repair(task):
    """Keep expert metadata/failed-plan semantics and undo all patches on exit."""
    entry = json.loads(os.environ["RAWVLA_EXPECTED_ENTRY"])
    if task.task_name not in TASKS or task.task_name != entry["task_name"]:
        raise ValueError("Expert repair is restricted to the three diagnosed task classes")
    if os.environ.get("ROBOTWIN_FORCE_EVAL_SEED") != "1":
        raise ValueError("Expert repair requires the existing fixed-seed evaluation contract")
    from envs._base_task import Base_Task

    evidence = Path(os.environ["ROBOTWIN_POLICY_AUDIT_DIR"])
    initial = setup_identity(task)
    events = []
    original_grasp = Base_Task.grasp_actor
    patched_grasp, proof = guarded_method(original_grasp, "grasp", events)
    original_play = type(task).play_once
    patches = [proof]
    if task.task_name == "put_bottles_dustbin":
        patched_play, proof = guarded_method(original_play, "bottles", events)
        patches.append(proof)
    else:
        patched_play = original_play
    result, error = None, None
    Base_Task.grasp_actor = patched_grasp
    type(task).play_once = patched_play
    try:
        result = task.play_once()
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        Base_Task.grasp_actor = original_grasp
        type(task).play_once = original_play
        write_json(
            evidence / "expert-repair.json",
            {
                "phase": "expert_check_only",
                "entry": entry,
                "setup_identity": initial,
                "patches": patches,
                "events": events,
                "expert_plan_success": bool(task.plan_success),
                "episode_info": result,
                "exception": error,
                "methods_restored_before_policy_setup": (
                    Base_Task.grasp_actor is original_grasp
                    and type(task).play_once is original_play
                ),
            },
        )


def prepare(parent, output):
    """Copy small immutable artifacts; carry 627 completed identities without rerunning them."""
    selection = json.loads((parent / "robotwin-paper-conditions.json").read_text())
    expected = {case["case_id"]: case for case in selection["entries"]}
    ledger = rows(parent / "robotwin-paper/episodes.jsonl")
    completed = {row["case_id"] for row in ledger}
    if len(ledger) != len(completed) or set(expected) - completed != set(PAPER_CASES):
        raise ValueError(
            "Parent ledger no longer has exactly the five diagnosed missing identities"
        )
    for row in ledger:
        if row["entry"] != expected[row["case_id"]]["entry"]:
            raise ValueError("Parent completed identity differs from original metadata")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent / "frozen", output / "frozen")
    shutil.copy2(__file__, output / "frozen/src/owac/runtime/rawvla_repair.py")
    shutil.copytree(parent / "runtime", output / "runtime")
    for name in ("protocol.json", "robotwin-paper-conditions.json"):
        shutil.copy2(parent / name, output / name)
    directory = output / "robotwin-paper"
    directory.mkdir()
    for name in ("protocol.json", "episodes.jsonl"):
        shutil.copy2(parent / "robotwin-paper" / name, directory / name)
    (output / "task_config").symlink_to((parent / "task_config").resolve())
    script = output / "runtime/robotwin_eval.py"
    code = script.read_text()
    anchor = "                episode_info = TASK_ENV.play_once()\n"
    if code.count(anchor) != 1:
        raise ValueError("Frozen expert-check call anchor changed")
    code = code.replace(
        anchor,
        "                from owac.runtime.rawvla_repair import expert_check_with_repair\n"
        "                episode_info = expert_check_with_repair(TASK_ENV)\n",
    )
    script.write_text(code)
    write_json(
        output / "repair-protocol.json",
        {
            "version": "owac-pi05-expert-infrastructure-repair-v1",
            "parent": str(parent),
            "parent_protocol_sha256": digest(parent / "protocol.json"),
            "selection_sha256": digest(parent / "robotwin-paper-conditions.json"),
            "inherited_ledger_sha256": digest(directory / "episodes.jsonl"),
            "inherited_completed": len(ledger),
            "repair_cases": list(PAPER_CASES),
            "repair_module_sha256": digest(__file__),
            "patched_eval_sha256": digest(script),
            "original_eval_sha256": digest(parent / "runtime/robotwin_eval.py"),
            "scope": "Empty expert grasp plans only; restore original methods before policy reset",
            "unchanged": "seed, scenes, assets, lighting/noise, language RNG, PI05 weights/"
            "normalization, policy actions, success criteria, max steps",
        },
    )


def run(root, output):
    from owac.runtime.rawvla_job_guard import terminate_children
    from owac.runtime.rawvla_paper_eval import run as evaluate

    protocol = json.loads((output / "repair-protocol.json").read_text())
    if protocol["repair_module_sha256"] != digest(__file__) or protocol[
        "patched_eval_sha256"
    ] != digest(output / "runtime/robotwin_eval.py"):
        raise ValueError("Repair source changed after preparation")
    signal.signal(signal.SIGTERM, terminate_children)
    signal.signal(signal.SIGINT, terminate_children)
    evaluate(root, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--root", default=Path("/workspace/owac/rawvla"), type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.parent, args.output)
    else:
        run(args.root, args.output)


if __name__ == "__main__":
    main()
