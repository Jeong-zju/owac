"""Expert failure handling must not change policy-time methods or completed identities."""

import json
import sys
from types import SimpleNamespace

import pytest

from owac.runtime.rawvla_repair import expert_check_with_repair, guarded_method


class FakeExpert:
    def __init__(self, poses):
        self.poses = poses
        self.plan_success = True

    def choose_grasp_pose(self):
        return self.poses

    def grasp_actor(self, arm_tag):
        pre_grasp_pose, grasp_pose = self.choose_grasp_pose()
        if pre_grasp_pose == grasp_pose:
            assert pre_grasp_pose is not None, "target_pose cannot be None for move action."
            return arm_tag, [pre_grasp_pose]
        return arm_tag, [pre_grasp_pose, grasp_pose]

    def play_once(self):
        self.grasp_actor("left")
        return {"info": {"{A}": "roller/base0"}}


def test_empty_plan_is_preserved_as_expert_failure_without_fabricated_action():
    events = []
    method, proof = guarded_method(FakeExpert.grasp_actor, "grasp", events)
    expert = FakeExpert((None, None))
    assert method(expert, "left") == ("left", [])
    assert expert.plan_success is False
    assert events == ["empty_grasp_plan"]
    assert proof["guards"] == 1


@pytest.mark.parametrize("poses", [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [4, 5, 6])])
def test_valid_expert_actions_are_byte_equivalent(poses):
    events = []
    method, _ = guarded_method(FakeExpert.grasp_actor, "grasp", events)
    expert = FakeExpert(poses)
    assert method(expert, "right") == FakeExpert.grasp_actor(expert, "right")
    assert expert.plan_success is True
    assert events == []


class BottleExpert:
    def __init__(self, actions):
        self.actions = actions
        self.count = 0
        self.info = {"info": {}}

    def grasp_actor(self):
        self.count += 1
        return "right", self.actions[self.count - 1]

    def play_once(self):
        for _ in range(1):
            right_action = self.grasp_actor()
            right_action[1][0].target_pose[2] += 0.06
            left_action = self.grasp_actor()
            left_action[1][0].target_pose[2] -= 0.06
        self.info["info"] = {"{A}": "bottle/base1"}
        return self.info


@pytest.mark.parametrize("actions", [[[]], [[SimpleNamespace(target_pose=[0, 0, 0])], []]])
def test_empty_bottle_plan_still_returns_author_metadata(actions):
    events = []
    method, proof = guarded_method(BottleExpert.play_once, "bottles", events)
    assert method(BottleExpert(actions)) == {"info": {"{A}": "bottle/base1"}}
    assert len(events) == 1
    assert proof["guards"] == 2


def test_unexpected_author_code_is_rejected():
    with pytest.raises(ValueError, match="anchor changed"):
        guarded_method(BottleExpert.grasp_actor, "grasp", [])


def expert_environment(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "envs", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "envs._base_task", SimpleNamespace(Base_Task=FakeExpert))
    monkeypatch.setenv("RAWVLA_EXPECTED_ENTRY", json.dumps({"task_name": "grab_roller"}))
    monkeypatch.setenv("ROBOTWIN_FORCE_EVAL_SEED", "1")
    monkeypatch.setenv("ROBOTWIN_POLICY_AUDIT_DIR", str(tmp_path))
    task = FakeExpert((None, None))
    task.task_name = "grab_roller"
    return task


def test_expert_methods_restored_before_any_policy_setup(monkeypatch, tmp_path):
    task = expert_environment(monkeypatch, tmp_path)
    grasp, play = FakeExpert.grasp_actor, FakeExpert.play_once
    assert expert_check_with_repair(task) == {"info": {"{A}": "roller/base0"}}
    assert FakeExpert.grasp_actor is grasp and FakeExpert.play_once is play
    audit = json.loads((tmp_path / "expert-repair.json").read_text())
    assert audit["methods_restored_before_policy_setup"]
    assert audit["expert_plan_success"] is False
    # Original failure semantics remain available outside the expert-only scope.
    with pytest.raises(AssertionError):
        task.grasp_actor("left")


def test_unrelated_error_is_not_swallowed_and_methods_restored(monkeypatch, tmp_path):
    task = expert_environment(monkeypatch, tmp_path)

    def unexpected(self):
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(FakeExpert, "play_once", unexpected)
    grasp = FakeExpert.grasp_actor
    with pytest.raises(RuntimeError, match="unrelated failure"):
        expert_check_with_repair(task)
    assert FakeExpert.grasp_actor is grasp and FakeExpert.play_once is unexpected
    assert json.loads((tmp_path / "expert-repair.json").read_text())["exception"]
