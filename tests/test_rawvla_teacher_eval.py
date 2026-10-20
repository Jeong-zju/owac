"""Closed-loop contract tests; no simulator or policy rollout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from owac.simulation.rawvla_teacher_eval import (
    Attempt,
    AttemptLedger,
    TeacherEvalBlocked,
    fixed_denominators,
    load_initial_states,
    paired_noise,
    read_native_loop_constants,
)


def test_actual_initial_states_are_frozen_10_20_when_present():
    manifest = Path("outputs/rawvla-teacher-20261020/initial-states/manifest.json")
    if not manifest.is_file():
        pytest.skip("actual initial-state manifest is absent in this checkout")
    import hashlib

    payload = load_initial_states(
        manifest, expected_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert len(payload["validation"]) == 10
    assert len(payload["test"]) == 20
    assert len({item["id"] for item in payload["episodes"]}) == 30


def test_shared_noise_is_episode_frame_bound_not_role_bound():
    a = paired_noise(seed=101, episode_id="validation-101000", frame=0)
    b = paired_noise(seed=101, episode_id="validation-101000", frame=0)
    c = paired_noise(seed=101, episode_id="validation-101001", frame=0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (1, 10, 32)


def test_attempt_ledger_never_retries_and_retains_failure(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    attempt = Attempt("learned_raw_teacher", "test-101100", "test", 101100, "a" * 64, "running")
    ledger.begin(attempt)
    ledger.complete(attempt, success=False, real_steps=12, replans=3)
    with pytest.raises(TeacherEvalBlocked, match="duplicate"):
        ledger.begin(attempt)
    text = (tmp_path / "attempts.jsonl").read_text()
    assert '"status": "failed"' in text
    assert '"real_steps": 12' in text


def test_ordinary_task_failure_remains_valid_control_evidence(tmp_path):
    import json

    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    attempt = Attempt("learned_raw_teacher", "test-timeout", "test", 101, "a" * 64, "running")
    ledger.begin(attempt)
    ledger.complete(attempt, success=False, real_steps=300, replans=60)
    row = json.loads(ledger.path.read_text().splitlines()[-1])
    assert row["success"] is False and row["status"] == "failed"
    assert row["valid_control_evidence"] is True


def test_fixed_denominators_and_native_constants_contract():
    result = fixed_denominators(
        validation_attempted=5, validation_success=4, test_attempted=20, test_success=16
    )
    assert result["test"]["success"] == 16
    assert result["test"]["attempted"] == 20
    assert result["screen"]["teacher_at_least_16_of_20"] is True
    constants = read_native_loop_constants()
    assert constants["dummy_action"] == [0.0] * 6 + [-1.0]
    assert constants["max_steps"]["libero_goal"] == 300


def test_driver_unpacks_step_tuples_and_calls_shared_controller_contract(tmp_path):
    import numpy as np

    from owac.runtime.rawvla_teacher_budget import PhaseBudget
    from owac.simulation.rawvla_teacher_eval import AttemptLedger, ClosedLoopDriver

    class FakeEnv:
        def __init__(self):
            self.steps = 0
            self.resets = 0

        def reset(self):
            self.resets += 1
            return {"raw": {"agentview": 0, "wrist": 0}, "robot_state": np.zeros(8)}

        def step(self, action):
            self.steps += 1
            return (
                {"raw": {"agentview": self.steps, "wrist": self.steps}, "robot_state": np.zeros(8)},
                0.0,
                False,
                {},
            )

        def check_success(self):
            return False

        def close(self):
            pass

    class FakeController:
        def __init__(self):
            self.reset_count = 0
            self.observe_count = 0
            self.act_count = 0

        def reset(self):
            self.reset_count += 1

        def observe(self, observation, step):
            self.observe_count += 1

        def act(self, observation, state, noise, *, episode_id, policy_step):
            self.act_count += 1
            return np.zeros((5, 7), dtype=np.float32), {"noise": np.asarray(noise).shape}

    budget = PhaseBudget(tmp_path / "budget")
    catalog = {
        "validation": [
            {
                "id": "validation-1",
                "split": "validation",
                "seed": 1,
                "state_sha256": __import__("hashlib")
                .sha256(np.zeros(79, dtype=np.float64).tobytes())
                .hexdigest(),
                "array_index": 0,
            }
        ]
    }
    driver = ClosedLoopDriver(
        catalog=catalog,
        budget=budget,
        ledger=AttemptLedger(tmp_path / "attempts.jsonl"),
        max_steps=2,
        setup_wait_steps=2,
        result_dir=tmp_path / "episodes",
    )
    controller = FakeController()
    with budget.lease("driver-test", gpu=False):
        results = driver.run_role(
            role="learned_raw_teacher",
            split="validation",
            env_factory=lambda entry, states, role: FakeEnv(),
            controller=controller,
            states_array=np.zeros((1, 79)),
        )
    assert results[0].steps == 2
    assert controller.reset_count == 1
    assert controller.observe_count == 4
    assert controller.act_count == 1
    assert (tmp_path / "episodes" / "learned_raw_teacher_validation-1.json").is_file()
    assert (tmp_path / "episodes" / "learned_raw_teacher_validation-1.npz").is_file()


def test_env_adapter_seeds_before_reset_sets_exact79_and_no_hidden_reset():
    import numpy as np

    from owac.simulation.rawvla_teacher_controller import LIBEROEnvAdapter

    class FakeEnv:
        def __init__(self):
            self.events = []
            self.steps = 0

        def seed(self, seed):
            self.events.append(("seed", seed))

        def reset(self):
            self.events.append(("reset",))
            return {
                "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.zeros(4),
                "robot0_gripper_qpos": np.zeros(2),
            }

        def get_sim_state(self):
            return np.zeros(79)

        def set_init_state(self, state):
            self.events.append(("set_init_state", tuple(np.asarray(state).shape)))
            return {
                "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.zeros(4),
                "robot0_gripper_qpos": np.zeros(2),
            }

        def step(self, action):
            self.events.append(("step", self.steps))
            self.steps += 1
            return (
                {
                    "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                    "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
                    "robot0_eef_pos": np.zeros(3),
                    "robot0_eef_quat": np.zeros(4),
                    "robot0_gripper_qpos": np.zeros(2),
                },
                0.0,
                False,
                {},
            )

        def check_success(self):
            return False

        def close(self):
            pass

    env = FakeEnv()
    adapter = LIBEROEnvAdapter(
        author_root="unused",
        libero_config="unused",
        env_factory=lambda task, seed: env,
    )
    observation = adapter.reset(
        {"seed": 17, "array_index": 0, "state_sha256": "a" * 64},
        np.zeros((1, 79), dtype=np.float64),
    )
    assert observation["robot_state"].shape == (8,)
    assert env.events[0] == ("seed", 17)
    assert env.events[1] == ("reset",)
    set_positions = [i for i, event in enumerate(env.events) if event[0] == "set_init_state"]
    assert set_positions and env.events[set_positions[0]][1] == (79,)
    assert not any(event[0] == "reset" for event in env.events[set_positions[0] + 1 :])
    assert sum(event[0] == "step" for event in env.events) == 0


def test_deployment_state_normalization_does_not_clip():
    from types import SimpleNamespace

    import numpy as np

    from owac.simulation.rawvla_teacher_controller import normalize_state8

    stats = SimpleNamespace(
        state={"q01": np.zeros(8, dtype=np.float32), "q99": np.ones(8, dtype=np.float32)}
    )
    out = normalize_state8(np.full(8, 2.0, dtype=np.float32), stats)
    assert out.shape == (8,)
    assert out[0] > 1.0


def test_bound_adapter_driver_and_real_controller_protocol(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    import numpy as np
    import torch

    from owac.runtime.rawvla_teacher_budget import PhaseBudget
    from owac.simulation.rawvla_teacher_controller import (
        LIBEROEnvAdapter,
        TeacherPolicyController,
    )
    from owac.simulation.rawvla_teacher_eval import AttemptLedger, ClosedLoopDriver

    raw_module = ModuleType("rawvla_bench.raw")

    def fake_raw(rgb, **kwargs):
        return np.zeros((256, 256, 3), dtype=np.uint8)

    raw_module.make_rawvla_observation = fake_raw
    monkeypatch.setitem(sys.modules, "rawvla_bench.raw", raw_module)

    import owac.simulation.rawvla_teacher_controller as controller_module

    monkeypatch.setattr(
        controller_module,
        "load_pinned_postprocessor",
        lambda: lambda normalized, state, stats, **kwargs: np.zeros((10, 7), dtype=np.float32),
    )

    class FrontendModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, raw, state=None, theta_prev=None):
            return SimpleNamespace(
                rgb=torch.zeros((1, 3, 256, 256)),
                state=torch.zeros((1, 128)),
                theta=torch.zeros((1, 64)),
            )

    class Policy:
        def __init__(self):
            self.training = False
            self.image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def _prepare_examples(self, examples, include_actions=False):
            assert include_actions is False
            assert len(examples[0]["image"]) == 2
            assert examples[0]["state"].shape == (8,)
            return {"image": np.zeros((1, 3, 3, 4, 4)), "state": np.zeros((1, 1, 32))}

        def _build_observation_from_batch(self, batch):
            return {"batch": batch}

        def _image_to_tensor(self, image):
            return image

        def sample_actions(self, observation, noise=None, num_steps=10):
            assert noise.shape == (1, 10, 32)
            return torch.zeros((1, 10, 32))

    teacher = SimpleNamespace(
        frontend=SimpleNamespace(
            model=FrontendModel(), eval=lambda: None, train=lambda mode=True: None
        ),
        policy=Policy(),
    )

    class NativeEnv:
        def __init__(self):
            self.steps = 0

        def seed(self, seed):
            self.seed_value = seed

        def reset(self):
            return self._obs()

        def get_sim_state(self):
            return np.zeros(79)

        def set_init_state(self, state):
            assert np.asarray(state).shape == (79,)
            return self._obs()

        def _obs(self):
            return {
                "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.zeros(4),
                "robot0_gripper_qpos": np.zeros(2),
            }

        def step(self, action):
            self.steps += 1
            return self._obs(), 0.0, False, {}

        def check_success(self):
            return False

        def close(self):
            pass

    adapter = LIBEROEnvAdapter(
        author_root="unused", libero_config="unused", env_factory=lambda task, seed: NativeEnv()
    )
    controller = TeacherPolicyController(
        teacher=teacher,
        stats=SimpleNamespace(
            state={"q01": np.zeros(8), "q99": np.ones(8)},
            action={"q01": np.zeros(7), "q99": np.ones(7)},
        ),
        role="learned_raw_teacher",
        author_root="unused",
    )
    budget = PhaseBudget(tmp_path / "budget")
    catalog = {
        "validation": [
            {
                "id": "v",
                "split": "validation",
                "seed": 1,
                "array_index": 0,
                "state_sha256": __import__("hashlib")
                .sha256(np.zeros(79, dtype=np.float64).tobytes())
                .hexdigest(),
            }
        ]
    }
    driver = ClosedLoopDriver(
        catalog=catalog,
        budget=budget,
        ledger=AttemptLedger(tmp_path / "attempts.jsonl"),
        max_steps=1,
        setup_wait_steps=10,
    )
    with budget.lease("bound-test", gpu=False):
        results = driver.run_role(
            role="learned_raw_teacher",
            split="validation",
            env_factory=lambda entry, states, role: adapter.bind(entry, states),
            controller=controller,
            states_array=np.zeros((1, 79)),
        )
    assert results[0].success is False
    assert results[0].steps == 1
    assert results[0].replans == 1
    assert adapter.env.steps == 10 + 1
    metadata = results[0].replan_metadata[0]
    assert metadata["camera_frame_indices"] == [5, 6, 7, 8, 9, 10]
    assert metadata["normalized_actions"].shape == (1, 10, 32)
    assert metadata["physical_actions"].shape == (10, 7)
    assert metadata["frontend_seconds"] >= 0.0
    assert metadata["full_seconds"] >= metadata["frontend_seconds"]


def test_keyboard_interrupt_persists_partial_episode_and_does_not_consume_next_state(
    tmp_path, monkeypatch
):
    import hashlib
    import json

    import imageio.v2 as imageio
    import numpy as np

    from owac.runtime.rawvla_teacher_budget import PhaseBudget
    from owac.simulation.rawvla_teacher_eval import AttemptLedger, ClosedLoopDriver

    closed = {"env": False, "media": False}

    class FakeWriter:
        def append_data(self, frame):
            pass

        def close(self):
            closed["media"] = True

    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: FakeWriter())

    class FakeEnv:
        def __init__(self):
            self.steps = 0

        def reset(self):
            return self._obs()

        def step(self, action):
            if self.steps == 12:  # 10 dummy + two executed policy actions
                raise KeyboardInterrupt("operator stop")
            self.steps += 1
            return self._obs(), 0.0, False, {}

        def _obs(self):
            return {
                "raw": {"agentview": np.zeros((256, 256, 3), dtype=np.uint8)},
                "robot_state": np.zeros(8),
                "native": {"agentview_image": np.zeros((256, 256, 3), dtype=np.uint8)},
            }

        def check_success(self):
            return False

        def close(self):
            closed["env"] = True

    class FakeController:
        def __init__(self):
            self.observations = 0

        def reset(self):
            pass

        def observe(self, observation, step):
            self.observations += 1

        def act(self, observation, state, noise, *, episode_id, policy_step):
            return np.zeros((5, 7), dtype=np.float32), {
                "normalized_actions": np.zeros((1, 10, 32), dtype=np.float32),
                "physical_actions": np.zeros((10, 7), dtype=np.float32),
                "full_seconds": 0.0,
                "frontend_seconds": 0.0,
            }

    states = np.zeros((2, 79), dtype=np.float64)
    state_hash = hashlib.sha256(states[0].tobytes()).hexdigest()
    catalog = {
        "validation": [
            {
                "id": "validation-interrupt",
                "split": "validation",
                "seed": 1,
                "array_index": 0,
                "state_sha256": state_hash,
            },
            {
                "id": "validation-after",
                "split": "validation",
                "seed": 2,
                "array_index": 1,
                "state_sha256": hashlib.sha256(states[1].tobytes()).hexdigest(),
            },
        ]
    }
    budget = PhaseBudget(tmp_path / "budget")
    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    driver = ClosedLoopDriver(
        catalog=catalog,
        budget=budget,
        ledger=ledger,
        max_steps=10,
        setup_wait_steps=10,
        media_dir=tmp_path / "media",
        result_dir=tmp_path / "results",
    )
    with budget.lease("interrupt-test", gpu=False):
        try:
            driver.run_role(
                role="learned_raw_teacher",
                split="validation",
                env_factory=lambda entry, all_states, role: FakeEnv(),
                controller=FakeController(),
                states_array=states,
            )
            raise AssertionError("KeyboardInterrupt was swallowed")
        except KeyboardInterrupt as exc:
            assert str(exc) == "operator stop"

    lines = (tmp_path / "attempts.jsonl").read_text().splitlines()
    assert len(lines) == 2
    begin = json.loads(lines[0])
    interrupted = json.loads(lines[1])
    assert begin["episode_id"] == "validation-interrupt"
    assert interrupted["status"] == "interrupted"
    assert interrupted["success"] is None
    assert interrupted["valid_control_evidence"] is False
    assert interrupted["interrupted"] is True
    result = json.loads(
        (tmp_path / "results" / "learned_raw_teacher_validation-interrupt.json").read_text()
    )
    assert result["status"] == "interrupted"
    assert result["success"] is None
    assert result["executed_actions_available"] == 2
    assert result["video_path"] is not None
    assert (tmp_path / "results" / "learned_raw_teacher_validation-interrupt.npz").is_file()
    assert closed["env"] is True
    assert closed["media"] is True
    assert not any("validation-after" in line for line in lines)
