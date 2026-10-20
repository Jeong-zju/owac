"""Fixture tests for the native author-teacher training plumbing.

No real policy/checkpoint or simulator is loaded. One toy optimizer step is
allowed; all production training remains authority-gated and unexecuted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from owac.training.rawvla_teacher import (
    TeacherAssets,
    TeacherAuthority,
    TeacherTrainer,
    TeacherTrainingBlocked,
    build_author_teacher,
    native_masked_action_loss,
)


def _authority(path: Path) -> Path:
    payload = {
        "schema": "owac.raw-teacher.authorization.v1",
        "authorization": {
            "teacher_optimizer_updates": True,
            "student_training": False,
            "physical_actions": False,
        },
        "limits": {
            "local_gpu_hours": 12,
            "additional_disk_bytes": 50 * 1024**3,
            "demonstrations_replayed_max": 50,
        },
        "training": {
            "max_cumulative_updates": 2000,
            "max_anchor_presentations": 16000,
            "effective_batch": 8,
            "microbatch": 1,
            "gradient_accumulation": 8,
            "lambda_native_action": 0.15,
            "lambda_brightness": 0.01,
            "lambda_chroma": 0.001,
        },
    }
    path.write_text(json.dumps(payload))
    return path


def _window(frames: int = 8):
    raw = np.zeros((6, 6, 2, 4, 4, 3), dtype=np.uint8)
    return SimpleNamespace(
        raw_uint8=raw,
        state=np.arange(8, dtype=np.float32),
        padded_actions=np.zeros((10, 32), dtype=np.float32),
        action_mask=np.array([True] * 6 + [False] * 4),
        source_demo_id="s",
        split="train",
    )


class _ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, examples, **kwargs):
        actions = torch.as_tensor(examples[0]["action"], dtype=torch.float32)[None]
        aux = self.linear(torch.ones(1, 1)).square().mean()
        return {
            "u_t": actions,
            "v_t": actions,
            "exposure_prior_loss": aux,
            "chroma_prior_loss": aux,
        }


class _ToyData:
    def __init__(self):
        self.manifest = SimpleNamespace(as_dict=lambda: {"data": "toy"})
        self._windows = [_window() for _ in range(2)]

    def iter_anchors(self, split):
        return iter([SimpleNamespace(source_demo_id=f"{split}_0", split=split, current_frame=0)])

    def window(self, anchor):
        return self._windows[0]

    def sampler(self, *, split, seed=101):
        return _ToySampler(self._windows)


class _ToySampler:
    def __init__(self, windows):
        self.windows = windows
        self.i = 0
        self.epoch = 0
        self.epoch = 0
        self.block_cursor = 0
        self.anchor_cursor = 0

    def next_anchor(self):
        value = SimpleNamespace(source_demo_id="toy", split="train", current_frame=0)
        self.i += 1
        return value

    def next(self):
        return self.windows[self.i % len(self.windows)]

    def state_dict(self):
        return {
            "i": self.i,
            "epoch": self.epoch,
            "block_cursor": self.block_cursor,
            "anchor_cursor": self.anchor_cursor,
        }

    def load_state_dict(self, state):
        self.i = int(state["i"])
        self.epoch = int(state.get("epoch", 0))
        self.block_cursor = int(state.get("block_cursor", 0))
        self.anchor_cursor = int(state.get("anchor_cursor", 0))


def test_authority_and_native_masked_loss():
    authority = TeacherAuthority.from_file(_authority(Path("/tmp/toy-authority.json")))
    assert authority.max_updates == 2000
    assert authority.lambda_action == 0.15
    u = torch.zeros(1, 10, 32)
    v = torch.zeros(1, 10, 32)
    v[0, :6, :7] = 1.0
    result = native_masked_action_loss(
        {"u_t": u, "v_t": v}, torch.tensor([[True] * 6 + [False] * 4])
    )
    assert result["valid_actions"] == 6
    assert result["real7_mse"] > 0
    assert result["padded32_mse"] > 0


def test_authority_rejects_oversized_budget(tmp_path):
    path = _authority(tmp_path / "bad.json")
    payload = json.loads(path.read_text())
    payload["training"]["max_cumulative_updates"] = 2001
    path.write_text(json.dumps(payload))
    with pytest.raises(TeacherTrainingBlocked):
        TeacherAuthority.from_file(path)


def test_build_teacher_blocks_missing_assets_without_loading(tmp_path):
    assets = TeacherAssets(
        author_root=tmp_path / "missing-author",
        starvla_root=tmp_path / "missing-starvla",
        checkpoint=tmp_path / "missing.safetensors",
        tokenizer=tmp_path / "missing.model",
        statistics=tmp_path / "missing.json",
        config_paths=(tmp_path / "missing.yaml",),
    )
    with pytest.raises(TeacherTrainingBlocked):
        build_author_teacher(assets)


def test_toy_teacher_trainer_one_update_and_checkpoint(tmp_path):
    authority = TeacherAuthority.from_file(_authority(tmp_path / "authority.json"))
    policy = _ToyPolicy()
    teacher = SimpleNamespace(
        policy=policy,
        frontend=policy.linear,
        config={"toy": True},
        backend_hashes={
            name: hashlib.sha256(
                parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for name, parameter in policy.named_parameters()
            if not name.startswith("linear.")
        },
        frontend_param_names=("weight", "bias"),
    )
    trainer = TeacherTrainer(
        teacher=teacher,
        data=_ToyData(),
        authority=authority,
        run_root=tmp_path / "run",
    )
    report = trainer.train(max_updates=1, validation=lambda: {"first5_mae7": 0.0})
    assert report["actual_updates"] == 1
    assert (tmp_path / "run" / "checkpoint_0000.pt").is_file()
    assert (tmp_path / "run" / "checkpoint_0001.pt").is_file()
    checkpoint = torch.load(tmp_path / "run" / "checkpoint_0001.pt", weights_only=True)
    assert checkpoint["schema"] == "owac.rawvla-teacher.checkpoint.v1"
    assert checkpoint["cumulative_updates"] == 1
    assert checkpoint["optimizer"] is not None


def test_preflight_fails_when_frontend_receives_no_gradient(tmp_path):
    authority = TeacherAuthority.from_file(_authority(tmp_path / "authority.json"))
    policy = _ToyPolicy()
    frontend = nn.Linear(1, 1)  # not used by the toy policy
    teacher = SimpleNamespace(
        policy=policy,
        frontend=frontend,
        config={"toy": True},
        backend_hashes={
            name: hashlib.sha256(
                parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for name, parameter in policy.named_parameters()
            if not name.startswith("linear.")
        },
        frontend_param_names=("weight", "bias"),
    )
    trainer = TeacherTrainer(
        teacher=teacher,
        data=_ToyData(),
        authority=authority,
        run_root=tmp_path / "run",
    )
    with pytest.raises(TeacherTrainingBlocked, match="no frontend gradients|zero"):
        trainer.preflight()


def test_unequal_future_lengths_match_one_masked_batch_gradient():
    weight = torch.tensor(0.7, requires_grad=True)
    masks = [torch.tensor([[True] + [False] * 9]), torch.ones(1, 10, dtype=torch.bool)]
    for scale, mask in zip((1.0, 2.0), masks, strict=True):
        prediction = (weight * scale).expand(1, 10, 32)
        result = native_masked_action_loss(
            {"u_t": torch.zeros_like(prediction), "v_t": prediction}, mask
        )
        (result["action_loss_sum"] / 11.0).backward()
    assert float(weight.grad) == pytest.approx(82.0 * 0.7 / 11.0, rel=1e-6)


def test_two_updates_equal_interrupted_adamw_resume_and_keep_best_zero(tmp_path):
    import copy

    authority = TeacherAuthority.from_file(_authority(tmp_path / "authority.json"))
    torch.manual_seed(307)
    initial = copy.deepcopy(_ToyPolicy().state_dict())

    def build(name):
        policy = _ToyPolicy()
        policy.load_state_dict(initial)
        teacher = SimpleNamespace(
            policy=policy,
            frontend=policy.linear,
            config={"toy": True},
            backend_hashes={
                "frozen": hashlib.sha256(
                    policy.frozen.detach().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
            },
            frontend_param_names=("weight", "bias"),
        )
        return TeacherTrainer(
            teacher=teacher, data=_ToyData(), authority=authority, run_root=tmp_path / name
        )

    uninterrupted = build("uninterrupted")
    uninterrupted.train(
        max_updates=2,
        validation=lambda: {"first5_mae7": 0.1 + uninterrupted.cumulative_updates},
    )
    interrupted = build("resumed")
    interrupted.train(
        max_updates=1,
        validation=lambda: {"first5_mae7": 0.1 + interrupted.cumulative_updates},
    )
    assert any(bool(state["exp_avg"].abs().sum()) for state in interrupted.optimizer.state.values())
    resumed = build("resumed")
    resumed.load_checkpoint(tmp_path / "resumed/checkpoint_resume.pt")
    resumed.train(
        max_updates=2, validation=lambda: {"first5_mae7": 0.1 + resumed.cumulative_updates}
    )
    for name, value in uninterrupted.teacher.frontend.state_dict().items():
        assert torch.equal(value, resumed.teacher.frontend.state_dict()[name])
    assert uninterrupted.scheduler.state_dict() == resumed.scheduler.state_dict()
    assert resumed.best_validation == {"first5_mae7": 0.1, "step": 0}
    best = torch.load(tmp_path / "resumed/checkpoint_best.pt", weights_only=True)
    assert best["cumulative_updates"] == 0
    for name, value in initial.items():
        if name.startswith("linear."):
            assert torch.equal(value, best["frontend"][name.removeprefix("linear.")])


def test_window_example_passes_native_state8_to_policy_helper():
    from types import SimpleNamespace

    from owac.training.rawvla_teacher import _window_example

    window = _window()
    window.normalized = lambda stats: {
        "state": np.arange(8, dtype=np.float32),
        "padded_actions": np.zeros((10, 32), dtype=np.float32),
    }
    stats = SimpleNamespace(
        state={"q01": np.zeros(8), "q99": np.ones(8)},
        action={"q01": np.zeros(7), "q99": np.ones(7)},
    )
    example = _window_example(window, stats=stats)
    assert example["state"].shape == (8,)
    assert example["action"].shape == (10, 32)


def test_effective_policy_contract_uses_instance_flag_not_class_default():
    from types import SimpleNamespace

    from owac.training.rawvla_teacher import effective_policy_contract

    framework = SimpleNamespace(
        discrete_state_input=False,
        tokenizer=SimpleNamespace(pad_state_value=-2.0),
        image_keys=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        image_resolution=[224, 224],
        action_dim=32,
        action_horizon=10,
        num_inference_steps=10,
        max_state_dim=32,
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(framework=framework), discrete_state_input=False
    )
    contract = effective_policy_contract(policy)
    assert contract["instance_discrete_state_input"] is False
    assert contract["effective_pad_state_value"] == 0.0
    assert contract["configured_pad_state_value"] == -2.0
    assert contract["tokenizer_state_mode"] == "none"
    policy.discrete_state_input = True
    true_contract = effective_policy_contract(policy)
    assert true_contract["effective_pad_state_value"] == -2.0
    assert true_contract["tokenizer_state_mode"] == "text_state"
