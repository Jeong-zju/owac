"""Zero-update training integration tests (fixture_contract only)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from owac.backends.rawvla_features_n02 import build_stub_bridge
from owac.baselines.raw_stem_siglip import (
    HistoryContract,
    RawStemInput,
    RawStemSiglipStudent,
)
from owac.training.raw_distillation import (
    DistillationBatch,
    RawDistillationTrainer,
    RawTeacherCheckpoint,
    RawVLATeacherAdapter,
    ResumeState,
    TeacherPreparationUnavailable,
    TrainingAuthority,
    TrainingUnavailable,
    _frozen_hashes,
    capture_resume_state,
    register_raw_teacher_adapter,
    restore_resume_state,
    token_distillation_loss,
    write_prepared_teacher_bundle,
)
from tests.test_raw_stem_siglip import make_policy


@pytest.fixture(autouse=True)
def _forbid_real_optimizer_step(monkeypatch):
    """Fail, never skip, if a test reaches a real torch optimizer update."""

    def forbidden(*args, **kwargs):
        raise AssertionError("engineering tests forbid every torch optimizer.step")

    for name in dir(torch.optim):
        cls = getattr(torch.optim, name)
        if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
            monkeypatch.setattr(cls, "step", forbidden)


def _student_and_bridge():
    policy = make_policy(seed=3)
    student = RawStemSiglipStudent.from_fixture(
        policy, input_mode="rawvla_rgb3", fixture_id="train"
    )
    built = build_stub_bridge(
        width=6,
        token_dim=6,
        tokens_per_camera=4,
        channels=3,
        frame_pixels=2,
        action_horizon=10,
        cameras=("camera_0",),
        with_frame=False,
    )
    return student, built["bridge"]


def _batch(student, teacher_tokens=None):
    raw = RawStemInput(
        values=torch.rand(1, 1, 3, 8, 8),
        input_mode="rawvla_rgb3",
        camera_ids=("camera_0",),
        units="normalized_0_1",
    )
    if teacher_tokens is None:
        teacher_tokens = torch.randn(1, 4, 6)
    return DistillationBatch(
        raw=raw,
        teacher_tokens=teacher_tokens,
        lang_tokens=torch.tensor([[1, 2, 3]]),
        lang_masks=torch.ones(1, 3, dtype=torch.bool),
        state=torch.zeros(1, 32),
        actions=torch.zeros(1, 10, 32),
        future_mask=torch.ones(1, 10, dtype=torch.bool),
        noise=torch.zeros(1, 10, 32),
        time=torch.tensor([0.5]),
    )


def test_token_loss_denominator_and_cosine_no_magnitude_renorm():
    student = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [9.0, 9.0]]])
    teacher = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    loss = token_distillation_loss(student, teacher, mask)
    assert loss.valid_tokens == 2
    assert loss.width == 2
    assert float(loss.mse) == 0.0
    assert float(loss.cosine) == 0.0
    assert float(loss.total) == 0.0
    # Missing mask rows are excluded, not treated as zeros.
    loss2 = token_distillation_loss(student, teacher, torch.tensor([[True, False, False]]))
    assert loss2.valid_tokens == 1
    with pytest.raises(ValueError, match="valid token"):
        token_distillation_loss(student, teacher, torch.zeros(1, 3, dtype=torch.bool))


def test_native_action_gradient_reaches_stem_and_frozen_backend_is_stable():
    student, bridge = _student_and_bridge()
    student.configure_trainable_scope("stem_only")
    before = {
        name: tensor.detach().clone()
        for name, tensor in bridge.base.named_parameters()
        if "vision_tower" in name or "multi_modal_projector" in name
    }
    batch = _batch(student)
    trainer = RawDistillationTrainer(
        student=student,
        bridge=bridge,
        config={"stem_lr": 1e-4, "weight_decay": 0.01, "warmup_updates": 100},
        history=HistoryContract(),
        teacher_ready=True,
        stage0_parity_ready=True,
        source_split_ready=True,
        resource_budget_ready=True,
        phase1_diagnosis={"passed": True},
        matched_cd_declaration={"declared": True},
    )
    out = trainer.diagnostic_forward(batch, include_action=True, backward=True)
    assert out["optimizer_updates"] == 0
    assert out["action_loss"] is not None and float(out["action_loss"]) >= 0
    assert out["frozen_stable"]
    assert student.stem.patch_embedding.weight.grad is not None
    assert student.stem.patch_embedding.weight.grad.abs().sum() > 0
    for name, value in bridge.base.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            assert value.grad is None
            assert torch.equal(value.detach(), before[name])
    with pytest.raises(TrainingUnavailable, match="optimizer.step"):
        trainer.forbid_update()


def test_phase_gate_blocks_native_action_before_diagnosis():
    student, bridge = _student_and_bridge()
    trainer = RawDistillationTrainer(
        student=student,
        bridge=bridge,
        config={},
        history=HistoryContract(),
        teacher_ready=True,
        stage0_parity_ready=True,
        source_split_ready=True,
        resource_budget_ready=True,
    )
    with pytest.raises(TrainingUnavailable, match="phase-1"):
        trainer.diagnostic_forward(_batch(student), include_action=True)


def test_resume_roundtrip_without_optimizer_step(tmp_path):
    student, bridge = _student_and_bridge()
    trainer = RawDistillationTrainer(
        student=student,
        bridge=bridge,
        config={"stem_lr": 1e-4, "weight_decay": 0.01, "warmup_updates": 100},
        history=HistoryContract(),
    )
    optimizer, scheduler = trainer.optimizer_and_scheduler()
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    state = capture_resume_state(
        optimizer=optimizer,
        scheduler=scheduler,
        sampler_permutation=(3, 1, 2),
        sampler_cursor=2,
        cumulative_updates=17,
        anchor_presentations=272,
        config_sha256="a" * 64,
        data_sha256="b" * 64,
        source_sha256="c" * 64,
    )
    path = state.save(tmp_path / "resume.pt")
    loaded = ResumeState.load(path)
    assert loaded.sampler_permutation == (3, 1, 2)
    assert loaded.cumulative_updates == 17
    assert loaded.anchor_presentations == 272
    optimizer2, scheduler2 = trainer.optimizer_and_scheduler()
    restore_resume_state(loaded, optimizer=optimizer2, scheduler=scheduler2)
    assert optimizer2.state_dict()["param_groups"][0]["lr"] == pytest.approx(1e-6)
    assert scheduler2.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert loaded.config_sha256 == "a" * 64


def test_teacher_adapter_fixture_receipt_and_real_gate(tmp_path):
    policy = make_policy(seed=4)

    # Add the original policy preprocessing boundary to the tiny fixture.
    def image_to_tensor(image):
        return (
            torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(8, 8), mode="bilinear", align_corners=False
            ).squeeze(0)
            * 2
            - 1
        )

    policy._image_to_tensor = image_to_tensor

    class Frontend(torch.nn.Module):
        def forward(self, raw):
            rgb = raw[:, :, :3].clamp(0, 1)
            return rgb

    register_raw_teacher_adapter("fixture-rawvla", Frontend)
    checkpoint = RawTeacherCheckpoint(
        frontend_checkpoint_sha256="0" * 64,
        frontend_config_sha256="1" * 64,
        code_revision="fixture",
        adapter_name="fixture-rawvla",
        source="fixture rawvla frontend",
        verified=False,
    )
    history = HistoryContract.from_frames((0, 1), (0.0, 0.1), stride=1)
    adapter = RawVLATeacherAdapter(
        policy=policy,
        frontend=Frontend(),
        checkpoint=checkpoint,
        history=history,
        evidence_level="fixture_contract",
        checkpoint_state=None,
    )
    raw = torch.rand(1, 2, 4, 8, 8)
    prepared = adapter.prepare(raw)
    assert prepared.tokens.shape == (1, 2, 4, 6)
    assert prepared.receipt.real_ready is False
    assert prepared.receipt.raw_sha256
    hashes = write_prepared_teacher_bundle(tmp_path / "prepared", prepared)
    assert set(hashes) == {"raw", "prepared_rgb", "teacher_tokens", "receipt"}
    assert (tmp_path / "prepared" / "receipt.json").is_file()
    with pytest.raises(TeacherPreparationUnavailable, match="learned RAW"):
        RawVLATeacherAdapter(
            policy=policy,
            frontend=Frontend(),
            checkpoint=RawTeacherCheckpoint(
                frontend_checkpoint_sha256="0" * 64,
                frontend_config_sha256="1" * 64,
                code_revision="fixture",
                adapter_name="fixture-rawvla",
                source="official pi05 only",
                verified=True,
                state_path=None,
            ),
            history=HistoryContract(),
            evidence_level="real_policy",
        )


def test_guarded_update_branch_and_dry_run_use_nonupdating_spies():
    student, bridge = _student_and_bridge()
    student.configure_trainable_scope("stem_only")
    trainer = RawDistillationTrainer(
        student=student,
        bridge=bridge,
        config={
            "authorization": {"optimizer_updates": True},
            "resource_limits": {"gpu_hours": 1.0, "training_disk_bytes": 100000000},
            "training": {
                "effective_batch": 1,
                "max_cumulative_updates_per_arm": 5000,
                "max_anchor_presentations_per_arm": 80000,
                "gradient_clip": 1.0,
            },
            "stem_lr": 1e-4,
        },
        history=HistoryContract(),
        authority=TrainingAuthority(
            path="fixture-authority",
            sha256="a" * 64,
            authorized=True,
            max_updates=5,
            max_anchors=10,
        ),
        teacher_ready=True,
        stage0_parity_ready=True,
        source_split_ready=True,
        resource_budget_ready=True,
        phase1_diagnosis={"passed": True},
        matched_cd_declaration={"declared": True},
    )

    class SpyOptimizer:
        def __init__(self):
            self.steps = 0
            self.zeroed = 0

        def zero_grad(self, *args, **kwargs):
            self.zeroed += 1

        def step(self):
            self.steps += 1

    class SpyScheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    optimizer, scheduler = SpyOptimizer(), SpyScheduler()
    report = trainer.run_accumulated_update(
        [_batch(student)],
        optimizer=optimizer,
        scheduler=scheduler,
        include_action=True,
        arm="D",
    )
    assert report["actual_updates"] == 1
    assert optimizer.steps == 1 and scheduler.steps == 1
    assert trainer.cumulative_updates == 1
    assert trainer.anchor_presentations == 1
    # Dry-run restores counters/RNG and reports proposed work separately.
    before = (trainer.cumulative_updates, trainer.anchor_presentations)
    dry = trainer.dry_run_accumulation([_batch(student)], include_action=True)
    assert dry["actual_updates"] == 0 and dry["proposed_updates"] == 1
    assert (trainer.cumulative_updates, trainer.anchor_presentations) == before
    # A later scheduled audit must compare to the INITIAL frozen identity,
    # not merely hash an already-corrupted backend immediately before/after.
    trainer.config["training"]["frozen_audit_interval"] = 2
    with torch.no_grad():
        next(bridge.base.parameters()).add_(0.125)  # fixture corruption, not training
    with pytest.raises(TrainingUnavailable, match="initial audit"):
        trainer.run_accumulated_update(
            [_batch(student)],
            optimizer=optimizer,
            scheduler=scheduler,
            arm="D",
            include_action=True,
        )
    assert optimizer.steps == 1


def test_update_branch_is_blocked_by_missing_authority_before_step():
    student, bridge = _student_and_bridge()
    trainer = RawDistillationTrainer(
        student=student,
        bridge=bridge,
        config={"authorization": {"optimizer_updates": False}, "training": {"effective_batch": 1}},
        history=HistoryContract(),
        teacher_ready=True,
        stage0_parity_ready=True,
        source_split_ready=True,
        resource_budget_ready=True,
    )

    class Spy:
        steps = 0

        def zero_grad(self, *args, **kwargs):
            return None

        def step(self):
            self.steps += 1

    spy = Spy()
    with pytest.raises(TrainingUnavailable, match="authority"):
        trainer.run_accumulated_update([_batch(student)], optimizer=spy, scheduler=spy)
    assert spy.steps == 0


def test_author_output_state_carry_strict_load_and_reset(tmp_path):
    from types import SimpleNamespace

    class Frontend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, raw, state=None, theta_prev=None):
            self.calls += 1
            return SimpleNamespace(
                rgb=raw[:, -1, :3],
                state=torch.full((1, 8), float(self.calls)),
                theta={"call": self.calls},
            )

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def _image_to_tensor(self, image):
            return image * 2 - 1

    state_path = tmp_path / "frontend.pt"
    torch.save({"w": torch.ones(1)}, state_path)
    register_raw_teacher_adapter("fixture-author-output", Frontend)
    checkpoint = RawTeacherCheckpoint(
        frontend_checkpoint_sha256=__import__("hashlib")
        .sha256(state_path.read_bytes())
        .hexdigest(),
        frontend_config_sha256="1" * 64,
        code_revision="fixture-rawvla-output",
        adapter_name="fixture-author-output",
        source="fixture RawVLA output",
        verified=True,
        state_path=str(state_path),
    )
    history = HistoryContract.from_frames((0, 1), (0.0, 0.1), stride=1)
    frontend = Frontend()
    adapter = RawVLATeacherAdapter(
        policy=Policy(),
        frontend=frontend,
        checkpoint=checkpoint,
        history=history,
        evidence_level="fixture_contract",
        state_path=state_path,
    )
    assert torch.equal(frontend.w.detach(), torch.ones(1))
    first = adapter.prepare(torch.ones(1, 2, 3, 8, 8))
    assert first.prepared_rgb.shape == (1, 3, 8, 8)
    assert first.receipt.state_before_sha256 != first.receipt.state_after_sha256
    assert first.receipt.causal_prefix_sha256
    second = adapter.prepare(torch.ones(1, 2, 3, 8, 8))
    assert second.receipt.state_before_sha256 == first.receipt.state_after_sha256
    adapter.reset()
    reset = adapter.prepare(torch.ones(1, 2, 3, 8, 8))
    assert reset.receipt.state_before_sha256 != second.receipt.state_after_sha256


def test_frozen_stability_hash_covers_language_and_action_parameters():
    _, bridge = _student_and_bridge()
    before = _frozen_hashes(bridge)
    assert any(".language_model." in name for name in before)
    assert any(".action_head." in name for name in before)
    parameter = next(bridge.base.vlm.language_model.parameters())
    with torch.no_grad():
        parameter.add_(1.0)
    after = _frozen_hashes(bridge)
    assert before != after


def test_cost_profiler_zero_update_mechanics_and_frozen_hashes():
    from owac.training.raw_distillation_profile import profile_train_step

    student, bridge = _student_and_bridge()
    trainer = RawDistillationTrainer(
        student=student,
        bridge=bridge,
        config={},
        history=HistoryContract(),
        teacher_ready=True,
        stage0_parity_ready=True,
        source_split_ready=True,
        resource_budget_ready=True,
    )
    profile = profile_train_step(
        trainer,
        [_batch(student)],
        evidence_level="fixture_contract",
        update_cap=5000,
        effective_batch=16,
        include_action=True,
    )
    assert profile.frozen_stable
    assert profile.context["include_action"] is True
    assert profile.context["backend_parameters"] > 0
    assert profile.gradient_norms
    assert profile.forward_backward_ms
    assert profile.forward_backward_ms[0] >= 0
    assert profile.peak_allocated_bytes is None
    assert "fixture_contract" in " ".join(profile.limitations)
    assert profile.assumptions["update_cap"] == 5000


def test_anchor_batch_iterator_builds_real_distillation_batch_fixture():
    from types import SimpleNamespace

    from owac.raw.distillation_pairs import RawFrameData
    from owac.training.raw_distillation import AnchorBatchIterator

    sample = SimpleNamespace(
        frame_key="s/base/0/rawvla_rgb3",
        source_id="s",
        camera_id="camera_0",
        variant="rawvla_rgb3",
        raw=RawFrameData(torch.ones(3, 8, 8), "normalized_0_1", "", (0, 0)),
        state=torch.arange(8).float(),
        actions=torch.zeros(10, 7),
        future_mask=torch.ones(10, dtype=torch.bool),
        language="pick",
    )
    teacher = SimpleNamespace(
        prepare=lambda raw, history=None: SimpleNamespace(
            tokens=torch.zeros(1, 4, 6), prepared_rgb=torch.zeros(1, 3, 8, 8)
        )
    )

    class Tokenizer:
        def tokenize(self, text, state=None):
            return [1, 2], [True, True]

    dataset = SimpleNamespace(
        manifest=SimpleNamespace(source_split=lambda source_id: "train"),
        sample=lambda key: sample,
    )
    context = SimpleNamespace(
        dataset=dataset,
        tokenizer=Tokenizer(),
        bridge=SimpleNamespace(
            base=SimpleNamespace(max_state_dim=32, tokenizer=Tokenizer()),
            protocol=SimpleNamespace(name="libero"),
        ),
        teacher_adapter=teacher,
        history=HistoryContract(),
        camera_anchors=lambda: (
            SimpleNamespace(
                source_id="s",
                frame_id=0,
                timestamp=0.0,
                variant="rawvla_rgb3",
                frame_keys=("s/base/0/rawvla_rgb3",),
                camera_order=("camera_0",),
            ),
        ),
    )
    batch = next(iter(AnchorBatchIterator(context, split="train")))
    assert batch.teacher_tokens.shape == (1, 4, 6)
    assert batch.raw.values.shape == (1, 1, 3, 8, 8)
    assert batch.lang_tokens.shape == (1, 2)
    assert batch.state.shape == (1, 32)
    assert batch.actions.shape == (1, 10, 32)


def test_future_loop_control_flow_checkpoints_without_torch_optimizer(tmp_path):
    student, bridge = _student_and_bridge()
    trainer = RawDistillationTrainer(
        student=student,
        bridge=bridge,
        config={
            "authorization": {"optimizer_updates": True},
            "resource_limits": {"gpu_hours": 1.0, "training_disk_bytes": 100000000},
            "training": {
                "effective_batch": 1,
                "max_cumulative_updates_per_arm": 5000,
                "max_anchor_presentations_per_arm": 80000,
                "gradient_clip": 1.0,
                "loss": {"lambda_f": 1.0, "lambda_a_D": 1.0},
            },
        },
        history=HistoryContract(),
        authority=TrainingAuthority(
            path="fixture", sha256="a" * 64, authorized=True, max_updates=5, max_anchors=10
        ),
        teacher_ready=True,
        stage0_parity_ready=True,
        source_split_ready=True,
        resource_budget_ready=True,
    )
    batch = _batch(student)

    class Iterator:
        def __init__(self):
            self.cursor = 0
            self.epoch = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.cursor += 1
            return batch

        def __len__(self):
            return 1

        def reset_epoch(self):
            self.epoch += 1
            self.cursor = 0

        def state_dict(self):
            return {
                "seed": 101,
                "epoch": self.epoch,
                "cursor": self.cursor,
                "arm": "C",
                "split": "train",
                "permutation": [["s/base/0"]],
                "record_keys": ["s/base/0"],
                "manifest_sha256": "m" * 64,
                "source_sha256": {"s": "s" * 64},
            }

        def load_state_dict(self, state):
            self.cursor = int(state["cursor"])

    class SpyOptimizer:
        def __init__(self):
            self.steps = 0

        def zero_grad(self, *args, **kwargs):
            return None

        def step(self):
            self.steps += 1

        def state_dict(self):
            return {"state": {}, "param_groups": []}

        def load_state_dict(self, state):
            return None

    class SpyScheduler:
        def step(self):
            return None

        def state_dict(self):
            return {"last_epoch": 0}

        def load_state_dict(self, state):
            return None

    optimizer, scheduler = SpyOptimizer(), SpyScheduler()
    from owac.training.raw_distillation import run_distillation_loop

    report = run_distillation_loop(
        trainer=trainer,
        iterator=Iterator(),
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=tmp_path,
        arm="C",
        max_updates=1,
        checkpoint_updates=(0, 1),
        validation=lambda: {
            "step": trainer.cumulative_updates,
            "first5_mae7": 0.1 + trainer.cumulative_updates,
        },
    )
    assert report["best_validation"]["cumulative_update"] == 0
    best_payload = torch.load(tmp_path / "checkpoint_best.pt", weights_only=True)
    assert best_payload["cumulative_updates"] == 0
    assert report["actual_updates"] == 1
    assert report["cumulative_updates"] == 1
    assert optimizer.steps == 1
    assert (tmp_path / "checkpoint_0.pt").is_file()
    assert (tmp_path / "checkpoint_1.pt").is_file()
    assert trainer.cumulative_updates == 1
    from owac.training.raw_distillation import load_training_checkpoint

    resumed = load_training_checkpoint(
        tmp_path / "checkpoint_1.pt",
        trainer=trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        iterator=Iterator(),
        arm="C",
        config=trainer.config,
    )
    assert resumed["cumulative_updates"] == 1


def test_future_loop_cumulative_resume_499_to_500_and_max_zero(tmp_path):

    from owac.training.raw_distillation import run_distillation_loop

    student, bridge = _student_and_bridge()
    trainer = RawDistillationTrainer(
        student=student,
        bridge=bridge,
        config={
            "authorization": {"optimizer_updates": True},
            "resource_limits": {"gpu_hours": 1.0, "training_disk_bytes": 100000000},
            "training": {
                "effective_batch": 1,
                "max_cumulative_updates_per_arm": 5000,
                "max_anchor_presentations_per_arm": 80000,
                "gradient_clip": 1.0,
                "loss": {"lambda_f": 1.0},
            },
        },
        history=HistoryContract(),
        authority=TrainingAuthority(
            path="fixture", sha256="a" * 64, authorized=True, max_updates=500, max_anchors=80000
        ),
        teacher_ready=True,
        stage0_parity_ready=True,
        source_split_ready=True,
        resource_budget_ready=True,
    )
    trainer.cumulative_updates = 499
    optimizer, scheduler = trainer.optimizer_and_scheduler()
    batch = _batch(student)

    class Iterator:
        def __init__(self):
            self.cursor = 0
            self.epoch = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.cursor += 1
            return batch

        def __len__(self):
            return 1

        def reset_epoch(self):
            self.epoch += 1
            self.cursor = 0

        def state_dict(self):
            return {
                "seed": 101,
                "epoch": self.epoch,
                "cursor": self.cursor,
                "arm": "C",
                "split": "train",
                "permutation": [["x"]],
                "record_keys": ["x"],
                "manifest_sha256": "m" * 64,
                "source_sha256": {"s": "s" * 64},
            }

        def load_state_dict(self, state):
            self.cursor = int(state["cursor"])

    class Spy:
        def zero_grad(self, *args, **kwargs):
            return None

        def step(self):
            raise AssertionError("torch optimizer must not be used")

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            return None

    # Use a non-torch spy for the actual step path.
    class SafeSpy(Spy):
        def step(self):
            return None

    opt, sched = SafeSpy(), SafeSpy()
    # max_updates=0 must not iterate or step.
    zero = run_distillation_loop(
        trainer=trainer,
        iterator=Iterator(),
        optimizer=opt,
        scheduler=sched,
        checkpoint_dir=tmp_path / "zero",
        arm="C",
        max_updates=0,
        checkpoint_updates=(0,),
    )
    assert zero["actual_updates"] == 0
    assert trainer.cumulative_updates == 499
    trainer.cumulative_updates = 499
    report = run_distillation_loop(
        trainer=trainer,
        iterator=Iterator(),
        optimizer=opt,
        scheduler=sched,
        checkpoint_dir=tmp_path / "resume",
        arm="C",
        max_updates=500,
        checkpoint_updates=(0, 500),
    )
    assert report["actual_updates"] == 1
    assert trainer.cumulative_updates == 500
    assert (tmp_path / "resume" / "checkpoint_500.pt").is_file()


def test_accumulated_valid_token_reduction_matches_unsplit_batch_without_updates():
    """Unequal microbatches and validity counts must preserve the full loss gradient."""
    import copy
    from dataclasses import replace

    student, bridge = _student_and_bridge()
    second = copy.deepcopy(student)
    item = _batch(student)
    values = torch.cat([item.raw.values, item.raw.values * 0.7, item.raw.values * 0.3])
    valid = torch.ones(3, 1, 8, 8, dtype=torch.bool)
    valid[2, :, :4, :4] = False
    target = item.teacher_tokens.expand(3, -1, -1).clone()

    def batch(lo, hi):
        return replace(
            item,
            raw=replace(item.raw, values=values[lo:hi], valid_mask=valid[lo:hi]),
            teacher_tokens=target[lo:hi],
            actions=item.actions.expand(hi - lo, -1, -1),
            future_mask=item.future_mask.expand(hi - lo, -1),
        )

    config = {
        "authorization": {"optimizer_updates": True},
        "training": {
            "effective_batch": 3,
            "gradient_clip": 0.0,
        },
    }
    authority = TrainingAuthority(
        path="fixture", sha256="a" * 64, authorized=True, max_updates=1, max_anchors=3
    )

    def driver(model, batches):
        trainer = RawDistillationTrainer(
            student=model,
            bridge=bridge,
            config=config,
            history=HistoryContract(),
            authority=authority,
            teacher_ready=True,
            stage0_parity_ready=True,
            source_split_ready=True,
            resource_budget_ready=True,
        )
        before = {k: v.clone() for k, v in model.state_dict().items()}

        class NoUpdate:
            def zero_grad(self, **kwargs):
                model.zero_grad(**kwargs)

            def step(self):
                pass

        report = trainer.run_accumulated_update(batches, optimizer=NoUpdate(), scheduler=NoUpdate())
        assert report["valid_token_denominator"] == 11
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}

    split = driver(student, [batch(0, 2), batch(2, 3)])
    unsplit = driver(second, [batch(0, 3)])
    for key in split:
        torch.testing.assert_close(split[key], unsplit[key], rtol=2e-5, atol=2e-6)


def test_b_arm_uses_actual_prepared_teacher_time_and_camera_axes():
    from types import SimpleNamespace as NS

    from owac.raw.distillation_pairs import RawFrameData
    from owac.training.raw_distillation import AnchorBatchIterator

    cameras = ("base_0_rgb", "left_wrist_0_rgb")
    samples = {
        key: NS(
            camera_id=key,
            variant="rawvla_rgb3",
            state=torch.zeros(8),
            actions=torch.zeros(10, 7),
            future_mask=torch.ones(10, dtype=torch.bool),
            language="pick",
            raw=RawFrameData(torch.ones(3, 8, 8), "normalized_0_1", "", (0, 0)),
        )
        for key in cameras
    }
    teacher = NS(
        prepare=lambda raw, history: NS(
            tokens=torch.ones(1, 1, 4, 6), prepared_rgb=torch.ones(1, 1, 3, 8, 8)
        ),
        reset=lambda: None,
    )
    tokenizer = NS(tokenize=lambda prompt, state: ([1, 2], [True, True]))
    context = NS(
        dataset=NS(manifest=NS(source_split=lambda _: "train"), sample=samples.__getitem__),
        teacher_adapter=teacher,
        history=HistoryContract(),
        bridge=NS(base=NS(max_state_dim=32, tokenizer=tokenizer), protocol=NS(name="libero")),
        camera_anchors=lambda: (
            NS(source_id="s", frame_keys=cameras, camera_order=cameras, variant="rawvla_rgb3"),
        ),
    )
    batch = next(iter(AnchorBatchIterator(context, split="train", arm="B")))
    assert batch.raw.input_mode == "rgb3_control"
    assert batch.raw.values.shape == (1, 2, 3, 8, 8)
    assert batch.teacher_tokens.shape == (1, 8, 6)
    assert batch.noise is None and batch.time is None


def test_recurrent_dataclass_state_hash_uses_full_tensor_bytes_and_detaches():
    from dataclasses import dataclass

    from owac.training.raw_distillation import _clone_state, _state_hash

    @dataclass
    class Theta:
        tone: torch.Tensor

    first = Theta(torch.tensor([0.12345671], requires_grad=True))
    second = Theta(torch.tensor([0.12345679], requires_grad=True))
    assert str(first) == str(second)  # A repr-based cache key would collide.
    assert _state_hash(None, first) != _state_hash(None, second)
    clone = _clone_state(first)
    assert not clone.tone.requires_grad
    assert clone.tone.data_ptr() != first.tone.data_ptr()
    torch.testing.assert_close(clone.tone, first.tone)
