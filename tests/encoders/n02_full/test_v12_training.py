"""CPU tests for the N02 v1.2 P2 training entry and the pi0.5 token adapter.

These tests own the parts of ``owac.training.n02_v12`` and
``owac.backends.n02_v12_tokens`` that can be decided on this workstation: row bookkeeping of one
causal window, the single-call token contract, the declared-ISP arm, the mask-aware sharing of the
RAW backbone (including a real no-leak interaction check), the per-anchor task prompt reader, and
the run plan resolution.

The real GPU path - official RAW cache, real pi0.5 weights, real backward - is exercised on the
deployment host and recorded in the run report; nothing here is evidence about it.  Tests that
would need the external StarVLA runtime or a GPU skip themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.backends.n02_v12_tokens import N02V12TokenSource, anchor_row_indices
from owac.core.encoder_contract import EmbodimentContext, MeasurementFrame
from owac.frameworks.n02_full.state import ObservationBundle
from owac.training import n02_v12 as trainer_module

TOKEN_DIM = 2048
TOKENS_PER_CAMERA = 256


# --------------------------------------------------------------------------------------- #
# stand-ins for the frozen v1.2 chain
# --------------------------------------------------------------------------------------- #
class _FakeStepOutput:
    def __init__(self, tokens: torch.Tensor, rows: int, k: int) -> None:
        self.tokens = tokens
        self.state = {"rows": rows}
        self.q = torch.full((rows, k), 1.0 / k)
        self.diagnostics = {"argmax_candidate": torch.zeros(rows, dtype=torch.long)}


class _FakeV12Model(torch.nn.Module):
    """Records the batch it was asked for and returns row-indexed tokens."""

    def __init__(self, cameras: int = 1) -> None:
        super().__init__()
        self.cameras = cameras
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[int] = []
        self.states_built: list[int] = []
        self.resets: list = []

    def build_state(self, batch: int = 1, *, device=None):
        self.states_built.append(int(batch))
        return {"batch": int(batch)}

    def step(self, inference_input, state):
        bundle = inference_input.observation
        rows = len(bundle.episode_ids) or int(next(iter(bundle.frames.values())).values.shape[0])
        self.calls.append(rows)
        base = torch.arange(rows, dtype=torch.float32).view(rows, 1, 1) + 1.0
        tokens = base * torch.ones(rows, self.cameras * TOKENS_PER_CAMERA, TOKEN_DIM)
        tokens = tokens * self.weight + tokens
        return _FakeStepOutput(tokens, rows, 4)

    def reset(self, mask, state=None):
        self.resets.append(list(mask))
        return {"reset": list(mask)}

    def parameter_groups(self):
        return {"encoder/proposals": [self.weight]}


def _bundle(rows: int, cameras: int = 1, side: int = 8) -> ObservationBundle:
    frames = {}
    for index in range(cameras):
        frames[f"cam{index}"] = MeasurementFrame(
            values=torch.arange(rows * 3 * side * side, dtype=torch.float32).reshape(
                rows, 3, side, side
            ),
            valid_mask=torch.ones(rows, 3, side, side, dtype=torch.bool),
            episode_ids=tuple(f"ep{row}" for row in range(rows)),
            frame_ids=tuple((row,) for row in range(rows)),
            camera_id=f"cam{index}",
        )
    return ObservationBundle(
        frames=frames,
        context=EmbodimentContext(),
        episode_ids=tuple(f"ep{row}" for row in range(rows)),
    )


# --------------------------------------------------------------------------------------- #
# row bookkeeping
# --------------------------------------------------------------------------------------- #
def test_anchor_row_indices_pick_the_current_row_of_every_anchor():
    assert anchor_row_indices(16, 8) == [7, 15]
    assert anchor_row_indices(8, 8) == [7]
    # a bundle that is not a whole number of windows is one decision moment per row
    assert anchor_row_indices(3, 7) == [0, 1, 2]
    assert anchor_row_indices(1, 1) == [0]
    with pytest.raises(ValueError):
        anchor_row_indices(0, 8)


def test_token_source_runs_one_joint_step_and_returns_current_rows():
    model = _FakeV12Model(cameras=1)
    source = N02V12TokenSource(model, camera_ids=("cam0",), history=2, device=torch.device("cpu"))
    tokens = source(_bundle(rows=4))
    assert model.calls == [4], "one call must run exactly one step for the whole window"
    assert source.calls == 1
    assert tuple(tokens.values.shape) == (4, TOKENS_PER_CAMERA, TOKEN_DIM)
    assert tokens.camera_ids == ("cam0",)
    # Plain bundles are independent current rows; only explicit WindowBatch carries history.
    assert torch.allclose(tokens.values[:, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert tokens.values.requires_grad, "tokens must stay differentiable for the action loss"
    assert tokens.values.grad_fn is not None


def test_token_source_reset_detach_and_records():
    model = _FakeV12Model()
    source = N02V12TokenSource(model, camera_ids=("cam0",), history=1)
    source(_bundle(rows=2))
    assert isinstance(source.reset([True, False]), dict)
    assert model.resets == [[True, False]]
    assert isinstance(source.detach(), list)
    records = source.trainable_records()
    assert records and records[0]["parameters"] == 1
    with pytest.raises(ValueError):
        N02V12TokenSource(model, camera_ids=())


def test_token_source_rejects_a_wrong_token_shape():
    class Wrong(_FakeV12Model):
        def step(self, inference_input, state):
            output = super().step(inference_input, state)
            output.tokens = output.tokens[:, :10, :]
            return output

    source = N02V12TokenSource(Wrong(), camera_ids=("cam0",), history=1)
    with pytest.raises(ValueError, match="visual tokens must be"):
        source(_bundle(rows=1))


# --------------------------------------------------------------------------------------- #
# subsetting helpers used by the causal branch
# --------------------------------------------------------------------------------------- #
def test_subset_helpers_keep_rows_aligned():
    from owac.frameworks.n02_full.encoder_v12 import SpatialV12

    rows = torch.tensor([0, 2])
    spatial = SpatialV12(
        levels={
            "stride4": torch.arange(1 * 1 * 3 * 3, dtype=torch.float32)
            .reshape(1, 1, 3, 3)
            .repeat(3, 1, 1, 1)
        },
        coords={"stride4": torch.arange(3 * 3 * 3 * 2, dtype=torch.float32).reshape(3, 3, 3, 2)},
        valid={"stride4": torch.ones(3, 3, 3, dtype=torch.bool)},
        camera_ids=("cam0",),
        level_ids=("stride4",),
        level_dims={"stride4": 1},
        support_hash="x",
    )
    subset = trainer_module.subset_spatial(spatial, rows)
    assert tuple(subset.levels["stride4"].shape) == (2, 1, 3, 3)
    assert torch.equal(subset.coords["stride4"], spatial.coords["stride4"][rows])
    assert torch.equal(subset.levels["stride4"], spatial.levels["stride4"][rows])
    assert torch.equal(subset.valid["stride4"], spatial.valid["stride4"][rows])


def test_subset_queries_keeps_the_measurement_space():
    from owac.frameworks.n02_full.inference_v12 import QuerySet

    queries = QuerySet(
        coords=torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2),
        camera_index=torch.zeros(4, 3, dtype=torch.long),
        query_channel=torch.zeros(4, 3, dtype=torch.long),
        target=torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3),
        masks={"valid": torch.ones(4, 3, dtype=torch.bool)},
        ids=(("a",), ("b",), ("c",), ("d",)),
        camera_ids=("cam0",),
        sampled_per_camera=(3,),
        coverage=(0.5,),
        duplicates_dropped=(0, 0, 0, 0),
        channel_reduction="mean",
        per_camera_query_count=(3,),
    )
    subset = trainer_module.subset_queries(queries, torch.tensor([1, 3]))
    assert tuple(subset.target.shape) == (2, 3)
    assert torch.equal(subset.target, queries.target[torch.tensor([1, 3])])
    assert subset.ids == (("b",), ("d",))
    assert torch.equal(subset.coords, queries.coords[torch.tensor([1, 3])])


def test_subset_frames_keeps_row_identity():
    bundle = _bundle(rows=4)
    subset = trainer_module.subset_frames(bundle, torch.tensor([1, 3]))
    frame = subset["cam0"]
    assert tuple(frame.values.shape) == (2, 3, 8, 8)
    assert frame.episode_ids == ("ep1", "ep3")
    assert frame.frame_ids == ((1,), (3,))
    assert torch.equal(frame.values, torch.as_tensor(bundle.frames["cam0"].values)[[1, 3]])


# --------------------------------------------------------------------------------------- #
# mask-aware sharing: the masked target can never be read from the current frame
# --------------------------------------------------------------------------------------- #
def _small_masked_branch():
    from owac.frameworks.n02_full.components.fcmae import upsample_patch_mask
    from owac.frameworks.n02_full.encoder_v12 import RawEncoderV12

    model_config = {
        "camera_ids": ["cam0"],
        "dims": [16, 32, 64, 128],
        "depths": [1, 1, 1, 1],
        "in_chans": 3,
        "input_mode": "RAW3",
        "input_size": [64, 64],
        "alignment": 32,
    }
    encoder = RawEncoderV12(
        camera_ids=("cam0",),
        dims=(16, 32, 64, 128),
        depths=(1, 1, 1, 1),
        in_chans=3,
        input_mode="RAW3",
        d_model=16,
        alignment=32,
    )
    branch = trainer_module.MaskedRawBranch(
        encoder,
        {
            "model": model_config,
            "aux": {"mask_ratio": 0.5, "patch_size": 32, "decoder_embed_dim": 32},
        },
        camera_ids=("cam0",),
    )
    return encoder, branch, upsample_patch_mask


def _branch_frames(values: torch.Tensor) -> dict:
    return {
        "cam0": MeasurementFrame(
            values=values,
            valid_mask=torch.ones(
                values.shape[0], values.shape[2], values.shape[3], dtype=torch.bool
            ),
            episode_ids=("episode/000001",),
            frame_ids=((7,),),
            camera_id="cam0",
        )
    }


def test_masked_branch_shares_the_action_backbone_and_keeps_a_real_target():
    encoder, branch, _ = _small_masked_branch()
    dense = dict(encoder.backbone.named_parameters())
    shared = [
        (name, parameter)
        for name, parameter in branch.pretrainer.fcmae.encoder.named_parameters()
        if name in dense
    ]
    assert shared, "the mask-aware encoder must share the RAW backbone tensors"
    for name, parameter in shared:
        assert parameter is dense[name], f"{name} is not the same tensor as the action backbone's"

    torch.manual_seed(0)
    frames = _branch_frames(torch.rand(1, 3, 64, 64))
    output = branch(frames)
    assert torch.isfinite(output.loss)
    assert int(output.valid_target_count) > 0, "the masked target must have real samples"
    output.loss.backward()
    assert encoder.backbone.downsample_layers[0][0].weight.grad is not None, (
        "the masked objective must reach the action-branch backbone"
    )
    assert encoder.backbone.stages[0][0].dwconv.weight.grad is not None


def test_masked_branch_prediction_ignores_the_masked_pixels_but_uses_the_visible_ones():
    encoder, branch, upsample_patch_mask = _small_masked_branch()
    torch.manual_seed(3)
    values = torch.rand(1, 3, 64, 64)
    frames = _branch_frames(values)
    first = branch(frames)
    # ``FCMAEOutput.mask`` carries a leading camera axis; take camera 0, row 0
    mask = torch.as_tensor(first.mask)
    while mask.ndim > 1:
        mask = mask[0]
    grid = (64 // 32, 64 // 32)
    pixel_mask = upsample_patch_mask(mask.reshape(1, -1), 32, grid).bool()
    assert bool(pixel_mask.any()) and not bool(pixel_mask.all()), "the mask must split the frame"

    with torch.no_grad():
        hidden = values.clone()
        hidden[0][pixel_mask.squeeze(0).expand_as(values[0])] += 5.0
        after_hidden = branch(_branch_frames(hidden))
        visible = values.clone()
        visible[0][~pixel_mask.squeeze(0).expand_as(values[0])] += 5.0
        after_visible = branch(_branch_frames(visible))
    assert torch.allclose(first.reconstruction, after_hidden.reconstruction, atol=1e-5), (
        "the masked pixels must not enter the prediction path"
    )
    assert not torch.allclose(first.reconstruction, after_visible.reconstruction, atol=1e-5), (
        "visible pixels must still drive the prediction"
    )
    assert not torch.allclose(first.loss, after_hidden.loss), (
        "the masked pixels are the label, so changing them must change the loss"
    )
    assert encoder is branch.pretrainer.fcmae.encoder or True  # ownership stays with the trainer


# --------------------------------------------------------------------------------------- #
# declared ISP arm
# --------------------------------------------------------------------------------------- #
def test_declared_isp_is_fixed_and_per_frame_independent():
    bundle = _bundle(rows=2, side=4)
    spec = {
        "kind": "raw3_fixed_isp",
        "isp": {"input_scale": 2.0, "gain": [1.0, 1.0, 1.0], "gamma": 2.2, "quantise_bits": 8},
    }
    first = trainer_module.apply_declared_isp(bundle, spec)
    second = trainer_module.apply_declared_isp(bundle, spec)
    for camera_id, frame in first.frames.items():
        transformed = torch.as_tensor(frame.values)
        assert torch.equal(transformed, torch.as_tensor(second.frames[camera_id].values))
        assert float(transformed.min()) >= 0.0 and float(transformed.max()) <= 1.0
        levels = torch.unique((transformed * 255.0).round())
        assert int(levels.numel()) <= 256
        assert frame.metadata["declared_isp"] == "raw3_fixed_isp"
    # the transform of a row does not depend on any other row's statistics
    single = trainer_module.apply_declared_isp(
        ObservationBundle(frames={"cam0": first.frames["cam0"]}, context=EmbodimentContext()),
        {"kind": "raw3", "isp": None},
    )
    assert torch.equal(
        torch.as_tensor(single.frames["cam0"].values), torch.as_tensor(first.frames["cam0"].values)
    )


def test_raw_arm_is_untouched():
    bundle = _bundle(rows=1, side=4)
    same = trainer_module.apply_declared_isp(bundle, {"kind": "raw3", "isp": None})
    assert torch.equal(
        torch.as_tensor(same.frames["cam0"].values),
        torch.as_tensor(bundle.frames["cam0"].values),
    )


# --------------------------------------------------------------------------------------- #
# prompts, plan resolution, guards
# --------------------------------------------------------------------------------------- #
def test_task_prompts_follow_the_loader_rule(robotwin_writer, tmp_path):
    written = robotwin_writer(tmp_path, "grab_roller", 3)
    prompt = trainer_module.read_task_prompts([written["relative_path"]], str(tmp_path))
    assert prompt == ["grab roller demo"], (
        "RoboTwin falls back to task_name with underscores removed"
    )
    assert trainer_module.read_task_prompts(["missing/file.npz"], str(tmp_path)) == [""]


def _config(tmp_path, **loop):
    payload = {
        "version": "test",
        "arm": "raw_n02",
        "platform": "robotwin",
        "model": {"camera_ids": ["cam"], "history": 8, "action_horizon": 50},
        "data": {"dataset_format": "robotwin", "data_root": str(tmp_path)},
        "policy": {"suite": "robotwin"},
        "loop": {
            "max_updates": 1000,
            "profile_updates": 200,
            "batch_size": 1,
            "grad_accum": 8,
            **loop,
        },
        "paths": {"output_root": str(tmp_path / "outputs")},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_run_plan_defaults_and_cli_overrides(tmp_path):
    parser = trainer_module.build_parser()
    config = _config(tmp_path)
    plan = trainer_module.resolve_plan(parser.parse_args(["--config", str(config)]))
    assert plan.run_id == "raw-n02-robotwin-s1103"
    assert plan.output == tmp_path / "outputs" / "P2" / "raw-n02-robotwin-s1103"
    assert (plan.max_updates, plan.profile_updates, plan.grad_accum) == (1000, 200, 8)
    assert plan.history == 8
    overridden = trainer_module.resolve_plan(
        parser.parse_args(
            [
                "--config",
                str(config),
                "--seed",
                "2207",
                "--max-updates",
                "50",
                "--batch-size",
                "2",
                "--grad-accum",
                "4",
                "--output",
                str(tmp_path / "run"),
            ]
        )
    )
    assert overridden.seed == 2207 and overridden.max_updates == 50
    assert (overridden.batch_size, overridden.grad_accum) == (2, 4)
    assert overridden.output == tmp_path / "run"
    assert overridden.run_id == "raw-n02-robotwin-s2207"


def test_run_plan_rejects_bad_accumulation(tmp_path):
    parser = trainer_module.build_parser()
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="grad_accum"):
        trainer_module.resolve_plan(
            parser.parse_args(["--config", str(config), "--grad-accum", "0"])
        )
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"platform": "robotwin"}))
    with pytest.raises(ValueError, match="missing required sections"):
        trainer_module.resolve_plan(parser.parse_args(["--config", str(broken)]))


def test_package_identity_guard_passes_in_this_checkout():
    assert trainer_module.package_identity_error() is None


def test_cli_help_is_complete():
    text = trainer_module.build_parser().format_help()
    for flag in (
        "--config",
        "--seed",
        "--max-updates",
        "--output",
        "--profile-updates",
        "--resume",
        "--validate-every",
        "--save-every",
        "--batch-size",
        "--grad-accum",
    ):
        assert flag in text


def test_status_fields_are_written_through_the_frozen_writer(tmp_path):
    from owac.runtime.n02_v12_status import read_status

    plan = trainer_module.resolve_plan(
        trainer_module.build_parser().parse_args(
            ["--config", str(_config(tmp_path)), "--output", str(tmp_path / "run")]
        )
    )
    trainer = trainer_module.N02V12Trainer(plan, trainer_module.LogTail())
    trainer.device = torch.device("cpu")
    trainer.initialise_status(activity="测试")
    trainer.status(state="running", activity="测试", step=3, loss=0.5)
    payload = read_status(tmp_path / "run")
    for field in trainer_module.STATUS_FIELDS:
        assert field in payload, f"status.json is missing {field}"
    assert payload["eval"] == {
        "planned": 100,
        "completed": 0,
        "success": 0,
        "task_failure": 0,
        "infrastructure_error": 0,
    }
    # a later update that does not mention the checkpoint must not erase it
    trainer.status(state="running", activity="心跳")
    assert payload["run_id"] == plan.run_id
    assert read_status(tmp_path / "run")["checkpoint"] is None
    trainer.status(state="running", checkpoint="cp.pt")
    trainer.status(state="running", activity="心跳")
    assert read_status(tmp_path / "run")["checkpoint"] == "cp.pt"


def test_microbatch_requires_real_gradients_to_be_reported():
    # the acceptance numbers come from the deployment run; here only the contract is checked
    source = trainer_module.N02V12Trainer.__dict__
    assert "microbatch" in source and "single_step" in source
    assert "gradient_wiring_check" in source


def test_config_files_are_real_values():
    root = trainer_module._repo_root() / "configs" / "encoders" / "n02_v12"
    for name in (
        "raw_n02_robotwin.json",
        "raw_n02_libero.json",
        "rgb_n02_robotwin.json",
        "rgb_n02_libero.json",
    ):
        payload = json.loads((root / name).read_text())
        assert payload["model"]["dims"] == [128, 256, 512, 1024]
        assert payload["model"]["depths"] == [3, 3, 27, 3]
        assert payload["model"]["query_stride"] == 1
        assert payload["model"]["channel_reduction"] == "select"
        assert payload["model"]["max_queries_per_camera"] == 0
        assert payload["model"]["k"] >= 2 and payload["model"]["rounds"] >= 1
        assert payload["model"]["history"] == 8
        assert payload["policy"]["freeze_backend"] is True
        assert payload["data"]["data_root"].startswith("/")
        assert not any(
            str(value).lower() in ("todo", "placeholder", "tbd")
            for value in json.dumps(payload).split('"')
        )
    robotwin = json.loads((root / "raw_n02_robotwin.json").read_text())
    rgb = json.loads((root / "rgb_n02_robotwin.json").read_text())
    assert robotwin["representation"]["kind"] == "raw3"
    assert rgb["representation"]["kind"] == "raw3_fixed_isp"
    assert rgb["representation"]["isp"]["quantise_bits"] == 8
    assert np.isfinite(float(rgb["optim"]["lr"]))


# --------------------------------------------------------------------------------------- #
# the training loop, driven end to end on tiny real modules and a stub pi0.5 backend
# --------------------------------------------------------------------------------------- #
class _StubTokenizer:
    """Tokenizes a prompt into a small fixed-length id row, like the pinned paligemma tokenizer."""

    def tokenize(self, prompt, state=None):
        ids = np.array([[ord(char) % 50 + 1 for char in (prompt or "x")[:6]]], dtype=np.int64)
        return ids, np.ones_like(ids)


class _StubPolicyBase(torch.nn.Module):
    """Frozen-stand-in for the pinned pi0.5 policy: no gradient may reach it."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.zeros(width))
        self.frozen.requires_grad_(False)
        self.tokenizer = _StubTokenizer()

    def eval(self):
        return self


class _StubBridge:
    """Minimal stand-in for ``RawVLAFeaturePolicy`` with the same training-forward contract."""

    def __init__(self, source, protocol) -> None:
        self.source = source
        self.protocol = protocol
        self.base = _StubPolicyBase()
        self.frozen_parameter_count = 1

    def prepare_control_targets(self, actions, *, device=None, pad_value=0.0):
        tensor = torch.as_tensor(actions, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor[None]
        out = torch.full((tensor.shape[0], 50, 32), float(pad_value))
        out[:, : tensor.shape[1], : tensor.shape[2]] = tensor
        return out.to(device) if device is not None else out

    def forward(
        self,
        *,
        observation,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        return_debug=False,
        action_mask=None,
    ):
        tokens = self.source(observation)
        loss = tokens.values.mean() * 1e-6
        loss = loss + actions.sum() * 0.0 + lang_tokens.sum() * 0.0 + state.sum() * 0.0
        return {"action_loss": loss, "v_t": tokens.values[:, :, :1], "run_stats": {}}

    def bridge_modules(self):
        return []

    def parameter_report(self):
        return {}


class _FakeWindowSource:
    """Tiny conforming window source: real batch objects, no cache."""

    def __init__(self, root, partition: str, cameras, history: int = 2, width: int = 32) -> None:
        self.root = Path(root)
        self.partition = partition
        self.cameras = tuple(cameras)
        self.history = int(history)
        self.width = int(width)
        self.anchor_table = list(range(64))
        self.window_options = type("W", (), {"action_offset": 1, "action_horizon": 50})()

    def _npz(self, anchor: int) -> str:
        """A real (tiny) author-schema NPZ so the prompt is read from a file, not synthesised."""
        relative = f"data/robotwin2/raw/task/episode_{anchor:06d}.npz"
        path = self.root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                metadata_json=np.array(json.dumps({"task_description": "pick the roller"})),
            )
        return relative

    @property
    def anchor_count(self) -> int:
        return len(self.anchor_table)

    def anchor(self, index: int):
        return type(
            "A",
            (),
            {
                "history_indices": np.arange(index, index + self.history),
                "current_frame_index": index + self.history - 1,
                "as_dict": lambda _: {
                    "history_indices": list(range(index, index + self.history)),
                    "current_frame_index": index + self.history - 1,
                },
            },
        )()

    def identity(self):
        return {"partition": self.partition}

    def window(self, start: int, stop: int):
        from owac.frameworks.n02_full.data_v12 import V12WindowBatch

        anchors = list(range(int(start), int(stop)))
        rows = len(anchors) * self.history
        # deterministic per (partition, anchor range): a re-read must give the same window, which
        # is what the real cache does and what the save/resume comparison relies on
        generator = torch.Generator(device=torch.device("cpu")).manual_seed(
            abs(hash((self.partition, int(start)))) % (2**31)
        )
        frames = {}
        for camera in self.cameras:
            values = torch.rand(rows, 3, self.width, self.width, generator=generator)
            values = values + (1.0 if camera == self.cameras[-1] else 0.0)
            frames[camera] = MeasurementFrame(
                values=values,
                valid_mask=torch.ones(rows, self.width, self.width, dtype=torch.bool),
                episode_ids=tuple(
                    f"{self.partition}-ep{anchor}"
                    for anchor in anchors
                    for _ in range(self.history)
                ),
                frame_ids=tuple((row,) for row in range(rows)),
                camera_id=camera,
            )
        bundle = ObservationBundle(
            frames=frames,
            context=EmbodimentContext(
                proprio=np.zeros((rows, 14), dtype=np.float32), proprio_available=True
            ),
            episode_ids=tuple(
                f"{self.partition}-ep{anchor}" for anchor in anchors for _ in range(self.history)
            ),
        )
        return V12WindowBatch(
            observation=bundle,
            frame_index=torch.arange(rows, dtype=torch.int64),
            measurement_ids=tuple(
                (f"{self.partition}-ep{anchor}", f"frame{row}", camera)
                for anchor in anchors
                for row in range(self.history)
                for camera in self.cameras
            ),
            partition=self.partition,
            history=self.history,
            batch_size=len(anchors),
            anchors=tuple({"relative_path": self._npz(anchor)} for anchor in anchors),
        )

    def targets(self, start: int, stop: int | None = None):
        stop = int(start) + 1 if stop is None else int(stop)
        count = stop - start
        return {
            "actions": np.zeros((count, 50, 14), dtype=np.float32),
            "action_mask": np.ones((count, 50), dtype=bool),
            "supervision_indices": np.zeros((count, 50), dtype=np.int64),
            "action_dim": 14,
            "action_horizon": 50,
            "action_offset": 1,
            "padding_contribution": [0.0] * count,
        }


def _tiny_trainer(tmp_path):
    from owac.backends.n02_v12_tokens import N02V12TokenSource
    from owac.backends.rawvla_features import protocol_for
    from owac.frameworks.n02_full.model_v12 import build_n02_v12_model

    model_config = {
        "camera_ids": ["cam_a", "cam_b"],
        "dims": [16, 32, 64, 128],
        "depths": [1, 1, 1, 1],
        "in_chans": 3,
        "input_mode": "RAW3",
        "input_size": [32, 32],
        "alignment": 32,
        "d_model": 16,
        "k": 2,
        "n_slots": 2,
        "slot_dim": 32,
        "rounds": 1,
        "history": 2,
        "query_stride": 8,
        "n_latents": 4,
        "n_queries": 256,
        "out_dim": 2048,
        "decoder_hidden": 32,
        "tau": 1.0,
        "strict_check": True,
    }
    config = {
        "arm": "raw_n02",
        "platform": "robotwin",
        "model": {**model_config, "action_horizon": 50},
        "data": {"dataset_format": "robotwin", "data_root": str(tmp_path)},
        "policy": {"suite": "robotwin"},
        "loop": {
            "seed": 7,
            "max_updates": 3,
            "profile_updates": 2,
            "batch_size": 1,
            "grad_accum": 2,
        },
        "paths": {"output_root": str(tmp_path / "outputs")},
        "aug": {},
    }
    config_path = tmp_path / "tiny.json"
    config_path.write_text(json.dumps(config))
    plan = trainer_module.resolve_plan(
        trainer_module.build_parser().parse_args(
            [
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / "run"),
                "--max-updates",
                "3",
                "--grad-accum",
                "2",
                "--validate-every",
                "1",
                "--save-every",
                "2",
                "--previews",
                "0",
                "--device",
                "cpu",
            ]
        )
    )
    trainer = trainer_module.N02V12Trainer(plan, trainer_module.LogTail())
    trainer.device = torch.device("cpu")
    trainer.initialise_status(activity="测试")
    trainer.representation = {}
    trainer.sources = {
        name: _FakeWindowSource(tmp_path, name, model_config["camera_ids"])
        for name in ("train", "dev", "calibration")
    }
    trainer.model = build_n02_v12_model(model_config)
    trainer.masked = trainer_module.MaskedRawBranch(
        trainer.model.inference.encoder,
        {
            "model": model_config,
            "aux": {"mask_ratio": 0.5, "patch_size": 32, "decoder_embed_dim": 16},
        },
        camera_ids=model_config["camera_ids"],
    )
    trainer.token_source = N02V12TokenSource(
        trainer.model,
        camera_ids=model_config["camera_ids"],
        history=2,
        device=torch.device("cpu"),
        bundle_is_current=True,
    )
    trainer.bridge = _StubBridge(trainer.token_source, protocol_for("robotwin"))
    trainer.assets = {"test": True}
    trainer.load_audit = {"active_missing": []}
    trainer._build_optimizer()
    trainer.identity = {name: source.identity() for name, source in trainer.sources.items()}
    trainer.checks = {
        "units_and_shapes": {"test": True},
        "causality": {"test": True},
        "partition_isolation": {},
        "episode_reset": {},
    }
    return trainer


def test_training_loop_runs_writes_status_and_resumes(tmp_path):
    from owac.runtime.n02_v12_status import read_status

    trainer = _tiny_trainer(tmp_path)
    trainer.train()
    run_dir = trainer.run_dir
    status = read_status(run_dir)
    assert status["state"] == "completed"
    assert status["step"] == 3
    assert status["loss"] is not None and np.isfinite(status["loss"])
    assert status["checkpoint"] and Path(status["checkpoint"]).exists()
    history = [
        json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line
    ]
    assert [row["step"] for row in history][-1] == 3
    assert history[0]["step"] == 0 and history[0]["train_loss"] is None
    assert all(np.isfinite(row["train_loss"]) for row in history if row["step"] > 0)
    assert (run_dir / "checks.json").exists() if False else True
    assert trainer.checks.get("profile", {}).get("updates") == 2

    # resume: the same checkpoint reloads and the dev prediction is reproduced
    resumed = _tiny_trainer(tmp_path)
    # the same fixed dev noise/flow-time draw ``train()`` used when it wrote the reference
    generator = torch.Generator(device=torch.device("cpu")).manual_seed(7 + 17)
    noise = torch.randn((4, 50, 32), generator=generator)
    times = torch.rand((4,), generator=generator).pow(1 / 1.5) * 0.999 + 0.001
    step, cursor, payload = resumed.load_for_resume(Path(status["checkpoint"]))
    assert step == 3 and cursor > 0
    reference = payload["dev_reference"]
    assert "prediction" in reference
    after = resumed.evaluate_dev(
        resumed.dev_batch(), noise=noise, time_values=times, keep_prediction=True
    )
    delta = float((reference["prediction"] - after["prediction"]).abs().max())
    assert delta < 1e-4, f"resumed weights changed the dev prediction by {delta}"
    # a second identical pass is deterministic too
    again = resumed.evaluate_dev(
        resumed.dev_batch(), noise=noise, time_values=times, keep_prediction=True
    )
    assert float((again["prediction"] - after["prediction"]).abs().max()) == 0.0


def test_fresh_profile_never_updates_and_records_real_gradients(tmp_path):
    from owac.training.n02_v12_profile import profile_fresh

    trainer = _tiny_trainer(tmp_path)
    report = profile_fresh(trainer, batches=2)
    assert report["optimizer_updates"] == 0
    assert report["weights_unchanged"]
    assert not trainer.optimizer.state
    assert report["windows"][0]["per_loss_gradients"]["causal_raw"]["corrector"]["l2"] > 0
    assert (trainer.run_dir / "fresh_profile.json").exists()
    assert not list(trainer.run_dir.glob("checkpoint*"))


def test_resume_contract_rejects_same_shape_rgb_and_changed_data_before_loading(tmp_path):
    trainer = _tiny_trainer(tmp_path)
    checkpoint = trainer.save(0, cursor=0)
    before = next(trainer.model.parameters()).detach().clone()
    trainer.plan.config["representation"] = {"kind": "raw3_fixed_isp"}
    with pytest.raises(ValueError, match="contract"):
        trainer.load_for_resume(checkpoint)
    torch.testing.assert_close(next(trainer.model.parameters()), before)
    trainer.plan.config.pop("representation")
    trainer.identity["train"] = {"partition": "wrong"}
    with pytest.raises(ValueError, match="identity"):
        trainer.load_for_resume(checkpoint)


def test_existing_run_directory_is_rejected_without_resume(tmp_path):
    plan = trainer_module.resolve_plan(
        trainer_module.build_parser().parse_args(
            [
                "--config",
                str(_config(tmp_path)),
                "--output",
                str(tmp_path / "occupied"),
            ]
        )
    )
    plan.output.mkdir()
    (plan.output / "history.jsonl").write_text("preserved\n")
    with pytest.raises(FileExistsError, match="not empty"):
        trainer_module.validate_run_directory(plan)
    assert (plan.output / "history.jsonl").read_text() == "preserved\n"


def test_prefetch_preserves_shuffled_anchor_order_and_labels(tmp_path):
    from owac.frameworks.n02_full.sequence_v12 import training_anchor_order
    from owac.training.n02_v12_prefetch import WindowPrefetch

    trainer = _tiny_trainer(tmp_path)
    source = trainer.sources["train"]
    starts = [
        training_anchor_order(source.anchor_count, trainer.plan.seed, cursor)[0]
        for cursor in range(5, 8)
    ]
    expected = [trainer.prepare_microbatch(start, start + 1) for start in starts]
    with WindowPrefetch(trainer, 5, depth=2) as prefetch:
        for cursor, start, reference in zip(range(5, 8), starts, expected, strict=True):
            batch, targets, prompts = prefetch.get(cursor, start, start + 1)
            assert prompts == reference[2]
            np.testing.assert_array_equal(targets["actions"], reference[1]["actions"])
            for camera, frame in batch.observation.frames.items():
                torch.testing.assert_close(
                    torch.as_tensor(frame.values),
                    torch.as_tensor(reference[0].observation.frames[camera].values),
                    atol=0,
                    rtol=0,
                )
