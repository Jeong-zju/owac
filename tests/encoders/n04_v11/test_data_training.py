"""N04 v1.1 S2 数据、损失、checkpoint、进度与 eval100 的合成单测。

全部用假数据／假编码器／假后端，验证可改变结论的事实：窗口与监督点数量、本体上下文不含
未来信息、损失权重与均值口径、checkpoint 往返、progress.json 原子写与字段完整、eval100
的 5 域 × 10 结构，以及闭环计数口径。
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from owac.core.encoder_contract_n04 import VisualTokens
from owac.evaluation.n04_v11 import SubprocessEpisodeRunner, load_eval_list, run_closed_loop
from owac.raw.n04_v11_data import (
    PLATFORM_SPECS,
    ActionNormalizer,
    V11DataIndex,
    load_window_batch,
    window_candidates,
)
from owac.raw.rawvla_sequence_n04 import RawVLASequenceLoader, v11_window_plan
from owac.training.n04_v11 import (
    DEFAULT_LOSS_WEIGHTS,
    PROGRESS_FIELDS,
    V11Trainer,
    apply_lora,
    build_optimizer,
    forecast_nll,
    freeze_parameters,
    load_config,
    masked_mean,
    read_jsonl,
    summarize_eval_results,
    vicreg_terms,
    weighted_total,
    write_progress,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL100 = REPO_ROOT / "configs/n04/v1_1/eval100.json"


# --------------------------------------------------------------------------------------
# 合成官方数据
# --------------------------------------------------------------------------------------


def _write_libero(path: Path, frames: int, *, seed: int, phase_shift: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    actions = (rng.normal(size=(frames, 7)) + phase_shift).astype(np.float32)
    np.savez(
        path,
        agentview_raw_uint8=rng.integers(0, 255, (frames, 6, 6, 3), dtype=np.uint8),
        wrist_raw_uint8=rng.integers(0, 255, (frames, 6, 6, 3), dtype=np.uint8),
        robot_states=rng.normal(size=(frames, 8)).astype(np.float32),
        actions=actions,
        metadata_json=json.dumps({"task_description": "pick the cube", "demo_key": "demo_0"}),
    )


def _write_robotwin(path: Path, frames: int, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # HDR-derived：范围可以超过 1，平台固定缩放必须是 identity
    np.savez(
        path,
        head_camera_raw_float32=(rng.random((frames, 6, 6, 3)) * 4.0).astype(np.float32),
        left_camera_raw_float32=(rng.random((frames, 6, 6, 3)) * 4.0).astype(np.float32),
        right_camera_raw_float32=(rng.random((frames, 6, 6, 3)) * 4.0).astype(np.float32),
        actions=rng.normal(size=(frames, 14)).astype(np.float32),
        robot_states=rng.normal(size=(frames, 16)).astype(np.float32),
        metadata_json=json.dumps({"task_name": "grab_roller"}),
    )


def _assignments(relpaths: dict[str, tuple[str, str, str]]) -> dict:
    items = [
        {
            "platform": platform,
            "source_trajectory_id": source,
            "split": split,
            "task_key": relpath.split("/")[0],
            "relpaths": [relpath],
        }
        for relpath, (platform, source, split) in relpaths.items()
    ]
    return {"schema": "n04-s0-splits-v1", "seed": 7, "main": {"assignments": items}}


@pytest.fixture()
def dataset(tmp_path: Path) -> dict:
    root = tmp_path / "data"
    frames = 80
    _write_libero(root / "libero/raw/suite_a/episode_000001/variant_00.npz", frames, seed=1)
    _write_libero(root / "libero/raw/suite_a/episode_000002/variant_00.npz", frames, seed=2)
    _write_libero(root / "libero/raw/suite_b/episode_000003/variant_00.npz", frames, seed=3)
    _write_robotwin(root / "robotwin2/raw/grab_roller/episode_000001.npz", frames, seed=4)
    relpaths = {
        "suite_a/episode_000001/variant_00.npz": ("libero", "libero:suite_a:demo_1", "train"),
        "suite_a/episode_000002/variant_00.npz": ("libero", "libero:suite_a:demo_2", "train"),
        "suite_b/episode_000003/variant_00.npz": ("libero", "libero:suite_b:demo_3", "dev"),
        "grab_roller/episode_000001.npz": (
            "robotwin",
            "robotwin:grab_roller:episode_1",
            "train",
        ),
    }
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps(_assignments(relpaths)))
    norm_stats = tmp_path / "norm_stats.json"
    norm_stats.write_text(
        json.dumps(
            {
                "state": {"q01": [-1.0] * 8, "q99": [1.0] * 8},
                "action": {"q01": [0.0] * 7, "q99": [1.0] * 7},
                "actions": {"mean": [0.0] * 14, "std": [1.0] * 14},
            }
        )
    )
    return {
        "root": root,
        "split_file": split_file,
        "norm_stats": norm_stats,
        "relpaths": relpaths,
        "libero_file": root / "libero/raw/suite_a/episode_000001/variant_00.npz",
    }


def _config(dataset: dict, tmp_path: Path, **overrides) -> dict:
    config = {
        "schema": "test",
        "stage": "S2",
        "platform": "libero",
        "arm": "D",
        "seed": 101,
        "data": {
            "root": str(dataset["root"]),
            "platform_dirs": {"libero": "libero/raw", "robotwin": "robotwin2/raw"},
            "split_file": str(dataset["split_file"]),
            "splits": ["train", "dev"],
            "train_split": "train",
            "dev_split": "dev",
            "window": {"burn_in": 16, "gradient": 32, "stride": 16, "supervision_points": 4},
            "proprio_history": 2,
            "dev_windows": 1,
            "norm_stats": {
                "libero": str(dataset["norm_stats"]),
                "robotwin": str(dataset["norm_stats"]),
            },
        },
        "model": {
            "tokens_per_camera": 4,
            "token_dim": 8,
            "name": "fake",
            "proprio_dim": 24,  # (history+1) × LIBERO 本体维度
        },
        "backend": {"action_dim": 32, "num_denoise_steps": 2, "adapter": {}},
        "loss": {"weights": dict(DEFAULT_LOSS_WEIGHTS)},
        "optim": {
            "batch": 2,
            "micro_batch": 2,
            "steps": 4,
            "warmup_steps": 2,
            "grad_clip": 1.0,
            "amp": False,
            "betas": [0.9, 0.95],
            "unmatched": "pi05_adapt",
            "groups": [
                {
                    "name": "convnext",
                    "lr": 1e-4,
                    "weight_decay": 0.05,
                    "patterns": ["encoder"],
                },
                {
                    "name": "pi05_adapt",
                    "lr": 1e-5,
                    "weight_decay": 0.0,
                    "patterns": ["backend"],
                },
            ],
        },
        "monitor": {"log_every": 1, "checkpoint_every": 2, "eval_every": 2},
    }
    for key, value in overrides.items():
        if isinstance(value, dict):
            config[key] = {**config.get(key, {}), **value}
        else:
            config[key] = value
    return config


# --------------------------------------------------------------------------------------
# 假编码器 / 假后端（只实现契约里的调用面）
# --------------------------------------------------------------------------------------


class FakeEncoder(nn.Module):
    """WS-C ``step`` 的单帧调用面：``values`` 是一帧的多相机列表（每项 ``[C,H,W]``）。"""

    def __init__(
        self,
        *,
        tokens_per_camera: int = 4,
        token_dim: int = 8,
        proprio_dims: tuple[int, ...] = (9, 15),
        emit_losses: bool = False,
        graph_state: bool = True,
    ):
        super().__init__()
        self.tokens_per_camera = tokens_per_camera
        self.token_dim = token_dim
        self.proprio_dims = proprio_dims
        self.emit_losses = emit_losses
        self.graph_state = graph_state
        self.proj = nn.Linear(3, token_dim)
        self.config = SimpleNamespace(
            input_mode="rawvla_rgb3", scale_name="libero_uint8", saturation_level=1.0
        )
        self.reset_calls = 0
        self.step_calls = 0
        self.proprio_seen: list[int] = []

    def reset(self, batch_or_episode=None):
        self.reset_calls += 1
        return None

    def step(self, values, proprio_context, state=None, **kwargs):
        self.step_calls += 1
        frames = [torch.as_tensor(item) for item in values]
        assert all(frame.dim() == 3 for frame in frames), "step 只接受单帧 [C,H,W]"
        proprio = torch.as_tensor(proprio_context.proprio).reshape(-1)
        self.proprio_seen.append(int(proprio.numel()))
        stacked = torch.stack(frames, dim=0)  # [C_cam, C, H, W]
        pooled = stacked.mean(dim=(2, 3))  # [C_cam, C]
        projected = self.proj(pooled)
        tokens = projected[:, None, :].expand(-1, self.tokens_per_camera, -1)
        values_out = tokens.reshape(1, -1, self.token_dim)
        debug = {"frame_key": kwargs.get("frame_key")}
        # WS-C 目前只放 correspondence；forecast 由训练器自算（Lead 裁定）
        if self.emit_losses:
            debug["losses"] = {"correspondence": values_out.square().mean()}
        state = values_out.mean(dim=1)
        return SimpleNamespace(
            prefix_tokens=values_out,
            state=state if self.graph_state else state.detach(),
            debug=debug,
        )

    def readout_tokens(self, output, tokens_per_camera):
        assert tokens_per_camera == self.tokens_per_camera
        mask = torch.ones(output.prefix_tokens.shape[:2], dtype=torch.bool)
        return VisualTokens(values=output.prefix_tokens, mask=mask)


class FakeTokenizer:
    """真后端 tokenizer 返回 numpy：这里复刻该行为，覆盖训练器的 dtype 转换。"""

    def tokenize(self, text, state=None):
        # 真 `LazyPaliGemmaTokenizer.tokenize` 返回一维 [L] 的 int64/bool
        length = 3
        return (
            np.zeros((length,), dtype=np.int64),
            np.ones((length,), dtype=bool),
        )


class FakeBackend(nn.Module):
    """与 PI0 同签名的最小可执行后端：动作损失经 prefix 回传到编码器。"""

    def __init__(self, *, control_dim: int = 7, horizon: int = 10, width: int = 8):
        super().__init__()
        self.max_state_dim = 8
        self.action_dim = 32
        self.action_horizon = horizon
        self.discrete_state_input = False
        self.state_pad_value = 0.0
        self.tokenizer = FakeTokenizer()
        self.embedding = nn.Embedding(32, width)
        self.vlm = SimpleNamespace(embed_language_tokens=self.embedding.forward)
        self.action_in = nn.Linear(32, width)
        self.action_out = nn.Linear(width, 32)
        self.control_dim = control_dim

    @staticmethod
    def _preprocess_observation(observation):
        return (
            (),
            (),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def forward_from_observation(
        self, observation, actions, noise=None, time=None, return_debug=False
    ):
        prefix, _, _ = self.embed_prefix(
            (), (), observation.tokenized_prompt, observation.tokenized_prompt_mask
        )
        hidden = prefix.mean(dim=1, keepdim=True)
        velocity = self.action_out(self.action_in(actions) + hidden)
        return {"v_t": velocity, "u_t": torch.zeros_like(velocity)}

    def sample_actions(self, observation, noise=None, num_steps=10):
        return torch.zeros(1, self.action_horizon, 32)


class FakeMeasurement(nn.Module):
    """最小测量替身：`_target_features` 只用 backbone_levels/fuse + 一个可训练参数。"""

    OBS_DIM = 8

    def __init__(self) -> None:
        super().__init__()
        from owac.frameworks.n04.encoder_v11 import default_measurement_config

        self.config = default_measurement_config()
        self.weight = nn.Parameter(torch.ones(1))

    def backbone_levels(self, values):
        return [values * self.weight]

    def fuse(self, levels):
        reduced = levels[0][:, :1, ::4, ::4]
        return reduced.repeat(1, self.OBS_DIM, 1, 1)


class FakeQualityHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, values, support, saturated, latent, xy_px, grid=None):
        return torch.ones(xy_px.shape[0], xy_px.shape[1], FakeMeasurement.OBS_DIM) * (
            self.scale.clamp_min(0.1)
        )


class FakeStateStack(nn.Module):
    """预测侧替身：`predict` 加一个非零可训练偏置（roll 会改变均值，便于验证 horizon 差异）。"""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((1,), 0.5))
        self.calls = 0
        self.proprio_seen: list[torch.Tensor] = []

    def predict(self, state, proprio_context):
        self.calls += 1
        self.proprio_seen.append(proprio_context.detach().clone())
        delta = torch.zeros_like(state.mode_mean)
        delta[..., 0] = self.bias
        return type(state)(
            xy=state.xy,
            ids=state.ids,
            age=state.age,
            valid=state.valid,
            fresh=state.fresh,
            mode_mean=state.mode_mean + delta,
            mode_cov=state.mode_cov,
            mode_weights=state.mode_weights,
            sensor_size_hw=state.sensor_size_hw,
        )


class FakeForecastEncoder(FakeEncoder):
    """带测量／质量头／状态预测替身的编码器：返回 per-camera 后验状态与质量头 debug。"""

    SUPPORTS = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.measurement = FakeMeasurement()
        self.quality_head = FakeQualityHead()
        self.state_stack = FakeStateStack()
        self.config = SimpleNamespace(
            input_mode="rawvla_rgb3", scale_name="libero_uint8", saturation_level=1.0
        )

    def _posterior(self, cameras, device, size_hw):
        from owac.frameworks.n04.state_estimator import SupportState

        n, obs = self.SUPPORTS, FakeMeasurement.OBS_DIM
        xy = torch.tensor([[2.0, 2.0], [4.0, 2.0], [2.0, 4.0], [4.0, 4.0]], device=device)[None]
        return {
            camera: SupportState(
                xy=xy.clone(),
                ids=torch.arange(n, device=device)[None],
                age=torch.zeros(1, n, dtype=torch.long, device=device),
                valid=torch.ones(1, n, dtype=torch.bool, device=device),
                fresh=torch.zeros(1, n, dtype=torch.bool, device=device),
                mode_mean=torch.zeros(1, n, 2, 2 * obs, device=device),
                mode_cov=tuple(torch.full((1, n, 2, obs), 0.5, device=device) for _ in range(3)),
                mode_weights=torch.full((1, n, 2), 0.5, device=device),
                sensor_size_hw=size_hw,
            )
            for camera in cameras
        }

    def step(self, values, proprio_context, state=None, **kwargs):
        output = super().step(values, proprio_context, state, **kwargs)
        cameras = list(kwargs.get("camera_id") or ["camera0"])
        device = output.prefix_tokens.device
        size_hw = (int(values[0].shape[-2]), int(values[0].shape[-1]))
        posterior = self._posterior(cameras, device, size_hw)
        # Source encoder's live features and R must receive forecast gradients.
        posterior = {
            cam: replace(state, mode_mean=state.mode_mean + output.prefix_tokens.mean())
            for cam, state in posterior.items()
        }
        output.state = SimpleNamespace(per_camera=posterior)
        output.debug["flow"] = {
            camera: torch.zeros(1, 2, *size_hw, device=device) for camera in cameras
        }
        output.debug["quality"] = {
            camera: {
                "R_diag": torch.full(
                    (1, self.SUPPORTS, FakeMeasurement.OBS_DIM), 0.5, device=device
                )
                * self.quality_head.scale
            }
            for camera in cameras
        }
        return output


def _trainer(dataset: dict, tmp_path: Path, **overrides) -> V11Trainer:
    config = _config(dataset, tmp_path, **overrides)
    encoder = FakeEncoder()
    return V11Trainer(
        config,
        seed=101,
        run_dir=tmp_path / "run",
        device="cpu",
        encoder=encoder,
        backend=FakeBackend(),
    )


# --------------------------------------------------------------------------------------
# 窗口与监督点
# --------------------------------------------------------------------------------------


def test_window_plan_and_supervision_point_counts():
    plans = v11_window_plan(80, burn_in=16, gradient=32, stride=16)
    assert [plan["start"] for plan in plans] == [0, 16, 32]
    first = plans[0]
    assert first["length"] == 48 and first["burn_in"] == 16 and first["gradient"] == 32
    assert first["supervision_offsets"] == (4, 12, 20, 28)
    assert first["supervision_index"] == (20, 28, 36, 44)
    assert len(first["supervision_index"]) == 4
    assert v11_window_plan(40) == []  # 比窗口短的文件不进训练
    with pytest.raises(ValueError):
        v11_window_plan(80, gradient=30, supervision_points=4)


def test_window_targets_match_full_sequence_and_platform_phase(dataset):
    index = V11DataIndex(root=dataset["root"], split_file=dataset["split_file"])
    entry = index.entries("libero", "train")[0]
    window = load_window_batch(
        entry, start=16, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )
    assert len(window) == 48
    assert window.camera_ids == PLATFORM_SPECS["libero"].camera_ids
    assert window.frame_ids[0] == 16 and window.frame_ids[-1] == 63
    assert [position for position, _, _ in window.supervision_points()] == [20, 28, 36, 44]
    full = RawVLASequenceLoader(suite="libero").load(entry.path)
    for position, actions, _ in window.supervision_points():
        global_frame = window.start + position
        assert torch.allclose(actions, full.supervised_action[global_frame])
    assert window.raw["agentview"].max() > 1.0  # 原始单位 uint8 域 [0,255]，这里不再缩放

    robot = index.entries("robotwin", "train")[0]
    robot_window = load_window_batch(
        robot, start=0, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )
    robot_full = RawVLASequenceLoader(suite="robotwin").load(robot.path)
    assert robot_window.proprio.shape[-1] == 14  # 当前 14 维关节状态，不是 16 维末端表示
    assert torch.allclose(robot_window.proprio[0, -1], robot_full.proprio[0])
    position, actions, mask = robot_window.supervision_points()[0]
    expected = robot_full.supervised_action[position]
    assert torch.allclose(actions, expected)
    assert float(mask.sum()) > 0
    with np.load(robot.path) as npz:
        original = npz["head_camera_raw_float32"][0].transpose(2, 0, 1)
    assert torch.allclose(robot_window.raw["head"][0], torch.as_tensor(original))  # FP32 原值
    assert robot_window.raw["head"].max() > 1.0


def test_proprio_context_never_reads_future_frames(dataset, tmp_path):
    index = V11DataIndex(root=dataset["root"], split_file=dataset["split_file"])
    entry = index.entries("libero", "train")[0]
    window = load_window_batch(
        entry, start=0, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )
    frame = 20
    before = window.proprio[frame].clone()
    window.proprio[frame + 1 :] = 0.0
    window.targets[frame + 1 :] = 0.0
    assert torch.allclose(before, window.proprio[frame])
    # 上下文里没有未来动作：targets 与 proprio 是两个不同张量
    assert window.proprio is not window.targets
    assert window.proprio_padded[0] == 2  # 开头不足 history+1 帧，重复真实首帧
    assert window.proprio_padded[frame] == 0

    loader = RawVLASequenceLoader(suite="libero")
    loaded = loader.load_window(entry.path, length=48, start=0)
    context_at = loader.proprio_context_at(loaded, frame, history=2)
    assert context_at.proprio.shape[0] == 3
    assert context_at.executed_action is None
    assert context_at.executed_action_available is False
    assert context_at.metadata["future_action_excluded"] is True
    assert torch.allclose(context_at.proprio[-1], before[-1])

    # 改动 t 之后的帧不改变 t 时刻上下文：重写文件后再取一次
    fresh = index.entries("libero", "train")[0]
    npz = np.load(fresh.path)
    payload = {key: npz[key] for key in npz.files}
    npz.close()
    payload["robot_states"] = payload["robot_states"].copy()
    payload["robot_states"][frame + 1 :] = 99.0
    np.savez(fresh.path, **payload)
    reloaded = RawVLASequenceLoader(suite="libero").load_window(fresh.path, length=48, start=0)
    context_after = RawVLASequenceLoader(suite="libero").proprio_context_at(
        reloaded, frame, history=2
    )
    assert torch.allclose(context_at.proprio, context_after.proprio)


# --------------------------------------------------------------------------------------
# 损失口径
# --------------------------------------------------------------------------------------


def test_masked_mean_and_weighted_total():
    values = torch.tensor([1.0, 3.0, 5.0])
    mask = torch.tensor([1.0, 0.0, 1.0])
    assert torch.allclose(masked_mean(values, mask), torch.tensor(3.0))
    assert torch.allclose(masked_mean(values), torch.tensor(3.0))
    assert torch.allclose(masked_mean(values, torch.zeros(3)), torch.tensor(0.0))

    terms = {
        "action": torch.tensor(2.0),
        "correspondence": torch.tensor(4.0),
        "forecast": None,
        "representation": torch.tensor(1.0),
    }
    total, report = weighted_total(
        terms, {"action": 1.0, "correspondence": 0.1, "forecast": 0.1, "representation": 0.01}
    )
    assert report["action"] == 2.0 and report["correspondence"] == 4.0
    assert report["forecast"] is None  # 缺失项记 null，不按 0 计入
    assert math.isclose(float(total), 2.0 + 0.4 + 0.01, rel_tol=1e-6)
    empty, empty_report = weighted_total({"action": None}, {"action": 1.0})
    assert empty is None and empty_report == {"action": None}


def test_vicreg_and_forecast_nll_keep_full_terms():
    constant = torch.ones(64, 8)
    terms = vicreg_terms(constant, eps=1e-4)
    # 常数特征：std 只有 eps 的正则项，方差项 ≈ 1 - sqrt(eps)，协方差项为 0
    assert float(terms["variance"]) == pytest.approx(1.0 - math.sqrt(1e-4), abs=1e-4)
    assert float(terms["covariance"]) == pytest.approx(0.0, abs=1e-6)
    varied = torch.randn(256, 8)
    assert float(vicreg_terms(varied)["variance"]) < 1.0

    target = torch.zeros(1, 4, 3)
    mean = torch.zeros(1, 4, 3)
    var = torch.ones(1, 4, 3)
    unit = float(forecast_nll(mean, var, target))
    assert unit == pytest.approx(0.5 * (0.0 + math.log(2 * math.pi)), abs=1e-5)
    larger = float(forecast_nll(mean, var, target + 2.0))
    assert larger > unit  # 二次项没有被截断
    wide = float(forecast_nll(mean, var * 4, target))
    assert wide != unit  # log variance 仍参与
    assert float(forecast_nll(mean, var, target, mask=torch.zeros(1, 4, 3))) == 0.0


# --------------------------------------------------------------------------------------
# 优化器分组 / LoRA
# --------------------------------------------------------------------------------------


def test_optimizer_groups_weight_decay_and_lora():
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    names = [f"encoder.{name}" for name, _ in model.named_parameters()]
    groups = [
        {"name": "convnext", "lr": 1e-4, "weight_decay": 0.05, "patterns": ["encoder"]},
        {"name": "pi05_adapt", "lr": 1e-5, "weight_decay": 0.0, "patterns": ["backend"]},
    ]
    optimizer, report = build_optimizer(
        list(zip(names, model.parameters(), strict=True)), groups=groups
    )
    decay = {group["name"]: group for group in optimizer.param_groups}
    assert decay["convnext:decay"]["lr"] == 1e-4 and decay["convnext:decay"]["weight_decay"] == 0.05
    assert decay["convnext:no_decay"]["weight_decay"] == 0.0
    assert report["groups"]["convnext"]["params"] > 0

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(4, 4)
            self.mlp = nn.Linear(4, 4)
            self.action_head = nn.Linear(4, 4)

    backend = Backbone()
    lora = apply_lora(backend, rank=2)
    assert any("q_proj" in name for name in lora["wrapped"])
    freeze = freeze_parameters(backend, trainable_prefixes=("action_head.",))
    assert any("action_head" in name for name in freeze["trainable"])
    assert any("lora_" in name for name in freeze["trainable"])
    assert not backend.mlp.weight.requires_grad
    assert not backend.self_attn.q_proj.base.weight.requires_grad
    assert backend.self_attn.q_proj.lora_a.weight.requires_grad
    out = backend.self_attn.q_proj(torch.randn(2, 4))
    assert out.shape == (2, 4)
    # 冻结后端会读 `.weight.dtype`：包装后属性面必须保持
    assert backend.self_attn.q_proj.weight is backend.self_attn.q_proj.base.weight
    assert backend.self_attn.q_proj.bias is backend.self_attn.q_proj.base.bias
    assert any("q_proj.base.weight" in key for key in backend.state_dict())


# --------------------------------------------------------------------------------------
# 训练一步 / checkpoint / 进度
# --------------------------------------------------------------------------------------


def test_train_step_steps_encoder_once_per_frame_and_writes_monitor(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    info = trainer.train_step(0)
    assert info["train_loss"] is not None and info["train_loss"] > 0
    assert info["terms"]["action"] is not None
    assert info["supervision_points"] == 4 * 2  # 2 段序列 × 4 个监督点
    assert trainer.encoder.step_calls % 48 == 0
    # 每帧只融合一次：2 段序列 × 48 帧；监督帧也走同一次 encoder.step
    assert trainer.encoder.step_calls == 2 * 48
    assert trainer.encoder.reset_calls == 2  # 每段窗口边界 reset，不跨 episode 继承状态
    assert set(trainer.encoder.proprio_seen) == {24}  # (history+1) × 8 维，未被静默截断
    trainer._write_monitor(
        trainer.run_dir / "metrics.jsonl", trainer.run_dir / "progress.json", 4, info
    )
    rows = read_jsonl(trainer.run_dir / "metrics.jsonl")
    assert rows and rows[-1]["terms"]["action"] is not None
    assert trainer.encoder.reset_calls >= 1
    progress = json.loads((trainer.run_dir / "progress.json").read_text())
    assert set(PROGRESS_FIELDS) <= set(progress)
    assert not (trainer.run_dir / "progress.json.tmp").exists()


@pytest.mark.parametrize("missing", ["correspondence", "forecast"])
def test_missing_required_objective_stops_before_optimizer(dataset, tmp_path, missing):
    trainer = _trainer(dataset, tmp_path)
    trainer.config["loss"]["require_all_objectives"] = True
    trainer.loss_weights = {
        name: float(name in ("action", missing)) for name in DEFAULT_LOSS_WEIGHTS
    }
    before = {name: value.detach().clone() for name, value in trainer._trainable_parameters()}
    epoch = trainer.scheduler.last_epoch
    with pytest.raises(FloatingPointError, match=f"Missing required objectives.*{missing}"):
        trainer.train_step(0)
    assert trainer.scheduler.last_epoch == epoch and trainer.step == 0
    assert not trainer.optimizer.state
    for name, value in trainer._trainable_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert value.grad is None
    assert trainer.activity["objective_coverage"][missing]["valid_items"] == 0


def test_required_objectives_report_actual_coverage(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    trainer.encoder.emit_losses = True
    trainer.config["loss"]["require_all_objectives"] = True
    trainer.loss_weights = {
        name: float(name in ("action", "correspondence")) for name in DEFAULT_LOSS_WEIGHTS
    }
    info = trainer.train_step(0)
    assert info["objective_coverage"]["action"]["windows"] == trainer.batch
    assert info["objective_coverage"]["action"]["valid_items"] == trainer.batch * 4
    assert info["objective_coverage"]["correspondence"]["valid_items"] == trainer.batch * 32
    assert trainer.step == 1


def test_action_gradient_reaches_encoder(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss, info = trainer._forward_windows(
        [
            load_window_batch(
                entry,
                start=0,
                burn_in=16,
                gradient=32,
                supervision_points=4,
                proprio_history=2,
            )
            for entry, _ in trainer.train_candidates[:1]
        ],
        train=True,
    )
    assert loss is not None and info["terms"]["action"] is not None
    # 逐监督点 backward 已在前向内完成：返回的标量只用于报告
    assert not loss.requires_grad
    grad = trainer.encoder.proj.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0


def test_per_point_backward_matches_single_backward_scale(dataset, tmp_path):
    """逐监督点 backward 与旧的一次性 backward 的梯度逐值等价（TBPTT 无影响时）。

    替身的携带状态不接图（``graph_state=False``），因此"段边界切断状态图"不改变梯度；
    等价性验证的是**有效 loss 尺度**：每段 action/P + 辅助项/N，外层 1/W。
    """
    config = _config(dataset, tmp_path)
    encoder = FakeEncoder(emit_losses=True, graph_state=False)
    trainer = V11Trainer(
        config,
        seed=101,
        run_dir=tmp_path / "run",
        device="cpu",
        encoder=encoder,
        backend=FakeBackend(),
    )
    entry, start = trainer.train_candidates[0]
    window = load_window_batch(
        entry, start=start, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )

    def run_once(single_backward: bool):
        trainer.optimizer.zero_grad(set_to_none=True)
        loss, info = trainer._forward_window(
            window, train=True, backward_weight=1.0, single_backward=single_backward
        )
        assert loss is not None
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in trainer._trainable_parameters()
            if parameter.grad is not None
        }
        assert grads
        return grads, info

    reference, reference_info = run_once(True)
    segmented, segmented_info = run_once(False)
    assert reference.keys() == segmented.keys()
    for name, value in reference.items():
        assert torch.allclose(value, segmented[name], rtol=1e-5, atol=1e-6), name
    assert segmented_info["denominators"] == reference_info["denominators"]
    assert segmented_info["segments"] == 4  # 监督点数：不因显存改动减少
    assert segmented_info["supervision_points"] == 4
    assert segmented_info["applied_terms"] == reference_info["applied_terms"]
    expected = sum(
        trainer.loss_weights[name] * value
        for name, value in segmented_info["applied_terms"].items()
        if value is not None
    )
    assert segmented_info["applied_train_loss"] == pytest.approx(expected)
    # 报告与实反传是同一目标；burn-in 不进入分子或分母。
    assert segmented_info["valid_loss_counts"]["correspondence"] == 32
    assert segmented_info["applied_terms"]["correspondence"] == pytest.approx(
        segmented_info["terms"]["correspondence"]
    )


def test_arm_ab_requires_named_isp_and_can_start(dataset, tmp_path):
    for arm in ("A", "B"):
        config = _config(dataset, tmp_path, arm=arm)
        with pytest.raises(ValueError, match="fixed_clip_srgb_v1"):
            V11Trainer(
                config,
                seed=101,
                run_dir=tmp_path / f"run{arm}",
                device="cpu",
                encoder=FakeEncoder(),
                backend=FakeBackend(),
            )
        config["model"]["isp"] = "fixed_clip_srgb_v1"
        trainer = V11Trainer(
            config,
            seed=101,
            run_dir=tmp_path / f"run{arm}",
            device="cpu",
            encoder=FakeEncoder(),
            backend=FakeBackend(),
        )
        assert trainer.config["arm"] == arm


def test_dev_uses_fixed_noise_preserves_rng_and_restores_modes(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    trainer.encoder.train()
    trainer.backend.train()
    seen = []

    def noisy_forward(windows, *, train):
        assert not train and not trainer.encoder.training and not trainer.backend.training
        assert trainer._representation_term(torch.randn(3, 4)) is None
        assert not trainer.forecast_enabled
        value = torch.rand(()) + random.random() + float(np.random.random())
        seen.append(float(value))
        return value, {"supervision_points": 4, "terms": {"action": float(value)}}

    trainer._forward_windows = noisy_forward
    before = torch.random.get_rng_state().clone()
    python_before, numpy_before = random.getstate(), np.random.get_state()
    first = trainer.evaluate_dev_action_loss()
    second = trainer.evaluate_dev_action_loss()
    assert first["dev_loss"] == second["dev_loss"]
    assert seen[0] == seen[1]
    assert torch.equal(before, torch.random.get_rng_state())
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert trainer.encoder.training and trainer.backend.training
    assert trainer.forecast_enabled and not trainer._evaluating_actions_only


def test_fixed_dev_restores_math_and_compiled_methods_on_failure(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    native = trainer.encoder.forward

    def compiled(*args, **kwargs):
        return None

    trainer.encoder.forward = compiled
    trainer._eager_methods = [(trainer.encoder, "forward", native)]
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )
    with pytest.raises(RuntimeError, match="reference failure"):
        with trainer._fixed_dev_math():
            assert trainer.encoder.forward == native
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.get_float32_matmul_precision() == "highest"
            assert not torch.backends.cudnn.benchmark
            assert torch.backends.cudnn.deterministic
            raise RuntimeError("reference failure")
    assert trainer.encoder.forward is compiled
    assert before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )


@pytest.mark.parametrize("shape", [(8, 32), (32, 8), (512, 2048)])
def test_vicreg_smaller_gram_preserves_covariance_and_gradients(shape):
    torch.manual_seed(101)
    x = torch.randn(*shape, requires_grad=True)
    center = x - x.mean(0, keepdim=True)
    covariance = center.T @ center / (shape[0] - 1)
    off = covariance - torch.diag_embed(covariance.diag())
    reference = off.square().sum() / shape[1]
    actual = vicreg_terms(x)["covariance"]
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
    (expected_grad,) = torch.autograd.grad(reference, (x,))
    (actual_grad,) = torch.autograd.grad(actual, (x,))
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)


def test_profile_window_records_stages_and_memory_settings(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    profile = trainer.profile_window()
    assert set(profile["stages"]) == {"window_forward_backward"}
    assert profile["segments"] == 4
    assert profile["memory_settings"]["per_segment_term_gradients"] is True
    assert profile["memory_settings"]["forecast_anchor_stride"] == 1

    trainer.config["monitor"]["profile"] = True
    trainer.run(steps=1)
    rows = read_jsonl(trainer.run_dir / "metrics.jsonl")
    assert any(row.get("phase") == "profile" for row in rows)
    report = json.loads((trainer.run_dir / "train-report.json").read_text())
    assert report["memory"]["per_segment_term_gradients"] is True
    assert report["memory"]["raft_resize_hw"] is None
    assert report["memory"]["gradient_checkpointing"] is False
    assert report["steps"] == 1


def test_checkpoint_roundtrip_and_config_guard(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    for step in range(2):
        trainer.train_step(step)
    path = trainer.save_checkpoint(tmp_path / "ckpt.pt")
    assert not Path(path + ".tmp").exists()
    resumed = _trainer(dataset, tmp_path / "again")
    resumed.load_checkpoint(path)
    assert resumed.step == trainer.step
    for (name, value), (other_name, other) in zip(
        trainer.encoder.state_dict().items(), resumed.encoder.state_dict().items(), strict=True
    ):
        assert name == other_name and torch.allclose(value, other)
    changed = _trainer(dataset, tmp_path / "changed")
    changed.config["model"] = {"tokens_per_camera": 8, "token_dim": 16}
    with pytest.raises(ValueError, match="结构"):
        changed.load_checkpoint(path)


def test_progress_json_atomic_and_contract_fields(tmp_path):
    path = tmp_path / "progress.json"
    config = {"stage": "S2", "platform": "libero", "arm": "D", "seed": 101}
    payload = write_progress(
        path,
        config=config,
        phase="train",
        step=12,
        budget_steps=100,
        train_loss=0.5,
        dev_loss=None,
        step_seconds=1.25,
        gpu_memory_gb=None,
        log_tail=["step 12"],
    )
    assert set(PROGRESS_FIELDS) <= set(payload)
    assert payload["dev_loss"] is None and payload["gpu_memory_gb"] is None
    assert json.loads(path.read_text())["step"] == 12
    assert not path.with_suffix(".json.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []


# --------------------------------------------------------------------------------------
# eval100 与闭环计数
# --------------------------------------------------------------------------------------


def test_eval100_has_five_domains_times_ten_per_platform():
    listing = load_eval_list(EVAL100)
    entries = listing["entries"]
    assert len(entries) == 100
    counts = listing["counts"]
    assert set(counts) == {"libero", "robotwin"}
    for platform in ("libero", "robotwin"):
        assert sum(counts[platform].values()) == 50
        assert set(counts[platform]) == set(listing["lighting_domains"])
        assert set(counts[platform].values()) == {10}
    identities = [entry["identity"] for entry in entries]
    assert len(set(identities)) == 100
    for entry in entries:
        assert {"platform", "task", "initial_state", "env_seed", "lighting_domain"} <= set(entry)
    libero_tasks = {(e["task"], e["initial_state"]) for e in entries if e["platform"] == "libero"}
    assert len(libero_tasks) == 10  # 10 个基身份 × 5 个光照域（跨域配对）


def test_closed_loop_counts_statuses_and_retries_errors(tmp_path):
    outcomes = ["success", "failure", "infrastructure_error"]
    seen: list[str] = []

    def runner(entry, *, checkpoint, artifacts_dir, config):
        seen.append(entry["identity"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "frame.txt").write_text("kept")
        return {"status": outcomes[len(seen) % 3], "success": None, "seconds": 0.5}

    config = {
        "stage": "S2",
        "platform": "libero",
        "arm": "D",
        "seed": 101,
        "eval": {"planned": 100},
    }
    summary = run_closed_loop(
        config, eval_list=EVAL100, checkpoint="ckpt", output=tmp_path / "eval", runner=runner
    )
    rows = read_jsonl(tmp_path / "eval" / "eval_results.jsonl")
    assert len(rows) == 100
    assert summary["planned"] == 100
    assert summary["completed"] + summary["errors"] == 100
    assert summary["completed"] == summary["successes"] + summary["failures"]
    assert summary["failures"] == 34 and summary["errors"] == 33
    assert all(row["success"] is False for row in rows if row["status"] == "failure")
    assert all(row["success"] is None for row in rows if row["status"] == "infrastructure_error")
    assert all("frame.txt" in artifact for row in rows for artifact in row["artifacts"])

    # 只补基础设施错误：已结算身份复用，不重复执行
    rerun_seen: list[str] = []

    def rerun(entry, *, checkpoint, artifacts_dir, config):
        rerun_seen.append(entry["identity"])
        return {"status": "success", "success": True, "seconds": 0.1}

    second = run_closed_loop(
        config, eval_list=EVAL100, checkpoint="ckpt", output=tmp_path / "eval", runner=rerun
    )
    assert len(rerun_seen) == summary["errors"]
    assert second["errors"] == 0
    assert second["completed"] == 100
    assert second["successes"] > summary["successes"]
    # 同一身份只按最后一行结算：行数增加，但分母仍是 100
    assert len(read_jsonl(tmp_path / "eval" / "eval_results.jsonl")) == 100 + summary["errors"]
    progress = json.loads((tmp_path / "eval" / "progress.json").read_text())
    assert progress["phase"] == "eval" and progress["eval"]["planned"] == 100
    assert (tmp_path / "eval" / "artifacts").exists()


def test_summarize_eval_results_uses_last_row_per_identity(tmp_path):
    path = tmp_path / "eval_results.jsonl"
    rows = [
        {"identity": "a", "platform": "libero", "status": "infrastructure_error"},
        {"identity": "a", "platform": "libero", "status": "success"},
        {"identity": "b", "platform": "robotwin", "status": "failure"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = summarize_eval_results(path, planned=3)
    assert summary["completed"] == 2 and summary["successes"] == 1
    assert summary["failures"] == 1 and summary["errors"] == 0
    assert summary["libero"]["successes"] == 1 and summary["robotwin"]["failures"] == 1


def test_config_include_merge(tmp_path):
    (tmp_path / "base.json").write_text(
        json.dumps(
            {
                "stage": "S2",
                "platform": "libero",
                "arm": "D",
                "data": {"root": "/x", "window": {"burn_in": 16}},
                "model": {},
                "loss": {},
                "optim": {},
                "monitor": {},
            }
        )
    )
    (tmp_path / "child.json").write_text(
        json.dumps({"include": ["base.json"], "arm": "C", "data": {"window": {"gradient": 32}}})
    )
    config = load_config(tmp_path / "child.json", data_root="/override")
    assert config["arm"] == "C"
    assert config["data"]["window"] == {"burn_in": 16, "gradient": 32}
    assert config["data"]["root"] == "/override"


def test_repo_configs_load_and_match_runs():
    for platform in ("libero", "robotwin"):
        for arm in ("D", "C", "B", "A"):
            path = REPO_ROOT / f"configs/n04/v1_1/runs/{platform}_{arm}.json"
            config = load_config(path)
            assert config["platform"] == platform and config["arm"] == arm
            assert config["seed"] == 101
            assert config["optim"]["steps"] == 100000
            assert config["optim"]["warmup_steps"] == 2000
            assert config["loss"]["weights"]["action"] == 1.0
    smoke = load_config(REPO_ROOT / "configs/n04/v1_1/smoke_profile.json")
    assert 20 <= smoke["optim"]["steps"] <= 50
    assert smoke["data"]["dev_windows"] == 1


def test_window_candidates_cover_all_train_sources(dataset):
    index = V11DataIndex(root=dataset["root"], split_file=dataset["split_file"])
    candidates = window_candidates(
        index, platform="libero", split="train", burn_in=16, gradient=32, stride=16
    )
    assert {entry.relpath for entry, _ in candidates} == {
        "suite_a/episode_000001/variant_00.npz",
        "suite_a/episode_000002/variant_00.npz",
    }
    assert len(candidates) == 6  # 每条 80 帧序列 3 个窗口
    report = index.report()
    assert report["entries"]["libero/train"] == 2 and report["entries"]["libero/dev"] == 1


def test_eval_list_rejects_incomplete_domain_coverage(tmp_path):
    payload = json.loads(EVAL100.read_text())
    payload["entries"] = [e for e in payload["entries"] if e["lighting_domain"] != "Over"] + [
        {
            "platform": "libero",
            "task": "x",
            "initial_state": 0,
            "env_seed": 0,
            "lighting_domain": "Over",
            "identity": "extra",
        }
    ]
    broken = tmp_path / "eval100.json"
    broken.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="期望 10"):
        load_eval_list(broken, domains=("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver"))


def test_closed_loop_requires_wired_executor():
    # 没有仿真器接线时必须明确拒绝，不能伪造成功率
    with pytest.raises(SystemExit, match="closed_loop.command"):
        SubprocessEpisodeRunner({})


def test_action_normalizer_supports_real_stat_schemas(tmp_path):
    """复刻远端两份真统计文件结构：单具名条目 + action 单数 / norm_stats + actions 复数。"""
    libero = {
        "openpi_pi05_libero_franka": {
            "state": {
                "mean": [0.5] * 8,
                "std": [0.25] * 8,
                "q01": [0.0] * 8,
                "q99": [1.0] * 8,
            },
            "action": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "q01": [-1.0] * 7,
                "q99": [1.0] * 7,
            },
        }
    }
    robotwin = {
        "norm_stats": {
            "actions": {
                "mean": [0.0] * 14,
                "std": [2.0] * 14,
                "min": [-3.0] * 14,
                "max": [3.0] * 14,
            },
            "state": {"mean": [1.0] * 14, "std": [4.0] * 14, "min": [-3.0] * 14, "max": [3.0] * 14},
        }
    }
    libero_path = tmp_path / "dataset_statistics.json"
    robotwin_path = tmp_path / "policy_norm_stats.json"
    libero_path.write_text(json.dumps(libero))
    robotwin_path.write_text(json.dumps(robotwin))

    libero_normalizer = ActionNormalizer.from_file("libero", libero_path)
    assert libero_normalizer.entry_name == "openpi_pi05_libero_franka"
    assert libero_normalizer.action_key == "action"
    assert libero_normalizer.normalize_state(torch.zeros(1, 8)).shape == (1, 8)
    assert torch.allclose(
        libero_normalizer.normalize_actions(torch.tensor([[0.0] * 7])),
        torch.zeros(1, 7),
        atol=1e-6,
    )

    robotwin_normalizer = ActionNormalizer.from_file("robotwin", robotwin_path)
    assert robotwin_normalizer.action_key == "actions"
    assert torch.allclose(
        robotwin_normalizer.normalize_actions(torch.full((1, 14), 2.0)), torch.ones(1, 14)
    )
    assert torch.allclose(
        robotwin_normalizer.normalize_state(torch.full((1, 14), 5.0)), torch.ones(1, 14)
    )

    # 缺键仍然报错，不猜也不补零
    with pytest.raises(KeyError, match="state"):
        ActionNormalizer(
            "robotwin", {"norm_stats": {"actions": robotwin["norm_stats"]["actions"]}}
        ).normalize_state(torch.zeros(1, 14))
    with pytest.raises(KeyError, match="action"):
        ActionNormalizer(
            "libero", {"only": {"state": libero["openpi_pi05_libero_franka"]["state"]}}
        ).normalize_actions(torch.zeros(1, 7))


def test_forecast_target_is_stop_gradient_and_gradients_route(dataset, tmp_path):
    """目标侧 stop-grad：latent 不带梯度、不产生 encoder 梯度；三项各自到达该到的模块。"""
    config = _config(dataset, tmp_path)
    config["loss"]["forecast_horizons"] = [1, 2]
    encoder = FakeForecastEncoder(emit_losses=True)
    trainer = V11Trainer(
        config,
        seed=101,
        run_dir=tmp_path / "run",
        device="cpu",
        encoder=encoder,
        backend=FakeBackend(),
    )
    entry, start = trainer.train_candidates[0]
    window = load_window_batch(
        entry, start=start, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )

    frame = trainer._frame(window, 20)
    targets = trainer._target_features(frame)
    assert targets is not None and set(targets) == set(window.camera_ids)
    for spec in targets.values():
        assert spec["latent"].requires_grad is False
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._target_features(frame)
    assert all(parameter.grad is None for _, parameter in trainer._trainable_parameters())

    def isolate(term: str):
        weights = {key: (1.0 if key == term else 0.0) for key in trainer.loss_weights}
        trainer.loss_weights = weights
        trainer.optimizer.zero_grad(set_to_none=True)
        loss, info = trainer._forward_window(window, train=True, backward_weight=1.0)
        assert loss is not None
        hits = {
            name
            for name, parameter in trainer._trainable_parameters()
            if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        }
        return info, hits

    action_info, action_hits = isolate("action")
    assert action_info["terms"]["action"] is not None
    assert any("encoder.proj" in name for name in action_hits)
    assert any("backend" in name for name in action_hits)

    corr_info, corr_hits = isolate("correspondence")
    assert corr_info["terms"]["correspondence"] is not None
    assert any("encoder.proj" in name for name in corr_hits)

    forecast_info, forecast_hits = isolate("forecast")
    assert forecast_info["terms"]["forecast"] is not None
    assert forecast_info["forecast"]["settled"] > 0  # 真的结算了 anchor，不是空跑
    assert any("encoder.state_stack" in name for name in forecast_hits)  # 预测侧
    assert not any("encoder.measurement" in name for name in forecast_hits)  # 目标侧 stop-grad
    assert any("encoder.proj" in name for name in forecast_hits)
    assert any("encoder.quality_head" in name for name in forecast_hits)
    trainer.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)


def test_forecast_roll_uses_only_t0_proprio_and_differs_per_horizon(dataset, tmp_path):
    """roll 只用 t0 本体（改未来本体不影响），且不同 horizon 的预测确实不同。"""
    config = _config(dataset, tmp_path)
    config["loss"]["forecast_horizons"] = [1, 2]
    # 锚点只落在 16/24/32/40（stride=8）：这样"改 41 帧之后的未来数据"不触及任何 t0
    config["loss"]["forecast_anchor_stride"] = 8
    encoder = FakeForecastEncoder()
    trainer = V11Trainer(
        config,
        seed=101,
        run_dir=tmp_path / "run",
        device="cpu",
        encoder=encoder,
        backend=FakeBackend(),
    )
    entry, start = trainer.train_candidates[0]
    window = load_window_batch(
        entry, start=start, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )

    def run(active):
        encoder.state_stack.calls = 0
        encoder.state_stack.proprio_seen = []
        _, info = trainer._forward_window(active, train=False)
        return info, [t.clone() for t in encoder.state_stack.proprio_seen]

    info, seen = run(window)
    assert info["forecast"]["settled"] > 0
    assert encoder.state_stack.calls > 0
    # 每个 anchor 的 roll 次数 = horizon（渐进推进只多 4 次 predict 的说法在 k≤2 时即 k 次）
    assert all(t.shape[-1] == window.proprio.shape[-1] * window.proprio.shape[1] for t in seen)
    # 不同 horizon 的预测不同：逐 horizon 仪表（r_rms）应有差异
    diagnostics = info["forecast"]["diagnostics"]
    assert set(diagnostics) == {"1", "2"}
    assert diagnostics["1"]["forecast_quad"] != diagnostics["2"]["forecast_quad"]

    tampered = copy.deepcopy(window)
    tampered.proprio[41:] = 0.0
    tampered.targets[41:] = 0.0
    _, seen_after = run(tampered)
    assert len(seen_after) == len(seen)
    for before, after in zip(seen, seen_after, strict=True):
        assert torch.allclose(before, after)  # settle 前的 roll 只吃 t0 及更早的本体


def test_forecast_engine_roll_counts_and_graph_release():
    """单元级：roll 次数 = k、prediction 从 posterior 起、settle 后不留需要梯度的张量。"""
    from owac.frameworks.n04.state_estimator import SupportState
    from owac.training.n04_v11 import ForecastEngine

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.full((1,), 0.25))
            self.calls = 0

        def predict(self, state, proprio_context):
            self.calls += 1
            delta = torch.zeros_like(state.mode_mean)
            delta[..., 0] = self.bias
            return type(state)(
                xy=state.xy,
                ids=state.ids,
                age=state.age,
                valid=state.valid,
                fresh=state.fresh,
                mode_mean=state.mode_mean + delta,
                mode_cov=state.mode_cov,
                mode_weights=state.mode_weights,
                sensor_size_hw=state.sensor_size_hw,
            )

    estimator = Recorder()
    n, obs = 3, 4
    posterior = SupportState(
        xy=torch.tensor([[[1.0, 1.0], [3.0, 2.0], [5.0, 3.0]]]),
        ids=torch.arange(n)[None],
        age=torch.zeros(1, n, dtype=torch.long),
        valid=torch.ones(1, n, dtype=torch.bool),
        fresh=torch.zeros(1, n, dtype=torch.bool),
        mode_mean=torch.zeros(1, n, 2, 2 * obs),
        mode_cov=tuple(torch.ones(1, n, 2, obs) for _ in range(3)),
        mode_weights=torch.full((1, n, 2), 0.5),
        sensor_size_hw=(8, 8),
    )
    engine = ForecastEngine(estimator, (2,), total_frames=8, weight=0.0)
    engine.add(
        1,
        "cam",
        posterior,
        torch.ones(1, n, obs),
        torch.zeros(1, 12),
        size_hw=(8, 8),
        segment_last=7,
    )
    assert engine.created == 1
    anchor = engine.anchors[0]
    assert anchor.prediction is None
    assert anchor.posterior.mode_mean.requires_grad is False
    assert estimator.calls == 0  # 建锚点不 roll
    target = torch.zeros(1, n, obs)
    loss = engine.settle(anchor, target)
    assert loss is not None and torch.isfinite(loss)
    assert estimator.calls == 2  # 恰好 roll k 步
    assert anchor.prediction is not None and anchor.prediction.mode_mean.requires_grad is False
    assert engine.report()["settled"] == 1


def test_forecast_diagnostics_are_readonly_and_loss_is_bit_identical(dataset, tmp_path):
    """仪表只读：开启后 forecast 项与直接调 forecast_loss 的结果逐位相同。"""
    from owac.frameworks.n04.evidence_update import forecast_loss
    from owac.frameworks.n04.state_estimator import SupportState
    from owac.training.n04_v11 import ForecastEngine

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.full((1,), 0.25))

        def predict(self, state, proprio_context):
            delta = torch.zeros_like(state.mode_mean)
            delta[..., 0] = self.bias
            return type(state)(
                xy=state.xy,
                ids=state.ids,
                age=state.age,
                valid=state.valid,
                fresh=state.fresh,
                mode_mean=state.mode_mean + delta,
                mode_cov=state.mode_cov,
                mode_weights=state.mode_weights,
                sensor_size_hw=state.sensor_size_hw,
            )

    estimator = Recorder()
    n, obs = 3, 4
    posterior = SupportState(
        xy=torch.tensor([[[1.0, 1.0], [3.0, 2.0], [5.0, 3.0]]]),
        ids=torch.arange(n)[None],
        age=torch.zeros(1, n, dtype=torch.long),
        valid=torch.ones(1, n, dtype=torch.bool),
        fresh=torch.zeros(1, n, dtype=torch.bool),
        mode_mean=torch.zeros(1, n, 2, 2 * obs),
        mode_cov=tuple(torch.ones(1, n, 2, obs) for _ in range(3)),
        mode_weights=torch.full((1, n, 2), 0.5),
        sensor_size_hw=(8, 8),
    )
    target = torch.full((1, n, obs), 0.5)
    R = torch.ones(1, n, obs)

    reference_engine = ForecastEngine(estimator, (2,), total_frames=8, weight=0.0)
    reference_engine.add(1, "cam", posterior, R, torch.zeros(1, 12), size_hw=(8, 8), segment_last=7)
    reference_anchor = reference_engine.anchors[0]
    prediction = reference_anchor.posterior
    for _ in range(2):
        prediction = estimator.predict(prediction, reference_anchor.proprio)
    expected = forecast_loss(target, prediction, R, valid=reference_anchor.valid)

    engine = ForecastEngine(estimator, (2,), total_frames=8, weight=0.0)
    engine.add(1, "cam", posterior, R, torch.zeros(1, 12), size_hw=(8, 8), segment_last=7)
    instrumented = engine.settle(engine.anchors[0], target)
    assert instrumented is not None and torch.equal(instrumented, expected.detach())

    diagnostics = engine.report()["diagnostics"]["2"]
    for field in (
        "forecast_quad",
        "forecast_logdet",
        "forecast_mode_w0",
        "forecast_mode_w1",
        "forecast_S_mean",
        "forecast_S_min",
        "forecast_S_max",
        "forecast_R_mean",
        "forecast_R_min",
        "forecast_R_max",
    ):
        assert field in diagnostics
    # 判据字段：quad 是对维度求和量；d=4 的小替身下 quad>0，S 谱健康（min<=mean<=max）
    assert diagnostics["forecast_quad"] > 0
    assert (
        diagnostics["forecast_S_min"]
        <= diagnostics["forecast_S_mean"]
        <= diagnostics["forecast_S_max"]
    )
    assert diagnostics["forecast_R_min"] > 0


def test_metrics_and_progress_carry_loss_definition_and_diagnostics(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    trainer.config["loss_definition"] = "v1_1_forecast_perdim_v2"
    info = trainer.train_step(0)
    trainer._write_monitor(
        trainer.run_dir / "metrics.jsonl", trainer.run_dir / "progress.json", 4, info
    )
    row = read_jsonl(trainer.run_dir / "metrics.jsonl")[-1]
    assert row["forecast_diag"] is None or isinstance(row["forecast_diag"], dict)
    progress = json.loads((trainer.run_dir / "progress.json").read_text())
    assert progress["loss_definition"] == "v1_1_forecast_perdim_v2"
    # NaN 一律写 null，不写 0
    for value in (row.get("forecast_diag") or {}).values():
        assert value is None or math.isfinite(value)


def test_resume_notes_written_on_resume(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    trainer.train_step(0)
    checkpoint = trainer.save_checkpoint(tmp_path / "ckpt.pt")
    resumed = _trainer(dataset, tmp_path / "again")
    resumed.config["loss_definition"] = "v1_1_forecast_perdim_v2"
    resumed.run(steps=1, resume=checkpoint)
    notes = (resumed.run_dir / "resume-notes.md").read_text()
    assert "v1_1_forecast_perdim_v2" in notes and "不可直接比较" in notes


def test_single_scaling_libero_raw_units_and_model_input(dataset, tmp_path):
    """缩放只在编码器 InputScale 一处：窗口 raw 是 uint8 域，模型输入落在 [0,1]。"""
    from owac.frameworks.n04.contracts import input_scale_for
    from owac.frameworks.n04.encoder_v11 import default_measurement_config
    from owac.frameworks.n04.measurement import preprocess_frames

    index = V11DataIndex(root=dataset["root"], split_file=dataset["split_file"])
    entry = index.entries("libero", "train")[0]
    window = load_window_batch(
        entry, start=0, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )
    raw = window.raw["agentview"]
    assert float(raw.max()) > 1.0  # 原始 uint8 域，没被提前 /255（否则模型侧会再缩一次）
    assert float(raw.min()) >= 0.0
    assert input_scale_for("libero_uint8").factor == pytest.approx(1.0 / 255.0)

    trainer = _trainer(dataset, tmp_path)
    frame = trainer._frame(window, 0)
    batch = preprocess_frames(trainer._contract_frames(frame), default_measurement_config())
    assert float(batch.values.min()) >= 0.0 and float(batch.values.max()) <= 1.0
    # pad 区域是 0，均值会被稀释；用动态范围判断"不是被压成近常数"
    assert float(batch.values.max()) - float(batch.values.min()) > 0.2
    assert float(batch.values.max()) > 0.2


def test_saturation_mask_in_raw_units_and_encoder_domain(dataset, tmp_path):
    """LIBERO 饱和在原始域（≥255）可非空；RoboTwin 仍是 identity（HDR 原值 >1）。"""
    from owac.frameworks.n04.encoder_v11 import default_measurement_config
    from owac.frameworks.n04.measurement import preprocess_frames
    from owac.raw.n04_v11_data import saturation_mask

    image = torch.full((3, 4, 4), 100.0)
    assert not bool(saturation_mask(image, "libero").any())
    image[0, 0, 0] = 255.0
    mask = saturation_mask(image, "libero")
    assert mask is not None and bool(mask.any()) and mask.shape == (1, 4, 4)

    trainer = _trainer(dataset, tmp_path)
    index = V11DataIndex(root=dataset["root"], split_file=dataset["split_file"])
    entry = index.entries("libero", "train")[0]
    window = load_window_batch(
        entry, start=0, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )
    window.raw["agentview"][0, 0, 0, 0] = 255.0
    frame = trainer._frame(window, 0)
    masks = trainer._saturation(window, 0)
    assert masks is not None and bool(masks[0].any())  # 原始域的饱和确实可能非空
    batch = preprocess_frames(trainer._contract_frames(frame), default_measurement_config())
    assert bool(batch.saturated.any())  # 缩放后编码器侧的饱和平面同样非空

    robot = V11DataIndex(root=dataset["root"], split_file=dataset["split_file"])
    robot_entry = robot.entries("robotwin", "train")[0]
    robot_window = load_window_batch(
        robot_entry, start=0, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )
    assert float(robot_window.raw["head"].max()) > 1.0  # RoboTwin identity


def test_resume_allows_deliberate_loss_definition_change(dataset, tmp_path):
    """结构与优化器必须一致；仅损失定义变更允许 resume，并把 from/to 写进 notes 与 report。"""
    trainer = _trainer(dataset, tmp_path)
    trainer.config["loss_definition"] = "v1_1_forecast_perdim_v1"
    trainer.train_step(0)
    checkpoint = trainer.save_checkpoint(tmp_path / "ckpt.pt")

    resumed = _trainer(dataset, tmp_path / "again")
    resumed.config["loss_definition"] = "v1_1_forecast_perdim_v2"
    resumed.config["loss"] = {
        "weights": {"action": 1.0, "correspondence": 0.1, "forecast": 0.1, "representation": 0.01},
        "forecast": True,
        "forecast_horizons": [1],
    }
    report = resumed.run(steps=1, resume=checkpoint)
    assert resumed.loss_definition_change is not None
    change = resumed.loss_definition_change
    assert change["from_definition"] == "v1_1_forecast_perdim_v1"
    assert change["to_definition"] == "v1_1_forecast_perdim_v2"
    notes = (resumed.run_dir / "resume-notes.md").read_text()
    assert "v1_1_forecast_perdim_v1" in notes and "v1_1_forecast_perdim_v2" in notes
    assert report["loss_definition_change"] == change
    assert report["resumed_from"]["step"] == 1


def test_observation_vicreg_rejects_query_only_diversity():
    from owac.training.n04_v11 import observation_vicreg_terms

    q = torch.cat((torch.eye(32), -torch.eye(32))) * math.sqrt(32)
    x = q[None].expand(4, -1, -1).clone().requires_grad_()
    old = vicreg_terms(x)
    assert float(old["variance"]) < 0.01 and float(old["covariance"]) < 1e-5
    repaired = observation_vicreg_terms(x)
    assert float(repaired["variance"]) > 0.98
    assert float(repaired["covariance"]) == 0
    torch.manual_seed(41)
    varied = torch.randn(5, 3, 7, requires_grad=True)
    actual = observation_vicreg_terms(varied)
    explicit = []
    for query in range(3):
        centered = varied[:, query] - varied[:, query].mean(0)
        cov = centered.T @ centered / 4
        explicit.append((cov - torch.diag_embed(cov.diag())).square().sum() / 7)
    assert torch.allclose(actual["covariance"], torch.stack(explicit).mean(), atol=1e-6)
    sum(actual.values()).backward()
    assert varied.grad is not None and float(varied.grad.abs().sum()) > 0


def test_independent_scene_gradient_cache_matches_joint_backward():
    from owac.training.n04_v11 import (
        cache_representation_gradients,
        observation_vicreg_terms,
        summarize_prefix,
    )

    torch.manual_seed(67)
    model = nn.Linear(5, 7)
    raw = torch.randn(4, 2, 6, 5)
    observations = torch.stack([summarize_prefix(model(x).reshape(1, 12, 7), 2) for x in raw])
    terms = observation_vicreg_terms(observations)
    joint = 25 * terms["variance"] + terms["covariance"]
    (0.01 * joint).backward()
    reference = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    cache = cache_representation_gradients(observations.detach())
    for index, x in enumerate(raw):
        live = summarize_prefix(model(x).reshape(1, 12, 7), 2)
        replay = cache.replay(index, live)
        assert torch.allclose(replay, joint.detach(), atol=1e-5)
        (0.01 * replay / 4).backward()
    for parameter, expected in zip(model.parameters(), reference, strict=True):
        torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-5)


def test_independent_scene_cache_rejects_positional_diversity_and_mismatched_replay():
    from owac.training.n04_v11 import cache_representation_gradients, summarize_prefix

    q = torch.randn(1, 12, 9) * 50
    scenes = torch.stack([summarize_prefix(q, 2)] * 4)
    cache = cache_representation_gradients(scenes)
    assert cache.diagnostics["variance"] == pytest.approx(0.99)
    assert cache.diagnostics["std"] == pytest.approx(0)
    with pytest.raises(RuntimeError, match="replay"):
        cache.replay(0, scenes[0] + 1.0)


def test_representation_prepass_is_causal_and_restores_rng(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    plan = [trainer.train_candidates[0], trainer.train_candidates[-1]]
    windows = list(trainer._training_windows(plan))
    original = windows[0]
    trainer._seed_window(3, 0)
    expected = trainer._representation_observation(original)
    changed = replace(
        original,
        raw={k: v.clone() for k, v in original.raw.items()},
        proprio=original.proprio.clone(),
        targets=original.targets.clone(),
    )
    for values in changed.raw.values():
        values[17:] = 0
    changed.proprio[17:] = float("nan")
    changed.targets[:] = float("nan")
    trainer._seed_window(3, 0)
    actual = trainer._representation_observation(changed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    trainer.representation_mode = "independent_windows"
    before = torch.get_rng_state().clone()
    trainer._representation_batch(plan, 3)
    assert torch.equal(before, torch.get_rng_state())


def test_independent_representation_reaches_encoder_in_full_train_step(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    trainer.representation_mode = "independent_windows"
    trainer.loss_weights = {name: float(name == "representation") for name in trainer.loss_weights}
    # Different source sequences rather than two offsets from one video.
    from owac.training import n04_v11

    plan = []
    identities = set()
    for candidate in trainer.train_candidates:
        if candidate[0].source_id not in identities:
            identities.add(candidate[0].source_id)
            plan.append(candidate)
        if len(plan) == 2:
            break
    original = n04_v11.sample_windows
    n04_v11.sample_windows = lambda *args, **kwargs: plan
    before = trainer.encoder.proj.weight.detach().clone()
    try:
        report = trainer.train_step(0)
    finally:
        n04_v11.sample_windows = original
    assert not torch.equal(before, trainer.encoder.proj.weight)
    assert report["representation_diag"]["windows"] == 2
    assert report["terms"]["representation"] > 0


def test_independent_objective_cannot_be_silently_skipped(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    trainer.representation_mode = "independent_windows"
    window = next(trainer._training_windows(trainer.train_candidates[:1]))
    with pytest.raises(ValueError, match="batch gradient cache"):
        trainer._forward_window(window, train=True)


def test_independent_sampler_repairs_duplicate_source_batch_deterministically(
    dataset, tmp_path, monkeypatch
):
    from owac.training import n04_v11

    trainer = _trainer(dataset, tmp_path)
    trainer.representation_mode = "independent_windows"
    duplicate = trainer.train_candidates[0]
    monkeypatch.setattr(n04_v11, "sample_windows", lambda *args, **kwargs: [duplicate, duplicate])
    first = trainer._training_plan(3)
    assert first == trainer._training_plan(3)
    assert len(first) == 2 and len({entry.source_id for entry, _ in first}) == 2
    assert all(spec in trainer.train_candidates for spec in first)


def test_compiled_autograd_allows_exact_retained_graph_fallback():
    from torch._functorch import config as aot_config

    from owac.training.n04_v11 import configure_compiled_autograd

    previous = (aot_config.donated_buffer, aot_config.backward_pass_autocast)
    try:
        aot_config.donated_buffer = True
        configure_compiled_autograd()

        class SavedActivation(nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = nn.Parameter(torch.zeros(3, 3))

            def forward(self, x):
                return x.relu() + self.offset

        eager = SavedActivation()
        compiled = torch.compile(copy.deepcopy(eager), fullgraph=True)
        reference_input = torch.randn(3, 3, requires_grad=True)
        compiled_input = reference_input.detach().clone().requires_grad_(True)
        y, z = eager(reference_input), compiled(compiled_input)
        for index, objective in enumerate(
            (lambda value: value.sum(), lambda value: value.square().mean())
        ):
            expected = torch.autograd.grad(
                objective(y), (reference_input, eager.offset), retain_graph=index == 0
            )
            actual = torch.autograd.grad(
                objective(z), (compiled_input, compiled.offset), retain_graph=index == 0
            )
            for got, wanted in zip(actual, expected, strict=True):
                torch.testing.assert_close(got, wanted)
    finally:
        aot_config.donated_buffer, aot_config.backward_pass_autocast = previous


def test_fixed_auxiliary_validation_is_written_without_updates(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    trainer.monitor["auxiliary_dev"] = True
    # The fixture supplies one dev sequence; use two distinct observation starts.
    entry, _ = trainer.dev_windows[0]
    trainer.dev_windows = [(entry, 0), (entry, 16)]
    before = {
        name: parameter.detach().clone() for name, parameter in trainer._trainable_parameters()
    }
    trainer.evaluate_dev_action_loss()
    auxiliary = json.loads((trainer.run_dir / "auxiliary-validation.json").read_text())
    assert auxiliary["windows"] == 2 and auxiliary["representation"]["windows"] == 2
    history = (trainer.run_dir / "auxiliary-validation.jsonl").read_text().splitlines()
    assert len(history) == 1 and json.loads(history[0]) == auxiliary
    for name, parameter in trainer._trainable_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_exact_fallback_restores_compiled_routines_and_preserves_gradients(dataset, tmp_path):
    from owac.training.n04_v11 import configure_compiled_autograd

    reference = _trainer(dataset, tmp_path / "reference")
    actual = _trainer(dataset, tmp_path / "actual")
    actual.encoder.load_state_dict(reference.encoder.state_dict())
    actual.backend.load_state_dict(reference.backend.state_dict())
    window = next(reference._training_windows(reference.train_candidates[:1]))
    torch.manual_seed(41)
    expected, _ = reference._forward_window(window, train=True, _fast_counts=False)
    eager = actual.encoder.proj.forward
    configure_compiled_autograd()
    compiled = torch.compile(eager, fullgraph=True)
    actual.encoder.proj.forward = compiled
    actual._eager_methods = [(actual.encoder.proj, "forward", eager)]
    actual.performance["verified_joint_backward"] = True
    torch.manual_seed(41)
    got, info = actual._forward_window(window, train=True)
    assert info["valid_count_fallback"] and info["gradient_eager_fallback"]
    assert actual.encoder.proj.forward is compiled
    torch.testing.assert_close(got, expected)
    for (_, want), (_, parameter) in zip(
        reference._trainable_parameters(), actual._trainable_parameters(), strict=True
    ):
        if want.grad is not None:
            torch.testing.assert_close(parameter.grad, want.grad)


def test_known_compiled_backward_error_retries_complete_native_window(dataset, tmp_path):
    trainer = _trainer(dataset, tmp_path)
    window = next(trainer._training_windows(trainer.train_candidates[:1]))
    eager = trainer.encoder.proj.forward

    class BrokenSavedBuffer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            raise RuntimeError("saved tensor modified by an inplace operation")

    def compiled(x):
        return BrokenSavedBuffer.apply(eager(x))

    trainer.encoder.proj.forward = compiled
    trainer._eager_methods = [(trainer.encoder.proj, "forward", eager)]
    trainer.performance["verified_joint_backward"] = True
    _, info = trainer._forward_window(window, train=True)
    assert info["gradient_eager_fallback"] and "inplace" in info["compiled_backward_retry"]
    assert trainer.encoder.proj.forward is compiled
    assert (trainer.run_dir / "compiled-backward-retries.jsonl").is_file()
    assert all(bool(torch.isfinite(p.grad).all()) for p in trainer.encoder.parameters())


def test_forecast_cross_segment_gradients_match_reference(dataset, tmp_path):
    config = _config(dataset, tmp_path)
    config["loss"]["forecast_horizons"] = [1, 2, 4]
    trainer = V11Trainer(
        config,
        seed=101,
        run_dir=tmp_path / "r",
        device="cpu",
        encoder=FakeForecastEncoder(),
        backend=FakeBackend(),
    )
    entry, start = trainer.train_candidates[0]
    window = load_window_batch(
        entry, start=start, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )

    def run(single):
        trainer.optimizer.zero_grad(set_to_none=True)
        _, info = trainer._forward_window(window, train=True, single_backward=single)
        return info, {
            n: p.grad.clone() for n, p in trainer._trainable_parameters() if p.grad is not None
        }

    reference, ref_grads = run(True)
    segmented, grads = run(False)
    assert segmented["forecast"]["settled"] == (31 + 30 + 28) * 2
    assert segmented["valid_loss_counts"]["forecast"] == segmented["forecast"]["settled"]
    assert not segmented["forecast"]["skip_reasons"].get("segment_boundary", 0)
    assert segmented["terms"] == reference["terms"]
    assert grads.keys() == ref_grads.keys()
    for name in grads:
        assert torch.allclose(grads[name], ref_grads[name], atol=1e-5, rtol=1e-5), name


def test_user_stop_marker_blocks_all_optimizer_entrypoints(dataset, tmp_path, monkeypatch):
    trainer = _trainer(dataset, tmp_path)
    root = tmp_path / "remote-root"
    root.mkdir()
    (root / "TRAINING_STOPPED_BY_USER.json").write_text("{}")
    monkeypatch.setenv("N04V11_ROOT", str(root))
    for action in (lambda: trainer.train_step(0), trainer.profile_window, trainer.run):
        with pytest.raises(RuntimeError, match="Training stopped by user"):
            action()
    assert trainer.step == 0 and not trainer.optimizer.state


def test_checkpoint_rejects_old_semantics_and_restores_rng(dataset, tmp_path):
    import random

    from owac.frameworks.n04.compatibility import IMPLEMENTATION_VERSION

    trainer = _trainer(dataset, tmp_path)
    checkpoint = tmp_path / "repair.pt"
    trainer.save_checkpoint(checkpoint)
    expected = (random.random(), np.random.random(), torch.rand(4))
    resumed = _trainer(dataset, tmp_path)
    payload = resumed.load_checkpoint(checkpoint)
    assert payload["implementation_version"] == IMPLEMENTATION_VERSION
    assert random.random() == expected[0] and np.random.random() == expected[1]
    assert torch.equal(torch.rand(4), expected[2])
    payload.pop("implementation_version")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="implementation_version"):
        resumed.load_checkpoint(checkpoint)


def test_reused_bfloat16_target_is_sampled_in_fp32(dataset, tmp_path):
    trainer = V11Trainer(
        _config(dataset, tmp_path),
        seed=101,
        run_dir=tmp_path / "r",
        device="cpu",
        encoder=FakeForecastEncoder(),
        backend=FakeBackend(),
    )
    entry, start = trainer.train_candidates[0]
    window = load_window_batch(
        entry, start=start, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )
    spec = trainer._target_features(trainer._frame(window, 20))[window.camera_ids[0]]
    spec["latent"] = spec["latent"].bfloat16()
    sample = trainer._sample_latent(spec, torch.tensor([[[2.0, 2.0]]]))
    assert sample.dtype == torch.float32 and bool(torch.isfinite(sample).all())


@pytest.mark.parametrize("emit_losses", [True, False])
def test_verified_joint_backward_matches_reference_and_falls_back(dataset, tmp_path, emit_losses):
    trainer = V11Trainer(
        _config(dataset, tmp_path),
        seed=101,
        run_dir=tmp_path / "run",
        device="cpu",
        encoder=FakeForecastEncoder(emit_losses=emit_losses),
        backend=FakeBackend(),
    )
    entry, start = trainer.train_candidates[0]
    window = load_window_batch(
        entry, start=start, burn_in=16, gradient=32, supervision_points=4, proprio_history=2
    )

    def run(fast):
        trainer.optimizer.zero_grad(set_to_none=True)
        _, info = trainer._forward_window(window, train=True, _fast_counts=fast)
        return info, {
            n: p.grad.clone() for n, p in trainer._trainable_parameters() if p.grad is not None
        }

    slow, gs = run(False)
    fast, gf = run(True)
    assert slow["terms"] == fast["terms"]
    assert gf.keys() == gs.keys()
    for name in gs:
        assert torch.allclose(gs[name], gf[name], rtol=2e-5, atol=1e-5), name
    if not emit_losses:
        assert fast["valid_count_fallback"] and not fast["gradient_fast_path"]
    else:
        assert fast["gradient_fast_path"]


@pytest.mark.parametrize("platform", ("libero", "robotwin"))
def test_codex_masked_gpu_launch_preserves_training_contract(platform):
    """Each supervisor exposes one physical card as logical cuda:0; compilation is math only."""
    from owac.training.n04_v11 import loss_fingerprint, structural_fingerprint

    root = REPO_ROOT / "configs/n04/v1_1"
    eager = load_config(root / f"codex_{platform}_D.json")
    compiled = load_config(root / f"codex_compiled_{platform}.json")
    assert eager["device"] == compiled["device"] == "cuda:0"
    assert eager["platform"] == platform
    assert eager["owner"] == compiled["owner"] == "Codex"
    assert structural_fingerprint(eager) == structural_fingerprint(compiled)
    assert loss_fingerprint(eager) == loss_fingerprint(compiled)
    assert eager["data"] == compiled["data"]
    assert eager["optim"]["batch"] == 8
    assert eager["data"]["window"]["gradient"] == 32
    assert eager["model"]["depths"] == [3, 3, 27, 3]
    assert eager["model"]["raft_iterations"] == 12
    assert eager["model"]["readout_blocks"] == 6
    assert eager["loss"]["forecast_horizons"] == [1, 2, 4]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_compiled_gradient_layout_allows_real_fused_update(device):
    from owac.training.n04_v11 import align_optimizer_gradients

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    parameter = nn.Parameter(torch.randn(8, 4, 3, 3, device=device))
    reference = nn.Parameter(parameter.detach().clone())
    values = torch.randn_like(parameter)
    parameter.grad = values.contiguous(memory_format=torch.channels_last)
    reference.grad = values.clone()
    optimizer = torch.optim.AdamW([parameter], fused=device == "cuda")
    expected = torch.optim.AdamW([reference], fused=False, foreach=False)
    assert parameter.stride() != parameter.grad.stride()
    assert align_optimizer_gradients(optimizer) == 1
    assert parameter.stride() == parameter.grad.stride()
    torch.testing.assert_close(parameter.grad, values, rtol=0, atol=0)
    optimizer.step()
    expected.step()
    torch.testing.assert_close(parameter, reference)
    assert align_optimizer_gradients(optimizer) == 0
