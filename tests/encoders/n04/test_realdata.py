"""N04 真实 RawVLA NPZ 路径测试。

真实数据与权重不在本机（文档记录部署在服务器 ``/workspace/owac/rawvla/``）。这里用
**测试自造的小 NPZ**（严格按作者键名、维度与对齐）验证：键解析、数值单位、窗口与监督
对齐、本体相对变换接入、动作缺失标记、训练前向/反向、CLI 报缺行为。这**不冒充**真实
数据验收，只证明代码路径在真实键与真实维度下可执行。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig, N04TrainConfig
from owac.frameworks.n04.proprio import (
    LIBERO_STATE_LAYOUT,
    ROBOTWIN_JOINT_LAYOUT,
    proprio_delta,
)
from owac.raw.rawvla_sequence_n04 import RawVLASequenceLoader
from owac.training.n04 import (
    DOCUMENTED_RAW_DATA_ROOT,
    RawVLAFrame,
    build_real_context,
    discover_npz,
    load_checkpoint,
    main,
    make_rawvla_batches,
    split_npz_paths,
    train,
)

HEIGHT = 16
WIDTH = 16
LENGTH = 12


def _libero_npz(
    path: Path, *, length: int = LENGTH, seed: int = 0, saturation: bool = True
) -> Path:
    """写一个键名/维度与 LIBERO 缓存一致的 NPZ。"""
    rng = np.random.default_rng(seed)
    agentview = rng.integers(0, 200, size=(length, HEIGHT, WIDTH, 3), dtype=np.uint8)
    wrist = rng.integers(0, 200, size=(length, HEIGHT, WIDTH, 3), dtype=np.uint8)
    if saturation:
        #: 制造一个真正饱和的像素：白电平 255 归一化后为 1.0。
        agentview[0, 0, 0, :] = 255
    states = np.zeros((length, 8), dtype=np.float32)
    states[:, 0] = np.linspace(0.0, 0.5, length)
    #: 两个轴的旋转以不同相位变化（且穿过零点）：同轴等比旋转下 `R_t R_{t-1}^T` 恰好
    #: 等于逐元素差，只有轴向量方向真的变化时才能暴露“盲减”错误。
    phase = np.linspace(0.0, math.pi, length)
    states[:, 3] = 0.8 * np.sin(phase)
    states[:, 5] = 0.6 * np.cos(phase)
    states[:, 6] = 0.4
    actions = rng.normal(0, 0.1, size=(length, 7)).astype(np.float32)
    np.savez(
        path,
        agentview_raw_uint8=agentview,
        wrist_raw_uint8=wrist,
        robot_states=states,
        actions=actions,
    )
    return path


def _robotwin_npz(path: Path, *, length: int = LENGTH, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    #: HDR-derived：量程超过 1.0，用来验证加载器不做 uint8/255 式的静默归一化。
    head = (rng.random((length, HEIGHT, WIDTH, 3), dtype=np.float32) * 4.0).astype(np.float32)
    left = (rng.random((length, HEIGHT, WIDTH, 3), dtype=np.float32) * 4.0).astype(np.float32)
    right = (rng.random((length, HEIGHT, WIDTH, 3), dtype=np.float32) * 4.0).astype(np.float32)
    actions = rng.normal(0, 0.2, size=(length + 60, 14)).astype(np.float32)
    np.savez(
        path,
        head_camera_raw_float32=head,
        left_camera_raw_float32=left,
        right_camera_raw_float32=right,
        actions=actions,
    )
    return path


def _config(*, camera: str = "agentview", predictor: str = "learned_proprio") -> N04Config:
    return N04Config(
        input_mode="rawvla_rgb3",
        measurement_size=(HEIGHT, WIDTH),
        channels=3,
        camera_ids=(camera,),
        geometry=N04GeometryConfig(feature_size=(HEIGHT, WIDTH), pixel_stride=1.0, max_flow=4.0),
        model=N04ModelConfig(
            name="n04-realnpz-test",
            width=8,
            blocks=1,
            scales=1,
            scale_strides=(1,),
            coarse_stride=4,
            hidden=16,
            camera_embedding=4,
            readout_tokens=8,
            readout_dim=16,
            predictor=predictor,
        ),
    )


def _train_config(**overrides: object) -> N04TrainConfig:
    base = dict(
        sequence_length=6,
        tbptt=3,
        batch_size=2,
        steps=2,
        lr=3e-4,
        burn_in=1,
        log_every=0,
        seed=0,
        device="cpu",
        threads=2,
    )
    base.update(overrides)
    return N04TrainConfig(**base)  # type: ignore[arg-type]


def _config_json(path: Path, **train_overrides: object) -> Path:
    payload = {
        "name": "n04-realnpz-test",
        "input_mode": "rawvla_rgb3",
        "measurement_size": [HEIGHT, WIDTH],
        "channels": 3,
        "camera_ids": ["agentview"],
        "seed": 0,
        "geometry": {"feature_size": [HEIGHT, WIDTH], "pixel_stride": 1.0, "max_flow": 4.0},
        "model": {
            "name": "n04-realnpz-test",
            "width": 8,
            "blocks": 1,
            "scales": 1,
            "scale_strides": [1],
            "coarse_stride": 4,
            "hidden": 16,
            "camera_embedding": 4,
            "readout_tokens": 8,
            "readout_dim": 16,
            "predictor": "learned_proprio",
        },
        "train": {
            "sequence_length": 6,
            "tbptt": 3,
            "batch_size": 2,
            "steps": 1,
            "lr": 0.0003,
            "burn_in": 1,
            "log_every": 0,
            "seed": 0,
            "device": "cpu",
            "threads": 2,
            **train_overrides,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# -- 加载器 ---------------------------------------------------------------------------


def test_libero_npz_key_parsing_and_units(tmp_path: Path) -> None:
    path = _libero_npz(tmp_path / "libero_ep0.npz")
    sequence = RawVLASequenceLoader(suite="libero").load(path)
    assert len(sequence) == LENGTH
    assert sequence.camera_ids == ("agentview", "wrist")
    assert sequence.proprio.shape == (LENGTH, 8)
    assert sequence.supervised_action.shape == (LENGTH, 10, 7)
    #: 动作尾部 padding 必须被 mask 标出，损失不重复计入填充项。
    assert float(sequence.supervised_action_mask[-1].sum()) < 10.0
    #: 缓存里没有可靠的已执行动作记录。
    assert sequence.executed_action is None
    assert sequence.availability.executed_action is False
    assert sequence.time_unit == "frame"
    #: 原始数值单位保留（uint8 未归一化），范围单独记录。
    assert sequence.value_range["agentview"][1] == pytest.approx(255.0)
    assert "pseudo-RAW" in sequence.metadata["note"]


def test_robotwin_cache_uses_actions_as_current_joint_state(tmp_path: Path) -> None:
    path = _robotwin_npz(tmp_path / "robotwin_ep0.npz")
    sequence = RawVLASequenceLoader(suite="robotwin").load(path)
    assert sequence.proprio.shape[-1] == 14
    #: 作者缓存协议：``actions[t]`` 是当前关节状态，监督从 ``actions[t+1]`` 起。
    npz = np.load(path)
    assert torch.allclose(sequence.proprio[0], torch.as_tensor(npz["actions"][0]))
    assert torch.allclose(sequence.supervised_action[0, 0], torch.as_tensor(npz["actions"][1]))
    assert sequence.supervised_action.shape[1] == 50
    assert sequence.executed_action is None


def test_robotwin_require_executed_action_rejects_missing_record(tmp_path: Path) -> None:
    path = _robotwin_npz(tmp_path / "robotwin_ep1.npz")
    with pytest.raises(ValueError, match="已执行动作"):
        RawVLASequenceLoader(suite="robotwin", require_executed_action=True).load(path)


# -- batch 构造 -----------------------------------------------------------------------


def test_make_rawvla_batches_shapes_masks_and_metadata(tmp_path: Path) -> None:
    _libero_npz(tmp_path / "ep0.npz", seed=0)
    _libero_npz(tmp_path / "ep1.npz", seed=1)
    batches = make_rawvla_batches(
        cfg=_config(),
        train_cfg=_train_config(),
        npz_dir=tmp_path,
        suite="libero",
        length=6,
        batch=2,
        steps=2,
        val_fraction=0.0,
    )
    assert len(batches) == 2
    frame = batches[0][0]
    assert isinstance(frame, RawVLAFrame)
    assert frame.measured.shape == (2, 3, HEIGHT, WIDTH)
    #: LIBERO uint8 归一化后必须落在 [0, 1]。
    assert float(frame.measured.min()) >= 0.0 and float(frame.measured.max()) <= 1.0
    assert frame.proprio.shape == (2, 8)
    assert frame.valid_mask.shape == (2, 1, HEIGHT, WIDTH)
    assert frame.saturated_mask.shape == (2, 1, HEIGHT, WIDTH)
    meta = frame.metadata
    assert meta["suite"] == "libero"
    assert meta["camera"] == "agentview"
    assert meta["executed_action_available"] is False
    assert meta["time_unit"] == "frame"
    assert "joint_action/vector" in meta["executed_action_note"]
    #: 真实帧没有合成标签：不能伪造。
    assert frame.true_object_translation is None
    assert frame.object_mask is None
    assert frame.oracle_camera_flow is None
    #: 时长与窗口对齐：seq_len - burn_in 之后的帧都是新测量。
    assert len(batches[0]) == 6


def test_saturated_pixel_is_marked(tmp_path: Path) -> None:
    _libero_npz(tmp_path / "sat.npz", length=LENGTH, saturation=True)
    batches = make_rawvla_batches(
        cfg=_config(),
        train_cfg=_train_config(),
        npz_dir=tmp_path,
        suite="libero",
        length=4,
        batch=1,
        steps=1,
        val_fraction=0.0,
    )
    first = batches[0][0]
    if float(first.measured.max()) >= 1.0:
        assert float(first.saturated_mask.sum()) >= 1.0


def test_resolution_mismatch_is_rejected(tmp_path: Path) -> None:
    _libero_npz(tmp_path / "ep0.npz")
    cfg = _config()
    bad = N04Config(
        input_mode=cfg.input_mode,
        measurement_size=(32, 32),
        channels=3,
        camera_ids=cfg.camera_ids,
        geometry=N04GeometryConfig(feature_size=(32, 32), pixel_stride=1.0),
        model=cfg.model,
    )
    with pytest.raises(ValueError, match="measurement-size"):
        make_rawvla_batches(
            cfg=bad,
            train_cfg=_train_config(),
            npz_dir=tmp_path,
            suite="libero",
            length=4,
            batch=1,
            steps=1,
            val_fraction=0.0,
        )


def test_missing_npz_dir_reports_documented_server_path(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError) as excinfo:
        discover_npz(missing)
    assert DOCUMENTED_RAW_DATA_ROOT in str(excinfo.value)
    assert str(missing) in str(excinfo.value)


def test_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    paths = [_libero_npz(tmp_path / f"ep{i}.npz", seed=i) for i in range(6)]
    train_a, val_a = split_npz_paths(paths, val_fraction=0.5, seed=0)
    train_b, val_b = split_npz_paths(paths, val_fraction=0.5, seed=0)
    assert [p.name for p in train_a] == [p.name for p in train_b]
    assert [p.name for p in val_a] == [p.name for p in val_b]
    assert set(train_a).isdisjoint(val_a)
    assert len(train_a) + len(val_a) == len(paths)


# -- 上下文与训练 ---------------------------------------------------------------------


def test_real_context_uses_relative_rotation_and_marks_missing_action(tmp_path: Path) -> None:
    _libero_npz(tmp_path / "ep0.npz", seed=3)
    batches = make_rawvla_batches(
        cfg=_config(),
        train_cfg=_train_config(),
        npz_dir=tmp_path,
        suite="libero",
        length=4,
        batch=1,
        steps=1,
        val_fraction=0.0,
    )
    first, second = batches[0][0], batches[0][1]
    first_context = build_real_context(first, None)
    assert first_context.proprio_delta_available is False
    assert first_context.executed_action_available is False
    assert first_context.dt is None and first_context.dt_available is False
    second_context = build_real_context(second, first)
    assert second_context.proprio_delta_available is True
    expected = proprio_delta(
        second.proprio, first.proprio, layout=LIBERO_STATE_LAYOUT, available=True
    )
    assert torch.allclose(second_context.proprio_delta, expected.delta, atol=1e-6)
    #: 旋转通道绝不能等于逐元素差。第一对从零旋转出发时两者恰好相同，所以要在
    #: 已经转起来的相邻帧上验证（多轴旋转下 R_t R_{t-1}^T 与轴角之差不同）。
    naive_differs = False
    for index in range(1, len(batches[0]) - 1):
        left, right = batches[0][index], batches[0][index + 1]
        context = build_real_context(right, left)
        naive = right.proprio[:, 3:6] - left.proprio[:, 3:6]
        if not torch.allclose(context.proprio_delta[:, 3:6], naive, atol=1e-6):
            naive_differs = True
    assert naive_differs
    assert second_context.info["executed_action_available"] is False


def test_train_on_real_batches_is_finite_and_checkpoints(tmp_path: Path) -> None:
    _libero_npz(tmp_path / "ep0.npz", seed=0)
    _libero_npz(tmp_path / "ep1.npz", seed=1)
    cfg = _config()
    train_cfg = _train_config(steps=2)
    batches = make_rawvla_batches(
        cfg=cfg,
        train_cfg=train_cfg,
        npz_dir=tmp_path,
        suite="libero",
        length=6,
        batch=2,
        steps=2,
        val_fraction=0.0,
    )
    out = tmp_path / "run"
    result = train(cfg, train_cfg, batches=batches, output_dir=out)
    assert result.steps == 2
    assert all(np.isfinite(list(record.values())).all() for record in result.history)
    #: 真实数据没有变化/ego 标签：这两项必须为 0，而不是拿代理当真值监督。
    assert all(record["change"] == 0.0 for record in result.history)
    assert all(record["ego_warp"] == 0.0 for record in result.history)
    assert (out / "checkpoint.pt").exists()
    payload = load_checkpoint(out / "checkpoint.pt")
    assert payload["step"] == 2


def test_resume_from_checkpoint_continues_steps(tmp_path: Path) -> None:
    _libero_npz(tmp_path / "ep0.npz", seed=0)
    _libero_npz(tmp_path / "ep1.npz", seed=1)
    cfg = _config()
    train_cfg = _train_config(steps=3)
    batches = make_rawvla_batches(
        cfg=cfg,
        train_cfg=train_cfg,
        npz_dir=tmp_path,
        suite="libero",
        length=6,
        batch=2,
        steps=3,
        val_fraction=0.0,
    )
    out = tmp_path / "resume"
    first = train(cfg, train_cfg, batches=batches[:1], output_dir=out)
    assert first.steps == 1
    resumed = train(cfg, train_cfg, batches=batches, output_dir=out, resume=out / "checkpoint.pt")
    assert resumed.steps == 3
    assert len(resumed.history) == 3


# -- CLI -----------------------------------------------------------------------------


def test_cli_reports_missing_assets_with_exit_code_two(tmp_path: Path, capsys) -> None:
    config = _config_json(tmp_path / "cfg.json")
    missing = tmp_path / "not-mounted"
    code = main(
        [
            "--config",
            str(config),
            "--npz-dir",
            str(missing),
            "--steps",
            "1",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out.strip())
    assert DOCUMENTED_RAW_DATA_ROOT in payload["error"]


def test_cli_requires_synthetic_or_npz_dir(tmp_path: Path) -> None:
    config = _config_json(tmp_path / "cfg.json")
    with pytest.raises(SystemExit):
        main(["--config", str(config), "--steps", "1", "--output", str(tmp_path / "out")])


def test_cli_runs_on_local_npz_and_writes_data_summary(tmp_path: Path, capsys) -> None:
    _libero_npz(tmp_path / "ep0.npz", seed=0)
    _libero_npz(tmp_path / "ep1.npz", seed=1)
    config = _config_json(tmp_path / "cfg.json")
    out = tmp_path / "cli-run"
    code = main(
        [
            "--config",
            str(config),
            "--npz-dir",
            str(tmp_path),
            "--suite",
            "libero",
            "--steps",
            "1",
            "--npz-limit",
            "2",
            "--val-fraction",
            "0",
            "--measurement-size",
            str(HEIGHT),
            str(WIDTH),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    summary = json.loads((out / "data.json").read_text(encoding="utf-8"))
    assert summary["source"] == "rawvla_npz"
    assert summary["suite"] == "libero"
    assert summary["measurement_scale"] == "libero:uint8/255"
    assert summary["executed_action_available"] is False
    assert summary["sequences"]
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["data"] == "rawvla_npz"
    assert (out / "checkpoint.pt").exists()


def test_robotwin_batches_keep_float32_scale(tmp_path: Path) -> None:
    _robotwin_npz(tmp_path / "robotwin_ep0.npz", length=LENGTH)
    batches = make_rawvla_batches(
        cfg=_config(camera="head"),
        train_cfg=_train_config(),
        npz_dir=tmp_path,
        suite="robotwin",
        length=4,
        batch=1,
        steps=1,
        val_fraction=0.0,
    )
    frame = batches[0][0]
    #: RoboTwin float32 HDR-derived RAW 不除以 255、也不按帧最大值归一化。
    assert float(frame.measured.max()) > 1.0
    assert frame.metadata["measurement_scale"] == "robotwin:as-is"
    #: 饱和是未知量：不伪造饱和掩码。
    assert float(frame.saturated_mask.sum()) == 0.0
    layout = ROBOTWIN_JOINT_LAYOUT
    assert frame.proprio.shape[-1] == layout[-1].stop


def test_checkpoint_prefix_alignment_for_ablations(tmp_path: Path) -> None:
    """消融训练保存的 checkpoint 带 ``encoder.`` 前缀，两种目标都必须能装进去。

    此前评测加载消融 checkpoint 直接 RuntimeError（unexpected/missing keys），
    “候选 vs 消融”的同预算对照因此完全跑不出来。
    """
    from owac.baselines.n04 import build_baseline
    from owac.frameworks.n04.model import N04Encoder

    _libero_npz(tmp_path / "ep0.npz", seed=0)
    _libero_npz(tmp_path / "ep1.npz", seed=1)
    cfg = _config()
    train_cfg = _train_config(steps=1)
    batches = make_rawvla_batches(
        cfg=cfg,
        train_cfg=train_cfg,
        npz_dir=tmp_path,
        suite="libero",
        length=6,
        batch=2,
        steps=1,
        val_fraction=0.0,
    )
    out = tmp_path / "ablation"
    #: 用只改上下文、不改结构的消融：内层 N04Encoder 才能与 checkpoint 架构一致。
    train(cfg, train_cfg, batches=batches, output_dir=out, baseline="n04_time_shifted_context")
    state = load_checkpoint(out / "checkpoint.pt")["encoder"]
    assert any(key.startswith("encoder.") for key in state), "消融 checkpoint 应带 wrapper 前缀"

    wrapper = build_baseline("n04_time_shifted_context", cfg)
    load_checkpoint(out / "checkpoint.pt", encoder=wrapper)

    inner = N04Encoder(cfg)
    load_checkpoint(out / "checkpoint.pt", encoder=inner)
    expected = {
        key[len("encoder.") :]: value for key, value in state.items() if key.startswith("encoder.")
    }
    for key, value in expected.items():
        assert torch.allclose(inner.state_dict()[key], value)


def test_checkpoint_prefix_alignment_rejects_wrong_architecture(tmp_path: Path) -> None:
    _libero_npz(tmp_path / "ep0.npz", seed=0)
    cfg = _config()
    train_cfg = _train_config(steps=1)
    batches = make_rawvla_batches(
        cfg=cfg,
        train_cfg=train_cfg,
        npz_dir=tmp_path,
        suite="libero",
        length=6,
        batch=1,
        steps=1,
        val_fraction=0.0,
    )
    out = tmp_path / "run"
    train(cfg, train_cfg, batches=batches, output_dir=out)
    other = _config()
    other = N04Config(
        input_mode=other.input_mode,
        measurement_size=other.measurement_size,
        channels=other.channels,
        camera_ids=other.camera_ids,
        geometry=other.geometry,
        model=other.model.without(width=16, blocks=2),
    )
    from owac.frameworks.n04.model import N04Encoder

    with pytest.raises((RuntimeError, ValueError)):
        load_checkpoint(out / "checkpoint.pt", encoder=N04Encoder(other))
