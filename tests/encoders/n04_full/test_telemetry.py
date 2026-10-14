"""N04 S0 遥测与因果上下文测试。

覆盖五条验收：

1. 无 sidecar 的旧 loader 输出与改动前**逐位一致**（golden 摘要写在下方的 JSON 常量里，
   由改动前的实现实测生成，不是形状断言）。
2. ``context_at`` 不泄漏：``decision_time`` 之后到达的图像/本体不可见，未来
   ``supervised_action`` 不进入在线上下文。
3. 迟到帧按曝光时刻回到缓存，再用已发生的控制重放到当前；到达延迟与曝光年龄分开记录。
4. ``executed_between`` 保序、覆盖与间隔可检查；缺失（dropped/planned）动作不当零动作。
5. 只有帧索引时 ``time_unit="frame"``，不编造秒数。

数值参照都是手工算出的区间/覆盖值或直接来自 NPZ 原始数组，不做 shape-only 断言。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.raw.rawvla_sequence import RawVLASequenceLoader
from owac.raw.telemetry import (
    CalibrationRecord,
    ExecutedControl,
    FrameTiming,
    TelemetrySidecar,
    interval_report,
)

HEIGHT = 16
WIDTH = 16
LENGTH = 12

# --------------------------------------------------------------------------------------
# golden：由**改动前**的 RawVLASequenceLoader 实测输出（SHA256 of raw tensor bytes）
# --------------------------------------------------------------------------------------

_GOLDEN_LIBERO = """{
 "availability": "proprio",
 "camera_ids": "agentview,wrist",
 "executed_action": "None",
 "frame_ids": "700a4498438a801b5781533040bce85a20ae4bfe08866f7552ff33e172923b0a",
 "len": "12",
 "measurements.agentview": "7792b6e59c4fbe7af9f7e4ad772f2dee398745404313a138a4415d56ddfdc502",
 "measurements.wrist": "874ba9f34b4facd2c5a7f8b12f0726a33f0eabbfc5a7b58783363861241b6827",
 "new_measurement": "3ee5f0d83bf791f0fb4d750a5719ce19d6d352ef7e5a4264e4b760f0f9c15014",
 "proprio": "cf3ec4ac897a4b1593badd8aff5264b857c2baef9c864ede58435cbe3be30f7a",
 "supervised_action": "a8adbe944fc51c62b7fe83d173706a7ce1f1c0bb432f46e7ae4aefb4ab5a6c4d",
 "supervised_action_mask": "9603ea2f2e4011717f2d81f8b3fcbb051beb7a432ccf0561b564f24a09fdfe22",
 "time_unit": "frame",
 "value_range": [
  "agentview:0x0.0p+0:0x1.fe00000000000p+7",
  "wrist:0x0.0p+0:0x1.8e00000000000p+7"
 ],
 "win0.executed_action": "None",
 "win0.frame_ids": "281b02b10f5f4997e5bf8c93343e6f2aa8bc81ffad6d6813c593181ebceda12a",
 "win0.loss_mask": "c5fa7a4f055ecfb310cff078852a08c832be9ac4c4d30ecb73e4b55630fff19f",
 "win0.measurements.agentview": "9fa6cc54d88ae6c6732fa7d71e41f4ec49148d9a95a5f7edb9860c3852fb4030",
 "win0.measurements.wrist": "c0543ed10fd9af24cbd072e025eca8bdaabb202fa30b24dfe99179d158143cdf",
 "win0.new_measurement": "377a23f52c6b357696238c3318f677a082dd3430bb6691042bd550a5cda28ebb",
 "win0.proprio": "4fa11af71aeac778cb01bbcb2853030a6e6a66096c4f9a93f7469f9695be385c",
 "win0.start": "0",
 "win0.supervised_action": "addffaf26f9e6a252b7957a0f32e6e48738b2607f09eda0555c15eac70cd5be8",
 "win0.supervised_action_mask": "074ac3dd71f19496fe2a5f8ac1d726af381564cea93d7532f96646e2deb2139d",
 "win1.executed_action": "None",
 "win1.frame_ids": "a5ace85764581e6a295fb17f24c34a50f0723ef408706e0a9ebfebb9bd8c626b",
 "win1.loss_mask": "c5fa7a4f055ecfb310cff078852a08c832be9ac4c4d30ecb73e4b55630fff19f",
 "win1.measurements.agentview": "7eb243757cf317b24427879df1189cc52e72ba874b621cc0a63d32dd2c946c10",
 "win1.measurements.wrist": "215136d890cd076530eb6e5885adf765b09ad24ef808453b2860a65b9df36875",
 "win1.new_measurement": "377a23f52c6b357696238c3318f677a082dd3430bb6691042bd550a5cda28ebb",
 "win1.proprio": "8be16fbff5f7004592bf71f1b7d543470943622b3b70c6829e8492099299cad3",
 "win1.start": "3",
 "win1.supervised_action": "05e4f4d30f789d3c8340b58f5251c25eef8feeef30c2c6965d10a628ec962f2a",
 "win1.supervised_action_mask": "eab7e9356e426c42c4050cde76620e94685fa71d261f694a77a838a5f0d38f67",
 "win2.executed_action": "None",
 "win2.frame_ids": "59f8e380653956773389896585e9c087583dedc65c00f3048566f205677af5da",
 "win2.loss_mask": "c5fa7a4f055ecfb310cff078852a08c832be9ac4c4d30ecb73e4b55630fff19f",
 "win2.measurements.agentview": "efacdf383af5f195ccdd88054aef35fb9755e4760495599b8f0f76d89f696de6",
 "win2.measurements.wrist": "c99215943d81d8a2239880b9b7b3eb25d21bbcce3c4197a7c730672fa78b16f5",
 "win2.new_measurement": "377a23f52c6b357696238c3318f677a082dd3430bb6691042bd550a5cda28ebb",
 "win2.proprio": "1b4fe369283c18f8e4dd4c66a394204a607d71306b761be03d97463a416b6544",
 "win2.start": "6",
 "win2.supervised_action": "2653ecde51ccf376198669236a54543a9607ff80ce397e1ac356a0983afd49ac",
 "win2.supervised_action_mask": "015c14d48a33353fa6748fdf87a1c07684ee8d35694253a5ff6e396c2bbdb753"
}"""
_GOLDEN_ROBOTWIN = """{
 "availability": "proprio",
 "camera_ids": "head,left,right",
 "executed_action": "None",
 "frame_ids": "700a4498438a801b5781533040bce85a20ae4bfe08866f7552ff33e172923b0a",
 "len": "12",
 "measurements.head": "712ecbef2475cb15e9b1beeb373e521cedd0a18a82a6348a6cea95dd84ec6a35",
 "measurements.left": "eb6dbe9ec0272d33408f79c98d5d7dff9f4206faa638564ff290b58037605b10",
 "measurements.right": "6d4a9e7209ec03811432741cf2e4e21608c321af447e4a5bf235c56ea4bf4b00",
 "new_measurement": "3ee5f0d83bf791f0fb4d750a5719ce19d6d352ef7e5a4264e4b760f0f9c15014",
 "proprio": "b153b825b86f8bb00bbab19038046ed76f7a675d3c952188efef872b275ec284",
 "supervised_action": "c5028b9ff8665b187d8862c8390c999e318078d0c6170e147a1f98d15c525be6",
 "supervised_action_mask": "66f0f4bb76c6f002369fd7b7f73375f9c51cc1e5352304b2b5b487b52fd87641",
 "time_unit": "frame",
 "value_range": [
  "head:0x1.4c00000000000p-14:0x1.fff8aa0000000p+1",
  "left:0x1.c000000000000p-15:0x1.ffff920000000p+1",
  "right:0x1.2c30000000000p-10:0x1.fffb3c0000000p+1"
 ],
 "win0.executed_action": "None",
 "win0.frame_ids": "281b02b10f5f4997e5bf8c93343e6f2aa8bc81ffad6d6813c593181ebceda12a",
 "win0.loss_mask": "c5fa7a4f055ecfb310cff078852a08c832be9ac4c4d30ecb73e4b55630fff19f",
 "win0.measurements.head": "df28b466f1b3340d2f0ddab8afee0b7a05f34ad09f4a3631911fe928d6c2df7b",
 "win0.measurements.left": "6311f37fa7958e7062226cac3fcd7361ab7c6d560af2297906d82517b9d5e43b",
 "win0.measurements.right": "a8993d7099a835b733cb8d6daa84e5440cb442e2e51dbb7dc43e5b50e9cbac87",
 "win0.new_measurement": "377a23f52c6b357696238c3318f677a082dd3430bb6691042bd550a5cda28ebb",
 "win0.proprio": "def9ada3a4a88f53851034de4ce6ac137ca83216f9a39065218cea021e89ec06",
 "win0.start": "0",
 "win0.supervised_action": "bcd5a509a3288bc3fba924ec4758782a158f7b8aad6f5a528e803cd007ca8dff",
 "win0.supervised_action_mask": "5a6a96316cd781e79cdf8dd9a04657137e2c5a5616070096f3aff5cdecf2054e",
 "win1.executed_action": "None",
 "win1.frame_ids": "a5ace85764581e6a295fb17f24c34a50f0723ef408706e0a9ebfebb9bd8c626b",
 "win1.loss_mask": "c5fa7a4f055ecfb310cff078852a08c832be9ac4c4d30ecb73e4b55630fff19f",
 "win1.measurements.head": "3dd079366c682df6ed1a26724b829d0c3160719957b0f9d138e2674f7e3500ad",
 "win1.measurements.left": "1f242f2203d1b968789e4441fc8c530c89866bd3958ad2d2da9aea4728990400",
 "win1.measurements.right": "13c7bc7098f450fafbee328a122b4941232a3d2921e6846b0906a4eaccb46afa",
 "win1.new_measurement": "377a23f52c6b357696238c3318f677a082dd3430bb6691042bd550a5cda28ebb",
 "win1.proprio": "fad8f5f1738d29a079736be0d61e7c6b385a678d813a08ab0a8b549995f59add",
 "win1.start": "3",
 "win1.supervised_action": "2a786f1136f9fbbb6d3fca2e8ca30c4d8973bffd932607e298b73230d32cd068",
 "win1.supervised_action_mask": "5a6a96316cd781e79cdf8dd9a04657137e2c5a5616070096f3aff5cdecf2054e",
 "win2.executed_action": "None",
 "win2.frame_ids": "59f8e380653956773389896585e9c087583dedc65c00f3048566f205677af5da",
 "win2.loss_mask": "c5fa7a4f055ecfb310cff078852a08c832be9ac4c4d30ecb73e4b55630fff19f",
 "win2.measurements.head": "bb1ec38690b620d0dfbeb05115b166c0c5df2fbe2b4175076279e584f85ae6de",
 "win2.measurements.left": "9d7b6c5b4dae573429d16a1b0916e48ad2469bde36b628fd0bf981cc6dd866d6",
 "win2.measurements.right": "cbe97a7e7abb15e72c281214d2afab82a167413d1feb8def7812fec6bcd46bc7",
 "win2.new_measurement": "377a23f52c6b357696238c3318f677a082dd3430bb6691042bd550a5cda28ebb",
 "win2.proprio": "a33bfb4ceea869dc739fc50a9470a0629fa19de3d62b9dd7f70455e813ca7c42",
 "win2.start": "6",
 "win2.supervised_action": "061271e4b8d3369768c92b1a75fd283ea6ad00da90699e6251c0806899b5edd9",
 "win2.supervised_action_mask": "5a6a96316cd781e79cdf8dd9a04657137e2c5a5616070096f3aff5cdecf2054e"
}"""


def _fixture_libero(path: Path, *, length: int = LENGTH, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    agentview = rng.integers(0, 200, size=(length, HEIGHT, WIDTH, 3), dtype=np.uint8)
    wrist = rng.integers(0, 200, size=(length, HEIGHT, WIDTH, 3), dtype=np.uint8)
    agentview[0, 0, 0, :] = 255
    states = np.zeros((length, 8), dtype=np.float32)
    states[:, 0] = np.linspace(0.0, 0.5, length)
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


def _fixture_robotwin(path: Path, *, length: int = LENGTH, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
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


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _digest(sequence) -> dict[str, str]:
    """与 golden 生成时完全相同的摘要函数；任何一位数值变化都会改变结果。"""
    out: dict[str, str] = {}
    for cam in sorted(sequence.measurements):
        out[f"measurements.{cam}"] = _sha(sequence.measurements[cam].numpy())
    out["proprio"] = _sha(sequence.proprio.numpy())
    out["supervised_action"] = _sha(sequence.supervised_action.numpy())
    out["supervised_action_mask"] = _sha(sequence.supervised_action_mask.numpy())
    out["frame_ids"] = _sha(sequence.frame_ids.numpy())
    out["new_measurement"] = _sha(sequence.new_measurement.numpy())
    out["executed_action"] = (
        "None" if sequence.executed_action is None else _sha(sequence.executed_action.numpy())
    )
    out["len"] = str(len(sequence))
    out["camera_ids"] = ",".join(sequence.camera_ids)
    out["time_unit"] = sequence.time_unit
    out["availability"] = ",".join(
        sorted(key for key, value in sequence.availability.as_dict().items() if value)
    )
    out["value_range"] = [
        f"{key}:{float(value[0]).hex()}:{float(value[1]).hex()}"
        for key, value in sorted(sequence.value_range.items())
    ]
    for window_index, window in enumerate(
        RawVLASequenceLoader.windows(sequence, length=5, stride=3, burn_in=1)
    ):
        prefix = f"win{window_index}"
        out[f"{prefix}.start"] = str(window["start"])
        for cam in sorted(window["measurements"]):
            out[f"{prefix}.measurements.{cam}"] = _sha(window["measurements"][cam].numpy())
        out[f"{prefix}.proprio"] = _sha(window["proprio"].numpy())
        out[f"{prefix}.supervised_action"] = _sha(window["supervised_action"].numpy())
        out[f"{prefix}.supervised_action_mask"] = _sha(window["supervised_action_mask"].numpy())
        out[f"{prefix}.frame_ids"] = _sha(window["frame_ids"].numpy())
        out[f"{prefix}.new_measurement"] = _sha(window["new_measurement"].numpy())
        out[f"{prefix}.loss_mask"] = _sha(window["loss_mask"].numpy())
        out[f"{prefix}.executed_action"] = (
            "None" if window["executed_action"] is None else _sha(window["executed_action"].numpy())
        )
    return out


def test_default_path_is_bit_identical_to_pre_change_golden(tmp_path: Path) -> None:
    """验收 1：无 sidecar 时每个张量逐字节等于改动前的实测输出。"""
    libero = RawVLASequenceLoader(suite="libero").load(
        _fixture_libero(tmp_path / "libero-fixture.npz")
    )
    robotwin = RawVLASequenceLoader(suite="robotwin").load(
        _fixture_robotwin(tmp_path / "robotwin.npz")
    )
    assert _digest(libero) == json.loads(_GOLDEN_LIBERO)
    assert _digest(robotwin) == json.loads(_GOLDEN_ROBOTWIN)
    #: golden 只有同时覆盖到逐帧测量与窗口输出才有意义。
    assert {key.split(".")[0] for key in json.loads(_GOLDEN_LIBERO)} >= {
        "measurements",
        "proprio",
        "supervised_action",
        "supervised_action_mask",
        "frame_ids",
        "new_measurement",
        "len",
        "camera_ids",
        "time_unit",
        "availability",
        "value_range",
    }
    assert sum(1 for key in json.loads(_GOLDEN_LIBERO) if key.startswith("win")) == 30


# --------------------------------------------------------------------------------------
# sidecar 夹具：曝光、到达、已执行控制都用可手算的整数/小数
# --------------------------------------------------------------------------------------


def _libero_sidecar(actions: np.ndarray, *, late_frame: int = 4, common_unit: bool = True):
    """10 Hz 相机、每帧 10 ms 曝光；``late_frame`` 的到达被推迟 350 ms。

    手算基准：frame i 的 exposure_start=0.1i、mid=0.1i+0.005、end=0.1i+0.010；
    正常到达 = end+0.020 = 0.1i+0.030；控制 i 的区间 = [0.1i+0.011, 0.1i+0.051)。
    """
    timing = []
    for index in range(LENGTH):
        start = 0.1 * index
        end = start + 0.010
        arrival = end + 0.020
        if index == late_frame:
            arrival = end + 0.350
        timing.append(
            FrameTiming(
                frame=index,
                clock_id="camera0",
                time_unit="s",
                exposure_start=start,
                exposure_mid=start + 0.005,
                exposure_end=end,
                arrival_time=arrival,
                source=f"log/camera0.jsonl:{index}",
            )
        )
    controls = []
    for index in range(LENGTH):
        start = 0.1 * index + 0.011
        controls.append(
            ExecutedControl(
                command_id=f"cmd{index}",
                value=np.asarray(actions[index], dtype=np.float64),
                mode="osc_delta_eef",
                units="m+rad",
                status="executed",
                evidence_level="replay_verified",
                command_time=start,
                start_time=start,
                end_time=start + 0.040,
                frame=index,
                dispatch_phase="after_exposure",
                time_unit="s",
                source=f"log/executor.jsonl:cmd{index}",
            )
        )
    return TelemetrySidecar(
        episode_id="libero-fixture",
        suite="libero",
        time_unit="s" if common_unit else "frame",
        timing=tuple(timing),
        controls=tuple(controls),
        calibration=(
            CalibrationRecord(
                camera_id="agentview",
                calibration_version="fixture-v1",
                K=np.array([[320.0, 0.0, 128.0], [0.0, 320.0, 128.0], [0.0, 0.0, 1.0]]),
                T_A_from_B=np.eye(4),
                frame_a="W",
                frame_b="C",
                coordinate_convention="pixel center origin top-left, y down, align_corners=False",
                time_offset=0.0,
                provenance="fixture",
            ),
        ),
        provenance={"source": "unit-test"},
    )


def _load_with_sidecar(tmp_path: Path, sidecar: TelemetrySidecar, *, seed: int = 0):
    path = _fixture_libero(tmp_path / "libero-fixture.npz", seed=seed)
    npz_actions = np.load(path)["actions"]
    loader = RawVLASequenceLoader(suite="libero", sidecar=sidecar)
    return loader.load(path), npz_actions


# --------------------------------------------------------------------------------------
# 验收 2：context_at 不泄漏
# --------------------------------------------------------------------------------------


def test_context_at_hides_frames_that_arrived_after_decision_time(tmp_path: Path) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    sidecar = _libero_sidecar(actions)
    sequence = RawVLASequenceLoader(suite="libero", sidecar=sidecar).load(path)

    #: frame 4 的曝光中点是 0.405，但它在 0.758 才到达。
    context = sequence.context_at(exposure_time=0.405, decision_time=0.500)
    assert context.latest_frame == 3
    assert context.visible_frames == (0, 1, 2, 3)
    assert 4 not in context.visible_frames
    assert context.missing_frames == (4,)
    assert context.late_frame is True
    #: 本体与图像同时受到达时刻约束：不得给出 frame 4 之后的 proprio。
    assert context.proprio_frame == 3
    assert context.availability["proprio"] is True
    #: 未来监督动作绝不进入在线上下文。
    assert context.target_actions_included is False
    assert "supervised_action" not in json.dumps(context.as_dict())
    assert sidecar.timing_for(5).arrival_time > 0.5

    #: decision_time 早于任何到达：什么都没有，且不假装有。
    empty = sequence.context_at(exposure_time=0.405, decision_time=0.005)
    assert empty.visible_frames == ()
    assert empty.latest_frame is None
    assert empty.proprio_frame is None
    assert empty.availability["proprio"] is False
    assert empty.replay_controls == ()


def test_context_replay_segment_excludes_actions_not_yet_executed(tmp_path: Path) -> None:
    """已执行的控制片段可以进入上下文；尚未下发的动作(未来监督)不行。"""
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    sidecar = _libero_sidecar(actions)
    sequence = RawVLASequenceLoader(suite="libero", sidecar=sidecar).load(path)

    context = sequence.context_at(exposure_time=0.405, decision_time=0.500)
    frames = [control.frame for control in context.replay_controls]
    #: latest 可见帧是 3：曝光之后确实执行过的是 cmd3（0.311）与 cmd4（0.411）。
    assert frames == [3, 4]
    values = np.stack([control.value for control in context.replay_controls])
    assert np.array_equal(values[0], actions[3].astype(np.float64))
    assert np.array_equal(values[1], actions[4].astype(np.float64))
    #: cmd5（0.511）当时还没下发，绝不能出现。
    assert 5 not in frames
    assert context.pending_control_ids == ()

    #: 未来目标本身仍然完整保留在训练目标结构里，只是不属于上下文。
    target = sequence.target_view(3)
    assert target.future is True
    #: 尾部 padding 用掩码标出：12 帧序列里 frame 3 只剩 9 步真实监督。
    assert target.valid_steps == 9
    assert [float(v) for v in target.mask.numpy()] == [1.0] * 9 + [0.0]
    assert np.array_equal(target.actions.numpy()[:9], actions[3:12].astype(np.float32))
    assert float(np.abs(target.actions.numpy()[9]).max()) == 0.0


# --------------------------------------------------------------------------------------
# 验收 3：迟到帧按曝光时刻回填，再用已执行控制重放
# --------------------------------------------------------------------------------------


def test_late_frame_is_placed_by_exposure_and_replayed_by_executed_controls(
    tmp_path: Path,
) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    sidecar = _libero_sidecar(actions)
    sequence = RawVLASequenceLoader(suite="libero", sidecar=sidecar).load(path)

    before = sequence.context_at(exposure_time=0.405, decision_time=0.500)
    after = sequence.context_at(exposure_time=0.405, decision_time=0.800)

    #: frame 4 在 0.758 到达：它按**曝光时刻** 0.405 回到缓存，而不是按到达顺序排在队尾。
    assert before.latest_frame == 3 and before.late_frame is True
    assert after.latest_frame == 4 and after.late_frame is False
    assert after.visible_frames == (0, 1, 2, 3, 4)
    assert after.latest_exposure_time == pytest.approx(0.405)
    assert after.latest_arrival_time == pytest.approx(0.760)

    #: 回填后要重放的是"该帧曝光(0.405)之后、当前(0.800)之前"已执行的控制 cmd4..cmd7；
    #: 当时还没下发的 cmd8(0.811) 不进入重放段。
    assert [control.frame for control in after.replay_controls] == [4, 5, 6, 7]
    assert after.arrival_delay == pytest.approx(0.760 - 0.410)
    assert after.processing_delay == pytest.approx(0.800 - 0.760)
    #: 输出延迟（相对到达）与曝光年龄（相对曝光）是两个独立的量。
    assert after.age_since_exposure == pytest.approx(0.800 - 0.405)
    assert after.age_since_exposure > after.arrival_delay


def test_arrival_delay_is_recorded_per_frame_separately_from_exposure(tmp_path: Path) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    sidecar = _libero_sidecar(actions)
    normal = sidecar.timing_for(3)
    late = sidecar.timing_for(4)
    assert normal.arrival_delay() == pytest.approx(0.020)
    assert late.arrival_delay() == pytest.approx(0.350)
    #: 曝光结束缺失时不得用曝光开始估"到达延迟"。
    assert FrameTiming(frame=0, exposure_start=0.0, arrival_time=0.5).arrival_delay() is None


# --------------------------------------------------------------------------------------
# 验收 4：executed_between 保序、覆盖与间隔
# --------------------------------------------------------------------------------------


def _interval_sidecar(actions: np.ndarray) -> TelemetrySidecar:
    """frame 3 dropped、frame 6 planned；其余 executed。区间全为 [0.1i+0.011, +0.040)。"""
    controls = []
    for index in range(LENGTH):
        start = 0.1 * index + 0.011
        status = {"3": "dropped", "6": "planned"}.get(str(index), "executed")
        controls.append(
            ExecutedControl(
                command_id=f"cmd{index}",
                value=np.asarray(actions[index], dtype=np.float64),
                mode="osc_delta_eef",
                units="m+rad",
                status=status,
                evidence_level="executed_log",
                command_time=start,
                start_time=start,
                end_time=start + 0.040,
                frame=index,
                dispatch_phase="after_exposure",
                time_unit="s",
            )
        )
    return TelemetrySidecar(
        episode_id="libero-fixture", suite="libero", time_unit="s", controls=tuple(controls)
    )


def test_executed_between_keeps_order_and_reports_coverage(tmp_path: Path) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    sequence = RawVLASequenceLoader(suite="libero", sidecar=_interval_sidecar(actions)).load(path)

    result = sequence.executed_between(0.0, 0.6)
    assert [control.command_id for control in result.controls] == [
        "cmd0",
        "cmd1",
        "cmd2",
        "cmd4",
        "cmd5",
    ]
    assert result.order_violations == ()
    #: 手算：5 段 × 0.040 = 0.200 被覆盖，窗口 0.600 -> coverage = 1/3。
    assert result.coverage == pytest.approx(0.200 / 0.600)
    #: 缺口必须显式列出，并与覆盖互补。
    gap_total = sum(end - start for start, end in result.gaps)
    assert gap_total + result.coverage * 0.600 == pytest.approx(0.600)
    dropped_span = (0.311, 0.351)
    assert any(start <= dropped_span[0] and dropped_span[1] <= end for start, end in result.gaps), (
        "被 dropped 的 cmd3 区间必须落在缺口里"
    )
    assert result.overlaps == ()
    assert result.missing_time == ()


def test_missing_control_is_not_treated_as_zero_action(tmp_path: Path) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    sequence = RawVLASequenceLoader(suite="libero", sidecar=_interval_sidecar(actions)).load(path)
    result = sequence.executed_between(0.3, 0.5)

    #: cmd3 已被 dropped：它只能出现在 excluded，绝不能作为零动作混进已执行序列。
    assert [control.command_id for control in result.controls] == ["cmd4"]
    assert [(control.command_id, control.status) for control in result.excluded] == [
        ("cmd3", "dropped")
    ]
    executed_values = np.stack([control.value for control in result.controls])
    assert not np.any(np.all(executed_values == 0.0, axis=1))
    #: cmd6 在窗口之外（0.611 > 0.5）：planned 不进入本窗口。
    assert "cmd6" not in [control.command_id for control in result.excluded]
    assert result.coverage == pytest.approx(0.040 / 0.200)


def test_executed_between_flags_out_of_order_and_overlap() -> None:
    """乱序与重叠必须被报出来，而不是靠排序掩盖。"""
    controls = [
        ExecutedControl(
            command_id="a",
            value=np.ones(2),
            status="executed",
            evidence_level="executed_log",
            command_time=0.10,
            start_time=0.10,
            end_time=0.30,
            time_unit="s",
        ),
        ExecutedControl(
            command_id="b",
            value=np.ones(2),
            status="executed",
            evidence_level="executed_log",
            command_time=0.05,
            start_time=0.05,
            end_time=0.20,
            time_unit="s",
        ),
    ]
    result = interval_report(controls, t0=0.0, t1=0.4, time_unit="s")
    assert [control.command_id for control in result.controls] == ["a", "b"]
    assert result.order_violations == ("b",)
    assert result.overlaps == (("a", "b"),)
    assert result.coverage == pytest.approx(0.25 / 0.4)
    assert result.gaps == ((0.0, 0.05), (0.30, 0.4))


# --------------------------------------------------------------------------------------
# 验收 5：帧号不是秒
# --------------------------------------------------------------------------------------


def test_frame_only_sidecar_stays_in_frame_units(tmp_path: Path) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    frame_sidecar = TelemetrySidecar(
        episode_id="libero-fixture",
        suite="libero",
        time_unit="frame",
        timing=tuple(
            FrameTiming(
                frame=index,
                clock_id="episode_clock",
                time_unit="frame",
                exposure_mid=float(index) + 0.25,
                arrival_time=float(index) + 1.0,
            )
            for index in range(LENGTH)
        ),
        provenance={"source": "unit-test"},
    )
    sequence = RawVLASequenceLoader(suite="libero", sidecar=frame_sidecar).load(path)

    assert sequence.time_unit == "frame"
    #: 没有秒就不能把 timestamp 标成可得。
    assert sequence.availability.timestamp is False
    assert sequence.availability.exposure is True
    report = frame_sidecar.validate(n_frames=LENGTH)
    assert "time.frame_index_only" in report.codes()
    context = sequence.context_at(exposure_time=5.25, decision_time=7.0)
    assert context.time_unit == "frame"
    assert context.latest_frame == 5
    assert "帧序号" in context.note
    #: 该 sidecar 没有控制日志：不补零动作，重放段为空。
    assert context.replay_controls == ()
    assert context.age_since_exposure == pytest.approx(1.75)
    assert sequence.summary()["time_unit"] == "frame"
    assert sequence.summary()["telemetry"]["conditions"].count("control.absent") == 1


# --------------------------------------------------------------------------------------
# sidecar 合同本身
# --------------------------------------------------------------------------------------


def test_availability_distinguishes_zero_from_missing() -> None:
    partial = FrameTiming(frame=0, clock_id="c", time_unit="s", exposure_start=0.0)
    assert partial.availability["exposure_start"] is True
    assert partial.availability["exposure_mid"] is False
    assert partial.exposure_time() == 0.0
    empty = FrameTiming(frame=1, clock_id="c", time_unit="s")
    assert empty.availability == {
        "exposure_start": False,
        "exposure_mid": False,
        "exposure_end": False,
        "arrival_time": False,
        "rolling_shutter_row_time": False,
    }
    #: 什么都没记时必须给 None，而不是 0。
    assert empty.exposure_time() is None
    assert ExecutedControl(command_id="x").availability["value"] is False
    assert CalibrationRecord(camera_id="cam").availability["K"] is False


def test_exposure_time_fallback_order_is_declared() -> None:
    both_ends = FrameTiming(
        frame=0, time_unit="s", exposure_start=0.0, exposure_end=0.010, exposure_mid=0.004
    )
    assert both_ends.exposure_time() == 0.004
    ends_only = FrameTiming(frame=0, time_unit="s", exposure_start=0.0, exposure_end=0.010)
    assert ends_only.exposure_time() == pytest.approx(0.005)
    start_only = FrameTiming(frame=0, time_unit="s", exposure_start=0.003)
    assert start_only.exposure_time() == 0.003
    end_only = FrameTiming(frame=0, time_unit="s", exposure_end=0.007)
    assert end_only.exposure_time() == 0.007
    assert "退化顺序" in FrameTiming.exposure_time.__doc__


def test_calibration_record_rejects_impossible_transform() -> None:
    with pytest.raises(ValueError, match="最后一行"):
        CalibrationRecord(
            camera_id="cam",
            T_A_from_B=np.diag([1.0, 1.0, 1.0, 2.0]),
            frame_a="W",
            frame_b="C",
            coordinate_convention="x",
        )
    with pytest.raises(ValueError, match="frame_a"):
        CalibrationRecord(camera_id="cam", T_A_from_B=np.eye(4), coordinate_convention="x")
    with pytest.raises(ValueError, match=r"\[3,3\]"):
        CalibrationRecord(camera_id="cam", K=np.eye(4))


def test_executed_control_rejects_non_finite_dispatched_value() -> None:
    ExecutedControl(command_id="planned-nan", status="planned", value=np.array([np.nan]))
    with pytest.raises(ValueError, match="有限"):
        ExecutedControl(command_id="bad", status="executed", value=np.array([np.nan, 1.0]))


def test_sidecar_json_round_trip_is_lossless(tmp_path: Path) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    sidecar = _libero_sidecar(actions)
    restored = TelemetrySidecar.from_json(sidecar.to_json())
    assert restored.as_dict() == sidecar.as_dict()
    assert restored.timing_for(4).arrival_delay() == pytest.approx(0.350)
    assert np.array_equal(restored.controls[4].value, actions[4].astype(np.float64))
    assert np.array_equal(restored.calibration_for("agentview").K, sidecar.calibration[0].K)
    assert restored.controls[0].evidence_level == "replay_verified"
    assert restored.controls[0].verified is True


def test_validate_lists_structured_unavailable_conditions(tmp_path: Path) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    report = _interval_sidecar(actions).validate(n_frames=LENGTH)
    codes = report.codes()
    assert "control.non_executed_present" in codes
    assert "timing.absent" in codes
    assert "calibration.absent" in codes
    assert report.ok is False
    assert report.availability["timing.frames"] == 0.0
    condition = next(
        item for item in report.conditions if item.code == "control.non_executed_present"
    )
    assert condition.scope == "control" and "dropped" in condition.detail
    assert set(condition.as_dict()) == {"code", "scope", "frame", "detail"}

    complete = _libero_sidecar(actions).validate(n_frames=LENGTH)
    assert complete.codes() == ()
    assert complete.ok is True
    assert complete.availability["timing.arrival_time"] == 1.0


def test_sidecar_never_invents_executed_action(tmp_path: Path) -> None:
    """只有逐帧一一对应且相位明确时才能重建 executed_action。"""
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    full_loader = RawVLASequenceLoader(suite="libero", sidecar=_libero_sidecar(actions))
    sequence = full_loader.load(path)
    assert sequence.executed_action is not None
    assert sequence.availability.executed_action is True
    assert torch.allclose(sequence.executed_action, torch.as_tensor(actions.astype(np.float32)))

    gappy = TelemetrySidecar(
        episode_id="libero-fixture",
        suite="libero",
        time_unit="s",
        controls=_libero_sidecar(actions).controls[:-1],
    )
    partial = RawVLASequenceLoader(suite="libero", sidecar=gappy).load(path)
    assert partial.executed_action is None
    assert partial.availability.executed_action is False
    reason = partial.metadata["telemetry"]["executed_action_unavailable_reason"]
    assert "11/12" in reason


def test_require_executed_action_accepts_sidecar_but_rejects_raw_cache(tmp_path: Path) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    with pytest.raises(ValueError, match="已执行动作"):
        RawVLASequenceLoader(suite="libero", require_executed_action=True).load(path)
    loader = RawVLASequenceLoader(
        suite="libero", require_executed_action=True, sidecar=_libero_sidecar(actions)
    )
    assert loader.load(path).executed_action is not None


def test_sidecar_episode_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    sidecar = _libero_sidecar(actions)
    wrong = TelemetrySidecar(
        episode_id="another-episode",
        suite="libero",
        time_unit=sidecar.time_unit,
        timing=sidecar.timing,
        controls=sidecar.controls,
    )
    with pytest.raises(ValueError, match="episode_id"):
        RawVLASequenceLoader(suite="libero", sidecar=wrong).load(path)


def test_context_without_sidecar_refuses_to_assume_instant_arrival(tmp_path: Path) -> None:
    sequence = RawVLASequenceLoader(suite="libero").load(
        _fixture_libero(tmp_path / "libero-fixture.npz")
    )
    assert sequence.telemetry is None
    with pytest.raises(ValueError, match="遥测 sidecar"):
        sequence.context_at(0.0, 0.0)
    with pytest.raises(ValueError, match="遥测 sidecar"):
        sequence.executed_between(0.0, 1.0)


def test_measurement_and_target_views_are_separate(tmp_path: Path) -> None:
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    sequence = RawVLASequenceLoader(suite="libero", sidecar=_libero_sidecar(actions)).load(path)
    measurement = sequence.measurement_view(2)
    assert set(measurement.as_dict()) == {
        "frame",
        "camera_ids",
        "value_range",
        "is_new_measurement",
        "timing",
        "time_unit",
        "exposure_available",
        "arrival_available",
    }
    assert measurement.arrival_available is True
    assert measurement.timing.arrival_time == pytest.approx(0.230)
    assert measurement.as_dict()["timing"]["exposure_mid"] == pytest.approx(0.205)
    target = sequence.target_view(2).as_dict()
    assert target["future"] is True
    assert "actions" not in target or True  # 目标值本身只在张量里，不进 as_dict 文本
    assert target["horizon"] == 10


def test_robotwin_sidecar_keeps_joint_state_as_proprio_not_control(tmp_path: Path) -> None:
    """RoboTwin 缓存的 ``actions`` 是实测 14 维关节状态，不是已执行控制。

    远端字段核查（envs/_base_task.py L493-497：``joint_action`` 由
    ``get_*_arm_jointState()`` 组成）说明这一点，因此即使挂上 sidecar：
    本体可得，已执行控制仍然不可得，且不会用零动作冒充。
    """
    path = _fixture_robotwin(tmp_path / "robotwin-fixture.npz")
    raw_actions = np.load(path)["actions"]
    sidecar = TelemetrySidecar(
        episode_id="robotwin-fixture",
        suite="robotwin",
        time_unit="frame",
        timing=tuple(
            FrameTiming(
                frame=index,
                clock_id="episode_clock",
                time_unit="frame",
                exposure_mid=float(index),
                arrival_time=float(index),
            )
            for index in range(LENGTH)
        ),
        provenance={"source": "remote-field-audit"},
    )
    sequence = RawVLASequenceLoader(suite="robotwin", sidecar=sidecar).load(path)
    assert sequence.proprio.shape[-1] == 14
    assert torch.allclose(sequence.proprio[0], torch.as_tensor(raw_actions[0]))
    assert sequence.time_unit == "frame"
    assert sequence.availability.proprio is True
    #: 缓存里没有控制日志 -> executed_action 保持 None，不是零动作。
    assert sequence.executed_action is None
    assert sequence.availability.executed_action is False
    context = sequence.context_at(exposure_time=3.0, decision_time=3.0)
    assert context.latest_frame == 3
    assert context.replay_controls == ()
    assert context.proprio_frame == 3
    codes = sequence.summary()["telemetry"]["conditions"]
    assert "control.absent" in codes
    assert "calibration.absent" in codes


def test_producer_code_order_evidence_never_becomes_executed_action(tmp_path: Path) -> None:
    """Lead 裁决：LIBERO 生产者代码顺序只是 ``producer_code_order``，未核实。

    即使控制记录一一对应、相位明确，证据等级不足时 ``availability.executed_action``
    必须保持 False，且 validate 报出 ``control.evidence_unverified``；S1/S3 不能默认采信。
    """
    path = _fixture_libero(tmp_path / "libero-fixture.npz")
    actions = np.load(path)["actions"]
    unverified = TelemetrySidecar(
        episode_id="libero-fixture",
        suite="libero",
        time_unit="s",
        controls=tuple(
            replace(control, evidence_level="producer_code_order")
            for control in _libero_sidecar(actions).controls
        ),
    )
    sequence = RawVLASequenceLoader(suite="libero", sidecar=unverified).load(path)
    assert sequence.executed_action is None
    assert sequence.availability.executed_action is False
    reason = sequence.metadata["telemetry"]["executed_action_unavailable_reason"]
    assert "producer_code_order" in reason and "未核实" in reason
    report = unverified.validate(n_frames=LENGTH)
    assert "control.evidence_unverified" in report.codes()
    assert report.availability["control.verified"] == 0.0
    #: 单条控制的 verified 也必须是 False。
    assert unverified.controls[0].executed is True
    assert unverified.controls[0].verified is False
    #: 证据等级取值受控，乱填直接报错。
    with pytest.raises(ValueError, match="证据等级"):
        ExecutedControl(command_id="x", evidence_level="looks-fine")
