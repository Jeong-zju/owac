"""N04 在 RawVLA NPZ 上的测量序列加载器。

建立在 `src/owac/raw/rawvla_sequence_n04.py` 之上，**只输出测量张量与元数据**，不继承作者
的 PIL 图像输出与 RGB chroma target。按部署记录固定以下键与对齐：

LIBERO
  ``agentview_raw_uint8`` / ``wrist_raw_uint8``：两路三通道 uint8 pseudo-RAW
  （来源为 RGB 逆处理，不是原始 Bayer CFA）。
  ``robot_states[t]``：8 维本体；监督 ``actions[t:t+10]`` 为 7 维。

RoboTwin
  ``head_camera_raw_float32`` / ``left_camera_raw_float32`` / ``right_camera_raw_float32``：
  三路三通道 float32 HDR-derived RAW。
  作者缓存把 ``actions[t]`` 当作**当前 14 维关节状态**，监督从 ``actions[t+1:t+51]`` 开始；
  NPZ 的 16 维 ``robot_states`` 是末端位姿表示，不是关节状态。
  在线当前状态应来自真实观测的 ``joint_action/vector``。缓存没有可靠的“已执行控制输入”
  时，动作上下文标记为**缺失**，不拿未来监督动作当成已执行动作。

``availability`` 显式区分：本体缺失、已执行动作缺失、时间戳缺失、噪声模型缺失。
只有帧索引时 ``time_unit="frame"``，不编造秒数。开头 padding 的重复帧用
``new_measurement=False`` 标识；动作尾部的 padding 用独立监督 mask。

S0 之后本模块支持可选 **遥测 sidecar**（:mod:`owac.raw.telemetry`）。三条结构被显式分开：

* **普通测量**：``measurements`` / ``proprio`` / ``frame_ids``，由 :meth:`measurement_view` 取用；
* **训练目标**：``supervised_action`` / ``supervised_action_mask``，由 :meth:`target_view`
  取用，永远标记 ``future=True``；
* **在线可用上下文**：:meth:`context_at`，只含 ``decision_time`` 之前已到达的帧与已经
  执行的控制，绝不包含未来监督动作。

``sidecar=None``（默认）时行为与加入 sidecar 之前逐位一致：不补时间、不补已执行动作，
``time_unit`` 仍是 ``"frame"``。没有 sidecar 时调用 :meth:`context_at` /
:meth:`executed_between` 会明确报错，而不是假定"帧立即到达"。
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from owac.core.encoder_contract_n04 import EmbodimentContext
from owac.raw.telemetry import (
    VERIFIED_EVIDENCE_LEVELS,
    ExecutedControl,
    ExecutedInterval,
    FrameTiming,
    OnlineContext,
    TelemetrySidecar,
    interval_report,
)

__all__ = [
    "LIBERO_STATE_DIM",
    "LIBERO_ACTION_DIM",
    "ROBOTWIN_JOINT_DIM",
    "MeasurementView",
    "RawVLASequence",
    "RawVLASequenceLoader",
    "SequenceAvailability",
    "TargetView",
]

LIBERO_STATE_DIM = 8
LIBERO_ACTION_DIM = 7
LIBERO_ACTION_HORIZON = 10
ROBOTWIN_JOINT_DIM = 14
ROBOTWIN_ACTION_HORIZON = 50
ROBOTWIN_END_EFFECTOR_STATE_DIM = 16  # NPZ 的 robot_states：与 14 维关节状态不同

LIBERO_CAMERA_KEYS = {
    "agentview": "agentview_raw_uint8",
    "wrist": "wrist_raw_uint8",
}
ROBOTWIN_CAMERA_KEYS = {
    "head": "head_camera_raw_float32",
    "left": "left_camera_raw_float32",
    "right": "right_camera_raw_float32",
}


@dataclass
class SequenceAvailability:
    """每一项信息是否真的可得。缺失写 False，不用零值冒充。"""

    proprio: bool = False
    executed_action: bool = False
    timestamp: bool = False
    noise_model: bool = False
    exposure: bool = False
    intrinsics: bool = False
    pose: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "proprio": self.proprio,
            "executed_action": self.executed_action,
            "timestamp": self.timestamp,
            "noise_model": self.noise_model,
            "exposure": self.exposure,
            "intrinsics": self.intrinsics,
            "pose": self.pose,
        }


@dataclass(frozen=True)
class MeasurementView:
    """**普通测量**：某一帧实际测到的内容与其时间。

    这个结构里没有任何未来动作，也没有控制记录——它只回答"这一帧测到了什么、
    什么时候曝光、什么时候到达"。
    """

    frame: int
    camera_ids: tuple[str, ...]
    value_range: dict[str, tuple[float, float]]
    is_new_measurement: bool
    timing: FrameTiming | None
    time_unit: str
    exposure_available: bool
    arrival_available: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "camera_ids": list(self.camera_ids),
            "value_range": {k: list(v) for k, v in self.value_range.items()},
            "is_new_measurement": self.is_new_measurement,
            "timing": None if self.timing is None else self.timing.as_dict(),
            "time_unit": self.time_unit,
            "exposure_available": self.exposure_available,
            "arrival_available": self.arrival_available,
        }


@dataclass(frozen=True)
class TargetView:
    """**训练目标**：这一帧之后的专家动作，``future=True``。

    这个结构只能进损失，不能进在线上下文；把它当"已执行控制"使用会直接把未来信息
    泄漏进预测。``valid_steps`` 是去掉尾部 padding 后真正参与损失的步数。
    """

    frame: int
    actions: torch.Tensor
    mask: torch.Tensor
    horizon: int
    action_dim: int
    future: bool = True

    @property
    def valid_steps(self) -> int:
        return int((self.mask > 0).sum())

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "valid_steps": self.valid_steps,
            "future": self.future,
        }


@dataclass
class RawVLASequence:
    """一条已经读入内存的序列。

    ``measurements[camera]`` 形状 ``[T, C, H, W]``，浮点；``value_range`` 记录该数组在
    原始文件里的数值范围与归一化方式，禁止无条件按 255 归一化或按帧最大值归一化。
    """

    suite: Literal["libero", "robotwin"]
    measurements: dict[str, torch.Tensor]
    proprio: torch.Tensor
    #: ``None`` 表示缓存里**没有**可靠的已执行动作记录（不是"全零动作"）。
    executed_action: torch.Tensor | None
    supervised_action: torch.Tensor
    supervised_action_mask: torch.Tensor
    frame_ids: torch.Tensor
    episode_id: str
    camera_ids: tuple[str, ...]
    new_measurement: torch.Tensor
    time_unit: str
    availability: SequenceAvailability
    value_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    #: 可选遥测 sidecar。``None`` 表示这条序列**没有**到达时刻与控制日志记录。
    telemetry: TelemetrySidecar | None = None

    def __len__(self) -> int:
        return int(self.proprio.shape[0])

    def camera(self, name: str) -> torch.Tensor:
        return self.measurements[name]

    # -- 三类结构的显式分离 -----------------------------------------------------------

    def measurement_view(self, frame: int) -> MeasurementView:
        """普通测量视图：不含任何未来动作。"""
        timing = None if self.telemetry is None else self.telemetry.timing_for(frame)
        return MeasurementView(
            frame=frame,
            camera_ids=self.camera_ids,
            value_range=self.value_range,
            is_new_measurement=bool(self.new_measurement[frame]),
            timing=timing,
            time_unit=self.time_unit,
            exposure_available=bool(timing is not None and timing.has_exposure),
            arrival_available=bool(timing is not None and timing.arrival_time is not None),
        )

    def target_view(self, frame: int) -> TargetView:
        """训练目标视图：未来专家动作，永远 ``future=True``。"""
        return TargetView(
            frame=frame,
            actions=self.supervised_action[frame],
            mask=self.supervised_action_mask[frame],
            horizon=int(self.supervised_action.shape[1]),
            action_dim=int(self.supervised_action.shape[2]),
        )

    def telemetry_of(self) -> TelemetrySidecar:
        if self.telemetry is None:
            raise ValueError(
                "该序列没有遥测 sidecar：没有到达时刻、没有已执行控制记录。"
                "帧号与测量本身不含这些信息，不能假定帧立即到达或动作已经执行。"
            )
        return self.telemetry

    def context_at(self, exposure_time: float, decision_time: float) -> OnlineContext:
        """``decision_time`` 时刻在线真正可得的上下文（见 `OnlineContext`）。"""
        return self.telemetry_of().context_for(
            exposure_time=exposure_time,
            decision_time=decision_time,
            n_frames=len(self),
            proprio_available=bool(self.availability.proprio),
        )

    def executed_between(self, t0: float, t1: float) -> ExecutedInterval:
        """``(t0, t1]`` 内已经执行的控制片段，保序并检查覆盖与间隔。"""
        sidecar = self.telemetry_of()
        return interval_report(
            sidecar.controls,
            t0=t0,
            t1=t1,
            time_unit=sidecar.time_unit,
            exposure_lookup=sidecar.exposure_lookup(),
        )

    def summary(self) -> dict[str, Any]:
        payload = {
            "suite": self.suite,
            "episode_id": self.episode_id,
            "length": len(self),
            "cameras": list(self.camera_ids),
            "proprio_dim": int(self.proprio.shape[-1]),
            "executed_action_dim": None
            if self.executed_action is None
            else int(self.executed_action.shape[-1]),
            "supervised_action_dim": int(self.supervised_action.shape[-1]),
            "time_unit": self.time_unit,
            "availability": self.availability.as_dict(),
            "value_range": {k: list(v) for k, v in self.value_range.items()},
            "new_measurement_false": int((~self.new_measurement).sum()),
        }
        if self.telemetry is not None:
            report = self.telemetry.validate(n_frames=len(self))
            payload["telemetry"] = {
                "schema": self.telemetry.schema,
                "conditions": [condition.code for condition in report.conditions],
                "availability": report.availability,
            }
        return payload


def _require(npz, names: tuple[str, ...], *, where: str) -> None:
    missing = [n for n in names if n not in npz.files]
    if missing:
        raise KeyError(f"{where} 缺少 NPZ 键 {missing}；实际键：{sorted(npz.files)}")


def _to_chw(array: np.ndarray) -> torch.Tensor:
    """``[T,H,W,C]``（或 ``[T,H,W]``）-> ``[T,C,H,W]`` 的 float32 张量。

    **不做量程变换**：原始数值单位保留，范围另行记录。LIBERO uint8 与 RoboTwin
    float32 的单位不同，调用方必须按 ``value_range`` 决定后续处理。
    """
    tensor = torch.as_tensor(np.ascontiguousarray(array))
    if tensor.dim() == 3:  # [T,H,W] -> [T,1,H,W]
        tensor = tensor.unsqueeze(1)
    elif tensor.dim() == 4:
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"无法解释的测量数组形状 {tuple(tensor.shape)}")
    return tensor.to(torch.float32)


def npz_metadata(npz):
    if "metadata_json" not in npz.files:
        return {}
    raw = npz["metadata_json"].item()
    return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))


def source_identity(path, metadata=None):
    """Group all photometric/RGB/RAW variants of one source before splitting."""
    path = Path(path)
    if not path.exists():
        return path.stem
    if metadata is None:
        with np.load(path, allow_pickle=False) as npz:
            metadata = npz_metadata(npz)
    if metadata.get("dataset_name") and metadata.get("demo_key") and metadata.get("task_name"):
        return ":".join(str(metadata[k]) for k in ("dataset_name", "task_name", "demo_key"))
    if metadata.get("task_name") and "dataset_seed" in metadata:
        return f"robotwin:{metadata['task_name']}:{metadata['dataset_seed']}"
    if metadata.get("source_episode_id"):
        return str(metadata["source_episode_id"])
    # Conservative fallback for fixture layouts; variants share their episode directory.
    if path.parent.name.startswith("episode_"):
        return f"{path.parent.parent.name}:{path.parent.name}"
    return path.stem


class RawVLASequenceLoader:
    """把 RawVLA-Bench NPZ 读成 `RawVLASequence`。

    本加载器不做训练/评测的窗口切分；切片由 `windows` 提供，保证同一来源的 RGB/RAW/
    照明衍生版本落在同一侧（拆分由调用方按 ``episode_id`` 完成）。
    """

    def __init__(
        self,
        *,
        suite: Literal["libero", "robotwin"],
        require_executed_action: bool = False,
        sidecar: TelemetrySidecar | None = None,
    ) -> None:
        if suite not in ("libero", "robotwin"):
            raise ValueError(f"未知套件 {suite!r}")
        self.suite = suite
        self.require_executed_action = require_executed_action
        self.sidecar = sidecar

    def load(self, path: Path | str, *, episode_id: str | None = None) -> RawVLASequence:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到 NPZ 文件：{path}")
        with np.load(path, allow_pickle=False) as npz:
            if self.require_executed_action and self.sidecar is None:
                raise ValueError("该缓存没有可靠的已执行动作记录；不能使用未来监督动作")
            if self.suite == "libero":
                sequence = self._load_libero(npz, path, episode_id)
            else:
                sequence = self._load_robotwin(npz, path, episode_id)
        if self.sidecar is not None:
            sequence = self._apply_sidecar(sequence, self.sidecar)
            if self.require_executed_action and sequence.executed_action is None:
                raise ValueError(
                    "sidecar 没有给出逐帧一一对应的已执行控制；不能把未来监督动作当成已执行动作"
                )
        return sequence

    # -- sidecar -----------------------------------------------------------------

    @staticmethod
    def _apply_sidecar(sequence: RawVLASequence, sidecar: TelemetrySidecar) -> RawVLASequence:
        """把遥测挂到序列上，**只**把确实有记录的字段标成可得。

        * ``episode_id`` 不一致直接报错：挂错 sidecar 比没有 sidecar 更危险。
        * ``time_unit`` 只在 sidecar 有真实秒时才改为秒；帧号 sidecar 保持 ``"frame"``。
        * ``executed_action`` 只有在每一帧都恰好有一条 ``after_exposure`` 的已执行控制、
          数值维度一致、**且证据等级已核实**（``executed_log`` / ``replay_verified``）时才
          建立；否则保持 ``None`` 并写下原因，不用零动作补。仅有生产者代码顺序证据
          （``producer_code_order``）不算核实，见 `owac.raw.telemetry` 的证据等级说明。
        """
        if sidecar.episode_id and sequence.episode_id and sidecar.episode_id != sequence.episode_id:
            raise ValueError(
                f"sidecar episode_id {sidecar.episode_id!r} 与序列 {sequence.episode_id!r} 不一致"
            )
        length = len(sequence)
        if any(record.frame < 0 or record.frame >= length for record in sidecar.timing):
            raise ValueError("sidecar 含超出序列长度的帧号")
        availability = replace(
            sequence.availability,
            timestamp=sidecar.has_real_seconds
            and any(record.has_exposure for record in sidecar.timing),
            exposure=any(record.has_exposure for record in sidecar.timing),
            intrinsics=any(record.K is not None for record in sidecar.calibration),
            pose=any(record.T_A_from_B is not None for record in sidecar.calibration),
        )
        executed_action, reason = RawVLASequenceLoader._executed_action_from_sidecar(
            sidecar, length=length
        )
        availability = replace(availability, executed_action=executed_action is not None)
        metadata = dict(sequence.metadata)
        metadata["telemetry"] = {
            "schema": sidecar.schema,
            "time_unit": sidecar.time_unit,
            "provenance": dict(sidecar.provenance),
            "executed_action_unavailable_reason": reason,
            "unavailable_conditions": [
                condition.code for condition in sidecar.validate(n_frames=length).conditions
            ],
        }
        return replace(
            sequence,
            time_unit=sidecar.time_unit,
            availability=availability,
            telemetry=sidecar,
            executed_action=executed_action,
            metadata=metadata,
        )

    @staticmethod
    def _executed_action_from_sidecar(
        sidecar: TelemetrySidecar, *, length: int
    ) -> tuple[torch.Tensor | None, str | None]:
        """从 sidecar 重建逐帧已执行动作；条件不满足时返回 ``(None, 原因)``。"""
        by_frame: dict[int, list[ExecutedControl]] = {}
        for control in sidecar.controls:
            if not control.executed or control.frame is None:
                continue
            by_frame.setdefault(control.frame, []).append(control)
        if len(by_frame) != length:
            return None, f"只有 {len(by_frame)}/{length} 帧带已执行控制记录"
        dims = set()
        for controls in by_frame.values():
            if len(controls) != 1:
                return None, (
                    f"第 {controls[0].frame} 帧有 {len(controls)} 条已执行控制，不是一一对应"
                )
            if controls[0].dispatch_phase != "after_exposure":
                return None, (
                    f"第 {controls[0].frame} 帧的 dispatch_phase="
                    f"{controls[0].dispatch_phase!r}，无法确定相对该帧曝光的相位"
                )
            if controls[0].evidence_level not in VERIFIED_EVIDENCE_LEVELS:
                return None, (
                    f"第 {controls[0].frame} 帧的控制证据等级 "
                    f"{controls[0].evidence_level!r} 未核实（需 executed_log 或 "
                    "replay_verified）；真实 episode 级同版本重放核对前不建立 executed_action"
                )
            if controls[0].value is None:
                return None, f"第 {controls[0].frame} 帧的已执行控制没有数值"
            dims.add(np.asarray(controls[0].value, dtype=np.float64).shape)
        if len(dims) != 1:
            return None, f"各帧已执行控制数值维度不一致：{sorted(dims)}"
        dim = dims.pop()[0]
        if dim == 0:
            return None, "已执行控制数值为空"
        stacked = np.zeros((length, dim), dtype=np.float32)
        for frame, controls in by_frame.items():
            stacked[frame] = np.asarray(controls[0].value, dtype=np.float32)
        return torch.from_numpy(stacked), None

    # -- 在线因果上下文 ---------------------------------------------------------------

    def context_at(
        self, sequence: RawVLASequence, exposure_time: float, decision_time: float
    ) -> OnlineContext:
        """``decision_time`` 时刻在线可得的上下文；不含未来监督动作。"""
        return sequence.context_at(exposure_time, decision_time)

    def executed_between(self, sequence: RawVLASequence, t0: float, t1: float) -> ExecutedInterval:
        """``(t0, t1]`` 内已执行的控制片段，保序并检查覆盖与间隔。"""
        return sequence.executed_between(t0, t1)

    # -- LIBERO ------------------------------------------------------------------

    def _load_libero(self, npz, path: Path, episode_id: str | None) -> RawVLASequence:
        _require(npz, tuple(LIBERO_CAMERA_KEYS.values()), where="LIBERO")
        measurements = {
            name: _to_chw(npz[key]) for name, key in LIBERO_CAMERA_KEYS.items() if key in npz.files
        }
        length = next(iter(measurements.values())).shape[0]
        if any(m.shape[0] != length for m in measurements.values()):
            raise ValueError("camera frame counts do not align")
        metadata = npz_metadata(npz)
        _require(npz, ("robot_states", "actions"), where="LIBERO")
        proprio = torch.as_tensor(np.asarray(npz["robot_states"], dtype=np.float32))
        if proprio.shape[0] < length:
            raise ValueError("proprio frames shorter than measurements")
        proprio = proprio[:length]
        if proprio.shape[-1] != LIBERO_STATE_DIM:
            raise ValueError(
                f"LIBERO robot_states 期望 {LIBERO_STATE_DIM} 维，实际 {proprio.shape[-1]}"
            )
        supervised, mask = _slice_supervision(
            npz["actions"], length=length, start_offset=0, horizon=LIBERO_ACTION_HORIZON
        )
        if supervised.shape[-1] != LIBERO_ACTION_DIM:
            raise ValueError(
                f"LIBERO actions 期望 {LIBERO_ACTION_DIM} 维，实际 {supervised.shape[-1]}"
            )
        value_range = {name: (float(m.min()), float(m.max())) for name, m in measurements.items()}
        #: LIBERO 缓存没有可靠的“已执行控制输入”字段：不把未来的监督动作当成已执行动作。
        availability = SequenceAvailability(
            proprio=True,
            executed_action=False,
            timestamp=False,
            noise_model="raw_sensor_shot_noise" in metadata,
            exposure="lighting_ev" in metadata,
            intrinsics=False,
            pose=False,
        )
        return RawVLASequence(
            suite="libero",
            measurements={k: v[:length] for k, v in measurements.items()},
            proprio=proprio,
            executed_action=None,
            supervised_action=supervised,
            supervised_action_mask=mask,
            frame_ids=torch.arange(length),
            episode_id=episode_id or source_identity(path, metadata),
            camera_ids=tuple(measurements.keys()),
            new_measurement=torch.ones(length, dtype=torch.bool),
            time_unit="frame",
            availability=availability,
            value_range=value_range,
            metadata={
                **metadata,
                "source": str(path),
                "raw_dtype": "uint8",
                "note": "三通道 uint8 pseudo-RAW（RGB 逆处理），不是原始 Bayer CFA",
                "states_key": "robot_states",
                "supervision": f"actions[t:t+{LIBERO_ACTION_HORIZON}]",
            },
        )

    # -- RoboTwin ----------------------------------------------------------------

    def _load_robotwin(self, npz, path: Path, episode_id: str | None) -> RawVLASequence:
        _require(npz, tuple(ROBOTWIN_CAMERA_KEYS.values()), where="RoboTwin")
        measurements = {
            name: _to_chw(npz[key])
            for name, key in ROBOTWIN_CAMERA_KEYS.items()
            if key in npz.files
        }
        length = next(iter(measurements.values())).shape[0]
        if any(m.shape[0] != length for m in measurements.values()):
            raise ValueError("camera frame counts do not align")
        metadata = npz_metadata(npz)
        _require(npz, ("actions",), where="RoboTwin")
        actions_all = torch.as_tensor(np.asarray(npz["actions"], dtype=np.float32))
        #: 作者缓存协议：``actions[t]`` 是**当前** 14 维关节状态，监督从 ``actions[t+1]`` 起。
        if actions_all.shape[0] < length:
            raise ValueError("current joint state frames shorter than measurements")
        proprio = actions_all[:length]
        if proprio.shape[-1] != ROBOTWIN_JOINT_DIM:
            raise ValueError(
                f"RoboTwin actions 期望 {ROBOTWIN_JOINT_DIM} 维关节向量，实际 {proprio.shape[-1]}"
            )
        supervised, mask = _slice_supervision(
            actions_all,
            length=length,
            start_offset=1,
            horizon=ROBOTWIN_ACTION_HORIZON,
        )
        value_range = {name: (float(m.min()), float(m.max())) for name, m in measurements.items()}
        availability = SequenceAvailability(
            proprio=True,
            #: 缓存状态映射只用于复用作者数据协议：这里没有“已执行控制输入”的记录。
            executed_action=False,
            timestamp=False,
            noise_model="raw_sensor_shot_noise" in metadata,
            exposure="lighting_ev" in metadata,
            intrinsics=False,
            pose=False,
        )
        needs_executed = self.require_executed_action and self.sidecar is None
        if needs_executed and not availability.executed_action:
            raise ValueError("该缓存没有可靠的已执行动作记录；已按要求拒绝把它当作已执行动作使用")
        return RawVLASequence(
            suite="robotwin",
            measurements={k: v[:length] for k, v in measurements.items()},
            proprio=proprio,
            executed_action=None,
            supervised_action=supervised,
            supervised_action_mask=mask,
            frame_ids=torch.arange(length),
            episode_id=episode_id or source_identity(path, metadata),
            camera_ids=tuple(measurements.keys()),
            new_measurement=torch.ones(length, dtype=torch.bool),
            time_unit="frame",
            availability=availability,
            value_range=value_range,
            metadata={
                **metadata,
                "source": str(path),
                "raw_dtype": "float32",
                "note": "三通道 float32 HDR-derived RAW；范围依作者元数据，不无条件除以 255",
                "states_key": "actions[t]（当前关节状态）",
                "end_effector_states_dim": ROBOTWIN_END_EFFECTOR_STATE_DIM,
                "supervision": f"actions[t+1:t+{ROBOTWIN_ACTION_HORIZON + 1}]",
                "executed_action_source": "在线应使用真实观测的 joint_action/vector",
            },
        )

    # -- 窗口 ---------------------------------------------------------------------

    @staticmethod
    def windows(
        sequence: RawVLASequence,
        *,
        length: int,
        stride: int | None = None,
        burn_in: int = 0,
    ) -> Iterator[dict[str, Any]]:
        """按固定长度切窗口。

        * 起点之前的**开头 padding** 不存在：窗口从 0 开始时，前 ``burn_in`` 帧用
          ``new_measurement=False`` 标出（它们用于构造状态，不作为新证据重复计入损失）。
        * 动作尾部的 padding 由 ``supervised_action_mask`` 单独表达。
        """
        total = len(sequence)
        stride = stride or length
        starts = list(range(0, max(1, total - length + 1), stride))
        if not starts:
            starts = [0]
        for start in starts:
            end = min(start + length, total)
            if end - start < 2:
                continue
            index = torch.arange(start, end)
            new_measurement = torch.ones(end - start, dtype=torch.bool)
            frame_ids = sequence.frame_ids[index]
            # Burn-in contains real measurements. Loss exclusion is independent.
            loss_mask = torch.arange(end - start) >= burn_in
            yield {
                "start": start,
                "index": index,
                "measurements": {
                    cam: values[index] for cam, values in sequence.measurements.items()
                },
                "proprio": sequence.proprio[index],
                "executed_action": None
                if sequence.executed_action is None
                else sequence.executed_action[index],
                "supervised_action": sequence.supervised_action[index],
                "supervised_action_mask": sequence.supervised_action_mask[index],
                "frame_ids": frame_ids,
                "new_measurement": new_measurement,
                "loss_mask": loss_mask,
                "time_unit": sequence.time_unit,
                "availability": sequence.availability,
                "episode_id": sequence.episode_id,
                "metadata": sequence.metadata,
            }

    # -- v1.1 追加接口（见文件末尾说明） -------------------------------------------

    def load_window(
        self,
        path: Path | str,
        *,
        length: int,
        start: int = 0,
        episode_id: str | None = None,
    ) -> RawVLASequence:
        """只读 ``[start, start+length)`` 帧，其余语义与 :meth:`load` 一致。

        ``actions`` 会按目标相位多读 ``V11_ACTION_SLACK[self.suite]`` 帧，所以窗口内每帧的
        ``target_view`` 与整段加载逐位一致；``frame_ids`` 保留全局帧号。文件在该区间更短时
        返回较短序列，调用方按 ``len(sequence)`` 判断。带遥测 sidecar 时按整段语义处理，
        不在这里做窗口化（v1.1 不补 sidecar）。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到 NPZ 文件：{path}")
        if self.sidecar is not None:
            raise ValueError("load_window 不做 sidecar 窗口化；请用整段 load 或去掉 sidecar")
        if self.require_executed_action:
            raise ValueError("该缓存没有可靠的已执行动作记录；不能使用未来监督动作")
        view = _NpzWindowView(path, start=start, length=length, slack=V11_ACTION_SLACK[self.suite])
        if self.suite == "libero":
            sequence = self._load_libero(view, path, episode_id)
        else:
            sequence = self._load_robotwin(view, path, episode_id)
        if not len(sequence):
            raise ValueError(f"{path} 在 [{start}, {start + length}) 上没有帧")
        return replace(sequence, frame_ids=torch.arange(start, start + len(sequence)))

    def v11_windows(
        self,
        sequence: RawVLASequence,
        *,
        burn_in: int = 16,
        gradient: int = 32,
        stride: int | None = None,
        supervision_points: int = 4,
        history: int = 4,
    ) -> Iterator[dict[str, Any]]:
        """窗口计划 + 每帧本体上下文（只用 ``≤ t`` 的记录，未来动作不进入）。"""
        for plan in v11_window_plan(
            len(sequence),
            burn_in=burn_in,
            gradient=gradient,
            stride=stride,
            supervision_points=supervision_points,
        ):
            record = dict(plan)
            record["episode_id"] = sequence.episode_id
            record["frame_ids"] = tuple(
                int(value)
                for value in sequence.frame_ids[plan["start"] : plan["start"] + plan["length"]]
            )
            record["proprio_contexts"] = [
                proprio_context_at(sequence, plan["start"] + offset, history=history)
                for offset in range(plan["length"])
            ]
            yield record

    def proprio_context_at(
        self, sequence: RawVLASequence, frame: int, *, history: int = 4
    ) -> EmbodimentContext:
        """``frame`` 时刻的本体上下文，见模块级 `proprio_context_at`。"""
        return proprio_context_at(sequence, frame, history=history)


def _slice_supervision(
    actions: np.ndarray, *, length: int, start_offset: int, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """切出监督动作窗口并标记尾部 padding。

    返回 ``(supervised [T, horizon, A], mask [T, horizon])``。末尾不足 ``horizon`` 的部分
    用零填充并把 ``mask`` 置 0，损失不重复计入填充项。
    """
    array = np.asarray(actions, dtype=np.float32)
    dim = array.shape[-1]
    out = np.zeros((length, horizon, dim), dtype=np.float32)
    mask = np.zeros((length, horizon), dtype=np.float32)
    for t in range(length):
        start = t + start_offset
        end = min(start + horizon, array.shape[0])
        if end <= start:
            continue
        count = end - start
        out[t, :count] = array[start:end]
        mask[t, :count] = 1.0
    return torch.from_numpy(out), torch.from_numpy(mask)


# --------------------------------------------------------------------------------------
# v1.1 S2 追加接口（只新增名字；既有方法语义、签名不变）
# --------------------------------------------------------------------------------------

#: 目标相位在窗口末尾之后额外需要的帧数：
#: LIBERO ``actions[t:t+10]``、RoboTwin ``actions[t+1:t+51]``。
V11_ACTION_SLACK = {"libero": LIBERO_ACTION_HORIZON - 1, "robotwin": ROBOTWIN_ACTION_HORIZON}


def v11_supervision_offsets(gradient: int, points: int = 4) -> tuple[int, ...]:
    """把梯度窗口等分成 ``points`` 段并取每段中点，作为动作监督点（不按难易挑选）。"""
    if points < 1 or gradient < points or gradient % points:
        raise ValueError(f"gradient={gradient} 必须能被监督点数 {points} 整除")
    stride = gradient // points
    return tuple(index * stride + stride // 2 for index in range(points))


def v11_window_plan(
    total: int,
    *,
    burn_in: int = 16,
    gradient: int = 32,
    stride: int | None = None,
    supervision_points: int = 4,
) -> list[dict[str, Any]]:
    """``burn_in + gradient`` 帧窗口计划（默认 16 + 32）。

    窗口只取同一 episode 内的连续记录，不跨边界、不补造开头帧；``start`` 是全局帧号，
    ``supervision_index`` 是窗口内的动作监督帧（相对窗口起点）。
    """
    if burn_in < 0 or gradient < 1:
        raise ValueError("burn_in 不能为负且 gradient 至少为 1")
    length = burn_in + gradient
    step = length if stride is None else int(stride)
    if step < 1:
        raise ValueError("stride 至少为 1")
    offsets = v11_supervision_offsets(gradient, supervision_points)
    return [
        {
            "start": start,
            "length": length,
            "burn_in": burn_in,
            "gradient": gradient,
            "supervision_offsets": offsets,
            "supervision_index": tuple(burn_in + offset for offset in offsets),
        }
        for start in range(0, max(0, int(total) - length + 1), step)
    ]


def proprio_context_at(
    sequence: RawVLASequence, frame: int, *, history: int = 4
) -> EmbodimentContext:
    """无 sidecar 的本体上下文：只用 ``≤ frame`` 的真实记录。

    * 缓存里没有可靠的已执行控制时 ``executed_action=None``、``executed_action_available=False``，
      不拿未来监督动作补上；
    * 窗口开头不足 ``history`` 帧时重复该 episode 的第一帧真实值，并把重复帧数写进
      ``metadata["padded_frames"]``（不编造数值、不借用未来帧）；
    * 返回结构里没有任何未来动作／未来本体：``supervised_action`` 只由 `target_view` 取用。
    """
    total = len(sequence)
    if not 0 <= frame < total:
        raise IndexError(f"帧号 {frame} 超出序列长度 {total}")
    if not sequence.availability.proprio:
        raise ValueError("该序列没有本体记录，不能构造本体上下文")
    width = max(1, int(history) + 1)
    first = max(0, frame - int(history))
    index = list(range(first, frame + 1))
    padded = width - len(index)
    index = [index[0]] * padded + index
    proprio = sequence.proprio[torch.as_tensor(index, dtype=torch.long)]
    executed = None if sequence.executed_action is None else sequence.executed_action[frame]
    return EmbodimentContext(
        proprio=proprio,
        proprio_available=True,
        executed_action=executed,
        executed_action_available=executed is not None,
        time_unit=sequence.time_unit,
        metadata={
            "history_frames": width,
            "padded_frames": padded,
            "frame": int(frame),
            "platform": sequence.suite,
            "proprio_key": sequence.metadata.get("states_key", "proprio"),
            "time_unit": sequence.time_unit,
            "future_action_excluded": True,
        },
    )


def _npy_member_header(stream) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError(f"不支持的 npy 版本 {version}")


def npz_member_shape(path: Path | str, key: str) -> tuple[int, ...]:
    """只解析成员头得到数组形状：计划窗口时不必解压整段测量。"""
    path = Path(path)
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise KeyError(f"{path} 缺少成员 {member}；实际成员：{sorted(archive.namelist())}")
        with archive.open(member) as stream:
            shape, _, _ = _npy_member_header(stream)
    return tuple(int(value) for value in shape)


class _NpzWindowView:
    """把 NPZ 的帧区间伪装成 ``np.load`` 结果，供既有 ``_load_*`` 复用。

    RoboTwin 单文件可达 GB 级；整段 ``np.load`` 再切片会读完整份成员。这里按 npy 头计算
    帧字节偏移，只读 ``[start, start+length)``（``actions`` 另加目标相位需要的 slack）。
    """

    #: 小于此字节数的成员（如 ``metadata_json``）整段读出，不做偏移计算。
    SMALL_MEMBER_BYTES = 1 << 20

    def __init__(self, path: Path | str, *, start: int, length: int, slack: int = 0) -> None:
        self._path = Path(path)
        self._start = max(0, int(start))
        self._length = int(length)
        self._slack = int(slack)
        if self._length < 1:
            raise ValueError("窗口长度至少为 1")
        with zipfile.ZipFile(self._path) as archive:
            self._members = {
                info.filename[: -len(".npy")]: info
                for info in archive.infolist()
                if info.filename.endswith(".npy")
            }

    @property
    def files(self) -> list[str]:
        return list(self._members)

    def __getitem__(self, key: str):
        member = self._members.get(key)
        if member is None:
            raise KeyError(f"{self._path} 没有成员 {key!r}；实际成员：{sorted(self._members)}")
        extra = self._slack if key == "actions" else 0
        if member.file_size <= self.SMALL_MEMBER_BYTES:
            with np.load(self._path, allow_pickle=False) as npz:
                array = npz[key]
            if array.ndim == 0:
                return array
            start = min(self._start, int(array.shape[0]))
            rows = max(0, min(self._length + extra, int(array.shape[0]) - start))
            return array[start : start + rows]
        with zipfile.ZipFile(self._path) as archive, archive.open(member) as stream:
            shape, fortran, dtype = _npy_member_header(stream)
            if fortran or not shape:
                raise ValueError(f"{self._path} 成员 {key!r} 形状 {shape} 不能按帧切分")
            header_bytes = stream.tell()
            frame_bytes = int(dtype.itemsize) * int(np.prod(shape[1:], dtype=np.int64))
            start = min(self._start, int(shape[0]))
            rows = max(0, min(self._length + extra, int(shape[0]) - start))
            stream.seek(header_bytes + start * frame_bytes)
            payload = stream.read(rows * frame_bytes)
            if len(payload) != rows * frame_bytes:
                raise ValueError(f"{self._path} 成员 {key!r} 读取不完整")
            return np.frombuffer(payload, dtype=dtype).reshape((rows, *shape[1:]))

    def __contains__(self, key: str) -> bool:
        return key in self._members
