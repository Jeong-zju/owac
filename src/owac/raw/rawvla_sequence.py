"""N04 在 RawVLA NPZ 上的测量序列加载器。

建立在 `src/owac/raw/rawvla_sequence.py` 之上，**只输出测量张量与元数据**，不继承作者
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
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

__all__ = [
    "LIBERO_STATE_DIM",
    "LIBERO_ACTION_DIM",
    "ROBOTWIN_JOINT_DIM",
    "RawVLASequence",
    "RawVLASequenceLoader",
    "SequenceAvailability",
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

    def __len__(self) -> int:
        return int(self.proprio.shape[0])

    def camera(self, name: str) -> torch.Tensor:
        return self.measurements[name]

    def summary(self) -> dict[str, Any]:
        return {
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
    ) -> None:
        if suite not in ("libero", "robotwin"):
            raise ValueError(f"未知套件 {suite!r}")
        self.suite = suite
        self.require_executed_action = require_executed_action

    def load(self, path: Path | str, *, episode_id: str | None = None) -> RawVLASequence:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到 NPZ 文件：{path}")
        with np.load(path, allow_pickle=False) as npz:
            if self.require_executed_action:
                raise ValueError("该缓存没有可靠的已执行动作记录；不能使用未来监督动作")
            if self.suite == "libero":
                return self._load_libero(npz, path, episode_id)
            return self._load_robotwin(npz, path, episode_id)

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
        if self.require_executed_action and not availability.executed_action:
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
