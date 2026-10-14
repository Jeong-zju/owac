"""N04 S0 遥测 sidecar：一帧**实际测到了什么、什么时候到达、上一帧后真正执行了哪些控制**。

本模块只保存数据与校验语义，不导入 torch、不初始化设备、不产生任何"猜测值"。三条硬规则：

1. **缺失即 None + availability False**：任何物理时间字段没有记录就写 ``None``，
   ``availability`` 里对应项为 ``False``；不用 ``0``、不用帧号、不用"上一次的值"冒充。
2. **帧号不是秒**：只有帧索引时 ``time_unit="frame"``。此时时间字段的数值是帧序号，
   仍是可排序的真实顺序量，但不是秒，禁止把它换算成秒。
3. **未执行的动作不是零动作**：``planned`` / ``dropped`` / ``unknown`` 状态的控制永远
   不进入"已执行"序列；它们只出现在不可用/缺口清单里。

``availability`` 在每个类型上都是**由字段派生**的只读视图（``None`` -> ``False``），
避免出现"标了 True 但值是 None"或"值写了 0 就当有测量"的不一致。

证据等级（schema 字段 ``ExecutedControl.evidence_level``）
----------------------------------------------------------

控制记录"是不是真的执行过"不是二值的，必须显式记录它凭什么可信：

* ``executed_log``——执行器侧日志直接记录了下发值（最强，可进入执行重放）；
* ``replay_verified``——同版本重放后逐帧核对一致（可进入执行重放）；
* ``producer_code_order``——只有数据生产代码的控制流顺序支持（例如"先渲染后 step"），
  **没有**任何 episode 级核对；不得据此把 ``availability.executed_action`` 置 True；
* ``inferred``——由其它字段推断出来，既非日志也非重放；
* ``unknown``——未声明，等价于不可采信。

只有 :data:`VERIFIED_EVIDENCE_LEVELS` 里的等级才允许进入"已执行控制"的正式使用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

__all__ = [
    "CONTROL_STATUSES",
    "EVIDENCE_LEVELS",
    "EXECUTED_STATUSES",
    "VERIFIED_EVIDENCE_LEVELS",
    "CalibrationRecord",
    "ExecutedControl",
    "ExecutedInterval",
    "FrameTiming",
    "OnlineContext",
    "TelemetryReport",
    "TelemetrySidecar",
    "UnavailableCondition",
    "interval_report",
]

#: 控制记录允许的执行状态。``planned`` 不是"还没执行"以外的任何意思，绝不参与重放。
ControlStatus = Literal["executed", "clipped", "dropped", "planned", "unknown"]
CONTROL_STATUSES: tuple[str, ...] = ("executed", "clipped", "dropped", "planned", "unknown")
#: 只有这两个状态代表"确实作用到了执行器"（``clipped`` 表示实际下发值被限幅改写）。
EXECUTED_STATUSES: frozenset[str] = frozenset({"executed", "clipped"})

#: 一条控制记录凭什么可信（见模块文档"证据等级"）。
EvidenceLevel = Literal[
    "executed_log", "replay_verified", "producer_code_order", "inferred", "unknown"
]
EVIDENCE_LEVELS: tuple[str, ...] = (
    "executed_log",
    "replay_verified",
    "producer_code_order",
    "inferred",
    "unknown",
)
#: 允许进入执行重放/``executed_action`` 重建的等级；其余一律视为未核实。
VERIFIED_EVIDENCE_LEVELS: frozenset[str] = frozenset({"executed_log", "replay_verified"})

TimeUnit = Literal["s", "ms", "frame"]
#: 控制相对该帧曝光的下发相位；未知写 ``unknown``，不默认成任何一侧。
DispatchPhase = Literal["before_exposure", "after_exposure", "unknown"]


def _frozen(array: np.ndarray | None, dtype: Any = np.float64) -> np.ndarray | None:
    if array is None:
        return None
    out = np.array(array, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _finite_times(**values: float | None) -> None:
    for name, value in values.items():
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{name} must be finite or None")


@dataclass(frozen=True)
class FrameTiming:
    """单帧的曝光与到达时间。

    ``time_unit`` 是所有这些数值的单位；``clock_id`` 标识这些数值用的是哪个时钟
    （不同时钟的数值不可直接相减）。没有任何时间记录时保留 ``clock_id=""`` 与全
    ``None``，而不是补一个 0。

    ``availability`` 是派生属性：字段为 ``None`` 即为 ``False``。因此"0.0 秒的曝光
    开始"与"没有曝光开始记录"是两件不同的事。

    ``rolling_shutter_row_time`` 只有确实存在滚动快门行读出时间时才填；全局快门或
    未记录时保持 ``None``，``rolling_shutter_rows`` 同理。
    """

    frame: int
    clock_id: str = ""
    time_unit: TimeUnit = "frame"
    exposure_start: float | None = None
    exposure_mid: float | None = None
    exposure_end: float | None = None
    arrival_time: float | None = None
    rolling_shutter_row_time: float | None = None
    rolling_shutter_rows: int | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if self.time_unit not in ("s", "ms", "frame"):
            raise ValueError(f"未知 time_unit {self.time_unit!r}")
        _finite_times(
            exposure_start=self.exposure_start,
            exposure_mid=self.exposure_mid,
            exposure_end=self.exposure_end,
            arrival_time=self.arrival_time,
            rolling_shutter_row_time=self.rolling_shutter_row_time,
        )
        if (self.exposure_start is not None and self.exposure_end is not None) and (
            self.exposure_start > self.exposure_end
        ):
            raise ValueError(
                f"frame {self.frame}: exposure_start {self.exposure_start} 晚于 exposure_end "
                f"{self.exposure_end}"
            )
        if self.rolling_shutter_row_time is not None and self.rolling_shutter_rows is None:
            raise ValueError(
                f"frame {self.frame}: 有行读出时间却没有行数；行读出总时间不可推算，不补默认值"
            )

    @property
    def availability(self) -> dict[str, bool]:
        return {
            "exposure_start": self.exposure_start is not None,
            "exposure_mid": self.exposure_mid is not None,
            "exposure_end": self.exposure_end is not None,
            "arrival_time": self.arrival_time is not None,
            "rolling_shutter_row_time": self.rolling_shutter_row_time is not None,
        }

    @property
    def has_exposure(self) -> bool:
        return any(
            value is not None
            for value in (self.exposure_start, self.exposure_mid, self.exposure_end)
        )

    def exposure_time(self) -> float | None:
        """本帧的代表曝光时刻，退化顺序如下。

        1. ``exposure_mid``——真实曝光中点，物理上最接近"这一帧测量对应哪一刻"。
        2. ``exposure_start`` 与 ``exposure_end`` 的算术平均——只有两端时用中点近似，
           它**不是**实测中点，调用方如需区分请读 ``availability["exposure_mid"]``。
        3. ``exposure_start``。
        4. ``exposure_end``。
        5. 都没有则 ``None``（禁止用帧号或 0 兜底）。

        返回值单位是 ``time_unit``：``"frame"`` 时它是帧序号而不是秒。
        """
        if self.exposure_mid is not None:
            return self.exposure_mid
        if self.exposure_start is not None and self.exposure_end is not None:
            return 0.5 * (self.exposure_start + self.exposure_end)
        if self.exposure_start is not None:
            return self.exposure_start
        if self.exposure_end is not None:
            return self.exposure_end
        return None

    def arrival_delay(self) -> float | None:
        """到达相对本帧曝光结束的延迟（``arrival_time - exposure_end``）。

        曝光结束时间缺失时返回 ``None``：拿曝光开始算出的"延迟"会系统性偏大，
        不能当成同一个量。
        """
        if self.arrival_time is None or self.exposure_end is None:
            return None
        return float(self.arrival_time - self.exposure_end)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "clock_id": self.clock_id,
            "time_unit": self.time_unit,
            "exposure_start": self.exposure_start,
            "exposure_mid": self.exposure_mid,
            "exposure_end": self.exposure_end,
            "arrival_time": self.arrival_time,
            "rolling_shutter_row_time": self.rolling_shutter_row_time,
            "rolling_shutter_rows": self.rolling_shutter_rows,
            "source": self.source,
            "availability": self.availability,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrameTiming:
        data = dict(payload)
        data.pop("availability", None)
        return cls(**data)


@dataclass(frozen=True)
class ExecutedControl:
    """一条**下发后**的控制记录。

    ``value`` 是真正下发出去的数值（限幅/变换之后的），不是规划器的期望值，也不是
    未来监督动作。``mode`` / ``units`` 说明它的物理含义（例如
    ``mode="osc_delta_eef"``、``units="m+rad"``），换一套坐标系或百分比夹爪时必须另
    记一条，不能沿用同一个 ``units``。

    ``frame`` 是这条控制下发时对应的测量帧（如果有）；``dispatch_phase`` 说明它相对该
    帧曝光是发生在之前还是之后。这个相位决定"帧 t 的观测能看到哪些控制"，缺失时写
    ``unknown``，调用方不得自行假定。

    ``source`` 记录来源日志位置（文件+行号/记录 id），便于回溯；没有来源就留空串，
    并把 ``availability["source"]`` 置 False。
    """

    command_id: str
    value: np.ndarray | None = None
    mode: str = "unknown"
    units: str = "unknown"
    status: ControlStatus = "unknown"
    #: 该记录凭什么可信；默认 ``"unknown"``，即默认**不采信**。
    evidence_level: EvidenceLevel = "unknown"
    command_time: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    frame: int | None = None
    dispatch_phase: DispatchPhase = "unknown"
    time_unit: TimeUnit = "s"
    source: str = ""
    note: str = ""
    # Empty inherits the sidecar's single clock; other clocks require explicit conversion.
    clock_id: str = ""

    def __post_init__(self) -> None:
        _finite_times(
            command_time=self.command_time, start_time=self.start_time, end_time=self.end_time
        )
        if self.status not in CONTROL_STATUSES:
            raise ValueError(f"未知执行状态 {self.status!r}；允许 {CONTROL_STATUSES}")
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ValueError(f"未知证据等级 {self.evidence_level!r}；允许 {EVIDENCE_LEVELS}")
        if self.time_unit not in ("s", "ms", "frame"):
            raise ValueError(f"未知 time_unit {self.time_unit!r}")
        if self.start_time is not None and self.end_time is not None:
            if self.start_time > self.end_time:
                raise ValueError(
                    f"{self.command_id}: start_time {self.start_time} 晚于 end_time {self.end_time}"
                )
        if self.executed and self.value is not None:
            values = np.asarray(self.value, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{self.command_id}: 已执行控制的数值必须有限，实际含 NaN/Inf")

    @property
    def availability(self) -> dict[str, bool]:
        return {
            "value": self.value is not None,
            "command_time": self.command_time is not None,
            "interval": self.start_time is not None and self.end_time is not None,
            "frame": self.frame is not None,
            "source": bool(self.source),
        }

    @property
    def executed(self) -> bool:
        """``True`` 只意味着"确实作用到了执行器"，不表示数值没有被限幅。"""
        return self.status in EXECUTED_STATUSES

    @property
    def verified(self) -> bool:
        """证据等级是否达到可正式使用的门槛（``status`` 可信 + 来源已核实）。"""
        return self.executed and self.evidence_level in VERIFIED_EVIDENCE_LEVELS

    @property
    def has_interval(self) -> bool:
        return self.start_time is not None and self.end_time is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "clock_id": self.clock_id,
            "value": None if self.value is None else [float(v) for v in self.value],
            "mode": self.mode,
            "units": self.units,
            "status": self.status,
            "evidence_level": self.evidence_level,
            "verified": self.verified,
            "command_time": self.command_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "frame": self.frame,
            "dispatch_phase": self.dispatch_phase,
            "time_unit": self.time_unit,
            "source": self.source,
            "note": self.note,
            "availability": self.availability,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutedControl:
        data = dict(payload)
        data.pop("availability", None)
        data.pop("verified", None)
        value = data.get("value")
        data["value"] = None if value is None else np.asarray(value, dtype=np.float64)
        return cls(**data)


@dataclass(frozen=True)
class CalibrationRecord:
    """一台相机的标定与坐标约定。

    ``K`` 为 ``[3,3]`` 像素内参，``T_A_from_B`` 为 ``[4,4]`` 齐次变换，表示把 **B 系**
    坐标变换到 **A 系**（``frame_a`` / ``frame_b`` 记录两个系的名称）。约定写进字段
    而不是靠注释：``coordinate_convention`` 明确像素原点、y 轴方向、``align_corners``
    等；解析不同来源的标定必须先显式转换，不能直接拼接。

    ``time_offset`` 是该相机时钟相对机器人时钟的偏移（单位 ``time_unit``）。未测出时
    写 ``None``；用 0 冒充会让"时间未标定"看起来像"已同步"。
    """

    camera_id: str
    calibration_version: str = ""
    K: np.ndarray | None = None
    T_A_from_B: np.ndarray | None = None
    frame_a: str = ""
    frame_b: str = ""
    distortion_model: str = "none"
    distortion: np.ndarray | None = None
    coordinate_convention: str = ""
    time_offset: float | None = None
    time_unit: TimeUnit = "s"
    provenance: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.K is not None and np.asarray(self.K).shape != (3, 3):
            raise ValueError(f"{self.camera_id}: K 必须是 [3,3]，实际 {np.asarray(self.K).shape}")
        if self.T_A_from_B is not None:
            matrix = np.asarray(self.T_A_from_B, dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"{self.camera_id}: T_A_from_B 必须是 [4,4]，实际 {matrix.shape}")
            if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
                raise ValueError(f"{self.camera_id}: T_A_from_B 最后一行必须是 [0,0,0,1]")
        if self.T_A_from_B is not None and (not self.frame_a or not self.frame_b):
            raise ValueError(f"{self.camera_id}: 有变换矩阵就必须声明 frame_a / frame_b")
        if self.time_unit not in ("s", "ms", "frame"):
            raise ValueError(f"未知 time_unit {self.time_unit!r}")

    @property
    def availability(self) -> dict[str, bool]:
        return {
            "K": self.K is not None,
            "T_A_from_B": self.T_A_from_B is not None,
            "distortion": self.distortion is not None,
            "distortion_model": self.distortion_model not in ("", "none"),
            "coordinate_convention": bool(self.coordinate_convention),
            "time_offset": self.time_offset is not None,
            "provenance": bool(self.provenance),
        }

    @property
    def available(self) -> bool:
        """投影计算所需的最小集合：内参、外参、坐标约定。"""
        return (
            self.K is not None and self.T_A_from_B is not None and bool(self.coordinate_convention)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "calibration_version": self.calibration_version,
            "K": None if self.K is None else np.asarray(self.K, dtype=np.float64).tolist(),
            "T_A_from_B": None
            if self.T_A_from_B is None
            else np.asarray(self.T_A_from_B, dtype=np.float64).tolist(),
            "frame_a": self.frame_a,
            "frame_b": self.frame_b,
            "distortion_model": self.distortion_model,
            "distortion": None
            if self.distortion is None
            else [float(v) for v in np.asarray(self.distortion).reshape(-1)],
            "coordinate_convention": self.coordinate_convention,
            "time_offset": self.time_offset,
            "time_unit": self.time_unit,
            "provenance": self.provenance,
            "note": self.note,
            "availability": self.availability,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationRecord:
        data = dict(payload)
        data.pop("availability", None)
        data["K"] = None if data.get("K") is None else _frozen(data["K"])
        data["T_A_from_B"] = None if data.get("T_A_from_B") is None else _frozen(data["T_A_from_B"])
        data["distortion"] = None if data.get("distortion") is None else _frozen(data["distortion"])
        return cls(**data)


@dataclass(frozen=True)
class UnavailableCondition:
    """一个具体的"这项信息不可用/不一致"条目。"""

    code: str
    scope: str
    detail: str
    frame: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "scope": self.scope, "frame": self.frame, "detail": self.detail}


@dataclass(frozen=True)
class TelemetryReport:
    """episode 级校验结果：结构化不可用条件清单 + 各字段可得比例。"""

    episode_id: str
    n_frames: int | None
    conditions: tuple[UnavailableCondition, ...]
    availability: dict[str, float]

    @property
    def ok(self) -> bool:
        """没有任何**错误级**条件（``time.frame_index_only`` 之类是事实说明，不是错误）。"""
        return not self.conditions

    def codes(self) -> tuple[str, ...]:
        return tuple(condition.code for condition in self.conditions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "n_frames": self.n_frames,
            "ok": self.ok,
            "availability": dict(self.availability),
            "conditions": [condition.as_dict() for condition in self.conditions],
        }


@dataclass(frozen=True)
class OnlineContext:
    """``decision_time`` 这一刻**真正可得**的上下文。

    选择规则：一帧只有当它的 ``arrival_time <= decision_time`` 时才可见；这个约束对
    图像与本体同时生效（缓存里本体与图像同帧对齐，没有独立的更晚本体采样时刻）。
    ``requested_exposure_time`` 之后曝光、或截止 ``decision_time`` 仍未到达的帧一律
    不可见。

    ``replay_controls`` 是"迟到帧按曝光时刻回到缓存更新，再用已发生的控制重放到当前"
    里的重放段：``latest_exposure_time`` 之后、``decision_time`` 之前**已经执行**的控制，
    保持命令顺序。``planned`` / ``dropped`` / ``unknown`` 只进 ``pending_control_ids``。

    ``target_actions_included`` 恒为 ``False``：未来监督动作属于训练目标，不进在线上下文。
    它作为字段存在，是为了让调用方可以断言这一点，而不是靠约定。
    """

    decision_time: float
    time_unit: TimeUnit
    requested_exposure_time: float | None
    visible_frames: tuple[int, ...]
    latest_frame: int | None
    latest_exposure_time: float | None
    latest_arrival_time: float | None
    arrival_delay: float | None
    age_since_exposure: float | None
    late_frame: bool
    missing_frames: tuple[int, ...]
    replay_controls: tuple[ExecutedControl, ...]
    pending_control_ids: tuple[str, ...]
    proprio_frame: int | None
    availability: dict[str, bool]
    target_actions_included: bool = False
    note: str = ""
    processing_delay: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_time": self.decision_time,
            "time_unit": self.time_unit,
            "requested_exposure_time": self.requested_exposure_time,
            "visible_frames": list(self.visible_frames),
            "latest_frame": self.latest_frame,
            "latest_exposure_time": self.latest_exposure_time,
            "latest_arrival_time": self.latest_arrival_time,
            "arrival_delay": self.arrival_delay,
            "age_since_exposure": self.age_since_exposure,
            "processing_delay": self.processing_delay,
            "late_frame": self.late_frame,
            "missing_frames": list(self.missing_frames),
            "replay_controls": [control.as_dict() for control in self.replay_controls],
            "pending_control_ids": list(self.pending_control_ids),
            "proprio_frame": self.proprio_frame,
            "availability": dict(self.availability),
            "target_actions_included": self.target_actions_included,
            "note": self.note,
        }


@dataclass(frozen=True)
class ExecutedInterval:
    """``(t0, t1]`` 区间内"真正执行了什么"的检查结果。

    ``controls`` 保持 sidecar 里的命令顺序（不重排掩盖乱序）；``excluded`` 收集落在同一
    区间但不是已执行状态的控制，它们**不是零动作**，只是不能当成已执行的重放输入。
    ``gaps`` 是已执行区间没有覆盖到的子区间，``coverage`` 是覆盖比例；控制连时间都没有
    时进 ``missing_time``，不会静默按 0 处理。
    """

    t0: float
    t1: float
    time_unit: TimeUnit
    controls: tuple[ExecutedControl, ...]
    excluded: tuple[ExecutedControl, ...]
    order_violations: tuple[str, ...]
    gaps: tuple[tuple[float, float], ...]
    overlaps: tuple[tuple[str, str], ...]
    missing_time: tuple[str, ...]
    coverage: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "t0": self.t0,
            "t1": self.t1,
            "time_unit": self.time_unit,
            "controls": [control.command_id for control in self.controls],
            "excluded": [
                {"command_id": control.command_id, "status": control.status}
                for control in self.excluded
            ],
            "order_violations": list(self.order_violations),
            "gaps": [list(gap) for gap in self.gaps],
            "overlaps": [list(pair) for pair in self.overlaps],
            "missing_time": list(self.missing_time),
            "coverage": self.coverage,
        }


def dispatch_time(control: ExecutedControl, exposure_lookup: dict[int, float]) -> float | None:
    """控制的下发时刻。

    优先级：``command_time`` -> ``start_time``。仅有帧号/相位时不能定位执行时刻。
    ``exposure_lookup`` 为旧调用签名保留，不用于编造控制时间。
    """
    if control.command_time is not None:
        return float(control.command_time)
    if control.start_time is not None:
        return float(control.start_time)
    # A frame link / phase is not an execution timestamp. Never invent one.
    return None


def interval_report(
    controls: tuple[ExecutedControl, ...] | list[ExecutedControl],
    *,
    t0: float,
    t1: float,
    time_unit: TimeUnit,
    exposure_lookup: dict[int, float] | None = None,
) -> ExecutedInterval:
    """检查 ``(t0, t1]`` 内的执行覆盖。

    未执行状态（``planned`` / ``dropped`` / ``unknown``）一律进 ``excluded``；
    它们既不算覆盖，也不会被填成零动作。
    """
    _finite_times(t0=t0, t1=t1)
    if t1 <= t0:
        raise ValueError(f"区间必须满足 t0 < t1，实际 t0={t0}, t1={t1}")
    lookup = exposure_lookup or {}
    executed: list[ExecutedControl] = []
    excluded: list[ExecutedControl] = []
    missing_time: list[str] = []
    for control in controls:
        if control.time_unit != time_unit:
            raise ValueError("control time_unit differs from interval time_unit")
        moment = dispatch_time(control, lookup)
        placeable = moment is not None or control.has_interval
        if not placeable:
            missing_time.append(control.command_id)
        intersects = False
        if control.has_interval:
            intersects = control.start_time < t1 and control.end_time > t0  # type: ignore[operator]
        if moment is not None:
            intersects = intersects or (t0 < moment <= t1)
        if not control.verified or control.value is None:
            #: 未核实、缺值或非执行状态只按窗口收集；连时间都没有的不能静默丢弃。
            if intersects or not placeable:
                excluded.append(control)
            continue
        if placeable and intersects:
            if moment is not None and moment > t1:
                excluded.append(control)
                continue
            if control.has_interval:
                start = max(float(control.start_time), t0)
                end = min(float(control.end_time), t1)
                if end <= start:
                    continue
                control = replace(control, start_time=start, end_time=end)
            executed.append(control)

    order_violations: list[str] = []
    previous: float | None = None
    for control in executed:
        moment = dispatch_time(control, lookup)
        anchor = moment if moment is not None else control.start_time
        if anchor is None:
            continue
        if previous is not None and anchor < previous:
            order_violations.append(control.command_id)
        previous = anchor

    #: 覆盖按已执行控制的 [start, end) 与窗口求交；只给时刻的控制贡献零测度。
    spans = [
        (max(float(control.start_time), t0), min(float(control.end_time), t1))
        for control in executed
        if control.has_interval
    ]
    spans = [(start, end) for start, end in spans if end > start]
    spans.sort()
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 1e-12:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = sum(end - start for start, end in merged)
    gaps: list[tuple[float, float]] = []
    cursor = t0
    for start, end in merged:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < t1:
        gaps.append((cursor, t1))
    overlaps: list[tuple[str, str]] = []
    for index, left in enumerate(executed):
        if not left.has_interval:
            continue
        for right in executed[index + 1 :]:
            if not right.has_interval:
                continue
            if max(left.start_time, right.start_time) < min(left.end_time, right.end_time):  # type: ignore[arg-type,operator]
                overlaps.append((left.command_id, right.command_id))
    return ExecutedInterval(
        t0=t0,
        t1=t1,
        time_unit=time_unit,
        controls=tuple(executed),
        excluded=tuple(excluded),
        order_violations=tuple(order_violations),
        gaps=tuple(gaps),
        overlaps=tuple(overlaps),
        missing_time=tuple(missing_time),
        coverage=covered / (t1 - t0),
    )


@dataclass
class TelemetrySidecar:
    """一条 episode 的遥测容器：逐帧时间 + 已执行控制 + 标定，附校验语义。

    容器本身**不补齐**任何字段：没有的记录就是没有。``validate`` 返回结构化的不可用
    条件清单与各字段可得比例，供 S1/S3 决定哪些条件可以进入正式训练。
    """

    episode_id: str
    suite: str = ""
    time_unit: TimeUnit = "frame"
    timing: tuple[FrameTiming, ...] = ()
    controls: tuple[ExecutedControl, ...] = ()
    calibration: tuple[CalibrationRecord, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    schema: str = "owac-telemetry-sidecar-v1"
    note: str = ""

    def timing_for(self, frame: int) -> FrameTiming | None:
        for record in self.timing:
            if record.frame == frame:
                return record
        return None

    def calibration_for(self, camera_id: str) -> CalibrationRecord | None:
        for record in self.calibration:
            if record.camera_id == camera_id:
                return record
        return None

    def check_time_domain(self) -> None:
        """Reject incomparable clocks/units before any online time selection."""
        if any(r.time_unit != self.time_unit for r in (*self.timing, *self.controls)):
            raise ValueError("sidecar time_unit differs from a record time_unit")
        clocks = {r.clock_id for r in (*self.timing, *self.controls) if r.clock_id}
        if len(clocks) > 1:
            raise ValueError(f"mixed clock domains require explicit conversion: {sorted(clocks)}")
        frames = [r.frame for r in self.timing]
        if len(set(frames)) != len(frames):
            raise ValueError("duplicate frame timing")
        for r in self.timing:
            exposure = r.exposure_end if r.exposure_end is not None else r.exposure_time()
            if exposure is not None and r.arrival_time is not None and r.arrival_time < exposure:
                raise ValueError("arrival_time precedes exposure")

    def exposure_lookup(self) -> dict[int, float]:
        self.check_time_domain()
        lookup: dict[int, float] = {}
        for record in self.timing:
            moment = record.exposure_time()
            if moment is not None:
                lookup[record.frame] = float(moment)
        return lookup

    def context_for(
        self,
        *,
        exposure_time: float,
        decision_time: float,
        n_frames: int | None = None,
        proprio_available: bool = True,
    ) -> OnlineContext:
        """构造 ``decision_time`` 时刻的在线上下文。

        可见性判据只有一条：``arrival_time <= decision_time``。没有到达时间的帧**不可见**
        （不假定帧立即到达）；曝光时刻晚于 ``exposure_time`` 的帧也不可见（不拿未来观测）。
        ``n_frames`` 用于把超过序列长度的帧排除掉。
        """
        _finite_times(exposure_time=exposure_time, decision_time=decision_time)
        if not self.timing:
            raise ValueError("sidecar 没有帧时间记录：无法判定哪些帧已经到达")
        lookup = self.exposure_lookup()
        visible: list[int] = []
        late_frames: list[int] = []
        missing: list[int] = []
        for record in self.timing:
            frame = record.frame
            if n_frames is not None and (frame < 0 or frame >= n_frames):
                continue
            exposure = lookup.get(frame)
            if exposure is None or exposure > exposure_time:
                continue
            if record.arrival_time is None:
                missing.append(frame)
                continue
            if record.arrival_time <= decision_time:
                visible.append(frame)
            else:
                missing.append(frame)
                late_frames.append(frame)
        visible.sort(key=lambda frame: (lookup[frame], frame))
        eligible = [frame for frame in visible if lookup[frame] <= exposure_time]
        latest = eligible[-1] if eligible else None
        latest_exposure = None if latest is None else float(lookup[latest])
        latest_timing = None if latest is None else self.timing_for(latest)
        latest_arrival = None if latest_timing is None else latest_timing.arrival_time

        replay: tuple[ExecutedControl, ...] = ()
        pending: tuple[str, ...] = ()
        if latest_exposure is not None and latest_exposure < decision_time:
            window = interval_report(
                self.controls,
                t0=latest_exposure,
                t1=decision_time,
                time_unit=self.time_unit,
                exposure_lookup=lookup,
            )
            replay = window.controls
            pending = tuple(control.command_id for control in window.excluded)
        arrival_delay = None if latest_timing is None else latest_timing.arrival_delay()
        processing_delay = None if latest_arrival is None else float(decision_time - latest_arrival)
        age = None if latest_exposure is None else float(decision_time - latest_exposure)
        return OnlineContext(
            decision_time=float(decision_time),
            time_unit=self.time_unit,
            requested_exposure_time=float(exposure_time),
            visible_frames=tuple(visible),
            latest_frame=latest,
            latest_exposure_time=latest_exposure,
            latest_arrival_time=None if latest_arrival is None else float(latest_arrival),
            arrival_delay=arrival_delay,
            processing_delay=processing_delay,
            age_since_exposure=age,
            late_frame=bool(late_frames),
            missing_frames=tuple(sorted(missing)),
            replay_controls=replay,
            pending_control_ids=pending,
            proprio_frame=latest if proprio_available else None,
            availability={
                "timing": True,
                "arrival": latest_arrival is not None,
                "proprio": bool(proprio_available and latest is not None),
                "replay_controls": bool(replay),
                "control_log": bool(self.controls),
                "exposure_known": bool(lookup),
            },
            target_actions_included=False,
            note=(
                "time_unit='frame'：数值是帧序号而不是秒"
                if self.time_unit == "frame"
                else f"time_unit={self.time_unit!r}"
            ),
        )

    @property
    def has_real_seconds(self) -> bool:
        return self.time_unit in ("s", "ms")

    @property
    def availability(self) -> dict[str, bool]:
        return {
            "timing": bool(self.timing),
            "exposure": any(record.has_exposure for record in self.timing),
            "arrival": any(record.arrival_time is not None for record in self.timing),
            "rolling_shutter": any(
                record.rolling_shutter_row_time is not None for record in self.timing
            ),
            "controls": bool(self.controls),
            "calibration": bool(self.calibration),
        }

    def validate(self, *, n_frames: int | None = None) -> TelemetryReport:
        """episode 级完整性校验。

        ``n_frames`` 给出该 episode 实际应有的帧数（来自测量序列）时，还会检查每帧是否
        有对应时间记录。不传就只校验 sidecar 自身的一致性。
        """
        conditions: list[UnavailableCondition] = []
        frames = sorted(record.frame for record in self.timing)
        if n_frames is not None:
            missing = sorted(set(range(n_frames)) - set(frames))
            if missing:
                conditions.append(
                    UnavailableCondition(
                        code="timing.missing_frames",
                        scope="timing",
                        detail=f"{len(missing)}/{n_frames} 帧没有时间记录：前若干帧 {missing[:8]}",
                    )
                )
        if not self.timing:
            conditions.append(
                UnavailableCondition(
                    code="timing.absent", scope="timing", detail="整条 episode 没有任何帧时间记录"
                )
            )
        if self.timing and not any(record.has_exposure for record in self.timing):
            conditions.append(
                UnavailableCondition(
                    code="timing.exposure_absent",
                    scope="timing",
                    detail="有帧记录但没有任何曝光时间（start/mid/end 全缺失）",
                )
            )
        if self.timing and not any(record.arrival_time is not None for record in self.timing):
            conditions.append(
                UnavailableCondition(
                    code="timing.arrival_absent",
                    scope="timing",
                    detail="没有任何到达时间；迟到帧与输出延迟无法判定",
                )
            )
        clocks = sorted({record.clock_id for record in self.timing if record.clock_id})
        if len(clocks) > 1:
            conditions.append(
                UnavailableCondition(
                    code="timing.clock_mixed",
                    scope="timing",
                    detail=f"同一 episode 混用多个时钟 {clocks}；跨时钟数值不可直接相减",
                )
            )
        if not self.has_real_seconds:
            conditions.append(
                UnavailableCondition(
                    code="time.frame_index_only",
                    scope="time",
                    detail=f"time_unit={self.time_unit!r}：只有帧序号，没有秒；不得换算成秒",
                )
            )
        for record in self.timing:
            if (
                record.exposure_start is not None
                and record.exposure_end is not None
                and record.arrival_time is not None
                and record.arrival_time < record.exposure_end
            ):
                conditions.append(
                    UnavailableCondition(
                        code="timing.arrival_before_exposure_end",
                        scope="timing",
                        frame=record.frame,
                        detail=(
                            f"arrival {record.arrival_time} 早于 exposure_end "
                            f"{record.exposure_end}；两者单位或时钟不一致"
                        ),
                    )
                )
            if (
                record.exposure_start is not None
                and record.exposure_mid is not None
                and record.exposure_mid < record.exposure_start
            ) or (
                record.exposure_mid is not None
                and record.exposure_end is not None
                and record.exposure_mid > record.exposure_end
            ):
                conditions.append(
                    UnavailableCondition(
                        code="timing.exposure_nonmonotonic",
                        scope="timing",
                        frame=record.frame,
                        detail="exposure_mid 不在 start 与 end 之间",
                    )
                )
        if not self.controls:
            conditions.append(
                UnavailableCondition(
                    code="control.absent",
                    scope="control",
                    detail="没有已执行控制记录；不能把它当成全零动作序列",
                )
            )
        unverified_levels = sorted(
            {
                control.evidence_level
                for control in self.controls
                if control.executed and control.evidence_level not in VERIFIED_EVIDENCE_LEVELS
            }
        )
        if unverified_levels:
            conditions.append(
                UnavailableCondition(
                    code="control.evidence_unverified",
                    scope="control",
                    detail=(
                        f"已执行控制只有证据等级 {unverified_levels}；"
                        "未达 executed_log/replay_verified，不得当作已执行动作采信"
                    ),
                )
            )
        non_executed = [control for control in self.controls if not control.executed]
        if non_executed:
            counts: dict[str, int] = {}
            for control in non_executed:
                counts[control.status] = counts.get(control.status, 0) + 1
            conditions.append(
                UnavailableCondition(
                    code="control.non_executed_present",
                    scope="control",
                    detail=f"{len(non_executed)} 条非执行状态控制 {counts}；不得当作已执行",
                )
            )
        without_time = [
            control.command_id
            for control in self.controls
            if control.command_time is None and not control.has_interval and control.frame is None
        ]
        if without_time:
            conditions.append(
                UnavailableCondition(
                    code="control.missing_time",
                    scope="control",
                    detail=f"{len(without_time)} 条控制没有任何可定位时间：{without_time[:8]}",
                )
            )
        previous: float | None = None
        for control in self.controls:
            if control.executed and control.command_time is not None:
                if previous is not None and control.command_time < previous:
                    conditions.append(
                        UnavailableCondition(
                            code="control.out_of_order",
                            scope="control",
                            detail=f"{control.command_id} 的命令时间早于前一条；顺序不可信",
                        )
                    )
                    break
                previous = control.command_time
        for control in self.controls:
            if control.executed and control.has_interval:
                duration = float(control.end_time - control.start_time)  # type: ignore[operator]
                if duration <= 0.0:
                    conditions.append(
                        UnavailableCondition(
                            code="control.zero_duration",
                            scope="control",
                            detail=f"{control.command_id} 的执行区间长度为 0",
                        )
                    )
        if self.controls and all(control.frame is None for control in self.controls):
            conditions.append(
                UnavailableCondition(
                    code="control.frame_unlinked",
                    scope="control",
                    detail="控制记录没有关联帧号，无法与测量逐帧对齐",
                )
            )
        if not self.calibration:
            conditions.append(
                UnavailableCondition(
                    code="calibration.absent",
                    scope="calibration",
                    detail="没有标定记录；内参/外参/时间偏移均不可用",
                )
            )
        else:
            for record in self.calibration:
                if not record.available:
                    missing = sorted(key for key, value in record.availability.items() if not value)
                    conditions.append(
                        UnavailableCondition(
                            code="calibration.partial",
                            scope="calibration",
                            detail=f"{record.camera_id} 不能用于投影，缺失 {missing}",
                        )
                    )
                elif record.time_offset is None:
                    conditions.append(
                        UnavailableCondition(
                            code="calibration.no_time_offset",
                            scope="calibration",
                            detail=f"{record.camera_id} 没有相机-机器人时间偏移；时间未标定",
                        )
                    )
        availability = self._availability_fractions(n_frames)
        return TelemetryReport(
            episode_id=self.episode_id,
            n_frames=n_frames,
            conditions=tuple(conditions),
            availability=availability,
        )

    def _availability_fractions(self, n_frames: int | None) -> dict[str, float]:
        total = n_frames if n_frames is not None else max(len(self.timing), 1)
        out: dict[str, float] = {}
        for key in ("exposure_start", "exposure_mid", "exposure_end", "arrival_time"):
            available = sum(1 for record in self.timing if record.availability[key])
            out[f"timing.{key}"] = available / total
        out["timing.frames"] = len(self.timing) / total
        controls = len(self.controls)
        out["control.executed"] = (
            0.0 if controls == 0 else sum(1 for c in self.controls if c.executed) / controls
        )
        out["control.value"] = (
            0.0
            if controls == 0
            else sum(1 for c in self.controls if c.value is not None) / controls
        )
        out["control.verified"] = (
            0.0 if controls == 0 else sum(1 for c in self.controls if c.verified) / controls
        )
        out["control.frame"] = (
            0.0
            if controls == 0
            else sum(1 for c in self.controls if c.frame is not None) / controls
        )
        cameras = max(len(self.calibration), 1)
        out["calibration.available"] = (
            sum(1 for record in self.calibration if record.available) / cameras
        )
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "suite": self.suite,
            "time_unit": self.time_unit,
            "timing": [record.as_dict() for record in self.timing],
            "controls": [control.as_dict() for control in self.controls],
            "calibration": [record.as_dict() for record in self.calibration],
            "provenance": json.loads(json.dumps(self.provenance, default=str)),
            "note": self.note,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TelemetrySidecar:
        return cls(
            episode_id=str(payload["episode_id"]),
            suite=str(payload.get("suite", "")),
            time_unit=payload.get("time_unit", "frame"),
            timing=tuple(FrameTiming.from_dict(item) for item in payload.get("timing", ())),
            controls=tuple(ExecutedControl.from_dict(item) for item in payload.get("controls", ())),
            calibration=tuple(
                CalibrationRecord.from_dict(item) for item in payload.get("calibration", ())
            ),
            provenance=dict(payload.get("provenance", {})),
            schema=str(payload.get("schema", "owac-telemetry-sidecar-v1")),
            note=str(payload.get("note", "")),
        )

    @classmethod
    def from_json(cls, text: str) -> TelemetrySidecar:
        return cls.from_dict(json.loads(text))
