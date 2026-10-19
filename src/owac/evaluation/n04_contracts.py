"""N04 S0 合同审计套件（工作流 D：合同验证与预注册门槛）。

用法::

    python -m owac.evaluation.n04_contracts \\
        --config configs/n04/full/s0_data.json --output <run_dir>/contracts.json

职责：

* 把 S0 提示词的「必做验证」逐条实现为可复跑的检查，输出 ``passed`` / ``failed`` /
  ``skipped`` 与数值证据。依赖其他工作流产物的检查在依赖缺失时记 ``skipped`` 并保留
  重跑入口，不伪造输入。
* 校验 ``configs/n04/full/protocol.json`` 的冻结门槛与 ``s0_data.json`` 的 nominal
  camera 换算一致（C0），避免门槛被静默改写。

边界：

* 本模块只做审计与参照计算，**不修改**被测实现。失败项如实报告，交由对应工作流修复。
* C2.1–C2.4 对 ``frameworks/n04/geometry.py`` 基元与 CFA 相位律做独立 FP64 对照；
  C2.5–C2.6 对 ws-geometry ``measurement_geometry.py`` 的 crop/resize 内参、原图→特征
  单位链与有限差分做对照。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCHEMA = "n04-s0-contracts-v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = "configs/n04/full/protocol.json"

#: S0 提示词要求逐条覆盖的检查（顺序即报告顺序）。
REQUIRED_CHECKS = (
    ("C1", "D0 逐样本图像—标签对应（独立投影重算）"),
    ("C2", "已知 K / 90 度旋转 / 复合逆 / 深度符号 / 像素中心 / CFA 四相位"),
    ("C3", "未观测值 NaN / 极大值：更新与梯度不受影响"),
    ("C4", "未来动作与当前图像置乱；已执行控制改变上下文选择"),
    ("C5", "迟到帧按曝光时刻回补并重放已执行控制；未到达本体不泄漏"),
    ("C6", "sensor residual 数值精度；曝光增益改变均值/方差；饱和/遮挡独立"),
    ("C7", "manifest 两次生成一致、hash 稳定、拆分源 ID 交集为零"),
)

STATUSES = ("passed", "failed", "skipped", "partial")


class DependencyMissing(RuntimeError):
    """检查所需的兄弟工作流产物/运行时不可用：整组记 skipped。"""


# --------------------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ("git", *args), cwd=REPO_ROOT, capture_output=True, text=True, timeout=20
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "head": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "porcelain_lines": None if status is None else len(status.splitlines()),
    }


def _probe_import(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - 探测需要吞掉任意导入错误
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"import ok: {getattr(module, '__file__', '<builtin>')}"


def _dependency_sources(names: tuple[str, ...]) -> dict[str, Any]:
    """记录依赖文件的 sha256，便于把审计结论钉在具体兄弟工作流版本上。"""
    out: dict[str, Any] = {}
    for name in names:
        ok, detail = _probe_import(name)
        digest = None
        if ok:
            file = getattr(importlib.import_module(name), "__file__", None)
            if file and Path(file).is_file():
                digest = _sha256_file(Path(file))
        out[name] = {"importable": ok, "sha256": digest, "detail": detail}
    return out


def _probe_module_attrs(name: str, attrs: tuple[str, ...]) -> dict[str, Any]:
    ok, detail = _probe_import(name)
    found: list[str] = []
    if ok:
        module = importlib.import_module(name)
        found = [attr for attr in attrs if hasattr(module, attr)]
    return {"module": name, "importable": ok, "detail": detail, "found_attrs": found}


def _sub(
    check_id: str,
    title: str,
    status: str,
    reason: str = "",
    **evidence: Any,
) -> dict[str, Any]:
    assert status in STATUSES, status
    return {
        "id": check_id,
        "title": title,
        "status": status,
        "reason": reason,
        "evidence": evidence,
    }


def _group(
    check_id: str,
    title: str,
    subchecks: list[dict[str, Any]],
    *,
    required: bool = True,
    note: str = "",
) -> dict[str, Any]:
    """组合状态规则：任一 failed → failed；passed 与 skipped 并存 → partial；
    全 skipped → skipped；否则 passed。partial 表示代码语义已测、真实数据缺失。"""
    statuses = {sub["status"] for sub in subchecks}
    if "failed" in statuses:
        status = "failed"
    elif "skipped" in statuses and "passed" in statuses:
        status = "partial"
    elif "skipped" in statuses:
        status = "skipped"
    else:
        status = "passed"
    return {
        "id": check_id,
        "title": title,
        "status": status,
        "required": required,
        "note": note,
        "subchecks": subchecks,
    }


def _torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise DependencyMissing(
            f"torch 不可用（{type(exc).__name__}: {exc}）；请用 --extra encoder 环境运行"
        ) from exc
    return torch


def _numpy():
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise DependencyMissing(f"numpy 不可用（{type(exc).__name__}: {exc}）") from exc
    return np


def _finite_all(torch, tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def _tiny_encoder(
    torch,
    *,
    seed: int = 0,
    predictor: str = "identity",
    innovation_reference: str = "task_state",
    width: int = 8,
):
    from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig
    from owac.frameworks.n04.model import N04Encoder

    torch.manual_seed(seed)
    config = N04Config(
        measurement_size=(16, 16),
        camera_ids=("head",),
        geometry=N04GeometryConfig(feature_size=(16, 16), pixel_stride=1.0),
        model=N04ModelConfig(
            width=width,
            hidden=8,
            blocks=1,
            scales=1,
            scale_strides=(1,),
            readout_tokens=4,
            readout_dim=16,
            predictor=predictor,
            innovation_reference=innovation_reference,
        ),
    )
    return N04Encoder(config)


def _context(
    torch,
    *,
    actions: list[list[float]],
    proprio: list[list[float]] | None = None,
    proprio_available: bool = False,
    action_available: bool = True,
):
    from owac.frameworks.n04.ego import EgoMotionContext

    action_tensor = torch.as_tensor(actions, dtype=torch.float32).reshape(-1, 3)
    batch = action_tensor.shape[0]
    if proprio is None:
        proprio_tensor = torch.zeros(batch, 4, dtype=torch.float32)
    else:
        proprio_tensor = torch.as_tensor(proprio, dtype=torch.float32).reshape(batch, 4)
    return EgoMotionContext(
        proprio_delta=proprio_tensor,
        proprio_delta_available=proprio_available,
        executed_action=action_tensor,
        executed_action_available=action_available,
        dt=torch.ones(batch),
        dt_available=False,
        expected_width=8,
        camera_id="head",
    )


def _perturb_head(torch, encoder, seed: int, scale: float = 0.05) -> bool:
    """审计专用结构扰动：未训练 head 恒为零初始化，无法暴露上下文路由。"""
    head = getattr(encoder.propagator, "head", None)
    if head is None:
        return False
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        head.weight.normal_(0.0, scale, generator=generator)
        head.bias.normal_(0.0, scale, generator=generator)
    return True


def _step_with_grads(torch, encoder, values, context, mask):
    state = encoder.init_state(1)["head"]
    out = encoder.step(values, context, state, camera_id="head", measurement_valid=mask)
    encoder.zero_grad(set_to_none=True)
    (out.state.stable.sum() + out.state.change.sum() + out.state.uncertainty.sum()).backward()
    grads = {
        name: (None if param.grad is None else param.grad.detach().clone())
        for name, param in encoder.named_parameters()
    }
    return {
        "stable": out.state.stable.detach().clone(),
        "change": out.state.change.detach().clone(),
        "uncertainty": out.state.uncertainty.detach().clone(),
        "representation": out.representation.detach().clone(),
        "grads": grads,
    }


def _grad_diff(torch, lhs: dict, rhs: dict) -> dict[str, Any]:
    nonfinite = 0
    compared = 0
    max_diff = 0.0
    for name, grad in rhs.items():
        if grad is None:
            continue
        if not _finite_all(torch, grad):
            nonfinite += 1
            continue
        other = lhs.get(name)
        if other is None:
            continue
        compared += 1
        max_diff = max(max_diff, float((grad - other).abs().max()))
    total = sum(1 for grad in rhs.values() if grad is not None)
    return {
        "param_grads_total": total,
        "param_grads_compared": compared,
        "param_grads_nonfinite": nonfinite,
        "max_grad_abs_diff_vs_clean": max_diff,
    }


# --------------------------------------------------------------------------------------
# C0 预注册门槛一致性（附加，不计入必做验证）
# --------------------------------------------------------------------------------------


def _check_c0(ctx: RunContext) -> dict[str, Any]:
    subs: list[dict[str, Any]] = []
    protocol = ctx.protocol
    config = ctx.config
    if protocol is None:
        return _group(
            "C0",
            "protocol.json 冻结与换算一致性（附加）",
            [_sub("C0.0", "读取 protocol.json", "skipped", f"未找到 {ctx.protocol_path}")],
            required=False,
        )
    subs.append(
        _sub(
            "C0.1",
            "protocol 已冻结",
            "passed" if protocol.get("frozen") is True else "failed",
            "" if protocol.get("frozen") is True else "frozen 字段不是 true，门槛可能被静默修改",
            frozen=protocol.get("frozen"),
            frozen_at=protocol.get("frozen_at"),
        )
    )
    recorded = str(protocol.get("source_config", {}).get("sha256", ""))
    actual = _sha256_file(ctx.config_path)
    subs.append(
        _sub(
            "C0.2",
            "source_config.sha256 与 s0_data.json 一致",
            "passed" if recorded == actual else "failed",
            "" if recorded == actual else "protocol 记录的配置 hash 与当前配置不符，需重新冻结",
            recorded=recorded,
            actual=actual,
        )
    )
    reference_2d = config.get("protocol", {}).get("reference_2d", {})
    inherited = protocol.get("inherited_2d", {})
    rel_ok = math.isclose(
        float(inherited.get("relative_improvement_min", -1)),
        float(reference_2d.get("relative_improvement_min", -2)),
        rel_tol=1e-12,
    )
    abs_ok = math.isclose(
        float(inherited.get("absolute_improvement_px_min", -1)),
        float(reference_2d.get("absolute_improvement_px_min", -2)),
        rel_tol=1e-12,
    )
    subs.append(
        _sub(
            "C0.3",
            "二维门槛继承与配置镜像一致",
            "passed" if (rel_ok and abs_ok) else "failed",
            "" if (rel_ok and abs_ok) else "protocol.json 与 s0_data.json 的二维门槛不一致",
            protocol_relative=inherited.get("relative_improvement_min"),
            config_relative=reference_2d.get("relative_improvement_min"),
            protocol_absolute_px=inherited.get("absolute_improvement_px_min"),
            config_absolute_px=reference_2d.get("absolute_improvement_px_min"),
        )
    )
    filled = _three_dimensional_derivation(config)
    frozen = protocol.get("three_dimensional", {})
    gate = frozen.get("absolute_gate", {})
    original = gate.get("original_pixel_space", {})
    feature = gate.get("feature_space_equivalents", {})
    checks = {
        "original_pixel_space.min_improvement_m": (
            float(original.get("min_improvement_m", -1)),
            filled["resolution_original_m"],
        ),
        "original_pixel_space.inherited_floor_m": (
            float(original.get("inherited_floor_m", -1)),
            filled["inherited_original_m"],
        ),
        "feature_space_equivalents.feature_scale_m_per_feature_px": (
            float(feature.get("feature_scale_m_per_feature_px", -1)),
            filled["feature_scale_m"],
        ),
        "feature_space_equivalents.inherited_floor_feature_px": (
            float(feature.get("inherited_floor_feature_px", -1)),
            filled["inherited_feature_px"],
        ),
        "feature_space_equivalents.measurement_resolution_feature_px_equivalent": (
            float(feature.get("measurement_resolution_feature_px_equivalent", -1)),
            filled["resolution_feature_px"],
        ),
        "feature_space_equivalents.gate_m": (
            float(feature.get("gate_m", -1)),
            filled["resolution_original_m"],
        ),
    }
    mismatches: dict[str, Any] = {}
    for name, (value, expected) in checks.items():
        if not math.isclose(value, expected, rel_tol=1e-12):
            mismatches[name] = {"frozen": value, "recomputed": expected}
    diagnostic_ok = (
        frozen.get("calibration_available") is False
        and frozen.get("status") == "diagnostic_reference_only"
        and protocol.get("source_theory", {}).get("path") == "research/theory/n04_v1.md"
    )
    subs.append(
        _sub(
            "C0.4",
            "三维米制门槛与 nominal_camera / theory §7 换算一致",
            "passed" if (not mismatches and diagnostic_ok) else "failed",
            "" if (not mismatches and diagnostic_ok) else "冻结门槛与重算值不符或缺少诊断参照标注",
            recomputed_m=filled,
            mismatches=mismatches,
            diagnostic_reference_labeled=diagnostic_ok,
            theory_source=protocol.get("source_theory", {}).get("path"),
        )
    )
    errors = []
    config_error = config.get("protocol", {}).get("error_separation", {})
    h1 = protocol.get("h1_error_separation", {})
    for key, expected_key in (
        ("dev_false_positive_budget", "dev_false_positive_budget"),
        ("negative_false_positive_relative_reduction_min", "false_positive_relative_reduction_min"),
        ("event_recall_drop_max_percentage_points", "event_recall_drop_max_pp"),
    ):
        left, right = float(h1.get(key, math.nan)), float(config_error.get(expected_key, math.nan))
        if not math.isclose(left, right, rel_tol=1e-12):
            errors.append(f"H1 {key}: frozen={left} config={right}")
    config_dyn = config.get("protocol", {}).get("dynamic_revision", {})
    h2 = protocol.get("h2_dynamic_revision", {})
    for key, expected_key in (
        (
            "stop_reverse_integration_error_reduction_min",
            "stop_reverse_integration_error_reduction_min",
        ),
        ("missing_phase_error_degradation_max", "missing_phase_error_degradation_max"),
    ):
        left, right = float(h2.get(key, math.nan)), float(config_dyn.get(expected_key, math.nan))
        if not math.isclose(left, right, rel_tol=1e-12):
            errors.append(f"H2 {key}: frozen={left} config={right}")
    subs.append(
        _sub(
            "C0.5",
            "H1/H2 门槛与配置一致",
            "passed" if not errors else "failed",
            "; ".join(errors),
            mismatches=errors,
        )
    )
    return _group("C0", "protocol.json 冻结与换算一致性（附加）", subs, required=False)


def _three_dimensional_derivation(config: dict[str, Any]) -> dict[str, float]:
    camera = config["protocol"]["nominal_camera"]
    focal = float(camera["focal_px"])
    depth = float(camera["depth_m"])
    stride = float(camera["feature_stride"])
    return {
        "focal_px": focal,
        "depth_m": depth,
        "feature_stride": stride,
        "inherited_original_m": 0.05 * depth / focal,
        "resolution_original_m": 1.0 * depth / focal,
        "inherited_feature_px": 0.05 / stride,
        "resolution_feature_px": 1.0 / stride,
        "feature_scale_m": stride * depth / focal,
    }


# --------------------------------------------------------------------------------------
# C1 D0 逐样本图像—标签对应
# --------------------------------------------------------------------------------------


def _reference_project_points_fp64(points_world, K, T_world_from_cam):
    """独立 FP64 针孔重投影（numpy 向量化版本）。"""
    np = _numpy()
    points = np.asarray(points_world, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    transform = np.asarray(T_world_from_cam, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points_cam = (points - translation) @ rotation
    z = points_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsics[0, 0] * points_cam[:, 0] / z + intrinsics[0, 2]
        v = intrinsics[1, 1] * points_cam[:, 1] / z + intrinsics[1, 2]
    return np.stack((u, v), axis=-1), z


def _reference_project_batch_fp64(points_world, K, transforms):
    """逐样本批版参照投影：``points_world [B,3]``、``transforms [B,4,4]``。"""
    np = _numpy()
    points = np.asarray(points_world, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    uv_rows = []
    depth_rows = []
    for index in range(points.shape[0]):
        uv, depth = _reference_project_points_fp64(points[index : index + 1], K, transforms[index])
        uv_rows.append(uv[0])
        depth_rows.append(float(depth[0]))
    return np.asarray(uv_rows, dtype=np.float64), np.asarray(depth_rows, dtype=np.float64)


def _loop_project_points_fp64(points_world, K, T_world_from_cam):
    """逐点循环的独立实现：用于交叉验证上面的向量化参照核。"""
    np = _numpy()
    intrinsics = np.asarray(K, dtype=np.float64)
    transform = np.asarray(T_world_from_cam, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    out = []
    depths = []
    for point in np.asarray(points_world, dtype=np.float64):
        cam = rotation.T @ (point - translation)
        depths.append(cam[2])
        out.append(
            (
                intrinsics[0, 0] * cam[0] / cam[2] + intrinsics[0, 2],
                intrinsics[1, 1] * cam[1] / cam[2] + intrinsics[1, 2],
            )
        )
    return np.asarray(out, dtype=np.float64), np.asarray(depths, dtype=np.float64)


def _check_c1(ctx: RunContext) -> dict[str, Any]:
    subs: list[dict[str, Any]] = []
    try:
        subs.append(_c1_reference_selftest(ctx))
    except DependencyMissing as exc:
        subs.append(_sub("C1.1", "独立 FP64 重投影参照核自检", "skipped", str(exc)))
    ok, detail = _probe_import("owac.raw.n04_mechanism_data")
    if not ok:
        subs.append(
            _sub(
                "C1.2",
                "D0 随机逐样本渲染标签重算（B=1/2/4、多物体、多相机、置换不串位）",
                "skipped",
                f"D0 生产者尚未就绪：owac.raw.n04_mechanism_data {detail}",
                rerun="依赖就绪后重新运行本 CLI；期望接口见 "
                "docs/prompts/n04_v1_0/S0_data_contracts.md",
                dependency={"module": "owac.raw.n04_mechanism_data", "importable": False},
            )
        )
    else:
        subs.append(_c1_d0_correspondence(ctx))
    return _group(
        "C1",
        "D0 逐样本图像—标签对应（独立投影重算）",
        subs,
        note="参照核已验证；D0 数据交互在生产者就绪后自动启用"
        "（接口不匹配记 skipped，不伪造输入）。",
    )


def _c1_reference_selftest(ctx: RunContext) -> dict[str, Any]:
    np = _numpy()
    rng = np.random.default_rng(ctx.seed)
    K = np.array([[320.0, 0.0, 127.5], [0.0, 320.0, 127.5], [0.0, 0.0, 1.0]])
    max_diff = 0.0
    max_perm_diff = 0.0
    shapes = []
    for objects in (1, 2, 4):
        points = rng.uniform(-0.4, 0.4, size=(objects, 3))
        points[:, 2] = rng.uniform(0.3, 1.5, size=objects) + 0.4
        for angle in (0.0, 0.7):
            transform = np.eye(4)
            transform[:3, :3] = np.array(
                [
                    [math.cos(angle), 0.0, math.sin(angle)],
                    [0.0, 1.0, 0.0],
                    [-math.sin(angle), 0.0, math.cos(angle)],
                ]
            )
            transform[:3, 3] = np.array([0.05, -0.02, 0.1])
            uv_vec, z_vec = _reference_project_points_fp64(points, K, transform)
            uv_loop, z_loop = _loop_project_points_fp64(points, K, transform)
            max_diff = max(max_diff, float(np.abs(uv_vec - uv_loop).max()))
            max_diff = max(max_diff, float(np.abs(z_vec - z_loop).max()))
            order = rng.permutation(objects)
            uv_perm, _ = _reference_project_points_fp64(points[order], K, transform)
            max_perm_diff = max(max_perm_diff, float(np.abs(uv_perm - uv_vec[order]).max()))
            shapes.append([objects, 2])
    status = "passed" if max_diff <= 1e-12 and max_perm_diff <= 1e-12 else "failed"
    return _sub(
        "C1.1",
        "独立 FP64 重投影参照核自检",
        status,
        "" if status == "passed" else "向量化参照核与逐点实现不一致",
        object_counts=[1, 2, 4],
        cameras=2,
        batch_shapes=shapes,
        max_abs_diff_vec_vs_loop=max_diff,
        max_abs_diff_permutation=max_perm_diff,
        tolerance=1e-12,
        scope="参照核自身正确性；D0 样本级对应仍需生产者数据",
    )


def _c1_d0_correspondence(ctx: RunContext) -> dict[str, Any]:
    """D0 生产者就绪后的样本级重算：独立 FP64 投影 + batch 置换隔离。"""
    from dataclasses import replace as dataclass_replace

    import numpy as np
    import torch

    from owac.raw import n04_mechanism_data as d0

    title = "D0 随机逐样本渲染标签重算（B=1/2/4、多物体、多相机、置换不串位）"
    config = d0.D0Config.from_config(ctx.config)
    producer_error: str | None = None
    producer_failures: list[Any] = []
    try:
        audit = d0.audit_d0(config, batch_sizes=(1, 2, 4))
        producer_failures = list(audit.get("failures", []))
    except Exception as exc:  # noqa: BLE001 - 生产者审计异常本身就是要退回的证据
        producer_error = f"{type(exc).__name__}: {exc}"

    per_batch: dict[str, Any] = {}
    max_error = 0.0
    for batch in (1, 2, 4):
        worlds, _ = d0.build_factor_worlds(
            d0.resolve_plan("static"), config, batch=batch, frames=1, seed=config.seed + batch
        )
        world = worlds[0]
        render = d0.render_world(world)
        labels = d0.frame_labels(world, render)
        error = 0.0
        for view, camera in enumerate(world.cameras):
            intrinsics = np.array(
                [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]]
            )
            transform = world.T_world_from_cam[camera.name].detach().cpu().numpy()
            for name, entry in labels["bodies"].items():
                track = next(item for item in world.bodies if item.body.name == name)
                points = track.T_world_from_body[:, :3, 3].detach().cpu().numpy()
                recomputed, depth = _reference_project_batch_fp64(points, intrinsics, transform)
                declared = entry["centre_px"][:, view].detach().cpu().numpy()
                finite = np.isfinite(declared)
                if finite.any():
                    error = max(error, float(np.abs(recomputed[finite] - declared[finite]).max()))
                present = track.present.detach().cpu().numpy().astype(bool)
                if bool((depth[present] <= 0).any()):
                    producer_error = producer_error or "独立重算出现非正深度"
        max_error = max(max_error, error)
        per_batch[f"B{batch}"] = {
            "bodies": len(labels["bodies"]),
            "cameras": len(world.cameras),
            "max_centre_error_px": error,
        }

    worlds, _ = d0.build_factor_worlds(
        d0.resolve_plan("static"), config, batch=4, frames=1, seed=config.seed + 41
    )
    world = worlds[0]
    render = d0.render_world(world)
    permutation = torch.tensor([3, 1, 0, 2])
    inverse = torch.argsort(permutation)
    permuted = dataclass_replace(
        world,
        T_world_from_cam={
            name: value[permutation] for name, value in world.T_world_from_cam.items()
        },
        bodies=tuple(
            dataclass_replace(
                track,
                T_world_from_body=track.T_world_from_body[permutation],
                present=track.present[permutation],
            )
            for track in world.bodies
        ),
    )
    render_permuted = d0.render_world(permuted)
    body_index_mismatch = int(
        (render_permuted.body_index[inverse] != render.body_index).sum().item()
    )
    labels = d0.frame_labels(world, render)
    labels_permuted = d0.frame_labels(permuted, render_permuted)
    label_perm_error = 0.0
    for name, entry in labels["bodies"].items():
        other = labels_permuted["bodies"][name]
        label_perm_error = max(
            label_perm_error,
            float((entry["centre_px"] - other["centre_px"][inverse]).abs().nan_to_num().max()),
            float((entry["visible_pixels"] - other["visible_pixels"][inverse]).abs().max()),
        )

    multi_ok = all(item["bodies"] >= 2 and item["cameras"] >= 2 for item in per_batch.values())
    ok = (
        producer_error is None
        and not producer_failures
        and max_error <= 1e-6
        and body_index_mismatch == 0
        and label_perm_error <= 1e-9
        and multi_ok
    )
    reason = ""
    if producer_error is not None:
        reason = (
            "D0 生产者的自带审计 audit_d0 抛异常，D0 生成路径被阻断"
            "（generate_d0 先调用它）：" + producer_error
        )
    elif producer_failures:
        reason = f"audit_d0 报告 failures：{producer_failures[:3]}"
    elif not ok:
        reason = "独立重投影误差超限、batch 置换不隔离，或多物体/多相机覆盖不足"
    return _sub(
        "C1.2",
        title,
        "passed" if ok else "failed",
        reason,
        producer_audit_error=producer_error,
        producer_audit_failures=producer_failures[:5],
        per_batch=per_batch,
        independent_reprojection_max_err_px=max_error,
        permutation_body_index_mismatch_pixels=body_index_mismatch,
        permutation_label_max_err=label_perm_error,
        tolerance_px=1e-6,
        source="owac.raw.n04_mechanism_data"
        "（audit_d0 / build_factor_worlds / render_world / frame_labels）",
        note="独立重投影使用审计侧 numpy FP64 参照核；"
        "置换检查把同一 world 的 batch 维重排后重渲染。",
    )


# --------------------------------------------------------------------------------------
# C2 几何参照
# --------------------------------------------------------------------------------------


def _c2_hand_cases() -> list[dict[str, Any]]:
    """K 已知的 90 度旋转手算用例（f=10 px, 图像 256x256, anchor 深度 0.6 m）。"""
    return [
        {
            "name": "yaw+90",
            "axis": "y",
            "angle_deg": 90.0,
            "pixel": (207.5, 127.5),
            "expect": (126.25, 127.5),
        },
        {
            "name": "roll+90",
            "axis": "z",
            "angle_deg": 90.0,
            "pixel": (207.5, 127.5),
            "expect": (127.5, 47.5),
        },
        {
            "name": "pitch-90",
            "axis": "x",
            "angle_deg": -90.0,
            "pixel": (127.5, 207.5),
            "expect": (127.5, 126.25),
        },
    ]


def _c2_torch() -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    from owac.frameworks.n04.geometry import (
        GridSpec,
        compose_transforms,
        forward_project_points,
        invert_transform,
        project_points,
    )

    return torch, {
        "GridSpec": GridSpec,
        "compose_transforms": compose_transforms,
        "forward_project_points": forward_project_points,
        "invert_transform": invert_transform,
        "project_points": project_points,
    }


def _check_c2(ctx: RunContext) -> dict[str, Any]:
    subs: list[dict[str, Any]] = []
    try:
        torch, geometry = _c2_torch()
    except DependencyMissing as exc:
        return _group(
            "C2",
            "已知 K / 90 度旋转 / 复合逆 / 深度符号 / 像素中心 / CFA 四相位",
            [_sub("C2.0", "几何运行环境", "skipped", str(exc))],
        )
    for name, fn in (
        ("C2.1", _c2_pixel_center),
        ("C2.2", _c2_rotations),
        ("C2.3", _c2_depth_sign),
    ):
        try:
            subs.append(fn(torch, geometry))
        except DependencyMissing as exc:
            subs.append(_sub(name, "几何基元检查", "skipped", str(exc)))
    try:
        subs.append(_c2_cfa(torch))
    except DependencyMissing as exc:
        subs.append(_sub("C2.4", "CFA 四相位坐标律", "skipped", str(exc)))
    ok_geometry, geometry_detail = _probe_import("owac.frameworks.n04.measurement_geometry")
    if ok_geometry:
        subs.extend(_c2_geometry_module(torch))
    else:
        subs.append(
            _sub(
                "C2.5",
                "crop/resize 内参更新与像素仿射链一致（FP64）",
                "skipped",
                f"measurement_geometry 不可导入：{geometry_detail}",
                rerun="ws-geometry 交付后重跑同一 CLI",
            )
        )
        subs.append(
            _sub(
                "C2.6",
                "原图→特征单位链、有限差分 Jacobian、负深度支持与 CFA 坐标",
                "skipped",
                f"measurement_geometry 不可导入：{geometry_detail}",
                rerun="ws-geometry 交付后重跑同一 CLI",
            )
        )
    return _group(
        "C2",
        "已知 K / 90 度旋转 / 复合逆 / 深度符号 / 像素中心 / CFA 四相位",
        subs,
        note="C2.1–C2.4 针对 geometry.py 基元与 CFA 相位律；C2.5–C2.6 针对 ws-geometry 的 "
        "measurement_geometry（crop/resize 内参、原图→特征单位链、有限差分）。",
    )


def _c2_geometry_module(torch) -> list[dict[str, Any]]:
    """ws-geometry measurement_geometry 的独立 FP64 对照。"""
    from owac.frameworks.n04 import measurement_geometry as mg
    from owac.frameworks.n04.synthetic import cfa_plane_offsets as synthetic_offsets

    dtype = torch.float64
    K0 = mg.CameraIntrinsics(
        K=torch.tensor([[320.0, 0.0, 127.5], [0.0, 320.0, 127.5], [0.0, 0.0, 1.0]], dtype=dtype),
        unit="original_px",
    )
    camera = mg.se3_identity("cam", dtype=dtype)
    points = torch.tensor(
        [[0.10, 0.05, 0.60], [-0.15, 0.10, 1.20], [0.20, -0.08, 0.80]], dtype=dtype
    )
    base_pixels = mg.project(K0, camera, points)

    resize = mg.resize_pixel_affine((256, 256), (128, 128), dtype=dtype)
    K_resized = K0.resized((256, 256), (128, 128))
    resize_err = float(
        (mg.apply_pixel_affine(resize, base_pixels).xy - mg.project(K_resized, camera, points).xy)
        .abs()
        .max()
    )
    # offset_xy = (x, y)：x 减 10（宽 246），y 减 20（高 236）
    crop = mg.crop_pixel_affine((10.0, 20.0), dtype=dtype)
    chain = mg.compose_pixel_affine(resize, crop)
    cropped = mg.CameraIntrinsics(K=K0.K.clone(), unit="original_px")
    cropped.K[..., 0, 2] = cropped.K[..., 0, 2] - 10.0
    cropped.K[..., 1, 2] = cropped.K[..., 1, 2] - 20.0
    K_chain = cropped.resized((236, 246), (118, 123))
    chain_pixels = mg.apply_pixel_affine(chain, base_pixels)
    chain_err = float((chain_pixels.xy - mg.project(K_chain, camera, points).xy).abs().max())
    inverse_err = float(
        (mg.apply_pixel_affine(mg.invert_pixel_affine(chain), chain_pixels).xy - base_pixels.xy)
        .abs()
        .max()
    )
    crop_ok = resize_err <= 1e-9 and chain_err <= 1e-9 and inverse_err <= 1e-9
    c25 = _sub(
        "C2.5",
        "crop/resize 内参更新与像素仿射链一致（FP64）",
        "passed" if crop_ok else "failed",
        "" if crop_ok else "resize/crop 后的内参与像素仿射作用不等价",
        resize_only_max_err_px=resize_err,
        crop_then_resize_max_err_px=chain_err,
        chain_inverse_max_err_px=inverse_err,
        tolerance_px=1e-9,
        chain="crop(offset=(x=10,y=20)) -> resize(236x246 -> 118x123)，中心对齐公式",
    )

    stride_affine = mg.stride_pixel_affine(32.0, center_aligned=True, dtype=dtype)
    K_feature = K0.on_feature_grid(32.0, center_aligned=True)
    stride_err = float(
        (
            mg.apply_pixel_affine(stride_affine, base_pixels).xy
            - mg.project(K_feature, camera, points).xy
        )
        .abs()
        .max()
    )
    expected_cx = (127.5 + 0.5) / 32.0 - 0.5
    intrinsics_ok = (
        abs(float(K_feature.K[..., 0, 0]) - 10.0) <= 1e-12
        and abs(float(K_feature.K[..., 0, 2]) - expected_cx) <= 1e-12
    )

    def fn(x):
        return torch.stack((x[..., 0] * x[..., 0] + 3.0 * x[..., 1], x[..., 0] * x[..., 1]), dim=-1)

    jacobian = mg.finite_difference_jacobian(fn, torch.tensor([0.3, -0.7], dtype=dtype))
    analytic = torch.tensor([[0.6, 3.0], [-0.7, 0.3]], dtype=dtype)
    fd_err = float((jacobian - analytic).abs().max())
    behind = mg.project_with_support(
        K0, camera, torch.tensor([[0.1, 0.05, -0.5]], dtype=dtype), grid_size_hw=(256, 256)
    )
    behind_ok = bool(behind.in_front.reshape(-1)[0] == 0) and float(behind.support.max()) == 0.0
    cfa_ok = all(
        tuple(mg.cfa_plane_offsets(pattern)) == tuple(synthetic_offsets(pattern))
        for pattern in ("RGGB", "BGGR", "GRBG", "GBRG")
    )
    cfa_coordinate = mg.cfa_plane_coordinates("GRBG", "R", (3, 5)).tolist()
    coordinate_ok = cfa_coordinate == [6, 11]
    unit_ok = (
        stride_err <= 1e-9
        and intrinsics_ok
        and fd_err <= 1e-6
        and behind_ok
        and cfa_ok
        and coordinate_ok
    )
    c26 = _sub(
        "C2.6",
        "原图→特征单位链、有限差分 Jacobian、负深度支持与 CFA 坐标",
        "passed" if unit_ok else "failed",
        "" if unit_ok else "特征网格单位链/有限差分/负深度/CFA 坐标之一不符",
        stride_vs_feature_intrinsics_max_err_px=stride_err,
        feature_intrinsics=[float(K_feature.K[..., 0, 0]), float(K_feature.K[..., 0, 2])],
        feature_intrinsics_expected=[10.0, expected_cx],
        finite_difference_max_err=fd_err,
        behind_camera_in_front=[int(v) for v in behind.in_front.reshape(-1).tolist()],
        behind_camera_support=float(behind.support.max()),
        cfa_offsets_match_synthetic=cfa_ok,
        cfa_coordinate_grbg_r_3_5=cfa_coordinate,
        tolerance={"stride_px": 1e-9, "finite_difference": 1e-6},
        threshold_link="protocol.json 三维门槛的 feature/original px 换算依赖该 stride=32 单位链",
    )
    return [c25, c26]


def _c2_pixel_center(torch, geometry) -> dict[str, Any]:
    spec = geometry["GridSpec"](8, 8, pixel_stride=4)
    xy = torch.tensor([[[0.0, 0.0], [7.0, 7.0], [3.5, 2.5]]], dtype=torch.float64)
    normalized = spec.normalized_from_xy(xy)
    expected_first = (-0.875, -0.875)
    expected_last = (0.875, 0.875)
    roundtrip = float((spec.xy_from_normalized(normalized) - xy).abs().max())
    feature = spec.pixel_xy_to_feature_xy(xy)
    stride_err = float((feature * 4.0 - xy).abs().max())
    corner_in = bool(spec.in_bounds_xy(torch.tensor([[[-0.5, -0.5]]], dtype=torch.float64))[0, 0])
    outside = bool(spec.in_bounds_xy(torch.tensor([[[8.5, 8.5]]], dtype=torch.float64))[0, 0])
    errors = [
        abs(float(normalized[0, 0, 0]) - expected_first[0]),
        abs(float(normalized[0, 0, 1]) - expected_first[1]),
        abs(float(normalized[0, 1, 0]) - expected_last[0]),
        abs(float(normalized[0, 1, 1]) - expected_last[1]),
        roundtrip,
        stride_err,
    ]
    max_err = max(errors)
    ok = max_err <= 1e-12 and corner_in and not outside
    return _sub(
        "C2.1",
        "像素中心 / 归一化坐标 / stride 换算",
        "passed" if ok else "failed",
        "" if ok else "像素中心约定与手算不符或边界判定错误",
        normalized_first=[float(v) for v in normalized[0, 0]],
        normalized_last=[float(v) for v in normalized[0, 1]],
        hand_first=list(expected_first),
        hand_last=list(expected_last),
        roundtrip_max_err=roundtrip,
        stride_roundtrip_max_err=stride_err,
        in_bounds_minus_half=corner_in,
        in_bounds_eight_and_half=outside,
        tolerance=1e-12,
    )


def _c2_rotations(torch, geometry) -> dict[str, Any]:
    focal = 10.0
    K = torch.tensor(
        [[focal, 0.0, 127.5], [0.0, focal, 127.5], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    identity = torch.eye(4, dtype=torch.float64)

    def rotation(axis: str, angle_deg: float) -> Any:
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        if axis == "x":
            matrix = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)
        elif axis == "y":
            matrix = torch.tensor([[c, 0, s], [0, 1, 0], [-s, c, 0]], dtype=torch.float64)
        else:
            matrix = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)
        transform = torch.eye(4, dtype=torch.float64)
        transform[:3, :3] = matrix
        return transform

    rows = []
    max_err = 0.0
    for case in _c2_hand_cases():
        transform = rotation(case["axis"], case["angle_deg"])
        uv, in_front = geometry["forward_project_points"](
            torch.tensor([[[list(case["pixel"])]]], dtype=torch.float64),
            torch.tensor([[[0.6]]], dtype=torch.float64),
            K,
            K,
            identity,
            transform,
        )
        got = (float(uv[0, 0, 0, 0]), float(uv[0, 0, 0, 1]))
        err = max(abs(got[0] - case["expect"][0]), abs(got[1] - case["expect"][1]))
        max_err = max(max_err, err)
        rows.append(
            {
                "name": case["name"],
                "got_px": list(got),
                "hand_px": list(case["expect"]),
                "in_front": bool(in_front[0, 0, 0]),
                "error_px": err,
            }
        )
    rotate_z = rotation("z", 90.0)
    rotate_y = rotation("y", 90.0)
    composite_ab = geometry["compose_transforms"](rotate_z, rotate_y)
    composite_ba = geometry["compose_transforms"](rotate_y, rotate_z)
    noncommute = float((composite_ab - composite_ba).abs().max())
    inverse_error = float(
        (
            geometry["invert_transform"](composite_ab)
            - geometry["compose_transforms"](
                geometry["invert_transform"](rotate_y), geometry["invert_transform"](rotate_z)
            )
        )
        .abs()
        .max()
    )
    identity_error = float(
        (
            geometry["compose_transforms"](geometry["invert_transform"](composite_ab), composite_ab)
            - torch.eye(4, dtype=torch.float64)
        )
        .abs()
        .max()
    )
    ok = (
        max_err <= 1e-9
        and all(row["in_front"] for row in rows)
        and noncommute > 1e-6
        and inverse_error <= 1e-12
        and identity_error <= 1e-12
    )
    return _sub(
        "C2.2",
        "K 已知的 90 度旋转与非交换复合/逆",
        "passed" if ok else "failed",
        "" if ok else "90 度旋转手算对照或复合/逆恒等式不成立",
        cases=rows,
        max_hand_error_px=max_err,
        noncommutativity_max_abs=noncommute,
        inverse_order_error=inverse_error,
        inverse_identity_error=identity_error,
        tolerance_rotation_px=1e-9,
        hand_derivation="X_cur = R_cam^T X_prev, u = f*x/z + c_u（numpy 独立参照另有 C1.1 核）",
    )


def _c2_depth_sign(torch, geometry) -> dict[str, Any]:
    K = torch.tensor(
        [[320.0, 0.0, 127.5], [0.0, 320.0, 127.5], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    points = torch.tensor([[[0.15, 0.0, -0.6], [0.15, 0.0, 0.6]]], dtype=torch.float64)
    uv, in_front = geometry["project_points"](points, K)
    uv = uv.reshape(-1, 2)
    front = in_front.reshape(-1)
    negative_ok = not bool(front[0]) and bool(front[1])
    expected_negative = (47.5, 127.5)
    uv_err = max(
        abs(float(uv[0, 0]) - expected_negative[0]),
        abs(float(uv[0, 1]) - expected_negative[1]),
    )
    rotate_180 = torch.eye(4, dtype=torch.float64)
    rotate_180[:3, :3] = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float64
    )
    _, forward_front = geometry["forward_project_points"](
        torch.tensor([[[[207.5, 127.5]]]], dtype=torch.float64),
        torch.tensor([[[0.6]]], dtype=torch.float64),
        K,
        K,
        torch.eye(4, dtype=torch.float64),
        rotate_180,
    )
    occluded = bool(forward_front.reshape(-1)[0])
    ok = negative_ok and uv_err <= 1e-12 and not occluded
    return _sub(
        "C2.3",
        "正/负深度与 in_front 支持掩码",
        "passed" if ok else "failed",
        "" if ok else "负深度未被标记为 in_front=False，或 180 度旋转后的遮挡未被标记",
        negative_pixel=[float(uv[0, 0]), float(uv[0, 1])],
        negative_expected_px=list(expected_negative),
        negative_uv_error_px=uv_err,
        in_front_positive=bool(front[1]),
        in_front_negative=bool(front[0]),
        behind_camera_in_front=occluded,
    )


def _c2_cfa(torch) -> dict[str, Any]:
    from owac.frameworks.n04.model import linear_measurement_planes, measurement_from_linear_planes
    from owac.frameworks.n04.synthetic import cfa_plane_offsets

    patterns = ("RGGB", "BGGR", "GRBG", "GBRG")
    law_rows = []
    law_ok = True
    for pattern in patterns:
        red_row = 0 if "R" in pattern[:2] else 1
        expected = []
        for index in range(4):
            dy, dx = divmod(index, 2)
            channel = pattern[dy * 2 + dx]
            if channel == "G":
                channel = "Gr" if dy == red_row else "Gb"
            expected.append((channel, dy, dx))
        declared = [(str(c), int(dy), int(dx)) for c, dy, dx in cfa_plane_offsets(pattern)]
        match = declared == expected
        law_ok = law_ok and match
        law_rows.append(
            {"pattern": pattern, "declared": declared, "expected": expected, "match": match}
        )
    samples = torch.rand(2, 1, 8, 8, dtype=torch.float64)
    planes, _ = linear_measurement_planes(samples, torch.ones_like(samples), "bayer_cfa")
    restored = measurement_from_linear_planes(planes, "bayer_cfa")
    roundtrip = float((restored - samples).abs().max())
    ok = law_ok and roundtrip <= 1e-15
    return _sub(
        "C2.4",
        "CFA 四相位坐标律与平面打包往返",
        "passed" if ok else "failed",
        "" if ok else "CFA 相位命名/坐标律不满足标准 Bayer 定义或打包往返不精确",
        patterns=patterns,
        phase_law=law_rows,
        roundtrip_max_err=roundtrip,
        roundtrip_tolerance=1e-15,
        law="平面 (i,j) 对应原始格点 (2i+dy, 2j+dx)；G 与 R 同行记 Gr，否则记 Gb",
    )


# --------------------------------------------------------------------------------------
# C3 未观测值替换
# --------------------------------------------------------------------------------------


def _check_c3(ctx: RunContext) -> dict[str, Any]:
    torch = _torch()
    encoder = _tiny_encoder(torch, seed=ctx.seed + 3, predictor="identity")
    context = _context(torch, actions=[[0.0, 0.0, 0.0]])
    mask = torch.ones(1, 1, 16, 16)
    mask[:, :, :8, :] = 0.0
    generator = torch.Generator().manual_seed(ctx.seed + 4)
    base = torch.rand(1, 3, 16, 16, generator=generator)
    clean = base * mask
    nan_values = torch.where(mask > 0, base, torch.full_like(base, float("nan")))
    huge_values = torch.where(mask > 0, base, torch.full_like(base, 1e30))
    sanitized = torch.where(mask > 0, nan_values, torch.zeros_like(nan_values))

    reference = _step_with_grads(torch, encoder, clean, context, mask)
    rows = []
    for label, values in (("nan", nan_values), ("huge", huge_values), ("sanitized", sanitized)):
        outcome = _step_with_grads(torch, encoder, values, context, mask)
        fields = ("stable", "change", "uncertainty", "representation")
        finite = all(_finite_all(torch, outcome[key]) for key in fields)
        equal = all(bool(torch.equal(outcome[key], reference[key])) for key in fields)
        grads = _grad_diff(torch, reference["grads"], outcome["grads"])
        rows.append(
            {
                "substitution": label,
                "posterior_stable_finite": _finite_all(torch, outcome["stable"]),
                "uncertainty_finite": _finite_all(torch, outcome["uncertainty"]),
                "change_finite": _finite_all(torch, outcome["change"]),
                "representation_finite": _finite_all(torch, outcome["representation"]),
                "all_fields_finite": finite,
                "equal_to_clean_masked_run": equal,
                **grads,
            }
        )
    nan_row, huge_row, sanitized_row = rows
    subs = []
    nan_ok = bool(
        nan_row["all_fields_finite"]
        and nan_row["equal_to_clean_masked_run"]
        and nan_row["param_grads_nonfinite"] == 0
    )
    subs.append(
        _sub(
            "C3.1",
            "未观测区替换 NaN：更新与梯度不受影响",
            "passed" if nan_ok else "failed",
            ""
            if nan_ok
            else "未观测像素的 NaN 进入更新路径：后验/协方差出现非有限值或参数梯度非有限；"
            "合同要求在调用 step 前用 where(valid, value, 有限占位) 清洗"
            "（implementation_contract.md CAUSAL API）",
            **nan_row,
        )
    )
    huge_ok = bool(
        huge_row["all_fields_finite"]
        and huge_row["equal_to_clean_masked_run"]
        and huge_row["param_grads_nonfinite"] == 0
    )
    subs.append(
        _sub(
            "C3.2",
            "未观测区替换极大值 1e30：更新与梯度不受影响",
            "passed" if huge_ok else "failed",
            "" if huge_ok else "未观测区极大值改变了后验或梯度",
            **huge_row,
        )
    )
    sanitized_ok = bool(
        sanitized_row["all_fields_finite"]
        and sanitized_row["equal_to_clean_masked_run"]
        and sanitized_row["param_grads_nonfinite"] == 0
    )
    subs.append(
        _sub(
            "C3.3",
            "按合同清洗（where(valid, value, 0)）后与干净掩码运行一致",
            "passed" if sanitized_ok else "failed",
            "" if sanitized_ok else "清洗后仍与干净运行不一致，问题不止在 NaN×0",
            **sanitized_row,
        )
    )
    return _group(
        "C3",
        "未观测值 NaN / 极大值：更新与梯度不受影响",
        subs,
        note="被测接口为现存 N04Encoder.step（N04FullEncoder 尚未实现）；后验均值、变化单元、"
        "不确定度与全部参数梯度均纳入比较，未只检查有限性。",
    )


# --------------------------------------------------------------------------------------
# C4 因果上下文与置乱
# --------------------------------------------------------------------------------------


def _check_c4(ctx: RunContext) -> dict[str, Any]:
    torch = _torch()
    subs = [
        _c4_measurement_blindness(ctx, torch),
        _c4_context_selection(ctx, torch),
        _c4_routing(ctx, torch),
    ]
    return _group(
        "C4",
        "未来动作与当前图像置乱；已执行控制改变上下文选择",
        subs,
        note="模型效果留 S3；此处检查信息路由与上下文选择。head 为未训练零初始化，"
        "路由检查使用审计专用的确定性结构扰动并在证据中标注。",
    )


def _c4_measurement_blindness(ctx: RunContext, torch) -> dict[str, Any]:
    from owac.frameworks.n04.model import N04Encoder

    signature_params = list(inspect.signature(N04Encoder.predict_before_observe).parameters)
    measurement_in_signature = [name for name in signature_params if "measure" in name.lower()]
    encoder = _tiny_encoder(torch, seed=ctx.seed + 11, predictor="learned_proprio")
    perturbed = _perturb_head(torch, encoder, seed=ctx.seed + 12)
    actions = [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
    context = _context(torch, actions=actions)
    generator = torch.Generator().manual_seed(ctx.seed + 13)
    values = torch.rand(2, 3, 16, 16, generator=generator)
    first = encoder.step(values, context, encoder.init_state(2)["head"], camera_id="head")
    second = encoder.step(values[[1, 0]], context, encoder.init_state(2)["head"], camera_id="head")
    permuted = second.state.prediction[[1, 0]]
    diff = float((first.state.prediction.detach() - permuted.detach()).abs().max())
    ok = not measurement_in_signature and diff <= 1e-12
    return _sub(
        "C4.1",
        "预测签名与输出不读当前图像（batch 置换逐样本不变）",
        "passed" if ok else "failed",
        "" if ok else "predict 路径读取了当前测量或 batch 置换改变了归属",
        predict_parameters=signature_params,
        measurement_parameters=measurement_in_signature,
        head_perturbed=perturbed,
        permutation="batch [0,1] -> [1,0]",
        max_prior_abs_diff_after_permutation=diff,
        tolerance=1e-12,
    )


def _c4_context_selection(ctx: RunContext, torch) -> dict[str, Any]:
    from dataclasses import replace as dataclass_replace

    from owac.training.n04 import RawVLAFrame, build_real_context

    signature_params = list(__import__("inspect").signature(build_real_context).parameters)
    base = RawVLAFrame(
        measured=torch.zeros(1, 3, 16, 16),
        valid_mask=torch.ones(1, 1, 16, 16),
        saturated_mask=torch.zeros(1, 1, 16, 16),
        proprio=torch.zeros(1, 8),
        action=torch.zeros(1, 7),
        metadata={"suite": "libero"},
    )
    changed = dataclass_replace(base, action=torch.full((1, 7), 3.0))
    context_a = build_real_context(base, None)
    context_b = build_real_context(changed, None)
    action_delta = float((context_a.executed_action - context_b.executed_action).abs().max())
    unavailable_a = context_a.motion_vector()[0]
    unavailable_b = context_b.motion_vector()[0]
    gated_delta = float((unavailable_a - unavailable_b).abs().max())
    available = dataclass_replace(
        changed, metadata={"suite": "libero", "executed_action_available": True}
    )
    available_base = dataclass_replace(
        base, metadata={"suite": "libero", "executed_action_available": True}
    )
    available_delta = float(
        (
            build_real_context(available, None).motion_vector()[0]
            - build_real_context(available_base, None).motion_vector()[0]
        )
        .abs()
        .max()
    )
    ok = (
        action_delta > 0
        and gated_delta <= 1e-12
        and available_delta > 0
        and signature_params == ["frame", "previous", "camera_index"]
        and not hasattr(base, "supervised_action")
        and context_a.info["executed_action_available"] is False
    )
    return _sub(
        "C4.2",
        "上下文选择跟随已执行控制；未标记可用的值不泄漏；未来监督字段不进入上下文",
        "passed" if ok else "failed",
        "" if ok else "上下文构造未正确按已执行/可用性选择信息",
        context_parameters=signature_params,
        executed_action_delta=action_delta,
        motion_vector_delta_when_unavailable=gated_delta,
        motion_vector_delta_when_available=available_delta,
        raw_frame_has_supervised_action=hasattr(base, "supervised_action"),
        executed_action_available=context_a.info["executed_action_available"],
    )


def _c4_routing(ctx: RunContext, torch) -> dict[str, Any]:
    """已执行控制改变先验的结构路由检查（需非零 head）。"""
    encoder = _tiny_encoder(torch, seed=ctx.seed + 21, predictor="learned_proprio")
    if not _perturb_head(torch, encoder, seed=ctx.seed + 22):
        return _sub(
            "C4.3",
            "已执行控制改变传播先验（结构路由）",
            "skipped",
            "传播器没有 head，无法做路由检查",
        )
    state = encoder.init_state(2)["head"]
    identical = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(ctx.seed + 23))
    values = identical.expand(2, -1, -1, -1).contiguous()
    context_ab = _context(torch, actions=[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    context_ba = _context(torch, actions=[[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]])
    stepped = encoder.step(values, context_ab, state, camera_id="head").state
    prior_ab = encoder.predict_before_observe(stepped, context_ab)[0]
    prior_ba = encoder.predict_before_observe(stepped, context_ba)[0]
    own_action_error = max(
        float((prior_ab[0].detach() - prior_ba[1].detach()).abs().max()),
        float((prior_ab[1].detach() - prior_ba[0].detach()).abs().max()),
    )
    action_effect = float((prior_ab[0].detach() - prior_ab[1].detach()).abs().max())
    ok = own_action_error <= 1e-12 and action_effect > 0
    return _sub(
        "C4.3",
        "已执行控制改变传播先验（结构路由，未训练权重）",
        "passed" if ok else "failed",
        "" if ok else "改变已执行控制未进入传播输出，或上下文在 batch 内串位",
        identical_measurements=True,
        max_prior_diff_same_context=own_action_error,
        max_prior_diff_action_0_vs_5=action_effect,
        tolerance_same_context=1e-12,
        scope="审计专用 head 结构扰动；不代表已训练模型效果（S3）",
    )


# --------------------------------------------------------------------------------------
# C5 迟到帧与因果时间
# --------------------------------------------------------------------------------------


def _c5_telemetry_context(loader_methods: dict[str, bool]) -> dict[str, Any]:
    """用合成 sidecar 检查冻结的时间合同（不伪造真实数据）。"""
    import numpy as np

    from owac.raw.telemetry import ExecutedControl, FrameTiming, TelemetrySidecar

    def timing(frame: int, mid: float, arrival: float) -> FrameTiming:
        return FrameTiming(
            frame=frame,
            clock_id="audit-clock",
            time_unit="s",
            exposure_start=mid - 0.005,
            exposure_mid=mid,
            exposure_end=mid + 0.005,
            arrival_time=arrival,
            source=f"audit-synthetic:frame{frame}",
        )

    def control(
        command_id: str, status: str, start: float, end: float, dispatched: float
    ) -> ExecutedControl:
        return ExecutedControl(
            command_id=command_id,
            value=np.zeros(7),
            mode="audit-synthetic",
            units="m+rad",
            status=status,
            evidence_level="executed_log",
            command_time=dispatched,
            start_time=start,
            end_time=end,
            time_unit="s",
            source=f"audit-synthetic:{command_id}",
        )

    sidecar = TelemetrySidecar(
        episode_id="audit-synthetic",
        suite="libero",
        time_unit="s",
        timing=(
            timing(0, 0.005, 0.02),
            timing(1, 0.105, 1.50),
            timing(2, 0.205, 0.25),
        ),
        controls=(
            control("c0", "executed", 0.00, 0.02, 0.00),
            control("c1", "executed", 0.12, 0.20, 0.12),
            control("c2", "executed", 0.22, 0.30, 0.22),
            control("c3", "planned", 0.32, 0.40, 0.32),
            control("c4", "planned", 0.25, 0.29, 0.25),
        ),
        provenance={"origin": "n04_contracts audit synthetic sidecar"},
    )
    early = sidecar.context_for(exposure_time=0.21, decision_time=0.30)
    late = sidecar.context_for(exposure_time=0.15, decision_time=1.60)
    no_proprio = sidecar.context_for(
        exposure_time=0.21, decision_time=0.30, proprio_available=False
    )

    import torch

    from owac.raw.rawvla_sequence_n04 import (
        RawVLASequence,
        RawVLASequenceLoader,
        SequenceAvailability,
    )

    sequence = RawVLASequence(
        suite="libero",
        measurements={"agentview": torch.zeros(3, 3, 4, 4)},
        proprio=torch.zeros(3, 8),
        executed_action=None,
        supervised_action=torch.zeros(3, 1, 7),
        supervised_action_mask=torch.zeros(3, 1, dtype=torch.bool),
        frame_ids=torch.arange(3),
        episode_id="audit-synthetic",
        camera_ids=("agentview",),
        new_measurement=torch.ones(3, dtype=torch.bool),
        time_unit="s",
        availability=SequenceAvailability(proprio=True, timestamp=True, exposure=True),
        metadata={},
        telemetry=sidecar,
    )
    loader = RawVLASequenceLoader(suite="libero", sidecar=sidecar)
    sequence_context = sequence.context_at(0.21, 0.30)
    loader_context = loader.context_at(sequence, 0.21, 0.30)
    sequence_interval = sequence.executed_between(0.105, 1.60)
    loader_interval = loader.executed_between(sequence, 0.105, 1.60)
    binding = {
        "sequence_context_type": type(sequence_context).__name__,
        "loader_context_type": type(loader_context).__name__,
        "context_matches_sidecar": sequence_context.as_dict()
        == early.as_dict()
        == loader_context.as_dict(),
        "sequence_interval_type": type(sequence_interval).__name__,
        "interval_controls": [c.command_id for c in sequence_interval.controls],
        "interval_matches_loader": sequence_interval.as_dict() == loader_interval.as_dict(),
        "loader_context_signature": str(inspect.signature(RawVLASequenceLoader.context_at)),
        "loader_executed_between_signature": str(
            inspect.signature(RawVLASequenceLoader.executed_between)
        ),
    }
    checks = {
        "late_frame_hidden_before_arrival": list(early.visible_frames) == [0, 2],
        "late_frame_flagged": bool(early.late_frame) and 1 in early.missing_frames,
        "latest_visible_frame": early.latest_frame == 2,
        "replay_executed_in_command_order": [c.command_id for c in early.replay_controls] == ["c2"],
        "planned_not_replayed": list(early.pending_control_ids) == ["c4"],
        "late_frame_visible_after_arrival": list(late.visible_frames) == [0, 1],
        "late_window_replay_order": [c.command_id for c in late.replay_controls] == ["c1", "c2"],
        "late_window_pending": list(late.pending_control_ids) == ["c3", "c4"],
        "arrival_delay_recorded": abs(float(late.arrival_delay) - 1.39) <= 1e-9,
        "target_actions_included_false": early.target_actions_included is False
        and late.target_actions_included is False,
        "proprio_follows_visible_frame": early.proprio_frame == 2
        and early.proprio_frame in early.visible_frames,
        "proprio_not_leaked_when_unavailable": no_proprio.proprio_frame is None
        and no_proprio.availability["proprio"] is False,
        "loader_time_selection_methods": all(loader_methods.values())
        and set(loader_methods) == {"context_at", "executed_between"},
        "loader_wrapper_binding": binding["context_matches_sidecar"]
        and binding["interval_matches_loader"]
        and binding["sequence_context_type"] == "OnlineContext"
        and binding["loader_context_type"] == "OnlineContext"
        and binding["sequence_interval_type"] == "ExecutedInterval"
        and binding["interval_controls"] == ["c1", "c2"],
    }
    ok = all(checks.values())
    return _sub(
        "C5.1",
        "迟到帧按曝光时刻回补缓存并重放已执行控制（telemetry 合成 sidecar）",
        "passed" if ok else "failed",
        "" if ok else "telemetry 上下文选择与冻结的时间合同不符",
        checks=checks,
        early_visible_frames=list(early.visible_frames),
        early_missing_frames=list(early.missing_frames),
        early_replay=[c.command_id for c in early.replay_controls],
        early_pending=list(early.pending_control_ids),
        late_visible_frames=list(late.visible_frames),
        late_replay=[c.command_id for c in late.replay_controls],
        late_pending=list(late.pending_control_ids),
        arrival_delay_s=None if late.arrival_delay is None else float(late.arrival_delay),
        age_since_exposure_s=(
            None if late.age_since_exposure is None else float(late.age_since_exposure)
        ),
        loader_methods=loader_methods,
        loader_binding=binding,
        source="owac.raw.telemetry.TelemetrySidecar.context_for；"
        "RawVLASequenceLoader.context_at/executed_between 为同一实现的加载器入口",
        scope="合成 sidecar 的语义检查；真实 NPZ 时间戳覆盖由 ws-telemetry 的远端运行报告",
    )


def _c5_adversarial_time_controls() -> dict[str, Any]:
    from dataclasses import replace

    import numpy as np

    from owac.raw.telemetry import ExecutedControl, FrameTiming, TelemetrySidecar

    timing = FrameTiming(
        frame=0,
        clock_id="shared",
        time_unit="s",
        exposure_start=0.0,
        exposure_mid=0.01,
        exposure_end=0.02,
        arrival_time=0.04,
    )
    control = ExecutedControl(
        command_id="c",
        value=np.ones(2),
        status="executed",
        evidence_level="executed_log",
        command_time=0.1,
        start_time=0.1,
        end_time=0.9,
        time_unit="s",
        source="fixture",
    )
    sidecar = TelemetrySidecar(
        episode_id="audit", time_unit="s", timing=(timing,), controls=(control,)
    )
    checks = {}
    for level in ("unknown", "inferred", "producer_code_order"):
        altered = replace(sidecar, controls=(replace(control, evidence_level=level),))
        checks[f"reject_{level}"] = not altered.context_for(
            exposure_time=0.01, decision_time=0.5
        ).replay_controls
    context = sidecar.context_for(exposure_time=0.01, decision_time=0.5)
    checks["no_future_duration"] = context.replay_controls[0].end_time == 0.5
    checks["separate_delay_units"] = (
        abs(context.arrival_delay - 0.02) < 1e-12 and abs(context.processing_delay - 0.46) < 1e-12
    )
    for name, altered in (
        ("mixed_unit", replace(sidecar, controls=(replace(control, time_unit="ms"),))),
        (
            "mixed_clock",
            replace(sidecar, timing=(timing, replace(timing, frame=1, clock_id="other"))),
        ),
    ):
        try:
            altered.context_for(exposure_time=0.01, decision_time=0.5)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    return _sub(
        "C5.4",
        "未核实控制、未来区间与时钟单位反例",
        "passed" if all(checks.values()) else "failed",
        "",
        checks=checks,
        scope="synthetic adversarial contract; not real-data coverage",
    )


def _check_c5(ctx: RunContext) -> dict[str, Any]:
    torch = _torch()
    subs: list[dict[str, Any]] = []
    ok_telemetry, detail = _probe_import("owac.raw.telemetry")
    loader_attrs: dict[str, bool] = {}
    ok_loader, loader_detail = _probe_import("owac.raw.rawvla_sequence_n04")
    if ok_loader:
        module = importlib.import_module("owac.raw.rawvla_sequence_n04")
        loader = getattr(module, "RawVLASequenceLoader", None)
        loader_attrs = (
            {name: hasattr(loader, name) for name in ("context_at", "executed_between")}
            if loader is not None
            else {}
        )
    if ok_telemetry:
        try:
            subs.append(_c5_telemetry_context(loader_attrs))
            subs.append(_c5_adversarial_time_controls())
        except Exception:  # noqa: BLE001 - 生产者接口异常本身就是要退回的证据
            subs.append(
                _sub(
                    "C5.1",
                    "迟到帧按曝光时刻回补缓存并重放已执行控制（telemetry 合成 sidecar）",
                    "failed",
                    "telemetry 在审计合成 sidecar 上抛异常：" + traceback.format_exc(limit=4),
                    loader_methods=loader_attrs,
                )
            )
    else:
        subs.append(
            _sub(
                "C5.1",
                "迟到帧按曝光时刻回补缓存并重放已执行控制（telemetry 合成 sidecar）",
                "skipped",
                f"owac.raw.telemetry 不可导入：{detail}",
                rerun="ws-telemetry 交付后重跑同一 CLI",
                dependency={"telemetry_importable": False, "loader_methods": loader_attrs},
            )
        )
    encoder = _tiny_encoder(torch, seed=ctx.seed + 31, predictor="learned_proprio")
    perturbed = _perturb_head(torch, encoder, seed=ctx.seed + 32)
    state = encoder.init_state(1)["head"]
    context_a = _context(
        torch, actions=[[9.0, 9.0, 9.0]], proprio=[[1e3, 1e3, 1e3, 1e3]], proprio_available=False
    )
    context_b = _context(
        torch,
        actions=[[-9.0, -9.0, -9.0]],
        proprio=[[-1e3, -1e3, -1e3, -1e3]],
        proprio_available=False,
    )
    prior_a = encoder.predict_before_observe(state, context_a)[0]
    prior_b = encoder.predict_before_observe(state, context_b)[0]
    leak = float((prior_a.detach() - prior_b.detach()).abs().max())
    subs.append(
        _sub(
            "C5.2",
            "未到达（availability=False）的本体/动作不能泄漏进先验",
            "passed" if leak <= 1e-12 else "failed",
            "" if leak <= 1e-12 else "标记不可用的值改变了先验",
            head_perturbed=perturbed,
            max_prior_diff_between_unavailable_values=leak,
            tolerance=1e-12,
            scope="可用性门控（编码器侧）；按物理时间的选择由 C5.1 覆盖",
        )
    )
    if ctx.live_data_root is not None:
        from owac.evaluation.n04_live_contracts import audit

        live = audit(Path(ctx.live_data_root))
        subs.append(
            _sub(
                "C5.3",
                "新独立扩展集的真实采集/到达/控制与几何合同",
                "passed" if live["passed"] else "failed",
                "按数组和执行日志重新核验；原 2403 NPZ 缺失字段仍不可用",
                live_audit=live,
                scope="新仿真扩展；host 到达与 sim 物理时间分别记账；非硬件曝光",
            )
        )
    else:
        subs.append(
            _sub(
                "C5.3",
                "真实数据的到达时刻/标定/已执行动作可用性",
                "skipped",
                "未提供新扩展数据根；原缓存时间、标定或执行记录不足",
                rerun="使用 --live-data-root 指向独立扩展集，逐文件复算",
            )
        )
    return _group(
        "C5",
        "迟到帧按曝光时刻回补缓存并重放已执行控制；未到达本体不泄漏",
        subs,
        note="C5.1/2/4 验证工程语义；C5.3 仅对显式提供的新数据根逐文件验收。",
    )


# --------------------------------------------------------------------------------------
# C6 sensor residual / 曝光增益 / 支持掩码
# --------------------------------------------------------------------------------------


def _check_c6(ctx: RunContext) -> dict[str, Any]:
    torch = _torch()
    subs = [
        _c6_static_residual(ctx, torch),
        _c6_exposure_gain(ctx),
        _c6_support_independence(ctx, torch),
    ]
    return _group(
        "C6",
        "sensor residual 数值精度；曝光增益改变均值/方差；饱和/遮挡独立",
        subs,
        note="静态 residual 在被测编码器的 sensor_history 参照路径上检查；曝光增益使用仓库内"
        "Poisson–Gaussian 诊断采样器做经验均值/方差标定（非生产 ISP）。",
    )


def _c6_static_residual(ctx: RunContext, torch) -> dict[str, Any]:
    from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
    from owac.training.n04 import build_context

    encoder = _tiny_encoder(
        torch, seed=ctx.seed + 41, predictor="identity", innovation_reference="sensor_history"
    )
    scene = SceneConfig(
        height=16, width=16, noise_sigma=0, object_colors=((0.4, 0.2, 0.1),), seed=ctx.seed + 42
    )
    sequence = SyntheticSequenceGenerator(scene).sequence(
        length=3, batch=2, condition="static", seed=ctx.seed + 43
    )
    state = encoder.init_state(2)["head"]
    previous = None
    residuals = []
    for frame in sequence:
        out = encoder.step(
            frame.measured,
            build_context(frame, previous),
            state,
            saturation=frame.saturated_mask,
            measurement_valid=frame.valid_mask,
        )
        state = out.state
        previous = frame
        residuals.append(float(out.innovation.residual.detach().abs().max()))
    after_history = residuals[1:]
    max_after = max(after_history) if after_history else float("nan")
    ok = max_after <= 1e-6
    return _sub(
        "C6.1",
        "静态无噪声/恒等几何/同曝光：sensor residual 接近数值精度",
        "passed" if ok else "failed",
        "" if ok else "有历史后的 sensor residual 超过数值精度容差",
        residual_max_by_step=residuals,
        residual_max_after_first_frame=max_after,
        first_frame_excluded_reason="首帧没有历史，残差=当前测量本身，不代表静态一致性",
        tolerance=1e-6,
        reference="innovation_reference=sensor_history, predictor=identity",
    )


def _c6_exposure_gain(ctx: RunContext) -> dict[str, Any]:
    try:
        from owac.evaluation.n04_raw_diagnostics import shot_read
    except Exception as exc:  # noqa: BLE001
        return _sub(
            "C6.2",
            "正曝光增益同时改变均值与方差",
            "skipped",
            f"Poisson–Gaussian 诊断采样器不可用：{type(exc).__name__}: {exc}",
            rerun="owac.evaluation.n04_raw_diagnostics.shot_read 可用后重跑本 CLI",
        )
    torch = _torch()
    generator = torch.Generator().manual_seed(ctx.seed + 51)
    clean = 0.1 + 0.25 * torch.rand(1, 1, 64, 64, generator=generator)
    alpha, beta, samples = 0.005, 1e-4, 400

    def collect(scale: float):
        draws = torch.stack(
            [shot_read(scale * clean, alpha, beta, generator) for _ in range(samples)]
        )
        mean = float(draws.mean())
        variance = float(draws.var(dim=0, unbiased=True).mean())
        return mean, variance

    mean_low, variance_low = collect(1.0)
    mean_high, variance_high = collect(2.0)
    mean_ratio = mean_high / mean_low
    variance_ratio = variance_high / variance_low
    predicted_variance_ratio = (alpha * 2.0 * float(clean.mean()) + beta) / (
        alpha * float(clean.mean()) + beta
    )
    ok = (
        abs(mean_ratio - 2.0) <= 0.02
        and abs(variance_ratio - predicted_variance_ratio) <= 0.1
        and variance_ratio > 1.5
    )
    return _sub(
        "C6.2",
        "正曝光增益同时改变均值与方差",
        "passed" if ok else "failed",
        "" if ok else "曝光增益只改变均值、或方差缩放与 Poisson–Gaussian 预测不符",
        samples=samples,
        mean_low=mean_low,
        mean_high=mean_high,
        mean_ratio=mean_ratio,
        variance_low=variance_low,
        variance_high=variance_high,
        variance_ratio=variance_ratio,
        predicted_variance_ratio=predicted_variance_ratio,
        alpha=alpha,
        beta=beta,
        sensor_model="owac.evaluation.n04_raw_diagnostics.shot_read"
        "（诊断用 Poisson–Gaussian 采样器）",
        scope="数据/传感器合同检查，不是生产 ISP 或机制结果",
    )


def _c6_support_independence(ctx: RunContext, torch) -> dict[str, Any]:
    from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
    from owac.training.n04 import build_context

    scene = SceneConfig(
        height=16, width=16, noise_sigma=0, object_colors=((0.4, 0.2, 0.1),), seed=ctx.seed + 61
    )
    sequence = SyntheticSequenceGenerator(scene).sequence(
        length=2, batch=1, condition="static", seed=ctx.seed + 62
    )
    frame = sequence[-1]
    region = torch.zeros(1, 1, 16, 16)
    region[:, :, :8, :] = 1.0
    rows = []
    for label, saturation, measurement_valid in (
        ("saturation_only", region, None),
        ("occlusion_only", None, 1.0 - region),
        ("none", None, None),
    ):
        encoder = _tiny_encoder(torch, seed=ctx.seed + 63, predictor="identity")
        first = encoder.step(
            frame.measured, build_context(frame, None), encoder.init_state(1)["head"]
        )
        second = encoder.step(
            frame.measured,
            build_context(frame, frame),
            first.state,
            saturation=saturation,
            measurement_valid=measurement_valid,
        )
        gate = second.state.decomposition["g_update"].detach()
        rows.append(
            {
                "case": label,
                "g_update_max_in_masked_region": float(gate[:, :, :8, :].max()),
                "g_update_max_outside": float(gate[:, :, 8:, :].max()),
            }
        )
    saturation_row, occlusion_row, none_row = rows
    ok = (
        saturation_row["g_update_max_in_masked_region"] <= 1e-12
        and occlusion_row["g_update_max_in_masked_region"] <= 1e-12
        and none_row["g_update_max_in_masked_region"] > 0
        and saturation_row["g_update_max_outside"] > 0
        and occlusion_row["g_update_max_outside"] > 0
    )
    return _sub(
        "C6.3",
        "饱和与遮挡各自独立地关闭更新支持",
        "passed" if ok else "failed",
        "" if ok else "饱和或遮挡掩码未独立关闭更新门",
        cases=rows,
        region="特征场上半 [:, :, :8, :]",
        tolerance=1e-12,
    )


# --------------------------------------------------------------------------------------
# C7 manifest 与拆分
# --------------------------------------------------------------------------------------


def _hash_unit(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def _reference_group_split(
    ids: list[str], seed: int, ratio: dict[str, float]
) -> dict[str, list[str]]:
    """审计侧参照拆分：按 (hash(seed:id), id) 排序后按比例切分。"""
    ordered = sorted(ids, key=lambda item: (_hash_unit(item, seed), item))
    total = len(ordered)
    train_end = int(math.floor(total * float(ratio["train"])))
    dev_end = train_end + int(
        math.floor(total * (float(ratio["train"]) + float(ratio["dev"]))) - train_end
    )
    return {
        "train": ordered[:train_end],
        "dev": ordered[train_end:dev_end],
        "final": ordered[dev_end:],
    }


def _check_c7(ctx: RunContext) -> dict[str, Any]:
    subs = []
    try:
        subs.append(_c7_split_kernel(ctx))
    except DependencyMissing as exc:
        subs.append(_sub("C7.1", "拆分参照核", "skipped", str(exc)))
    ok_manifest, manifest_detail = _probe_import("owac.raw.n04_manifest")
    if ok_manifest:
        try:
            subs.append(_c7_manifest_split(ctx))
        except Exception:  # noqa: BLE001 - producer 接口异常本身就是要退回的证据
            subs.append(
                _sub(
                    "C7.2",
                    "ws-manifest 分层拆分：确定性、源 ID 互斥、变体不拆散、成员表 hash",
                    "failed",
                    "build_split 在审计合成记录上抛异常：" + traceback.format_exc(limit=4),
                )
            )
    else:
        subs.append(
            _sub(
                "C7.2",
                "ws-manifest 分层拆分：确定性、源 ID 互斥、变体不拆散、成员表 hash",
                "skipped",
                f"owac.raw.n04_manifest 不可导入：{manifest_detail}",
                rerun="ws-manifest 交付后重跑同一 CLI",
            )
        )
    if ctx.manifest_path is not None:
        subs.append(_verify_manifest_file(ctx))
    else:
        producer_main = None
        if ok_manifest:
            producer_main = callable(
                getattr(importlib.import_module("owac.raw.n04_manifest"), "main", None)
            )
        subs.append(
            _sub(
                "C7.3",
                "真实 manifest 产物复核（拆分互斥 / 成员表 hash / 二次生成一致）",
                "skipped",
                "本地没有远端 NPZ 数据根，未运行完整 manifest；需 ws-manifest 的 splits.json 产物",
                rerun="python -m owac.raw.n04_manifest --config configs/n04/full/s0_data.json "
                "--output <run_dir>/inventory.json；再运行本 CLI 并加 "
                "--manifest <run_dir>/splits.json",
                producer_cli="python -m owac.raw.n04_manifest --config <s0_data.json> "
                "--output <inventory.json>",
                producer_main_present=producer_main,
            )
        )
    return _group(
        "C7",
        "manifest 两次生成一致、hash 稳定、拆分源 ID 交集为零",
        subs,
        note="C7.1 验证审计侧拆分参照核；C7.2 在合成元数据上验证 ws-manifest build_split "
        "的确定性/互斥/覆盖/成员表 hash；C7.3 需远端真实产物或 --manifest。",
    )


def _c7_manifest_split(ctx: RunContext) -> dict[str, Any]:
    """ws-manifest 拆分函数在合成记录上的行为检查（不伪造真实数据）。"""
    from owac.raw import n04_manifest as manifest

    records = []
    for platform_name, task in (
        ("libero", "task_a"),
        ("libero", "task_b"),
        ("robotwin2", "task_a"),
        ("robotwin2", "task_b"),
    ):
        for index in range(12):
            source = f"{platform_name}_{task}_src{index:02d}"
            for variant in ("raw0", "raw1"):
                records.append(
                    {
                        "source_trajectory_id": source,
                        "platform": platform_name,
                        "task_key": task,
                        "relpath": f"{source}/{variant}.npz",
                    }
                )
    ratios = ctx.config.get("split_ratio", {"train": 0.8, "dev": 0.1, "final": 0.1})
    seed = int(ctx.config.get("split_seed", ctx.seed))
    first = manifest.build_split(records, seed=seed, ratios=ratios, name="audit-synthetic")
    second = manifest.build_split(records, seed=seed, ratios=ratios, name="audit-synthetic")
    deterministic = (
        first["member_table_sha256"] == second["member_table_sha256"]
        and first["assignments"] == second["assignments"]
    )
    assignments = first["assignments"]
    sources = [item["source_trajectory_id"] for item in assignments]
    unique = len(set(sources)) == len(sources)
    overlap = first["source_overlap"]
    no_overlap = all(len(values) == 0 for values in overlap.values())
    files_per_source = {item["source_trajectory_id"]: item["n_files"] for item in assignments}
    cohesion = all(files_per_source.get(source) == 2 for source in set(sources))
    member_table = [
        {"source_trajectory_id": item["source_trajectory_id"], "split": item["split"]}
        for item in assignments
    ]
    recomputed = manifest.sha256_bytes(manifest.canonical_json(member_table))
    hash_ok = recomputed == first["member_table_sha256"]
    counts = first["counts_by_split"]
    coverage_ok = sum(counts.values()) == len(set(sources)) == first["n_sources"]
    ok = deterministic and unique and no_overlap and cohesion and hash_ok and coverage_ok
    return _sub(
        "C7.2",
        "ws-manifest 分层拆分：确定性、源 ID 互斥、变体不拆散、成员表 hash",
        "passed" if ok else "failed",
        "" if ok else "合成记录上的拆分不满足确定性/互斥/覆盖/成员表 hash 之一",
        synthetic_sources=len(set(sources)),
        synthetic_files=len(records),
        deterministic=deterministic,
        unique_sources=unique,
        source_overlap=overlap,
        variants_kept_together=cohesion,
        member_table_sha256=first["member_table_sha256"],
        recomputed_member_table_sha256=recomputed,
        counts_by_split=counts,
        seed=seed,
        ratios=ratios,
        producer_cli="python -m owac.raw.n04_manifest --config <s0_data.json> "
        "--output <inventory.json>",
        scope="拆分函数在合成元数据上的行为；完整 inventory 需要远端 NPZ，未在本地运行",
    )


def _c7_split_kernel(ctx: RunContext) -> dict[str, Any]:
    ratio = ctx.config.get("split_ratio", {"train": 0.8, "dev": 0.1, "final": 0.1})
    ids = [f"traj_{index:04d}" for index in range(100)]
    first = _reference_group_split(ids, ctx.seed, ratio)
    second = _reference_group_split(ids, ctx.seed, ratio)
    stable = first == second
    sets = {name: set(values) for name, values in first.items()}
    pairwise = {
        "train_dev": len(sets["train"] & sets["dev"]),
        "train_final": len(sets["train"] & sets["final"]),
        "dev_final": len(sets["dev"] & sets["final"]),
    }
    union = sets["train"] | sets["dev"] | sets["final"]
    proportions = {name: len(values) / len(ids) for name, values in first.items()}
    ratio_error = max(
        abs(proportions[name] - float(ratio[name])) for name in ("train", "dev", "final")
    )
    ok = stable and not any(pairwise.values()) and len(union) == len(ids) and ratio_error <= 0.02
    return _sub(
        "C7.1",
        "拆分参照核：确定性、源 ID 交集为零、比例符合冻结 seed/比例",
        "passed" if ok else "failed",
        "" if ok else "参照拆分不满足确定性/互斥/覆盖/比例要求",
        seed=ctx.seed,
        counts={name: len(values) for name, values in first.items()},
        proportions=proportions,
        target_ratio=ratio,
        max_ratio_error=ratio_error,
        pairwise_intersection_sizes=pairwise,
        union_size=len(union),
        ids_total=len(ids),
        hash_recipe="sha256(f'{seed}:{source_id}')[:16] as unit float, then sort tie-break by id",
        scope="审计侧参照实现；ws-manifest 产物仍须用 --manifest 复核",
    )


def _manifest_member_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("source_trajectory_id", "source_id", "id", "trajectory_id"):
            if key in value:
                return str(value[key])
        return None
    return None


def _extract_manifest_splits(data: dict[str, Any]) -> dict[str, list[str]] | None:
    raw: Any = data.get("splits")
    if not isinstance(raw, dict):
        if all(key in data for key in ("train", "dev", "final")):
            raw = {key: data[key] for key in ("train", "dev", "final")}
        else:
            return None
    splits: dict[str, list[str]] = {}
    for name, values in raw.items():
        if not isinstance(values, list):
            return None
        ids = [
            member_id for member_id in (_manifest_member_id(item) for item in values) if member_id
        ]
        splits[str(name)] = ids
    if not splits:
        return None
    return splits


def _verify_manifest_file(ctx: RunContext) -> dict[str, Any]:
    """复核 ws-manifest 产物：拆分互斥、成员表 hash、二次生成一致。"""
    path = Path(ctx.manifest_path)
    title = "真实 manifest 产物复核（拆分互斥 / 成员表 hash / 二次生成一致）"
    if not path.is_file():
        return _sub("C7.3", title, "failed", f"--manifest 路径不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    main = data.get("main")
    if isinstance(main, dict) and isinstance(main.get("assignments"), list):
        return _verify_producer_splits(path, main, title)
    splits = _extract_manifest_splits(data)
    if splits is None:
        return _sub(
            "C7.3",
            title,
            "skipped",
            "manifest schema 无法识别（需要 main.assignments、splits 或顶层 train/dev/final）",
            manifest_path=str(path),
            sha256=_sha256_file(path),
        )
    sets = {name: set(values) for name, values in splits.items()}
    names = sorted(sets)
    overlap = {
        f"{left}|{right}": len(sets[left] & sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    union = set().union(*sets.values()) if sets else set()
    total = sum(len(values) for values in sets.values())
    ok = total == len(union) and not any(overlap.values())
    return _sub(
        "C7.3",
        title,
        "passed" if ok else "failed",
        "" if ok else "拆分成员存在重复或源 ID 交集不为零",
        manifest_path=str(path),
        sha256=_sha256_file(path),
        splits={name: len(values) for name, values in splits.items()},
        overlap=overlap,
        union_size=len(union),
        members_total=total,
    )


def _verify_producer_splits(path: Path, main: dict[str, Any], title: str) -> dict[str, Any]:
    from owac.raw import n04_manifest as manifest

    assignments = main["assignments"]
    sources = [str(item["source_trajectory_id"]) for item in assignments]
    unique = len(set(sources)) == len(sources)
    by_split: dict[str, set[str]] = {}
    for item in assignments:
        by_split.setdefault(str(item["split"]), set()).add(str(item["source_trajectory_id"]))
    names = sorted(by_split)
    overlap = {
        f"{left}|{right}": len(by_split[left] & by_split[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    counts = {name: len(ids) for name, ids in by_split.items()}
    coverage_ok = sum(counts.values()) == len(set(sources)) == int(main.get("n_sources", -1))
    cohesion = all(int(item["n_files"]) == len(item.get("relpaths", [])) for item in assignments)

    member_table = [
        {"source_trajectory_id": str(item["source_trajectory_id"]), "split": str(item["split"])}
        for item in assignments
    ]
    recomputed = manifest.sha256_bytes(manifest.canonical_json(member_table))
    recorded = main.get("member_table_sha256")
    hash_ok = recomputed == recorded

    records = []
    for item in assignments:
        for relpath in item.get("relpaths", []):
            records.append(
                {
                    "source_trajectory_id": str(item["source_trajectory_id"]),
                    "platform": str(item["platform"]),
                    "task_key": str(item["task_key"]),
                    "relpath": str(relpath),
                }
            )
    regenerated_hash = None
    regenerated_ok = None
    if records:
        regenerated = manifest.build_split(
            records,
            seed=int(main["seed"]),
            ratios=dict(main["ratios"]),
            name=str(main.get("name", "main")),
        )
        regenerated_hash = regenerated["member_table_sha256"]
        regenerated_ok = regenerated_hash == recorded

    siblings: dict[str, Any] = {}
    hashes_path = path.parent / "inventory-hashes.json"
    if hashes_path.is_file():
        sibling = json.loads(hashes_path.read_text(encoding="utf-8"))
        siblings["inventory-hashes.main_member_table_sha256"] = sibling.get(
            "main_member_table_sha256"
        )
    stdout_path = path.parent / "stdout.json"
    if stdout_path.is_file():
        sibling = json.loads(stdout_path.read_text(encoding="utf-8"))
        siblings["stdout.main_split.member_table_sha256"] = (sibling.get("main_split") or {}).get(
            "member_table_sha256"
        )
    sibling_ok = all(value == recorded for value in siblings.values()) if siblings else None

    ok = (
        unique
        and not any(overlap.values())
        and coverage_ok
        and cohesion
        and hash_ok
        and regenerated_ok is not False
        and sibling_ok is not False
    )
    return _sub(
        "C7.3",
        title,
        "passed" if ok else "failed",
        "" if ok else "拆分互斥/覆盖/成员表 hash/二次生成一致之一不成立",
        manifest_path=str(path),
        sha256=_sha256_file(path),
        counts_by_split=counts,
        source_overlap=overlap,
        unique_sources=unique,
        coverage_ok=coverage_ok,
        variants_kept_together=cohesion,
        seed=main.get("seed"),
        ratios=main.get("ratios"),
        member_table_sha256=recorded,
        recomputed_member_table_sha256=recomputed,
        regenerated_member_table_sha256=regenerated_hash,
        regenerated_from_same_records=regenerated_ok,
        sibling_hashes=siblings,
        sibling_hashes_match=sibling_ok,
        source="ws-manifest research/experiments/results/2026-10-14_n04_manifest/splits.json",
    )


# --------------------------------------------------------------------------------------
# 运行器
# --------------------------------------------------------------------------------------


class RunContext:
    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path,
        protocol: dict[str, Any] | None,
        protocol_path: Path,
        seed: int,
        manifest_path: str | None,
        threads: int,
        live_data_root: str | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.protocol = protocol
        self.protocol_path = protocol_path
        self.seed = seed
        self.manifest_path = manifest_path
        self.threads = threads
        self.live_data_root = live_data_root


def _safe_check(
    check_id: str, title: str, fn: Callable[[RunContext], dict[str, Any]], ctx: RunContext
) -> dict[str, Any]:
    try:
        return fn(ctx)
    except DependencyMissing as exc:
        return {
            "id": check_id,
            "title": title,
            "status": "skipped",
            "required": False,
            "note": str(exc),
            "subchecks": [],
        }
    except Exception:  # noqa: BLE001 - 审计不应因单条检查崩溃而丢失其余证据
        return {
            "id": check_id,
            "title": title,
            "status": "failed",
            "required": False,
            "note": "未预期异常：" + traceback.format_exc(limit=4),
            "subchecks": [],
        }


def run_audit(ctx: RunContext, selected: list[str] | None = None) -> dict[str, Any]:
    dispatch: dict[str, Callable[[RunContext], dict[str, Any]]] = {
        "C0": _check_c0,
        "C1": _check_c1,
        "C2": _check_c2,
        "C3": _check_c3,
        "C4": _check_c4,
        "C5": _check_c5,
        "C6": _check_c6,
        "C7": _check_c7,
    }
    checks: list[dict[str, Any]] = []
    entries = [("C0", "protocol.json 冻结与换算一致性（附加）", False)]
    entries.extend((check_id, title, True) for check_id, title in REQUIRED_CHECKS)
    for check_id, title, required in entries:
        result = _safe_check(check_id, title, dispatch[check_id], ctx)
        result.setdefault("id", check_id)
        result.setdefault("title", title)
        if required:
            result["required"] = True
        checks.append(result)
    if selected:
        keep = set(selected) | {"C0"}
        checks = [check for check in checks if check["id"] in keep]
    summary = {
        "total": len(checks),
        "passed": sum(1 for check in checks if check["status"] == "passed"),
        "failed": sum(1 for check in checks if check["status"] == "failed"),
        "skipped": sum(1 for check in checks if check["status"] == "skipped"),
        "partial": sum(1 for check in checks if check["status"] == "partial"),
        "required_total": sum(1 for check in checks if check.get("required", False)),
    }
    return {"checks": checks, "summary": summary}


def build_report(ctx: RunContext, selected: list[str] | None, started: float) -> dict[str, Any]:
    results = run_audit(ctx, selected)
    report = {
        "schema": SCHEMA,
        "stage": "S0",
        "workflow": "D 合同验证与预注册门槛",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - started, 3),
        "command": " ".join(sys.argv),
        "repo": _git_info(),
        "source": {
            "config": {"path": str(ctx.config_path), "sha256": _sha256_file(ctx.config_path)},
            "module": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "protocol": (
                {"path": str(ctx.protocol_path), "sha256": _sha256_file(ctx.protocol_path)}
                if ctx.protocol_path.is_file()
                else {"path": str(ctx.protocol_path), "sha256": None}
            ),
            "protocol_frozen": None if ctx.protocol is None else bool(ctx.protocol.get("frozen")),
            "dependencies": _dependency_sources(
                (
                    "owac.raw.n04_mechanism_data",
                    "owac.raw.n04_manifest",
                    "owac.raw.telemetry",
                    "owac.raw.rawvla_sequence_n04",
                    "owac.frameworks.n04.measurement_geometry",
                    "owac.frameworks.n04.geometry",
                    "owac.frameworks.n04.model",
                    "owac.evaluation.n04_raw_diagnostics",
                )
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": _module_version("torch"),
            "numpy": _module_version("numpy"),
        },
        "availability_policy": {
            "scope": "以下缺失字段只描述原 2403 缓存；新扩展独立由 C5.3 实读验收",
            "new_extension_root": ctx.live_data_root,
            "timestamps": "unavailable",
            "calibration": "unavailable（K / T / 畸变 / 时间偏移）",
            "executed_action_libero": "partial_provisional"
            "（evidence_level=producer_code_order, verified=false）",
            "executed_action_robotwin": "unavailable",
            "policy": "缺失项记 skipped 并写明原因；不得用 fixture 冒充真实数据合同通过",
            "recorded_at": "2026-10-14",
            "source": "ws-telemetry 远端实测结论（Lead 转发）",
        },
        "readme": {
            "check_status_rule": "组内：任一 failed → failed；passed 与 skipped 并存 → partial；"
            "全 skipped → skipped；否则 passed",
            "skipped_policy": "依赖兄弟工作流产物缺失时记 skipped 并保留同一 CLI 重跑入口；"
            "不伪造输入",
            "scope": "C1/C5/C7 的生产者模块由 ws-mechanism/ws-telemetry/ws-manifest 交付；"
            "C2.5 由 ws-geometry 交付；失败项只报告不代改",
        },
        **results,
    }
    return report


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None
    return getattr(module, "__version__", None)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m owac.evaluation.n04_contracts",
        description="N04 S0 合同审计：逐条输出 passed/failed/skipped 与数值证据。",
    )
    parser.add_argument(
        "--config",
        default="configs/n04/full/s0_data.json",
        help="S0 数据合同配置（默认 configs/n04/full/s0_data.json）",
    )
    parser.add_argument("--output", required=True, help="审计 JSON 输出路径")
    parser.add_argument(
        "--protocol",
        default=DEFAULT_PROTOCOL,
        help="预注册门槛文件（默认 configs/n04/full/protocol.json）",
    )
    parser.add_argument("--live-data-root", default=None, help="新独立扩展根目录；C5 直接重读数组")
    parser.add_argument("--manifest", default=None, help="ws-manifest 产物路径（可选，用于 C7.2）")
    parser.add_argument(
        "--checks", default=None, help="逗号分隔的检查子集，如 C2,C3（C0 始终运行）"
    )
    parser.add_argument("--seed", type=int, default=20261014, help="审计内部随机种子")
    parser.add_argument(
        "--threads", type=int, default=4, help="Torch CPU 线程数（多 agent 共享机器）"
    )
    parser.add_argument(
        "--fail-on-failed",
        action="store_true",
        help="兼容选项；failed 检查始终返回非零退出码",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    config_path = Path(args.config).resolve()
    protocol_path = Path(args.protocol).resolve()
    if not config_path.is_file():
        print(f"[n04_contracts] 配置不存在: {config_path}", file=sys.stderr)
        return 2
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else None
    )
    try:
        torch = _torch()
        if args.threads > 0:
            torch.set_num_threads(int(args.threads))
    except DependencyMissing:
        pass
    selected = (
        [item.strip().upper() for item in args.checks.split(",") if item.strip()]
        if args.checks
        else None
    )
    context = RunContext(
        config=config,
        config_path=config_path,
        protocol=protocol,
        protocol_path=protocol_path,
        seed=int(args.seed),
        manifest_path=args.manifest,
        threads=int(args.threads),
        live_data_root=args.live_data_root,
    )
    report = build_report(context, selected, started)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    print(
        f"[n04_contracts] checks={summary['total']} passed={summary['passed']} "
        f"failed={summary['failed']} skipped={summary['skipped']} "
        f"partial={summary['partial']} -> {output_path}"
    )
    for check in report["checks"]:
        print(f"  {check['id']}: {check['status']}  {check['title']}")
    if summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
