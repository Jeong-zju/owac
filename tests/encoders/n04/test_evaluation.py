"""N04 机制评测指标测试。

重点回归一个曾经的真实缺陷：把变化幅值**先压成标量**再做掩码加权时，"目标处"与
"目标外"恒等（曾得到 change_contrast = 4e-12），指标完全失去分辨力。
"""

from __future__ import annotations

import math

import pytest
import torch

from owac.evaluation.n04 import (
    MECHANISM_CONDITIONS,
    _normalised_contrast,
    evaluate_condition,
    run_mechanism_suite,
)
from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig
from owac.frameworks.n04.model import N04Encoder
from owac.frameworks.n04.synthetic import SceneConfig


def _encoder() -> N04Encoder:
    cfg = N04Config(
        input_mode="rawvla_rgb3",
        measurement_size=(32, 32),
        channels=3,
        camera_ids=("head",),
        geometry=N04GeometryConfig(feature_size=(32, 32), pixel_stride=1.0, max_flow=4.0),
        model=N04ModelConfig(
            name="n04-eval-test",
            width=8,
            blocks=1,
            scales=1,
            scale_strides=(1,),
            coarse_stride=4,
            hidden=16,
            camera_embedding=4,
            readout_tokens=8,
            readout_dim=16,
        ),
    )
    torch.manual_seed(0)
    return N04Encoder(cfg).eval()


def _scene() -> SceneConfig:
    return SceneConfig(
        height=32,
        width=32,
        input_mode="rawvla_rgb3",
        camera_translation=(1.2, 1.2),
        camera_rotation=(0.0, 0.4),
        object_velocity=(0.8, 1.2),
        illumination=(0.7, 1.3),
        noise_sigma=0.01,
        seed=0,
    )


def test_mechanism_conditions_cover_required_matrix() -> None:
    required = {
        "camera_only",
        "object_only",
        "both",
        "illumination",
        "same_direction",
        "occlusion",
        "calibration_error",
        "action_delay",
        "exposure_step",
    }
    assert required.issubset(set(MECHANISM_CONDITIONS))


def test_change_metric_is_spatially_resolved() -> None:
    """目标处与目标外的变化能量必须能不同：标量幅值会让它们恒等。"""
    with torch.no_grad():
        result = evaluate_condition(
            _encoder(), condition="object_only", scene=_scene(), length=6, batch=1, seed=0
        )
    assert result.change_at_object == result.change_at_object  # 非 NaN
    assert result.change_off_object == result.change_off_object
    assert abs(result.change_at_object - result.change_off_object) > 1e-9
    assert result.extra["change_contrast"] == pytest.approx(
        result.change_at_object - result.change_off_object, rel=1e-6, abs=1e-12
    )


def test_mechanism_report_has_counterfactuals_and_finite_metrics() -> None:
    with torch.no_grad():
        report = run_mechanism_suite(
            _encoder(),
            base_scene=_scene(),
            conditions=("camera_only", "object_only", "illumination"),
            length=5,
            batch=1,
            seed=0,
        )
    names = [c.condition for c in report.conditions]
    assert names == ["camera_only", "object_only", "illumination"]
    for condition in report.conditions:
        assert math.isfinite(condition.location_error)
        assert condition.latency_p95_ms >= condition.latency_p50_ms >= 0.0
        assert condition.state_bytes > 0
        assert 0.0 <= condition.mask_coverage <= 1.0
    #: 成对反例必须同时给出两项，且比值有定义。
    pairs = {tuple(item["pair"]) for item in report.counterfactuals}
    assert ("camera_only", "object_only") in pairs
    assert ("object_only", "illumination") in pairs
    for item in report.counterfactuals:
        assert math.isfinite(item["ratio"])
    assert "camera_vs_object_change" in report.contrast
    assert "illumination_false_alarm" in report.contrast
    assert report.notes and "合成" in report.notes[0]


def test_normalised_contrast_handles_degenerate_inputs() -> None:
    assert _normalised_contrast([], []) != _normalised_contrast([], [])  # NaN
    assert _normalised_contrast([2.0], [0.0]) == pytest.approx(1.0)
    assert _normalised_contrast([1.0], [1.0]) == pytest.approx(0.0)
    assert _normalised_contrast([1.0], [3.0]) == pytest.approx(-0.5)
