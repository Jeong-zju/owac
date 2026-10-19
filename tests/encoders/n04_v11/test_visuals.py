"""WS-G 验收：可视化的固定显示映射、同帧对齐与“未提供”分支。

本地没有官方数据、也没有 π0.5 权重：这里用合成帧与人造 debug 张量验证像素级对齐、
固定映射不随帧变化、缺失 RGB 走“未提供”分支，以及 CLI 契约。真图由远端 checkpoint +
官方 dev 帧产生（``scripts/n04_full/n04_v11_visuals.sh``），不在本地伪造。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
Image = pytest.importorskip("PIL.Image")

from owac.evaluation.n04_v11_visuals import (  # noqa: E402
    ATTENTION_VMAX,
    INFO_LINE_WIDTH,
    QUALITY_LOG10_RANGE,
    RAW_DISPLAY,
    FrameVisual,
    attention_to_map,
    build_parser,
    compose_frame,
    display_columns,
    display_scale_for,
    expected_png_size,
    fixed_display,
    flow_arrows,
    flow_hsv,
    format_summary_table,
    grid_cell_pixels,
    grid_to_pixel,
    info_lines,
    overlay_scalar,
    points_hw2,
    raw_display_rgb,
    rgb_member_candidates,
    rgb_panel,
    select_clips,
    short_relpath,
    summarize_frames,
    support_scatter,
    write_png,
)
from owac.simulation.raw_preview import raw_display as project_raw_display  # noqa: E402

SIZE_HW = (256, 256)
LATENT_HW = (64, 64)
DENSE_TOKENS = LATENT_HW[0] * LATENT_HW[1]
SUPPORT_TOKENS = 128


# --------------------------------------------------------------------------------------
# RAW 固定映射
# --------------------------------------------------------------------------------------


def test_fixed_display_matches_project_log_compression_convention():
    """与既有 ``simulation.raw_preview.raw_display`` 用同一固定 log 压缩公式（同参数同色）。"""
    counts = np.array([[0, 1, 7, 64, 128, 200, 255]], dtype=np.uint32)
    reference = project_raw_display(counts, white_level=255, log_gain=15)
    ours = np.rint(
        fixed_display(counts.astype(np.float32), white_level=255.0, log_gain=15.0) * 255.0
    ).astype(np.uint8)
    assert np.array_equal(reference, ours)
    assert reference[0, 0] == 0 and reference[0, -1] == 255


def test_raw_display_follows_platform_value_scale():
    """显示量纲必须跟数据模块的 value_scale 走。

    真实回归（2026-10-16）：post-fix 后 LIBERO 输入保持 uint8 计数 [0,255]，
    而显示仍乘 255 → RAW 面板整块被打成全白。
    """
    from owac.raw.n04_v11_data import PLATFORM_SPECS

    expected = 255.0 if PLATFORM_SPECS["libero"].value_scale == "uint8_div255" else 1.0
    assert display_scale_for("libero") == expected
    assert display_scale_for("robotwin") == 1.0

    counts = np.full((2, 2, 3), 128.0, dtype=np.float32)  # 中灰计数
    model_input = counts / 255.0 if expected == 255.0 else counts
    image = raw_display_rgb(model_input, "libero")
    assert 150 < int(image[0, 0, 0]) < 255  # 可见的中灰，不是全白
    assert not np.all(image == 255)


def test_raw_display_is_frame_independent_not_auto_stretched():
    """同一 RAW 值在任何帧里同色；暗帧不会被逐帧拉亮。"""
    value = 0.2  # 以"平台原始单位/255"表示的同一灰度，下面按当前 value_scale 换算成模型输入
    scale = display_scale_for("libero")
    same_value = raw_display_rgb(
        np.full((4, 4, 3), value * 255.0 / scale, dtype=np.float32), "libero"
    )
    mixed = raw_display_rgb(
        np.array(
            [
                [[0.0, 0.0, 0.0], [value, value, value]],
                [[1.0, 1.0, 1.0], [value, value, value]],
            ],
            dtype=np.float32,
        )
        * 255.0
        / scale,
        "libero",
    )
    assert np.array_equal(same_value[0, 0], mixed[0, 1])
    assert np.array_equal(same_value[0, 0], mixed[1, 1])
    assert not np.array_equal(same_value[0, 0], mixed[0, 0])

    dark_frame = raw_display_rgb(
        np.full((4, 4, 3), 0.02 * 255.0 / scale, dtype=np.float32), "libero"
    )
    full_frame = raw_display_rgb(np.full((4, 4, 3), 255.0 / scale, dtype=np.float32), "libero")
    assert int(dark_frame.max()) < 128 < int(full_frame.max())
    assert np.all(dark_frame == dark_frame[0, 0])


def test_robotwin_fixed_mapping_keeps_dim_episodes_visible():
    """RoboTwin HDR float32 用固定 log 压缩：0.0016 量级的深色 episode 也可见。"""
    dark = raw_display_rgb(np.full((2, 2, 3), 0.0016, dtype=np.float32), "robotwin")
    bright = raw_display_rgb(np.full((2, 2, 3), 1.0, dtype=np.float32), "robotwin")
    assert 20 < int(dark[0, 0, 0]) < int(bright[0, 0, 0]) <= 255
    assert np.all(dark == dark[0, 0])


def test_raw_display_clips_above_white_level():
    scale = display_scale_for("libero")
    at_white = raw_display_rgb(np.full((2, 2, 3), 255.0 / scale, dtype=np.float32), "libero")
    above = raw_display_rgb(np.full((2, 2, 3), 4 * 255.0 / scale, dtype=np.float32), "libero")
    assert np.array_equal(at_white, above)  # 只裁剪，不重新拉伸


def test_overlay_near_zero_values_leave_the_base_frame_untouched():
    """固定色标下的近零值不上色：attention 几乎为 0 时不能把整帧染成 colormap 的零值。"""
    base = np.full((8, 8, 3), 200, dtype=np.uint8)
    assert np.array_equal(
        overlay_scalar(base, np.zeros((8, 8), np.float32), vmin=0.0, vmax=3.0), base
    )
    tiny = np.full((8, 8), 1e-4, dtype=np.float32)
    assert np.array_equal(overlay_scalar(base, tiny, vmin=0.0, vmax=3.0), base)
    hot = np.full((8, 8), ATTENTION_VMAX, dtype=np.float32)
    assert not np.array_equal(overlay_scalar(base, hot, vmin=0.0, vmax=3.0), base)


def test_display_constants_are_documented_per_platform():
    for platform in ("libero", "robotwin"):
        spec = RAW_DISPLAY[platform]
        assert spec.white_level > 0 and spec.log_gain > 0
        assert "log1p" in spec.formula() and spec.measured_on


# --------------------------------------------------------------------------------------
# flow：固定编码与像素坐标
# --------------------------------------------------------------------------------------


def test_flow_hsv_encoding_is_fixed_and_angle_ordered():
    right = flow_hsv(np.tile(np.array([[[4.0, 0.0]]], dtype=np.float32), (4, 6, 1)))
    left = flow_hsv(np.tile(np.array([[[-4.0, 0.0]]], dtype=np.float32), (4, 6, 1)))
    down = flow_hsv(np.tile(np.array([[[0.0, 4.0]]], dtype=np.float32), (4, 6, 1)))
    assert right[0, 0, 0] > right[0, 0, 1] + 30 and right[0, 0, 0] > right[0, 0, 2] + 30
    assert left[0, 0, 1] > left[0, 0, 0] or left[0, 0, 2] > left[0, 0, 0]
    assert not np.array_equal(right, left) and not np.array_equal(right, down)
    weak = flow_hsv(np.tile(np.array([[[1.0, 0.0]]], dtype=np.float32), (4, 6, 1)))
    assert int(weak[0, 0, 0]) < int(right[0, 0, 0])  # 模长单调


def test_flow_arrows_stay_in_original_pixel_coordinates():
    height, width = 32, 40
    flow = np.zeros((height, width, 2), dtype=np.float32)
    flow[..., 0], flow[..., 1] = 4.0, -3.0
    x, y, u, v = flow_arrows(flow, stride=8)
    assert x.shape == y.shape == u.shape == v.shape
    # stride//2 起，横纵各自均匀采样（网格是 xs × ys 的外积）
    assert np.array_equal(x[0], np.arange(4, width, 8, dtype=np.float32))
    assert np.array_equal(y[:, 0], np.arange(4, height, 8, dtype=np.float32))
    assert np.array_equal(u, np.full_like(x, 4.0)) and np.array_equal(v, np.full_like(y, -3.0))
    # 箭头头 = 尾 + flow（原始像素，1:1）
    assert np.allclose(x + u, x + 4.0) and np.allclose(y + v, y - 3.0)


# --------------------------------------------------------------------------------------
# 特征网格 -> 像素：同帧对齐
# --------------------------------------------------------------------------------------


def test_grid_to_pixel_and_cell_are_inverse_of_px_to_grid():
    import torch

    from owac.frameworks.n04.correspondence import px_to_grid

    iy, ix = 13, 57
    x, y = grid_to_pixel(iy, ix, SIZE_HW, LATENT_HW)
    assert (x, y) == pytest.approx((229.5, 53.5))
    grid = px_to_grid(torch.tensor([[x, y]], dtype=torch.float32), SIZE_HW).numpy()
    expected = np.array([[(ix + 0.5) * 2 / LATENT_HW[1] - 1, (iy + 0.5) * 2 / LATENT_HW[0] - 1]])
    assert np.allclose(grid, expected, atol=1e-5)
    assert grid_cell_pixels(iy, ix, SIZE_HW, LATENT_HW) == (228, 52, 231, 55)


def test_attention_map_keeps_dense_grid_layout():
    media = DENSE_TOKENS + SUPPORT_TOKENS
    attention = np.zeros((1, 1, media), dtype=np.float32)
    iy, ix = 13, 57
    attention[0, 0, iy * LATENT_HW[1] + ix] = 1.0
    attention[0, 0, DENSE_TOKENS] = 5.0  # 支撑 token 的注意力不能混进像素图
    mapped = attention_to_map(attention, dense_tokens=DENSE_TOKENS, latent_hw=LATENT_HW)
    assert mapped.shape == LATENT_HW
    assert np.unravel_index(int(np.argmax(mapped)), LATENT_HW) == (iy, ix)
    assert float(mapped[iy, ix]) == pytest.approx(media, rel=1e-6)  # a * M：1.0 = 均匀读取


def test_overlay_lands_on_the_same_pixels_as_the_input_frame():
    base = np.zeros((*SIZE_HW, 3), dtype=np.uint8)
    heat = np.zeros(SIZE_HW, dtype=np.float32)
    iy, ix = 13, 57
    x0, y0, x1, y1 = grid_cell_pixels(iy, ix, SIZE_HW, LATENT_HW)
    heat[y0 : y1 + 1, x0 : x1 + 1] = ATTENTION_VMAX
    blended = overlay_scalar(base, heat, vmin=0.0, vmax=ATTENTION_VMAX, cmap="magma", alpha=0.6)
    changed = np.argwhere(np.any(blended != base, axis=-1))
    assert changed.min(axis=0).tolist() == [y0, x0]
    assert changed.max(axis=0).tolist() == [y1, x1]


def test_upsampled_attention_stays_inside_its_grid_cell():
    media = DENSE_TOKENS + SUPPORT_TOKENS
    attention = np.zeros((1, 1, media), dtype=np.float32)
    iy, ix = 40, 3
    attention[0, 0, iy * LATENT_HW[1] + ix] = 1.0
    mapped = attention_to_map(attention, dense_tokens=DENSE_TOKENS, latent_hw=LATENT_HW)
    base = np.zeros((*SIZE_HW, 3), dtype=np.uint8)
    blended = overlay_scalar(base, mapped, vmin=0.0, vmax=ATTENTION_VMAX, cmap="magma", alpha=0.6)
    hot = np.argwhere(blended[..., 0] > 128)
    assert len(hot) > 0
    x0, y0, x1, y1 = grid_cell_pixels(iy, ix, SIZE_HW, LATENT_HW)
    assert x0 - 2 <= hot[:, 1].min() and hot[:, 1].max() <= x1 + 2
    assert y0 - 2 <= hot[:, 0].min() and hot[:, 0].max() <= y1 + 2
    # 固定 [0, 3] 色标下该格点会饱和成一小块平台，但重心必须落在该格点的像素中心附近
    centroid_y, centroid_x = hot.mean(axis=0)
    assert abs(centroid_x - (x0 + x1) / 2) <= 2
    assert abs(centroid_y - (y0 + y1) / 2) <= 2
    center_x, center_y = grid_to_pixel(iy, ix, SIZE_HW, LATENT_HW)
    assert abs(centroid_x - center_x) <= 2 and abs(centroid_y - center_y) <= 2


def test_support_scatter_marks_outside_and_occluded():
    xy = np.array([[10.0, 10.0], [300.0, 10.0], [10.0, -4.0], [20.0, 20.0]], dtype=np.float32)
    valid = np.array([True, True, True, False])
    groups = support_scatter(xy, SIZE_HW, valid=valid)
    assert groups["inside"].tolist() == [True, False, False, True]
    assert groups["valid"].tolist() == [True, False, False, False]
    assert groups["outside"].tolist() == [False, True, True, False]
    assert groups["occluded"].tolist() == [False, False, False, True]
    assert groups["clamped"][1].tolist() == [255.0, 10.0]  # 越界点画在边框上


# --------------------------------------------------------------------------------------
# 一帧 PNG：尺寸、缺失 RGB 分支
# --------------------------------------------------------------------------------------


def _synthetic_visual(*, rgb: np.ndarray | None, size_hw: tuple[int, int] = SIZE_HW) -> FrameVisual:
    height, width = size_hw
    latent = (height // 4, width // 4)
    dense = latent[0] * latent[1]
    media = dense + SUPPORT_TOKENS
    attention = np.zeros((1, 256, media), dtype=np.float32)
    attention[0, :, dense // 2 + latent[1] // 3] = 1.0 / 256.0
    flow = np.zeros((height, width, 2), dtype=np.float32)
    flow[..., 0] = np.linspace(-6.0, 6.0, width, dtype=np.float32)
    flow[..., 1] = 2.0
    rng = np.random.default_rng(0)
    columns, rows = 16, 8
    grid_x, grid_y = np.meshgrid(
        np.linspace(4.0, width - 5.0, columns, dtype=np.float32),
        np.linspace(4.0, height - 5.0, rows, dtype=np.float32),
    )
    xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
    assert xy.shape == (SUPPORT_TOKENS, 2)
    return FrameVisual(
        platform="libero",
        camera="agentview",
        clip_id="clip_01",
        frame_index=42,
        raw_hwc=rng.random((height, width, 3), dtype=np.float32),
        rgb_hwc=rgb,
        rgb_source=None if rgb is None else "agentview_rgb_uint8",
        flow_px=flow,
        flow_valid=True,
        quality_xy=xy,
        quality_scalar=rng.normal(0.0, 1.0, size=xy.shape[0]).astype(np.float32),
        mode_revision=rng.random(xy.shape[0]).astype(np.float32),
        support_xy=xy,
        support_valid=np.ones(xy.shape[0], dtype=bool),
        support_ids=np.arange(xy.shape[0], dtype=np.float32),
        support_age=np.linspace(0, 30, xy.shape[0], dtype=np.float32),
        trajectory=np.stack([xy, xy + 1.0, xy + 2.0], axis=0),
        attention_map=attention_to_map(attention, dense_tokens=dense, latent_hw=latent),
        attention_dense_mass=0.9,
        notes=["unit test"],
    )


def test_render_without_cjk_font_draws_no_missing_glyphs(tmp_path):
    """远端没有中文字体（cjk=False）：图上不能出现缺字豆腐块（matplotlib 会发缺字警告）。"""
    import warnings

    figure, _ = compose_frame(
        _synthetic_visual(rgb=np.full((*SIZE_HW, 3), 90, np.uint8)), dpi=50, cjk=False
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_png(figure, tmp_path / "ascii_only.png", dpi=50)
        missing = [str(item.message) for item in caught if "missing from font" in str(item.message)]
        assert not missing, missing[:3]
    finally:
        matplotlib.pyplot.close(figure)


def test_compose_frame_without_rgb_reports_not_provided(tmp_path):
    visual = _synthetic_visual(rgb=None)
    figure, meta = compose_frame(visual, dpi=60, cjk=False)
    try:
        assert meta["rgb"]["available"] is False
        assert "未提供" in meta["rgb"]["label"]
        assert meta["camera"] == "agentview" and meta["frame_index"] == 42
        assert meta["raw"]["mapping"] == RAW_DISPLAY["libero"].formula()
        assert meta["quality"]["log10_range"] == list(QUALITY_LOG10_RANGE)
        assert meta["attention"]["range"] == [0.0, ATTENTION_VMAX]
        path = write_png(figure, tmp_path / "no_rgb.png", dpi=60)
        assert path.exists()
        with Image.open(path) as image:
            assert image.size == expected_png_size(60)
    finally:
        matplotlib.pyplot.close(figure)


def test_points_hw2_flattens_the_debug_batch_dimension():
    """``StepOutput.debug`` 的支撑坐标是 ``[1,N,2]``：点数必须按 N 取，不能把 batch 当点。"""
    batched = np.arange(5 * 2, dtype=np.float32).reshape(1, 5, 2)
    assert points_hw2(batched).shape == (5, 2)
    assert points_hw2(np.zeros((5, 2), dtype=np.float32)).shape == (5, 2)


def test_compose_frame_accepts_batched_debug_shapes(tmp_path):
    """用真实 debug 形状（带 batch 维）渲染：质量／模式面板要按 N 个点画。"""
    visual = _synthetic_visual(rgb=None)
    visual.flow_px = np.transpose(np.asarray(visual.flow_px), (2, 0, 1))[None]  # [1,2,H,W]
    visual.quality_xy = np.asarray(visual.quality_xy)[None]  # [1,N,2]
    visual.support_xy = np.asarray(visual.support_xy)[None]
    visual.support_valid = np.asarray(visual.support_valid)[None]
    visual.support_ids = np.asarray(visual.support_ids)[None]
    visual.support_age = np.zeros((1,), dtype=np.float32)  # 编码器里 age 是 [B]
    figure, meta = compose_frame(visual, dpi=50, cjk=False)
    try:
        assert meta["quality"]["n"] == SUPPORT_TOKENS
        assert meta["mode"]["n"] == SUPPORT_TOKENS
        assert meta["support"]["n"] == SUPPORT_TOKENS
        assert meta["flow"]["available"] is True
        assert write_png(figure, tmp_path / "batched.png", dpi=50).exists()
    finally:
        matplotlib.pyplot.close(figure)


def test_compose_frame_with_rgb_reports_available(tmp_path):
    rgb = np.full((*SIZE_HW, 3), 128, dtype=np.uint8)
    figure, meta = compose_frame(_synthetic_visual(rgb=rgb), dpi=60, cjk=False)
    try:
        assert meta["rgb"]["available"] is True
        assert meta["rgb"]["source"] == "agentview_rgb_uint8"
        path = write_png(figure, tmp_path / "with_rgb.png", dpi=60)
        with Image.open(path) as image:
            assert image.size == expected_png_size(60)
    finally:
        matplotlib.pyplot.close(figure)


def test_compose_frame_without_support_state_still_renders(tmp_path):
    visual = _synthetic_visual(rgb=None)
    visual.quality_xy = None
    visual.quality_scalar = None
    visual.mode_revision = None
    visual.support_xy = None
    visual.trajectory = None
    visual.attention_map = None
    figure, meta = compose_frame(visual, dpi=60, cjk=False)
    try:
        assert meta["quality"]["available"] is False
        assert meta["mode"]["available"] is False
        assert meta["support"]["available"] is False
        assert meta["attention"]["available"] is False
        path = write_png(figure, tmp_path / "no_state.png", dpi=60)
        assert path.exists()
    finally:
        matplotlib.pyplot.close(figure)


def test_info_panel_lines_fit_the_panel_width():
    """信息面板每行必须放得下（否则 PNG 上会被面板裁掉看不见）。"""
    long_entry = "libero_object_no_noops_1.0.0_lerobot/episode_000116/variant_00.npz"
    for zh in (False, True):
        for rgb in (None, np.zeros((*SIZE_HW, 3), dtype=np.uint8)):
            visual = _synthetic_visual(rgb=rgb)
            visual.rgb_file = short_relpath(long_entry)
            for line in info_lines(visual, SIZE_HW, zh=zh):
                assert display_columns(line) <= INFO_LINE_WIDTH, (zh, line)
    assert len(short_relpath(long_entry)) < len(long_entry)
    assert short_relpath("a/b.npz") == "a/b.npz"
    assert display_columns("中文ab") == 6


def test_rgb_panel_never_fabricates_from_raw():
    assert rgb_panel(None, "agentview") == (None, "同源 RGB（agentview）：未提供")
    image, label = rgb_panel(np.zeros((4, 4, 3), dtype=np.uint8), "head")
    assert image is not None and image.dtype == np.uint8 and image.shape == (4, 4, 3)
    assert "head" in label


def test_rgb_member_candidates_cover_both_platforms():
    assert rgb_member_candidates("agentview_raw_uint8")[0] == "agentview_rgb_uint8"
    assert rgb_member_candidates("wrist_raw_uint8")[0] == "wrist_rgb_uint8"
    assert rgb_member_candidates("head_camera_raw_float32")[0] == "head_camera_rgb_uint8"
    assert rgb_member_candidates("left_camera_raw_float32")[0] == "left_camera_rgb_uint8"


# --------------------------------------------------------------------------------------
# 固定片段挑选与 CLI 契约
# --------------------------------------------------------------------------------------


class _Entry:
    def __init__(self, relpath: str, frames: int) -> None:
        self.relpath = relpath
        self.frames = frames


def test_select_clips_is_deterministic_and_filters_short_entries():
    entries = [_Entry(f"episode_{index:03d}/variant_00.npz", 40 + index) for index in range(12)]
    counts = {entry.relpath: entry.frames for entry in entries}
    first = select_clips(
        entries, frames=8, clips=3, seed=101, frame_count=lambda e: counts[e.relpath]
    )
    second = select_clips(
        entries, frames=8, clips=3, seed=101, frame_count=lambda e: counts[e.relpath]
    )
    assert [(entry.relpath, start) for entry, start in first] == [
        (entry.relpath, start) for entry, start in second
    ]
    assert len(first) == 3
    for entry, start in first:
        assert 0 <= start <= counts[entry.relpath] - 8
    short = [_Entry("short.npz", 4), *entries]
    chosen = select_clips(short, frames=8, clips=12, seed=101, frame_count=lambda e: e.frames)
    assert all(entry.relpath != "short.npz" for entry, _ in chosen)
    assert len(chosen) == 12
    other_seed = select_clips(
        entries, frames=8, clips=3, seed=102, frame_count=lambda e: counts[e.relpath]
    )
    assert {(entry.relpath, start) for entry, start in other_seed} != {
        (entry.relpath, start) for entry, start in first
    }


def test_encoder_is_built_from_the_checkpoint_model_config_priority():
    """源码检查：checkpoint 自带 model 段优先，避免配置漂移导致 state_dict 对不上。"""
    from pathlib import Path as _Path

    source = _Path("src/owac/evaluation/n04_v11_visuals.py").read_text()
    assert 'checkpoint_model = (payload.get("config") or {}).get("model")' in source
    assert "build_v11_encoder(model_cfg, platform=platform, arm=arm)" in source
    assert 'build_v11_encoder(config["model"], platform=platform, arm=arm)' not in source


def test_summarize_frames_groups_by_step_and_reports_ranges(tmp_path):
    """--summary-only 的汇总：按 step 分组，给出跨帧 [min,max]，首帧无 flow 的样本不计入。"""
    root = tmp_path / "libero-D-seed101"
    for clip, frames in (("clip_01", [10, 11]),):
        for index, frame in enumerate(frames):
            meta = {
                "checkpoint_step": 100,
                "clip_id": clip,
                "frame_index": frame,
                "flow": {
                    "valid": index > 0,
                    "median_magnitude_px": 2.0 + index,
                    "p95_magnitude_px": 3.0 + index,
                },
                "quality": {"observed_log10_min": -0.01 * frame, "observed_log10_max": -0.001},
                "mode": {"mean_revision": 0.1 + index, "frac_above_half": 0.0},
                "attention": {"dense_mass": 0.9 + 0.01 * index},
                "support": {
                    "inside_valid": 128 - frame,
                    "occluded": frame,
                    "outside": 0,
                    "id_max": 130 + frame,
                },
            }
            (root / clip).mkdir(parents=True, exist_ok=True)
            (root / clip / f"frame_{frame:07d}.json").write_text(json.dumps(meta))
    later = root / "libero-D-seed101-step0500" / "clip_02"
    later.mkdir(parents=True)
    (later / "frame_0000050.json").write_text(
        json.dumps(
            {
                "checkpoint_step": 500,
                "clip_id": "clip_02",
                "frame_index": 50,
                "flow": {"valid": True, "median_magnitude_px": 9.0, "p95_magnitude_px": 12.0},
                "quality": {"observed_log10_min": -0.03, "observed_log10_max": -0.02},
                "mode": {"mean_revision": 0.3, "frac_above_half": 0.0},
                "attention": {"dense_mass": 0.8},
                "support": {
                    "inside_valid": 120,
                    "occluded": 8,
                    "outside": 0,
                    "id_max": 180,
                },
            }
        )
    )
    summary = summarize_frames(root)
    assert summary["steps"] == [100, 500]
    groups = {record["step"]: record for record in summary["groups"]}
    entry = groups[100]
    assert entry["dir"] == "libero-D-seed101"
    assert entry["frames"] == 2
    assert entry["frames_with_flow"] == 1  # clip_01 的第 0 帧没有上一帧
    assert entry["flow_median_px"] == [3.0, 3.0]
    assert entry["support_id_max"] == [140.0, 141.0]
    assert entry["births_per_frame_mean"] is None  # 老 meta 没有 births 字段
    assert groups[500]["flow_median_px"] == [9.0, 9.0]
    table = format_summary_table(summary)
    assert "libero-D-seed101" in table and "step 500" in table
    # 新 checkpoint 的图放在子目录（--tag）也不能漏掉，且不同 step 自动分列
    tagged = root / "libero-D-seed101-step0500" / "clip_01"
    tagged.mkdir(parents=True)
    (tagged / "frame_0000001.json").write_text(
        json.dumps(
            {
                "checkpoint_step": 500,
                "clip_id": "clip_01",
                "frame_index": 1,
                "flow": {"valid": True, "median_magnitude_px": 1.0, "p95_magnitude_px": 2.0},
                "quality": {"observed_log10_min": -0.02, "observed_log10_max": -0.01},
                "mode": {"mean_revision": 0.2, "frac_above_half": 0.1},
                "attention": {"dense_mass": 0.5},
                "support": {"inside_valid": 100, "occluded": 28, "outside": 0, "id_max": 200},
            }
        )
    )
    merged = summarize_frames(root)
    assert merged["steps"] == [100, 500]
    tagged_group = next(r for r in merged["groups"] if r["dir"].endswith("step0500"))
    assert tagged_group["support_id_max"] == [180.0, 200.0]
    assert tagged_group["tag"] == "step0500"
    # 同一 step 的不同目录必须分列，不能合并
    merged_dirs = {r["dir"] for r in merged["groups"]}
    assert merged_dirs == {"libero-D-seed101", "libero-D-seed101-step0500"}
    assert next(r for r in merged["groups"] if r["dir"].endswith("seed101"))["tag"] is None
    other = summarize_frames(tmp_path / "missing")
    assert other["steps"] == []
    assert "没有可汇总" in format_summary_table(other)


def test_thread_limit_is_opt_in_and_recorded():
    """--threads 只在显式给正数时生效（训练同机时避免抢 CPU），0/负数不动全局设置。"""
    from owac.evaluation.n04_v11_visuals import apply_thread_limit

    assert apply_thread_limit(0) == 0
    assert apply_thread_limit(-4) == 0
    assert apply_thread_limit(2) == 2
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            "c.json",
            "--output",
            "out",
            "--checkpoint",
            "ckpt.pt",
            "--threads",
            "3",
        ]
    )
    assert args.threads == 3
    assert parser.parse_args(["--config", "c", "--output", "o", "--checkpoint", "k"]).threads == 0


def test_cli_exposes_required_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/n04/v1_1/runs/libero_D.json",
            "--data-root",
            "/data",
            "--output",
            "/run/artifacts",
            "--checkpoint",
            "/run/checkpoint-step0001000.pt",
            "--device",
            "cpu",
            "--platform",
            "libero",
            "--clips",
            "3",
            "--frames",
            "8",
        ]
    )
    assert args.platform == "libero" and args.clips == 3 and args.frames == 8
    assert args.device == "cpu" and args.output == "/run/artifacts"
    assert args.split == "dev"
    assert args.tag is None and args.summary_only is False


def test_compose_frame_meta_reports_same_frame_index_for_all_panels():
    figure, meta = compose_frame(_synthetic_visual(rgb=None), dpi=50, cjk=False)
    try:
        assert meta["frame_index"] == 42
        assert meta["size_hw"] == list(SIZE_HW)
        assert "原始像素" in meta["same_coordinates"]
        assert meta["flow"]["valid"] is True and meta["flow"]["arrow_stride"] > 0
        assert meta["support"]["n"] == SUPPORT_TOKENS
    finally:
        matplotlib.pyplot.close(figure)
