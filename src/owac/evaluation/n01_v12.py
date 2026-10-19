"""N01 v1.2 P3 evaluation: fixed validation, source intervention and cases.

Three entry points, each producing evidence that can be checked against the run
that produced it:

``validate``   the fixed-order validation subset with a fixed RNG per sample, so
               two validation points are comparable to each other
``intervene``  a *content-path* source deletion: one source is removed, the
               fusion is recomputed from the remaining legal measurements, and
               the token and flow-target differences are reported
``cases``      the synchronisation figures: same-condition RAW and frozen-ISP RGB,
               their local difference, the channel-mean fusion weight overlaid on
               the original frame, and the support region state

Whole-source deletion rebuilds all reachable motion/noise/support/content decisions
from the already bounded surviving window. Fixed-decision and validity interventions
are separate modes. Predicted actions use fixed denoising noise; no simulator action
or learned improvement is inferred from this read-only command.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from owac.frameworks.n01_full.model import N01V12Config, build_model

__all__ = ["main", "validate", "intervene", "cases"]

# Four-state colours shared by every support figure, in the order of SUPPORT_*.
SUPPORT_COLOURS = {
    0: (120, 120, 120),  # invalid
    1: (40, 150, 70),  # accepted
    2: (235, 170, 40),  # deferred
    3: (200, 45, 45),  # retracted
}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _datasets(args: argparse.Namespace, config: N01V12Config, region: str):
    from owac.raw.n01_full_manifest import dataset_from_protocol

    repo_root = Path(args.repo_root)
    protocol_path = (
        Path(args.protocol)
        if args.protocol
        else repo_root / "configs/encoders/n01_full/protocol_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    return dataset_from_protocol(
        protocol,
        config.platform,
        region,
        repo_root=str(repo_root),
        window=config.window_frames,
        **({"observation_stride": config.observation.history_stride} if config.observation else {}),
    )


def _load_config(path: str | Path) -> N01V12Config:
    return N01V12Config.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _backend(args: argparse.Namespace, config: N01V12Config, model: Any, device: Any):
    """The frozen π0.5 policy plus this arm's encoder and bridge.

    The bridge is built around *this* model's encoder and postprocessor, so the
    tokens under test are the ones the arm actually produces.
    """

    from owac.backends.rawvla_features import ACTION_PROTOCOLS, RawVLAFeaturePolicy
    from owac.training.n01_action import load_backend

    deploy_root = Path(args.deploy_root)
    converted = deploy_root / "source" / "openpi_converted_protocol"
    if config.platform == "libero":
        weights = converted / "pi05_libero_starvla" / "bfloat16" / "model.safetensors"
        suite = "libero"
    else:
        weights = deploy_root / "checkpoints" / "pi05_robotwin" / "model.safetensors"
        suite = "robotwin"
    tokenizer = converted / "paligemma_tokenizer.model"
    policy, audit = load_backend(suite, weights, tokenizer, device)
    bridge = RawVLAFeaturePolicy(
        policy,
        encoder=model.encoder,
        postprocessor=model.bridge,
        action_protocol=ACTION_PROTOCOLS[suite],
    )
    bridge.freeze_backend_parameters()
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is None and not getattr(args, "allow_untrained", False):
        raise ValueError("supply --checkpoint, or explicitly select --allow-untrained diagnostics")
    if checkpoint is not None:
        from owac.runtime.n01_v12_service import load_arm_checkpoint

        restored = load_arm_checkpoint(model, Path(checkpoint), device, policy=policy)
        audit = {**audit, "n01_checkpoint": restored}
    else:
        audit = {
            **audit,
            "n01_checkpoint": {"trained": False, "note": "explicit untrained diagnostic"},
        }
    return policy, bridge, audit


# A CJK font is needed for the Chinese figure labels.  The server has none by
# default, so one is fetched once into the deployment assets directory; when it is
# missing the figures fall back to an ASCII label set instead of rendering tofu
# boxes, and the Chinese wording still reaches the reader through the case index.
CJK_FONT_CANDIDATES = (
    os.environ.get("OWAC_CJK_FONT", ""),
    "/workspace/owac/rawvla/assets/fonts/NotoSansSC-Regular.otf",
    str(Path.home() / ".fonts/NotoSansSC-Regular.otf"),
)
_CJK_FONT_PATH: str | None = None
_CJK_FONT_RESOLVED = False


def _cjk_font() -> str | None:
    """Register a CJK font with matplotlib once, or report that there is none."""

    global _CJK_FONT_PATH, _CJK_FONT_RESOLVED
    if _CJK_FONT_RESOLVED:
        return _CJK_FONT_PATH
    _CJK_FONT_RESOLVED = True
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    for candidate in CJK_FONT_CANDIDATES:
        if candidate and Path(candidate).is_file():
            try:
                font_manager.fontManager.addfont(candidate)
                name = font_manager.FontProperties(fname=candidate).get_name()
                matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                _CJK_FONT_PATH = candidate
                return _CJK_FONT_PATH
            except Exception:  # pragma: no cover - a broken font must not kill the run
                continue
    return None


def _t(chinese: str, ascii_fallback: str) -> str:
    """Chinese when a CJK font is active, plain ASCII otherwise."""

    return chinese if _cjk_font() else ascii_fallback


def _unique_frames(frame_ids: Any, member: int = 0) -> int:
    """Distinct frame identities in one collated window.

    ``batch["windows"][camera]`` is a ``MeasurementWindowBatch``, which has no
    ``unique_frame_count``; the per-sample method only exists on the uncollated
    ``MeasurementWindow``.  Counting here keeps the same meaning: repeated frame
    identities are one measurement, which is exactly why a 32-slot window can hold
    far fewer real frames.
    """

    ids = frame_ids[member] if getattr(frame_ids, "ndim", 0) == 2 else frame_ids
    return int(len({int(value) for value in ids.reshape(-1).tolist()}))


def _run_window(model: Any, batch: Any, device: Any, *, no_grad: bool, tbptt: int | None = None):
    import torch

    from owac.training.n01_v12 import encode_windows_v12

    with torch.no_grad() if no_grad else torch.enable_grad():
        return encode_windows_v12(model, batch, device, tbptt=tbptt)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """Fixed-order validation subset with a per-sample fixed RNG."""

    import torch

    from owac.raw.rawvla_sequence import collate_sequence_samples
    from owac.training.n01_action import action_masked_loss, observation_from_batch

    config = _load_config(args.config)
    device = torch.device(args.device)
    torch.manual_seed(config.seed)
    model = build_model(config, device=device)
    policy, bridge, audit = _backend(args, config, model, device)
    dataset = _datasets(args, config, "validation")
    count = min(int(args.samples), len(dataset))
    losses = []
    from owac.training.n01_v12_sampling import validation_indices

    for index in validation_indices(len(dataset), count, config.seed):
        torch.manual_seed(config.seed * 7919 + index)
        sample = dataset[index]
        batch = collate_sequence_samples([sample])
        if not bool(batch["action_mask"].any()):
            continue
        tokens, _ = _run_window(model, batch, device, no_grad=True)
        observation = observation_from_batch(policy, batch, device)
        actions = batch["action"].to(device)
        actions = torch.nn.functional.pad(
            actions, (0, max(int(policy.action_dim) - int(actions.shape[-1]), 0))
        )
        result = bridge.forward_from_observation_with_features(
            observation, tokens, actions, return_debug=True
        )
        loss = action_masked_loss(
            result["v_t"],
            result["u_t"],
            batch["action_mask"].to(device),
            dataset.protocol.action_dim,
        )
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("validation subset has no genuine action targets")
    receipt = {
        "stage": "P3",
        "command": "validate",
        "model_audit": audit,
        "arm": config.arm,
        "platform": config.platform,
        "seed": config.seed,
        "validation_region_samples": len(dataset),
        "sampled": count,
        "valid_action_samples": len(losses),
        "fixed_order": True,
        "fixed_flow_rng": f"torch.manual_seed(seed*7919 + index) = {config.seed}*7919 + index",
        "mean_action_loss": sum(losses) / max(len(losses), 1),
        "per_sample_action_loss": losses,
        "config": config.to_dict(),
    }
    if args.output:
        _write_json_atomic(Path(args.output), receipt)
    return receipt


def counterfactual_content_replay(
    model: Any, batch: Any, device: Any, camera: str, deleted_sources: set[int]
) -> dict[str, Any]:
    """Whole-source deletion on the neural content path.

    Establish the original finite window, remove whole sources, then rebuild every
    dependent decision. A validity mask intervention retains source identity and
    is reported separately from this operation.

    No older frame is pulled in to refill the window: the window simply holds fewer
    unique sources, exactly as the contract requires.
    """

    from owac.core.encoder_contract import ComputeBudget, MeasurementFrame

    window = batch["windows"][camera]
    values = window.values.to(device)
    size, length, _, height, width = values.shape
    capacity = model.config.window_frames
    state = model.init_state(size, {"camera_ids": [camera], "shape": (height, width)}, device)
    # Establish the original window before removing anything. Skipping sources during
    # ingestion could pull older observations back into a full-capacity window.
    for index in range(length):
        frame = MeasurementFrame(
            values[:, index],
            valid_mask=window.valid_mask[:, index].to(device),
            bad_mask=window.bad_mask[:, index].to(device),
            saturated_mask=window.saturated_mask[:, index].to(device),
            frame_ids=window.frame_ids[:, index].to(device),
            camera_id=camera,
            timestamp=window.timestamp[:, index].to(device),
            time_unit=window.time_unit,
            new_measurement=window.new_measurement[:, index].to(device),
            sample_xy=window.sample_xy,
        )
        output = model.step(
            frame, state=state, budget=ComputeBudget(max_updates=0, max_history=capacity)
        )
        state = output.state
    deleted_ids = set()
    for slot in deleted_sources:
        if not 0 <= slot < length:
            raise IndexError("deleted slot outside provided source window")
        deleted_ids.update(window.frame_ids[:, slot].tolist())
    output = model.delete_sources(state, camera, deleted_ids)
    if not bool(output.state.cameras[camera].observed.any()):
        raise ValueError("every source was deleted; there is nothing to read out")
    tokens = model.readout(output)
    return {
        "tokens": tokens,
        "diagnostics": output.diagnostics,
        "arrivals": length - len(deleted_sources),
        "buffer_unique_frames": int(output.state.cameras[camera].filled.max()),
        "requested_window": capacity,
    }


def intervene(args: argparse.Namespace) -> dict[str, Any]:
    """Delete one source and measure the content-path and flow-target difference."""

    import torch

    from owac.raw.rawvla_sequence import collate_sequence_samples
    from owac.training.n01_action import observation_from_batch

    config = _load_config(args.config)
    device = torch.device(args.device)
    torch.manual_seed(config.seed)
    model = build_model(config, device=device)
    policy, bridge, audit = _backend(args, config, model, device)
    dataset = _datasets(args, config, args.region)
    sample = dataset[int(args.index)]
    batch = collate_sequence_samples([sample])
    camera = args.camera or sorted(batch["windows"])[0]
    slot = int(args.slot)

    mode = str(args.mode)
    if mode not in {"mask", "fixed", "delete"}:
        raise ValueError("mode must be mask, fixed or delete")
    window = batch["windows"][camera]
    if not 0 <= slot < window.frame_ids.shape[1]:
        raise IndexError("source slot outside the supplied window")
    deleted_ids = set(window.frame_ids[:, slot].tolist())

    def flow_target(changed):
        from dataclasses import replace

        from owac.training.n01_v12 import encode_windows_v12

        torch.manual_seed(config.seed * 104729)
        current = batch
        if changed and mode == "mask":
            valid = window.valid_mask.clone()
            removed = (window.frame_ids[..., None] == torch.tensor(sorted(deleted_ids))).any(-1)
            valid &= ~removed[..., None, None, None]
            current = {
                **batch,
                "windows": {**batch["windows"], camera: replace(window, valid_mask=valid)},
            }

        def transform(name, output):
            if name != camera or not changed or mode == "mask":
                return output
            if mode == "fixed":
                return model.fixed_decision_readout(output.state, camera, deleted_ids)
            return model.delete_sources(output.state, camera, deleted_ids)

        with torch.no_grad():
            tokens, diagnostics = encode_windows_v12(model, current, device, transform=transform)
            if not bool(tokens.mask.any()):
                raise ValueError("intervention removed every valid observation")
            observation = observation_from_batch(policy, current, device)
            actions = current["action"].to(device)
            actions = torch.nn.functional.pad(
                actions, (0, max(int(policy.action_dim) - int(actions.shape[-1]), 0))
            )
            torch.manual_seed(config.seed * 104729)
            result = bridge.forward_from_observation_with_features(
                observation, tokens, actions, return_debug=True
            )
            torch.manual_seed(config.seed * 104729)
            predicted = bridge.sample_actions_with_features(observation, tokens, num_steps=10)
        return tokens.values, result["v_t"], result["u_t"], predicted, diagnostics

    tokens_before, v_before, u_before, a_before, _ = flow_target(False)
    tokens_after, v_after, u_after, a_after, diagnostics = flow_target(True)

    receipt = {
        "stage": "P3",
        "command": "intervene",
        "arm": config.arm,
        "platform": config.platform,
        "region": args.region,
        "sample_index": int(args.index),
        "camera": camera,
        "deleted_slot": slot,
        "deleted_frame_id": int(batch["windows"][camera].frame_ids[:, slot].reshape(-1)[0]),
        "mode": mode,
        "recompute_scope": {
            "delete": "all dependencies rebuilt from surviving retained window; no backfill",
            "fixed": "direct content recomputed with frozen motion/noise/support decisions",
            "mask": "validity mask intervention; source identity remains present",
        }[mode],
        "action_scope": "10-step denoised normalized prediction; not executed in simulator",
        "predicted_action_abs_diff_mean": float((a_after - a_before).abs().mean()),
        "predicted_action_abs_diff_max": float((a_after - a_before).abs().max()),
        "replay_diagnostics": diagnostics,
        "model_audit": audit,
        "fixed_rng": f"torch.manual_seed({config.seed}*104729) before each call",
        "token_abs_diff_mean": float((tokens_after - tokens_before).abs().mean()),
        "token_abs_diff_max": float((tokens_after - tokens_before).abs().max()),
        "tokens_changed": not bool(torch.equal(tokens_after, tokens_before)),
        "flow_target_v_abs_diff_mean": float((v_after - v_before).abs().mean()),
        "flow_target_v_abs_diff_max": float((v_after - v_before).abs().max()),
        "flow_target_u_abs_diff_mean": float((u_after - u_before).abs().mean()),
        "config": config.to_dict(),
    }
    if args.output:
        _write_json_atomic(Path(args.output), receipt)
    return receipt


# ---------------------------------------------------------------------------
# Synchronisation figures
# ---------------------------------------------------------------------------


def _to_view8(values: Any) -> Any:
    """The stored uint8 RAW view, i.e. what the cache actually holds."""

    import numpy as np

    array = values.detach().float().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        array = array.transpose(1, 2, 0)
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def _figure_pair(raw: Any, rgb: Any, out: Path, title: str, note: str) -> None:
    """RAW view, frozen-ISP RGB, and their absolute difference, with units."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    diff = np.abs(rgb.astype(np.float32) - raw.astype(np.float32)).mean(-1)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=110)
    axes[0].imshow(raw)
    axes[0].set_title(
        _t(
            "同源 RAW（存储视图 uint8 = rint(x·255)）",
            "Same-source RAW (stored view uint8 = rint(x*255))",
        ),
        fontsize=10,
    )
    axes[1].imshow(rgb)
    axes[1].set_title(_t("冻结 Default ISP 输出 RGB", "Frozen Default ISP output RGB"), fontsize=10)
    image = axes[2].imshow(diff, cmap="magma")
    axes[2].set_title(
        _t("|RAW − RGB| 通道均值（0–255 级）", "|RAW - RGB| channel mean (0-255 levels)"),
        fontsize=10,
    )
    figure.colorbar(
        image, ax=axes[2], fraction=0.046, label=_t("8-bit 级差", "8-bit level difference")
    )
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.text(0.01, 0.01, note, fontsize=8, color="#333333")
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out)
    plt.close(figure)


def _figure_detail(
    raw: Any, rgb: Any, out: Path, box: tuple[int, int, int, int], title: str
) -> None:
    """Nearest-neighbour zoom of one local box, so pixel-level loss is visible."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    top, left, height, width = box
    raw_box = raw[top : top + height, left : left + width]
    rgb_box = rgb[top : top + height, left : left + width]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.4), dpi=110)
    axes[0].imshow(raw_box, interpolation="nearest")
    axes[0].set_title(
        _t(
            f"RAW 局部 {width}×{height}（最近邻放大）",
            f"RAW local {width}x{height} (nearest-neighbour)",
        ),
        fontsize=10,
    )
    axes[1].imshow(rgb_box, interpolation="nearest")
    axes[1].set_title(
        _t("ISP-RGB 同一局部（最近邻放大）", "ISP-RGB same local box (nearest-neighbour)"),
        fontsize=10,
    )
    histogram = axes[2]
    bins = np.arange(0, 257, 4)
    histogram.hist(raw_box.reshape(-1), bins=bins, alpha=0.7, label="RAW", color="#1f77b4")
    histogram.hist(rgb_box.reshape(-1), bins=bins, alpha=0.6, label="ISP-RGB", color="#d62728")
    histogram.set_xlabel(_t("8-bit 值", "8-bit value"))
    histogram.set_ylabel(_t("像素数", "pixel count"))
    histogram.set_title(
        _t("局部直方图（对比度/量化）", "Local histogram (contrast / quantisation)"), fontsize=10
    )
    histogram.legend(fontsize=8)
    for axis in axes[:2]:
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out)
    plt.close(figure)


def _figure_fusion(raw: Any, weights: Any, frame_ids: list[int], out: Path, title: str) -> None:
    """Channel-mean fusion weight overlaid on the original frame, one shared scale."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weight = weights.detach().float().cpu().numpy()
    if weight.ndim == 5:
        weight = weight[0]
    mean_weight = weight.mean(1)  # channel mean, as required by the contract
    sources = mean_weight.shape[0]
    figure, axes = plt.subplots(1, sources + 1, figsize=(3.1 * (sources + 1), 3.9), dpi=110)
    if sources + 1 == 1:
        axes = [axes]
    images = []
    for index in range(sources):
        axis = axes[index]
        axis.imshow(raw)
        image = axis.imshow(mean_weight[index], cmap="inferno", alpha=0.55)
        images.append(image)
        label = (
            _t("参考（最新）", "reference (newest)")
            if index == 0
            else _t(f"源 t−{index}", f"source t-{index}")
        )
        axis.set_title(label, fontsize=9)
        axis.axis("off")
    shared = max(float(mean_weight.max()), 1e-6)
    for image in images:
        image.set_clim(0.0, shared)
    figure.colorbar(
        images[0],
        ax=axes[-1],
        fraction=0.5,
        label=_t("平均融合权重（未按帧归一化）", "mean fusion weight (not per-frame normalised)"),
    )
    axes[-1].axis("off")
    figure.suptitle(title, fontsize=11)
    if len(set(frame_ids)) < len(frame_ids):
        figure.text(
            0.01,
            0.028,
            _t(
                "注意：本窗口记录的 frame_id 有重复，因此不作为来源身份标注；来源按时间年龄区分。",
                "Note: this window's recorded frame_ids collide, so they are not used as "
                "source identities; sources are distinguished by age.",
            ),
            fontsize=8,
            color="#b03030",
        )
    figure.text(
        0.01,
        0.005,
        _t(
            "名称：平均融合权重。它既不是后验概率，也不是因果贡献；所有源共用同一色标，"
            "低权重不会被抬亮。",
            "Name: mean fusion weight. Not a posterior, not a causal contribution; all "
            "sources share one colour scale so low weights are not brightened.",
        ),
        fontsize=8,
        color="#333333",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out)
    plt.close(figure)


def _figure_motion(reference, source, offsets, mask, out, title):
    """Actual pull correspondences and warp residual; never labelled GT flow error."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from owac.baselines.burst_features import align_features

    flow = offsets.detach().float().cpu()
    source_tensor = torch.as_tensor(source.copy()).permute(2, 0, 1).float()[None]
    pair = torch.stack([source_tensor, source_tensor], 1)
    pull = torch.stack([torch.zeros_like(flow), flow], 1)
    warped, valid, _ = align_features(pair, torch.ones_like(pair[:, :, :1], dtype=torch.bool), pull)
    warp = warped[0, 1].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
    legal = valid[0, 1, 0].numpy() & mask.detach().cpu().numpy().astype(bool)
    height, width = reference.shape[:2]
    yy, xx = np.mgrid[0 : height : max(height // 12, 1), 0 : width : max(width // 12, 1)]
    dx, dy = flow[0, :, yy, xx].numpy()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(reference)
    axes[0].quiver(xx, yy, dx, dy, angles="xy", scale_units="xy", scale=1, color="cyan")
    axes[0].set_title("reference -> source pull (pixels)")
    axes[1].imshow(warp)
    axes[1].imshow(~legal, cmap="gray", alpha=(~legal).astype(float) * 0.6, vmin=0, vmax=1)
    axes[1].set_title("warped source; gray = unusable")
    residual = np.abs(reference.astype(float) - warp.astype(float)).mean(-1)
    axes[2].imshow(np.where(legal, residual, np.nan), cmap="magma", vmin=0, vmax=255)
    axes[2].set_title("alignment residual (8-bit); no GT flow")
    for axis in axes:
        axis.set_xlim(0, width - 1)
        axis.set_ylim(height - 1, 0)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(out, dpi=140)
    plt.close(figure)


def _figure_support(raw: Any, state: Any, grid: tuple[int, int], out: Path, title: str) -> None:
    """Region-level support state expanded to cells, with the four-state legend."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    codes = state.detach().cpu().numpy()
    if codes.ndim == 3:
        codes = codes[0]
    rows, cols = grid
    # ``codes`` is [history slots, regions], so the region axis is the LAST one and the
    # time axis is the first.  ``reshape(rows, cols, -1)`` folded time into the region
    # axis and made this map a scramble: a state whose region r held code r%4 rendered
    # row 0 as the first sixteen time steps instead of the first sixteen regions.
    layer = codes.reshape(codes.shape[0], rows, cols).max(0)  # per region: worst present
    height, width = raw.shape[0], raw.shape[1]
    scale_rows = max(height // rows, 1)
    scale_cols = max(width // cols, 1)
    expanded = np.kron(layer, np.ones((scale_rows, scale_cols), dtype=layer.dtype))[:height, :width]
    colour = np.zeros((*expanded.shape, 3), dtype=np.uint8)
    for code, rgb in SUPPORT_COLOURS.items():
        colour[expanded == code] = rgb

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=110)
    axes[0].imshow(raw)
    axes[0].set_title(_t("原始画面", "original frame"), fontsize=10)
    axes[1].imshow(raw)
    axes[1].imshow(colour, alpha=0.5)
    axes[1].set_title(
        _t(
            f"支持区域状态（{rows}×{cols} 区域，保守感受野展开）",
            f"support region state ({rows}x{cols} regions, conservative expansion)",
        ),
        fontsize=10,
    )
    for axis in axes:
        axis.axis("off")
    legend = [
        Patch(facecolor=np.array(rgb) / 255.0, label=name)
        for code, name in ((0, "invalid"), (1, "accepted"), (2, "deferred"), (3, "retracted"))
        for rgb in [SUPPORT_COLOURS[code]]
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=False)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out)
    plt.close(figure)


def cases(args: argparse.Namespace) -> dict[str, Any]:
    """Generate the synchronisation figures and the monitor case index."""

    import torch

    from owac.raw.n01_full_manifest import dataset_from_protocol
    from owac.raw.n01_paired_views import paired_views
    from owac.raw.rawvla_sequence import collate_sequence_samples

    config = _load_config(args.config)
    device = torch.device(args.device)
    torch.manual_seed(config.seed)
    output_dir = Path(args.output_dir)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(args.repo_root)
    protocol_path = (
        Path(args.protocol)
        if args.protocol
        else repo_root / "configs/encoders/n01_full/protocol_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset = dataset_from_protocol(
        protocol,
        config.platform,
        args.region,
        repo_root=str(repo_root),
        window=config.window_frames,
        **({"observation_stride": config.observation.history_stride} if config.observation else {}),
    )

    model = build_model(config, device=device)
    model.eval()
    if getattr(args, "checkpoint", None):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if N01V12Config.from_dict(payload["config"]).to_dict() != config.to_dict():
            raise ValueError("case checkpoint configuration mismatch")
        model.load_state_dict(payload["model"])
    elif not getattr(args, "allow_untrained", False):
        raise ValueError("cases require --checkpoint or explicit --allow-untrained")

    # Documented selection rule: evenly spaced strata of the named region, plus the
    # scan's most-retracted sample as the "event" case.  Failure cases need closed
    # loop results, which this stage does not have yet, so none are claimed.
    scan = min(int(args.scan), len(dataset))
    stride = max(len(dataset) // max(scan, 1), 1)
    candidates = [(index * stride) % len(dataset) for index in range(scan)]
    scored: list[tuple[int, int]] = []
    with torch.no_grad():
        for index in candidates:
            batch = collate_sequence_samples([dataset[index]])
            _, diagnostics = _run_window(model, batch, device, no_grad=True)
            total = 0
            for entry in diagnostics.values():
                counts = (entry.get("support") or {}).get("state_counts") or {}
                total += int(counts.get("retracted", 0))
            scored.append((total, index))
    scored.sort(reverse=True)
    chosen: list[tuple[str, int]] = []
    if scored:
        chosen.append(("event_max_retracted", scored[0][1]))
    even = sorted({index for _, index in scored})
    if even:
        chosen.append(("even_first", even[0]))
        chosen.append(("even_middle", even[len(even) // 2]))
        chosen.append(("even_last", even[-1]))
    seen: set[int] = set()
    unique: list[tuple[str, int]] = []
    for label, index in chosen:
        if index in seen:
            continue
        seen.add(index)
        unique.append((label, index))

    artifacts: list[dict[str, Any]] = []
    case_receipts: list[dict[str, Any]] = []
    detail_box = tuple(int(part) for part in str(args.detail_box).split(","))
    for case_label, index in unique:
        sample = dataset[index]
        batch = collate_sequence_samples([sample])
        episode = str(batch["meta"][0].get("episode_key", index))
        with torch.no_grad():
            captures = {}

            def capture(camera, output, captures=captures):
                import copy

                captures[camera] = (
                    copy.copy(getattr(model.encoder, "last_fusion", None)),
                    getattr(model.encoder, "last_support", None),
                )
                return output

            from owac.training.n01_v12 import encode_windows_v12

            _, diagnostics = encode_windows_v12(model, batch, device, transform=capture)
        for camera in sorted(batch["windows"]):
            window = batch["windows"][camera]
            values = window.values
            # Show the frame the fusion actually used as its reference, not slot 0 of
            # the stored window: with left padding those are different frames, and the
            # figure must describe what the model saw.
            fusion, decision = captures[camera]
            if fusion is not None and fusion.get("camera_id") == camera:
                frame_ids = [int(value) for value in fusion["frame_ids"]]
                unique = _unique_frames(window.frame_ids)
            else:  # pragma: no cover - only when the window never reached the content path
                # Drop the unmatched fusion rather than leaving it bound: the weight figure
                # below tests ``fusion is not None`` only, so a leftover from the previous
                # camera would be plotted under this camera's name.
                fusion = None
                frame_ids = [int(value) for value in window.frame_ids[0].reshape(-1).tolist()]
                unique = _unique_frames(window.frame_ids)
            source_slot = int((window.frame_ids[0] == frame_ids[0]).nonzero()[0])
            pair = paired_views(values[0, source_slot], platform=config.platform)
            raw_view = _to_view8(pair.raw)
            rgb_view = _to_view8(pair.rgb)

            stem = f"{case_label}_{camera}"
            pair_path = artifacts_dir / f"{stem}_pair.png"
            _figure_pair(
                raw_view,
                rgb_view,
                pair_path,
                f"{config.platform} / {camera} / 同源配对与局部差异",
                (
                    f"episode={episode} 参考 frame_id={frame_ids[0]} 唯一帧={unique} / 槽 "
                    f"{int(values.shape[1])}；provenance={pair.provenance['module']}；"
                    f"cache={pair.provenance['cache_dtype']}；RAW 预览用作者存储视图，"
                    "未做额外色调映射。"
                ),
            )
            detail_path = artifacts_dir / f"{stem}_detail.png"
            _figure_detail(
                raw_view,
                rgb_view,
                detail_path,
                detail_box,
                f"{config.platform} / {camera} / 局部细节与量化",
            )
            artifacts.append(
                {
                    "id": f"{stem}_pair",
                    "case_id": case_label,
                    "case_title": f"{case_label}（{config.platform}/{camera}）",
                    "kind": "image",
                    "url": f"artifacts/{pair_path.name}",
                    "caption": "同源 RAW 与冻结 Default ISP 的 RGB 及局部差异",
                    "camera": camera,
                    "episode": episode,
                    "frame": frame_ids[0],
                    "legend": "|RAW − RGB| 通道均值，单位 8-bit 级",
                    "source_time": f"frame_id={frame_ids[0]}",
                    "unit": "uint8 8-bit 级",
                }
            )
            artifacts.append(
                {
                    "id": f"{stem}_detail",
                    "case_id": case_label,
                    "case_title": f"{case_label}（{config.platform}/{camera}）",
                    "kind": "image",
                    "url": f"artifacts/{detail_path.name}",
                    "caption": "局部最近邻放大与直方图：量化与对比度",
                    "camera": camera,
                    "episode": episode,
                    "frame": frame_ids[0],
                    "legend": f"局部框 top,left,h,w={detail_box}",
                    "source_time": f"frame_id={frame_ids[0]}",
                    "unit": "8-bit 值 / 像素数",
                }
            )
            if fusion is not None and fusion.get("weights") is not None:
                fusion_path = artifacts_dir / f"{stem}_fusion.png"
                _figure_fusion(
                    raw_view,
                    fusion["weights"],
                    frame_ids,
                    fusion_path,
                    f"{config.platform} / {camera} / 平均融合权重（DBSR）",
                )
                artifacts.append(
                    {
                        "id": f"{stem}_fusion",
                        "case_id": case_label,
                        "case_title": f"{case_label}（{config.platform}/{camera}）",
                        "kind": "image",
                        "url": f"artifacts/{fusion_path.name}",
                        "caption": "真实历史融合权重按固定通道均值聚合到原画面",
                        "camera": camera,
                        "episode": episode,
                        "frame": frame_ids[0],
                        "legend": "平均融合权重（不是后验、不是因果贡献）",
                        "source_time": f"sources={frame_ids}",
                        "unit": "softmax 权重（所有源共用色标）",
                    }
                )
            if fusion is not None and fusion.get("offsets_xy") is not None and len(frame_ids) > 1:
                own_reference = _to_view8(fusion["window_values"][0, 0])
                own_source = _to_view8(fusion["window_values"][0, 1])
                motion_path = artifacts_dir / f"{stem}_motion.png"
                _figure_motion(
                    own_reference,
                    own_source,
                    fusion["offsets_xy"][:, 1],
                    fusion["cell_mask"][0, 1].all(0),
                    motion_path,
                    f"{config.platform}/{camera} sources={frame_ids[:2]} same-domain pull",
                )
                artifacts.append(
                    {
                        "id": f"{stem}_motion",
                        "case_id": case_label,
                        "kind": "image",
                        "url": f"artifacts/{motion_path.name}",
                        "camera": camera,
                        "caption": "Actual own-domain pull/warp; residual is not GT flow error",
                        "frame": frame_ids[0],
                        "source_time": f"history={frame_ids[1]}",
                        "unit": "pixels / 8-bit alignment residual",
                    }
                )
            if decision is not None and bool(
                (diagnostics.get(camera) or {}).get("p3_support_applied")
            ):
                support_path = artifacts_dir / f"{stem}_support.png"
                grid = model.config.support.region_grid
                _figure_support(
                    raw_view,
                    decision.state,
                    (grid, grid),
                    support_path,
                    f"{config.platform} / {camera} / 支持区域状态",
                )
                artifacts.append(
                    {
                        "id": f"{stem}_support",
                        "case_id": case_label,
                        "case_title": f"{case_label}（{config.platform}/{camera}）",
                        "kind": "image",
                        "url": f"artifacts/{support_path.name}",
                        "caption": "支持区域接受/暂缓/退出",
                        "camera": camera,
                        "episode": episode,
                        "frame": frame_ids[0],
                        "legend": "invalid/accepted/deferred/retracted（保守感受野展开）",
                        "source_time": f"history_sources={len(frame_ids) - 1}",
                        "unit": "区域状态码",
                    }
                )
            case_receipts.append(
                {
                    "case": case_label,
                    "sample_index": index,
                    "camera": camera,
                    "episode": episode,
                    "frame_id": int(frame_ids[0]),
                    "slots": int(values.shape[1]),
                    "unique_frames": unique,
                    "fused_reference_frame_id": frame_ids[0],
                    "fused_window_frame_ids": frame_ids,
                    "identical_source": pair.provenance["pairs"],
                    "support": (diagnostics.get(camera) or {}).get("support"),
                }
            )

    index = {
        "generated_at": os.popen("date -Iseconds").read().strip(),
        "stage": "P3",
        "arm": config.arm,
        "platform": config.platform,
        "region": args.region,
        "selection_rule": (
            f"evenly spaced strata of the {args.region} region (scan={scan}, stride={stride}) "
            "plus the scanned sample with the most retracted support regions as the event case. "
            "Failure cases require closed-loop results, which this stage has none of, so no "
            "failure case is claimed."
        ),
        "artifacts": artifacts,
    }
    _write_json_atomic(output_dir / "cases" / "index.json", index)
    receipt = {
        "stage": "P3",
        "command": "cases",
        "checkpoint": getattr(args, "checkpoint", None),
        "trained_weights": bool(getattr(args, "checkpoint", None)),
        "arm": config.arm,
        "platform": config.platform,
        "region": args.region,
        "cases": case_receipts,
        "artifact_count": len(artifacts),
        "index_path": str(output_dir / "cases" / "index.json"),
        "selection_rule": index["selection_rule"],
        "config": config.to_dict(),
    }
    if args.output:
        _write_json_atomic(Path(args.output), receipt)
    return receipt


def _complete_support_scales(args, config):
    import torch

    from owac.raw.rawvla_sequence import collate_sequence_samples

    model = build_model(config, device=args.device).eval()
    if getattr(args, "checkpoint", None):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if N01V12Config.from_dict(payload["config"]).to_dict() != config.to_dict():
            raise ValueError("support calibration checkpoint configuration mismatch")
        model.load_state_dict(payload["model"])
    elif not getattr(args, "allow_untrained", False):
        raise ValueError("supply --checkpoint or explicitly request --allow-untrained diagnostics")
    dataset = _datasets(args, config, args.region)
    count = min(args.samples, len(dataset))
    samples = []
    for i in range(count):
        batch = collate_sequence_samples([dataset[i * len(dataset) // max(count, 1)]])
        with torch.no_grad():
            _, diagnostics = _run_window(model, batch, args.device, no_grad=True)
        for camera, diagnostic in diagnostics.items():
            if getattr(args, "camera", None) and args.camera != camera:
                continue
            samples.append(
                {
                    "camera": camera,
                    "sample_index": i,
                    "support": diagnostic.get("support"),
                    "motion": diagnostic.get("p2_motion"),
                    "window_frames": diagnostic.get("window_frames"),
                }
            )
    receipt = {
        "command": "support-scales",
        "implementation": "gls_window",
        "region": args.region,
        "checkpoint": getattr(args, "checkpoint", None),
        "trained_weights": bool(getattr(args, "checkpoint", None)),
        "samples": samples,
        "config": config.to_dict(),
        "thresholds_auto_applied": False,
        "note": (
            "Actual motion/GLS/window-replay diagnostics. Unlabelled trajectory quantiles "
            "are not a physical false-alarm calibration; use labelled stable/change controls."
        ),
    }
    if args.output:
        _write_json_atomic(Path(args.output), receipt)
    return receipt


def support_scales(args: argparse.Namespace) -> dict[str, Any]:
    """Measure the support statistic on real calibration data so thresholds can be chosen.

    This mirrors the production support path exactly -- arrivals grow the window, the
    newest slot is the reference, one BOCPD update per arrival -- but it runs on CPU and
    skips the DBSR backbone, because the support mechanism makes no use of image
    features.  That means the scales can be re-derived from data without taking a GPU
    from training.

    It reports distributions and a proposed threshold set; it does **not** silently
    rewrite any configuration.  Choosing scales is a recorded decision, and the caller
    applies them.
    """

    import torch

    from owac.frameworks.n01_full.support import SpatialSupport

    config = _load_config(args.config)
    if config.support.implementation == "gls_window":
        return _complete_support_scales(args, config)
    dataset = _datasets(args, config, args.region)
    count = min(int(args.samples), len(dataset))
    support = SpatialSupport(config.support)
    capacity = max(int(config.window_frames) - 1, 1)

    logodds: list[float] = []
    residuals: list[float] = []
    accepted_fraction: list[float] = []
    retracted_fraction: list[float] = []
    deferred_fraction: list[float] = []
    per_camera: dict[str, int] = {}
    for index in range(count):
        sample = dataset[(index * max(len(dataset) // max(count, 1), 1)) % len(dataset)]
        for camera, window in sorted(sample.windows.items()):
            if getattr(args, "camera", None) and camera != args.camera:
                continue
            values = torch.as_tensor(window.values).float()
            valid = torch.as_tensor(window.valid_mask)
            # The per-sample layout is platform-dependent and `values` and
            # `valid_mask` always share it, so they are promoted together.  Measured:
            # LIBERO gives [B, T, C, H, W] and RoboTwin gives [T, C, H, W]; reading a
            # 4-dim mask as [B, T, H, W] would silently treat channels as batch.
            if values.ndim == 4:
                values = values[None]
                valid = valid[None]
            if valid.ndim != 5 or valid.shape[-3] != values.shape[2]:
                raise ValueError(
                    f"camera {camera}: unsupported layout values={tuple(values.shape)} "
                    f"valid_mask={tuple(valid.shape)}"
                )
            cell = valid.all(2, keepdim=True)
            batch, length = values.shape[0], values.shape[1]
            if length < 2:
                continue
            state = support.initial_state(
                batch, capacity, values.shape[-2], values.shape[-1], values.device
            )
            source_valid = torch.ones(batch, length, dtype=torch.bool, device=values.device)
            decision = None
            for t in range(1, length):
                # Newest-first, exactly as the buffer delivers the window.
                grown = values[:, : t + 1].flip(1)
                mask = cell[:, : t + 1].flip(1)
                state, decision = support.observe(state, grown, mask, source_valid[:, : t + 1])
            if decision is None:
                continue
            logodds.extend(float(v) for v in decision.change_logodds.reshape(-1))
            residuals.extend(float(v) for v in decision.region_residual.reshape(-1))
            codes = decision.state.reshape(-1)
            total = max(int(codes.numel()), 1)
            accepted_fraction.append(float((codes == 1).sum()) / total)
            deferred_fraction.append(float((codes == 2).sum()) / total)
            retracted_fraction.append(float((codes == 3).sum()) / total)
            per_camera[camera] = per_camera.get(camera, 0) + 1

    def percentiles(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)

        def pick(fraction: float) -> float:
            return round(ordered[min(int(fraction * (len(ordered) - 1)), len(ordered) - 1)], 4)

        summary = {
            "n": len(ordered),
            "min": round(ordered[0], 4),
            "max": round(ordered[-1], 4),
        }
        for fraction in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
            summary[f"q{int(fraction * 100):02d}"] = pick(fraction)
        return summary

    mean = lambda seq: (sum(seq) / len(seq)) if seq else None  # noqa: E731
    receipt = {
        "stage": "P3",
        "command": "support-scales",
        "platform": config.platform,
        "arm": config.arm,
        "region": args.region,
        "samples": count,
        "windows_measured": per_camera,
        "crc_note": (
            "the support statistic uses no image features, so this ran on CPU and did not "
            "take a GPU from training"
        ),
        "change_logodds": percentiles(logodds),
        "region_residual": percentiles(residuals),
        "decision_fraction_mean": {
            "accepted": mean(accepted_fraction),
            "deferred": mean(deferred_fraction),
            "retracted": mean(retracted_fraction),
        },
        "current_config": config.support.to_dict(),
        "derived_operating_point": _operating_point(logodds, residuals, args),
        "read_this_as": (
            "distributions on real data, not a convergence result. A threshold set is only "
            "defensible together with the decision fractions it produces and the calibration "
            "region it was measured on."
        ),
    }
    if args.output:
        _write_json_atomic(Path(args.output), receipt)
    return receipt


def _operating_point(
    logodds: list[float], residuals: list[float], args: argparse.Namespace
) -> dict[str, Any] | None:
    """Thresholds derived from the measured distribution, not chosen by taste.

    The changepoint statistic is unbounded and its sign flips between platforms, so one
    fixed threshold cannot transfer: on RoboTwin the median sits *below* the default
    ``retract_logodds`` and most regions are flagged changed, while on LIBERO the median
    is far above it.  This reports thresholds as recorded quantiles of the calibration
    distribution, so the decision fractions are comparable by construction and the rule
    that produced them travels with them.  It does not apply them: editing the arm's
    SupportConfig keeps the change reviewable in the configuration record.
    """

    if not logodds:
        return None
    ordered = sorted(logodds)

    def quantile(fraction: float) -> float:
        index = min(int(fraction * (len(ordered) - 1)), len(ordered) - 1)
        return round(ordered[index], 4)

    retract_q = float(getattr(args, "retract_quantile", 0.10))
    accept_q = float(getattr(args, "accept_quantile", 0.75))
    large_q = float(getattr(args, "large_quantile", 0.75))

    def residual_quantile(fraction: float) -> float | None:
        if not residuals:
            return None
        ordered_res = sorted(residuals)
        index = min(int(fraction * (len(ordered_res) - 1)), len(ordered_res) - 1)
        return round(ordered_res[index], 4)

    return {
        "rule": (
            f"retract_logodds = calibration q{int(retract_q * 100)}; "
            f"accept_logodds = calibration q{int(accept_q * 100)}; "
            f"region_residual q{int(large_q * 100)} reported as a diagnostic only "
            f"(not applied; see derived_operating_point.large_residual_threshold)"
        ),
        "retract_logodds": quantile(retract_q),
        "accept_logodds": quantile(accept_q),
        # Deliberately NOT emitted as a derived threshold.  `residual > 1 sigma` is a
        # physical statement, unlike the arbitrary zero point of change_logodds, so
        # replacing it with a quantile would flag a fixed fraction of regions whether or
        # not anything is suspicious.  The quantile is reported as a diagnostic only, next
        # to the cross-platform spread that is the actual finding.
        "large_residual_threshold": {
            "derived_quantile": residual_quantile(large_q),
            "do_not_apply": True,
            "why": (
                "residual > 1 sigma is physical; a quantile here is self-referential and "
                "would flag a fixed fraction unconditionally. A cross-platform spread in "
                "the residual median points at the noise estimator, not at this threshold."
            ),
        },
        "applied_automatically": False,
        "note": (
            "the quantile rule is a chosen operating point, not a data fact, so it is "
            "reported next to the distribution it came from"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(item: argparse.ArgumentParser) -> None:
        item.add_argument("--config", required=True)
        item.add_argument("--checkpoint", default=None)
        item.add_argument("--allow-untrained", action="store_true")
        item.add_argument("--repo-root", default=".")
        item.add_argument("--deploy-root", default=os.environ.get("N01_DEPLOY_ROOT", "."))
        item.add_argument("--protocol", default=None)
        item.add_argument("--device", default="cuda:0")
        item.add_argument("--output", default=None)

    validate_parser = sub.add_parser("validate", help="fixed-action validation subset")
    common(validate_parser)
    validate_parser.add_argument("--samples", type=int, default=8)
    validate_parser.set_defaults(func=validate)

    intervene_parser = sub.add_parser("intervene", help="content-path source deletion")
    common(intervene_parser)
    intervene_parser.add_argument("--region", default="validation")
    intervene_parser.add_argument("--index", type=int, default=0)
    intervene_parser.add_argument("--camera", default=None)
    intervene_parser.add_argument("--slot", type=int, default=1)
    intervene_parser.add_argument(
        "--mode",
        choices=("mask", "fixed", "delete"),
        default="delete",
        help=(
            "delete = whole-source deletion (the frame never arrives and all affected "
            "decisions are rebuilt); mask = validity-zeroing only. They are different "
            "operations and are labelled separately in the receipt."
        ),
    )
    intervene_parser.set_defaults(func=intervene)

    cases_parser = sub.add_parser("cases", help="synchronisation figures and case index")
    common(cases_parser)
    cases_parser.add_argument("--output-dir", required=True)
    cases_parser.add_argument("--region", default="validation")
    cases_parser.add_argument("--scan", type=int, default=8)
    cases_parser.add_argument("--detail-box", default="64,64,48,48")
    cases_parser.set_defaults(func=cases)

    scales = sub.add_parser(
        "support-scales", help="measure the support statistic on calibration data (CPU)"
    )
    common(scales)
    scales.add_argument("--region", default="calibration")
    scales.add_argument("--samples", type=int, default=8)
    scales.add_argument(
        "--camera",
        default=None,
        help=(
            "restrict to one camera. The decision fractions differ sharply between "
            "cameras -- agentview and wrist are not one population -- so an aggregate "
            "number must not be quoted as if it described both."
        ),
    )
    scales.add_argument("--retract-quantile", type=float, default=0.10)
    scales.add_argument("--accept-quantile", type=float, default=0.75)
    scales.add_argument("--large-quantile", type=float, default=0.75)
    scales.set_defaults(func=support_scales)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
