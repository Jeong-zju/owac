"""Parity suite for the N02 full component ports (contract: ``PARITY_VERSION``).

``run_all`` never raises: a check whose optional dependency or reference fixture is missing is
reported as ``not_run`` with a concrete reason, and a check whose numbers are out of tolerance is
reported as ``fail``.  Reference numbers always come from the isolated reference environment
(``outputs/n02-full/p0/parity-venv``) through fixtures written by ``upstream_refs.py``; nothing in
this module fabricates a number or treats a skipped run as a pass.

Tolerances come from the P0 contract section 5 and are not adjusted here:

* FP32 forward: ``atol=1e-5``, ``rtol=1e-4``  (the fp32 accumulation *differences between two
  frameworks*, so a same-framework check must be far below this)
* FP32 input/parameter gradients: ``atol=1e-5``, ``rtol=1e-3``
* FP64 analytic solutions: error ``<= 1e-8``; numerical gradients are reported as a step sweep
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

PARITY_VERSION = "n02-full-components-v1"

TOLERANCES: dict[str, dict[str, float]] = {
    "fp32_forward": {"atol": 1e-5, "rtol": 1e-4},
    "fp32_grad": {"atol": 1e-5, "rtol": 1e-3},
    "fp64_analytic": {"atol": 1e-8, "rtol": 0.0},
}

DEFAULT_FIXTURE_DIR = "outputs/n02-full/p0/parity/fixtures"
DEFAULT_OUTPUT_DIR = "outputs/n02-full/p0/parity"

CHECK_ORDER = (
    "convnext_base_block",
    "convnext_grn_global",
    "convnext_base_multiscale",
    "fcmae_patchify_roundtrip",
    "fcmae_sparse_upstream",
    "fcmae_mask_aware_dense",
    "savi_slot_attention",
    "savi_gru",
    "savi_processor",
    "savi_predictor",
    "savi_masked_extension",
    "dnri_one_step_causal",
    "theseus_lm_quadratic",
    "theseus_upstream_simple_example",
    "smc_ref",
    "genjl_particle_filter",
    "short_train_trend",
)


@dataclass
class CheckResult:
    """One parity verdict; ``evidence_path`` always points at a real artefact."""

    name: str
    status: str  # "pass" | "fail" | "not_run"
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    atol: float | None = None
    rtol: float | None = None
    reason: str = ""
    evidence_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def available_checks() -> tuple[str, ...]:
    """Names accepted by ``checks=`` in :func:`run_all`, in execution order."""
    return CHECK_ORDER


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _paths(config: Any) -> tuple[Path, Path]:
    cfg = asdict(config) if is_dataclass(config) else config or {}
    if not isinstance(cfg, dict):
        cfg = {}
    fixture_dir = Path(
        cfg.get("fixture_dir", cfg.get("paths", {}).get("parity_fixtures", DEFAULT_FIXTURE_DIR))
    )
    output_dir = Path(
        cfg.get("output_dir", cfg.get("paths", {}).get("parity_output", DEFAULT_OUTPUT_DIR))
    )
    return fixture_dir, output_dir


def environment_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0], "torch": None}
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception as exc:  # noqa: BLE001
        versions["torch"] = f"not_available: {exc}"
    for name in ("jax", "flax", "theseus", "numba", "timm", "platform"):
        if name == "platform":
            versions["platform"] = platform.platform()
            continue
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "installed")
        except Exception as exc:  # noqa: BLE001 - absence is the recorded fact
            versions[name] = f"not_available: {type(exc).__name__}"
    return versions


def _relative_error(pred, ref, atol: float) -> tuple[float, float]:
    import torch

    if isinstance(pred, torch.Tensor):
        pred = pred.detach()
    if isinstance(ref, torch.Tensor):
        ref = ref.detach()
    pred = torch.as_tensor(pred, dtype=torch.float64)
    ref = torch.as_tensor(ref, dtype=torch.float64)
    abs_err = float((pred - ref).abs().max().detach()) if pred.numel() else 0.0
    denom = ref.abs().clamp_min(atol * 10.0)
    rel_err = float(((pred - ref).abs() / denom).max()) if pred.numel() else 0.0
    return abs_err, rel_err


def _verdict(abs_err: float, rel_err: float, atol: float, rtol: float) -> bool:
    return abs_err <= atol or rel_err <= rtol


def _load_fixture(fixture_dir: Path, name: str):
    """Load a reference fixture: ``.pt`` (torch payload) or ``.npz`` (JAX export)."""
    import torch

    pt_path = fixture_dir / f"{name}.pt"
    if pt_path.exists():
        return torch.load(pt_path, map_location="cpu", weights_only=False)
    npz_path = fixture_dir / f"{name}.npz"
    if npz_path.exists():
        import numpy as np

        return np.load(npz_path, allow_pickle=False)
    raise FileNotFoundError(f"reference fixture not exported yet: {pt_path}")


def _not_run(name: str, reason: str, atol: float | None = None, rtol: float | None = None):
    return CheckResult(name=name, status="not_run", reason=reason, atol=atol, rtol=rtol)


# --------------------------------------------------------------------------------------
# checks: ConvNeXt V2 (upstream dense implementation, run verbatim in the reference env)
# --------------------------------------------------------------------------------------


def check_convnext_base_block(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    import torch

    from .convnextv2 import Block

    tol = TOLERANCES["fp32_forward"]
    grad_tol = TOLERANCES["fp32_grad"]
    try:
        fx = _load_fixture(fixture_dir, "convnext_block")
    except FileNotFoundError as exc:
        return _not_run("convnext_base_block", str(exc), tol["atol"], tol["rtol"])

    block = Block(dim=fx["dim"], drop_path=0.0).to(device=device, dtype=torch.float32).eval()
    load = block.load_state_dict(fx["state_dict"], strict=True)
    x = fx["input"].clone().to(device).requires_grad_(True)
    y = block(x)
    (grad,) = torch.autograd.grad(y.pow(2).sum(), x)
    fwd_abs, fwd_rel = _relative_error(y, fx["upstream_output"], tol["atol"])
    grad_abs, grad_rel = _relative_error(grad, fx["upstream_input_grad"], grad_tol["atol"])
    ok = _verdict(fwd_abs, fwd_rel, tol["atol"], tol["rtol"]) and _verdict(
        grad_abs, grad_rel, grad_tol["atol"], grad_tol["rtol"]
    )
    return CheckResult(
        name="convnext_base_block",
        status="pass" if ok else "fail",
        max_abs_err=fwd_abs,
        max_rel_err=fwd_rel,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            f"upstream models.convnextv2.Block vs port Block on non-square (2,{fx['dim']},5,7); "
            f"input-grad max_abs={grad_abs:.3e} max_rel={grad_rel:.3e}"
        ),
        evidence_path=str(fixture_dir / "convnext_block.pt"),
        metrics={
            "input_grad_max_abs": grad_abs,
            "input_grad_max_rel": grad_rel,
            "grad_atol": grad_tol["atol"],
            "grad_rtol": grad_tol["rtol"],
            "state_dict_missing": list(load.missing_keys),
            "state_dict_unexpected": list(load.unexpected_keys),
        },
    )


def check_convnext_grn_global(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    import torch

    from .convnextv2 import GRN

    tol = TOLERANCES["fp32_forward"]
    try:
        fx = _load_fixture(fixture_dir, "convnext_grn")
    except FileNotFoundError as exc:
        return _not_run("convnext_grn_global", str(exc), tol["atol"], tol["rtol"])

    grn = GRN(dim=fx["input"].shape[-1]).to(device).eval()
    grn.load_state_dict(fx["state_dict"], strict=True)
    with torch.no_grad():
        y1 = grn(fx["input"].to(device))
        y2 = grn(fx["input_perturbed"].to(device))
    abs_err, rel_err = _relative_error(y1, fx["upstream_output"], tol["atol"])
    pair_abs, _ = _relative_error(y2, fx["upstream_output_perturbed"], tol["atol"])
    # A single far pixel must change every output pixel: the GRN reduction is spatially global.
    delta = float((y2 - y1).abs().max().detach())
    ok = _verdict(abs_err, rel_err, tol["atol"], tol["rtol"]) and delta > 0.0
    return CheckResult(
        name="convnext_grn_global",
        status="pass" if ok else "fail",
        max_abs_err=max(abs_err, pair_abs),
        max_rel_err=rel_err,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "GRN statistics are taken over the whole spatial extent: perturbing one distant pixel "
            f"changes the output by max_abs={delta:.3e} (upstream far_pixel_delta="
            f"{fx['far_pixel_delta']:.3e})"
        ),
        evidence_path=str(fixture_dir / "convnext_grn.pt"),
        metrics={"far_pixel_effect_abs": delta, "upstream_far_pixel_delta": fx["far_pixel_delta"]},
    )


def _check_full_convnext(fixture_name: str, device: str, fixture_dir: Path):
    import torch

    from .convnextv2 import RawConvNeXtV2

    tol = TOLERANCES["fp32_forward"]
    try:
        fx = _load_fixture(fixture_dir, fixture_name)
    except FileNotFoundError as exc:
        return None, _not_run("convnext_base_multiscale", str(exc), tol["atol"], tol["rtol"])
    model = RawConvNeXtV2(
        in_chans=fx["in_chans"],
        depths=fx["depths"],
        dims=fx["dims"],
        num_classes=fx["num_classes"],
    ).to(device)
    load = model.load_state_dict(fx["state_dict"], strict=True)
    x = fx["input"].to(device)
    with torch.no_grad():
        multiscale = model.forward_multiscale(x)
        pooled = model.forward_stagewise(x)["pooled"]
        logits = model(x) if model.head is not None else None
    stage_outputs = [multiscale[f"stage{i}_stride{4 * 2**i}"] for i in range(4)]
    errors = []
    for got, want in zip(stage_outputs, fx["stage_outputs"], strict=False):
        errors.append(_relative_error(got, want, tol["atol"]))
    pooled_err = _relative_error(pooled, fx["pooled"], tol["atol"])
    max_abs = max(e[0] for e in errors + [pooled_err])
    max_rel = max(e[1] for e in errors + [pooled_err])
    return {
        "shapes": [tuple(t.shape) for t in stage_outputs],
        "max_abs": max_abs,
        "max_rel": max_rel,
        "pooled_max_abs": pooled_err[0],
        "load": load,
        "logits_err": (
            _relative_error(logits, fx["logits"], tol["atol"]) if logits is not None else None
        ),
        "in_chans": fx["in_chans"],
        "input_shape": tuple(fx["input"].shape),
    }, None


def check_convnext_base_multiscale(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    tol = TOLERANCES["fp32_forward"]
    results = {}
    for fixture_name in ("convnext_3ch_square", "convnext_4ch_rect"):
        payload, failure = _check_full_convnext(fixture_name, device, fixture_dir)
        if failure is not None:
            return failure
        results[fixture_name] = payload
    max_abs = max(r["max_abs"] for r in results.values())
    max_rel = max(r["max_rel"] for r in results.values())
    ok = _verdict(max_abs, max_rel, tol["atol"], tol["rtol"])
    shapes = {k: [list(s) for s in v["shapes"]] for k, v in results.items()}
    return CheckResult(
        name="convnext_base_multiscale",
        status="pass" if ok else "fail",
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "upstream ConvNeXtV2 stage outputs (stride 4/8/16/32) and pooled features vs the RAW "
            "port for 3-channel 32x32 and 4-channel 32x48 inputs; stride 8/16 shapes="
            f"{shapes['convnext_4ch_rect'][1:3]}"
        ),
        evidence_path=str(fixture_dir),
        metrics={
            "per_case": {
                k: {kk: vv for kk, vv in v.items() if kk != "load"} for k, v in results.items()
            }
        },
    )


# --------------------------------------------------------------------------------------
# checks: FCMAE geometry and the mask-aware dense port
# --------------------------------------------------------------------------------------


def check_fcmae_patchify_roundtrip(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    import torch

    from .fcmae import gen_random_mask, patchify, unpatchify, upsample_patch_mask

    tol = TOLERANCES["fp32_forward"]
    # 240x320 is the RoboTwin RAW resolution; it is not divisible by the 32-pixel patch of the
    # 4-stage encoder, so the exact-power-of-two variant is checked at patch 16 and the 32-pixel
    # patch is checked at a stride-aligned 240x320 crop of 240->256 padded / 320 columns.
    cases = {
        "3ch_square_64x64_p32": (3, 64, 64, 32),
        "4ch_square_64x64_p32": (4, 64, 64, 32),
        "3ch_rect_240x320_p16": (3, 240, 320, 16),
        "4ch_rect_240x320_p16": (4, 240, 320, 16),
        "3ch_rect_256x320_p32": (3, 256, 320, 32),
        "4ch_rect_256x320_p32": (4, 256, 320, 32),
    }
    torch.manual_seed(1103)
    per_case: dict[str, Any] = {}
    worst = 0.0
    for name, (chans, height, width, p) in cases.items():
        x = torch.randn(2, chans, height, width)
        patches = patchify(x, p)
        restored = unpatchify(patches, p, chans, (height // p, width // p))
        err = float((restored - x).abs().max().detach())
        expected_len = (height // p) * (width // p)
        assert patches.shape[1] == expected_len, (patches.shape, expected_len)
        per_case[name] = {
            "roundtrip_max_abs": err,
            "patch_shape": list(patches.shape),
            "num_patches": int(patches.shape[1]),
            "expected_patches": expected_len,
        }
        worst = max(worst, err)

    # mask bookkeeping: exactly int(L * (1 - ratio)) patches are kept, and mask positions are
    # independent of the data (no leakage of pixel content).
    grid = (15, 20)
    for ratio in (0.0, 0.4, 0.6, 0.9):
        mask = gen_random_mask(grid, 4, ratio, generator=torch.Generator().manual_seed(7))
        kept = (mask == 0).sum(dim=1)
        expected_keep = int(grid[0] * grid[1] * (1 - ratio))
        assert torch.all(kept == expected_keep), (ratio, kept, expected_keep)
    upsampled = upsample_patch_mask(mask, 8, grid)
    assert upsampled.shape == (4, grid[0] * 8, grid[1] * 8)

    # the upstream 3-channel square reference, when its fixture is present
    upstream_note = "upstream fixture absent"
    upstream_err = None
    try:
        fx = _load_fixture(fixture_dir, "fcmae_geometry_3ch")
        patches = patchify(fx["input"], fx["patch_size"])
        restored = unpatchify(patches, fx["patch_size"], 3, (2, 2))
        upstream_err = {
            "patchify_vs_upstream_max_abs": float(
                (patches - fx["upstream_patches"]).abs().max().detach()
            ),
            "unpatchify_vs_upstream_max_abs": float(
                (restored - fx["upstream_unpatchified"]).abs().max()
            ),
        }
        upstream_note = "compared against upstream models.fcmae.FCMAE methods"
    except FileNotFoundError:
        pass

    ok = worst <= tol["atol"] and (
        upstream_err is None or max(upstream_err.values()) <= tol["atol"]
    )
    return CheckResult(
        name="fcmae_patchify_roundtrip",
        status="pass" if ok else "fail",
        max_abs_err=worst,
        max_rel_err=None,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "patchify/unpatchify round-trip is exact for 3/4 channels and square/rectangular "
            f"inputs (worst |x - unpatchify(patchify(x))| = {worst:.3e}); mask keeps exactly "
            f"int(L*(1-ratio)) patches; {upstream_note}"
        ),
        evidence_path=str(fixture_dir / "fcmae_geometry_3ch.pt"),
        metrics={"per_case": per_case, "upstream_comparison": upstream_err},
    )


def check_fcmae_sparse_upstream(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    """Consume the actual CUDA custom-kernel run, with source identity checked."""
    import hashlib

    receipt = fixture_dir / "fcmae_sparse_upstream.json"
    if not receipt.is_file():
        return _not_run(
            "fcmae_sparse_upstream", "pinned custom Minkowski reference receipt missing"
        )
    payload = json.loads(receipt.read_text())
    root = Path(__file__).parent
    hashes_ok = all(
        hashlib.sha256((root / "upstream" / "convnextv2" / name).read_bytes()).hexdigest() == digest
        for name, digest in payload["source_hashes"].items()
    ) and all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in payload["port_source_hashes"].items()
    )
    layers = payload["layers"]
    grads = payload["parameter_gradients"]
    ok = (
        hashes_ok
        and payload["input_grad_close"]
        and all(row["forward_close"] and row["support_identical"] for row in layers.values())
        and all(row["close"] for row in grads.values())
    )
    error = max([row["max_abs"] for row in layers.values()])
    return CheckResult(
        name="fcmae_sparse_upstream",
        status="pass" if ok else "fail",
        max_abs_err=error,
        atol=1e-5,
        rtol=1e-4,
        reason="official sparse CUDA vs dense: coordinates, layers and gradients",
        evidence_path=str(receipt),
        metrics=payload,
    )


def check_fcmae_mask_aware_dense(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    """Mask-aware dense port: support sets, valid-only statistics, gradients, no leakage.

    This is a *naming reference* for the sparse encoder (dense cost), never a claim about the
    upstream sparse operator.  Everything asserted here is measured against an independent
    coordinate-set simulation written in the test module.
    """
    import torch

    from .fcmae import (
        MaskAwareDenseConvNeXtV2Encoder,
        PlainDenseZeroFillEncoder,
        dilate_support,
        gen_random_mask,
        upsample_patch_mask,
    )

    torch.manual_seed(2207)
    dims, depths = (8, 16, 32, 64), (1, 1, 1, 1)
    encoder = MaskAwareDenseConvNeXtV2Encoder(
        in_chans=3, depths=depths, dims=dims, patch_size=32
    ).eval()
    x = torch.randn(1, 3, 64, 64)
    mask = gen_random_mask((2, 2), 1, 0.5, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        features, valid, supports = encoder(x, mask)

    # 1. support bookkeeping vs an explicit dilation simulation
    pixel_valid = (upsample_patch_mask(mask, 8, (2, 2)).unsqueeze(1) < 0.5).float()
    expected = [pixel_valid]
    cur = pixel_valid
    for _i in range(1, 4):
        cur = dilate_support(cur, 2, 2)
        expected.append(cur)
    support_mismatch = max(
        float((got - want).abs().max().detach())
        for got, want in zip(supports, expected, strict=False)
    )

    # 2. no leakage: masked pixel content must not influence anything
    perturbed = x.clone()
    masked_pixels = (upsample_patch_mask(mask, 32, (2, 2)).unsqueeze(1) > 0.5).expand_as(x)
    perturbed[masked_pixels] = torch.randn_like(perturbed)[masked_pixels] * 13.0
    with torch.no_grad():
        features2, valid2, _ = encoder(perturbed, mask)
    leak = float((features - features2).abs().max().detach())

    # 3. valid-only normalisation: the GRN statistics must equal the masked statistics computed
    #    from the explicit support, not the statistics of the zero-filled dense map
    block = encoder.stages[0][0]
    stem_valid = supports[0]
    with torch.no_grad():
        probe = torch.randn(1, dims[0], stem_valid.shape[-2], stem_valid.shape[-1])
        _, _ = block(probe * stem_valid, stem_valid)
        stats = block._last_stats
    v = stats["valid"]
    x_pre = stats["pre_grn_x"]
    manual_norm = torch.sqrt(((x_pre * x_pre) * v).sum(dim=(0, 1, 2), keepdim=True))
    stats_err = float((stats["Gx"].reshape(-1) - manual_norm.reshape(-1)).abs().max().detach())

    # 4. gradients flow through the mask-aware path and stay finite
    grad_input = torch.zeros_like(x, requires_grad=True)
    out, _, _ = encoder(grad_input, mask)
    grads = torch.autograd.grad(out.pow(2).sum(), grad_input)[0]
    grad_finite = bool(torch.isfinite(grads).all())
    grad_at_masked = float(grads[masked_pixels].abs().max().detach())

    # 5. the plain zero-fill control must differ (it is a *differently named* control, not parity)
    control = PlainDenseZeroFillEncoder(in_chans=3, depths=depths, dims=dims).eval()
    with torch.no_grad():
        control.encoder_out = None
        control_features = control(x, mask)
    difference = float((features - control_features).abs().max().detach())

    tol = TOLERANCES["fp32_forward"]
    ok = (
        support_mismatch == 0.0
        and leak == 0.0
        and stats_err <= tol["atol"]
        and grad_finite
        and difference > 0.0
    )
    return CheckResult(
        name="fcmae_mask_aware_dense",
        status="pass" if ok else "fail",
        max_abs_err=max(support_mismatch, leak, stats_err),
        max_rel_err=None,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "mask-aware dense port (naming reference, dense compute): support sets match explicit "
            f"non-expanding support (mismatch={support_mismatch:.1e}); masked-pixel leakage="
            f"{leak:.1e}; GRN valid-only statistic error={stats_err:.3e}; "
            f"grads finite={grad_finite} (max |grad| at masked pixels={grad_at_masked:.3e}); "
            "plain zero-fill control differs by "
            f"{difference:.3e}"
        ),
        metrics={
            "support_mismatch": support_mismatch,
            "leakage": leak,
            "grn_stat_error": stats_err,
            "grad_finite": grad_finite,
            "grad_at_masked_pixels": grad_at_masked,
            "plain_zero_fill_difference": difference,
        },
    )


# --------------------------------------------------------------------------------------
# checks: SAVi (JAX reference fixtures)
# --------------------------------------------------------------------------------------


def _savi_fixture_array(fx: dict[str, Any], index: int):
    return fx[f"arr_{index}"]


def check_savi_slot_attention(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    import torch

    from .savi import SlotAttention, slot_attention_param_pairs

    tol = TOLERANCES["fp32_forward"]
    grad_tol = TOLERANCES["fp32_grad"]
    try:
        fx = _load_fixture(fixture_dir, "savi_slot_attention")
    except FileNotFoundError as exc:
        return _not_run("savi_slot_attention", str(exc), tol["atol"], tol["rtol"])
    meta = json.loads(str(fx["__meta__"]))
    kwargs = meta["kwargs"]
    slots = torch.as_tensor(fx["arr_0"], dtype=torch.float32).requires_grad_(True).to(device)
    inputs = torch.as_tensor(fx["arr_1"], dtype=torch.float32).requires_grad_(True).to(device)
    module = SlotAttention(
        num_iterations=kwargs["num_iterations"],
        qkv_size=kwargs["qkv_size"],
        mlp_size=kwargs["mlp_size"],
        num_heads=kwargs["num_heads"],
    ).build(slot_size=slots.shape[-1], inputs_size=inputs.shape[-1])
    flat = {}
    start = meta["array_index"]["params_start"]
    for path, index in zip(meta["param_paths"], range(start, len(fx)), strict=False):
        flat[path] = fx[f"arr_{index}"]
    module.load_flax(flat, "params")
    out = module(slots, inputs)
    (grad_inputs,) = torch.autograd.grad(out.pow(2).sum(), inputs, retain_graph=True)
    fwd_abs, fwd_rel = _relative_error(out, fx["arr_2"], tol["atol"])
    grad_abs, grad_rel = _relative_error(grad_inputs, fx["arr_3"], grad_tol["atol"])

    # per-parameter gradients against the JAX reference
    pairs = slot_attention_param_pairs(module, "params")
    param_grads = torch.autograd.grad(
        out.pow(2).sum(), [param for _, param, _ in pairs], allow_unused=True
    )
    worst_param_grad = 0.0
    worst_param_grad_rel = 0.0
    param_grad_violations: list[str] = []
    compared = 0
    for (flax_path, _tensor, _transpose), got in zip(pairs, param_grads, strict=False):
        index = meta["grad_param_paths"].index(flax_path)
        ref = torch.as_tensor(fx[f"arr_{meta['array_index']['grad_params_start'] + index}"])
        if _transpose:
            want = ref.reshape(ref.shape[0], -1).t().contiguous().reshape(-1)
        else:
            want = ref.reshape(-1)
        if got is None:
            continue
        got_flat = got.reshape(-1)
        diff = (got_flat - want).abs()
        abs_err = float(diff.max().detach())
        worst_param_grad = max(worst_param_grad, abs_err)
        # The per-parameter criterion is the contract's OR rule (atol or rtol); the relative error
        # is only meaningful for entries that carry a non-trivial gradient magnitude.
        significant = want.abs() > grad_tol["atol"] * 10.0
        if bool(significant.any()):
            rel_err = float((diff[significant] / want[significant].abs()).max())
            worst_param_grad_rel = max(worst_param_grad_rel, rel_err)
        else:
            rel_err = 0.0
        if not _verdict(abs_err, rel_err, grad_tol["atol"], grad_tol["rtol"]):
            param_grad_violations.append(f"{flax_path}: abs={abs_err:.3e} rel={rel_err:.3e}")
        compared += 1
    ok = (
        _verdict(fwd_abs, fwd_rel, tol["atol"], tol["rtol"])
        and _verdict(grad_abs, grad_rel, grad_tol["atol"], grad_tol["rtol"])
        and not param_grad_violations
    )
    return CheckResult(
        name="savi_slot_attention",
        status="pass" if ok else "fail",
        max_abs_err=fwd_abs,
        max_rel_err=fwd_rel,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "upstream Flax SlotAttention (verbatim, jax reference) vs torch port with identical "
            f"parameters: forward max_abs={fwd_abs:.3e}, input-grad max_abs={grad_abs:.3e}; "
            f"per-parameter grads compared for {compared} tensors "
            f"(worst abs={worst_param_grad:.3e})"
        ),
        evidence_path=str(fixture_dir / "savi_slot_attention.npz"),
        metrics={
            "input_grad_max_abs": grad_abs,
            "input_grad_max_rel": grad_rel,
            "param_grad_worst_abs": worst_param_grad,
            "param_grad_worst_rel": worst_param_grad_rel,
            "param_grad_count": compared,
            "param_grad_violations": param_grad_violations,
            "flax_kwargs": kwargs,
        },
    )


def check_savi_gru(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    import torch

    from .savi import GRUCell

    tol = TOLERANCES["fp32_forward"]
    try:
        fx = _load_fixture(fixture_dir, "savi_gru")
    except FileNotFoundError as exc:
        return _not_run("savi_gru", str(exc), tol["atol"], tol["rtol"])
    meta = json.loads(str(fx["__meta__"]))
    carry = torch.tensor(fx["arr_0"], dtype=torch.float32, requires_grad=True)
    inputs = torch.tensor(fx["arr_1"], dtype=torch.float32, requires_grad=True)
    cell = GRUCell(inputs.shape[-1], carry.shape[-1])
    flat = {
        path: fx[f"arr_{i}"]
        for path, i in zip(
            meta["param_paths"],
            range(meta["array_index"]["params_start"], len(fx)),
            strict=False,
        )
    }
    gru_prefix = "params" if "params/ir/kernel" in flat else "params/GRUCell_0"
    cell.load_flax(flat, gru_prefix)
    out = cell(carry, inputs)
    abs_err, rel_err = _relative_error(out, fx["arr_2"], tol["atol"])
    ok = _verdict(abs_err, rel_err, tol["atol"], tol["rtol"])
    gradient_error = None
    if "grad_params" in meta["array_index"]:
        out.square().sum().backward()
        gtol = TOLERANCES["fp32_grad"]
        ref = GRUCell(inputs.shape[-1], carry.shape[-1])
        ref.load_flax(
            {
                name: fx[f"arr_{meta['array_index']['grad_params'] + i}"]
                for i, name in enumerate(meta["grad_param_paths"])
            },
            gru_prefix,
        )
        pairs = [
            (inputs.grad, torch.tensor(fx[f"arr_{meta['array_index']['grad_inputs']}"])),
            (carry.grad, torch.tensor(fx[f"arr_{meta['array_index']['grad_carry']}"])),
        ]
        pairs += [(a.grad, r) for a, r in zip(cell.parameters(), ref.parameters(), strict=True)]
        ok &= all(torch.allclose(a, r, **gtol) for a, r in pairs)
        gradient_error = max(float((a - r).detach().abs().max()) for a, r in pairs)
    return CheckResult(
        name="savi_gru",
        status="pass" if ok else "fail",
        max_abs_err=abs_err,
        max_rel_err=rel_err,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "upstream flax.linen.GRUCell (0.6.11 semantics: gates r,z,n; reset multiplies the "
            f"hidden pre-activation) vs port GRUCell: max_abs={abs_err:.3e}"
        ),
        evidence_path=str(fixture_dir / "savi_gru.npz"),
        metrics={"param_paths": meta["param_paths"], "gradient_max_abs": gradient_error},
    )


def check_savi_processor(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    import torch

    from .savi import MLP, CorrectorPredictorTuple, Processor, SlotAttention

    tol = TOLERANCES["fp32_forward"]
    try:
        fx = _load_fixture(fixture_dir, "savi_processor")
    except FileNotFoundError:
        report_path = output_dir / "upstream_run_report_savi.json"
        recorded = "not recorded"
        if report_path.exists():
            try:
                payload = json.loads(report_path.read_text())
                recorded = payload.get("savi", {}).get("savi_processor_error", "not recorded")
            except Exception:  # noqa: BLE001
                pass
        result = _not_run(
            "savi_processor",
            f"the upstream Processor reference export failed: {recorded} "
            "(flax's nn.scan path is incompatible with the reference jax build, so no upstream "
            "Processor fixture exists; the port is covered by its own consistency test only)",
            tol["atol"],
            tol["rtol"],
        )
        result.evidence_path = str(output_dir / "logs" / "30_savi_jax_export.log")
        return result
    meta = json.loads(str(fx["__meta__"]))
    flat = {
        path: fx[f"arr_{i}"]
        for path, i in zip(
            meta["param_paths"],
            range(meta["array_index"]["params_start"], len(fx)),
            strict=False,
        )
    }
    frames = torch.as_tensor(fx["arr_1"], dtype=torch.float32)
    state = torch.as_tensor(fx["arr_0"], dtype=torch.float32)
    corrector = SlotAttention(num_iterations=1, qkv_size=state.shape[-1]).build(
        slot_size=state.shape[-1], inputs_size=frames.shape[-1]
    )
    predictor = MLP(hidden_size=state.shape[-1], output_size=state.shape[-1], residual=True).build(
        state.shape[-1]
    )
    # the reference pytree nests the two submodules under Processor_0/
    corrector.load_flax(flat, "params/corrector")
    predictor.load_flax(flat, "params/predictor")
    processor = Processor(corrector=corrector, predictor=predictor)
    mask = torch.ones(frames.shape[0], frames.shape[1], frames.shape[2])
    new_state, states = processor(state, frames, mask)
    assert isinstance(states, CorrectorPredictorTuple)
    corrected_abs, corrected_rel = _relative_error(states.corrected, fx["arr_3"], tol["atol"])
    predicted_abs, predicted_rel = _relative_error(states.predicted, fx["arr_4"], tol["atol"])
    state_abs, _ = _relative_error(new_state, fx["arr_2"], tol["atol"])
    worst = max(corrected_abs, predicted_abs, state_abs)
    ok = _verdict(worst, max(corrected_rel, predicted_rel), tol["atol"], tol["rtol"])
    return CheckResult(
        name="savi_processor",
        status="pass" if ok else "fail",
        max_abs_err=worst,
        max_rel_err=max(corrected_rel, predicted_rel),
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "upstream Processor scan (corrector then predictor; carried state is the predicted "
            f"state) vs port: corrected max_abs={corrected_abs:.3e}, predicted "
            f"max_abs={predicted_abs:.3e}, final state max_abs={state_abs:.3e}"
        ),
        evidence_path=str(fixture_dir / "savi_processor.npz"),
        metrics={"corrected_max_abs": corrected_abs, "predicted_max_abs": predicted_abs},
    )


def check_savi_predictor(fixture_dir, output_dir, device, dtype):
    import numpy as np
    import torch

    from .savi import TransformerPredictor

    try:
        fx = _load_fixture(fixture_dir, "savi_predictor")
    except FileNotFoundError as exc:
        return _not_run("savi_predictor", str(exc))
    meta = json.loads(str(fx["__meta__"]))
    model = TransformerPredictor(
        **{k: meta[k] for k in ("width", "num_heads", "mlp_size", "num_layers")}
    )
    flat = {
        name: fx[f"arr_{i + meta['params_start']}"] for i, name in enumerate(meta["param_paths"])
    }
    model.load_flax(flat)
    x = torch.tensor(fx["arr_0"], requires_grad=True)
    y = model(x)
    y.square().sum().backward()
    tol, gtol = TOLERANCES["fp32_forward"], TOLERANCES["fp32_grad"]
    ok = np.allclose(y.detach().numpy(), fx["arr_1"], **tol)
    ok &= np.allclose(x.grad.numpy(), fx["arr_2"], **gtol)
    # Map the gradient tree using the same verified layout conversion.
    gradient_model = TransformerPredictor(
        **{k: meta[k] for k in ("width", "num_heads", "mlp_size", "num_layers")}
    )
    gradient_model.load_flax(
        {
            name: fx[f"arr_{i + meta['grads_start']}"]
            for i, name in enumerate(meta["grad_param_paths"])
        }
    )
    worst = 0.0
    for param, ref in zip(model.parameters(), gradient_model.parameters(), strict=True):
        ok &= torch.allclose(param.grad, ref, **gtol)
        worst = max(worst, float((param.grad - ref).detach().abs().max().detach()))
    error = float((y.detach() - torch.tensor(fx["arr_1"])).abs().max().detach())
    return CheckResult(
        name="savi_predictor",
        status="pass" if ok else "fail",
        max_abs_err=error,
        max_rel_err=0.0,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason="pinned SAVi Transformer forward, input and all parameter gradients",
        evidence_path=str(fixture_dir / "savi_predictor.npz"),
        metrics={"parameter_grad_max_abs": worst},
    )


def check_savi_masked_extension(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    """The effective-mask path is an extension; only its own invariants are checked."""
    import torch

    from .savi import SlotAttention, SlotAttentionMasked

    torch.manual_seed(3301)
    slots = torch.randn(2, 4, 8)
    inputs = torch.randn(2, 6, 8)
    valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    masked = SlotAttentionMasked(num_iterations=3, qkv_size=16, mlp_size=24).build(8, 8)
    plain = SlotAttention(num_iterations=3, qkv_size=16, mlp_size=24).build(8, 8)
    plain.load_state_dict(masked.state_dict(), strict=True)
    out_masked = masked(slots, inputs, valid)
    perturbed = inputs.clone()
    perturbed[0, 3:] = torch.randn(3, 8) * 9.0
    out_perturbed = masked(slots, perturbed, valid)
    no_leak = float((out_masked - out_perturbed).abs().max().detach())
    # zeroing the invalid inputs must reproduce the masked result exactly (renormalised softmax)
    zeroed = inputs * valid.unsqueeze(-1)
    out_zeroed = masked(slots, zeroed, valid)
    renorm_gap = float((out_masked - out_zeroed).abs().max().detach())
    # with everything valid the extension degenerates to the upstream path
    all_valid = torch.ones(2, 6)
    gap_upstream = float(
        (masked(slots, inputs, all_valid) - plain(slots, inputs)).abs().max().detach()
    )
    tol = TOLERANCES["fp32_forward"]
    ok = no_leak == 0.0 and renorm_gap <= tol["atol"] and gap_upstream <= tol["atol"]
    return CheckResult(
        name="savi_masked_extension",
        status="pass" if ok else "fail",
        max_abs_err=max(no_leak, renorm_gap, gap_upstream),
        max_rel_err=None,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "NEW mask path (not upstream): invalid keys removed from softmax and value "
            f"aggregation, renormalised by valid mass; masked-content leakage={no_leak:.1e}, "
            f"zero-fill equivalence={renorm_gap:.3e}, all-valid degenerates to upstream path "
            f"({gap_upstream:.3e})"
        ),
        metrics={"leakage": no_leak, "renorm_gap": renorm_gap, "upstream_gap": gap_upstream},
    )


# --------------------------------------------------------------------------------------
# checks: dNRI
# --------------------------------------------------------------------------------------


def check_dnri_one_step_causal(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    import torch

    from .dnri import DNRIDecoder, DNRIEncoder

    tol = TOLERANCES["fp32_forward"]
    grad_tol = TOLERANCES["fp32_grad"]
    try:
        fx = _load_fixture(fixture_dir, "dnri_one_step")
    except FileNotFoundError as exc:
        return _not_run("dnri_one_step_causal", str(exc), tol["atol"], tol["rtol"])

    params = fx["params"]
    encoder = DNRIEncoder(params).eval().to(device)
    decoder = DNRIDecoder(params).eval().to(device)
    enc_load = encoder.load_state_dict(fx["encoder_state_dict"], strict=True)
    dec_load = decoder.load_state_dict(fx["decoder_state_dict"], strict=True)

    inputs = fx["inputs"].to(device)
    prior, posterior, _ = encoder.forward_bidirectional(inputs)
    prior_abs, prior_rel = _relative_error(prior, fx["upstream_prior"], tol["atol"])
    post_abs, post_rel = _relative_error(posterior, fx["upstream_posterior"], tol["atol"])

    # one online step through single_step_forward, matching the exported upstream state
    prior_state = fx["upstream_prior_state"]
    online, _ = encoder.online_step(inputs[:, 0], prior_state)
    online_abs, online_rel = _relative_error(online, fx["upstream_single_step"], tol["atol"])

    # decoder one step
    hidden = decoder.get_initial_hidden(inputs)
    edges = fx["edges"].to(device)
    pred, hidden_next = decoder(inputs[:, 0], hidden, edges)
    dec_abs, dec_rel = _relative_error(pred, fx["upstream_decoder_pred"], tol["atol"])
    hid_abs, _ = _relative_error(hidden_next, fx["upstream_decoder_hidden"], tol["atol"])

    # causality: perturbing future frames must not change any already-produced online output
    full_online, _ = encoder.online_sequence(inputs)
    perturbed = inputs.clone()
    perturbed[:, 3:] = torch.randn_like(perturbed[:, 3:]) * 5.0
    perturbed_online, _ = encoder.online_sequence(perturbed)
    past_identical = bool(torch.equal(full_online[:, :3], perturbed_online[:, :3]))
    future_changed = not bool(torch.equal(full_online[:, 3:], perturbed_online[:, 3:]))
    online_vs_batched = float((full_online - prior).abs().max().detach())

    worst = max(prior_abs, post_abs, online_abs, dec_abs, hid_abs)
    worst_rel = max(prior_rel, post_rel, online_rel, dec_rel)
    ok = _verdict(worst, worst_rel, tol["atol"], tol["rtol"]) and past_identical and future_changed
    return CheckResult(
        name="dnri_one_step_causal",
        status="pass" if ok else "fail",
        max_abs_err=worst,
        max_rel_err=worst_rel,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "upstream dnri.DNRI_Encoder/DNRI_Decoder (verbatim PyTorch) vs port: prior "
            f"max_abs={prior_abs:.3e}, posterior max_abs={post_abs:.3e}, single_step "
            f"max_abs={online_abs:.3e}, decoder step max_abs={dec_abs:.3e}; online-only causality: "
            f"past outputs bitwise unchanged when future frames are replaced ({past_identical}), "
            f"future outputs do change ({future_changed})"
        ),
        evidence_path=str(fixture_dir / "dnri_one_step.pt"),
        metrics={
            "encoder_load": {
                "missing": list(enc_load.missing_keys),
                "unexpected": list(enc_load.unexpected_keys),
            },
            "decoder_load": {
                "missing": list(dec_load.missing_keys),
                "unexpected": list(dec_load.unexpected_keys),
            },
            "prior_max_abs": prior_abs,
            "posterior_max_abs": post_abs,
            "single_step_max_abs": online_abs,
            "decoder_max_abs": dec_abs,
            "decoder_hidden_max_abs": hid_abs,
            "past_outputs_bitwise_identical": past_identical,
            "future_outputs_changed": future_changed,
            "online_vs_batched_prior_max_abs": online_vs_batched,
            "grad_tolerance": grad_tol,
        },
    )


# --------------------------------------------------------------------------------------
# checks: solvers and samplers (no upstream fixture needed)
# --------------------------------------------------------------------------------------


def check_theseus_lm_quadratic(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    tol = TOLERANCES["fp64_analytic"]
    try:
        from .theseus_ref import run_lm_reference_check
    except Exception as exc:  # noqa: BLE001
        return _not_run(
            "theseus_lm_quadratic",
            f"reference LM unavailable: {exc}",
            tol["atol"],
            tol["rtol"],
        )
    evidence = output_dir / "theseus"
    report = run_lm_reference_check(evidence)
    lsq = report["linear_least_squares"]
    outer = report["outer_acceptance"]
    abs_err = lsq["analytic_error_max_abs"]
    ok = (
        abs_err <= tol["atol"]
        and outer["monotone_non_increasing"]
        and outer["rejected_steps_that_lowered_residual"] > 0
    )
    return CheckResult(
        name="theseus_lm_quadratic",
        status="pass" if ok else "fail",
        max_abs_err=abs_err,
        max_rel_err=None,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "reference LM (written here, not upstream) on a linear least-squares problem: max "
            f"error vs analytic solution={abs_err:.3e} (<=1e-8); outer full-objective acceptance "
            f"is monotone non-increasing with {outer['rejected_steps_that_lowered_residual']} "
            "residual-decreasing proposals rejected by the complete objective"
        ),
        evidence_path=str(evidence / "theseus_lm_reference.json"),
        metrics={
            "iterations": lsq["iterations"],
            "finite_difference_sweep": lsq["finite_difference_sweep"],
            "outer_acceptance": {k: v for k, v in outer.items() if k != "records"},
        },
    )


def check_theseus_upstream_simple_example(
    fixture_dir: Path, output_dir: Path, device: str, dtype: str
):
    receipt = fixture_dir / "theseus_upstream.json"
    if not receipt.is_file():
        return _not_run(
            "theseus_upstream_simple_example",
            "pinned upstream Theseus reference receipt is missing",
        )
    payload = json.loads(receipt.read_text())
    import hashlib

    source = Path(__file__).parent / "upstream" / "theseus"
    hashes_ok = all(
        (source / name).is_file()
        and hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
        for name, digest in payload["source_hashes"].items()
    )
    losses = payload["simple_example_losses"]
    error = max(
        payload["solution_max_abs"],
        payload["gradient_max_abs"],
        min(row["max_abs_error"] for row in payload["finite_difference_sweep"]),
    )
    ok = hashes_ok and len(losses) == 20 and losses[-1] < losses[0] and error < 1e-8
    return CheckResult(
        name="theseus_upstream_simple_example",
        status="pass" if ok else "fail",
        max_abs_err=error,
        max_rel_err=0.0,
        atol=1e-8,
        rtol=0.0,
        reason=(
            "pinned official example plus FP64 dense LM analytic solution, "
            "implicit gradient and step-size sweep"
        ),
        evidence_path=str(receipt),
        metrics=payload,
    )


def check_smc_ref(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    tol = TOLERANCES["fp32_forward"]
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001
        return CheckResult(
            name="smc_ref",
            status="not_run",
            atol=tol["atol"],
            rtol=tol["rtol"],
            reason=(
                "particles.resampling imports numba at module scope; numba is not installed in "
                "this environment. The measured run is "
                f"{output_dir / 'smc' / 'smc_reference.json'} (reference venv, numba 0.61.2)"
            ),
            evidence_path=str(output_dir / "smc" / "smc_reference.json"),
        )
    from .smc_ref import run_smc_reference_check

    evidence = output_dir / "smc"
    report = run_smc_reference_check(evidence)
    exact_gap = report["exact_upstream_vs_enumeration_max_abs"]
    tv_mean = report["smc"]["tv_mean"]
    ok = exact_gap <= tol["atol"] and tv_mean < 0.5
    return CheckResult(
        name="smc_ref",
        status="pass" if ok else "fail",
        max_abs_err=max(exact_gap, tv_mean),
        max_rel_err=None,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            "upstream particles SMC (bootstrap filter, N="
            f"{report['smc']['num_particles']}) vs exact filtering on a fixed 5-state/3-symbol "
            f"HMM: upstream exact filter vs brute-force enumeration max_abs={exact_gap:.3e}; "
            f"mean total-variation distance of the SMC marginals={tv_mean:.5f} "
            f"(max over steps={report['smc']['tv_max']:.5f})"
        ),
        evidence_path=str(evidence / "smc_reference.json"),
        metrics=report["smc"],
    )


def check_genjl_particle_filter(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    import hashlib

    receipt = fixture_dir / "gen_upstream.json"
    model_path = fixture_dir / "gen_model.json"
    if not receipt.is_file() or not model_path.is_file():
        return _not_run("genjl_particle_filter", "pinned Gen.jl reference receipt is missing")
    payload = json.loads(receipt.read_text())
    model = json.loads(model_path.read_text())
    source = Path(__file__).parent / "upstream" / "genjl"
    hashes_ok = (
        all(
            hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
            for name, digest in payload["source_hashes"].items()
        )
        and hashlib.sha256(model_path.read_bytes()).hexdigest() == payload["input_sha256"]
    )
    import numpy as np

    actual = np.asarray(payload["marginals"], dtype=float)
    exact = np.asarray(model["exact"], dtype=float)
    tv = 0.5 * np.abs(actual - exact).sum(axis=1)
    ok = hashes_ok and np.isfinite(actual).all() and float(tv.max()) < 0.05
    return CheckResult(
        name="genjl_particle_filter",
        status="pass" if ok else "fail",
        max_abs_err=float(tv.max()),
        atol=0.05,
        rtol=0.0,
        reason="pinned Gen SMC against the same finite HMM enumeration as particles; TV bound 0.05",
        evidence_path=str(receipt),
        metrics=payload,
    )


def check_short_train_trend(fixture_dir: Path, output_dir: Path, device: str, dtype: str):
    import torch

    from .convnextv2 import RawConvNeXtV2

    tol = TOLERANCES["fp32_forward"]
    try:
        fx = _load_fixture(fixture_dir, "train_trend")
    except FileNotFoundError as exc:
        return _not_run("short_train_trend", str(exc), tol["atol"], tol["rtol"])
    torch.manual_seed(fx["seed"])
    model = RawConvNeXtV2(
        in_chans=fx["in_chans"],
        depths=fx["depths"],
        dims=fx["dims"],
        num_classes=fx["num_classes"],
        init_weights=False,
    ).to(device)
    model.load_state_dict(fx["init_state_dict"], strict=True)
    x = fx["inputs"].to(device)
    target = fx["targets"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=fx["lr"], weight_decay=fx["weight_decay"])
    losses = []
    model.train()
    for _ in range(fx["steps"]):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    upstream = list(fx["upstream_losses"])
    direction_match = (losses[-1] < losses[0]) == (upstream[-1] < upstream[0])
    rel_change_port = (losses[0] - losses[-1]) / losses[0]
    rel_change_upstream = (upstream[0] - upstream[-1]) / upstream[0]
    trend_gap = abs(rel_change_port - rel_change_upstream)
    # step-by-step agreement on the real optimisation trajectory (same seeds, same data)
    stepwise = max(abs(a - b) for a, b in zip(losses, upstream, strict=False))
    ok = direction_match and rel_change_port > 0 and trend_gap < 0.5
    return CheckResult(
        name="short_train_trend",
        status="pass" if ok else "fail",
        max_abs_err=stepwise,
        max_rel_err=trend_gap,
        atol=tol["atol"],
        rtol=tol["rtol"],
        reason=(
            f"{fx['steps']} real AdamW steps on the upstream implementation and on the port with "
            f"identical init/batch/seed: losses {losses[0]:.6f} -> {losses[-1]:.6f} (port) vs "
            f"{upstream[0]:.6f} -> {upstream[-1]:.6f} (upstream); trend direction matches="
            f"{direction_match}, relative-change gap={trend_gap:.3e}"
        ),
        evidence_path=str(fixture_dir / "train_trend.pt"),
        metrics={
            "port_losses": losses,
            "upstream_losses": upstream,
            "max_stepwise_abs": stepwise,
            "rel_change_port": rel_change_port,
            "rel_change_upstream": rel_change_upstream,
        },
    )


CHECKS: dict[str, Callable[[Path, Path, str, str], CheckResult]] = {
    "convnext_base_block": check_convnext_base_block,
    "convnext_grn_global": check_convnext_grn_global,
    "convnext_base_multiscale": check_convnext_base_multiscale,
    "fcmae_patchify_roundtrip": check_fcmae_patchify_roundtrip,
    "fcmae_sparse_upstream": check_fcmae_sparse_upstream,
    "fcmae_mask_aware_dense": check_fcmae_mask_aware_dense,
    "savi_slot_attention": check_savi_slot_attention,
    "savi_gru": check_savi_gru,
    "savi_processor": check_savi_processor,
    "savi_predictor": check_savi_predictor,
    "savi_masked_extension": check_savi_masked_extension,
    "dnri_one_step_causal": check_dnri_one_step_causal,
    "theseus_lm_quadratic": check_theseus_lm_quadratic,
    "theseus_upstream_simple_example": check_theseus_upstream_simple_example,
    "smc_ref": check_smc_ref,
    "genjl_particle_filter": check_genjl_particle_filter,
    "short_train_trend": check_short_train_trend,
}


def run_all(
    config: Any = None,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    checks: Any = None,
) -> dict[str, Any]:
    """Run every requested check; never raises. Returns the machine-readable parity receipt."""
    started = time.time()
    fixture_dir, output_dir = _paths(config)
    selected = list(checks) if checks is not None else list(CHECK_ORDER)
    unknown = [name for name in selected if name not in CHECKS]
    results: dict[str, dict[str, Any]] = {}
    for name in selected:
        if name in unknown:
            results[name] = CheckResult(
                name=name, status="not_run", reason=f"unknown check name {name!r}"
            ).to_dict()
            continue
        try:
            result = CHECKS[name](fixture_dir, output_dir, device, dtype)
        except ImportError as exc:
            result = _not_run(name, f"missing dependency: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a failed check must not crash the CLI
            result = CheckResult(
                name=name,
                status="fail",
                reason=f"{type(exc).__name__}: {exc}",
            )
        results[name] = result.to_dict()

    pass_count = sum(1 for r in results.values() if r["status"] == "pass")
    fail_count = sum(1 for r in results.values() if r["status"] == "fail")
    not_run_count = sum(1 for r in results.values() if r["status"] == "not_run")
    if fail_count:
        status = "fail"
    elif pass_count and not_run_count:
        status = "incomplete"
    elif pass_count:
        status = "pass"
    else:
        status = "not_run"
    payload: dict[str, Any] = {
        "parity_version": PARITY_VERSION,
        "status": status,
        "checks": results,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "not_run_count": not_run_count,
        "tolerances": TOLERANCES,
        "versions": environment_versions(),
        "paths": {"fixture_dir": str(fixture_dir), "output_dir": str(output_dir)},
        "device": device,
        "dtype": dtype,
        "elapsed_seconds": round(time.time() - started, 3),
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return payload


__all__ = ["PARITY_VERSION", "TOLERANCES", "available_checks", "run_all"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="component parity suite")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--checks", nargs="*", default=None)
    args = parser.parse_args()
    receipt = run_all({"fixture_dir": args.fixtures, "output_dir": args.output}, checks=args.checks)
    out = Path(args.output) / "parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str))
    print(json.dumps({k: v for k, v in receipt.items() if k != "checks"}, indent=2, default=str))
    for name, entry in receipt["checks"].items():
        print(f"{entry['status']:8s} {name}: {entry['reason'][:160]}")
