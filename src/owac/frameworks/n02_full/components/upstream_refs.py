"""Reference runner: executes *vendored upstream* modules verbatim and exports parity fixtures.

This module exports ConvNeXt, SAVi and dNRI fixtures. Dedicated upstream_sparse and
upstream_theseus runners execute the other pinned references.  It is meant to run inside the
isolated reference environment (``outputs/n02-full/p0/parity-venv``) where ``timm`` / ``jax`` are
installed; the project root environment cannot import ``timm`` and must not.

What is executed, and with which caveats:

* ``models/convnextv2.py`` and ``models/utils.py`` -- imported as-is.  ``models/utils.py`` imports
  ``MinkowskiEngine`` at module scope for its sparse layers, so a placeholder module object is
  inserted in ``sys.modules``; the dense ``LayerNorm``/``GRN``/``Block`` classes exercised here
  never touch it.  Nothing else is patched: the ``Block``/``ConvNeXtV2`` module objects are the
  upstream ones, and their parameters are the upstream parameters.
* ``models/fcmae.py`` -- module import needs ``MinkowskiEngine`` (placeholder, as above) and the
  class is instantiated without ``__init__`` (``object.__new__``) because ``__init__`` builds the
  sparse encoder.  The *methods* ``patchify``/``unpatchify``/``gen_random_mask``/``upsample_mask``
  are executed verbatim.
* ``models/convnextv2_sparse.py`` -- exported separately by ``upstream_sparse.py`` using the
  actual custom CUDA engine; this fixture runner does not execute it.

Fixtures are ``torch.save`` dictionaries containing the upstream parameters, inputs, outputs and
(when requested) gradients, plus provenance metadata.  ``parity_suite.py`` loads them and compares
against the local port, so the comparison code exists exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

UPSTREAM_ROOT = Path(__file__).resolve().parent / "upstream"

_UPSTREAM_PACKAGES = {
    "convnextv2": UPSTREAM_ROOT / "convnextv2",
    "dnri": UPSTREAM_ROOT / "dnri",
    "savi": UPSTREAM_ROOT / "savi",
}

# Modules taken from upstream at module scope but never executed in these reference runs.
_IMPORT_PLACEHOLDERS = ("MinkowskiEngine", "MinkowskiOps")


def ensure_upstream_path(package: str) -> Path:
    """Put one vendored upstream repo on ``sys.path`` (idempotent)."""
    root = _UPSTREAM_PACKAGES[package]
    if not root.is_dir():
        raise FileNotFoundError(f"vendored upstream tree missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def install_import_placeholders() -> list[str]:
    """Insert placeholder modules for Minkowski deps that cannot be installed.

    A placeholder is a module whose attribute lookup returns a *class* that raises when
    constructed.  This lets ``models/utils.py`` / ``models/fcmae.py`` be imported verbatim (their
    module-scope imports are the only place Minkowski names appear in the executed code paths)
    while any attempt to actually build a sparse operator fails loudly.  Returns the names that
    were newly inserted so the caller can record them as a caveat.
    """
    import types

    def _placeholder(name: str) -> type:
        def _raise(self, *args, **kwargs):
            raise RuntimeError(
                f"{name} is unavailable: MinkowskiEngine could not be installed in this environment"
            )

        return type(name, (), {"__init__": _raise})

    def _module_getattr(name: str):
        def __getattr__(attribute: str):
            if attribute.startswith("__") and attribute.endswith("__"):
                raise AttributeError(attribute)
            return _placeholder(f"{name}.{attribute}")

        return __getattr__

    inserted = []
    for name in _IMPORT_PLACEHOLDERS:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__getattr__ = _module_getattr(name)
            sys.modules[name] = module
            inserted.append(name)
    return inserted


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - recorded verbatim, never invented
        return f"unavailable: {type(exc).__name__}: {exc}"


def environment_manifest() -> dict[str, Any]:
    import numpy

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_head": git_revision(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for name in ("timm", "jax", "flax", "theseus", "MinkowskiEngine", "particles"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "installed(no __version__)")
        except Exception as exc:  # noqa: BLE001 - the failure itself is the recorded evidence
            info[name] = f"not_available: {type(exc).__name__}: {exc}"
    return info


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# check 1/2: ConvNeXt V2 block, GRN, multiscale
# --------------------------------------------------------------------------------------


def export_convnext_fixtures(out_dir: Path) -> dict[str, Any]:
    """Run upstream ConvNeXt V2 verbatim and dump fixtures for checks 1 and 2."""
    from owac.frameworks.n02_full.components import convnextv2 as port

    ensure_upstream_path("convnextv2")
    placeholders = install_import_placeholders()
    from models.convnextv2 import Block as UpBlock
    from models.convnextv2 import ConvNeXtV2 as UpConvNeXtV2
    from models.utils import GRN as UpGRN

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "environment": environment_manifest(),
        "import_placeholders": placeholders,
        "fixtures": {},
        "upstream_source_sha256": {
            "models/convnextv2.py": file_sha256(UPSTREAM_ROOT / "convnextv2/models/convnextv2.py"),
            "models/utils.py": file_sha256(UPSTREAM_ROOT / "convnextv2/models/utils.py"),
            "models/fcmae.py": file_sha256(UPSTREAM_ROOT / "convnextv2/models/fcmae.py"),
        },
    }

    def dump(name: str, payload: dict[str, Any]) -> None:
        payload["provenance"] = report["environment"]
        payload["upstream_file_sha256"] = report["upstream_source_sha256"]
        path = out_dir / f"{name}.pt"
        torch.save(payload, path)
        report["fixtures"][name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }

    # -- 1a: single upstream dense Block, non-square spatial input, gradients on input+params
    torch.manual_seed(1103)
    dim = 32
    up_block = UpBlock(dim=dim, drop_path=0.0).eval()
    port_block = port.Block(dim=dim, drop_path=0.0).eval()
    load_result = port_block.load_state_dict(up_block.state_dict(), strict=True)
    x = torch.randn(2, dim, 5, 7)
    x_up = x.clone().requires_grad_(True)
    x_port = x.clone().requires_grad_(True)
    y_up = up_block(x_up)
    y_port = port_block(x_port)
    (g_up,) = torch.autograd.grad(y_up.pow(2).sum(), x_up)
    (g_port,) = torch.autograd.grad(y_port.pow(2).sum(), x_port)
    dump(
        "convnext_block",
        {
            "state_dict": {k: v.detach().clone() for k, v in up_block.state_dict().items()},
            "input": x,
            "upstream_output": y_up.detach(),
            "upstream_input_grad": g_up.detach(),
            "port_output": y_port.detach(),
            "port_input_grad": g_port.detach(),
            "load_state_dict_result": {
                "strict": True,
                "missing": list(load_result.missing_keys),
                "unexpected": list(load_result.unexpected_keys),
            },
            "dim": dim,
        },
    )

    # -- 1b: GRN spatial coupling, measured on the upstream GRN itself
    torch.manual_seed(2207)
    up_grn = UpGRN(dim=8).eval()
    with torch.no_grad():
        up_grn.gamma.fill_(0.7)
        up_grn.beta.fill_(-0.2)
    gx = torch.randn(1, 3, 4, 8)
    gx2 = gx.clone()
    gx2[0, :, 0, 0] += 3.0
    with torch.no_grad():
        gy = up_grn(gx)
        gy2 = up_grn(gx2)
    dump(
        "convnext_grn",
        {
            "state_dict": {k: v.detach().clone() for k, v in up_grn.state_dict().items()},
            "input": gx,
            "input_perturbed": gx2,
            "upstream_output": gy.detach(),
            "upstream_output_perturbed": gy2.detach(),
            "far_pixel_delta": (gy2 - gy).abs().max().item(),
        },
    )

    # -- 1c/1d/2: full upstream network, 3-channel square and 4-channel rectangular
    cases = {
        "convnext_3ch_square": {"in_chans": 3, "shape": (1, 3, 32, 32)},
        "convnext_4ch_rect": {"in_chans": 4, "shape": (1, 4, 32, 48)},
    }
    for name, spec in cases.items():
        torch.manual_seed(3301)
        up_model = UpConvNeXtV2(
            in_chans=spec["in_chans"],
            num_classes=1000,
            depths=[2, 2, 2, 2],
            dims=[16, 32, 64, 128],
            drop_path_rate=0.0,
        ).eval()
        xs = torch.randn(*spec["shape"])
        stage_outputs = []
        h = xs
        for i in range(4):
            h = up_model.downsample_layers[i](h)
            h = up_model.stages[i](h)
            stage_outputs.append(h.detach())
        with torch.no_grad():
            pooled = up_model.forward_features(xs).detach()
            logits = up_model(xs).detach()
        dump(
            name,
            {
                "state_dict": {k: v.detach().clone() for k, v in up_model.state_dict().items()},
                "input": xs,
                "stage_outputs": stage_outputs,
                "pooled": pooled,
                "logits": logits,
                "in_chans": spec["in_chans"],
                "num_classes": 1000,
                "depths": [2, 2, 2, 2],
                "dims": [16, 32, 64, 128],
                "upstream_forward_feature_shapes": [tuple(t.shape) for t in stage_outputs],
            },
        )

    # -- 1e: upstream Base-dims single block (real 89M-scale widths, cheap single block)
    torch.manual_seed(4409)
    up_base_block = UpBlock(dim=128, drop_path=0.0).eval()
    xb = torch.randn(1, 128, 6, 6)
    with torch.no_grad():
        yb = up_base_block(xb)
    dump(
        "convnext_block_basewidth",
        {
            "state_dict": {k: v.detach().clone() for k, v in up_base_block.state_dict().items()},
            "input": xb,
            "upstream_output": yb,
            "dim": 128,
        },
    )
    return report


# --------------------------------------------------------------------------------------
# check 3: FCMAE patch/mask geometry (upstream method bodies, sparse encoder not run)
# --------------------------------------------------------------------------------------


def export_fcmae_fixtures(out_dir: Path) -> dict[str, Any]:
    ensure_upstream_path("convnextv2")
    placeholders = install_import_placeholders()
    from models.fcmae import FCMAE as UpFCMAE

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "environment": environment_manifest(),
        "import_placeholders": placeholders,
        "note": (
            "FCMAE.__init__ builds the Minkowski sparse encoder and is therefore never called; "
            "the upstream geometry methods are bound to a holder created with object.__new__ "
            "carrying only patch_size. No sparse operator was executed."
        ),
        "fixtures": {},
    }
    holder = object.__new__(UpFCMAE)

    def dump(name: str, payload: dict[str, Any]) -> None:
        payload["provenance"] = report["environment"]
        path = out_dir / f"{name}.pt"
        torch.save(payload, path)
        report["fixtures"][name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }

    # upstream 3-channel square case
    torch.manual_seed(5501)
    holder.patch_size = 32
    imgs = torch.randn(2, 3, 64, 64)
    patched = UpFCMAE.patchify(holder, imgs)
    restored = UpFCMAE.unpatchify(holder, patched)
    mask = UpFCMAE.gen_random_mask(holder, imgs, 0.6)
    upsampled = UpFCMAE.upsample_mask(holder, mask, 8)
    dump(
        "fcmae_geometry_3ch",
        {
            "patch_size": 32,
            "input": imgs,
            "upstream_patches": patched.detach(),
            "upstream_unpatchified": restored.detach(),
            "upstream_mask": mask.detach(),
            "upstream_mask_upsampled": upsampled.detach(),
        },
    )

    # upstream method bodies applied to the offsets needed by 240x320 / 4-channel inputs: the
    # upstream code cannot represent them (shape[1] and sqrt(L) are hard-coded), so we record the
    # real observed failure instead of a fabricated number.
    failures: dict[str, str] = {}
    for name, tensor, patch_size in (
        ("4ch_square", torch.randn(1, 4, 64, 64), 32),
        ("3ch_rect", torch.randn(1, 3, 240, 320), 32),
    ):
        holder.patch_size = patch_size
        try:
            UpFCMAE.patchify(holder, tensor)
            failures[name] = "no_error"
        except Exception as exc:  # noqa: BLE001 - the error text is the receipt
            failures[name] = f"{type(exc).__name__}: {exc}"
    holder.patch_size = 32
    holder.img_size = 240
    try:
        UpFCMAE.unpatchify(holder, torch.zeros(1, 60 * 80, 32 * 32 * 3))
        failures["rect_unpatchify"] = "no_error"
    except Exception as exc:  # noqa: BLE001
        failures["rect_unpatchify"] = f"{type(exc).__name__}: {exc}"
    report["upstream_geometry_failures"] = failures
    (out_dir / "fcmae_geometry_failures.json").write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------------------
# check 4: SAVi (JAX/Flax) reference export
# --------------------------------------------------------------------------------------


def _save_npz(path: Path, arrays: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    flattened: list[Any] = []
    for value in arrays.values():
        if isinstance(value, (list, tuple)):
            flattened.extend(value)
        else:
            flattened.append(value)
    payload = {f"arr_{i}": np.asarray(v) for i, v in enumerate(flattened)}
    payload["__arrays__"] = np.array(list(arrays.keys()))
    payload["__meta__"] = np.array(json.dumps(meta, default=str))
    np.savez_compressed(path, **payload)
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def patch_flax_grucell() -> dict[str, Any]:
    """Restore the removed ``nn.GRUCell()`` default so upstream ``misc.GRU`` can run.

    flax >= 0.7 made ``features`` a required field of ``linen.GRUCell`` while the 2022 upstream
    ``savi.modules.misc.GRU`` calls ``nn.GRUCell()`` with no argument.  The flax 0.10.4 cell body
    already derives ``hidden_features`` from the carry, so a subclass that only adds
    ``features: int | None = None`` is enough.  The patch is applied to the *library* inside the
    reference process; no upstream source is modified, and the executed cell math is flax's own.
    """
    import flax
    from flax import linen as nn

    try:
        nn.GRUCell()
    except TypeError as exc:

        class GRUCellCompat(nn.GRUCell):
            features: int | None = None

        # Flax derives submodule names from the class name; keep "GRUCell" so the exported
        # parameter paths match the upstream flax 0.6.11 layout (GRU_0/GRUCell_0/ir/kernel, ...).
        GRUCellCompat.__name__ = "GRUCell"
        nn.GRUCell = GRUCellCompat
        return {
            "applied": True,
            "flax_version": flax.__version__,
            "error_without_patch": str(exc),
            "note": "GRUCellCompat only restores the defaulted `features` field",
        }
    return {"applied": False, "flax_version": flax.__version__}


def export_savi_fixtures(out_dir: Path) -> dict[str, Any]:
    """Run the upstream SAVi modules verbatim under JAX and export inputs/params/outputs/grads.

    The upstream modules are imported unmodified.  ``flax`` must provide the era-matched
    ``nn.GRUCell`` (no ``features`` argument), which is why the reference environment pins
    ``flax==0.6.11``.
    """
    ensure_upstream_path("savi")
    import jax
    import jax.numpy as jnp
    from savi.modules import attention as up_attention
    from savi.modules import misc as up_misc
    from savi.modules import video as up_video

    from owac.frameworks.n02_full.components.savi import flatten_flax_params

    out_dir.mkdir(parents=True, exist_ok=True)
    grucell_patch = patch_flax_grucell()
    report: dict[str, Any] = {
        "environment": environment_manifest(),
        "flax_version": __import__("flax").__version__,
        "grucell_compat_patch": grucell_patch,
        "upstream_source_sha256": {
            "attention.py": file_sha256(UPSTREAM_ROOT / "savi/savi/modules/attention.py"),
            "video.py": file_sha256(UPSTREAM_ROOT / "savi/savi/modules/video.py"),
            "misc.py": file_sha256(UPSTREAM_ROOT / "savi/savi/modules/misc.py"),
        },
        "fixtures": {},
    }
    key = jax.random.PRNGKey(1103)

    def _grad_inputs(module, variables, *inputs, argnums=-1):
        def loss(*args):
            out = module.apply(variables, *args)
            return (out**2).sum()

        return jax.grad(loss, argnums=argnums)(*inputs)

    # -- A: SlotAttention with the MLP block, single head
    slots = jax.random.normal(jax.random.PRNGKey(2207), (2, 3, 8))
    inputs = jax.random.normal(jax.random.PRNGKey(3301), (2, 5, 8))
    sa = up_attention.SlotAttention(num_iterations=3, qkv_size=16, mlp_size=24, num_heads=1)
    variables = sa.init(key, slots, inputs)
    out = sa.apply(variables, slots, inputs)
    grad_in = _grad_inputs(sa, variables, slots, inputs)
    grad_params = jax.grad(lambda p: (sa.apply({"params": p}, slots, inputs) ** 2).sum())(
        variables["params"]
    )
    flat_params = flatten_flax_params(variables)
    flat_grads = flatten_flax_params({"params": grad_params})
    flat_grad_paths = list(flat_grads.keys())
    grad_arrays = list(flat_grads.values())
    report["fixtures"]["savi_slot_attention"] = _save_npz(
        out_dir / "savi_slot_attention.npz",
        {
            "slots": slots,
            "inputs": inputs,
            "output": out,
            "grad_inputs": grad_in,
            "grad_params": grad_arrays,
            "params": list(flat_params.values()),
        },
        {
            "case": "savi_slot_attention",
            "module": "savi.modules.attention.SlotAttention",
            "kwargs": {"num_iterations": 3, "qkv_size": 16, "mlp_size": 24, "num_heads": 1},
            "param_paths": list(flat_params.keys()),
            "grad_param_paths": flat_grad_paths,
            "loss": "sum(output**2)",
            "array_index": {
                "slots": 0,
                "inputs": 1,
                "output": 2,
                "grad_inputs": 3,
                "grad_params_start": 4,
                "params_start": 4 + len(grad_arrays),
            },
            "provenance": report["environment"],
        },
    )

    # -- B: two-head SlotAttention without the MLP block
    slots2 = jax.random.normal(jax.random.PRNGKey(4409), (2, 4, 8))
    inputs2 = jax.random.normal(jax.random.PRNGKey(5501), (2, 6, 8))
    sa2 = up_attention.SlotAttention(num_iterations=2, qkv_size=16, mlp_size=None, num_heads=2)
    v2 = sa2.init(key, slots2, inputs2)
    out2 = sa2.apply(v2, slots2, inputs2)
    flat2 = flatten_flax_params(v2)
    report["fixtures"]["savi_slot_attention_multhead"] = _save_npz(
        out_dir / "savi_slot_attention_multhead.npz",
        {
            "slots": slots2,
            "inputs": inputs2,
            "output": out2,
            "params": list(flat2.values()),
        },
        {
            "case": "savi_slot_attention_multhead",
            "module": "savi.modules.attention.SlotAttention",
            "kwargs": {"num_iterations": 2, "qkv_size": 16, "mlp_size": None, "num_heads": 2},
            "param_paths": list(flat2.keys()),
            "array_index": {"slots": 0, "inputs": 1, "output": 2, "params_start": 3},
            "provenance": report["environment"],
        },
    )

    # -- C: GRU cell (gate order / reset position)
    carry = jax.random.normal(jax.random.PRNGKey(6601), (2, 8))
    gru_in = jax.random.normal(jax.random.PRNGKey(7701), (2, 8))
    gru = up_misc.GRU()
    vg = gru.init(key, carry, gru_in)
    outg = gru.apply(vg, carry, gru_in)
    flatg = flatten_flax_params(vg)
    gg = jax.grad(lambda params: (gru.apply({"params": params}, carry, gru_in) ** 2).sum())(
        vg["params"]
    )
    flatgg = flatten_flax_params({"params": gg})
    gi = jax.grad(lambda inp: (gru.apply(vg, carry, inp) ** 2).sum())(gru_in)
    gc = jax.grad(lambda c: (gru.apply(vg, c, gru_in) ** 2).sum())(carry)
    report["fixtures"]["savi_gru"] = _save_npz(
        out_dir / "savi_gru.npz",
        {
            "carry": carry,
            "inputs": gru_in,
            "output": outg,
            "params": list(flatg.values()),
            "grad_inputs": gi,
            "grad_carry": gc,
            "grad_params": list(flatgg.values()),
        },
        {
            "case": "savi_gru",
            "module": "savi.modules.misc.GRU -> flax.linen.GRUCell",
            "param_paths": list(flatg.keys()),
            "array_index": {
                "carry": 0,
                "inputs": 1,
                "output": 2,
                "params_start": 3,
                "grad_inputs": 3 + len(flatg),
                "grad_carry": 4 + len(flatg),
                "grad_params": 5 + len(flatg),
            },
            "grad_param_paths": list(flatgg),
            "provenance": report["environment"],
        },
    )

    # -- D: MLP with pre-LayerNorm and residual
    mlp = up_misc.MLP(hidden_size=24, layernorm="pre", residual=True)
    vm = mlp.init(key, slots)
    outm = mlp.apply(vm, slots)
    flatm = flatten_flax_params(vm)
    report["fixtures"]["savi_mlp"] = _save_npz(
        out_dir / "savi_mlp.npz",
        {"inputs": slots, "output": outm, "params": list(flatm.values())},
        {
            "case": "savi_mlp",
            "module": "savi.modules.misc.MLP",
            "kwargs": {"hidden_size": 24, "layernorm": "pre", "residual": True},
            "param_paths": list(flatm.keys()),
            "array_index": {"inputs": 0, "output": 1, "params_start": 2},
            "provenance": report["environment"],
        },
    )

    # Original self-attention predictor, including parameter and input gradients.
    predictor = up_attention.Transformer(num_heads=2, qkv_size=8, mlp_size=16, num_layers=2)
    pv = predictor.init(key, slots)
    po = predictor.apply(pv, slots)
    pg = jax.grad(lambda params: (predictor.apply({"params": params}, slots) ** 2).sum())(
        pv["params"]
    )
    pi = jax.grad(lambda x: (predictor.apply(pv, x) ** 2).sum())(slots)
    pp, pgrad = flatten_flax_params(pv), flatten_flax_params({"params": pg})
    report["fixtures"]["savi_predictor"] = _save_npz(
        out_dir / "savi_predictor.npz",
        {
            "inputs": slots,
            "output": po,
            "grad_inputs": pi,
            "params": list(pp.values()),
            "grad_params": list(pgrad.values()),
        },
        {
            "param_paths": list(pp),
            "grad_param_paths": list(pgrad),
            "params_start": 3,
            "grads_start": 3 + len(pp),
            "width": 8,
            "num_heads": 2,
            "mlp_size": 16,
            "num_layers": 2,
            "provenance": report["environment"],
        },
    )

    # -- E: Processor (corrector/predictor scan order).  flax 0.10.4's nn.scan uses
    # jax.api_util.debug_info, which the reference jax removed; the failure is recorded
    # verbatim instead of fabricating a number.
    try:
        # -- E: Processor (corrector/predictor scan order)
        proc = up_video.Processor(
            corrector=up_attention.SlotAttention(num_iterations=1, qkv_size=8),
            predictor=up_misc.MLP(hidden_size=8, residual=True),
        )
        state0 = jax.random.normal(jax.random.PRNGKey(8801), (2, 3, 8))
        frames = jax.random.normal(jax.random.PRNGKey(9901), (2, 4, 5, 8))
        mask = jnp.ones((2, 4, 5), jnp.int32)
        vp = proc.init(key, state0, frames, mask, False)
        new_state, states = proc.apply(vp, state0, frames, mask, False)
        flatp = flatten_flax_params(vp)
        report["fixtures"]["savi_processor"] = _save_npz(
            out_dir / "savi_processor.npz",
            {
                "state": state0,
                "frames": frames,
                "new_state": new_state,
                "corrected": states.corrected,
                "predicted": states.predicted,
                "params": list(flatp.values()),
            },
            {
                "case": "savi_processor",
                "module": "savi.modules.video.Processor",
                "param_paths": list(flatp.keys()),
                "array_index": {
                    "state": 0,
                    "frames": 1,
                    "new_state": 2,
                    "corrected": 3,
                    "predicted": 4,
                    "params_start": 5,
                },
                "provenance": report["environment"],
            },
        )

    except Exception as exc:  # noqa: BLE001 - recorded evidence
        report["savi_processor_error"] = f"{type(exc).__name__}: {exc}"
    return report


# --------------------------------------------------------------------------------------
# check 5: dNRI (upstream PyTorch, run verbatim in the reference environment)
# --------------------------------------------------------------------------------------


DNRI_PARAMS = {
    "num_vars": 4,
    "num_edge_types": 2,
    "input_size": 3,
    "encoder_hidden": 16,
    "encoder_rnn_hidden": 16,
    "encoder_rnn_type": "lstm",
    "encoder_dropout": 0.0,
    "encoder_mlp_num_layers": 1,
    "prior_num_layers": 1,
    "decoder_hidden": 16,
    "decoder_dropout": 0.0,
    "skip_first": True,
    "gpu": False,
}


def export_dnri_fixtures(out_dir: Path) -> dict[str, Any]:
    """Run the upstream dNRI encoder/decoder verbatim and export one-step reference numbers."""
    ensure_upstream_path("dnri")
    from dnri.models.dnri import DNRI_Decoder as UpDecoder
    from dnri.models.dnri import DNRI_Encoder as UpEncoder

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "environment": environment_manifest(),
        "upstream_source_sha256": {
            "dnri/models/dnri.py": file_sha256(UPSTREAM_ROOT / "dnri/dnri/models/dnri.py"),
            "dnri/models/model_utils.py": file_sha256(
                UPSTREAM_ROOT / "dnri/dnri/models/model_utils.py"
            ),
        },
        "fixtures": {},
    }
    torch.manual_seed(1103)
    encoder = UpEncoder(DNRI_PARAMS).eval()
    decoder = UpDecoder(DNRI_PARAMS).eval()

    inputs = torch.randn(2, 5, DNRI_PARAMS["num_vars"], DNRI_PARAMS["input_size"])
    with torch.no_grad():
        prior, posterior, prior_state = encoder(inputs)
        single_step, _ = encoder.single_step_forward(inputs[:, 0], prior_state)
        hidden = decoder.get_initial_hidden(inputs)
        edges = torch.zeros(2, 12, DNRI_PARAMS["num_edge_types"])
        edges[..., 0] = 1.0
        decoder_pred, decoder_hidden = decoder(inputs[:, 0], hidden, edges)
    payload = {
        "params": DNRI_PARAMS,
        "inputs": inputs,
        "upstream_prior": prior,
        "upstream_posterior": posterior,
        "upstream_prior_state": prior_state,
        "upstream_single_step": single_step,
        "edges": edges,
        "upstream_decoder_pred": decoder_pred,
        "upstream_decoder_hidden": decoder_hidden,
        "encoder_state_dict": {k: v.clone() for k, v in encoder.state_dict().items()},
        "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
        "provenance": report["environment"],
    }
    path = out_dir / "dnri_one_step.pt"
    torch.save(payload, path)
    report["fixtures"]["dnri_one_step"] = {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    return report


# --------------------------------------------------------------------------------------
# check 9: short real training trend (upstream implementation vs port)
# --------------------------------------------------------------------------------------


TRAIN_TREND = {
    "steps": 20,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "seed": 1103,
    "in_chans": 3,
    "depths": [2, 2, 2, 2],
    "dims": [16, 32, 64, 128],
    "num_classes": 4,
    "batch_size": 2,
    "img_shape": [32, 32],
}


def export_train_trend_fixture(out_dir: Path) -> dict[str, Any]:
    """20 real AdamW steps on the upstream ConvNeXtV2, plus the exact init/data for the port."""
    ensure_upstream_path("convnextv2")
    install_import_placeholders()
    from models.convnextv2 import ConvNeXtV2 as UpConvNeXtV2

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TRAIN_TREND
    torch.manual_seed(cfg["seed"])
    model = UpConvNeXtV2(
        in_chans=cfg["in_chans"],
        num_classes=cfg["num_classes"],
        depths=cfg["depths"],
        dims=cfg["dims"],
        drop_path_rate=0.0,
    )
    inputs = torch.randn(cfg["batch_size"], cfg["in_chans"], *cfg["img_shape"])
    targets = torch.randn(cfg["batch_size"], cfg["num_classes"])
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    losses = []
    model.train()
    for _ in range(cfg["steps"]):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

    path = out_dir / "train_trend.pt"
    torch.save(
        {
            "config": cfg,
            "init_state_dict": init_state,
            "inputs": inputs,
            "targets": targets,
            "upstream_losses": losses,
            "steps": cfg["steps"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "seed": cfg["seed"],
            "in_chans": cfg["in_chans"],
            "depths": cfg["depths"],
            "dims": cfg["dims"],
            "num_classes": cfg["num_classes"],
            "provenance": environment_manifest(),
        },
        path,
    )
    return {
        "environment": environment_manifest(),
        "fixtures": {
            "train_trend": {
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        },
        "upstream_losses": losses,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export upstream parity fixtures")
    parser.add_argument(
        "--check",
        default="all",
        choices=["all", "convnext", "fcmae", "savi", "dnri", "trend"],
    )
    parser.add_argument("--out", default="outputs/n02-full/p0/parity/fixtures")
    parser.add_argument("--report", default="outputs/n02-full/p0/parity/upstream_run_report.json")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    report: dict[str, Any] = {}
    if args.check in ("all", "convnext"):
        report["convnext"] = export_convnext_fixtures(out_dir)
    if args.check in ("all", "fcmae"):
        report["fcmae"] = export_fcmae_fixtures(out_dir)
    if args.check in ("all", "savi"):
        report["savi"] = export_savi_fixtures(out_dir)
    if args.check in ("all", "dnri"):
        report["dnri"] = export_dnri_fixtures(out_dir)
    if args.check in ("all", "trend"):
        report["train_trend"] = export_train_trend_fixture(out_dir)
    report["command"] = " ".join(
        [
            os.path.basename(sys.executable),
            "-m",
            "owac.frameworks.n02_full.components.upstream_refs",
            "--check",
            args.check,
            "--out",
            args.out,
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps(
            {k: (v.get("fixtures") if isinstance(v, dict) else v) for k, v in report.items()},
            indent=2,
            default=str,
        )[:4000]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
