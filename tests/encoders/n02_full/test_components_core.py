"""Direct numerical tests for the N02 full component ports (root environment).

These tests assert real numbers produced by the local implementation.  Checks whose reference
numbers come from the isolated reference environment
(``outputs/n02-full/p0/parity-venv``) load the exported fixtures when they exist and are skipped
with an explicit reason otherwise; the upstream reference exports themselves are documented in
``outputs/n02-full/p0/parity/`` and re-verified by ``test_components_parity.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from owac.frameworks.n02_full.components.convnextv2 import (
    GRN,
    Block,
    RawConvNeXtV2,
    raw_convnextv2,
)
from owac.frameworks.n02_full.components.dnri import (
    DEFAULT_PARAMS,
    DNRIDecoder,
    DNRIEncoder,
)
from owac.frameworks.n02_full.components.fcmae import (
    MaskAwareDenseConvNeXtV2Encoder,
    PlainDenseZeroFillEncoder,
    RawFCMAE,
    dilate_support,
    gen_random_mask,
    patchify,
    unpatchify,
    upsample_patch_mask,
)
from owac.frameworks.n02_full.components.profile_targets import (
    build_profile_targets,
    profile_target_specs,
)
from owac.frameworks.n02_full.components.savi import SlotAttention, SlotAttentionMasked
from owac.frameworks.n02_full.components.theseus_ref import (
    LMConfig,
    fixed_structure_outer_loop,
    linear_least_squares_problem,
    lm_minimize,
    numerical_jacobian,
)

FIXTURE_DIR = Path("outputs/n02-full/p0/parity/fixtures")

FP32_ATOL = 1e-5
FP32_RTOL = 1e-4


def _fixture(name: str):
    path = FIXTURE_DIR / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"reference fixture {path} not exported (run the reference-environment export)")
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------------------
# ConvNeXt V2
# --------------------------------------------------------------------------------------


def test_convnext_block_matches_upstream_block_fixture():
    fx = _fixture("convnext_block")
    block = Block(dim=fx["dim"], drop_path=0.0).eval()
    block.load_state_dict(fx["state_dict"], strict=True)
    x = fx["input"].clone().requires_grad_(True)
    y = block(x)
    (grad,) = torch.autograd.grad(y.pow(2).sum(), x)
    assert torch.allclose(y, fx["upstream_output"], atol=FP32_ATOL, rtol=FP32_RTOL)
    assert torch.allclose(grad, fx["upstream_input_grad"], atol=FP32_ATOL, rtol=1e-3)


def test_convnext_full_model_matches_upstream_for_3ch_and_4ch_rect():
    for fixture_name in ("convnext_3ch_square", "convnext_4ch_rect"):
        fx = _fixture(fixture_name)
        model = RawConvNeXtV2(
            in_chans=fx["in_chans"],
            depths=fx["depths"],
            dims=fx["dims"],
            num_classes=fx["num_classes"],
        ).eval()
        model.load_state_dict(fx["state_dict"], strict=True)
        with torch.no_grad():
            multiscale = model.forward_multiscale(fx["input"])
            pooled = model.forward_stagewise(fx["input"])["pooled"]
        for i, want in enumerate(fx["stage_outputs"]):
            got = multiscale[f"stage{i}_stride{4 * 2**i}"]
            assert got.shape == want.shape
            assert torch.allclose(got, want, atol=FP32_ATOL, rtol=FP32_RTOL)
        assert torch.allclose(pooled, fx["pooled"], atol=FP32_ATOL, rtol=FP32_RTOL)


def test_convnext_multiscale_shapes_and_stride8_16_contract():
    model = raw_convnextv2("base", in_chans=4, num_classes=0)
    model.eval()
    with torch.no_grad():
        features = model.forward_multiscale(torch.randn(1, 4, 256, 256))
    assert features["stage1_stride8"].shape == (1, 256, 32, 32)
    assert features["stage2_stride16"].shape == (1, 512, 16, 16)
    selected = model.forward_features(torch.randn(1, 4, 256, 256))
    assert [tuple(t.shape) for t in selected] == [(1, 256, 32, 32), (1, 512, 16, 16)]


def test_convnext_base_parameter_count():
    model = raw_convnextv2("base", in_chans=4, num_classes=0)
    assert sum(p.numel() for p in model.parameters()) == 87_694_848
    model3 = raw_convnextv2("base", in_chans=3, num_classes=0)
    assert sum(p.numel() for p in model3.parameters()) == 87_692_800


def test_grn_has_global_spatial_support():
    torch.manual_seed(1103)
    grn = GRN(8).eval()
    with torch.no_grad():
        grn.gamma.fill_(0.7)
        grn.beta.fill_(-0.2)
    x = torch.randn(1, 3, 4, 8)
    perturbed = x.clone()
    perturbed[0, :, 0, 0] += 3.0
    with torch.no_grad():
        y = grn(x)
        y2 = grn(perturbed)
    # a change in the top-left corner must change every output position, including the far corner
    delta = (y2 - y).abs()
    assert delta.min() > 0
    assert delta[0, :, -1, -1].max() > 0


def test_grn_matches_upstream_fixture_and_far_pixel_effect():
    fx = _fixture("convnext_grn")
    grn = GRN(dim=fx["input"].shape[-1]).eval()
    grn.load_state_dict(fx["state_dict"], strict=True)
    with torch.no_grad():
        y = grn(fx["input"])
        y2 = grn(fx["input_perturbed"])
    assert torch.allclose(y, fx["upstream_output"], atol=FP32_ATOL, rtol=FP32_RTOL)
    assert torch.allclose(y2, fx["upstream_output_perturbed"], atol=FP32_ATOL, rtol=FP32_RTOL)
    assert float((y2 - y).abs().max()) == pytest.approx(fx["far_pixel_delta"], abs=1e-6)


# --------------------------------------------------------------------------------------
# FCMAE geometry, masks and the mask-aware dense encoder
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chans", "height", "width", "patch"),
    [
        (3, 64, 64, 32),
        (4, 64, 64, 32),
        (3, 240, 320, 16),
        (4, 240, 320, 16),
        (3, 256, 320, 32),
        (4, 256, 320, 32),
    ],
)
def test_patchify_unpatchify_roundtrip_is_exact(chans, height, width, patch):
    x = torch.randn(2, chans, height, width)
    patches = patchify(x, patch)
    assert patches.shape == (2, (height // patch) * (width // patch), patch * patch * chans)
    restored = unpatchify(patches, patch, chans, (height // patch, width // patch))
    assert torch.equal(restored, x)


def test_patchify_rejects_non_divisible_rectangular_input():
    x = torch.randn(1, 3, 240, 320)
    with pytest.raises(ValueError):
        patchify(x, 32)


def test_patchify_matches_upstream_3ch_square_fixture():
    fx = _fixture("fcmae_geometry_3ch")
    patches = patchify(fx["input"], fx["patch_size"])
    assert torch.equal(patches, fx["upstream_patches"])
    restored = unpatchify(patches, fx["patch_size"], 3, (2, 2))
    assert torch.equal(restored, fx["upstream_unpatchified"])
    mask = gen_random_mask((2, 2), 2, 0.6, generator=torch.Generator().manual_seed(0))
    assert mask.shape == fx["upstream_mask"].shape


def test_random_mask_keeps_exact_count_and_is_data_independent():
    grid = (15, 20)
    length = grid[0] * grid[1]
    for ratio in (0.0, 0.25, 0.6, 0.99):
        mask = gen_random_mask(grid, 4, ratio, generator=torch.Generator().manual_seed(3))
        assert torch.all(mask.sum(dim=1) == int(length * ratio))
    upsampled = upsample_patch_mask(mask, 8, grid)
    assert upsampled.shape == (4, grid[0] * 8, grid[1] * 8)
    # each pixel belongs to exactly one patch block
    assert set(upsampled.unique().tolist()) <= {0.0, 1.0}


def test_fcmae_rectangular_multichannel_forward_runs():
    model = RawFCMAE(
        img_size=(256, 320),
        in_chans=4,
        depths=(1, 1, 2, 1),
        dims=(8, 16, 32, 64),
        decoder_embed_dim=16,
    )
    loss, pred, mask = model(torch.randn(2, 4, 256, 320), 0.6)
    assert pred.shape == (2, 32 * 32 * 4, 8, 10)
    assert mask.shape == (2, 80)
    assert float(mask.mean()) == pytest.approx(0.6, abs=1e-6)
    assert torch.isfinite(loss)


def test_dilate_support_matches_explicit_coordinate_dilation():
    valid = torch.zeros(1, 1, 9, 9)
    valid[0, 0, 4, 4] = 1.0
    assert float(dilate_support(valid, 7).sum()) == 49.0
    assert float(dilate_support(valid, 2, 2).sum()) == 1.0
    valid[0, 0, 0, 0] = 1.0
    # the corner pixel contributes a clipped 4x4 block which overlaps the centre block in 3x3;
    # the union is computed independently here rather than hard-coded
    expected = set()
    for row, col in ((4, 4), (0, 0)):
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                r, c = row + dy, col + dx
                if 0 <= r < 9 and 0 <= c < 9:
                    expected.add((r, c))
    assert float(dilate_support(valid, 7).sum()) == len(expected)


def test_mask_aware_encoder_support_leakage_and_gradients():
    torch.manual_seed(2207)
    depths, dims = (1, 1, 1, 1), (8, 16, 32, 64)
    encoder = MaskAwareDenseConvNeXtV2Encoder(
        in_chans=3, depths=depths, dims=dims, patch_size=32
    ).eval()
    x = torch.randn(1, 3, 64, 64)
    mask = gen_random_mask((2, 2), 1, 0.5, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        features, valid, supports = encoder(x, mask)

    # support sets equal the explicit dilation sequence
    expected = [(upsample_patch_mask(mask, 8, (2, 2)).unsqueeze(1) < 0.5).float()]
    cur = expected[0]
    for _i in range(1, 4):
        cur = dilate_support(cur, 2, 2)
        expected.append(cur)
    for got, want in zip(supports, expected, strict=False):
        assert torch.equal(got, want)

    # masked pixel content cannot influence the output at all
    perturbed = x.clone()
    pixels = (upsample_patch_mask(mask, 32, (2, 2)).unsqueeze(1) > 0.5).expand_as(x)
    perturbed[pixels] = torch.randn_like(perturbed)[pixels] * 13.0
    with torch.no_grad():
        features2, _, _ = encoder(perturbed, mask)
    assert torch.equal(features, features2)

    # gradients are finite, including at all-zero inputs (upstream norm semantics)
    for label, base in (("random", x), ("zeros", torch.zeros_like(x))):
        probe = base.clone().requires_grad_(True)
        out, _, _ = encoder(probe, mask)
        (grad,) = torch.autograd.grad(out.pow(2).sum(), probe)
        assert torch.isfinite(grad).all(), label
    assert float(grad.abs().max()) >= 0.0


def test_plain_dense_zero_fill_control_differs_from_mask_aware_port():
    torch.manual_seed(3301)
    depths, dims = (1, 1, 1, 1), (8, 16, 32, 64)
    x = torch.randn(1, 3, 64, 64)
    mask = gen_random_mask((2, 2), 1, 0.5, generator=torch.Generator().manual_seed(5))
    aware = MaskAwareDenseConvNeXtV2Encoder(
        in_chans=3, depths=depths, dims=dims, patch_size=32
    ).eval()
    control = PlainDenseZeroFillEncoder(in_chans=3, depths=depths, dims=dims).eval()
    with torch.no_grad():
        aware_features, _, _ = aware(x, mask)
        control_features = control(x, mask)
    assert aware_features.shape == control_features.shape
    assert not torch.allclose(aware_features, control_features, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------------------
# SAVi
# --------------------------------------------------------------------------------------


def test_savi_masked_attention_blocks_invalid_keys_and_normalises():
    torch.manual_seed(3301)
    slots = torch.randn(2, 4, 8)
    inputs = torch.randn(2, 6, 8)
    valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    module = SlotAttentionMasked(num_iterations=2, qkv_size=16, mlp_size=16).build(8, 8)
    out = module(slots, inputs, valid)
    perturbed = inputs.clone()
    perturbed[0, 3:] = torch.randn(3, 8) * 9.0
    assert torch.equal(out, module(slots, perturbed, valid))
    zeroed = inputs * valid.unsqueeze(-1)
    assert torch.allclose(out, module(slots, zeroed, valid), atol=FP32_ATOL)


def test_savi_unmasked_path_degenerates_to_upstream_semantics():
    torch.manual_seed(4409)
    slots = torch.randn(2, 3, 8)
    inputs = torch.randn(2, 5, 8)
    masked = SlotAttentionMasked(num_iterations=3, qkv_size=16, mlp_size=24).build(8, 8)
    plain = SlotAttention(num_iterations=3, qkv_size=16, mlp_size=24).build(8, 8)
    plain.load_state_dict(masked.state_dict(), strict=True)
    all_valid = torch.ones(2, 5)
    assert torch.allclose(masked(slots, inputs, all_valid), plain(slots, inputs), atol=FP32_ATOL)


def test_savi_slot_attention_matches_jax_reference_fixture():
    path = FIXTURE_DIR / "savi_slot_attention.npz"
    if not path.exists():
        pytest.skip(f"JAX reference fixture {path} not exported")
    import json

    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    kwargs = meta["kwargs"]
    slots = torch.as_tensor(z["arr_0"]).requires_grad_(True)
    inputs = torch.as_tensor(z["arr_1"]).requires_grad_(True)
    module = SlotAttention(
        num_iterations=kwargs["num_iterations"],
        qkv_size=kwargs["qkv_size"],
        mlp_size=kwargs["mlp_size"],
        num_heads=kwargs["num_heads"],
    ).build(slots.shape[-1], inputs.shape[-1])
    start = meta["array_index"]["params_start"]
    flat = {
        p: z[f"arr_{i}"]
        for p, i in zip(meta["param_paths"], range(start, len(z.files)), strict=False)
    }
    module.load_flax(flat, "params")
    out = module(slots, inputs)
    (grad,) = torch.autograd.grad(out.pow(2).sum(), inputs)
    assert torch.allclose(out, torch.as_tensor(z["arr_2"]), atol=FP32_ATOL, rtol=FP32_RTOL)
    assert torch.allclose(grad, torch.as_tensor(z["arr_3"]), atol=FP32_ATOL, rtol=1e-3)


# --------------------------------------------------------------------------------------
# dNRI
# --------------------------------------------------------------------------------------


def _dnri_models():
    params = dict(DEFAULT_PARAMS)
    return DNRIEncoder(params).eval(), DNRIDecoder(params).eval()


def test_dnri_online_sequence_uses_prior_only_and_is_causal():
    encoder, _ = _dnri_models()
    torch.manual_seed(1103)
    inputs = torch.randn(2, 5, DEFAULT_PARAMS["num_vars"], DEFAULT_PARAMS["input_size"])
    priors, _ = encoder.online_sequence(inputs)
    prior_batched, _, _ = encoder.forward_bidirectional(inputs)
    assert torch.allclose(priors, prior_batched, atol=1e-6, rtol=1e-5)

    perturbed = inputs.clone()
    perturbed[:, 3:] = torch.randn_like(perturbed[:, 3:]) * 5.0
    priors_perturbed, _ = encoder.online_sequence(perturbed)
    # bitwise: already-produced online outputs must not change when future frames change
    assert torch.equal(priors[:, :3], priors_perturbed[:, :3])
    assert not torch.equal(priors[:, 3:], priors_perturbed[:, 3:])


def test_dnri_decoder_one_step_shapes_and_gradients():
    encoder, decoder = _dnri_models()
    torch.manual_seed(2207)
    inputs = torch.randn(2, 4, DEFAULT_PARAMS["num_vars"], DEFAULT_PARAMS["input_size"])
    hidden = decoder.get_initial_hidden(inputs)
    edges = torch.zeros(2, 12, DEFAULT_PARAMS["num_edge_types"])
    edges[..., 0] = 1.0
    pred, hidden_next = decoder(inputs[:, 0], hidden, edges)
    assert pred.shape == inputs[:, 0].shape
    assert hidden_next.shape == hidden.shape
    loss = (pred**2).sum() + (encoder.online_sequence(inputs)[0] ** 2).sum()
    loss.backward()
    assert torch.isfinite(encoder.mlp1.model[0].weight.grad).all()


def test_dnri_matches_upstream_fixture():
    fx = _fixture("dnri_one_step")
    params = fx["params"]
    encoder = DNRIEncoder(params).eval()
    decoder = DNRIDecoder(params).eval()
    encoder.load_state_dict(fx["encoder_state_dict"], strict=True)
    decoder.load_state_dict(fx["decoder_state_dict"], strict=True)
    inputs = fx["inputs"]
    prior, posterior, prior_state = encoder.forward_bidirectional(inputs)
    single_step, _ = encoder.online_step(inputs[:, 0], prior_state)
    pred, hidden_next = decoder(inputs[:, 0], decoder.get_initial_hidden(inputs), fx["edges"])
    assert torch.allclose(prior, fx["upstream_prior"], atol=FP32_ATOL, rtol=FP32_RTOL)
    assert torch.allclose(posterior, fx["upstream_posterior"], atol=FP32_ATOL, rtol=FP32_RTOL)
    assert torch.allclose(single_step, fx["upstream_single_step"], atol=FP32_ATOL, rtol=FP32_RTOL)
    assert torch.allclose(pred, fx["upstream_decoder_pred"], atol=FP32_ATOL, rtol=FP32_RTOL)
    assert torch.allclose(
        hidden_next, fx["upstream_decoder_hidden"], atol=FP32_ATOL, rtol=FP32_RTOL
    )


# --------------------------------------------------------------------------------------
# Theseus LM reference
# --------------------------------------------------------------------------------------


def test_lm_reference_matches_analytic_solution_and_gradient_sweep():
    problem = linear_least_squares_problem()
    a, b = problem["a"], problem["b"]

    def residual(x):
        return a @ x - b

    result = lm_minimize(residual, np.zeros(a.shape[1]), LMConfig(max_iterations=60))
    analytic = np.linalg.lstsq(a, b, rcond=None)[0]
    assert np.abs(result.x - analytic).max() <= 1e-8
    for step in (1e-4, 1e-5, 1e-6, 1e-7):
        grad = numerical_jacobian(residual, result.x, step=step).T @ residual(result.x)
        assert np.linalg.norm(grad) < 1e-8


def test_outer_acceptance_rejects_residual_decreasing_proposals_and_never_increases():
    problem = linear_least_squares_problem()
    a, b = problem["a"], problem["b"]

    def residual(x):
        return a @ x - b

    prior = 0.5 * np.ones(a.shape[1])

    def full_objective(x):
        resid = residual(x)
        return 0.5 * float(resid @ resid) + 2.5 * float(np.sum((x - prior) ** 2))

    outer = fixed_structure_outer_loop(
        residual, full_objective, np.zeros(a.shape[1]), n_iterations=20
    )
    values = [full_objective(np.zeros(a.shape[1]))] + [
        rec.full_objective_proposed for rec in outer.records
    ]
    assert all(b_ <= a_ + 1e-12 for a_, b_ in zip(values, values[1:], strict=False))
    assert outer.n_rejected > 0
    assert (
        sum(
            1
            for rec in outer.records
            if (not rec.accepted) and rec.residual_cost_proposed < rec.residual_cost_before
        )
        > 0
    )


# --------------------------------------------------------------------------------------
# SMC (upstream particles) — skipped without numba, which particles.resampling imports
# --------------------------------------------------------------------------------------


def test_smc_reference_runs_when_numba_is_available():
    pytest.importorskip("numba", reason="particles.resampling imports numba at module scope")
    from owac.frameworks.n02_full.components.smc_ref import run_smc_reference_check

    report = run_smc_reference_check(out_dir=None, num_particles=2048, seed=1103)
    assert report["exact_upstream_vs_enumeration_max_abs"] <= 1e-12
    assert report["smc"]["tv_mean"] < 0.05
    assert 0.0 <= report["smc"]["tv_max"] < 0.1


def test_smc_exact_enumeration_is_a_proper_distribution():
    from owac.frameworks.n02_full.components.smc_ref import (
        default_model,
        exact_filter_enumeration,
        fixed_observations,
    )

    model = default_model()
    obs = fixed_observations(model)
    posterior = exact_filter_enumeration(model, obs)
    assert posterior.shape == (len(obs), model.num_states)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert (posterior >= 0).all()


# --------------------------------------------------------------------------------------
# profile targets
# --------------------------------------------------------------------------------------


def test_profile_targets_are_modules_with_meta_and_expected_sizes():
    targets = build_profile_targets({"profile": {"img_size": 32, "batch_size": 2}})
    assert set(targets) == {
        "convnext_raw4",
        "convnext_raw3",
        "fcmae_patchify_unpatchify",
        "fcmae_mask_decoder",
        "fcmae_full",
        "savi_512",
    }
    for name, target in targets.items():
        assert isinstance(target, torch.nn.Module), name
        assert set(target.profile_meta) >= {
            "batch_size",
            "sequence_length",
            "input_shape",
            "kind",
            "notes",
        }
    assert targets["convnext_raw4"].profile_meta["input_shape"] == (2, 4, 32, 32)
    assert targets["convnext_raw4"].profile_meta["params"] == 87_694_848
    assert targets["convnext_raw3"].profile_meta["params"] == 87_692_800
    specs = profile_target_specs({"profile": {"img_size": 32}})
    assert isinstance(specs["fcmae_mask_decoder"]["module"], torch.nn.Module)
    assert specs["fcmae_mask_decoder"]["kind"] == targets["fcmae_mask_decoder"].profile_meta["kind"]
    assert specs["convnext_raw4"]["batch_size"] == 1


def test_profile_targets_forward_with_the_declared_shapes():
    targets = build_profile_targets({"profile": {"img_size": 32, "batch_size": 2}})
    encoder = targets["convnext_raw4"]
    x = torch.randn(*encoder.profile_meta["input_shape"])
    with torch.no_grad():
        features = encoder.forward_multiscale(x)
    assert features["stage2_stride16"].shape == (2, 512, 2, 2)
    decoder = targets["fcmae_mask_decoder"]
    mask = torch.zeros(2, 1)
    out = decoder(torch.randn(2, 1024, 1, 1), mask)
    assert out.shape == (2, 32 * 32 * 4, 1, 1)
    geometry = targets["fcmae_patchify_unpatchify"]
    assert torch.equal(geometry(x), x)
