# ruff: noqa: E402, I001
"""Interface-contract tests for ``src/owac/backends/rawvla_features.py`` (N01 task D).

HONEST SCOPE — read before trusting a green run
-----------------------------------------------
This machine has **no** importable StarVLA/pi0.5 and **no** pi0.5 weights
(``/workspace/owac/rawvla`` is absent; see :func:`test_missing_asset_manifest_is_truthful`).
Every policy test therefore drives ``_StubPI0Contract``: a minimal, independently written
replica of the PI0/PI05 interface contract with the *same signatures* as the frozen reference
sources (``self.vlm.embed_language_tokens``, ``vlm.forward_prefix``,
``action_head.prepare_flow_targets`` / ``embed_action_suffix`` / ``forward_with_action_expert`` /
``decode_action_velocity``, ``PI0.forward_with_action_expert``, ``PI0.denoise_step``).

What these tests *do* establish: the visual prefix is really used, the camera/sequence order and
pad/att mask conventions match ``PI0.embed_prefix``, one observation runs the encoder exactly
once across a 10-step flow solve, the action loss reaches the readout with the backend frozen,
the inference path runs under ``torch.inference_mode()``, partial-batch reset semantics, the
audit loader's vision-replacement behaviour, and Python 3.10 syntax.

What they *cannot* establish (unverified, not claimed): numerical agreement with the real
pi0.5 weights, real checkpoint keys/tensor layout, real KV-cache behaviour of the vendored
Gemma, CUDA/bf16 numerics at 2048 width, or any benchmark result.  A green run here is
interface evidence, not pi0.5 acceptance.
"""

from __future__ import annotations

import ast
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="encoder tests need torch (uv sync --extra encoder)")
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from owac.backends import rawvla_features as rvf  # noqa: E402
from owac.core.encoder_contract import VisualTokens  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/owac/backends/rawvla_features.py"
PATCH_DIR = REPO_ROOT / "third_party/patches/rawvla_features"
PATCH_PATH = PATCH_DIR / "vision_tower_optional.patch"
PATCH_README = PATCH_DIR / "README.md"
REFERENCE_CANDIDATES = (
    os.environ.get("OWAC_STARVLA_REFERENCE"),
    "/home/jeong/zeno/owac/repo/outputs/encoder-handoff-references/starvla",
    str(REPO_ROOT / "outputs/encoder-handoff-references/starvla"),
)

WIDTH = 16  # mini readout width; deliberately far from the deployed 2048
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb")


def _starvla_reference() -> Path | None:
    for candidate in REFERENCE_CANDIDATES:
        if candidate and (Path(candidate) / "starVLA/model/framework/VLM4A/PI0.py").is_file():
            return Path(candidate)
    return None


# --------------------------------------------------------------------------------------
# _StubPI0Contract: interface replica of the frozen PI0/PI05 contract (NOT pi0.5)
# --------------------------------------------------------------------------------------


def _sinusoidal(timestep: torch.Tensor, width: int) -> torch.Tensor:
    """Shape-compatible stand-in for create_sinusoidal_pos_embedding (ActionHead.py:24-40)."""
    fraction = torch.linspace(0.0, 1.0, width // 2, dtype=torch.float32)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    scaling = (1.0 / period) * 2 * math.pi
    angles = scaling[None, :] * timestep.to(torch.float32)[:, None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class _StubLanguageModel(nn.Module):
    """Exposes what ``PI0`` reaches for: ``.config`` and ``layers[0].self_attn.q_proj``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.q_proj = nn.Linear(width, width)
        self.layers = [SimpleNamespace(self_attn=SimpleNamespace(q_proj=self.q_proj))]


class _StubVLM(nn.Module):
    """Replica of the ``OpenPIPaliGemma`` surface used by ``PI0`` (lines 124-157)."""

    def __init__(self, width: int = WIDTH, vocab: int = 32) -> None:
        super().__init__()
        self.lang_embed = nn.Embedding(vocab, width)
        self.image_proj = nn.Linear(3, width)  # stands in for SigLIP + multi_modal_projector
        self.language_model = _StubLanguageModel(width)
        self.embed_image_calls = 0
        self.embed_language_calls = 0
        self.forward_prefix_calls = 0
        self.observed_prefix_dtype: object = None
        self.observed_grad_enabled: object = None
        self.observed_prefix_tokens: int = 0
        self.last_prefix_embs: object = None

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        self.embed_image_calls += 1
        pooled = image.mean(dim=(-2, -1))[:, :3]
        return self.image_proj(pooled.to(self.image_proj.weight.dtype))[:, None, :].expand(
            -1, 4, -1
        )

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        self.embed_language_calls += 1
        return self.lang_embed(tokens)

    def forward_prefix(
        self, *, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache
    ):
        self.forward_prefix_calls += 1
        self.observed_prefix_dtype = inputs_embeds.dtype
        self.observed_grad_enabled = torch.is_grad_enabled()
        self.observed_prefix_tokens = int(inputs_embeds.shape[1])
        self.last_prefix_embs = inputs_embeds
        return SimpleNamespace(
            past_key_values=SimpleNamespace(
                prefix_embs=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        )


class _StubActionHead(nn.Module):
    """Replica of ``OpenPIActionHeadBase`` plus the PI0/PI05 suffix variants."""

    def __init__(
        self,
        width: int = WIDTH,
        action_dim: int = 8,
        action_horizon: int = 4,
        max_state_dim: int = 6,
        pi05: bool = True,
    ) -> None:
        super().__init__()
        self.pi05 = bool(pi05)
        self.use_adarms = bool(pi05)
        self.width = width
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_state_dim = max_state_dim
        self.action_in_proj = nn.Linear(action_dim, width)
        self.action_out_proj = nn.Linear(width, action_dim)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        if pi05:
            self.time_mlp = nn.Linear(width, width)
        else:
            self.state_proj = nn.Linear(max_state_dim, width)
            self.action_time_mlp = nn.Linear(2 * width, width)

    def sample_action_noise(self, shape, device) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def sample_flow_time(self, bsize: int, device) -> torch.Tensor:
        return torch.full((bsize,), 0.5, dtype=torch.float32, device=device)

    def prepare_flow_targets(self, actions, noise=None, time=None):
        actions = actions.to(device=self.action_in_proj.weight.device, dtype=torch.float32)
        noise = (
            self.sample_action_noise(actions.shape, actions.device)
            if noise is None
            else noise.to(actions.device, torch.float32)
        )
        time = (
            self.sample_flow_time(actions.shape[0], actions.device)
            if time is None
            else time.to(actions.device, torch.float32)
        )
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        return x_t, noise - actions, noise, time

    def embed_action_suffix(self, state, noisy_actions, timestep):
        time_emb = _sinusoidal(timestep, self.width).to(self.action_in_proj.weight.dtype)
        action_emb = self.action_in_proj(noisy_actions.to(self.action_in_proj.weight.dtype))
        if self.pi05:
            adarms = F.silu(self.time_mlp(time_emb))
            pad = torch.ones(action_emb.shape[:2], dtype=torch.bool, device=action_emb.device)
            att = [1] + [0] * (self.action_horizon - 1)
            att_t = torch.tensor(att, dtype=torch.bool, device=action_emb.device)
            return action_emb, pad, att_t[None, :].expand(action_emb.shape[0], -1), adarms
        state_emb = self.state_proj(state.to(self.state_proj.weight.dtype))
        expanded = time_emb[:, None, :].expand_as(action_emb)
        action_time = self.action_time_mlp(torch.cat([action_emb, expanded], dim=2))
        embs = torch.cat([state_emb[:, None, :], action_time], dim=1)
        pad = torch.ones(embs.shape[:2], dtype=torch.bool, device=embs.device)
        att = torch.tensor([1] + [0] * self.action_horizon, dtype=torch.bool, device=embs.device)
        return embs, pad, att[None, :].expand(embs.shape[0], -1), None

    def attend(self, prefix_embs, suffix_embs, attention_mask):
        """Single-head attention over [prefix | suffix] using the frozen additive mask."""
        keys = self.k(torch.cat([prefix_embs, suffix_embs], dim=1))
        values = self.v(torch.cat([prefix_embs, suffix_embs], dim=1))
        logits = (self.q(suffix_embs) @ keys.transpose(-1, -2)) / math.sqrt(self.width)
        logits = logits[:, None, :, :] + attention_mask.to(logits.dtype)
        weights = torch.softmax(logits, dim=-1)
        return (weights @ values[:, None, :, :]).squeeze(1) + suffix_embs

    def decode_action_velocity(self, hidden):
        hidden = hidden[:, -self.action_horizon :].to(self.action_out_proj.weight.dtype)
        return self.action_out_proj(hidden).to(torch.float32)


class _StubPI0Contract(nn.Module):
    """Minimal replica of ``PI0``'s feature-facing interface.  NOT a pi0.5 implementation."""

    def __init__(
        self,
        width: int = WIDTH,
        action_dim: int = 8,
        action_horizon: int = 4,
        max_state_dim: int = 6,
        pi05: bool = True,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.vlm = _StubVLM(width=width)
        self.action_head = _StubActionHead(width, action_dim, action_horizon, max_state_dim, pi05)
        self.image_keys = list(rvf.PI0_IMAGE_KEYS)
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_state_dim = max_state_dim
        self.num_inference_steps = 10
        self.discrete_state_input = False
        self.state_pad_value = 0.0
        self.forward_with_action_expert_calls = 0
        self.denoise_step_calls = 0
        self.grad_enabled_in_suffix_path: object = None
        self.last_prefix_pad_masks: object = None
        if dtype != torch.float32:
            self.to(dtype)  # mirrors PI0.to_precision("bfloat16")

    def forward_with_action_expert(
        self, *, prefix_embs, suffix_embs, attention_mask, position_ids, adarms_cond
    ):
        self.forward_with_action_expert_calls += 1
        self.grad_enabled_in_suffix_path = torch.is_grad_enabled()
        self.last_full_attention_mask = attention_mask
        # PI0.forward_with_action_expert gets a mask over all streams; the suffix is the last
        # block, so its query rows are the trailing rows of the 4-D mask (PI0.py:456-460).
        suffix_len = suffix_embs.shape[1]
        suffix_rows = attention_mask[:, :, attention_mask.shape[2] - suffix_len :, :]
        return self.action_head.attend(prefix_embs, suffix_embs, suffix_rows)

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """Mirrors PI0.denoise_step (PI0.py:562-577): prefix comes from the KV cache only."""
        self.denoise_step_calls += 1
        self.grad_enabled_in_suffix_path = torch.is_grad_enabled()
        self.last_prefix_pad_masks = prefix_pad_masks
        suffix_embs, suffix_pad, suffix_att, adarms = self.action_head.embed_action_suffix(
            state, x_t, timestep
        )
        suffix_len = suffix_pad.shape[1]
        bsize, prefix_len = prefix_pad_masks.shape
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        att_2d = rvf._make_att_2d_masks_local(suffix_pad, suffix_att)
        mask4d = rvf._make_att_4d_masks_local(torch.cat([prefix_pad_2d, att_2d], dim=2))
        suffix_out = self.action_head.attend(
            past_key_values.prefix_embs, suffix_embs.to(past_key_values.prefix_embs.dtype), mask4d
        )
        return self.action_head.decode_action_velocity(suffix_out)


class _StubEncoder(nn.Module):
    """Counts readout/reset calls; exposes only what the policy requires."""

    def __init__(
        self, tokens_per_camera: int = 4, cameras: tuple[str, ...] = CAMERAS, in_dim: int = 8
    ) -> None:
        super().__init__()
        self.tokens_per_camera = tokens_per_camera
        self.cameras = tuple(cameras)
        self.in_dim = in_dim
        self.proj = nn.Linear(4, in_dim)
        self.readout_calls = 0
        self.reset_calls = 0
        self.reset_masks: list[object] = []

    @property
    def token_count(self) -> int:
        return self.tokens_per_camera * len(self.cameras)

    def readout(self, output, token_budget=None):
        self.readout_calls += 1
        representation = getattr(output, "representation", output)
        values = self.proj(representation)
        mask = torch.ones(values.shape[:2], dtype=torch.bool)
        return VisualTokens(values=values, mask=mask, camera_ids=list(self.cameras))

    def reset(self, state, batch_mask=None):
        self.reset_calls += 1
        self.reset_masks.append(batch_mask)
        return {"reset": True, "batch_mask": str(batch_mask)}


@dataclass
class _StubObservation:
    """Field-for-field stand-in for ``PI0.Observation`` (PI0.py:39-45)."""

    images: dict
    image_masks: dict
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    state: torch.Tensor


def _observation(batch: int = 2, lang_len: int = 5, state_dim: int = 6) -> _StubObservation:
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(1, 32, (batch, lang_len), generator=generator)
    mask = torch.ones(batch, lang_len, dtype=torch.bool)
    mask[:, -1] = False  # one padded language position
    return _StubObservation(
        images={},
        image_masks={},
        tokenized_prompt=tokens,
        tokenized_prompt_mask=mask,
        state=torch.randn(batch, 1, state_dim, generator=generator),
    )


def _visual_tokens(
    batch: int = 2, cameras: tuple[str, ...] = CAMERAS, in_dim: int = 8, seed: int = 0
) -> VisualTokens:
    generator = torch.Generator().manual_seed(seed)
    length = 4 * len(cameras)
    values = torch.randn(batch, length, in_dim, generator=generator)
    return VisualTokens(
        values=values, mask=torch.ones(batch, length, dtype=torch.bool), camera_ids=list(cameras)
    )


def _actions(batch: int = 2, horizon: int = 4, action_dim: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn(batch, horizon, action_dim, generator=generator)


def _make_policy(
    pi05: bool = True,
    encoder: _StubEncoder | None = None,
    dtype=torch.float32,
    batch_size: int = 1,
) -> tuple[rvf.RawVLAFeaturePolicy, _StubPI0Contract]:
    stub = _StubPI0Contract(pi05=pi05, dtype=dtype)
    policy = rvf.RawVLAFeaturePolicy(
        stub,
        rvf.MiniVisualTokenPostprocessor(),
        encoder=encoder,
        action_protocol=rvf.ACTION_PROTOCOLS["libero"],
        action_cache=rvf.ActionChunkCache(
            rvf.ACTION_PROTOCOLS["libero"].replan_steps, batch_size=batch_size
        ),
    )
    return policy, stub


# --------------------------------------------------------------------------------------
# 1. import discipline / no-torch survival
# --------------------------------------------------------------------------------------


def test_import_does_not_crash_without_torch_or_starvla():
    """`import owac.backends.rawvla_features` must survive a torch-free, starVLA-free env."""
    program = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['starVLA'] = None\n"
        "import owac.backends.rawvla_features as m\n"
        "assert m.torch is None, 'torch must stay unimported'\n"
        "assert sys.modules.get('starVLA') is None, 'starVLA must stay blocked'\n"
        "assert not [k for k, v in sys.modules.items()\n"
        "            if k.startswith('starVLA') and v is not None]\n"
        "for name in ('VisualTokenPostprocessor', 'RawVLAFeaturePolicy',\n"
        "             'AuditedVisualReplacementLoader', 'load_audited_with_replaced_vision'):\n"
        "    assert hasattr(m, name), name\n"
        "try:\n"
        "    m.VisualTokenPostprocessor()\n"
        "except RuntimeError as exc:\n"
        "    assert 'torch' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('expected RuntimeError without torch')\n"
        "print('no-torch import ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no-torch import ok" in result.stdout


def test_module_is_python310_and_avoids_match_and_type_statements():
    source = MODULE_PATH.read_text()
    tree = ast.parse(source)
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Match)], "match is banned"
    assert not [n for n in ast.walk(tree) if type(n).__name__ == "TypeAlias"], "type is banned"
    ast.parse(source, feature_version=(3, 10))  # deployment runtime is Python 3.10
    heavy = {"torch", "transformers", "starVLA", "safetensors", "omegaconf"}
    unguarded = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            unguarded += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            unguarded.append(node.module or "")
    assert not [name for name in unguarded if name.split(".")[0] in heavy], unguarded


# --------------------------------------------------------------------------------------
# 2. postprocessor: configuration, naming, parameter accounting
# --------------------------------------------------------------------------------------


def test_controlled_config_is_256x2048_and_mini_is_named_honestly():
    controlled = rvf.VisualPrefixConfig()
    assert controlled.tokens_per_camera == 256
    assert controlled.width == 2048 == rvf.PI05_LANGUAGE_WIDTH
    assert controlled.name == rvf.CONTROLLED_CONFIG_NAME
    assert controlled.camera_order == rvf.PI0_IMAGE_KEYS

    mini = rvf.MiniVisualTokenPostprocessor()
    assert mini.describe()["name"] == rvf.MINI_CONFIG_NAME
    assert "mini" in rvf.MINI_CONFIG_NAME and rvf.MINI_CONFIG_NAME != rvf.CONTROLLED_CONFIG_NAME
    assert mini.describe()["width"] == WIDTH
    assert issubclass(rvf.VisualTokenPostprocessor, nn.Module)

    report = rvf.parameter_report(mini)
    assert report["total"] == sum(report["by_name"].values()) > 0
    assert report["trainable"] == report["total"]
    # No in_proj.bias: a uniform per-row shift cancels in the following LayerNorm, so the
    # bias would be a dead parameter.
    assert set(report["by_name"]) == {
        "in_proj.weight",
        "norm.weight",
        "norm.bias",
        "camera_gain",
        "camera_embed.weight",
        "spatial_index.weight",
        "time_embed.weight",
    }
    assert "in_proj.bias" not in report["by_name"]
    # The first controlled configuration is a real, countable parameter set.
    full = rvf.VisualTokenPostprocessor()
    assert rvf.parameter_report(full)["total"] > 4_000_000
    assert full.describe()["prefix_tokens"] == 256 * len(rvf.PI0_IMAGE_KEYS)

    # The controlled config is derived from the shared readout budget (readout_tokens /
    # readout_width), without importing the framework config module.
    shared = SimpleNamespace(readout_tokens=256, readout_width=2048)
    derived = rvf.VisualPrefixConfig.from_readout_config(shared, camera_order=CAMERAS)
    assert derived == rvf.VisualPrefixConfig(camera_order=CAMERAS)
    assert derived.name == rvf.CONTROLLED_CONFIG_NAME
    with pytest.raises(ValueError, match="readout_tokens"):
        rvf.VisualPrefixConfig.from_readout_config(SimpleNamespace(readout_tokens=256))


# --------------------------------------------------------------------------------------
# 3. prefix layout: camera order, masks, language scaling, padding safety
# --------------------------------------------------------------------------------------


def test_prefix_layout_order_masks_and_language_scaling():
    policy, stub = _make_policy()
    observation = _observation(batch=2, lang_len=5)
    tokens = _visual_tokens(batch=2)

    embs, pad, att = policy.embed_prefix_from_visual_tokens(
        tokens, observation.tokenized_prompt, observation.tokenized_prompt_mask
    )
    visual_tokens_count = 4 * len(CAMERAS)
    assert embs.shape == (2, visual_tokens_count + 5, WIDTH)
    assert pad.shape == att.shape == (2, visual_tokens_count + 5)

    # Sequence order: camera-major blocks (base_0_rgb first, left_wrist second), language last.
    assert pad[:, :visual_tokens_count].all()
    assert torch.equal(pad[:, visual_tokens_count:], observation.tokenized_prompt_mask)
    assert not att.any(), "every prefix token stays in one bidirectional block (PI0.py:390/394)"

    postprocessor = policy.postprocessor
    single = postprocessor(
        VisualTokens(
            values=tokens.values[:, :4], mask=tokens.mask[:, :4], camera_ids=["base_0_rgb"]
        )
    )
    single_second = postprocessor(
        VisualTokens(
            values=tokens.values[:, 4:], mask=tokens.mask[:, 4:], camera_ids=["left_wrist_0_rgb"]
        )
    )
    # Same camera_order as the two-camera run: block 0 holds base_0_rgb, block 1 the wrist.
    assert single.embs.shape[1] == 8 and not single.pad_masks[:, 4:].any()
    assert torch.allclose(embs[:, :4], single.embs[:, :4])
    assert torch.allclose(embs[:, 4:8], single_second.embs[:, 4:8])
    # A one-camera order re-indexes the learned per-camera gain/embedding, so only slot 0 of
    # the shortened order can match the two-camera run (documented, order-keyed behaviour).
    shortened = postprocessor(
        VisualTokens(
            values=tokens.values[:, :4], mask=tokens.mask[:, :4], camera_ids=["base_0_rgb"]
        ),
        camera_order=["base_0_rgb"],
    )
    assert shortened.embs.shape[1] == 4
    assert torch.allclose(embs[:, :4], shortened.embs)

    # Scale convention: language is multiplied by sqrt(dim) (PI0.py:392); the OWAC visual
    # prefix is not, and the applied scale is reported instead.
    lang_emb = stub.vlm.embed_language_tokens(observation.tokenized_prompt)
    assert torch.allclose(embs[:, visual_tokens_count:], lang_emb * math.sqrt(WIDTH))
    assert torch.allclose(
        embs[:, :visual_tokens_count],
        torch.cat([single.embs[:, :4], single_second.embs[:, 4:8]], dim=1),
    )
    assert single.diagnostics["visual_prefix_scale"] == "none"
    assert single.diagnostics["scale_value"] == 1.0
    assert shortened.diagnostics["valid_tokens_per_camera_batch_sum"] == {"base_0_rgb": 8}
    assert shortened.diagnostics["prefix_tokens"] == 4
    assert single.diagnostics["synthetic_camera_blocks"] == ["left_wrist_0_rgb"]

    # Position convention used by PI0 (cumsum(pad_masks) - 1): first token 0, padding repeats.
    position_ids = torch.cumsum(pad, dim=1) - 1
    assert position_ids[0, 0].item() == 0
    assert position_ids[0, -1].item() == position_ids[0, -2].item()


def test_missing_camera_block_is_zero_padded_and_prefix_length_is_constant():
    postprocessor = rvf.MiniVisualTokenPostprocessor()
    tokens = _visual_tokens(batch=2, cameras=("left_wrist_0_rgb",))
    prefix = postprocessor(tokens)
    assert prefix.embs.shape[1] == 8, "prefix length must not depend on which cameras are valid"
    assert not prefix.pad_masks[:, :4].any()
    assert prefix.pad_masks[:, 4:].all()
    assert torch.equal(prefix.embs[:, :4], torch.zeros_like(prefix.embs[:, :4]))
    assert prefix.diagnostics["synthetic_camera_blocks"] == ["base_0_rgb"]


def test_masked_slots_with_nan_values_stay_exactly_zero():
    postprocessor = rvf.MiniVisualTokenPostprocessor()
    tokens = _visual_tokens(batch=2)
    tokens.values[:, 1] = float("nan")
    tokens.mask[:, 1] = False
    prefix = postprocessor(tokens)
    assert torch.equal(prefix.embs[:, 1], torch.zeros_like(prefix.embs[:, 1]))
    assert torch.isfinite(prefix.embs).all()


def test_time_slot_encoding_statistics_and_encoder_side_parameters():
    encoder = _StubEncoder()
    policy, _stub = _make_policy(encoder=encoder)
    tokens = _visual_tokens(batch=1)

    # Temporal position encoding is real: the mini config has a learned slot embedding, and a
    # different readout step must produce a different prefix.
    first = policy.postprocessor(tokens, time_slots=policy._time_slot)
    slot = policy.advance_time_slot()
    assert slot == 1
    second = policy.postprocessor(tokens, time_slots=slot)
    assert first.diagnostics["time_slot"] == 0 and second.diagnostics["time_slot"] == 1
    assert not torch.allclose(first.embs, second.embs)

    # Encoder-side parameters are exactly the readout plus the bound encoder.
    names = {id(parameter) for parameter in policy.encoder_side_parameters()}
    expected = {id(p) for p in policy.postprocessor.parameters()}
    expected |= {id(p) for p in encoder.parameters()}
    assert names == expected

    policy.stats["encoder_calls"] = 3
    policy.reset_stats()
    assert set(policy.stats.values()) == {0}


def test_postprocessor_rejects_token_count_and_width_mismatch():
    postprocessor = rvf.MiniVisualTokenPostprocessor()
    with pytest.raises(ValueError, match="token count"):
        postprocessor(VisualTokens(values=torch.zeros(1, 7, 8), mask=torch.ones(1, 7, dtype=bool)))
    with pytest.raises(ValueError, match="in_dim"):
        postprocessor(
            VisualTokens(
                values=torch.zeros(1, 4, 3),
                mask=torch.ones(1, 4, dtype=bool),
                camera_ids=["base_0_rgb"],
            )
        )
    with pytest.raises(ValueError, match="unknown camera"):
        postprocessor(
            VisualTokens(
                values=torch.zeros(1, 4, 8),
                mask=torch.ones(1, 4, dtype=bool),
                camera_ids=["head_camera"],
            )
        )


# --------------------------------------------------------------------------------------
# 4. the visual prefix is really used; encoder runs exactly once per observation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pi05", [True, False])
def test_encoder_runs_once_for_a_ten_step_flow_solve(pi05):
    encoder = _StubEncoder()
    policy, stub = _make_policy(pi05=pi05, encoder=encoder)
    observation = _observation(batch=2)
    encoder_output = SimpleNamespace(representation=torch.randn(2, 8, 4), state={"slot": 0})

    result = policy.predict_action_with_features(
        observation, encoder_output, num_steps=10, return_debug=True
    )

    assert encoder.readout_calls == 1, "one observation -> exactly one readout"
    assert policy.stats["encoder_calls"] == 1
    assert policy.stats["prefix_builds"] == 1
    assert policy.stats["denoise_steps"] == 10
    assert stub.denoise_step_calls == 10
    assert stub.vlm.embed_image_calls == 0, "the image path must not run"
    assert stub.forward_with_action_expert_calls == 0, "sampling uses the cached prefix"
    assert result["num_denoise_steps"] == 10
    assert result["actions"].shape == (2, 4, 8)
    assert result["encoder_calls"] == 1

    # The cached prefix is the full [visual | language] prefix, not the visual block alone.
    lang_len = observation.tokenized_prompt.shape[1]
    assert stub.vlm.observed_prefix_tokens == 8 + lang_len
    assert stub.last_prefix_pad_masks.shape == (2, 8 + lang_len)


def test_changing_visual_tokens_changes_the_sampled_actions():
    policy, _stub = _make_policy()
    observation = _observation(batch=1)
    noise = torch.zeros(1, 4, 8)
    first = policy.sample_actions_with_features(
        observation, _visual_tokens(batch=1, seed=0), noise=noise, num_steps=4
    )
    second = policy.sample_actions_with_features(
        observation, _visual_tokens(batch=1, seed=1), noise=noise, num_steps=4
    )
    assert (first - second).abs().max().item() > 1e-6
    assert torch.isfinite(first).all()


def test_prefix_cached_for_denoising_equals_embed_prefix_output():
    policy, stub = _make_policy()
    observation = _observation(batch=2)
    tokens = _visual_tokens(batch=2)
    expected, _pad, _att = policy.embed_prefix_from_visual_tokens(
        tokens, observation.tokenized_prompt, observation.tokenized_prompt_mask
    )
    policy.sample_actions_with_features(observation, tokens, num_steps=3)
    assert torch.allclose(stub.vlm.last_prefix_embs, expected)


def test_denoise_step_refuses_to_run_without_a_visual_prefix():
    policy, stub = _make_policy()
    with pytest.raises(RuntimeError, match="embed_prefix_from_visual_tokens"):
        policy.denoise_step_with_features(
            torch.zeros(1, 6),
            torch.ones(1, 9, dtype=bool),
            None,
            torch.zeros(1, 4, 8),
            torch.zeros(1),
        )
    assert stub.denoise_step_calls == 0


# --------------------------------------------------------------------------------------
# 5. gradient flow / freezing / inference mode
# --------------------------------------------------------------------------------------


def test_gradient_flow_check_is_a_real_backward_pass():
    encoder = _StubEncoder()
    policy, stub = _make_policy(encoder=encoder)
    observation = _observation(batch=2)
    tokens = _visual_tokens(batch=2)

    report = policy.gradient_flow_check(
        observation,
        tokens,
        _actions(batch=2),
        noise=torch.zeros(2, 4, 8),
        time=torch.full((2,), 0.5),
    )

    assert report["action_loss_finite"] and math.isfinite(report["action_loss"])
    assert report["encoder_output_grad_norm"] > 0.0, "loss must reach the readout tokens"
    assert report["postprocessor_output_grad_norm"] > 0.0
    assert report["encoder_param_grad_norm"] > 0.0
    assert report["encoder_params_with_nonzero_grad"] >= 1
    assert "postprocessor.in_proj.weight" in report["nonzero_param_names"]
    # The detached probe keeps the readout tensor as a leaf, so the bound encoder's own
    # parameters sit outside this graph and are reported instead of silently counted.
    assert report["encoder_params_without_graph"] == ["encoder.proj.weight", "encoder.proj.bias"]
    assert (
        report["encoder_params_with_nonzero_grad"] + len(report["encoder_params_without_graph"])
        == report["encoder_params_checked"]
    )
    assert report["graph_retained"] is True
    assert report["grad_enabled_during_forward"] is True
    assert report["inference_mode"] is False
    assert report["freeze_report"]["frozen_backend_tensors"] > 0

    # Frozen language/action parameters keep requires_grad False and receive no gradient.
    assert report["frozen_backend_params_with_grad"] == []
    assert policy._count_backend_requires_grad() == 0
    for _, parameter in stub.named_parameters():
        assert parameter.requires_grad is False
        assert parameter.grad is None
    assert all(p.requires_grad for p in policy.postprocessor.parameters())


def test_gradient_flow_check_reaches_the_encoder_parameters_through_the_readout():
    encoder = _StubEncoder()
    policy, _stub = _make_policy(encoder=encoder)
    observation = _observation(batch=2)
    encoder_output = SimpleNamespace(representation=torch.randn(2, 8, 4), state={"slot": 0})

    report = policy.gradient_flow_check(
        observation,
        actions=_actions(batch=2),
        noise=torch.zeros(2, 4, 8),
        time=torch.full((2,), 0.5),
        encoder_output=encoder_output,
    )

    assert encoder.readout_calls == 1
    assert report["encoder_params_without_graph"] == []
    assert report["encoder_params_checked"] == report["encoder_params_with_nonzero_grad"] == 9
    assert "encoder.proj.weight" in report["nonzero_param_names"]
    assert report["encoder_output_grad_norm"] > 0.0
    assert report["frozen_backend_params_with_grad"] == []


def test_gradient_flow_check_requires_a_token_source():
    policy, _stub = _make_policy()
    with pytest.raises(ValueError, match="visual_tokens or encoder_output"):
        policy.gradient_flow_check(_observation(), actions=_actions())


def test_training_forward_keeps_autograd_graph_and_inference_does_not():
    policy, stub = _make_policy()
    observation = _observation(batch=2)
    tokens = _visual_tokens(batch=2)
    out = policy.forward_from_observation_with_features(
        observation,
        tokens,
        _actions(batch=2),
        noise=torch.zeros(2, 4, 8),
        time=torch.full((2,), 0.5),
        return_debug=True,
    )
    assert stub.grad_enabled_in_suffix_path is True
    assert out["action_loss"].grad_fn is not None
    assert out["prefix_embs"].grad_fn is not None

    sampled = policy.sample_actions_with_features(observation, tokens, num_steps=3)
    assert stub.vlm.observed_grad_enabled is False, "inference path must disable grad"
    assert stub.grad_enabled_in_suffix_path is False
    assert sampled.grad_fn is None and sampled.requires_grad is False
    assert not torch.is_inference_mode_enabled(), "inference_mode must be restored afterwards"


def test_prefix_dtype_follows_the_frozen_language_model():
    policy, stub = _make_policy(dtype=torch.bfloat16)
    observation = _observation(batch=2)
    tokens = _visual_tokens(batch=2)
    out = policy.forward_from_observation_with_features(
        observation,
        tokens,
        _actions(batch=2),
        noise=torch.zeros(2, 4, 8),
        time=torch.full((2,), 0.5),
        return_debug=True,
    )
    assert out["prefix_embs"].dtype == torch.bfloat16
    prefix = policy.postprocessor(tokens)
    assert prefix.embs.dtype == torch.float32, "OWAC readout stays fp32 until the boundary"
    policy.sample_actions_with_features(observation, tokens, num_steps=2)
    assert stub.vlm.observed_prefix_dtype == torch.bfloat16


# --------------------------------------------------------------------------------------
# 6. reset semantics (full and partial batch)
# --------------------------------------------------------------------------------------


def test_partial_batch_reset_clears_only_finished_episodes():
    encoder = _StubEncoder()
    policy, _stub = _make_policy(encoder=encoder, batch_size=2)
    observation = _observation(batch=2)
    encoder_output = SimpleNamespace(representation=torch.randn(2, 8, 4), state={"slot": 0})
    policy.predict_action_with_features(observation, encoder_output, num_steps=2)
    policy.action_cache.record(torch.zeros(10, 8), step=0)
    policy.action_cache.record(torch.zeros(10, 8), step=0, batch_index=1)
    report = policy.reset_encoder_state(torch.tensor([True, False]))

    assert report["partial"] is True
    assert report["encoder_reset"] == "delegated_to_encoder"
    assert encoder.reset_calls == 1
    assert report["readout_tokens_cleared"] == [0]
    assert report["action_cache"]["indices"] == [0]
    assert policy.action_cache.actions[1] is not None, "running episode keeps its chunk"
    assert policy.action_cache.actions[0] is None
    assert policy._cached_visual_tokens is None

    # A new step after the reset must re-run the encoder (no stale prefix survives).
    policy.predict_action_with_features(observation, encoder_output, num_steps=2)
    assert encoder.readout_calls == 2


def test_full_reset_clears_every_episode_and_the_action_cache():
    encoder = _StubEncoder()
    policy, _stub = _make_policy(encoder=encoder, batch_size=2)
    observation = _observation(batch=2)
    policy.predict_action_with_features(
        observation,
        SimpleNamespace(representation=torch.randn(2, 8, 4), state={"slot": 0}),
        num_steps=2,
    )
    policy.action_cache.record(torch.zeros(10, 8), step=0, batch_index=0)
    policy.action_cache.record(torch.zeros(10, 8), step=0, batch_index=1)
    assert policy.action_cache.actions != [None, None]
    report = policy.reset_encoder_state()
    assert report["partial"] is False
    assert report["readout_tokens_cleared"] == "all"
    assert report["action_cache"]["indices"] == [0, 1]
    assert policy._cached_visual_tokens is None
    assert policy.action_cache.actions == [None, None]
    assert encoder.reset_masks[-1] is None


# --------------------------------------------------------------------------------------
# 7. action protocol: cadence read from the client source, never inferred
# --------------------------------------------------------------------------------------


def test_replan_cadence_is_read_from_the_client_source():
    protocols = rvf.load_action_protocols()
    robotwin = protocols["robotwin"]
    libero = protocols["libero"]

    assert robotwin.replan_steps == 50 and robotwin.action_horizon == 50
    assert libero.replan_steps == 5 and libero.action_horizon == 10
    assert libero.replan_steps != libero.action_horizon, "cadence is not the horizon"
    assert libero.control_dim == 7 and robotwin.control_dim == 14
    assert robotwin.model_action_dim == 32 and libero.model_action_dim == 32
    assert robotwin.provenance["derived_from_horizon"] is False
    assert libero.provenance["derived_from_horizon"] is False
    assert "rawvla_robotwin.py" in robotwin.provenance["source"]
    assert "rawvla_full_eval.py" in libero.provenance["source"]

    client_source = Path(robotwin.provenance["source"])
    guard = client_source.read_text().splitlines()[robotwin.provenance["line"] - 1]
    assert guard.strip() == "if self.actions is None or step - self.chunk_start >= 50:"
    assert robotwin.provenance["anchor"] == "step - self.chunk_start >= 50"


def test_action_chunk_cache_parity_with_the_frozen_client():
    cache = rvf.ActionChunkCache(replan_steps=50)
    actions = torch.arange(50).float()[:, None].expand(50, 14).clone()
    assert cache.should_replan(3)
    cache.record(actions, step=3)
    assert cache.step(3)[0].item() == 0
    assert not cache.should_replan(52)
    assert cache.step(52)[0].item() == 49
    assert cache.should_replan(53)
    report = cache.reset()
    assert report["slots"] == 1 and cache.actions == [None]


def test_replan_reader_fails_loudly_instead_of_guessing(tmp_path):
    forged = tmp_path / "client.py"
    forged.write_text("class Client:\n    def step(self, step):\n        return step\n")
    with pytest.raises(ValueError, match="replan guard"):
        rvf.read_replan_cadence(forged)
    with pytest.raises(FileNotFoundError):
        rvf.read_replan_cadence(tmp_path / "missing.py")


# --------------------------------------------------------------------------------------
# 8. audited loading with an explicitly replaced vision tower
# --------------------------------------------------------------------------------------


class _StubVisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = nn.Linear(3, 3)
        self.head = nn.Linear(3, 2)


class _StubCheckpointVLM(nn.Module):
    """Mirrors OpenPIPaliGemma's module layout: vlm.model.model.{vision_tower,language_model}."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.model = nn.Module()
        self.model.model.vision_tower = _StubVisionTower()
        self.model.model.language_model = nn.Linear(4, 4)
        self.model.lm_head = nn.Linear(4, 4)


class _StubCheckpointModel(nn.Module):
    """Replica of the ``PI0`` surface used by ``load_audited`` (rawvla.py:24-58)."""

    def __init__(self) -> None:
        super().__init__()
        self.vlm = _StubCheckpointVLM()
        self.action_head = nn.Linear(4, 4)

    def _normalize_state_dict(self, state):
        renamed = {}
        for key, value in state.items():
            clean = key
            for prefix in ("model.", "module.", "_orig_mod."):
                while clean.startswith(prefix):
                    clean = clean[len(prefix) :]
            if clean.startswith("paligemma_with_expert.paligemma.model."):
                clean = clean.replace(
                    "paligemma_with_expert.paligemma.model.", "vlm.model.model.", 1
                )
            elif clean.startswith("paligemma_with_expert.paligemma."):
                clean = clean.replace("paligemma_with_expert.paligemma.", "vlm.model.", 1)
            renamed[clean] = value
        return renamed

    def _is_expected_openpi_missing_key(self, key: str) -> bool:
        return key.endswith(".input_layernorm.weight")


def _checkpoint_state(model: _StubCheckpointModel) -> dict:
    return {key: value.clone() for key, value in model.state_dict().items()}


def test_audit_loader_skips_vision_weights_and_releases_the_tower():
    from owac.backends.rawvla import load_audited

    model = _StubCheckpointModel()
    state = _checkpoint_state(model)
    vision_keys = [k for k in state if k.startswith("vlm.model.model.vision_tower.")]
    assert len(vision_keys) == 4
    for key in vision_keys:
        state[key] = torch.full_like(state[key], float("nan"))

    # The unchanged baseline tries to load the vision tower (and rejects NaN-free? no: it loads).
    baseline = _StubCheckpointModel()
    audit = load_audited(baseline, state)
    assert audit["active_missing"] == [] and audit["shape_mismatch"] == []
    assert torch.isnan(baseline.vlm.model.model.vision_tower.vision_model.weight).all()

    # The replacement loader refuses to load them and frees the tower instead.
    replaced = _StubCheckpointModel()
    report = rvf.load_audited_with_replaced_vision(replaced, state)
    assert report["replaced_vision_keys"] == 4
    assert len(report["replaced_vision_key_sha256"]) == 64
    assert report["vision_release"]["vision_tower_detected"] is True
    assert report["vision_release"]["verified_removed"] is True
    assert report["vision_release"]["bytes_freed"] > 0
    assert not [n for n, _ in replaced.named_parameters() if "vision_tower" in n]
    assert not any(torch.isnan(p).any() for p in replaced.parameters())
    assert report["parameter_groups"]["language_model"]["missing"] == []
    assert report["parameter_groups"]["language_model"]["checked_tensors"] == 2
    assert report["parameter_groups"]["action_expert"]["model_tensors"] == 2


def test_audit_loader_still_checks_language_and_action_tensors_item_by_item():
    from owac.backends.rawvla import load_audited

    model = _StubCheckpointModel()
    good = _checkpoint_state(model)

    wrong_shape = dict(good)
    wrong_shape["action_head.weight"] = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="language/action"):
        rvf.load_audited_with_replaced_vision(_StubCheckpointModel(), wrong_shape)
    with pytest.raises(ValueError, match="does not match"):
        load_audited(_StubCheckpointModel(), wrong_shape)  # baseline still rejects it

    missing = dict(good)
    del missing["vlm.model.model.language_model.weight"]
    with pytest.raises(ValueError, match="language_model"):
        rvf.load_audited_with_replaced_vision(_StubCheckpointModel(), missing)

    # A wrongly shaped vision tensor no longer blocks loading: it is replaced by design.
    broken_vision = dict(good)
    broken_vision["vlm.model.model.vision_tower.head.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="does not match"):
        load_audited(_StubCheckpointModel(), broken_vision)
    report = rvf.load_audited_with_replaced_vision(_StubCheckpointModel(), broken_vision)
    assert report["replaced_vision_keys"] == 4


def test_audit_loader_normalizes_legacy_keys_before_classifying_vision():
    model = _StubCheckpointModel()
    state = _checkpoint_state(model)
    legacy = {
        key.replace("vlm.model.model.", "paligemma_with_expert.paligemma.model.", 1): value
        for key, value in state.items()
        if key.startswith("vlm.model.model.")
    }
    legacy.update({k: v for k, v in state.items() if not k.startswith("vlm.model.model.")})
    loader = rvf.AuditedVisualReplacementLoader()
    classified = loader.classify(model, legacy)
    assert classified["vision_keys"], "normalisation must run before vision classification"
    assert len(classified["vision_keys"]) == 4


# --------------------------------------------------------------------------------------
# 9. frozen-source parity and missing-asset truthfulness
# --------------------------------------------------------------------------------------


def _frozen_function(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace = {"torch": torch}
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - pinned reference
            return namespace[name]
    raise AssertionError(f"{name} not found in {path}")


def test_local_mask_helpers_match_the_frozen_upstream_source():
    reference = _starvla_reference()
    if reference is None:
        pytest.skip("frozen StarVLA reference copy not available for parity checking")
    action_head = reference / "starVLA/model/modules/action_model/OpenPI_ActionHead.py"
    upstream_2d = _frozen_function(action_head, "make_att_2d_masks")
    upstream_4d = _frozen_function(action_head, "make_att_4d_masks")
    pad = torch.tensor([[True, True, True, False], [True, False, True, True]])
    att = torch.tensor([[False, False, False, False], [False, True, False, False]])
    ours_2d = rvf._make_att_2d_masks_local(pad, att)
    assert torch.equal(ours_2d, upstream_2d(pad, att))
    assert torch.equal(rvf._make_att_4d_masks_local(ours_2d), upstream_4d(ours_2d))
    assert "local replica" in rvf.mask_helper_source(), "starVLA is not installed on this machine"


def test_missing_asset_manifest_is_truthful():
    entries = rvf.describe_missing_assets()
    kinds = {entry.kind for entry in entries}
    assert {
        "runtime_root",
        "source_tree",
        "weights",
        "statistics",
        "tokenizer",
        "environment",
    } <= kinds
    for entry in entries:
        assert entry.present == Path(entry.path).exists(), entry
    if not Path("/workspace/owac/rawvla").is_dir():
        assert all(not entry.present for entry in entries), "no pi0.5 asset is present locally"
    paths = " ".join(entry.path for entry in entries)
    assert "pi05_libero_starvla/bfloat16/model.safetensors" in paths
    assert "checkpoints/pi05_robotwin/model.safetensors" in paths
    assert "envs/starvla/bin/python" in paths


# --------------------------------------------------------------------------------------
# 10. third-party patch hygiene
# --------------------------------------------------------------------------------------


def test_third_party_patch_applies_to_the_frozen_sources(tmp_path):
    reference = _starvla_reference()
    if reference is None:
        pytest.skip("frozen StarVLA reference copy not available for patch checking")
    assert PATCH_PATH.is_file() and PATCH_README.is_file()
    targets = {
        "starVLA/model/modules/vlm/OpenPIPaliGemma.py": reference
        / "starVLA/model/modules/vlm/OpenPIPaliGemma.py",
        "starVLA/model/framework/VLM4A/PI0.py": reference / "starVLA/model/framework/VLM4A/PI0.py",
    }
    for relative, source in targets.items():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)
        assert "build_vision_tower" not in destination.read_text()

    result = subprocess.run(
        ["patch", "-p1", "-i", str(PATCH_PATH)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for relative in targets:
        text = (tmp_path / relative).read_text()
        assert "build_vision_tower" in text
        ast.parse(text, feature_version=(3, 10))


def test_vision_tower_policy_probe_only_uses_the_patched_keyword():
    class Unpatched:
        def __init__(self, config, *, use_adarms: bool = False, model_name_or_path: str = ""):
            del config, use_adarms, model_name_or_path

    class Patched:
        def __init__(self, config, *, use_adarms: bool = False, build_vision_tower: bool = True):
            del config, use_adarms, build_vision_tower

    assert rvf.vision_tower_policy_kwargs(Unpatched) == {}
    assert rvf.vision_tower_policy_kwargs(Patched) == {"build_vision_tower": False}
    assert rvf.vision_tower_policy_kwargs(object) == {}


def test_patch_readme_records_commits_applications_and_unverified_items():
    text = PATCH_README.read_text()
    assert "ba9c56b04da2c4931b70d8844710f5928d43775b" in text
    assert "57d508cb180d0510ac9060358009c0f6ffaac420" in text
    assert "patch -p1" in text and "build_vision_tower" in text
    assert "未验证" in text
    for block in text.split("```python")[1:]:
        ast.parse(block.split("```")[0], feature_version=(3, 10))
