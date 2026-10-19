"""Pinned StarVLA API contract fixtures; these are not pretrained-weight evidence."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from owac.backends.rawvla_features_n04 import (
    FeatureBridgeConfig,
    RawVLAFeaturePolicy,
    verify_gradient_path,
)
from owac.core.encoder_contract_n04 import MeasurementFrame, VisualTokens
from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig
from owac.frameworks.n04.ego import EgoMotionContext
from owac.frameworks.n04.model import N04Encoder


class FixtureVLM(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.embed = nn.Embedding(64, dim)
        q_proj = nn.Linear(dim, dim)
        self.language_model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=q_proj))],
            config=SimpleNamespace(_attn_implementation="eager"),
        )

    def embed_language_tokens(self, tokens):
        return self.embed(tokens)

    def forward_prefix(
        self, *, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache
    ):
        support = attention_mask[:, 0, 0] == 0
        pooled = (inputs_embeds * support[..., None]).sum(1) / support.sum(1).clamp_min(1)[:, None]
        return SimpleNamespace(past_key_values=pooled)


class FixtureActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_in_proj = nn.Linear(32, 16)
        self.action_out_proj = nn.Linear(16, 32)
        self.time_mlp_in, self.time_mlp_out = nn.Linear(16, 16), nn.Linear(16, 16)
        self.width, self.action_horizon = 16, 10

    def prepare_flow_targets(self, actions, noise=None, time=None):
        noise = torch.zeros_like(actions) if noise is None else noise
        time = torch.ones(actions.shape[0]) * 0.5 if time is None else time
        return (
            time[:, None, None] * noise + (1 - time[:, None, None]) * actions,
            noise - actions,
            noise,
            time,
        )

    def embed_action_suffix(self, state, noisy_actions, timestep):
        x = self.action_in_proj(noisy_actions)
        pad = torch.ones(x.shape[:2], dtype=torch.bool)
        block = torch.zeros_like(pad)
        block[:, 0] = True
        return x, pad, block, timestep[:, None].expand(-1, 16)

    def decode_action_velocity(self, output):
        return self.action_out_proj(output[:, -self.action_horizon :]).float()

    def sample_action_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def forward(self, *, attention_mask, position_ids, past_key_values, suffix_embs, adarms_cond):
        return self.decode_action_velocity(suffix_embs + past_key_values[:, None])


def make_att_2d_masks(pad, block):
    groups = block.cumsum(1)
    return (groups[:, None, :] <= groups[:, :, None]) & pad[:, None, :] & pad[:, :, None]


def make_att_4d_masks(mask):
    return torch.where(mask[:, None], 0.0, -2.3819763e38)


class FixtureBackend(nn.Module):
    """Same signatures as PI0, independently executable on CPU."""

    def __init__(self):
        super().__init__()
        self.vlm = FixtureVLM()
        self.action_head = FixtureActionHead()
        self.max_state_dim, self.action_dim, self.action_horizon = 32, 32, 10

    @staticmethod
    def _preprocess_observation(obs):
        return (), (), obs.tokenized_prompt, obs.tokenized_prompt_mask, obs.state

    def forward_with_action_expert(
        self, *, prefix_embs, suffix_embs, attention_mask, position_ids, adarms_cond
    ):
        support = attention_mask[:, 0, -1, : prefix_embs.shape[1]] == 0
        pooled = (prefix_embs * support[..., None]).sum(1) / support.sum(1).clamp_min(1)[:, None]
        return suffix_embs + pooled[:, None] + adarms_cond[:, None]

    def forward_from_observation(
        self, observation, actions, noise=None, time=None, return_debug=False
    ):
        _, _, tokens, mask, state = self._preprocess_observation(observation)
        prefix, pad, block = self.embed_prefix((), (), tokens, mask)
        x, u, noise, time = self.action_head.prepare_flow_targets(actions, noise=noise, time=time)
        suffix, spad, sblock, cond = self.action_head.embed_action_suffix(state.squeeze(1), x, time)
        pads, blocks = torch.cat((pad, spad), 1), torch.cat((block, sblock), 1)
        hidden = self.forward_with_action_expert(
            prefix_embs=prefix,
            suffix_embs=suffix,
            attention_mask=make_att_4d_masks(make_att_2d_masks(pads, blocks)),
            position_ids=pads.long().cumsum(1) - 1,
            adarms_cond=cond,
        )
        velocity = self.action_head.decode_action_velocity(hidden)
        return {"v_t": velocity, "u_t": u, "action_loss": (velocity - u).square().mean()}

    def sample_actions(self, observation, noise=None, num_steps=10):
        _, _, tokens, mask, state = self._preprocess_observation(observation)
        prefix, pad, block = self.embed_prefix((), (), tokens, mask)
        cache = self.vlm.forward_prefix(
            inputs_embeds=prefix,
            attention_mask=make_att_4d_masks(make_att_2d_masks(pad, block)),
            position_ids=pad.long().cumsum(1) - 1,
            past_key_values=None,
            use_cache=True,
        ).past_key_values
        value = (
            self.action_head.sample_action_noise((state.shape[0], 10, 32), state.device)
            if noise is None
            else noise
        )
        for i in range(num_steps):
            velocity = self.denoise_step(
                state.squeeze(1),
                pad,
                cache,
                value,
                torch.full((state.shape[0],), 1 - i / num_steps),
            )
            value = value - velocity / num_steps
        return value

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix, pad, block, cond = self.action_head.embed_action_suffix(state, x_t, timestep)
        return self.action_head(
            attention_mask=None,
            position_ids=None,
            past_key_values=past_key_values,
            suffix_embs=suffix,
            adarms_cond=cond,
        )


def bridge():
    cfg = N04Config(
        measurement_size=(16, 16),
        geometry=N04GeometryConfig(feature_size=(16, 16), pixel_stride=1),
        model=N04ModelConfig(
            width=8, hidden=16, scales=1, blocks=1, readout_tokens=4, readout_dim=16
        ),
    )
    encoder = N04Encoder(cfg)
    backend = FixtureBackend()
    policy = RawVLAFeaturePolicy(
        encoder, backend, FeatureBridgeConfig(tokens_per_camera=4, token_dim=16)
    )
    return policy, encoder, backend


def context(batch=1):
    return EgoMotionContext(
        proprio_delta=torch.ones(batch, 3),
        proprio_delta_available=True,
        executed_action=torch.ones(batch, 3),
        executed_action_available=True,
        dt=torch.ones(batch),
        camera_id="head",
    )


def frame(batch=1, index=0):
    return MeasurementFrame(
        values=torch.rand(batch, 3, 16, 16) * 0.8,
        camera_id="head",
        frame_ids=(index,) * batch,
        episode_ids=tuple(f"e{i}" for i in range(batch)),
    )


def test_prefix_preserves_language_padding_scaling_and_mask_positions():
    policy, _, backend = bridge()
    values = VisualTokens(torch.randn(1, 4, 16), torch.tensor([[True, False, True, False]]))
    tokens, mask = torch.zeros(1, 3, dtype=torch.long), torch.tensor([[True, True, False]])
    prefix = policy.embed_prefix_from_visual_tokens(values, tokens, mask)
    assert prefix["prefix_mask"].tolist() == [[True, False, True, False, True, True, False]]
    assert prefix["positions"].tolist() == [[0, 0, 1, 1, 2, 3, 3]]
    assert torch.allclose(
        prefix["prefix"][:, 4:], backend.vlm.embed_language_tokens(tokens) * math.sqrt(16)
    )


def test_first_measurement_and_invalid_regions_reach_readout_mask():
    policy, encoder, _ = bridge()
    observation = frame()
    observation.valid_mask = torch.ones(1, 1, 16, 16)
    observation.valid_mask[:, :, :, 8:] = 0
    tokens, _ = policy.encode_observation(observation, context(), encoder.init_state(1)["head"])
    assert tokens.mask.tolist() == [[True, False, True, False]]


def test_one_observation_ten_denoise_steps_and_no_graph():
    policy, encoder, _ = bridge()
    result = policy.sample_actions(
        frame(),
        context(),
        encoder.init_state(1)["head"],
        torch.zeros(1, 3, dtype=torch.long),
        torch.ones(1, 3, dtype=torch.bool),
        backend_state=torch.zeros(1, 8),
    )
    assert result["actions"].shape == (1, 10, 32)
    assert result["control_actions"].shape == (1, 10, 7)
    assert policy.counter.calls == 1
    assert not result["state"].stable.requires_grad


def test_gradients_reach_propagation_update_and_readout_when_backend_frozen():
    policy, encoder, backend = bridge()
    backend.requires_grad_(False)
    state = encoder.init_state(1)["head"]
    _, state = policy.encode_observation(frame(index=0), context(), state)
    output = policy.forward_from_observation(
        frame(index=1),
        context(),
        state,
        torch.randn(1, 10, 7),
        torch.zeros(1, 3, dtype=torch.long),
        torch.ones(1, 3, dtype=torch.bool),
        backend_state=torch.zeros(1, 8),
    )
    output["loss"].backward()
    report = verify_gradient_path(
        policy,
        encoder_trainable_parameters=list(encoder.named_parameters()),
        forbidden_parameters=list(backend.named_parameters()),
    )
    assert report["ok"]
    for prefix in ("propagator.head.", "decomposer.head.", "readout.proj."):
        assert sum(v for k, v in report["gradient_norms"].items() if k.startswith(prefix)) > 0


def test_action_tail_padding_has_zero_loss_and_missing_state_is_rejected():
    policy, encoder, _ = bridge()
    args = (
        frame(),
        context(),
        encoder.init_state(1)["head"],
        torch.randn(1, 10, 7),
        torch.zeros(1, 3, dtype=torch.long),
        torch.ones(1, 3, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="backend_state"):
        policy.forward_from_observation(*args)
    result = policy.forward_from_observation(
        *args, backend_state=torch.zeros(1, 8), action_mask=torch.zeros(1, 10)
    )
    assert result["loss"].item() == 0
