"""N01 v1.2 P3 four-arm factory and the claim that support changes the content.

The load-bearing check here is ``test_support_changes_the_fused_content``: with
identical weights, the Base arm and the N01 arm must produce *different* tokens
once support withdraws a source.  A diagnostic-only support that leaves the
content untouched would pass every shape assertion and fail this one.
"""

from __future__ import annotations

import pytest
import torch

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.frameworks.n01_full.model import (
    ARM_METHODS,
    ARMS,
    PLATFORMS,
    N01V12Config,
    N01V12Model,
    arm_matrix,
)
from owac.frameworks.n01_full.support import SupportConfig


def _config(arm: str, *, window: int = 4, **support_overrides) -> N01V12Config:
    support = SupportConfig(
        region_grid=4,
        multiscale_radii=(0,),
        receptive_field_radius=0,
        **support_overrides,
    )
    return N01V12Config(arm=arm, platform="libero", window_frames=window, support=support)


def _frames(model: N01V12Model, values: torch.Tensor, camera: str):
    """Push each frame slot of a [B,T,C,H,W] window as one arrival (oldest first)."""

    state = model.init_state(values.shape[0], {"camera_ids": [camera], "shape": values.shape[-2:]})
    budget = ComputeBudget(max_updates=1, max_history=values.shape[1])
    output = None
    for index in range(values.shape[1]):
        frame = MeasurementFrame(
            values=values[:, index],
            camera_id=camera,
            frame_ids=[index] * values.shape[0],
            episode_ids=["episode"] * values.shape[0],
        )
        output = model.step(frame, state=state, budget=budget)
        state = output.state
    assert output is not None
    return model.readout(output)


def _window(batch: int = 1, frames: int = 4, size: int = 32, seed: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((batch, frames, 3, size, size), generator=generator)


def test_arm_matrix_is_two_platforms_by_four_arms_by_one_seed() -> None:
    matrix = arm_matrix()
    assert len(matrix) == 8
    assert {config.platform for config in matrix} == set(PLATFORMS)
    assert {config.arm for config in matrix} == set(ARM_METHODS)
    assert {config.seed for config in matrix} == {101}
    assert all(config.window_frames == 32 for config in matrix)


def test_every_arm_shares_one_parameter_count() -> None:
    counts = {}
    for method in ARM_METHODS:
        model = N01V12Model(_config(method))
        report = model.parameter_report()
        assert report["support_parameters"] == 0
        counts[method] = report["total_trainable"]
    assert len(set(counts.values())) == 1, counts
    assert counts["raw_n01"] > 0


def test_support_is_off_for_base_arms_and_reported_truthfully() -> None:
    values = _window()

    base = N01V12Model(_config("raw_base"))
    base_tokens = _frames(base, values, "agentview")

    n01 = N01V12Model(_config("raw_n01"))
    n01.load_state_dict(base.state_dict())
    n01_tokens = _frames(n01, values, "agentview")

    assert base.support is None
    assert n01.support is not None
    assert base_tokens.values.shape == n01_tokens.values.shape


def test_support_changes_the_fused_content() -> None:
    """Identical weights, different content: the load-bearing requirement D check."""

    values = _window()
    shifted = values.clone()
    # A sustained offset is a structural change, not plausible measurement noise.
    shifted[:, 1:] += 6.0

    base = N01V12Model(_config("raw_base"))
    n01 = N01V12Model(_config("raw_n01"))
    n01.load_state_dict(base.state_dict())

    base_tokens = _frames(base, shifted, "agentview")
    n01_tokens = _frames(n01, shifted, "agentview")

    assert base_tokens.values.shape == n01_tokens.values.shape
    assert not torch.allclose(base_tokens.values, n01_tokens.values, atol=1e-6), (
        "support left the fused content untouched; it would be a diagnostic field, not a "
        "mechanism that changes what reaches the bridge"
    )


def test_support_keeps_a_near_static_history_and_the_token_check_is_blunt() -> None:
    """A near-static history stays usable -- measured, and measured to be a weak guard.

    Two things were established by flipping the comparison and re-running (threshold
    unchanged so the config stays valid):

    * the arm-vs-arm token assertions do NOT catch the inversion: the fusion concentrates
      nearly all of its weight on the reference slot, so including or dropping the history
      barely moves the tokens.  "The two arms agree/differ" is therefore a blunt instrument
      for whether history was used;
    * this assertion does not catch it either -- the recovery step makes the inverted
      comparison self-cancelling here: a stationary series has low odds, so the inverted
      rule withdraws it and the recovery branch immediately restores it.

    The guard that does catch the inversion is
    ``test_support.py::test_a_long_stationary_history_stays_usable``, which failed under the
    flip.  This test is kept for what it does verify -- that a near-static history is
    usable, and that the token equality is too weak to carry that claim.
    """

    arrivals = 16
    reference = _window(frames=1)
    jitter = torch.randn((1, arrivals, 3, 32, 32), generator=torch.Generator().manual_seed(17))
    near_static = (reference.repeat(1, arrivals, 1, 1, 1) + 0.01 * jitter).clamp(0.0, 1.0)

    base = N01V12Model(_config("raw_base", window=arrivals))
    n01 = N01V12Model(_config("raw_n01", window=arrivals))
    n01.load_state_dict(base.state_dict())

    base_tokens = _frames(base, near_static, "agentview")
    n01_tokens = _frames(n01, near_static, "agentview")

    decision = n01.encoder.last_support
    assert decision is not None, "a 16-arrival window must reach the support decision"
    usable = decision.source_usable
    assert bool(usable.any()), (
        "a near-static history must keep sources usable; if support withdraws them it is "
        "withdrawing exactly the evidence it exists to accumulate"
    )
    assert int((decision.state == 0).sum()) == 0 or bool(usable.any())

    # Weak instrument, kept and labelled: identical-shape equality here does not by itself
    # demonstrate that the history was used.
    assert base_tokens.values.shape == n01_tokens.values.shape


def test_raw_and_rgb_arms_share_the_input_shape_but_not_the_domain() -> None:
    raw = _config("raw_n01")
    rgb = _config("rgb_n01")
    assert raw.adapter_config().input_channels == rgb.adapter_config().input_channels == 3
    assert raw.spec.input_domain == "RAW"
    assert rgb.spec.input_domain == "RGB"
    assert raw.spec.is_raw and not rgb.spec.is_raw


def test_config_round_trips_through_its_effective_record() -> None:
    config = _config("rgb_n01", window=8)
    payload = config.to_dict()
    assert payload["input_domain"] == "RGB"
    assert payload["support_enabled"] is True
    assert payload["data_kind"] in {"hdr_derived", "libero_dequantized", "real_cfa"}
    restored = N01V12Config.from_dict(payload)
    assert restored.arm == config.arm
    assert restored.window_frames == config.window_frames
    assert restored.support == config.support
    assert restored.to_dict() == payload


def test_unknown_arm_and_platform_are_refused() -> None:
    with pytest.raises(ValueError):
        N01V12Config(arm="raw_n02")
    with pytest.raises(ValueError):
        N01V12Config(platform="droid")
    with pytest.raises(ValueError):
        N01V12Config(arm="raw_base", window_frames=1)


def test_camera_order_comes_from_the_real_loader_protocol() -> None:
    libero = N01V12Config(arm="raw_base", platform="libero")
    assert libero.cameras == ("agentview", "wrist")
    robotwin = N01V12Config(arm="raw_base", platform="robotwin2")
    assert len(robotwin.cameras) == 3
    assert all(isinstance(name, str) and name for name in robotwin.cameras)


def test_arm_specs_describe_the_helperd_protocol() -> None:
    assert ARMS["raw_base"].is_raw and not ARMS["raw_base"].is_n01
    assert ARMS["raw_n01"].is_raw and ARMS["raw_n01"].is_n01
    assert not ARMS["rgb_base"].is_raw and not ARMS["rgb_base"].is_n01
    assert not ARMS["rgb_n01"].is_raw and ARMS["rgb_n01"].is_n01


def test_rgb_arms_consume_the_frozen_isp_not_the_raw_tensor() -> None:
    """The RGB arms must read a different tensor from the RAW arms.

    ``model.py`` claimed the RGB arms read a ``rendered_rgb3`` tensor produced by
    ``owac.raw.n01_paired_views``, but ``N01V12Config`` hard-coded ``input_mode`` to the
    RAW view for every arm, so ``rgb_n01`` and ``raw_n01`` consumed byte-identical input
    and the "same mechanism, different input" comparison had never been run.
    """

    import torch

    from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
    from owac.frameworks.n01_full.model import N01V12Config, N01V12Model

    def build(arm: str) -> N01V12Model:
        model = N01V12Model(N01V12Config(arm=arm, platform="libero", window_frames=4))
        model.eval()
        return model

    raw = build("raw_n01")
    rgb = build("rgb_n01")
    assert raw.encoder.config.input_mode == "rawvla_rgb3"
    assert rgb.encoder.config.input_mode == "rendered_rgb3"

    torch.manual_seed(11)
    values = torch.rand((1, 3, 32, 32))
    budget = ComputeBudget(max_updates=4, max_history=4)

    captured: dict[str, torch.Tensor] = {}
    original = type(raw.encoder).encode_window

    def spy(self, window_values, *args, **kwargs):
        captured[self.config.input_mode] = window_values.detach().clone()
        return original(self, window_values, *args, **kwargs)

    type(raw.encoder).encode_window = spy
    try:
        for index in range(2):
            for model in (raw, rgb):
                state = model.init_state(
                    1, {"camera_ids": ["agentview"], "shape": (32, 32)}, torch.device("cpu")
                )
                frame = MeasurementFrame(
                    values=values,
                    valid_mask=torch.ones(1, dtype=torch.bool),
                    bad_mask=torch.zeros(1, dtype=torch.bool),
                    saturated_mask=torch.zeros(1, dtype=torch.bool),
                    frame_ids=torch.tensor([index]),
                    camera_id="agentview",
                    timestamp=torch.tensor([float(index)]),
                    time_unit="frame",
                    new_measurement=torch.ones(1, dtype=torch.bool),
                )
                output = model.step(frame, None, state, budget)
                captured[f"{model.config.arm}:{index}"] = model.readout(output).values.clone()
    finally:
        type(raw.encoder).encode_window = original

    assert "rawvla_rgb3" in captured and "rendered_rgb3" in captured
    assert not torch.allclose(captured["rawvla_rgb3"], captured["rendered_rgb3"], atol=1e-6), (
        "the RGB arm's encoder input is identical to the RAW arm's: the ISP was not applied"
    )
    assert not torch.allclose(captured["raw_n01:0"], captured["rgb_n01:0"], atol=1e-6), (
        "the two arms produced identical tokens, so only the label differs"
    )


def test_episode_reset_clears_the_support_detector() -> None:
    """A new episode must not inherit the previous episode's support state.

    ``FullState.support`` was the one field ``reset`` did not rebuild, so the BOCPD
    posterior, the per-source age, the defer counts and the state codes all survived an
    episode boundary -- letting one rollout's measurements decide the support of the next
    rollout's first frames.
    """

    import torch

    from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
    from owac.frameworks.n01_full.model import N01V12Config, N01V12Model

    model = N01V12Model(N01V12Config(arm="raw_n01", platform="libero", window_frames=6))
    state = model.init_state(
        1, {"camera_ids": ["agentview"], "shape": (16, 16)}, torch.device("cpu")
    )
    budget = ComputeBudget(max_updates=6, max_history=6)
    values = torch.rand((1, 3, 16, 16))
    for index in range(4):
        frame = MeasurementFrame(
            values=values,
            valid_mask=torch.ones(1, dtype=torch.bool),
            bad_mask=torch.zeros(1, dtype=torch.bool),
            saturated_mask=torch.zeros(1, dtype=torch.bool),
            frame_ids=torch.tensor([index]),
            camera_id="agentview",
            timestamp=torch.tensor([float(index)]),
            time_unit="frame",
            new_measurement=torch.ones(1, dtype=torch.bool),
        )
        state = model.step(frame, None, state, budget).state

    detector = state.support["agentview"]["detector"]
    assert detector.step > 0, "the detector should have advanced during the episode"
    assert int(state.support["agentview"]["age"].sum()) > 0

    reset_state = model.reset(state)
    cleared = reset_state.support["agentview"]
    assert cleared["detector"].step == 0, "the episode reset left the detector at its old arrival"
    assert int(cleared["age"].sum()) == 0, "the episode reset left the per-source age"
    assert int(cleared["defer_count"].sum()) == 0, "the episode reset left the defer counts"
    assert bool((cleared["state"] == 0).all()), (
        "the episode reset left state codes from the previous episode"
    )


@pytest.mark.parametrize("platform", ["libero", "robotwin2"])
@pytest.mark.parametrize("arm", ["raw_base", "raw_n01", "rgb_base", "rgb_n01"])
def test_every_arm_configuration_builds_and_runs_a_forward_pass(arm: str, platform: str) -> None:
    """All eight arm x platform configurations must actually execute.

    The matrix was only asserted at the metadata level for ``rgb_base``: it appeared in a
    comparison of ``ArmSpec`` flags but was never constructed, so the factory's claim to
    produce four working arms rested on three.  This runs a forward pass through each
    configuration and checks the two things the design promises about it: the input domain
    follows the arm, and only the N01 arms attach the support mechanism.
    """

    from owac.core.encoder_contract import ComputeBudget, MeasurementFrame

    model = N01V12Model(
        N01V12Config(
            arm=arm,
            platform=platform,
            window_frames=4,
            # Narrow and shallow: this is a wiring check, and the full-width defaults make
            # eight forward passes needlessly slow.
            feature_width=32,
            stem_width=8,
            residual_blocks=1,
        )
    )
    spec = ARMS[arm]
    expected_mode = "rawvla_rgb3" if spec.is_raw else "rendered_rgb3"
    assert model.encoder.config.input_mode == expected_mode, (
        f"{arm} resolved input mode {model.encoder.config.input_mode!r}, expected {expected_mode!r}"
    )
    assert (model.support is not None) is spec.is_n01, (
        f"{arm} attached support={model.support is not None}, expected {spec.is_n01}"
    )

    camera = model.config.cameras[0]
    state = model.init_state(1, {"camera_ids": [camera], "shape": (16, 16)}, torch.device("cpu"))
    budget = ComputeBudget(max_updates=4, max_history=4)
    output = None
    for index in range(3):
        frame = MeasurementFrame(
            values=torch.rand((1, 3, 16, 16), generator=torch.Generator().manual_seed(index)),
            valid_mask=torch.ones(1, dtype=torch.bool),
            bad_mask=torch.zeros(1, dtype=torch.bool),
            saturated_mask=torch.zeros(1, dtype=torch.bool),
            frame_ids=torch.tensor([index]),
            camera_id=camera,
            timestamp=torch.tensor([float(index)]),
            time_unit="frame",
            new_measurement=torch.ones(1, dtype=torch.bool),
        )
        output = model.step(frame, None, state, budget)
        state = output.state
    tokens = model.readout(output).values
    assert tokens.shape[0] == 1 and tokens.shape[1] > 0
    assert bool(torch.isfinite(tokens).all()), "a forward pass produced non-finite tokens"


def test_the_unique_buffer_saturates_at_the_exact_window_cap() -> None:
    """The window must hold exactly the newest ``MAX_EXACT_UNIQUE_FRAMES``, not drift.

    Saturation was observed while recording a real rollout (the reported unique-frame count
    climbed to 32 and stayed there), but nothing asserted it.  What matters for the
    exact-replay claim is not only that the count stops at the cap but that the frames kept
    are the *newest* ones: a window holding an arbitrary 32 arrivals would still report a
    saturated count while replaying the wrong history.

    Writing this test caught an error in my own first inspection: I read ``frame_ids[0]``
    (slot 0 only) and concluded the buffer had lost its id history, when the correct slice
    is ``frame_ids[:, 0]``.
    """

    from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
    from owac.frameworks.n01.config import MAX_EXACT_UNIQUE_FRAMES

    cap = MAX_EXACT_UNIQUE_FRAMES
    arrivals = cap + 5
    model = N01V12Model(
        N01V12Config(
            arm="raw_n01",
            platform="libero",
            window_frames=cap,
            feature_width=32,
            stem_width=8,
            residual_blocks=1,
        )
    )
    camera = model.config.cameras[0]
    state = model.init_state(1, {"camera_ids": [camera], "shape": (32, 32)}, torch.device("cpu"))
    budget = ComputeBudget(max_updates=cap, max_history=cap)
    for index in range(arrivals):
        frame = MeasurementFrame(
            values=torch.rand((1, 3, 32, 32), generator=torch.Generator().manual_seed(index)),
            valid_mask=torch.ones(1, dtype=torch.bool),
            bad_mask=torch.zeros(1, dtype=torch.bool),
            saturated_mask=torch.zeros(1, dtype=torch.bool),
            frame_ids=torch.tensor([index]),
            camera_id=camera,
            timestamp=torch.tensor([float(index)]),
            time_unit="frame",
            new_measurement=torch.ones(1, dtype=torch.bool),
        )
        state = model.step(frame, None, state, budget).state
        filled = int(state.cameras[camera].filled.max())
        assert filled == min(index + 1, cap), (
            f"after {index + 1} arrivals the buffer holds {filled}; "
            f"expected min({index + 1}, {cap})"
        )

    held = [int(value) for value in state.cameras[camera].frame_ids[:, 0].tolist()]
    assert held == list(range(arrivals - 1, arrivals - cap - 1, -1)), (
        f"the window must hold the newest {cap} ids in newest-first order, got {held[:4]}..."
    )


def test_a_still_buffered_frame_is_not_counted_twice() -> None:
    """Re-sending a frame that is still in the window must not create a second observation.

    "Exact replay" depends on the buffer counting each measurement once; if a repeated id
    were pushed again, the same measurement would enter the fusion twice and every count
    downstream would be inflated.  The arrival must still be causal, so the duplicate is
    sent at a later timestamp -- which is the realistic case of a client resending a frame.
    """

    from owac.core.encoder_contract import ComputeBudget, MeasurementFrame

    model = N01V12Model(
        N01V12Config(
            arm="raw_n01",
            platform="libero",
            window_frames=8,
            feature_width=32,
            stem_width=8,
            residual_blocks=1,
        )
    )
    camera = model.config.cameras[0]
    state = model.init_state(1, {"camera_ids": [camera], "shape": (32, 32)}, torch.device("cpu"))
    budget = ComputeBudget(max_updates=8, max_history=8)

    def push(frame_id: int, stamp: float):
        frame = MeasurementFrame(
            values=torch.rand((1, 3, 32, 32), generator=torch.Generator().manual_seed(frame_id)),
            valid_mask=torch.ones(1, dtype=torch.bool),
            bad_mask=torch.zeros(1, dtype=torch.bool),
            saturated_mask=torch.zeros(1, dtype=torch.bool),
            frame_ids=torch.tensor([frame_id]),
            camera_id=camera,
            timestamp=torch.tensor([stamp]),
            time_unit="frame",
            new_measurement=torch.ones(1, dtype=torch.bool),
        )
        return model.step(frame, None, state, budget).state

    for index in range(4):
        state = push(index, float(index))
    before = int(state.cameras[camera].filled.max())
    assert before == 4

    state = push(1, 10.0)  # id 1 is still buffered
    after = int(state.cameras[camera].filled.max())
    assert after == before, "a still-buffered frame was counted as a new measurement"
    assert [int(v) for v in state.cameras[camera].frame_ids[:, 0].tolist()][:4] == [3, 2, 1, 0]

    # A genuinely new id must still be accepted, or the check would be a blanket refusal.
    state = push(99, 11.0)
    assert int(state.cameras[camera].filled.max()) == before + 1
