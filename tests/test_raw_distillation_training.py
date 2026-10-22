"""Minimal shared fixture helpers for legacy raw-distillation runtime tests."""

from __future__ import annotations

from owac.backends.rawvla_features_n02 import build_stub_bridge
from owac.baselines.raw_stem_siglip import RawStemSiglipStudent
from tests.test_raw_stem_siglip import make_policy


def _student_and_bridge():
    policy = make_policy(seed=3)
    student = RawStemSiglipStudent.from_fixture(
        policy, input_mode="rawvla_rgb3", fixture_id="train"
    )
    built = build_stub_bridge(
        width=6,
        token_dim=6,
        tokens_per_camera=4,
        channels=3,
        frame_pixels=2,
        action_horizon=10,
        cameras=("camera_0",),
        with_frame=False,
    )
    return student, built["bridge"]
