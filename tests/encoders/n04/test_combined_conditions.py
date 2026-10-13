"""Factor combinations have actual simultaneous causes with no oracle context leakage."""

from dataclasses import replace

import pytest
import torch

from owac.frameworks.n04.config import N04TrainConfig
from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from owac.training.n04 import _change_target, make_synthetic_batches

from .test_six_fixes import config


@pytest.mark.parametrize(
    "name,camera,object_motion",
    [
        ("camera_illumination", True, False),
        ("object_illumination", False, True),
        ("both_illumination", True, True),
    ],
)
def test_joint_cause_truths_and_context_independence(name, camera, object_motion):
    scene = SceneConfig(
        camera_translation=1, object_velocity=1, illumination=(0.6, 1.3), noise_sigma=0
    )
    generator = SyntheticSequenceGenerator(scene)
    seq = generator.sequence(batch=4, length=5, condition=name, seed=999)
    baseline = generator.sequence(
        batch=4, length=5, condition="camera_only" if camera else "static", seed=999
    )
    for t, (frame, base) in enumerate(zip(seq, baseline, strict=True)):
        torch.testing.assert_close(frame.proprio, base.proprio)
        torch.testing.assert_close(frame.action, base.action)
        if t:
            assert bool(frame.true_camera_translation.abs().sum()) == camera
            assert bool(frame.true_object_translation.abs().sum()) == object_motion
            assert bool((frame.true_illumination - seq[t - 1].true_illumination).abs().sum())
            target, _ = _change_target(frame, (64, 64))
            assert bool(target.sum()) == object_motion
            assert frame.metadata["object_moving"] == [object_motion] * 4


def test_joint_illumination_has_same_rendered_geometry_as_motion_control():
    scene = SceneConfig(
        camera_translation=1, object_velocity=1, illumination=(0.5, 0.8), noise_sigma=0
    )
    gen = SyntheticSequenceGenerator(scene)
    joint = gen.sequence(batch=2, length=4, condition="both_illumination", seed=12)
    motion = gen.sequence(batch=2, length=4, condition="both", seed=12)
    for a, b in zip(joint, motion, strict=True):
        torch.testing.assert_close(a.object_mask, b.object_mask)
        torch.testing.assert_close(
            a.measured / a.true_illumination[:, None, None, None], b.measured
        )


def test_training_condition_schedule_is_explicit_and_reproducible():
    recipe = N04TrainConfig(
        batch_size=2, sequence_length=3, synthetic={"conditions": ["static", "both_illumination"]}
    )
    a = make_synthetic_batches(cfg=config(), train_cfg=recipe, steps=4, seed=3)
    b = make_synthetic_batches(cfg=config(), train_cfg=recipe, steps=4, seed=3)
    assert [seq[0].condition for seq in a] == ["static", "both_illumination"] * 2
    for seq1, seq2 in zip(a, b, strict=True):
        for f1, f2 in zip(seq1, seq2, strict=True):
            torch.testing.assert_close(f1.measured, f2.measured)
    with pytest.raises(ValueError, match="nonempty"):
        make_synthetic_batches(
            cfg=config(), train_cfg=replace(recipe, synthetic={"conditions": []}), steps=1, seed=3
        )
    with pytest.raises(ValueError, match="unknown synthetic condition"):
        SyntheticSequenceGenerator().sequence(condition="typo")
