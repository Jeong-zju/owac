"""Rendered objects and supervision are independent across samples and share geometry."""

import pytest
import torch

from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator


def render(batch, *, motion=None, camera=None, rotation=None, base=None, colors=None):
    cfg = SceneConfig(
        noise_sigma=0,
        background_level=0,
        object_colors=colors or ((0.4, 0.2, 0.1),),
    )
    generator = SyntheticSequenceGenerator(cfg)
    if motion is None:
        motion = torch.tensor([[12.0, 0], [-12.0, 0], [0, 12.0], [0, -12.0]])[:batch]
    return generator.frame_batch(
        camera_translation=torch.zeros(batch, 2) if camera is None else camera,
        camera_rotation=torch.zeros(batch) if rotation is None else rotation,
        object_translation=motion,
        illumination=torch.ones(batch),
        base_object_centers=torch.tensor([[[56.0, 56.0]]]).expand(batch, -1, -1).clone()
        if base is None
        else base,
        base_camera_offset=torch.zeros(batch, 2),
    )


@pytest.mark.parametrize("batch", (1, 2, 4))
def test_each_sample_has_its_own_requested_object(batch):
    frame = render(batch)
    torch.testing.assert_close(frame.oracle_clean[:, :1] / 0.4, frame.object_mask)
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    mask = frame.object_mask
    area = mask.sum((1, 2, 3))
    actual = torch.stack(((mask * x).sum((1, 2, 3)) / area, (mask * y).sum((1, 2, 3)) / area), -1)
    torch.testing.assert_close(actual, frame.true_object_translation + 32)
    torch.testing.assert_close(area, torch.full((batch,), 113.0))


def test_changing_one_sample_does_not_change_other_images_or_labels():
    motion = torch.tensor([[12.0, 0], [-12.0, 0], [0, 12.0], [0, -12.0]])
    before = render(4, motion=motion)
    motion[2] = torch.tensor([-10.0, -10.0])
    after = render(4, motion=motion)
    for name in ("measured", "object_mask", "oracle_clean"):
        assert torch.equal(getattr(before, name)[[0, 1, 3]], getattr(after, name)[[0, 1, 3]])
        assert not torch.equal(getattr(before, name)[2], getattr(after, name)[2])


def test_batch_permutation_only_permutes_samples():
    motion = torch.tensor([[12.0, 0], [-12.0, 0], [0, 12.0], [0, -12.0]])
    camera = torch.tensor([[0.2, 0.1], [-0.3, 0.5], [1.1, -0.1], [0.4, 0.8]])
    rotation = torch.tensor([0.0, 2.0, -5.0, 8.0])
    order = torch.tensor([2, 0, 3, 1])
    original = render(4, motion=motion, camera=camera, rotation=rotation)
    permuted = render(4, motion=motion[order], camera=camera[order], rotation=rotation[order])
    for name in ("measured", "object_mask", "oracle_camera_flow"):
        torch.testing.assert_close(getattr(permuted, name), getattr(original, name)[order])


@pytest.mark.parametrize("rotation", (0.0, 3.2, 17.3))
def test_fractional_crop_and_rotation_sample_scene_and_label_identically(rotation):
    frame = render(
        2,
        motion=torch.tensor([[0.35, -0.2], [-0.6, 0.8]]),
        camera=torch.tensor([[1.25, -0.75], [0.35, 0.6]]),
        rotation=torch.tensor([rotation, -rotation]),
    )
    torch.testing.assert_close(
        frame.oracle_clean[:, :1] / 0.4, frame.object_mask, atol=2e-6, rtol=1e-5
    )


def test_multiple_rendered_objects_have_matching_union_label():
    base = torch.tensor([[[48.0, 56.0], [64.0, 56.0]], [[56.0, 48.0], [56.0, 64.0]]])
    frame = render(
        2, motion=torch.zeros(2, 2), base=base, colors=((0.4, 0.2, 0.1), (0.4, 0.2, 0.1))
    )
    torch.testing.assert_close(frame.oracle_clean[:, :1] / 0.4, frame.object_mask)
    torch.testing.assert_close(frame.object_mask.sum((1, 2, 3)), torch.full((2,), 226.0))


def test_unrendered_objects_are_rejected():
    with pytest.raises(ValueError, match="rendered color"):
        render(2, base=torch.zeros(2, 2, 2))
