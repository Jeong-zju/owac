import numpy as np
import torch

from owac.simulation.raw_student_closed_loop import (
    _burst,
    _packet,
    _wilson,
    paired_noise,
)


def _ring():
    return [
        {
            "agentview": np.full((256, 256, 3), i, np.uint8),
            "wrist": np.full((256, 256, 3), i + 10, np.uint8),
        }
        for i in range(6)
    ]


def test_six_frame_burst_and_packet_padding_causality():
    ring = _ring()
    burst = _burst(ring, "agentview")
    assert burst.shape == (6, 3, 256, 256)
    packet = _packet("ep", 10, ring, ("base_0_rgb", "left_wrist_0_rgb"), torch.device("cpu"))
    assert len(packet.frames) == 6 and packet.reset is True
    assert packet.frame_ids == (5, 6, 7, 8, 9, 10)
    assert packet.padding_mask == (False,) * 6
    assert packet.frames[0].values.shape == (1, 2, 3, 224, 224)
    assert packet.frames[-1].values.shape == packet.frames[0].values.shape
    later = _packet("ep", 15, ring, ("base_0_rgb", "left_wrist_0_rgb"), torch.device("cpu"))
    assert later.reset is False


def test_paired_noise_is_identical_for_same_episode_and_decision():
    a = paired_noise(101, "episode_1", 3)
    b = paired_noise(101, "episode_1", 3)
    c = paired_noise(101, "episode_1", 4)
    assert torch.equal(a, b) and a.shape == (1, 10, 32)
    assert not torch.equal(a, c)


def test_wilson_interval_is_denominator_correct():
    assert _wilson(0, 0) is None
    lo, hi = _wilson(16, 20)
    assert 0.0 <= lo < 16 / 20 < hi <= 1.0


def test_author_replay_loader_uses_file_without_old_hash_module(tmp_path):
    from owac.simulation.rawvla_teacher_controller import _load_author_replay_module

    script = tmp_path / "scripts/replay/replay_libero_train_rawvla_light_npz.py"
    script.parent.mkdir(parents=True)
    script.write_text("MARKER = 'direct-import-ok'\n", encoding="utf-8")
    module = _load_author_replay_module(tmp_path)
    assert module.MARKER == "direct-import-ok"


def test_physical_frame_ring_does_not_duplicate_frame_ten():
    from owac.simulation.raw_student_closed_loop import _push_raw

    ring = [{"frame_index": index} for index in range(11)]
    ring = _push_raw(ring, {"frame_index": 10})
    assert [item["frame_index"] for item in ring] == [5, 6, 7, 8, 9, 10]
    ring = _push_raw(ring, {"frame_index": 11})
    assert [item["frame_index"] for item in ring] == [6, 7, 8, 9, 10, 11]
