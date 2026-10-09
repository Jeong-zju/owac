import numpy as np

from owac.simulation.raw_preview import cfa_colored_display, raw_display


def test_display_is_fixed_across_frames_and_does_not_mutate_raw():
    a = np.array([[0, 100000], [1000000, 16777215]], dtype=np.uint32)
    b = a.copy()
    result = raw_display(a)
    assert result[0, 0] == 0 and result[1, 1] == 255
    assert result[0, 1] == raw_display(np.full((2, 2), 100000, np.uint32))[0, 0]
    assert np.array_equal(a, b)


def test_bayer_display_preserves_sites_and_only_colors_their_native_channel():
    raw = np.full((2, 2), 16777215, dtype=np.uint32)
    actual = cfa_colored_display(raw)
    expected = np.array([[[0, 255, 0], [255, 0, 0]], [[0, 0, 255], [0, 255, 0]]], np.uint8)
    assert np.array_equal(actual, expected)
