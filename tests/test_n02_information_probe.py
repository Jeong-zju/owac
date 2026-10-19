"""Check holdout isolation, physical units, and a diagnostic readout positive control."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from owac.evaluation.n02_full.information_probe import (  # noqa: E402
    fit_readout,
    metrics,
    split_indices,
)


def dataset():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(120, 4)).astype(np.float32)
    y = x @ rng.normal(size=(4, 8)).astype(np.float32)
    records = [
        dict(group=i // 4, split="train" if i < 80 else "val" if i < 100 else "test")
        for i in range(120)
    ]
    return x, y, records


def test_rejects_adjacent_frame_leakage():
    _, _, rows = dataset()
    rows[-1]["group"] = rows[0]["group"]
    with pytest.raises(ValueError, match="leaks"):
        split_indices(rows)


def test_test_labels_cannot_change_normalizers_or_selected_weights():
    torch.set_num_threads(1)
    x, y, rows = dataset()
    info, pred = fit_readout(x, y, rows, nonlinear=False, updates=100)
    changed = y.copy()
    changed[100:] += 10000
    info2, pred2 = fit_readout(x, changed, rows, nonlinear=False, updates=100)
    assert info == info2
    np.testing.assert_array_equal(pred, pred2)


def test_small_probe_can_recover_known_signal_and_shuffle_removes_it():
    torch.set_num_threads(1)
    x, y, rows = dataset()
    _, pred = fit_readout(x, y, rows, nonlinear=False, updates=1000)
    _, shuffled = fit_readout(x, y, rows, nonlinear=False, shuffle=True, updates=1000)
    assert (
        np.square(pred[100:] - y[100:]).mean() < 0.15 * np.square(shuffled[100:] - y[100:]).mean()
    )


def test_coordinates_are_centimetres_and_openness_classes_are_balanced():
    y = np.zeros((4, 17))
    y[:, -2:] = [[0, 1], [1, 0], [0, 1], [1, 0]]
    pred = y.copy()
    pred[:, 0] += 0.03
    result = metrics(pred, y, "controlled")
    assert result["object"]["mean_cm"] == pytest.approx(3)
    assert result["object"]["within_2cm"] == 0
    assert result["object"]["within_5cm"] == 1
    assert result["openness"]["balanced_accuracy"] == 1


def test_head_training_cannot_backpropagate_into_frozen_feature_producer():
    x, y, rows = dataset()
    features = torch.tensor(x, requires_grad=True)
    before = features.detach().clone()
    fit_readout(features, y, rows, nonlinear=False, updates=100)
    assert features.grad is None
    torch.testing.assert_close(features.detach(), before, rtol=0, atol=0)


def test_capture_reader_waits_for_a_complete_zip_archive(monkeypatch, tmp_path):
    import runpy
    import zipfile
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1] / "research/experiments/n02_v1_2/"
        "information_r13_20261019/extract.py"
    )
    namespace = runpy.run_path(str(script))
    path = tmp_path / "snapshot.npz"
    np.savez(path, y=np.arange(17))
    original = np.load
    calls = []

    def loading(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise zipfile.BadZipFile("writer has not finished")
        return original(*args, **kwargs)

    monkeypatch.setattr(namespace["np"], "load", loading)
    monkeypatch.setattr(namespace["time"], "sleep", lambda _: None)
    with namespace["wait_snapshot"](path) as data:
        np.testing.assert_array_equal(data["y"], np.arange(17))
    assert len(calls) == 2
