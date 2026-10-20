"""Effective-config state-padding tests against the pinned PI0 implementation.

These tests extract the actual ``_prepare_examples`` and
``_build_observation_from_batch`` methods from the pinned StarVLA source and
drive them with a fake module shell.  No model weights, GPU, or simulator.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PI0_SOURCE = (
    ROOT / "outputs/raw-distillation-20261020-sol/starvla-pinned-source/"
    "starVLA/model/framework/VLM4A/PI0.py"
)


def _methods():
    if not PI0_SOURCE.is_file():
        pytest.skip("pinned PI0 source is absent")
    tree = ast.parse(PI0_SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PI0")
    functions = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_prepare_examples",
            "_build_observation_from_batch",
        }:
            source = textwrap.dedent(ast.get_source_segment(PI0_SOURCE.read_text(), node))
            namespace = {"np": np, "torch": torch}
            exec(compile(source, str(PI0_SOURCE), "exec"), namespace)
            functions[node.name] = namespace[node.name]
    return functions


class _Observation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Tokenizer:
    def __init__(self):
        self.received = []

    def tokenize(self, prompt, state=None):
        self.received.append(None if state is None else np.asarray(state).copy())
        return np.asarray([1, 2, 3], dtype=np.int64), np.asarray([True, True, True])


class _FakeSelf:
    def __init__(self, *, discrete: bool, pad: float):
        self.image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        self.max_state_dim = 32
        self.state_pad_value = float(pad)
        self.config = SimpleNamespace(
            framework=SimpleNamespace(discrete_state_input=bool(discrete))
        )
        self.action_head = SimpleNamespace(action_in_proj=SimpleNamespace(weight=torch.zeros(1, 1)))
        self.tokenizer = _Tokenizer()

    def _image_to_tensor(self, image):
        return torch.as_tensor(image, dtype=torch.float32)

    def _pad_array_2d(self, value, rows, width, fill):
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[None]
        result = np.full((rows, width), fill, dtype=np.float32)
        copy_rows = min(rows, array.shape[0])
        copy_cols = min(width, array.shape[1])
        result[:copy_rows, :copy_cols] = array[:copy_rows, :copy_cols]
        return result


def _batch(discrete: bool, pad: float, state):
    functions = _methods()
    shell = _FakeSelf(discrete=discrete, pad=pad)
    batch = functions["_prepare_examples"](
        shell,
        [{"image": [torch.zeros(1, 3, 4, 4)], "lang": "task", "state": state}],
        include_actions=False,
    )
    return shell, batch, functions


def test_actual_libero_false_config_zero_pad_is_bitwise_equivalent_to_state8():
    state8 = np.linspace(-1, 1, 8, dtype=np.float32)
    old_zero32 = np.zeros(32, dtype=np.float32)
    old_zero32[:8] = state8
    shell8, batch8, functions = _batch(False, -2.0, state8)
    shell32, batch32, _ = _batch(False, -2.0, old_zero32)
    assert torch.equal(batch8["state"], batch32["state"])
    assert np.allclose(batch8["state"].numpy()[0, 0, 8:], 0.0)
    assert torch.equal(batch8["image"], batch32["image"])
    assert torch.equal(batch8["image_mask"], batch32["image_mask"])
    # Drive the actual observation builder with an injected Observation shell.
    namespace = {"np": np, "torch": torch, "Observation": _Observation}
    node = None
    tree = ast.parse(PI0_SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PI0")
    node = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_build_observation_from_batch"
    )
    source = textwrap.dedent(ast.get_source_segment(PI0_SOURCE.read_text(), node))
    exec(compile(source, str(PI0_SOURCE), "exec"), namespace)
    builder = namespace["_build_observation_from_batch"]
    builder(shell8, batch8)
    builder(shell32, batch32)
    assert shell8.tokenizer.received == [None]
    assert shell32.tokenizer.received == [None]


def test_true_branch_uses_actual_configured_pad_and_tokenizes_state():
    state8 = np.linspace(-1, 1, 8, dtype=np.float32)
    zero32 = np.zeros(32, dtype=np.float32)
    shell8, batch8, functions = _batch(True, -2.0, state8)
    shell32, batch32, _ = _batch(True, -2.0, zero32)
    assert not torch.equal(batch8["state"], batch32["state"])
    assert np.allclose(batch8["state"].numpy()[0, 0, 8:], -2.0)
    namespace = {"np": np, "torch": torch, "Observation": _Observation}
    tree = ast.parse(PI0_SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PI0")
    node = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_build_observation_from_batch"
    )
    source = textwrap.dedent(ast.get_source_segment(PI0_SOURCE.read_text(), node))
    exec(compile(source, str(PI0_SOURCE), "exec"), namespace)
    builder = namespace["_build_observation_from_batch"]
    builder(shell8, batch8)
    builder(shell32, batch32)
    assert shell8.tokenizer.received[0] is not None
    assert shell32.tokenizer.received[0] is not None
    assert not np.array_equal(shell8.tokenizer.received[0], shell32.tokenizer.received[0])


def test_effective_libero_configs_are_false_and_historical_zero_pad():
    author = (
        ROOT / "outputs/raw-distillation-20261020-sol/teacher-search-v2/author-main/"
        "configs/libero/pi05.yaml"
    ).read_text()
    historical = (
        ROOT / "outputs/checkpoints/raw-distillation/pi05_libero_starvla/config.yaml"
    ).read_text()
    assert "discrete_state_input: false" in author
    assert "pad_state_value: -2.0" in author
    assert "discrete_state_input: false" in historical
    assert "pad_state_value: 0.0" in historical
    backend = (ROOT / "src/owac/backends/rawvla.py").read_text()
    assert '"discrete_state_input": suite != "libero"' in backend
    assert '"pad_state_value": 0.0' in backend
