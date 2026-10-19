"""Official embedding provenance, retraining and information boundaries."""

import ast
import copy
import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from owac.baselines.n02_measurement import RetrainedMeasurementScorer  # noqa: E402
from owac.baselines.n02_measurement_adapters import adapt_raw  # noqa: E402
from owac.evaluation.n02_retrained import (  # noqa: E402
    allow_extension,
    measurement_acceptance,
    paired_training_data,
)


def test_upstream_cnn_functions_and_classes_retain_official_wheel_ast():
    expected = {
        "FCEmbedding": "0c88c7d852b139626c0d33baded3144707365e26e1b8f69e9ccfa19c6ddef58e",
        "calculate_filter_output_size": (
            "4296dd29ccf6ce0e87478322adcb6dcccc613f896a5f1ed3826fcc33234a1985"
        ),
        "get_new_cnn_output_size": (
            "9a00c7f1faec59c7a8d9501c964b18122ec18e7392dd678142fdddeb4858ced4"
        ),
        "CNNEmbedding": "dcb3721b6eff7c5c44754574d11edcce9baab338506520852c17f27391560649",
    }
    for node in ast.parse(Path("src/owac/baselines/sbi_embedding.py").read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            assert digest == expected.pop(node.name)
    assert not expected


def test_retrained_model_is_causal_and_every_branch_learns_from_raw():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    model = RetrainedMeasurementScorer(torch.zeros(3), torch.ones(3)).eval()
    raw = torch.randn(2, 4, 3, 64, 64)
    logits = model(raw)
    altered = raw.clone()
    altered[:, 2:] += 30
    torch.testing.assert_close(model(altered)[:, :2], logits[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(model(raw[:, :2]), logits[:, :2], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        model.scores(raw)[..., 1] - model.scores(raw)[..., 0],
        logits[..., 1] - logits[..., 0],
        rtol=1e-5,
        atol=1e-6,
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = torch.nn.functional.cross_entropy(logits[:, -1], torch.tensor([0, 1]))
    loss.backward()
    for branch in ("embedding.cnn_subnet", "embedding.linear_subnet", "temporal", "classifier"):
        assert (
            sum(float(p.grad.norm()) for n, p in model.named_parameters() if n.startswith(branch))
            > 0
        )
    optimizer.step()
    for branch in ("embedding.cnn_subnet", "embedding.linear_subnet", "temporal", "classifier"):
        assert any(
            not torch.equal(p, before[n])
            for n, p in model.named_parameters()
            if n.startswith(branch)
        )
    with pytest.raises(ValueError, match="without extra"):
        model(torch.randn(2, 4, 4, 64, 64))
    with pytest.raises(ValueError, match="finite"):
        RetrainedMeasurementScorer(torch.zeros(3), torch.full((3,), float("inf")))


def test_paired_training_is_balanced_with_identical_noninformative_first_frame():
    train = paired_training_data([21998, 21999])
    assert train["truth"].tolist() == [0, 1, 0, 1]
    torch.testing.assert_close(train["raw"][::2, 0], train["raw"][1::2, 0], rtol=0, atol=0)
    assert [r["background_seed"] for r in train["identities"]] == [21998, 21998, 21999, 21999]


def test_measurement_adapters_are_causal_and_preserve_the_no_cue_pair():
    train = paired_training_data([21998])
    raw = train["raw"]
    for mode in ("center", "innovation", "center_innovation"):
        value = adapt_raw(raw, mode)
        torch.testing.assert_close(value[:, :3], adapt_raw(raw[:, :3], mode), rtol=0, atol=0)
        torch.testing.assert_close(value[0, 0], value[1, 0], rtol=0, atol=0)
        if "innovation" in mode:
            assert (value[:, 0] == 0).all()
        else:
            # Centering is an integer permutation, not a lossy resize/crop.
            torch.testing.assert_close(
                raw.flatten(2).sort(-1).values, value.flatten(2).sort(-1).values, rtol=0, atol=0
            )
    tiny = torch.zeros_like(raw)
    torch.testing.assert_close(adapt_raw(tiny, "center"), tiny, rtol=0, atol=0)


def test_extension_and_final_acceptance_use_behavior_and_same_seeds():
    row = {
        "pair_both_correct": 8,
        "no_cue_pass": 12,
        "wrong_prior_corrected": {"0": 0, "1": 0},
        "train_final_accuracy": 0.8,
        "qualified": False,
    }
    dev = {str(i): {"100": copy.deepcopy(row), "200": copy.deepcopy(row)} for i in range(3)}
    assert allow_extension(dev, False)["extend"]
    dev["0"]["200"]["pair_both_correct"] = 0
    dev["1"]["200"]["pair_both_correct"] = 0
    assert not allow_extension(dev, False)["extend"]
    for i in range(3):
        dev[str(i)]["200"]["qualified"] = i != 2
    held = {str(i): {"qualified": i != 0} for i in range(3)}
    assert not measurement_acceptance(dev, held, 200, False)["allow_mechanism_comparison"]
    held["0"]["qualified"] = True
    assert measurement_acceptance(dev, held, 200, False)["allow_mechanism_comparison"]
