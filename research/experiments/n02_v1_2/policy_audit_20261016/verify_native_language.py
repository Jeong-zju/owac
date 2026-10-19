"""Real tokenizer + upstream input methods, without loading model weights or updating them."""

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from starVLA.model.framework.VLM4A.PI0 import PI0, LazyPaliGemmaTokenizer

from owac.backends.rawvla_language import POLICY_INPUT_CONTRACT
from owac.evaluation.n02_full.v12 import _language_batch
from owac.runtime.rawvla_full_eval import write_json
from owac.runtime.rawvla_robotwin import normalize_state
from owac.training.n02_v12 import N02V12Trainer


def main():
    root = Path("outputs/n02-v12/policy-audit-20261016")
    sample = np.load(root / "original-normal/initial_observation.npz")
    reference = json.loads((root / "bridge-parity/parity.json").read_text())
    stats = json.loads(
        Path("/workspace/owac/rawvla/checkpoints/pi05_robotwin/policy_norm_stats.json").read_text()
    )["norm_stats"]
    state = normalize_state(sample["state"], stats["state"])[None]
    # The upstream methods need input attributes, not transformer/vision weights.
    base = SimpleNamespace(
        max_state_dim=32,
        state_pad_value=0.0,
        image_keys=("head", "left", "right"),
        image_resolution=(224, 224),
        model_name="PI05",
        config=SimpleNamespace(framework=SimpleNamespace(discrete_state_input=True)),
        action_head=SimpleNamespace(action_in_proj=torch.nn.Linear(1, 1), max_state_dim=32),
        tokenizer=LazyPaliGemmaTokenizer(
            200, "/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model"
        ),
    )
    base._image_to_tensor = MethodType(PI0._image_to_tensor, base)
    base._pad_array_2d = MethodType(PI0._pad_array_2d, base)
    example = {
        "image": [sample[f"rgb_{i}"] for i in range(3)],
        "state": state,
        "lang": reference["instruction"],
    }
    batch = PI0._prepare_examples(base, [example], include_actions=False)
    native = PI0._build_observation_from_batch(base, batch)
    trainer = object.__new__(N02V12Trainer)
    trainer.bridge = SimpleNamespace(base=base, protocol=SimpleNamespace(name="robotwin"))
    trainer.device = torch.device("cpu")
    actual = trainer.language([example["lang"]], state)
    evaluation = _language_batch(trainer.bridge, [example["lang"]], state, None)
    report = {
        "optimizer_updates": 0,
        "weights_loaded": False,
        "input_contract": POLICY_INPUT_CONTRACT,
        "train_tokens_identical_to_upstream": torch.equal(actual[0], native.tokenized_prompt),
        "train_masks_identical_to_upstream": torch.equal(actual[1], native.tokenized_prompt_mask),
        "eval_tokens_identical_to_upstream": torch.equal(evaluation[0], native.tokenized_prompt),
        "eval_masks_identical_to_upstream": torch.equal(
            evaluation[1], native.tokenized_prompt_mask
        ),
        "state_identical_to_upstream": torch.equal(actual[2], native.state[:, 0]),
        "active_tokens": int(actual[1].sum()),
    }
    write_json(root / "r6-language-parity.json", report)
    print(json.dumps(report), flush=True)
    assert all(value for key, value in report.items() if "identical" in key)


if __name__ == "__main__":
    main()
