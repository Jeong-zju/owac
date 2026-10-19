"""Canonical π0.5 language and proprio boundary shared by training and inference."""

from __future__ import annotations

import numpy as np
import torch

POLICY_INPUT_CONTRACT = "pi05-native-language-state-v1"


def canonical_task_prompt(task, supplied, instructions):
    """One declared task-to-instruction map for replay, validation and live policy.

    A configured map is authoritative; missing/unknown task IDs fail instead of
    silently choosing text or inferring a task from a sentence.
    """
    if instructions:
        if not isinstance(task, str) or task not in instructions:
            raise ValueError(f"no canonical instruction for task {task!r}")
        text = instructions[task]
    else:
        text = supplied
    if not isinstance(text, str) or not text.strip():
        raise ValueError("task instruction is missing")
    return text.strip()


def prepare_language_state(policy, prompts, proprio, *, device):
    """Pad proprio BEFORE discrete tokenization, as native PI0._prepare_examples does.

    RoboTwin's fourteen measured joints become a thirty-two-dimensional state.
    Tokenizing only the measured dimensions changes the pretrained prompt format.
    LIBERO retains its declared continuous-state protocol and gets no state text.
    """
    prompts = list(prompts)
    if not prompts:
        raise ValueError("at least one official task instruction is required")
    if torch.is_tensor(proprio):
        proprio = proprio.detach().cpu().numpy()
    values = np.asarray(proprio, dtype=np.float32)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0]
    width = int(
        getattr(
            policy.base,
            "max_state_dim",
            getattr(
                getattr(policy.base, "action_head", None),
                "max_state_dim",
                getattr(policy.protocol, "max_state_dim", 32),
            ),
        )
    )
    if (
        values.ndim != 2
        or values.shape[0] != len(prompts)
        or not 0 < values.shape[1] <= width
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"expected finite proprio [batch, 1..{width}], got {values.shape}")
    discrete = policy.protocol.name == "robotwin"
    fill = float(getattr(policy.base, "state_pad_value", 0.0)) if discrete else 0.0
    padded = np.full((len(prompts), width), fill, dtype=np.float32)
    padded[:, : values.shape[1]] = values
    token_rows, mask_rows = [], []
    for index, prompt in enumerate(prompts):
        if prompt is None or not str(prompt).strip():
            raise ValueError("official task instruction is missing")
        tokens, masks = policy.base.tokenizer.tokenize(
            str(prompt), padded[index] if discrete else None
        )
        tokens = np.asarray(tokens).reshape(-1)
        masks = np.asarray(masks, dtype=bool).reshape(-1)
        if tokens.size != masks.size or not masks.any():
            raise ValueError("tokenizer returned invalid tokens/masks")
        token_rows.append(tokens)
        mask_rows.append(masks)
    length = max(row.size for row in token_rows)
    tokens = torch.zeros((len(prompts), length), dtype=torch.long, device=device)
    masks = torch.zeros_like(tokens, dtype=torch.bool)
    for index, (row, mask) in enumerate(zip(token_rows, mask_rows, strict=True)):
        tokens[index, : row.size] = torch.as_tensor(row, dtype=torch.long, device=device)
        masks[index, : mask.size] = torch.as_tensor(mask, dtype=torch.bool, device=device)
    return tokens, masks, torch.as_tensor(padded, device=device)
