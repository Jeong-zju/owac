"""Quantify image-dependent changes added after spatial fusion, without policy updates."""

import json
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/owac/worktrees/n02-measurement-completion")
OUT = ROOT / "outputs/n02-v12/information-r13"
EVIDENCE = ROOT / "research/experiments/n02_v1_2/information_r13_20261019"


def main():
    rows = {}
    for arm in ("r11-task850", "r12-one256"):
        rows[arm] = {}
        for panel in ("controlled", "cached"):
            with np.load(OUT / arm / (panel + ".npz"), allow_pickle=False) as a:
                fusion = a["spatial_fusion"]
                actual = a["pre_pi_actual"]
                fixed = a["pre_pi_fixed_state"]
            delta = actual - fusion
            varying = float(np.mean(np.var(delta, axis=0)) ** 0.5)
            base_varying = float(np.mean(np.var(fusion, axis=0)) ** 0.5)
            rows[arm][panel] = dict(
                actual_vs_fixed_max=float(np.abs(actual - fixed).max()),
                fusion_rms=float(np.mean(fusion**2) ** 0.5),
                added_term_rms=float(np.mean(delta**2) ** 0.5),
                fusion_sample_centered_rms=base_varying,
                added_term_sample_centered_rms=varying,
                varying_added_to_fusion_ratio=varying / max(base_varying, 1e-12),
                added_camera_channel_centered_rms=float(
                    np.mean((delta - delta.mean(axis=(0, 3, 4), keepdims=True)) ** 2) ** 0.5
                ),
            )
    result = dict(
        rows=rows,
        policy_updates=0,
        labels_as_policy_inputs=False,
        context_scope="Actual/fixed describes supplied context. "
        "N02 visual path does not read proprio "
        "when action_dim=0; pi state prompt is a separate input. "
        "Do not infer that entire policy ignores state.",
        code_paths=[
            "src/owac/frameworks/n02_full/inference_v12.py:_evidence",
            "src/owac/runtime/n02_v12_robotwin.py:Policy.predict_action",
        ],
        limits="Added term includes bridge BF16 rounding. Near-constant pooled output is evidence "
        "about this output branch and these images, not proof that every latent is constant.",
    )
    (EVIDENCE / "context_scope_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
