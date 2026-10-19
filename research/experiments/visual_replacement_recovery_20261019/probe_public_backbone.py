"""CPU deployment probe with public weights and synthetic planes; no policy training."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from owac.baselines.single_frame_visual import make_convnext_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(101)
    started = time.perf_counter()
    model = make_convnext_base(
        args.checkpoint,
        sha256=args.sha256,
        camera_ids=("libero_agentview", "libero_wrist"),
    ).eval()
    inputs = torch.randn(1, 2, 3, 256, 256, requires_grad=True)
    tokens = model(inputs)
    if tokens.values.shape != (1, 512, 2048) or not torch.isfinite(tokens.values).all():
        raise ValueError("invalid real backbone token output")
    tokens.values.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise ValueError("missing or nonfinite used frontend parameter gradients")
    if inputs.grad.abs().sum() <= 0:
        raise ValueError("measurement gradient disconnected")
    with torch.no_grad():
        repeated = model(inputs.detach()).values
    if not torch.equal(tokens.values.detach(), repeated):
        raise ValueError("fixed-input CPU output changed")
    result = {
        "status": "passed",
        "scope": "public pretrained backbone, synthetic floating planes, CPU forward/backward only",
        "torch": torch.__version__,
        "seed": 101,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.sha256,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape": list(tokens.values.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "used_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "all_used_parameter_gradients_finite": True,
        "fixed_input_repeat_exact": True,
        "wall_seconds_including_load": time.perf_counter() - started,
        "optimizer_updates": 0,
        "policy_loaded": False,
        "real_observations": 0,
        "robot_closed_loops": 0,
        "capability_evidence": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
